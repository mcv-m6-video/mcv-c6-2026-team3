"""
File containing the main model.
"""

#Standard imports
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F
from util.eval_spotting import evaluate


#Local imports
from model.modules import BaseRGBModel, FCLayers, step, FCBottleneckLayers

class TemporalResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError('temporal_kernel_size must be odd to preserve sequence length.')

        padding = dilation * (kernel_size // 2)

        self.conv1 = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn1 = nn.BatchNorm1d(out_dim)
        self.conv2 = nn.Conv1d(
            out_dim,
            out_dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Identity() if in_dim == out_dim else nn.Conv1d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout(out)

        out = out + residual
        out = self.activation(out)
        return out

class TemporalConvHead(nn.Module):

    def __init__(self, input_dim, hidden_dim=None, kernel_size=3, dilations=(1, 2, 4), dropout=0.2):
        super().__init__()
        hidden_dim = input_dim if hidden_dim is None else hidden_dim

        blocks = []
        current_dim = input_dim
        for dilation in dilations:
            blocks.append(
                TemporalResidualBlock(
                    current_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            current_dim = hidden_dim

        self.blocks = nn.Sequential(*blocks)
        self.out_proj = nn.Identity() if hidden_dim == input_dim else nn.Conv1d(hidden_dim, input_dim, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.blocks(x)
        x = self.out_proj(x)
        x = x.transpose(1, 2)  # [B, T, D]
        return x

class Model(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, args = None):
            super().__init__()
            self._feature_arch = args.feature_arch

            if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny004': 'regnety_004',
                    'rny008': 'regnety_008',
                }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
                feat_dim = features.head.fc.in_features

                # Remove final classification layer
                features.head.fc = nn.Identity()
                self._d = feat_dim

            else:
                raise NotImplementedError(args._feature_arch)

            self._features = features

            #Temporal neck
            if args.use_temporal_head:
                self._temporal_head = TemporalConvHead(
                    input_dim=self._d,
                    hidden_dim=getattr(args, 'temporal_hidden_dim', None),
                    kernel_size=getattr(args, 'temporal_kernel_size', 3),
                    dilations=tuple(getattr(args, 'temporal_dilations', [1, 2, 4])),
                    dropout=getattr(args, 'temporal_dropout', 0.2),
                )
            else:
                self._temporal_head = nn.Identity()

            # MLP for classification
            self.use_delta = args.use_delta
            self._fc = FCLayers(self._d, args.num_classes+1) # +1 for background class (we now perform per-frame classification with softmax, therefore we have the extra background class)

            if self.use_delta:

                if args.use_bottleneck:
                    self._fc = FCBottleneckLayers(self._d*2, args.num_classes+1)
                else:
                    self._fc = FCLayers(self._d*2, args.num_classes+1)

            #Augmentations and crop
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
                T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.GaussianBlur(5)], p = 0.25),
                T.RandomHorizontalFlip(),
            ])

            #Standarization
            self.standarization = T.Compose([
                T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) #Imagenet mean and std
            ])

        def forward(self, x):
            x = self.normalize(x) #Normalize to 0-1
            batch_size, clip_len, channels, height, width = x.shape #B, T, C, H, W

            if self.training:
                x = self.augment(x) #augmentation per-batch

            x = self.standarize(x) #standarization imagenet stats
                        
            im_feat = self._features(
                x.view(-1, channels, height, width)
            ).reshape(batch_size, clip_len, self._d) #B, T, D

            im_feat = self._temporal_head(im_feat)

            if self.use_delta:
                delta = torch.zeros_like(im_feat)
                delta[:, 1:, :] = im_feat[:, 1:, :] - im_feat[:, :-1, :]
                im_feat = torch.cat([im_feat, delta], dim=-1)

            #MLP
            im_feat = self._fc(im_feat) #B, T, num_classes+1

            return im_feat 
        
        def normalize(self, x):
            return x / 255.
        
        def augment(self, x):
            for i in range(x.shape[0]):
                x[i] = self.augmentation(x[i])
            return x

        def standarize(self, x):
            for i in range(x.shape[0]):
                x[i] = self.standarization(x[i])
            return x

        def print_stats(self):
            print('Model params:',
                sum(p.numel() for p in self.parameters()))

    def __init__(self, args=None):
        self.device = "cpu"
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
            self.device = "cuda"

        self._model = Model.Impl(args=args)
        self._model.print_stats()
        self._args = args

        self._model.to(self.device)
        self._num_classes = args.num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        weights = torch.tensor([1.0] + [5.0] * (self._num_classes), dtype=torch.float32).to(self.device)

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device).long()

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    pred = pred.view(-1, self._num_classes + 1) # B*T, num_classes
                    label = label.view(-1) # B*T
                    loss = F.cross_entropy(
                            pred, label, reduction='mean', weight = weights)

                if optimizer is not None:
                    step(optimizer, scaler, loss,
                        lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)     # Avg loss

    def predict(self, seq):

        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4: # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits = self._model(seq)

            # apply sigmoid
            pred = torch.softmax(logits, dim=-1)
            
            return pred.cpu().numpy(), logits
