"""
File containing the main model.

Modified version:
- adds optional BCEWithLogitsLoss(pos_weight=...)
"""

from contextlib import nullcontext

import timm
import torch
import torchvision.transforms.v2 as T
from torch import nn
from tqdm import tqdm
from util.focal_loss import FocalLoss

from model.modules import BaseRGBModel, FCLayers, step


class Model(BaseRGBModel):
    class Impl(nn.Module):
        def __init__(self, args=None):
            super().__init__()

            self._feature_arch = args.feature_arch

            if self._feature_arch.startswith(("rny002", "rny004", "rny008")):

                features = timm.create_model(
                    {
                        "rny002": "regnety_002",
                        "rny004": "regnety_004",
                        "rny008": "regnety_008",
                    }[self._feature_arch.rsplit("_", 1)[0]],
                    pretrained=True,
                )

                if args.freeze_backbone:

                    for param in features.parameters():
                        param.requires_grad = False

                feat_dim = features.head.fc.in_features
                features.head.fc = nn.Identity()
                self._d = feat_dim

            else:
                raise NotImplementedError(args.feature_arch)

            self._features = features
            self._fc = FCLayers(self._d, args.num_classes)


            #Temporal manager from assumption 1

            match args.temporal_handler:
                case 'avg_pooling':
                    
                    def avg_pol(x):
                        return torch.mean(x, dim=1)

                    self.temporal_handler = avg_pol

                case 'convolution':

                    self.temporal_conv = nn.Conv1d(
                        in_channels=self._d,
                        out_channels=self._d,
                        kernel_size=3,
                        padding=1
                    )

                    self.temporal_dropout = nn.Dropout()

                    def conv_pol(x : torch.Tensor):

                        x = x.transpose(1, 2)
                        im_feat = self.temporal_conv(x)
                        im_feat = torch.relu(im_feat)
                        im_feat = self.temporal_dropout(im_feat)

                        #Time dimension still needs to be collapsed
                        im_feat = torch.mean(im_feat, dim=2)

                        return im_feat

                    self.temporal_handler = conv_pol

                case 'convolution_max':

                    self.temporal_conv = nn.Conv1d(
                        in_channels=self._d,
                        out_channels=self._d,
                        kernel_size=3,
                        padding=1
                    )

                    self.temporal_dropout = nn.Dropout()

                    def conv_pol_max(x : torch.Tensor):

                        x = x.transpose(1, 2)
                        im_feat = self.temporal_conv(x)
                        im_feat = torch.relu(im_feat)
                        im_feat = self.temporal_dropout(im_feat)

                        #Time dimension still needs to be collapsed
                        im_feat = torch.max(im_feat, dim=2)[0]

                        return im_feat

                    self.temporal_handler = conv_pol_max

                case _:

                    def max_pol(x):
                        return torch.max(x, dim=1)[0]

                    self.temporal_handler = max_pol


            self.standarization = T.Compose(
                [
                    T.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    )
                ]
            )

        def forward(self, x):
            x = self.normalize(x)

            batch_size, clip_len, channels, height, width = x.shape

            if self.training:
                x = self.augment(x) #augmentation per-batch

            x = self.standarize(x) #standarization imagenet stats

            im_feat = self._features(
                x.view(-1, channels, height, width)
            ).reshape(batch_size, clip_len, self._d)

            # Temporal max-pooling
            im_feat = self.temporal_handler(im_feat)

            # Classification head
            im_feat = self._fc(im_feat)

            return im_feat

        def normalize(self, x):
            return x / 255.0

        def augment(self, x):
            """
            Apply the SAME random transform to all frames of a clip
            by passing the whole clip tensor [L, C, H, W] to torchvision transforms.
            """

            B = x.size(0)

            #We apply it clip wise so it is consistent among all frames from the same clip.
            for b in range(B):
                clip = x[b]

                if torch.rand(1) < 0.5:
                    clip = torch.flip(clip, dims=[3])

                if torch.rand(1) < 0.25:
                    brightness = torch.empty(1).uniform_(0.7, 1.2).item()
                    contrast = torch.empty(1).uniform_(0.7, 1.2).item()
                    saturation = torch.empty(1).uniform_(0.7, 1.2).item()
                    hue = torch.empty(1).uniform_(-0.2, 0.2).item()

                    clip = T.functional.adjust_brightness(clip, brightness)
                    clip = T.functional.adjust_contrast(clip, contrast)
                    clip = T.functional.adjust_saturation(clip, saturation)
                    clip = T.functional.adjust_hue(clip, hue)

                if torch.rand(1) < 0.25:
                    clip = T.functional.gaussian_blur(clip, kernel_size=5)

                x[b] = clip

            return x

        def standarize(self, x):
            for i in range(x.shape[0]):
                x[i] = self.standarization(x[i])
            return x

        def print_stats(self):
            print("Model params:", sum(p.numel() for p in self.parameters()))

    def __init__(self, args=None, pos_weight=None):
        self.device = "cpu"
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
            self.device = "cuda"

        self._model = Model.Impl(args=args)
        self._model.print_stats()
        self._args = args
        self._model.to(self.device)
        self._num_classes = args.num_classes

        if pos_weight is not None:
            pos_weight = pos_weight.to(self.device)
            print("Using BCEWithLogitsLoss with pos_weight:")
            print(pos_weight.detach().cpu().tolist())
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:

            if args.use_focal_loss:
                self.loss_fn = FocalLoss()
            else:
                print("Using standard BCEWithLogitsLoss without class weights.")
                self.loss_fn = nn.BCEWithLogitsLoss()

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):
        if optimizer is None:
            self._model.eval()
        else:
            optimizer.zero_grad()
            self._model.train()

        epoch_loss = 0.0

        with torch.no_grad() if optimizer is None else nullcontext():
            for _, batch in enumerate(tqdm(loader)):
                frame = batch["frame"].to(self.device).float()
                label = batch["label"].to(self.device).float()

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    loss = self.loss_fn(pred, label)

                if optimizer is not None:
                    step(optimizer, scaler, loss, lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)

    def predict(self, seq):
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)

        if len(seq.shape) == 4:
            seq = seq.unsqueeze(0)

        if seq.device != self.device:
            seq = seq.to(self.device)

        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = self._model(seq)
                pred = torch.sigmoid(pred)

        return pred.cpu().numpy()