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
from typing import List


#Local imports
from model.modules import BaseRGBModel, FCLayers, step, FCBottleneckLayers

class _ConvBlock(nn.Module):
    """Single convolutional block: Conv1d → BatchNorm → ReLU → Dropout."""
 
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout_p: float,
        stride: int = 1,
    ):
        super().__init__()
        # 'same'-style padding so that stride-1 convolutions preserve length
        # and stride-2 convolutions halve it (±1 depending on parity).
        padding = kernel_size // 2
 
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
 
class UNetTemporalHeadMaxPool(nn.Module):
    """
    1-D temporal U-Net head with max-pool downsampling.
 
    Parameters
    ----------
    feature_size : int
        Number of input features *F* per time-step.
    kernel_size : int
        Kernel width shared by every convolution (must be odd).
    dropout_p : float
        Dropout probability applied after every convolutional block.
    hidden_dims : list[int]
        Channel widths at each encoder level.  ``len(hidden_dims)`` defines
        the depth of the U-Net.
 
    Input
    -----
    x : Tensor of shape ``(B, T, F)``
 
    Output
    ------
    Tensor of shape ``(B, T, C)`` where ``C = hidden_dims[0]``.
    """
 
    def __init__(
        self,
        feature_size: int,
        kernel_size: int,
        dropout_p: float,
        hidden_dims: List[int],
    ):
        super().__init__()
 
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one element.")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd for symmetric padding.")
 
        self.depth = len(hidden_dims)
        self.output_channels = hidden_dims[0]
 
        # ---- Encoder --------------------------------------------------------
        # Each encoder level: Conv1d(stride=1) → MaxPool1d(2)
        # The skip connection is taken BEFORE the pool so it retains the
        # higher temporal resolution needed by the decoder.
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
 
        in_ch = feature_size
        for dim in hidden_dims:
            self.encoders.append(
                _ConvBlock(in_ch, dim, kernel_size, dropout_p=dropout_p)
            )
            self.pools.append(nn.MaxPool1d(kernel_size=2, stride=2))
            in_ch = dim
 
        # ---- Decoder --------------------------------------------------------
        self.decoders = nn.ModuleList()
        N = self.depth
 
        for i in range(N):
            if i < N - 1:
                up_ch = hidden_dims[N - 1 - i]
                skip_ch = hidden_dims[N - 2 - i]
                out_ch = hidden_dims[N - 2 - i]
            else:
                # Last stage: merge with the original input
                up_ch = hidden_dims[0]
                skip_ch = feature_size
                out_ch = hidden_dims[0]
 
            self.decoders.append(
                _ConvBlock(up_ch + skip_ch, out_ch, kernel_size, dropout_p=dropout_p)
            )
 
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape ``(B, T, F)``
 
        Returns
        -------
        Tensor, shape ``(B, T, C)``  with ``C = hidden_dims[0]``
        """
        # Conv1d expects (B, C, L)
        x = x.permute(0, 2, 1)
 
        # ---------- Encoder ---------------------------------------------------
        skips: List[torch.Tensor] = [x]  # original input is the shallowest skip
 
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)     # conv (preserves temporal length)
            skips.append(x)    # save BEFORE pooling (full-resolution skip)
            x = pool(x)        # downsample by 2
 
        # The last element in skips is the pre-pool output of the deepest
        # encoder.  ``x`` after the final pool is the bottleneck.
        # We don't need a skip from the deepest level — pop it.
        skips.pop()
 
        # ---------- Decoder ---------------------------------------------------
        for decoder in self.decoders:
            skip = skips.pop()
            x = F.interpolate(
                x, size=skip.shape[2], mode="linear", align_corners=False
            )
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
 
        # Back to (B, T, C)
        x = x.permute(0, 2, 1)
        return x

class UNetTemporalHead(nn.Module):
    """
    1-D temporal U-Net head.
 
    Parameters
    ----------
    feature_size : int
        Number of input features *F* per time-step.
    kernel_size : int
        Kernel width shared by every convolution in the network.
        Must be an odd number so that padding = kernel_size // 2 gives
        symmetric, 'same'-style padding.
    dropout_p : float
        Dropout probability applied after every convolutional block.
    hidden_dims : list[int]
        Channel widths at each encoder level.  ``len(hidden_dims)`` defines
        the depth of the U-Net (number of down/up-sampling stages).
        Example: ``[64, 128, 256]`` → 3 encoder blocks, 3 decoder blocks.
 
    Input
    -----
    x : Tensor of shape ``(B, T, F)``
 
    Output
    ------
    Tensor of shape ``(B, T, C)`` where ``C = hidden_dims[0]``.
    This is intended to be fed into a final ``nn.Linear(C, num_classes)``
    for frame-level or dense prediction.
    """
 
    def __init__(
        self,
        feature_size: int,
        kernel_size: int,
        dropout_p: float,
        hidden_dims: List[int],
    ):
        super().__init__()
 
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one element.")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd for symmetric padding.")
 
        self.depth = len(hidden_dims)
        self.output_channels = hidden_dims[0]  # C in the output (B, T, C)
 
        # ---- Encoder --------------------------------------------------------
        self.encoders = nn.ModuleList()
        in_ch = feature_size
        for dim in hidden_dims:
            self.encoders.append(
                _ConvBlock(in_ch, dim, kernel_size, stride=2, dropout_p=dropout_p)
            )
            in_ch = dim
 
        # ---- Decoder --------------------------------------------------------
        # There are ``self.depth`` decoder stages, each performing:
        #   1. Upsample the current tensor to match the skip's temporal length.
        #   2. Concatenate with the skip connection along the channel axis.
        #   3. Apply a stride-1 ConvBlock to fuse the information.
        self.decoders = nn.ModuleList()
        N = self.depth
 
        for i in range(N):
            if i < N - 1:
                # Merge with an intermediate encoder skip
                up_ch = hidden_dims[N - 1 - i]
                skip_ch = hidden_dims[N - 2 - i]
                out_ch = hidden_dims[N - 2 - i]
            else:
                # Last decoder stage: merge with the original input
                up_ch = hidden_dims[0]
                skip_ch = feature_size
                out_ch = hidden_dims[0]
 
            self.decoders.append(
                _ConvBlock(
                    up_ch + skip_ch, out_ch, kernel_size, stride=1, dropout_p=dropout_p
                )
            )
 
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape ``(B, T, F)``
 
        Returns
        -------
        Tensor, shape ``(B, T, C)``  with ``C = hidden_dims[0]``
        """
        # Conv1d expects (B, C, L) so permute from (B, T, F) → (B, F, T)
        x = x.permute(0, 2, 1)
 
        # ---------- Encoder (collect skip connections) --------------------
        skips: List[torch.Tensor] = [x]  # original input is the shallowest skip
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
 
        # Pop the bottleneck — it is the starting point for the decoder,
        # not a skip connection.
        x = skips.pop()
 
        # ---------- Decoder (consume skips deepest → shallowest) ----------
        for decoder in self.decoders:
            skip = skips.pop()
            # Upsample to the skip's temporal length (handles any T, even odd)
            x = F.interpolate(x, size=skip.shape[2], mode="linear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
 
        # Permute back to (B, T, C)
        x = x.permute(0, 2, 1)
        return x

class ModelUnet(BaseRGBModel):

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

            constructed_dims = [self._d]

            if args.custom_hidden:
                constructed_dims = args.unet_dims
            else:
                for dim_mult in args.unet_dims:
                    constructed_dims.append(constructed_dims[-1]*dim_mult)

            #Temporal neck
            if args.unet_type == "max":
                self._temporal_head = UNetTemporalHeadMaxPool(
                    self._d,
                    kernel_size=args.unet_kernel_size,
                    dropout_p=args.dropout_unet,
                    hidden_dims=constructed_dims
                )
            else:
                self._temporal_head = UNetTemporalHead(
                    self._d,
                    kernel_size=args.unet_kernel_size,
                    dropout_p=args.dropout_unet,
                    hidden_dims=constructed_dims
                )

            # MLP for classification
            self.use_delta = args.use_delta
            feature_num = self._d
            if args.custom_hidden:
                feature_num = args.unet_dims[0]
            self._fc = FCLayers(feature_num, args.num_classes+1) # +1 for background class (we now perform per-frame classification with softmax, therefore we have the extra background class)

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

        self._model = ModelUnet.Impl(args=args)
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
