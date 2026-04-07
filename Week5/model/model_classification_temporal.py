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
import pytorchvideo.models.hub as hub

import torchvision.models.video as video_models

class ModelTemporal(BaseRGBModel):
    class Impl(nn.Module):
        def __init__(self, args=None):
            super().__init__()

            self._feature_arch = args.feature_arch

            if self._feature_arch == "r2plus1d_18":
                self._features = video_models.r2plus1d_18(weights=video_models.R2Plus1D_18_Weights.DEFAULT)
            
            elif self._feature_arch == "r3d_18":
                self._features = video_models.r3d_18(weights=video_models.R3D_18_Weights.DEFAULT)

            elif self._feature_arch == "r2plus1d_34":
                self._features = torch.hub.load(
                    "moabitcoin/ig65m-pytorch",
                    "r2plus1d_34_8_kinetics",
                    num_classes=400,
                    pretrained=True,
                )

            self._d = self._features.fc.in_features

            for param in self._features.parameters():
                param.requires_grad = False

            for param in self._features.layer4[1].conv2.parameters():
                param.requires_grad = True

            self._features.fc = nn.Identity()

            self._fc = FCLayers(self._d, args.num_classes)

            self.standarization = T.Compose([
                T.Normalize(mean = (0.43216, 0.394666, 0.37645), std = (0.22803, 0.22145, 0.216989))
            ])


        def forward(self, x):
            x = self.normalize(x)

            batch_size, clip_len, channels, height, width = x.shape

            if self.training:
                x = self.augment(x) #augmentation per-batch

            x = self.standarize(x) #standarization imagenet stats

            x = x.permute(0, 2, 1, 3, 4)

            im_feat = self._features(x)

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

        self._model = ModelTemporal.Impl(args=args)
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