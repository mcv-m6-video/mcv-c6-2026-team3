import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from pathlib import Path
from tqdm import tqdm


class _FeatureReduction(nn.Module):
    def __init__(self, in_channels=2048, out_channels=51):
        super().__init__()
        self.pool_kernel = 40
        self.pool_stride = 40
        self.out_channels = out_channels

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B * H * W, 1, C)
        x = F.avg_pool1d(x, kernel_size=self.pool_kernel, stride=self.pool_stride)
        out_c = x.shape[-1]
        x = x.reshape(B, H, W, out_c).permute(0, 3, 1, 2)  # (B, out_c, H, W)
        return x


class _DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=2, padding=1, output_padding=1):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel,
                                          stride=stride, padding=padding,
                                          output_padding=output_padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.deconv(x))


class BgsCNNV2(nn.Module):
    """
    PyTorch re-implementation of bgsCNN_v2.
    
    Architecture (mirrors bgsCNN_v2.py):
      pre_conv  : 1x1 conv, 6 → 3 channels  (input is frame+bg concatenated)
      resnet_v2 : ResNet-50 backbone, output_stride=16 → spatial size = H/16
                  For 321x321 input → ~21x21x2048
      feat_red  : channel avg-pool 2048 → 51
      deconv_1  : 21→43,  51→32
      pool_1    : 43→41,  32→16  (channel halving via 3D pool)
      deconv_2  : 41→83,  16→8
      pool_2    : 83→81,  8→8    (spatial shrink only)
      deconv_3  : 81→163, 8→4
      pool_3    : 163→161, 4→4
      deconv_4  : 161→323, 4→1
      pool_4    : 323→321, 1→1
      conv      : 1x1, 1→1
      sigmoid   : output probability map
    """

    def __init__(self, image_height=321, image_width=321):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width

        self.pre_conv = nn.Conv2d(6, 3, kernel_size=1, bias=False)

        # ResNet-50 with ImageNet pretrained weights (mirrors bgsCNN_v2 which loads resnet_v2_50.ckpt)
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        # Remove avgpool and fc; use dilated conv for output_stride=16
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1   # stride accumulates to 4
        self.layer2 = backbone.layer2   # stride accumulates to 8
        self.layer3 = backbone.layer3   # stride accumulates to 16
        # layer4 with dilation to keep output_stride=16
        for block in backbone.layer4:
            block.conv2.stride = (1, 1)
            if hasattr(block, 'downsample') and block.downsample is not None:
                block.downsample[0].stride = (1, 1)
            block.conv2.dilation = (2, 2)
            block.conv2.padding = (2, 2)
        self.layer4 = backbone.layer4   # output_stride=16, channels=2048

        # Feature reduction: 2048 → 51
        self.feat_red = _FeatureReduction(in_channels=2048, out_channels=51)

        # deconv_1: 51 → 32, upsample 21 → 43
        self.deconv1 = nn.ConvTranspose2d(51, 32, kernel_size=3, stride=2, padding=1, output_padding=0)
        # channel reduction 32 → 16 via conv (mimics 3D pool halving channels)
        self.ch_red1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0)  # 43→41

        # deconv_2: 16 → 8, upsample 41 → 83
        self.deconv2 = nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=0)
        # spatial shrink 83→81
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)

        # deconv_3: 8 → 4, upsample 81 → 163
        self.deconv3 = nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=0)
        # spatial shrink 163→161
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)

        # deconv_4: 4 → 1, upsample 161 → 323
        self.deconv4 = nn.ConvTranspose2d(4, 1, kernel_size=3, stride=2, padding=1, output_padding=0)
        # spatial shrink 323→321
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=0)

        # final 1x1 conv
        self.final_conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)

    def forward(self, x):
        # x: (B, 6, H, W)  [frame_rgb + bg_rgb, normalised 0-1]
        x = self.pre_conv(x)                    # (B, 3, H, W)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)                      # (B, 2048, H/16, W/16)

        x = self.feat_red(x)                    # (B, 51, H/16, W/16)

        x = F.relu(self.deconv1(x))             # (B, 32, ~43, ~43)
        x = F.relu(self.ch_red1(x))             # (B, 16, 41, 41)

        x = F.relu(self.deconv2(x))             # (B, 8, 83, 83)
        x = self.pool2(x)                       # (B, 8, 81, 81)

        x = F.relu(self.deconv3(x))             # (B, 4, 163, 163)
        x = self.pool3(x)                       # (B, 4, 161, 161)

        x = F.relu(self.deconv4(x))             # (B, 1, 323, 323)
        x = self.pool4(x)                       # (B, 1, 321, 321)

        x = self.final_conv(x)                  # (B, 1, 321, 321)

        # Resize to target size in case of rounding differences
        if x.shape[-2:] != (self.image_height, self.image_width):
            x = F.interpolate(x, size=(self.image_height, self.image_width),
                              mode='bilinear', align_corners=False)

        return torch.sigmoid(x)                 # (B, 1, H, W)


TRAINED_DIR = Path(__file__).parent / "trained"


class BGsCNNModel:

    def __init__(self,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 epochs: int = 10,
                 batch_size: int = 32,
                 learning_rate: float = 1e-4,
                 image_height: int = 321,
                 image_width: int = 321,
                 checkpoint_name: str = "bgscnn_v2.pt"):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.image_height = image_height
        self.image_width = image_width
        self.model: BgsCNNV2 | None = None
        self.background_modeled = False
        self._bg_mean: np.ndarray | None = None
        self.checkpoint_path = TRAINED_DIR / checkpoint_name
        print(f"[BGsCNNModel] device={self.device}, epochs={self.epochs}, lr={self.learning_rate}")
        print(f"[BGsCNNModel] checkpoint={self.checkpoint_path}")

    def _prepare_input(self, frame_bgr: np.ndarray, bg_bgr: np.ndarray) -> torch.Tensor:
        frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
        bg_rgb    = cv.cvtColor(bg_bgr,    cv.COLOR_BGR2RGB).astype(np.float32) / 255.0

        frame_rgb = cv.resize(frame_rgb, (self.image_width, self.image_height))
        bg_rgb    = cv.resize(bg_rgb,    (self.image_width, self.image_height))

        cube = np.concatenate([frame_rgb, bg_rgb], axis=2)          # (H, W, 6)
        cube = torch.from_numpy(cube).permute(2, 0, 1).unsqueeze(0) # (1, 6, H, W)
        return cube.to(self.device)


    def _save_checkpoint(self):
        TRAINED_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "bg_mean": self._bg_mean,
            "image_height": self.image_height,
            "image_width": self.image_width,
        }, self.checkpoint_path)
        print(f"[BGsCNNModel] Checkpoint saved → {self.checkpoint_path}")

    def _load_checkpoint(self) -> bool:
        if not self.checkpoint_path.exists():
            return False
        print(f"[BGsCNNModel] Found checkpoint {self.checkpoint_path}, loading …")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.image_height = ckpt["image_height"]
        self.image_width  = ckpt["image_width"]
        self._bg_mean     = ckpt["bg_mean"]
        self.model = BgsCNNV2(self.image_height, self.image_width).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.background_modeled = True
        print("[BGsCNNModel] Checkpoint loaded, skipping training.\n")
        return True

    def modelize_back(self, frames: np.ndarray, gt_masks: np.ndarray | None = None):
        # Check for existing checkpoint first
        if self._load_checkpoint():
            return

        N = len(frames)
        print(f"[BGsCNNModel] Training on {N} frames …")

        if frames.ndim == 3:
            frames = np.stack([cv.cvtColor(f, cv.COLOR_GRAY2BGR) for f in frames])

        self._bg_mean = frames.mean(axis=0).astype(np.uint8)   # (H, W, 3)

        if gt_masks is None:
            gt_masks = np.zeros((N, frames.shape[1], frames.shape[2]), dtype=np.uint8)

        inputs_list, targets_list = [], []
        for i in tqdm(range(N), desc="  Preparing inputs", unit="fr"):
            inp = self._prepare_input(frames[i], self._bg_mean)
            inputs_list.append(inp)

            mask = gt_masks[i].astype(np.float32) / 255.0
            mask = cv.resize(mask, (self.image_width, self.image_height),
                             interpolation=cv.INTER_NEAREST)
            tgt = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).to(self.device)
            targets_list.append(tgt)

        self.model = BgsCNNV2(self.image_height, self.image_width).to(self.device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        epoch_bar = tqdm(range(self.epochs), desc="  Training", unit="epoch")
        for epoch in epoch_bar:
            indices = np.random.permutation(N)
            total_loss, n_batches = 0.0, 0

            for start in range(0, N, self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                inp_batch = torch.cat([inputs_list[j] for j in batch_idx], dim=0)
                tgt_batch = torch.cat([targets_list[j] for j in batch_idx], dim=0)

                optimizer.zero_grad()
                out  = self.model(inp_batch)
                loss = criterion(out, tgt_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches  += 1

            avg_loss = total_loss / n_batches
            epoch_bar.set_postfix(loss=f"{avg_loss:.4f}")

        self.model.eval()
        self.background_modeled = True
        self._save_checkpoint()
        print("[BGsCNNModel] Training complete.\n")

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if not self.background_modeled:
            raise RuntimeError("Call modelize_back() before inference.")

        if frame.ndim == 2:
            frame = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)

        orig_h, orig_w = frame.shape[:2]
        inp = self._prepare_input(frame, self._bg_mean)

        with torch.no_grad():
            prob = self.model(inp).squeeze().cpu().numpy()   # (Ht, Wt)

        prob = cv.resize(prob, (orig_w, orig_h), interpolation=cv.INTER_LINEAR)
        return prob > 0.5
