import numpy as np

class AdaptiveGaussianModel():
    """
    Background subtraction model with adaptive Gaussian distributions.
    """
    def __init__(self, K: float = 2.5, p: float = 0.05, use_median: bool = False):
        self.K = K
        self.p = p
        self.use_median = use_median
        self.background_modeled = False
        
    def modelize_back(self, frames):
        frames = frames.astype(np.float32)
        
        if self.use_median:
            self.means = np.median(frames, axis=0)
        else:
            self.means = np.mean(frames, axis=0)

        self.variances = np.var(frames, axis=0)
        self.stds = np.sqrt(self.variances) + 2
        
        self.background_modeled = True
       
    def _shadow_detection(self, img : np.ndarray):
        
        img = img.astype(float)
        
        dot = img * self.means
        back_norm = self.means * self.means
        BD = dot/ (back_norm + 1e-8)
        V = BD * self.means
        CD = np.abs(img - BD * self.means)
        
        return BD, CD, V
        
    def __call__(self, frame):
        if not self.background_modeled:
            raise RuntimeError("Background not modeled")
        
        frame = frame.astype(np.float32)
        
        lower_bound = self.means - self.K * self.stds
        upper_bound = self.means + self.K * self.stds
        mask = (frame < lower_bound) | (frame > upper_bound)
        
        BD, CD, V = self._shadow_detection(frame)
        shadow_mask = (CD < 10) & ( ((1 > BD) & (BD > 0.5)) |  ((1.25 > BD) & (BD > 1)))
        mask[shadow_mask] = 0
        
        bg_mask = ~mask
        
        self.means[bg_mask] = self.p * frame[bg_mask] + (1 - self.p) * self.means[bg_mask]
        self.variances[bg_mask] = self.p * ((frame[bg_mask] - self.means[bg_mask])**2) + (1 - self.p) * self.variances[bg_mask]
        
        self.stds[bg_mask] = np.sqrt(self.variances[bg_mask]) + 2
        
        return mask