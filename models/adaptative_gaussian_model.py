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
        
    def __call__(self, frame):
        if not self.background_modeled:
            raise RuntimeError("Background not modeled")
        
        frame = frame.astype(np.float32)
        
        lower_bound = self.means - self.K * self.stds
        upper_bound = self.means + self.K * self.stds
        mask = (frame < lower_bound) | (frame > upper_bound)
        
        bg_mask = ~mask
        
        self.means[bg_mask] = self.p * frame[bg_mask] + (1 - self.p) * self.means[bg_mask]
        self.variances[bg_mask] = self.p * ((frame[bg_mask] - self.means[bg_mask])**2) + (1 - self.p) * self.variances[bg_mask]
        
        self.stds[bg_mask] = np.sqrt(self.variances[bg_mask]) + 2
        
        return mask