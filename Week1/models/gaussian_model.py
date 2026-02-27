import numpy as np
from typing import Tuple

class GaussianModel():
    """
    Background substraction model. Each pixel is modeled independently as
    a gaussian distribution.
    """
    

    
    def __init__(self, K : int = 7, use_median : bool = False):
        
        self.K = K
        self.background_modeled = False
        self.use_median = use_median
        
    def modelize_back(self, frames):
        
        frames = frames.astype(np.float32)
        
        if self.use_median:
            self.means = np.median(frames, axis=0)
        else:
            self.means = np.mean(frames, axis=0)

        self.stds = np.std(frames, axis=0) + 2
        
        self.background_modeled = True
        
    def __call__(self, frame):
        if not self.background_modeled:
            raise RuntimeError("Background not modeled")
        
        lower_bound = self.means - self.K*self.stds
        upper_bound = self.means + self.K*self.stds
        mask = (frame < lower_bound) | (frame > upper_bound)
        
        return mask 


class GaussianModelShadow():
    """
    Background substraction model. Each pixel is modeled independently as
    a gaussian distribution.
    """
    

    
    def __init__(self, K : int = 7, use_median : bool = False):
        
        self.K = K
        self.background_modeled = False
        self.use_median = use_median
        
    def modelize_back(self, frames):
        
        frames = frames.astype(np.float32)
        
        if self.use_median:
            self.means = np.median(frames, axis=0)
        else:
            self.means = np.mean(frames, axis=0)

        self.stds = np.std(frames, axis=0) + 2
        
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
        
        lower_bound = self.means - self.K*self.stds
        upper_bound = self.means + self.K*self.stds
        mask = (frame < lower_bound) | (frame > upper_bound)
        
        BD, CD, V = self._shadow_detection(frame)
        shadow_mask = (CD < 10) & ( ((1 > BD) & (BD > 0.5)) |  ((1.25 > BD) & (BD > 1)))
        mask[shadow_mask] = 0
        
        return mask 