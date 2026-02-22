import numpy as np
from typing import Tuple

class GaussianModel():
    """
    Background substraction model. Each pixel is modeled independently as
    a gaussian distribution.
    """
    

    
    def __init__(self, K : int = 11):
        
        self.K = K
        self.background_modeled = False
        
    def modelize_back(self, frames):
        
        frames = frames.astype(np.float32)
        
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