import cv2 as cv
import numpy as np

class MOGModel():
    def __init__(self, history=500, nmixtures=5, backgroundRatio=0.7, noiseSigma=0, learningRate=0.0):
        self.history = history
        self.nmixtures = nmixtures
        self.backgroundRatio = backgroundRatio
        self.noiseSigma = noiseSigma
        self.learningRate = learningRate
        self.bg_subtractor = None
        self.background_modeled = False
        
    def modelize_back(self, frames):
        self.bg_subtractor = cv.bgsegm.createBackgroundSubtractorMOG(
            history=self.history,
            nmixtures=self.nmixtures,
            backgroundRatio=self.backgroundRatio,
            noiseSigma=self.noiseSigma
        )
        
        # Train with background frames
        for frame in frames:
            self.bg_subtractor.apply(frame, learningRate=-1)
        
        self.background_modeled = True
        
    def __call__(self, frame):
        if not self.background_modeled:
            raise RuntimeError("Background not modeled")
        
        # Use the configured learning rate instead of 0
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learningRate)
        
        mask = fg_mask > 0
        
        return mask
