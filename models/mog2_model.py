import cv2 as cv
import numpy as np

class MOG2Model():
    def __init__(self, history=500, varThreshold=16, detectShadows=True, learningRate=0.0):
        self.history = history
        self.varThreshold = varThreshold
        self.detectShadows = detectShadows
        self.learningRate = learningRate
        self.bg_subtractor = None
        self.background_modeled = False
        
    def modelize_back(self, frames):
        self.bg_subtractor = cv.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.varThreshold,
            detectShadows=self.detectShadows
        )
        
        for frame in frames:
            self.bg_subtractor.apply(frame, learningRate=-1)
        
        self.background_modeled = True
        
    def __call__(self, frame):
        if not self.background_modeled:
            raise RuntimeError("Background not modeled")
        
        # Use the configured learning rate instead of 0
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.learningRate)
        
        mask = fg_mask == 255
        
        return mask
