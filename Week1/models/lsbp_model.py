import cv2 as cv
import numpy as np

class LSBPModel():
    def __init__(self, nSamples=20, LSBPRadius=16, Tlower=2.0, Tupper=32.0, 
                 Tinc=1.0, Tdec=0.05, Rscale=10.0, Rincdec=0.005, 
                 noiseRemovalThresholdFacBG=0.0004, noiseRemovalThresholdFacFG=0.0008,
                 LSBPthreshold=8, minCount=2, learningRate=0.0):
        self.nSamples = nSamples
        self.LSBPRadius = LSBPRadius
        self.Tlower = Tlower
        self.Tupper = Tupper
        self.Tinc = Tinc
        self.Tdec = Tdec
        self.Rscale = Rscale
        self.Rincdec = Rincdec
        self.noiseRemovalThresholdFacBG = noiseRemovalThresholdFacBG
        self.noiseRemovalThresholdFacFG = noiseRemovalThresholdFacFG
        self.LSBPthreshold = LSBPthreshold
        self.minCount = minCount
        self.learningRate = learningRate
        self.bg_subtractor = None
        self.background_modeled = False
        
    def modelize_back(self, frames):
        self.bg_subtractor = cv.bgsegm.createBackgroundSubtractorLSBP(
            nSamples=self.nSamples,
            LSBPRadius=self.LSBPRadius,
            Tlower=self.Tlower,
            Tupper=self.Tupper,
            Tinc=self.Tinc,
            Tdec=self.Tdec,
            Rscale=self.Rscale,
            Rincdec=self.Rincdec,
            noiseRemovalThresholdFacBG=self.noiseRemovalThresholdFacBG,
            noiseRemovalThresholdFacFG=self.noiseRemovalThresholdFacFG,
            LSBPthreshold=self.LSBPthreshold,
            minCount=self.minCount
        )
        
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
