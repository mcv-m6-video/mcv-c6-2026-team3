import cv2
import numpy as np

from models.subsense.Python.Subsense import Subsense, Lobster

class _BaseLBSPModel():
    def __init__(self, model_instance):
        self.model = model_instance
        self.background_modeled = False

    def modelize_back(self, frames):
        for frame in frames:
            # ctypes requieres contiguity in memory and strict uint8 type
            frame_c = np.ascontiguousarray(frame, dtype=np.uint8)
            _ = self.model.apply(frame_c)
            
        self.background_modeled = True

    def __call__(self, frame):
        if not self.background_modeled:
            raise RuntimeError("Background not modeled")
        
        frame_c = np.ascontiguousarray(frame, dtype=np.uint8)
        fg_mask = self.model.apply(frame_c)
        
        unique_values = np.unique(fg_mask)
        print("Unique values in fg_mask:", unique_values)

        # El modelo ya devuelve valores 0 o 255, convertir a booleano para consistencia
        mask = fg_mask > 0
        
        return mask
    
    def __del__(self):
        if hasattr(self, 'model') and self.model is not None:
            try:
                self.model.release()
            except:
                pass


class SubsenseModel(_BaseLBSPModel):
    """
    Spatio-Temporal Background Subtraction with Local Features and Feedback.
    """
    def __init__(self):
        super().__init__(Subsense())


class LobsterModel(_BaseLBSPModel):
    """
    Local Binary Similarity Patterns (WACV 2014).
    """
    def __init__(self):
        super().__init__(Lobster())