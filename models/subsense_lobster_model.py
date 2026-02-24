import cv2
import numpy as np

from models.subsense.Python.Subsense import Subsense, Lobster

class _BaseLBSPModel():
    def __init__(self, model_instance):
        self.model = model_instance
        self.background_modeled = False
        self.inference_frame_count = 0

    def modelize_back(self, frames):
        total_frames = len(frames)
        print(f"\n[{self.__class__.__name__}] Starting warmup with {total_frames} frames...")
        
        for idx, frame in enumerate(frames, 1):
            # ctypes requieres contiguity in memory and strict uint8 type
            frame_c = np.ascontiguousarray(frame, dtype=np.uint8)
            _ = self.model.apply(frame_c)
            
            if idx % max(1, total_frames // 10) == 0 or idx == total_frames:
                progress = (idx / total_frames) * 100
                print(f"  Warmup progress: {idx}/{total_frames} frames ({progress:.1f}%)")
            
        self.background_modeled = True
        print(f"[{self.__class__.__name__}] Warmup completed!\n")

    def __call__(self, frame):
        if not self.background_modeled:
            raise RuntimeError("Background not modeled")
        
        self.inference_frame_count += 1
        
        frame_c = np.ascontiguousarray(frame, dtype=np.uint8)
        fg_mask = self.model.apply(frame_c)

        mask = fg_mask > 0
        
        if self.inference_frame_count % 10 == 0:
            print("Inference frame count:", self.inference_frame_count)

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