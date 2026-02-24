import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add bgsCNN to path
bgscnn_path = Path(__file__).parent.parent / "bgsCNN"
sys.path.insert(0, str(bgscnn_path))

try:
    from model import BGsCNN
except ImportError:
    raise ImportError(f"Could not import BGsCNN model from {bgscnn_path}")


class BGsCNNModel():
    """
    Background subtraction using BGsCNN (Convolutional Neural Network).
    Trains on initial frames and performs inference on subsequent frames.
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 epochs=10, batch_size=1, learning_rate=1e-4):
        """
        Initialize BGsCNN model.
        
        Args:
            device: Device to run model on ('cuda' or 'cpu')
            epochs: Number of training epochs during background modeling
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.background_modeled = False
        self.inference_frame_count = 0
        
        print(f"[BGsCNNModel] Initialized with device: {self.device}")
        
    def modelize_back(self, frames):
        """
        Train the BGsCNN model on background frames.
        
        Args:
            frames: numpy array of shape (n_frames, height, width) - grayscale images
        """
        print(f"\n[BGsCNNModel] Starting training with {len(frames)} frames...")
        
        # Initialize model
        if len(frames.shape) == 3:
            # Grayscale
            height, width = frames[0].shape
            in_channels = 1
        else:
            raise ValueError("Expected grayscale frames of shape (n_frames, height, width)")
        
        self.model = BGsCNN(in_channels=in_channels).to(self.device)
        self.model.train()
        
        # Prepare optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()
        
        # Convert frames to tensor
        frames_tensor = torch.from_numpy(frames).float().unsqueeze(1) / 255.0  # (N, 1, H, W)
        
        # Create simple background mask (all zeros since these are background frames)
        bg_masks = torch.zeros((len(frames), 1, height, width))
        
        # Training loop
        for epoch in range(self.epochs):
            total_loss = 0
            n_batches = 0
            
            # Simple batching
            for i in range(0, len(frames), self.batch_size):
                batch_frames = frames_tensor[i:i+self.batch_size].to(self.device)
                batch_masks = bg_masks[i:i+self.batch_size].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_frames)
                loss = criterion(outputs, batch_masks)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            print(f"  Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
        
        self.model.eval()
        self.background_modeled = True
        print(f"[BGsCNNModel] Training completed!\n")
        
    def __call__(self, frame):
        """
        Apply background subtraction on a single frame.
        
        Args:
            frame: numpy array of shape (height, width) - grayscale image
            
        Returns:
            Binary mask where True indicates foreground
        """
        if not self.background_modeled:
            raise RuntimeError("Background not modeled. Call modelize_back() first.")
        
        self.inference_frame_count += 1
        
        # Ensure grayscale
        if len(frame.shape) == 3:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Prepare frame for model
        frame_tensor = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0) / 255.0  # (1, 1, H, W)
        frame_tensor = frame_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(frame_tensor)
            fg_prob = output.squeeze().cpu().numpy()
        
        # Threshold to get binary mask
        mask = fg_prob > 0.5
        
        if self.inference_frame_count % 50 == 0:
            print(f"  [BGsCNNModel] Inference frame: {self.inference_frame_count}")
        
        return mask
    
    def __del__(self):
        """Cleanup GPU memory"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
