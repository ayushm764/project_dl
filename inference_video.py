import os
import argparse
import sys
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

import config
from models.detector import AnomalyDetector


class VideoFrameDataset(Dataset):
    """Dataset for inference on extracted video frames."""
    def __init__(self, frames_tensor):
        self.frames = frames_tensor
        
    def __len__(self):
        return len(self.frames)
        
    def __getitem__(self, idx):
        return self.frames[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Video Anomaly Detection on a new video file.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video file (.mp4, .avi, etc.)")
    parser.add_argument("--model_path", type=str, default=config.MODEL_SAVE_PATH, help="Path to the trained autoencoder checkpoint.")
    parser.add_argument("--output_dir", type=str, default=config.OUTPUT_DIR, help="Directory to save output annotated video and plots.")
    return parser.parse_args()


def process_video():
    args = parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found at {args.video_path}")
        sys.exit(1)
        
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at {args.model_path}. You might need to train the model first.")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print(f"  UCSD Ped2 — Video Anomaly Inference")
    print(f"{'='*60}")
    print(f"  Input Video : {args.video_path}")
    print(f"  Model       : {args.model_path}")
    print(f"  Device      : {device}")
    
    # 1. Load the model and threshold
    print("\n[1/4] Loading model...")
    detector = AnomalyDetector(latent_dim=config.LATENT_DIM, device=device)
    detector.load(args.model_path)
    
    if detector.threshold is None:
        print("Error: The loaded model does not have a calibrated threshold. Please re-run calibration/training.")
        sys.exit(1)
        
    threshold = detector.threshold
        
    # 2. Extract frames from video
    print("\n[2/4] Reading frames from video...")
    cap = cv2.VideoCapture(args.video_path)
    original_frames = []
    processed_frames = []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        original_frames.append(frame) # Keep BGR frame for later visualization
        
        # Preprocess for the model: Grayscale -> Resize 64x64 -> Normalize [0, 1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        normalized = resized.astype(np.float32) / 255.0
        
        # Shape: (1, 64, 64)
        processed_frames.append(np.expand_dims(normalized, axis=0))
        
    cap.release()
    num_frames = len(original_frames)
    print(f"  Extracted {num_frames} frames.")
    
    if num_frames == 0:
        print("Error: No frames extracted. The video file might be empty or corrupt.")
        sys.exit(1)
        
    # 3. Run Inference
    print("\n[3/4] Running inference...")
    # Convert to tensor loader
    tensor_frames = torch.tensor(np.array(processed_frames), dtype=torch.float32)
    dataset = VideoFrameDataset(tensor_frames)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    predictions, errors = detector.predict(loader)
    
    num_anomalies = int(np.sum(predictions))
    print(f"  Found {num_anomalies} anomalous frames ({(num_anomalies/num_frames)*100:.1f}%).")
    
    # 4. Generate Output Video and Plot
    print("\n[4/4] Generating outputs...")
    
    # A) Plot the errors timeline
    plt.figure(figsize=(10, 5), dpi=config.PLOT_DPI)
    plt.plot(errors, label='Reconstruction Error', color='blue')
    plt.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.4f})')
    plt.fill_between(range(len(errors)), threshold, errors, where=(errors > threshold),
                     color='red', alpha=0.3, label='Anomalous Region')
    
    plt.title("Reconstruction Errors Over Time")
    plt.xlabel("Frame")
    plt.ylabel("MSE + SSIM Error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    video_basename = os.path.basename(args.video_path).split('.')[0]
    plot_path = os.path.join(args.output_dir, f"{video_basename}_inference_plot.png")
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    # B) Generate annotated video
    out_video_path = os.path.join(args.output_dir, f"{video_basename}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    
    for i, (frame, is_anomaly, err) in enumerate(zip(original_frames, predictions, errors)):
        annotated_frame = frame.copy()
        
        # Overlay error text
        text_color = (0, 0, 255) if is_anomaly else (0, 255, 0)
        label = "ANOMALY" if is_anomaly else "NORMAL"
        
        cv2.putText(annotated_frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(annotated_frame, f"Error: {err:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        cv2.putText(annotated_frame, label, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
        
        # Draw red border if anomalous
        if is_anomaly:
            cv2.rectangle(annotated_frame, (0, 0), (width-1, height-1), (0, 0, 255), 8)
            
        out_vid.write(annotated_frame)
        
    out_vid.release()
    
    print(f"\n{'='*60}")
    print(f"  Inference Complete!")
    print(f"{'='*60}")
    print(f"  Plot saved to  : {plot_path}")
    print(f"  Video saved to : {out_video_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    process_video()
