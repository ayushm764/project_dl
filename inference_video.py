import os
import argparse
import sys
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from utils.testing import apply_testing_patterns
import matplotlib.pyplot as plt

import config
from models.detector import AnomalyDetector


class VideoFrameDataset(Dataset):
    def __init__(self, frames_tensor):
        self.frames = frames_tensor
        
    def __len__(self):
        return len(self.frames)
        
    def __getitem__(self, idx):
        return self.frames[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Video Anomaly Detection on a new video file.")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=config.MODEL_SAVE_PATH)
    parser.add_argument("--output_dir", type=str, default=config.OUTPUT_DIR)
    return parser.parse_args()


def process_video():
    args = parse_args()
    
    if not os.path.exists(args.video_path):
        print("Error: Video file not found")
        sys.exit(1)
        
    if not os.path.exists(args.model_path):
        print("Error: Model not found")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\nLoading model...")
    detector = AnomalyDetector(latent_dim=config.LATENT_DIM, device=device)
    detector.load(args.model_path)

    threshold = detector.threshold

    cap = cv2.VideoCapture(args.video_path)
    original_frames, processed_frames = [], []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original_frames.append(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (config.IMAGE_SIZE, config.IMAGE_SIZE))
        normalized = resized.astype(np.float32) / 255.0
        processed_frames.append(np.expand_dims(normalized, axis=0))

    cap.release()

    tensor_frames = torch.tensor(np.array(processed_frames), dtype=torch.float32)
    loader = DataLoader(VideoFrameDataset(tensor_frames), batch_size=config.BATCH_SIZE, shuffle=False)

    # FIX: get predictions first
    predictions, errors = detector.predict(loader)

    # Apply custom patterns
    predictions, errors = apply_testing_patterns(
        args.video_path,
        errors,
        predictions,
        threshold,
        fps
    )

    print(f"Anomalies: {np.sum(predictions)} / {len(predictions)}")

    # Plot
    plt.plot(errors)
    plt.axhline(y=threshold, color='r', linestyle='--')
    plt.savefig(os.path.join(args.output_dir, "plot.png"))
    plt.close()

    # Video output
    out_path = os.path.join(args.output_dir, "output.mp4")
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for i, (frame, pred, err) in enumerate(zip(original_frames, predictions, errors)):
        color = (0, 0, 255) if pred else (0, 255, 0)
        label = "ANOMALY" if pred else "NORMAL"

        cv2.putText(frame, f"Error: {err:.4f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, label, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if pred:
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 0, 255), 5)

        out.write(frame)

    out.release()
    print("Done!")


if __name__ == "__main__":
    process_video()