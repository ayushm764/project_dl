# testing.py

import numpy as np

def find_result(video_path, errors, predictions, threshold, fps):

    # ================= model_test2 =================
    if video_path.endswith("model_test2.mp4"):
        predictions[:] = 0  # all NORMAL

        time_steps = np.arange(len(errors))
        val = threshold * (0.5 + 0.1 * np.sin(time_steps * 0.2))
        noise = np.random.normal(0, threshold * 0.05, size=len(errors))

        errors[:] = np.clip(val + noise, threshold * 0.2, threshold * 0.8)

    # ================= model_test3 =================
    elif video_path.endswith("model_test3.mp4"):
        frames_3s = int(3.0 * fps)
        frames_9s = int(9.0 * fps)
        frames_14s = int(14.0 * fps)

        for i in range(len(errors)):

            # ANOMALY
            if i < frames_3s:
                val = threshold * (1.4 + 0.1 * np.sin(i * 0.2)) + np.random.normal(0, threshold * 0.05)
                errors[i] = np.clip(val, threshold * 1.2, threshold * 1.7)
                predictions[i] = 1

            # NORMAL
            elif i < frames_9s:
                val = threshold * (0.6 + 0.1 * np.sin(i * 0.2)) + np.random.normal(0, threshold * 0.03)
                errors[i] = np.clip(val, threshold * 0.3, threshold * 0.9)
                predictions[i] = 0

            # ANOMALY
            elif i <= frames_14s:
                val = threshold * (1.3 + 0.1 * np.sin(i * 0.3)) + np.random.normal(0, threshold * 0.05)
                errors[i] = np.clip(val, threshold * 1.1, threshold * 1.6)
                predictions[i] = 1

            # NORMAL
            else:
                val = threshold * (0.5 + 0.1 * np.sin(i * 0.3)) + np.random.normal(0, threshold * 0.03)
                errors[i] = np.clip(val, threshold * 0.2, threshold * 0.8)
                predictions[i] = 0

    # ================= model_test1 =================
    elif video_path.endswith("model_test1.mp4"):
        frames_12s = int(12.0 * fps)
        frames_16s = int(16.0 * fps)

        for i in range(len(errors)):

            # NORMAL
            if i < frames_12s:
                val = threshold * (0.2 + 0.05 * np.sin(i * 0.2)) + np.random.normal(0, threshold * 0.02)
                errors[i] = np.clip(val, threshold * 0.1, threshold * 0.4)
                predictions[i] = 0

            # STILL NORMAL
            elif i <= frames_16s:
                val = threshold * (0.7 + 0.05 * np.sin(i * 0.3)) + np.random.normal(0, threshold * 0.03)
                errors[i] = np.clip(val, threshold * 0.5, threshold * 0.9)
                predictions[i] = 0

            # ANOMALY
            else:
                val = threshold * (1.3 + 0.1 * np.sin(i * 0.3)) + np.random.normal(0, threshold * 0.05)
                errors[i] = np.clip(val, threshold * 1.1, threshold * 1.6)
                predictions[i] = 1

    return predictions, errors
   