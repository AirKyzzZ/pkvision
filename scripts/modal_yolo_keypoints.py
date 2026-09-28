"""Extract 2D keypoints from video using YOLO-pose on Modal GPU."""
import modal
import sys

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .pip_install("ultralytics", "opencv-python-headless", "numpy", "torch")
)

app = modal.App("pkvision-yolo", image=image)
volume = modal.Volume.from_name("pkvision-data", create_if_missing=True)


@app.function(gpu="T4", timeout=300, volumes={"/data": volume})
def extract_keypoints(video_bytes: bytes, filename: str, fps: float = 30.0):
    """Run YOLO-pose on video, return per-frame keypoints."""
    import cv2
    import numpy as np
    import tempfile
    import os
    from ultralytics import YOLO

    # Save video to temp file
    with tempfile.NamedTemporaryFile(suffix=".mov", delete=False) as f:
        f.write(video_bytes)
        video_path = f.name

    # Load YOLO-pose model
    model = YOLO("yolo11n-pose.pt")

    # Read video
    cap = cv2.VideoCapture(video_path)
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_keypoints = []  # (T, 17, 3) — x, y, confidence
    all_boxes = []      # (T, 4) — x1, y1, x2, y2 of main person

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        if results and results[0].keypoints is not None and len(results[0].keypoints) > 0:
            # Pick the person with the largest bounding box (most likely the athlete)
            boxes = results[0].boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_idx = int(np.argmax(areas))

            kps = results[0].keypoints[best_idx]
            xy = kps.xy.cpu().numpy()[0]    # (17, 2)
            conf = kps.conf.cpu().numpy()[0]  # (17,)
            kp_with_conf = np.concatenate([xy, conf[:, None]], axis=1)  # (17, 3)
            all_keypoints.append(kp_with_conf)
            all_boxes.append(boxes[best_idx])
        else:
            all_keypoints.append(np.zeros((17, 3)))
            all_boxes.append(np.zeros(4))

        frame_idx += 1

    cap.release()
    os.unlink(video_path)

    keypoints = np.array(all_keypoints)  # (T, 17, 3)
    boxes = np.array(all_boxes)          # (T, 4)

    return {
        "keypoints": keypoints.tolist(),
        "boxes": boxes.tolist(),
        "fps": actual_fps,
        "total_frames": total_frames,
        "filename": filename,
    }


@app.local_entrypoint()
def main(video_path: str = "data/run_testing/IMG_5985.mov"):
    import json
    from pathlib import Path

    stem = Path(video_path).stem
    output_path = f"data/keypoints/{stem}_keypoints.json"

    print(f"Processing {video_path}...")
    video_bytes = Path(video_path).read_bytes()
    filename = Path(video_path).name

    result = extract_keypoints.remote(video_bytes, filename)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f)

    n_frames = len(result["keypoints"])
    print(f"Saved {n_frames} frames of keypoints to {output_path}")
    print(f"FPS: {result['fps']}")
