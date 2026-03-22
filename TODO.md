# PkVision — Roadmap

## Infrastructure

- [ ] **Docker + docker-compose** — Containerize the full stack (API + worker + Redis + PostgreSQL). Single `docker-compose up` to run everything. GPU passthrough support for training with NVIDIA Container Toolkit. Separate `Dockerfile.api` (lightweight, inference only) and `Dockerfile.train` (full ML deps + CUDA/MPS).
- [ ] **CI/CD with GitHub Actions** — Auto-run `pytest` on every push/PR. Lint with `ruff`. Type-check with `mypy`. Block merge if tests fail. Separate workflows for `test.yml` (fast, no GPU), `train-smoke.yml` (weekly, runs 5-epoch training on fixture data to catch regressions).
- [ ] **PostgreSQL migration** — Switch from SQLite to PostgreSQL for production. Alembic for schema migrations. Keep SQLite as dev default.
- [ ] **Pre-commit hooks** — ruff format + ruff check + mypy on staged files.

## Real-time Pipeline

- [ ] **Live webcam analysis** — WebSocket endpoint that streams detections in real-time. YOLO processes frames at ~15fps, detection runs on sliding windows, scores update live.
- [ ] **RTMP/RTSP stream input** — Accept live video feeds from competition cameras. Integrate with OBS or professional streaming setups.
- [ ] **Low-latency mode** — Optimized pipeline for sub-second detection: skip frames, reduce YOLO input resolution, batch inference.
- [ ] **Live overlay** — OpenCV overlay on video feed showing skeleton, detected trick name, confidence bar, running score.

## Benchmark & Metrics

- [ ] **Accuracy benchmark suite** — Standard test set of labeled clips with ground truth. Report per-trick precision, recall, F1. Compare angle threshold vs ST-GCN accuracy.
- [ ] **Confusion matrix dashboard** — Visual confusion matrix after each training run. Identify which tricks get confused (e.g. gainer vs back flip).
- [ ] **Latency profiling** — Measure end-to-end time: video load → pose extraction → detection → scoring. Track per-component timing. Target: < 2x video duration for offline, < 100ms per frame for real-time.
- [ ] **Model versioning** — Track model versions with metrics (accuracy, loss, training data size). MLflow or W&B integration for experiment tracking.
- [ ] **Regression tests** — Golden test set that must maintain >= X% accuracy. Fail CI if a model change drops below threshold.

## Detection Improvements

- [ ] **3D pose estimation** — Integrate MotionBERT or VideoPose3D for monocular 3D pose lifting. Critical for twist detection (rotations along camera axis).
- [ ] **Multi-camera fusion** — Combine 2+ camera angles for true 3D keypoints. Triangulation pipeline. Required for competition-grade accuracy.
- [ ] **Execution quality scoring** — Rate landing stability, body alignment, height. Separate D-score (difficulty) and E-score (execution) like FIG gymnastics.
- [ ] **Combo detection** — Detect trick sequences (back flip → twist → landing). Score combos with flow/transition bonuses.
- [ ] **Trick phase visualization** — Show approach/takeoff/execution/landing phases overlaid on the video timeline.

## Community & Ecosystem

- [ ] **Web dashboard** — Next.js frontend for uploading videos, viewing results, browsing the trick catalog. Dark theme, responsive.
- [ ] **Clip submission portal** — Web form (not just GitHub Issues) for athletes to submit clips. Upload to S3/Blob, auto-notify maintainers.
- [ ] **Leaderboard** — Public leaderboard of highest-scoring runs submitted by the community.
- [ ] **Mobile app** — React Native app for filming + instant analysis. Film a trick, get immediate feedback on what was detected.
- [ ] **Multilingual catalog** — Expand beyond EN/FR: ES, DE, PT, JP, AR. Community-contributed translations.
- [ ] **Plugin system** — Allow third-party detection strategies. Community can train specialized models for niche tricks and share them.

## Competition Integration

- [ ] **FIG notation export** — Export analysis results in FIG-compatible notation format. Align difficulty ratings with official Code of Points.
- [ ] **Judge tablet interface** — iPad-optimized UI for competition judges. View AI suggestions, apply overrides, submit final scores.
- [ ] **Multi-athlete tracking** — Detect and track multiple athletes in the same frame. Assign tricks to specific athletes.
- [ ] **Competition mode** — Locked-down mode with audit logging, no model updates during competition, tamper-evident results.
- [ ] **Replay system** — Slow-motion replay with skeleton overlay for judges to review contested detections.
