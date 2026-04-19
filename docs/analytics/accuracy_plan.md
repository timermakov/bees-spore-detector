# Analytics and Accuracy Plan

## Current baseline

- Existing OpenCV-based approach is approximately `~60%` quality in challenging microscopy scenes.
- Existing YOLO tooling in the repo supports stronger detection and tiled inference.

## Target

- Reach `~90%` detection quality on a curated validation subset.

## Protocol

1. Build fixed validation set with expert labels.
2. Evaluate OpenCV baseline: precision/recall/F1.
3. Evaluate YOLO mode with current best weights.
4. Run ablation:
   - confidence threshold sweep
   - image size sweep
   - SAHI tile size / overlap tuning
5. Select operating point maximizing F1 while keeping false positives acceptable for titration.

## Backend integration note

- API endpoint `POST /probes/{probe_id}/analyze` accepts mode (`yolo`/`opencv`), stores method in `probe_results.method`, and makes comparative analysis auditable.
