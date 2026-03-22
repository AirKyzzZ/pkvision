#!/bin/bash
# PkVision — Quick training pipeline for backflip detection
# Run this after downloading clips and filling training_batch.json
#
# Usage: bash scripts/train_backflip.sh

set -e

echo "=========================================="
echo "  PkVision — Backflip Training Pipeline"
echo "=========================================="

CLIPS_DIR="data/clips"
LABELS="data/clips/labels.json"
KEYPOINTS_DIR="data/clips/keypoints"
MODEL_PATH="data/models/stgcn_best.pt"

# Step 1: Download clips from batch file (if batch exists and has entries)
if [ -f "$CLIPS_DIR/training_batch.json" ]; then
    ENTRY_COUNT=$(python3 -c "import json; data=json.load(open('$CLIPS_DIR/training_batch.json')); print(len([x for x in data if isinstance(x, dict) and 'url' in x]))")
    if [ "$ENTRY_COUNT" -gt 0 ]; then
        echo ""
        echo "[1/4] Downloading $ENTRY_COUNT clips from YouTube..."
        python3 scripts/download_clips.py --batch "$CLIPS_DIR/training_batch.json" --output "$CLIPS_DIR"
    else
        echo ""
        echo "[1/4] No batch entries found, using existing clips..."
    fi
else
    echo ""
    echo "[1/4] No batch file found, using existing clips..."
fi

# Check we have labels
if [ ! -f "$LABELS" ] || [ "$(python3 -c "import json; print(len(json.load(open('$LABELS'))))")" -eq 0 ]; then
    echo ""
    echo "ERROR: No labeled clips found in $LABELS"
    echo "Run: python3 scripts/label.py --clips-dir $CLIPS_DIR"
    exit 1
fi

LABEL_COUNT=$(python3 -c "import json; print(len(json.load(open('$LABELS'))))")
echo "  Found $LABEL_COUNT labeled clips"

# Step 2: Extract poses
echo ""
echo "[2/4] Extracting poses with YOLO..."
python3 scripts/extract_poses.py --clips-dir "$CLIPS_DIR" --labels "$LABELS" --output "$KEYPOINTS_DIR"

# Step 3: Train
echo ""
echo "[3/4] Training ST-GCN model..."
python3 scripts/train.py --keypoints-dir "$KEYPOINTS_DIR" --labels "$LABELS" --output "$MODEL_PATH" --epochs 100 --batch-size 8

# Step 4: Test
echo ""
echo "[4/4] Training complete!"
echo ""
echo "Model saved to: $MODEL_PATH"
echo ""
echo "Test it with:"
echo "  python3 scripts/analyze.py --input YOUR_VIDEO.mp4"
echo ""
echo "=========================================="
