#!/bin/bash
# Extract keypoints on Mac, then deploy + train on GPU — fully automated
# Run with: nohup bash scripts/extract_then_train.sh > overnight.log 2>&1 &

set -e

REMOTE="pc@100.113.246.33"
SSH_OPTS="-o RemoteCommand=none -o RequestTTY=no"
PYTHON="C:\\Users\\pc\\gvhmr_env\\Scripts\\python.exe"
PKL="data/keypoints/parkourtheory_keypoints.pkl"

echo "$(date) === STEP 1: Wait for keypoint extraction ==="

# Wait for extraction to finish (check for the .pkl file)
while [ ! -f "$PKL" ]; do
    echo "$(date) Waiting for extraction... (checking every 60s)"
    sleep 60
done

SIZE=$(du -h "$PKL" | cut -f1)
echo "$(date) Extraction done! Keypoints: $SIZE"

echo ""
echo "$(date) === STEP 2: Transfer to windows-dev ==="
scp $SSH_OPTS "$PKL" "$REMOTE:C:/Users/pc/pkvision/data/keypoints/"
echo "$(date) Transfer complete"

echo ""
echo "$(date) === STEP 3: Install deps on windows-dev ==="
ssh $SSH_OPTS $REMOTE "$PYTHON -m pip install scipy ultralytics 2>&1 | tail -3"

echo ""
echo "$(date) === STEP 4: Start training on GPU (200 epochs) ==="
ssh $SSH_OPTS $REMOTE "cd C:\\Users\\pc\\pkvision && $PYTHON -u scripts/train_skeleton_classifier.py --skip-extract --epochs 200 --batch-size 32 --lr 3e-4 --device cuda" 2>&1 | tee data/keypoints/train_output.txt

echo ""
echo "$(date) === TRAINING COMPLETE ==="
echo "Check results in data/keypoints/train_output.txt"
