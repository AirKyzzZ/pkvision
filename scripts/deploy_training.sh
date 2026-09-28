#!/bin/bash
# Deploy keypoints + start training on windows-dev GPU
# Run this after keypoint extraction finishes on Mac
#
# Usage: bash scripts/deploy_training.sh

set -e

REMOTE="pc@100.113.246.33"
SSH_OPTS="-o RemoteCommand=none -o RequestTTY=no"
PYTHON="C:\\Users\\pc\\gvhmr_env\\Scripts\\python.exe"

echo "=== Checking keypoint extraction ==="
if [ ! -f "data/keypoints/parkourtheory_keypoints.pkl" ]; then
    echo "ERROR: Keypoints not extracted yet!"
    echo "Check: tail -f data/keypoints/extract_log.txt"
    exit 1
fi

SIZE=$(du -h data/keypoints/parkourtheory_keypoints.pkl | cut -f1)
echo "Keypoints file: $SIZE"

echo ""
echo "=== Transferring keypoints to windows-dev ==="
scp $SSH_OPTS data/keypoints/parkourtheory_keypoints.pkl "$REMOTE:C:/Users/pc/pkvision/data/keypoints/"
echo "Transfer complete"

echo ""
echo "=== Starting training on GPU ==="
ssh $SSH_OPTS $REMOTE "cd C:\\Users\\pc\\pkvision && start /b $PYTHON -u scripts/train_skeleton_classifier.py --skip-extract --epochs 200 --batch-size 32 --lr 3e-4 --device cuda > train_skeleton.log 2>&1"

echo ""
echo "Training started on windows-dev!"
echo "Monitor: ssh windows-dev 'type C:\\Users\\pc\\pkvision\\train_skeleton.log'"
echo "Check GPU: ssh windows-dev 'nvidia-smi'"
