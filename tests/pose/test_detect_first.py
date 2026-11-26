import numpy as np

from core.pose.detect_first import detected_people


class FakeBody:
    def __init__(self, boxes):
        self.boxes = boxes
        self.pose_calls = 0

    def det_model(self, frame):
        return self.boxes

    def pose_model(self, frame, bboxes):
        self.pose_calls += 1
        n = len(bboxes)
        return np.ones((n, 26, 2)), np.full((n, 26), 0.9)


def test_empty_detection_never_calls_pose():
    for boxes in (None, np.zeros((0, 4))):
        body = FakeBody(boxes)
        keypoints, scores = detected_people(body, np.zeros((4, 4, 3)))
        assert body.pose_calls == 0
        assert keypoints.shape == (0, 17, 2) and scores.shape == (0, 17)


def test_keeps_coco17_for_each_detected_person():
    keypoints, scores = detected_people(FakeBody(np.zeros((2, 4))), np.zeros((4, 4, 3)))
    assert keypoints.shape == (2, 17, 2) and scores.shape == (2, 17)
