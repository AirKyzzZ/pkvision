import numpy as np

from core.pose.athlete_select import build_tracks, select_athlete

FPS = 30.0


def person(cx, cy, height=100.0, angle_deg=0.0, conf=0.9):
    """17 joints spread along a body axis at angle_deg; shoulders above hips when upright."""
    a = np.radians(angle_deg)
    axis, side = np.array([np.sin(a), -np.cos(a)]), np.array([np.cos(a), np.sin(a)])
    joints = np.zeros((17, 3), np.float32)
    along = np.linspace(0.5, -0.5, 17)
    for i in range(17):
        joints[i, :2] = np.array([cx, cy]) + height * along[i] * axis + height * 0.1 * ((-1) ** i) * side
    for left, right, offset in ((5, 6, 0.3), (11, 12, -0.1)):
        for idx, sign in ((left, 1), (right, -1)):
            joints[idx, :2] = np.array([cx, cy]) + height * offset * axis + sign * height * 0.12 * side
    joints[:, 2] = conf
    return joints


def flip_frames(n=60, spectator_height=300.0):
    """A small rotating athlete and a large, still spectator."""
    frames = []
    for t in range(n):
        athlete = person(100 + 2 * t, 300, angle_deg=t * 18.0)
        spectator = person(600, 300, height=spectator_height)
        frames.append(np.stack([spectator, athlete]))
    return frames


def test_rotating_athlete_beats_larger_still_spectator():
    out = select_athlete(flip_frames(), FPS)
    assert np.allclose(out[10, 0, 0], person(120, 300, angle_deg=180.0)[0, 0], atol=1e-2)


def test_order_of_people_in_frame_does_not_matter():
    a = select_athlete(flip_frames(), FPS)
    b = select_athlete([f[::-1] for f in flip_frames()], FPS)
    assert np.allclose(a, b)


def test_only_one_person_is_kept_through_detector_dropout():
    frames = flip_frames()
    for t in range(20, 26):
        frames[t] = frames[t][:1]  # athlete undetected for 6 frames (0.2 s)
    out = select_athlete(frames, FPS)
    assert (out[20:26, :, 2] == 0).all()
    assert (out[19, :, 2] > 0).all() and (out[26, :, 2] > 0).all()
    assert out[26, :, 0].mean() < 400  # back on the athlete, not the spectator at x=600


def test_two_people_do_not_merge_into_one_track():
    tracks = build_tracks(flip_frames(n=30), FPS)
    assert len(tracks) == 2


def test_camera_cut_starts_a_new_track():
    frames = [person(100, 100)[None] for _ in range(10)] + [person(900, 700)[None] for _ in range(10)]
    assert len(build_tracks(frames, FPS)) == 2


def test_empty_and_low_confidence_frames_give_zeros():
    frames = [np.zeros((0, 17, 3), np.float32), person(50, 50, conf=0.1)[None]]
    out = select_athlete(frames, FPS)
    assert out.shape == (2, 17, 3) and (out == 0).all()


def test_still_keypoint_jitter_does_not_outscore_a_flip():
    rng = np.random.default_rng(0)
    frames = flip_frames()
    for f in frames:
        f[0, :, :2] += rng.normal(0, 1.0, (17, 2))
    out = select_athlete(frames, FPS)
    assert out[10, :, 0].mean() < 400
