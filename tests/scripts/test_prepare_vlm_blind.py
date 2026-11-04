import numpy as np

from scripts.prepare_vlm_blind import COLS, GRID_FRAMES, ROWS, TILE, grids, opaque_id, read_frames, write_video


def test_opaque_id_is_stable_and_hides_the_name():
    assert opaque_id("double_full") == opaque_id("double_full")
    assert "double" not in opaque_id("double_full")


def test_grids_cover_24_frames_on_two_sheets():
    frames = [np.full((120, 200, 3), i % 255, np.uint8) for i in range(100)]
    sheets = grids(frames)
    assert GRID_FRAMES == ROWS * COLS * 2 and len(sheets) == 2
    assert all(s.shape == (ROWS * TILE, COLS * TILE, 3) for s in sheets)


def test_short_clip_still_fills_the_grid():
    assert len(grids([np.zeros((50, 50, 3), np.uint8)] * 3)) == 2


def test_written_video_keeps_frame_order(tmp_path):
    frames = [np.full((64, 64, 3), v, np.uint8) for v in (20, 120, 220)]
    path = tmp_path / "x.mp4"
    write_video(frames[::-1], 30.0, path)
    decoded, _ = read_frames(path)
    means = [f.mean() for f in decoded]
    assert len(decoded) == 3 and means[0] > means[1] > means[2]
