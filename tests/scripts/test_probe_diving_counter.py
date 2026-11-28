from scripts import probe_diving_counter as probe


def fake_annotations():
    ann = {}
    for video in (1, 2, 7):
        for dive in range(1, 201):
            ann[(video, dive)] = {"tw_no": dive % 8, "ss_no": 5, "start_frame": dive * 300, "end_frame": dive * 300 + 90}
    return ann


def test_pilot_is_deterministic_stratified_and_skips_low_resolution_videos():
    ann = fake_annotations()
    chosen = probe.pilot(ann)
    assert chosen == probe.pilot(ann) and len(chosen) == len(set(chosen)) == probe.PILOT
    assert all(video in probe.VIDEOS_720P for video, _ in chosen)
    counts = {tw: sum(ann[k]["tw_no"] == tw for k in chosen) for tw in range(8)}
    assert counts[7] == 5 and counts[6] == 11 and counts[1] == 20 and counts[0] == probe.PILOT - sum(probe.QUOTA.values())


def test_an_unavailable_dive_is_replaced_within_its_stratum():
    ann = fake_annotations()
    chosen = probe.pilot(ann)
    gone = next(k for k in chosen if ann[k]["tw_no"] == 3)
    again = probe.pilot(ann, {gone})
    assert gone not in again and len(again) == probe.PILOT
    assert sum(ann[k]["tw_no"] == 3 for k in again) == probe.QUOTA[3]


def test_verdict_thresholds():
    assert probe.verdict(0.85, 0.80, 0.9) == "counter validated"
    assert probe.verdict(0.85, 0.80, 0.6) == "inconclusive"
    assert probe.verdict(0.85, 0.70, 0.9) == "inconclusive"
    assert probe.verdict(0.85, 0.59, 0.9) == "skeleton or counter is the problem"
    assert probe.verdict(0.9, 0.9, 0.4) == "skeleton or counter is the problem"


def test_score_counts_abstentions_in_coverage_only():
    rows = [{"flip": 2.5, "twist": 1.0, "true_somersault": 2.5, "true_twist": 1.0, "valid_frac": 1.0, "twist_shoulder": 1.0},
            {"flip": 2.0, "twist": 0.0, "true_somersault": 2.5, "true_twist": 0.0, "valid_frac": 0.8, "twist_shoulder": 0.0},
            {"flip": None, "twist": None, "true_somersault": 1.5, "true_twist": 0.0, "valid_frac": 0.2, "twist_shoulder": None}]
    out = probe.score(rows)
    assert out["answered"] == 2 and out["coverage"] == 0.667
    assert out["somersault_exact"] == 0.5 and out["twist_exact"] == 1.0 and out["somersault_within_half"] == 1.0
