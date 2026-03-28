"""Hybrid trick matcher — combines 3D physics + 2D motion signatures.

Uses two independent recognition paths and fuses their results:
1. Physics path: GVHMR SMPL params -> rotation tracking -> FIG matcher
   Best for: single flips, twist counting, axis classification
2. Signature path: ViTPose 2D keypoints -> DTW matching vs reference clips
   Best for: trick family classification, view-invariant recognition

The hybrid score weights each path based on its confidence and the trick complexity.

Usage:
    matcher = HybridMatcher(signature_db_path="data/signature_db_v3.pt")
    results = matcher.match_segment(
        tracking, start, end, global_orient, body_pose, transl,
        vitpose_segment, fps=30.0
    )
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from core.recognition.matcher import Matcher3D, TrickMatch
from core.recognition.motion_signature import SignatureDatabase, MotionSignature, SignatureMatch
from core.pose.rotation_tracker import extract_trick_physics
from ml.trick_physics import TRICK_DEFINITIONS


@dataclass
class HybridMatch:
    """Combined match result from both recognition paths."""
    trick_name: str
    trick_id: str
    hybrid_confidence: float
    physics_confidence: float
    signature_confidence: float
    d_score: float
    # Physics details
    flip_count: float
    twist_count: float
    direction: str
    axis: str
    body_shape: str
    went_inverted: bool
    # Source info
    physics_match_name: str
    signature_match_name: str


class HybridMatcher:
    """Combines physics-based and signature-based trick recognition."""

    def __init__(
        self,
        signature_db_path: str = "data/signature_db_v3.pt",
        physics_weight: float = 0.4,
        signature_weight: float = 0.6,
        min_segment_frames: int = 10,
    ):
        # Physics matcher
        self.physics_matcher = Matcher3D()

        # Signature matcher
        db_data = torch.load(signature_db_path, map_location="cpu", weights_only=False)
        self.sig_db = SignatureDatabase(norm_length=db_data["norm_length"])
        for ref_data in db_data["references"]:
            sig = MotionSignature(**{k: v for k, v in ref_data.items()})
            self.sig_db.references.append(sig)

        self.physics_weight = physics_weight
        self.signature_weight = signature_weight
        self.min_segment_frames = min_segment_frames

    def match_segment(
        self,
        tracking: dict,
        start: int,
        end: int,
        global_orient: np.ndarray,
        body_pose: np.ndarray,
        transl: np.ndarray,
        vitpose_segment: np.ndarray,
        fps: float = 30.0,
        top_k: int = 5,
    ) -> tuple[list[HybridMatch], dict]:
        """Match a trick segment using both paths.

        Returns:
            (matches, physics_dict) — hybrid matches and raw physics data
        """
        from scripts.analyze_3d import tracking_to_signature

        # Skip very short segments (likely noise)
        if end - start < self.min_segment_frames:
            return [], {}

        # ── Physics Path ──
        physics = extract_trick_physics(tracking, start, end, global_orient, body_pose, transl)
        signature = tracking_to_signature(physics, start, end, fps, transl)
        physics_matches = self.physics_matcher.match(signature, top_k=top_k * 2)

        # ── Signature Path (DTW) ──
        sig_matches = self.sig_db.match(vitpose_segment, fps=fps, top_k=top_k * 2)

        # ── Fusion ──
        # Build a unified candidate list from both paths
        candidates = {}

        # Add physics candidates
        for pm in physics_matches:
            tid = pm.trick_id
            candidates[tid] = {
                "trick_name": pm.trick_name,
                "trick_id": tid,
                "physics_conf": pm.confidence,
                "signature_conf": 0.0,
                "physics_match": pm.trick_name,
                "signature_match": "",
            }

        # Add/merge signature candidates
        for sm in sig_matches:
            # Find the FIG trick ID for this signature match name
            sig_tid = self._name_to_trick_id(sm.name)
            if sig_tid and sig_tid in candidates:
                # Same trick found by both paths — boost confidence
                candidates[sig_tid]["signature_conf"] = sm.confidence
                candidates[sig_tid]["signature_match"] = sm.name
            elif sig_tid:
                candidates[sig_tid] = {
                    "trick_name": sm.name,
                    "trick_id": sig_tid,
                    "physics_conf": 0.0,
                    "signature_conf": sm.confidence,
                    "physics_match": "",
                    "signature_match": sm.name,
                }
            else:
                # Signature match doesn't map to a FIG trick ID
                fake_id = sm.name.lower().replace(" ", "_")
                if fake_id not in candidates:
                    candidates[fake_id] = {
                        "trick_name": sm.name,
                        "trick_id": fake_id,
                        "physics_conf": 0.0,
                        "signature_conf": sm.confidence,
                        "physics_match": "",
                        "signature_match": sm.name,
                    }
                else:
                    candidates[fake_id]["signature_conf"] = max(
                        candidates[fake_id]["signature_conf"], sm.confidence
                    )

        # Compute hybrid scores
        results = []
        for tid, c in candidates.items():
            # Adaptive weighting: trust physics more for simple tricks,
            # signatures more for complex tricks
            p_conf = c["physics_conf"]
            s_conf = c["signature_conf"]

            # If both paths agree (both > 0), boost significantly
            if p_conf > 0.3 and s_conf > 0.3:
                agreement_bonus = 0.15
            else:
                agreement_bonus = 0.0

            # Adaptive weights based on trick complexity
            # For multi-rotation tricks, physics is unreliable -> weight signature more
            if physics["flip_count"] >= 1.5 or physics["twist_count"] >= 1.5:
                pw, sw = 0.25, 0.75  # Trust signature more for complex tricks
            elif not physics["went_inverted"]:
                pw, sw = 0.3, 0.7  # Non-inverted = trust signature
            else:
                pw, sw = self.physics_weight, self.signature_weight

            hybrid = pw * p_conf + sw * s_conf + agreement_bonus
            hybrid = min(1.0, hybrid)

            # Get D-score
            td = TRICK_DEFINITIONS.get(tid)
            d_score = td.fig_score if td and hasattr(td, "fig_score") else 0.0

            results.append(HybridMatch(
                trick_name=c["trick_name"],
                trick_id=tid,
                hybrid_confidence=round(hybrid, 3),
                physics_confidence=round(p_conf, 3),
                signature_confidence=round(s_conf, 3),
                d_score=d_score,
                flip_count=physics["flip_count"],
                twist_count=physics["twist_count"],
                direction=physics["direction"],
                axis=physics["axis"],
                body_shape=physics["body_shape"],
                went_inverted=physics["went_inverted"],
                physics_match_name=c["physics_match"],
                signature_match_name=c["signature_match"],
            ))

        # Sort by hybrid confidence
        results.sort(key=lambda m: -m.hybrid_confidence)
        return results[:top_k], physics

    def _name_to_trick_id(self, name: str) -> str | None:
        """Convert a trick name to FIG trick ID."""
        # Direct lookup
        tid = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        if tid in TRICK_DEFINITIONS:
            return tid
        # Search by name
        for t_id, td in TRICK_DEFINITIONS.items():
            if td.name.lower() == name.lower():
                return t_id
        return None
