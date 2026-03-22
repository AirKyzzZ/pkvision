"""Explainability and audit trail — generates human-readable decision logs."""

from __future__ import annotations

from datetime import datetime

from core.models import AuditEntry, AuditEntryType, ScoreResult, TrickDetection


class AuditTracer:
    """Accumulates audit entries during a run analysis.

    All entries are immutable after creation. Human overrides create new entries
    rather than modifying existing ones.
    """

    def __init__(self):
        self.entries: list[AuditEntry] = []

    def log_system(self, message: str) -> None:
        """Log a system-level event (analysis start, config, etc.)."""
        self.entries.append(
            AuditEntry(
                entry_type=AuditEntryType.SYSTEM,
                reasoning=message,
            )
        )

    def log_detection(self, detection: TrickDetection) -> None:
        """Log a trick detection with full reasoning."""
        # Build angle match reasoning
        angle_details = []
        for match in detection.angle_matches:
            status = "MATCH" if match.matched else "MISS"
            angle_details.append(
                f"{match.joint}={match.measured:.1f} deg "
                f"(rule: {match.rule_min:.0f}-{match.rule_max:.0f} deg) [{status}]"
            )

        phase_details = []
        for phase_name, conf in detection.phase_confidences.items():
            phase_details.append(f"{phase_name}={conf:.1%}")

        reasoning_parts = [
            f"Detected {detection.trick_name} ({detection.trick_id})",
            f"Strategy: {detection.strategy_used}",
            f"Confidence: {detection.confidence:.1%}",
            f"Time: {detection.start_time_ms:.0f}ms - {detection.end_time_ms:.0f}ms",
            f"Frames: {detection.start_frame} - {detection.end_frame}",
        ]

        if phase_details:
            reasoning_parts.append(f"Phase confidences: {', '.join(phase_details)}")
        if angle_details:
            reasoning_parts.append(f"Angles: {'; '.join(angle_details)}")

        self.entries.append(
            AuditEntry(
                entry_type=AuditEntryType.DETECTION,
                timestamp_ms=detection.start_time_ms,
                trick_id=detection.trick_id,
                confidence=detection.confidence,
                reasoning=" | ".join(reasoning_parts),
            )
        )

    def log_scoring(self, score: ScoreResult) -> None:
        """Log the scoring decision."""
        top3_details = []
        for i, trick in enumerate(score.top3, 1):
            top3_details.append(
                f"#{i} {trick.trick_name} "
                f"(difficulty={trick.difficulty}, confidence={trick.confidence:.1%}, "
                f"score={trick.weighted_score:.2f})"
            )

        reasoning = (
            f"Top {len(score.top3)} tricks selected by difficulty | "
            f"Total score: {score.total_score:.2f} / {score.max_possible_score:.2f} | "
            + " | ".join(top3_details)
        )

        self.entries.append(
            AuditEntry(
                entry_type=AuditEntryType.SCORING,
                reasoning=reasoning,
            )
        )

    def log_override(
        self,
        trick_id: str,
        judge_name: str,
        original_confidence: float,
        new_confidence: float,
        reason: str,
    ) -> None:
        """Log a judge override. Preserves the original detection."""
        self.entries.append(
            AuditEntry(
                entry_type=AuditEntryType.OVERRIDE,
                trick_id=trick_id,
                confidence=new_confidence,
                reasoning=(
                    f"Judge override by {judge_name}: "
                    f"{trick_id} confidence {original_confidence:.1%} → {new_confidence:.1%} | "
                    f"Reason: {reason}"
                ),
                created_by=judge_name,
            )
        )

    def format_human_readable(self) -> str:
        """Generate a human-readable audit log string."""
        lines = ["=" * 70, "PKVISION — AUDIT TRAIL", "=" * 70, ""]

        for entry in self.entries:
            timestamp = ""
            if entry.timestamp_ms is not None:
                seconds = entry.timestamp_ms / 1000.0
                minutes = int(seconds // 60)
                secs = seconds % 60
                timestamp = f"[{minutes:02d}:{secs:05.2f}] "

            prefix = {
                AuditEntryType.SYSTEM: "SYS",
                AuditEntryType.DETECTION: "DET",
                AuditEntryType.SCORING: "SCR",
                AuditEntryType.OVERRIDE: "OVR",
            }.get(entry.entry_type, "???")

            lines.append(f"{timestamp}{prefix} | {entry.reasoning}")

        lines.append("")
        lines.append("=" * 70)
        return "\n".join(lines)
