"""Celery task stubs for future async video processing.

These are placeholders. The PoC processes videos synchronously.
When ready for async processing:
1. Set CELERY_BROKER_URL and CELERY_RESULT_BACKEND in .env
2. Run: celery -A worker.tasks worker --loglevel=info
"""

# from celery import Celery
#
# app = Celery("pkvision")
# app.config_from_object("django.conf:settings", namespace="CELERY")
#
# @app.task(bind=True)
# def analyze_video(self, run_id: str, video_path: str) -> dict:
#     """Process a video asynchronously."""
#     # Import here to avoid circular imports
#     # from core.pose.detector import PoseDetector
#     # from core.recognition.classifier import TrickClassifier
#     # from core.scoring.engine import ScoringEngine
#     pass
