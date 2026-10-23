"""Metaorder detectors behind one interface (plan section 6).

    FlowDetector (A1)                 rule-based: EWMA of the public flow imbalance, thresholded
    RunsDetector, CusumDetector,      held-out statistical detectors (A2), never used in training
    VolumeClockDetector, RegularityDetector
    LearnedDetector (A3)              logistic classifier on public features, trained on institutional labels
"""

from order_book.detectors.base import Detector
from order_book.detectors.features import FEATURE_NAMES, FeatureTracker
from order_book.detectors.flow import FlowDetector
from order_book.detectors.learned import LearnedDetector, LearnedDetectorParams
from order_book.detectors.statistical import (
    CusumDetector, RegularityDetector, RunsDetector, VolumeClockDetector, held_out_detectors,
)

__all__ = [
    "CusumDetector", "Detector", "FEATURE_NAMES", "FeatureTracker", "FlowDetector", "LearnedDetector",
    "LearnedDetectorParams", "RegularityDetector", "RunsDetector", "VolumeClockDetector", "held_out_detectors",
]
