"""
Fraud Detection System - Hackathon Finance Track
Main package initialization
"""

__version__ = "1.0.0"
__author__ = "Team Hack Fraud Detection"

from . import data_processing
from . import feature_engineering
from . import models
from . import evaluation
from . import utils

__all__ = [
    "data_processing",
    "feature_engineering",
    "models",
    "evaluation",
    "utils"
]
