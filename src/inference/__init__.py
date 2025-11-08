"""Inference module"""
from .predict import AgePredictor
from .generate import TemplateGenerator
from .uncertainty import UncertaintyEstimator

__all__ = ['AgePredictor', 'TemplateGenerator', 'UncertaintyEstimator']
