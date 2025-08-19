"""
Azure Integration Module

This module provides Azure ML integration for the algorithmic trading and option pricing project.
"""

from .ml_pipeline import AzureMLPipeline
from .deployment import AzureDeployment

__all__ = [
    'AzureMLPipeline',
    'AzureDeployment'
]