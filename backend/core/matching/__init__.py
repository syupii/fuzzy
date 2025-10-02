# core/matching/__init__.py
"""
マッチングモジュール
分野マッチングと統合マッチャーを提供
"""

from .field_matcher import FieldMatcher, FieldInterest
from .integrated_matcher import IntegratedMatcher, CompatibilityResult

__all__ = [
    'FieldMatcher',
    'FieldInterest',
    'IntegratedMatcher',
    'CompatibilityResult'
]