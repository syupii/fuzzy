# core/matching/__init__.py
"""
マッチングモジュール（パターンA版）
シンプルマッチャーと分野マッチングを提供
"""

from .simple_matcher import SimpleMatcher, CompatibilityResult
from .field_matcher import FieldMatcher, FieldInterest

__all__ = [
    'SimpleMatcher',
    'CompatibilityResult',
    'FieldMatcher',
    'FieldInterest'
]