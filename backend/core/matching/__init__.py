# backend/core/matching/__init__.py

from .simple_matcher import SimpleMatcher, CompatibilityResult
from .fuzzy_multipath_matcher import FuzzyMultiPathMatcher, FuzzyPath
from .field_matcher_corrected import FieldMatcherCorrected, FieldMatchResult

__all__ = [
    # 既存（改善版）
    'SimpleMatcher',
    'CompatibilityResult',
    
    # 新規（技術資料準拠版）
    'FuzzyMultiPathMatcher',
    'FuzzyPath',
    
    # 新規（フィールドマッチング修正版）
    'FieldMatcherCorrected',
    'FieldMatchResult',
]