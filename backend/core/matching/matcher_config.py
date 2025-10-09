# backend/config/matcher_config.py （新規作成）

from enum import Enum

class MatcherType(Enum):
    """マッチャータイプ"""
    SIMPLE = "simple"              # SimpleMatcher（改善版）
    FUZZY_MULTIPATH = "multipath"  # FuzzyMultiPathMatcher（技術資料準拠版）

# デフォルト設定
DEFAULT_MATCHER_TYPE = MatcherType.FUZZY_MULTIPATH  # ★ 技術資料準拠版を推奨

# マッチャー設定
MATCHER_CONFIG = {
    MatcherType.SIMPLE: {
        "name": "SimpleMatcher",
        "description": "改善版（σ=0.25、ボーナス付き）",
        "sigma": 0.25,
        "category_decay": 0.6,
        "no_match_penalty": 0.5
    },
    MatcherType.FUZZY_MULTIPATH: {
        "name": "FuzzyMultiPathMatcher",
        "description": "技術資料完全準拠版（複数パス統合）",
        "sigma": 0.2,
        "category_decay": 0.7,
        "no_match_penalty": 0.3
    }
}