# core/decision_tree/__init__.py - ファジィ決定木モジュール

# ノード関連
from .node import (
    FuzzyTreeNode, FuzzyInternalNode, FuzzyLeafNode, FuzzyRuleNode,
    SplitCondition, NodeTraverser
)

# 決定木関連
from .tree import FuzzyDecisionTree, TreeMetrics, PredictionResult

# 構築関連
from .builder import FuzzyTreeBuilder, BuilderConfig, SplitEvaluation

# 後方互換性のための定義（存在しないクラスの代替）
try:
    # 存在しない場合のフォールバック
    from enum import Enum
    from dataclasses import dataclass
    from typing import Dict, Any, Optional
    
    # NodeType の定義（存在しない場合）
    class NodeType(str, Enum):
        LEAF = "leaf"
        INTERNAL = "internal"
        RULE = "rule"
    
    # NodeStatistics の定義（存在しない場合）
    @dataclass
    class NodeStatistics:
        prediction_count: int = 0
        accuracy: float = 0.0
        confidence: float = 0.0
        samples_count: int = 0
    
    # FuzzyDecisionNode の別名（後方互換性）
    FuzzyDecisionNode = FuzzyTreeNode
    
    # EnhancedFuzzyDecisionTree の別名（後方互換性）
    EnhancedFuzzyDecisionTree = FuzzyDecisionTree
    
    # TreeConfig の定義（存在しない場合）
    @dataclass
    class TreeConfig:
        max_depth: int = 10
        min_samples_split: int = 2
        min_samples_leaf: int = 1
        fuzzy_threshold: float = 0.1
    
    # PredictionMode の定義（存在しない場合）
    class PredictionMode(str, Enum):
        CRISP = "crisp"
        FUZZY = "fuzzy"
        HYBRID = "hybrid"
    
    # SplitCriterion の定義（存在しない場合）
    class SplitCriterion(str, Enum):
        FUZZY_GAIN = "fuzzy_gain"
        GINI = "gini"
        ENTROPY = "entropy"

except ImportError:
    pass

__all__ = [
    # ノードクラス
    'FuzzyTreeNode',
    'FuzzyInternalNode', 
    'FuzzyLeafNode',
    'FuzzyRuleNode',
    'SplitCondition',
    'NodeTraverser',
    
    # 決定木クラス
    'FuzzyDecisionTree',
    'TreeMetrics',
    'PredictionResult',
    
    # 構築クラス
    'FuzzyTreeBuilder',
    'BuilderConfig',
    'SplitEvaluation',
    
    # 後方互換性
    'FuzzyDecisionNode',    # FuzzyTreeNode の別名
    'EnhancedFuzzyDecisionTree',  # FuzzyDecisionTree の別名
    'NodeType',
    'NodeStatistics',
    'TreeConfig',
    'PredictionMode',
    'SplitCriterion',
]