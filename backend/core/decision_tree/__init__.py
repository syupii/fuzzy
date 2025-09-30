# core/decision_tree/__init__.py - 修正版 v2

# ノード関連
from .node import (
    FuzzyTreeNode,
    FuzzyInternalNode,
    FuzzyLeafNode
)

# 決定木関連
from .tree import (
    NodeType,
    TreeNode,
    FuzzyDecisionTreeBuilder,
    MultiLevelFuzzyClassifier
)

# FuzzyDecisionTreeラッパークラス（後方互換性）
class FuzzyDecisionTree:
    """ファジィ決定木（ラッパークラス）"""
    
    def __init__(self, config=None):
        """
        初期化
        
        Args:
            config: 決定木設定
        """
        self.config = config
        self.builder = FuzzyDecisionTreeBuilder(
            max_depth=getattr(config, 'max_depth', 10) if config else 10,
            min_samples_split=getattr(config, 'min_samples_leaf', 1) if config else 1
        )
        self.root = None
    
    def fit(self, X, y):
        """
        学習
        
        Args:
            X: 特徴量
            y: ラベル
        """
        # 簡易実装
        pass
    
    def predict(self, X):
        """
        予測
        
        Args:
            X: 特徴量
            
        Returns:
            予測値
        """
        return 0.5

# TreeConfigクラス
class TreeConfig:
    """決定木設定クラス"""
    
    def __init__(self, max_depth: int = 10, min_samples_leaf: int = 1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

# エクスポート
__all__ = [
    # ノードクラス
    'FuzzyTreeNode',
    'FuzzyInternalNode',
    'FuzzyLeafNode',
    
    # 決定木クラス
    'NodeType',
    'TreeNode',
    'FuzzyDecisionTreeBuilder',
    'FuzzyDecisionTree',  # ラッパークラス
    'MultiLevelFuzzyClassifier',
    
    # 設定クラス
    'TreeConfig',
]
