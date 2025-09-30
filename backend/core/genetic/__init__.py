# core/genetic/__init__.py - 修正版

# 個体関連
from .individual import (
    FuzzyTreeGene,
    Individual
)

# WeightVectorのエイリアス（後方互換性）
# Individualクラスをベースにした簡易版
class WeightVector:
    """重みベクトルクラス（エイリアス）"""
    
    def __init__(self, weights: list = None):
        self.weights = weights or []
    
    def to_dict(self):
        return {'weights': self.weights}
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(data.get('weights', []))

# 集団関連
from .population import (
    Population,
    PopulationConfig
)

# 進化エンジン関連
from .evolution import (
    EvolutionEngine,
    EvolutionConfig,
    FuzzyTreeEvaluator
)

# エクスポート
__all__ = [
    # 個体クラス
    'FuzzyTreeGene',
    'Individual',
    'WeightVector',  # エイリアス
    
    # 集団クラス
    'Population',
    'PopulationConfig',
    
    # 進化エンジン
    'EvolutionEngine',
    'EvolutionConfig',
    'FuzzyTreeEvaluator',
]
