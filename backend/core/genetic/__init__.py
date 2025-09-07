# core/genetic/__init__.py - 遺伝的アルゴリズムモジュール

# 個体関連
from .individual import Individual, WeightVector, FuzzyTreeIndividual, GeneInfo

# 型定義関連（後方互換性のため）
try:
    from .types import IndividualType, FitnessComponents, GeneticIndividual
except ImportError:
    # types.py が存在しない場合は individual.py から直接インポート
    from .individual import Individual as GeneticIndividual
    from enum import Enum
    from dataclasses import dataclass
    
    class IndividualType(str, Enum):
        WEIGHT_VECTOR = "weight_vector"
        FUZZY_TREE = "fuzzy_tree"
        HYBRID = "hybrid"
    
    @dataclass
    class FitnessComponents:
        accuracy: float = 0.0
        diversity: float = 0.0
        complexity: float = 0.0

# 集団関連
from .population import Population, PopulationConfig, PopulationStatistics

# 進化エンジン関連
from .evolution import EvolutionEngine, EvolutionConfig, EvolutionResult

# 操作関連
from .operators import (
    OperatorConfig, SelectionMethod, CrossoverMethod, MutationMethod,
    OperatorFactory, GeneticOperator, SelectionOperator, 
    CrossoverOperator, MutationOperator
)

__all__ = [
    # 個体クラス
    'Individual',
    'WeightVector', 
    'FuzzyTreeIndividual',
    'GeneInfo',
    'GeneticIndividual',  # 後方互換性
    
    # 型定義
    'IndividualType',
    'FitnessComponents',
    
    # 集団クラス
    'Population',
    'PopulationConfig',
    'PopulationStatistics',
    
    # 進化エンジン
    'EvolutionEngine',
    'EvolutionConfig',
    'EvolutionResult',
    
    # 操作クラス
    'GeneticOperator',
    'SelectionOperator',
    'CrossoverOperator', 
    'MutationOperator',
    'OperatorConfig',
    'OperatorFactory',
    
    # 操作メソッド
    'SelectionMethod',
    'CrossoverMethod',
    'MutationMethod',
]