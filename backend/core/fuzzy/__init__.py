# core/fuzzy/__init__.py - 修正版

from .membership import (
    MembershipFunction, TriangularMF, GaussianMF, TrapezoidalMF,
    MembershipFunctionFactory, MembershipType
)

# rules.pyから実際に存在するクラスをインポート
from .rules import (
    FuzzyRule, 
    Condition as FuzzyCondition,      # エイリアス設定
    Conclusion as FuzzyConclusion,    # エイリアス設定
    FuzzyRuleSet as RuleBase,         # エイリアス設定
    LabMatchingRuleBuilder as RuleGenerator,  # エイリアス設定
    RuleOperator,
    LogicalOperator,
    ComparisonOperator,
    RuleParser
)

from .inference import (
    FuzzyInferenceEngine, SimpleFuzzyInferenceEngine, DefuzzificationMethod
)

# 後方互換性のためのエクスポート
__all__ = [
    # メンバーシップ関数
    'MembershipFunction',
    'TriangularMF', 
    'GaussianMF',
    'TrapezoidalMF',
    'MembershipFunctionFactory',
    'MembershipType',
    
    # ルール関連（実際のクラス名）
    'FuzzyRule',
    'Condition',
    'Conclusion', 
    'FuzzyRuleSet',
    'LabMatchingRuleBuilder',
    'RuleOperator',
    'LogicalOperator',
    'ComparisonOperator',
    'RuleParser',
    
    # ルール関連（エイリアス）
    'FuzzyCondition',
    'FuzzyConclusion',
    'RuleBase',
    'RuleGenerator',
    
    # 推論エンジン
    'FuzzyInferenceEngine',
    'SimpleFuzzyInferenceEngine',
    'DefuzzificationMethod',
]