# core/fuzzy/__init__.py - 修正版 v2

# メンバーシップ関数
from .membership import (
    MembershipFunction,
    TriangularMF,
    GaussianMF,
    TrapezoidalMF,
    GeneralizedBellMF,
    SigmoidMF,
    FuzzySet,
    FuzzyVariable,
    create_standard_fuzzy_variable
)

# MembershipFunctionFactoryのエイリアス（後方互換性）
class MembershipFunctionFactory:
    """メンバーシップ関数ファクトリクラス（エイリアス）"""
    
    @staticmethod
    def create_standard_sets(name: str, universe: tuple):
        """標準的なファジィ変数を作成"""
        return create_standard_fuzzy_variable(name, universe, n_sets=3)
    
    @staticmethod
    def create_five_level_sets(name: str, universe: tuple):
        """5レベルのファジィ変数を作成"""
        return create_standard_fuzzy_variable(name, universe, n_sets=5)

# 推論エンジン
from .inference import (
    PriorityAwareFuzzyInferenceEngine,
    PriorityAwareGeneticEvolutionEngine,
    PriorityAwareFuzzyDecisionTree
)

# SimpleFuzzyInferenceEngineラッパークラス（後方互換性）
class SimpleFuzzyInferenceEngine:
    """シンプルなファジィ推論エンジン（ラッパークラス）"""
    
    def __init__(self, features=None, target=None):
        """
        初期化
        
        Args:
            features: 特徴量のリスト（評価基準）
            target: ターゲット変数名
        """
        self.features = features or []
        self.target = target or "compatibility"
        self.engine = PriorityAwareFuzzyInferenceEngine()
    
    def predict(self, profile):
        """
        予測を実行
        
        Args:
            profile: 学生プロファイル辞書
            
        Returns:
            予測値（0-1の範囲）
        """
        # 簡易的な予測実装
        # すべての特徴の平均を取る
        if not self.features:
            return 0.5
        
        values = []
        for feature in self.features:
            if feature in profile:
                # 0-10の範囲を0-1に正規化
                normalized = profile[feature] / 10.0
                values.append(normalized)
        
        if not values:
            return 0.5
        
        return sum(values) / len(values)
    
    def infer_lab_compatibility(self, student_profile, lab_profile):
        """
        研究室適合度を推論
        
        Args:
            student_profile: 学生プロファイル
            lab_profile: 研究室プロファイル
            
        Returns:
            適合度スコア
        """
        # 基本的な距離ベースの適合度計算
        if not self.features:
            return 0.5
        
        lab_features = lab_profile.get('features', {})
        
        total_distance = 0.0
        count = 0
        
        for feature in self.features:
            student_val = student_profile.get(feature, 5.0) / 10.0
            lab_val = lab_features.get(feature, 5.0) / 10.0
            
            distance = abs(student_val - lab_val)
            total_distance += distance
            count += 1
        
        if count == 0:
            return 0.5
        
        # 距離を適合度に変換（距離が小さいほど適合度が高い）
        avg_distance = total_distance / count
        compatibility = 1.0 - avg_distance
        
        return max(0.0, min(1.0, compatibility))

# エクスポート
__all__ = [
    # メンバーシップ関数
    'MembershipFunction',
    'TriangularMF',
    'GaussianMF',
    'TrapezoidalMF',
    'FuzzySet',
    'FuzzyVariable',
    'create_standard_fuzzy_variable',
    'MembershipFunctionFactory',
    
    # 推論エンジン
    'PriorityAwareFuzzyInferenceEngine',
    'SimpleFuzzyInferenceEngine',  # ラッパークラス
    'PriorityAwareGeneticEvolutionEngine',
    'PriorityAwareFuzzyDecisionTree',
]
