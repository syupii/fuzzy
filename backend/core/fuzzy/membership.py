# core/fuzzy/membership.py - ファジィメンバーシップ関数

import numpy as np
from typing import Dict, List, Tuple, Callable
from models.schemas import FieldInterest

class MembershipFunction:
    """ファジィメンバーシップ関数クラス"""
    
    def __init__(self):
        # 基本メンバーシップ関数の定義
        self.functions = {
            "very_low": self._create_trapezoid(0, 0, 1, 2.5),
            "low": self._create_trapezoid(1, 2.5, 3.5, 5),
            "medium": self._create_trapezoid(3.5, 5, 5, 6.5),
            "high": self._create_trapezoid(5, 6.5, 7.5, 9),
            "very_high": self._create_trapezoid(7.5, 9, 10, 10)
        }
        
        # 興味度レベル分類
        self.interest_levels = {
            "no_interest": (0, 2),
            "slight_interest": (2, 4),
            "moderate_interest": (4, 6),
            "strong_interest": (6, 8),
            "very_strong_interest": (8, 10)
        }
        
        # 経験レベル分類
        self.experience_levels = {
            "novice": (0, 2),
            "beginner": (2, 4),
            "intermediate": (4, 6),
            "advanced": (6, 8),
            "expert": (8, 10)
        }
    
    def _create_trapezoid(self, a: float, b: float, c: float, d: float) -> Callable:
        """台形メンバーシップ関数を作成"""
        def trapezoid(x: float) -> float:
            if x <= a or x >= d:
                return 0.0
            elif a < x <= b:
                return (x - a) / (b - a) if b != a else 1.0
            elif b < x <= c:
                return 1.0
            elif c < x < d:
                return (d - x) / (d - c) if d != c else 1.0
            return 0.0
        return trapezoid
    
    def _create_triangle(self, a: float, b: float, c: float) -> Callable:
        """三角メンバーシップ関数を作成"""
        def triangle(x: float) -> float:
            if x <= a or x >= c:
                return 0.0
            elif a < x <= b:
                return (x - a) / (b - a) if b != a else 1.0
            elif b < x < c:
                return (c - x) / (c - b) if c != b else 1.0
            return 0.0
        return triangle
    
    def evaluate(self, value: float, level: str) -> float:
        """メンバーシップ値を計算"""
        if level in self.functions:
            return self.functions[level](value)
        return 0.0
    
    def field_compatibility(self, student_interest: FieldInterest, 
                          field_info: Dict) -> float:
        """分野適合性をファジィ計算"""
        
        # 興味度の評価
        interest_score = self._evaluate_interest(student_interest.interest_level)
        
        # 経験レベルと分野難易度のマッチング評価
        experience_match = self._evaluate_experience_match(
            student_interest.experience_level, 
            field_info.get("difficulty", "intermediate")
        )
        
        # 重要度の重み
        importance_weight = student_interest.importance_level / 10.0
        
        # ファジィ統合（重み付き平均）
        compatibility = (
            interest_score * 0.4 +
            experience_match * 0.3 +
            importance_weight * 0.3
        )
        
        return min(1.0, max(0.0, compatibility))
    
    def _evaluate_interest(self, interest_level: int) -> float:
        """興味度をファジィ評価"""
        # 高い興味を重視
        high_membership = self.evaluate(interest_level, "high")
        very_high_membership = self.evaluate(interest_level, "very_high")
        
        return max(high_membership, very_high_membership)
    
    def _evaluate_experience_match(self, experience_level: int, 
                                 difficulty: str) -> float:
        """経験レベルと分野難易度のマッチング評価"""
        
        # 難易度を数値に変換
        difficulty_map = {
            "beginner": 3,
            "intermediate": 6,
            "advanced": 9
        }
        
        target_experience = difficulty_map.get(difficulty, 6)
        
        # 経験レベルとのマッチング計算
        # 経験が足りない場合とオーバースペックの場合を考慮
        distance = abs(experience_level - target_experience)
        
        if distance <= 1:
            return 1.0  # 完全マッチ
        elif distance <= 2:
            return 0.8  # 良好なマッチ
        elif distance <= 3:
            return 0.6  # 中程度のマッチ
        elif distance <= 4:
            return 0.4  # やや困難
        else:
            return 0.2  # 困難
    
    def criteria_similarity(self, student_value: int, lab_value: float) -> float:
        """評価基準の類似度をファジィ計算"""
        
        # 距離ベースの類似度
        distance = abs(student_value - lab_value)
        
        # ガウシアンメンバーシップ関数を使用
        sigma = 2.0  # 標準偏差
        similarity = np.exp(-(distance ** 2) / (2 * sigma ** 2))
        
        return similarity
    
    def fuzzy_and(self, *values: float) -> float:
        """ファジィAND演算（最小値）"""
        return min(values)
    
    def fuzzy_or(self, *values: float) -> float:
        """ファジィOR演算（最大値）"""
        return max(values)
    
    def fuzzy_not(self, value: float) -> float:
        """ファジィNOT演算"""
        return 1.0 - value
    
    def defuzzify_centroid(self, membership_values: Dict[float, float]) -> float:
        """重心法によるファジィ値の明確化"""
        
        if not membership_values:
            return 0.0
        
        numerator = sum(value * membership for value, membership in membership_values.items())
        denominator = sum(membership_values.values())
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def linguistic_evaluation(self, value: float) -> str:
        """数値を言語的評価に変換"""
        
        if value >= 8.5:
            return "非常に良い"
        elif value >= 7.0:
            return "良い"
        elif value >= 5.5:
            return "普通"
        elif value >= 3.5:
            return "やや劣る"
        else:
            return "劣る"
    
    def calculate_confidence(self, field_interests: List[FieldInterest]) -> float:
        """学生の選択に対する信頼度を計算"""
        
        if not field_interests:
            return 0.0
        
        # 興味度の分散と平均を考慮
        interest_levels = [fi.interest_level for fi in field_interests]
        experience_levels = [fi.experience_level for fi in field_interests]
        
        interest_mean = np.mean(interest_levels)
        interest_std = np.std(interest_levels)
        
        experience_mean = np.mean(experience_levels)
        
        # 信頼度計算（高い興味と一貫性を重視）
        confidence = (
            (interest_mean / 10.0) * 0.5 +  # 平均興味度
            (1.0 - min(interest_std / 5.0, 1.0)) * 0.3 +  # 一貫性
            (experience_mean / 10.0) * 0.2  # 経験レベル
        )
        
        return min(1.0, max(0.0, confidence))