# core/matching/integrated_matcher.py
"""
統合マッチングシステム (修正版)
ファジィ推論 + 決定木 + 遺伝的アルゴリズム + 分野マッチング
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .field_matcher import FieldMatcher


@dataclass
class CompatibilityResult:
    """適合度計算結果"""
    total_compatibility: float
    field_score: float
    basic_score: float
    tree_score: float
    detailed_score: float
    field_weight: float
    basic_weight: float
    field_contribution: float
    basic_contribution: float
    decision_path: List[str]
    explanation: str
    breakdown: Dict


class IntegratedMatcher:
    """
    統合マッチングシステム
    全てのコンポーネントを統合して適合度を計算
    """
    
    def __init__(
        self, 
        fuzzy_engine=None,
        decision_tree=None,
        field_matcher=None,
        optimized_weights: Optional[np.ndarray] = None
    ):
        """
        Args:
            fuzzy_engine: ファジィ推論エンジン
            decision_tree: ファジィ決定木
            field_matcher: 分野マッチャー
            optimized_weights: 遺伝的アルゴリズムで最適化された重み (12項目用)
        """
        self.fuzzy_engine = fuzzy_engine
        self.decision_tree = decision_tree
        self.field_matcher = field_matcher or FieldMatcher()
        
        # 重みが指定されていない場合はデフォルト (12項目のみ)
        if optimized_weights is None:
            self.optimized_weights = np.ones(12) / 12
        else:
            # 12項目分の重みのみを使用
            self.optimized_weights = np.array(optimized_weights[:12])
        
        # 基本項目のリスト (research_field_matchを除く12項目)
        self.basic_criteria = [
            "research_intensity",
            "advisor_style",
            "team_work",
            "workload",
            "theory_practice",
            "skill_development",
            "lab_atmosphere",
            "flexibility",
            "publication_opportunity",
            "interdisciplinary",
            "communication_style"
        ]
    
    def calculate_compatibility(
        self, 
        student: Dict, 
        lab: Dict
    ) -> CompatibilityResult:
        """
        統合適合度を計算
        
        Args:
            student: 学生プロファイル
            lab: 研究室プロファイル
        
        Returns:
            適合度計算結果
        """
        
        # ===== ステップ1: ファジィ化 =====
        fuzzified_student = self._fuzzify_profile(student)
        
        # ===== ステップ2: ファジィ決定木による基本分類 =====
        tree_score, decision_path = self._fuzzy_tree_prediction(
            fuzzified_student,
            student,
            lab
        )
        
        # ===== ステップ3: 詳細マッチングスコア（遺伝的重み使用） =====
        detailed_score = self._calculate_detailed_matching(
            student, 
            lab, 
            self.optimized_weights  # 12項目
        )
        
        # ===== ステップ4: 基本項目スコアの統合 =====
        # 決定木の判断（40%）と詳細マッチング（60%）を統合
        basic_score = tree_score * 0.4 + detailed_score * 0.6
        
        # ===== ステップ5: 分野マッチングスコア =====
        field_score = self._calculate_field_matching(student, lab)
        
        # ===== ステップ6: 最終統合 (修正版) =====
        field_weight_pref = student.get("research_field_match", 5)
        alpha = field_weight_pref / 10  # 分野の比重 (0.0 ~ 1.0)
        beta = 1.0 - alpha  # 基本項目の比重
        
        # 分野と基本項目を統合 (追加の重みは不要)
        field_contribution = alpha * field_score
        basic_contribution = beta * basic_score
        
        total_compatibility = field_contribution + basic_contribution
        
        # 正規化（0-1の範囲に）
        total_compatibility = np.clip(total_compatibility, 0, 1)
        
        # ===== ステップ7: 説明文生成 =====
        explanation = self._generate_explanation(
            student, lab, total_compatibility, field_score, basic_score,
            alpha, beta
        )
        
        return CompatibilityResult(
            total_compatibility=float(total_compatibility),
            field_score=float(field_score),
            basic_score=float(basic_score),
            tree_score=float(tree_score),
            detailed_score=float(detailed_score),
            field_weight=float(alpha),
            basic_weight=float(beta),
            field_contribution=float(field_contribution),
            basic_contribution=float(basic_contribution),
            decision_path=decision_path,
            explanation=explanation,
            breakdown={
                "field": {
                    "score": float(field_score),
                    "weight": float(alpha),
                    "contribution": float(field_contribution)
                },
                "basic": {
                    "tree_score": float(tree_score),
                    "detailed_score": float(detailed_score),
                    "combined_score": float(basic_score),
                    "weight": float(beta),
                    "contribution": float(basic_contribution)
                }
            }
        )
    
    def _fuzzify_profile(self, profile: Dict) -> Dict[str, str]:
        """プロファイルをファジィ化"""
        fuzzified = {}
        
        for criterion in self.basic_criteria:
            value = profile.get(criterion, 5)
            
            # 三角型メンバーシップ関数
            mu_low = self._triangular_mf(value, 0, 2, 4)
            mu_medium = self._triangular_mf(value, 3, 5, 7)
            mu_high = self._triangular_mf(value, 6, 8, 10)
            
            # 最大メンバーシップ値を持つ言語値を選択
            memberships = {
                "low": mu_low,
                "medium": mu_medium,
                "high": mu_high
            }
            
            fuzzified[criterion] = max(memberships, key=memberships.get)
        
        return fuzzified
    
    def _triangular_mf(self, x: float, a: float, b: float, c: float) -> float:
        """三角型メンバーシップ関数"""
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a) if (b - a) > 0 else 0.0
        else:
            return (c - x) / (c - b) if (c - b) > 0 else 0.0
    
    def _fuzzy_tree_prediction(
        self,
        fuzzified_student: Dict[str, str],
        student: Dict,
        lab: Dict
    ) -> Tuple[float, List[str]]:
        """ファジィ決定木による予測（簡易版）"""
        
        decision_path = ["評価開始"]
        score = 0.5
        
        # 主要項目での判定（8層簡易版）
        important_criteria = [
            "research_intensity",
            "publication_opportunity",
            "team_work",
            "advisor_style",
            "workload",
            "lab_atmosphere",
            "flexibility",
            "skill_development"
        ]
        
        for i, criterion in enumerate(important_criteria):
            student_val = fuzzified_student.get(criterion, "medium")
            lab_val = student.get(criterion, 5)
            lab_fuzz = self._fuzzify_value(lab_val)
            
            if student_val == lab_fuzz:
                boost = 0.08 if i < 3 else 0.04  # 上位3項目は重み大
                score += boost
                decision_path.append(f"層{i+1} {criterion}: {student_val} - 一致 (+{boost:.2f})")
            else:
                decision_path.append(f"層{i+1} {criterion}: {student_val} vs {lab_fuzz}")
        
        score = np.clip(score, 0, 1)
        decision_path.append(f"決定木スコア: {score:.2f}")
        
        return score, decision_path
    
    def _fuzzify_value(self, value: float) -> str:
        """単一値をファジィ化"""
        mu_low = self._triangular_mf(value, 0, 2, 4)
        mu_medium = self._triangular_mf(value, 3, 5, 7)
        mu_high = self._triangular_mf(value, 6, 8, 10)
        
        memberships = {"low": mu_low, "medium": mu_medium, "high": mu_high}
        return max(memberships, key=memberships.get)
    
    def _calculate_detailed_matching(
        self,
        student: Dict,
        lab: Dict,
        weights: np.ndarray
    ) -> float:
        """詳細マッチングスコア（遺伝的重み使用）"""
        
        total_score = 0.0
        total_weight = 0.0
        
        for i, criterion in enumerate(self.basic_criteria):
            if i >= len(weights):
                break
            
            student_val = student.get(criterion, 5)
            lab_val = lab.get(criterion, 5)
            
            # 正規化（0-1の範囲に）
            if student_val > 1:
                student_val = student_val / 10
            if lab_val > 1:
                lab_val = lab_val / 10
            
            # ファジィマッチング（差分ベース）
            diff = abs(student_val - lab_val)
            similarity = 1.0 - diff
            
            # 重みを適用
            weight = weights[i]
            weighted_score = similarity * weight
            
            total_score += weighted_score
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    def _calculate_field_matching(self, student: Dict, lab: Dict) -> float:
        """分野マッチングスコア"""
        
        field_interests_data = student.get("field_interests", {})
        lab_field_id = lab.get("field_id", "unknown")
        
        # 分野興味データをパース
        field_interests = self.field_matcher.parse_field_interests(field_interests_data)
        
        return self.field_matcher.calculate_field_matching_score(
            field_interests,
            lab_field_id
        )
    
    def _generate_explanation(
        self,
        student: Dict,
        lab: Dict,
        total: float,
        field_score: float,
        basic_score: float,
        alpha: float,
        beta: float
    ) -> str:
        """説明文を生成"""
        
        parts = []
        
        # 総合評価
        if total >= 0.85:
            parts.append("✅ 非常に高い適合度")
        elif total >= 0.7:
            parts.append("⭐ 高い適合度")
        elif total >= 0.5:
            parts.append("⚠️ 中程度の適合度")
        else:
            parts.append("❌ 低い適合度")
        
        # 分野評価
        if alpha > 0.6:  # 分野重視
            if field_score >= 0.8:
                parts.append("興味分野と完全に一致")
            elif field_score >= 0.5:
                parts.append("興味分野と部分的に一致")
            else:
                parts.append("興味分野と異なる")
        
        # 基本項目評価
        if beta > 0.6:  # 基本項目重視
            if basic_score >= 0.8:
                parts.append("研究スタイルが非常に合う")
            elif basic_score >= 0.6:
                parts.append("研究スタイルが概ね合う")
        
        return " / ".join(parts)