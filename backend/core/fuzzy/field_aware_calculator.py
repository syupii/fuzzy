"""
backend/core/fuzzy/field_aware_calculator.py
分野考慮型ファジィ適合度計算器

機能:
- 基本12項目のファジィ評価
- 分野興味度の考慮
- research_field_matchによる比重調整
- メンバーシップ度の活用
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional


class FieldAwareFuzzyCompatibilityCalculator:
    """
    分野考慮型ファジィ適合度計算器
    
    計算式:
    最終スコア = β × 基本12項目スコア + α × 分野スコア
    
    where:
        α = research_field_match / 10  (分野の比重)
        β = 1 - α  (基本項目の比重)
    """
    
    # 基本12項目（innovation_risk除外）
    CRITERIA = [
        "research_intensity", "advisor_style", "team_work",
        "workload", "theory_practice", "skill_development",
        "lab_atmosphere", "flexibility", "publication_opportunity",
        "interdisciplinary", "communication_style", "innovation_risk"
    ]
    
    # 分野カテゴリマッピング
    FIELD_CATEGORIES = {
        # テクノロジー・システム
        "ai_ml": "technology",
        "image_processing": "technology",
        "network_security": "technology",
        "database_systems": "technology",
        "embedded_iot": "technology",
        "education_linguistics": "technology",
        "natural_science_math": "technology",
        "tourism_regional": "technology",
        "business_decision": "technology",
        "audio_processing": "technology",
        "system_ethics": "technology",
        
        # クリエイティブ
        "web_design": "creative",
        "design_visual": "creative",
        "video_animation": "creative",
        "computer_music": "creative",
        
        # エンターテイメント
        "game_esports": "entertainment",
        "vr_ar_media": "entertainment",
        
        # 人文・社会・体育
        "philosophy_humanities": "humanities",
        "sports_science": "humanities"
    }
    
    def __init__(self, gene=None):
        """
        Args:
            gene: FieldAwareFuzzyTreeGene (遺伝的アルゴリズムで最適化)
        """
        if gene is None:
            # デフォルト遺伝子を使用
            from core.genetic.field_aware_gene import FieldAwareFuzzyTreeGene
            self.gene = FieldAwareFuzzyTreeGene()
        else:
            self.gene = gene
    
    def calculate_compatibility(
        self, 
        student: Dict[str, Any], 
        lab: Dict[str, Any]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        統合適合度を計算
        
        Args:
            student: 学生プロファイル
                {
                    # 基本12項目 (1-10 または 0-1)
                    "research_intensity": 8.0,
                    "advisor_style": 6.0,
                    ...
                    
                    # 比重指数 (1-10)
                    "research_field_match": 7.0,
                    
                    # 分野興味度
                    "field_interests": {
                        "ai_ml": 9.0,
                        "robotics": 3.0
                    }
                }
            
            lab: 研究室プロファイル
                {
                    # 基本12項目
                    "research_intensity": 9.0,
                    ...
                    
                    # 研究分野
                    "field_id": "ai_ml"
                }
        
        Returns:
            (compatibility_score, detailed_breakdown)
        """
        
        # Step 1: research_field_matchから比重を決定
        research_field_match = student.get("research_field_match", 5.0)
        alpha = research_field_match / 10.0  # 分野の比重
        beta = 1.0 - alpha  # 基本項目の比重
        
        # Step 2: 基本12項目のスコア計算（ファジィ決定木）
        basic_score, memberships, criteria_scores = self._calculate_basic_score(
            student, lab
        )
        
        # Step 3: 分野スコア計算
        field_score = self._calculate_field_score(
            student.get("field_interests", {}),
            lab.get("field_id", "unknown")
        )
        
        # Step 4: 分野スコアにブースト適用
        field_boost = 1.0 + (alpha - 0.5) * (self.gene.field_boost_factor - 1.0)
        effective_field_score = np.clip(field_score * field_boost, 0, 1)
        
        # Step 5: 最終統合
        total_compatibility = beta * basic_score + alpha * effective_field_score
        total_compatibility = np.clip(total_compatibility, 0, 1)
        
        # 詳細情報
        breakdown = {
            "total_compatibility": float(total_compatibility),
            "alpha": float(alpha),
            "beta": float(beta),
            "research_field_match_value": research_field_match,
            "basic_score": float(basic_score),
            "field_score_raw": float(field_score),
            "field_score_effective": float(effective_field_score),
            "field_boost_factor": float(field_boost),
            "basic_contribution": float(beta * basic_score),
            "field_contribution": float(alpha * effective_field_score),
            "criteria_scores": {k: float(v) for k, v in criteria_scores.items()},
            "memberships": {k: {kk: float(vv) for kk, vv in v.items()} 
                           for k, v in memberships.items()}
        }
        
        return total_compatibility, breakdown
    
    def _calculate_basic_score(
        self, 
        student: Dict[str, Any], 
        lab: Dict[str, Any]
    ) -> Tuple[float, Dict, Dict]:
        """ファジィ決定木で基本12項目のスコアを計算"""
        
        # Level 1のメンバーシップ度計算
        level1_feature = self.gene.level1_feature
        level1_value = self._normalize_value(student.get(level1_feature, 5.0))
        
        level1_memberships = {
            "low": self._triangular_membership(
                level1_value, *self.gene.membership_params["low"]
            ),
            "medium": self._triangular_membership(
                level1_value, *self.gene.membership_params["medium"]
            ),
            "high": self._triangular_membership(
                level1_value, *self.gene.membership_params["high"]
            )
        }
        
        # 各ブランチでの適合度を計算（メンバーシップ度で重み付け）
        total_compatibility = 0.0
        
        for branch, membership in level1_memberships.items():
            if membership > 0.01:  # 閾値以上のみ考慮
                branch_score = self._calculate_branch_score(student, lab, branch)
                total_compatibility += membership * branch_score
        
        # 12項目すべての詳細スコア
        criteria_scores = self._calculate_all_criteria_scores(student, lab)
        
        memberships = {"level1": level1_memberships}
        
        return total_compatibility, memberships, criteria_scores
    
    def _calculate_branch_score(
        self, 
        student: Dict[str, Any], 
        lab: Dict[str, Any], 
        branch: str
    ) -> float:
        """特定ブランチでの適合度計算"""
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for i, criterion in enumerate(self.CRITERIA):
            student_val = self._normalize_value(student.get(criterion, 5.0))
            lab_val = self._normalize_value(lab.get(criterion, 5.0))
            
            # ガウス型ファジィ類似度
            similarity = np.exp(
                -0.5 * ((student_val - lab_val) / self.gene.similarity_sigma) ** 2
            )
            
            weight = self.gene.criteria_weights[i]
            weighted_sum += weight * similarity
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def _calculate_all_criteria_scores(
        self, 
        student: Dict[str, Any], 
        lab: Dict[str, Any]
    ) -> Dict[str, float]:
        """12項目すべての詳細スコア"""
        
        criteria_scores = {}
        
        for criterion in self.CRITERIA:
            student_val = self._normalize_value(student.get(criterion, 5.0))
            lab_val = self._normalize_value(lab.get(criterion, 5.0))
            
            similarity = np.exp(
                -0.5 * ((student_val - lab_val) / self.gene.similarity_sigma) ** 2
            )
            
            criteria_scores[criterion] = similarity
        
        return criteria_scores
    
    def _calculate_field_score(
        self, 
        field_interests: Dict[str, float], 
        lab_field_id: str
    ) -> float:
        """
        分野スコアを計算
        
        Args:
            field_interests: {field_id: interest_level, ...}
            lab_field_id: 研究室の分野ID
        
        Returns:
            0.0 ~ 1.0 のスコア
        """
        
        if not field_interests:
            return 0.0
        
        # 完全一致
        if lab_field_id in field_interests:
            interest_level = field_interests[lab_field_id]
            return self._normalize_value(interest_level)
        
        # カテゴリ一致（同じカテゴリなら70%のスコア）
        lab_category = self.FIELD_CATEGORIES.get(lab_field_id, "unknown")
        max_score = 0.0
        
        for field_id, interest_level in field_interests.items():
            field_category = self.FIELD_CATEGORIES.get(field_id, "unknown")
            
            if field_category == lab_category and field_category != "unknown":
                score = self._normalize_value(interest_level) * 0.7
                max_score = max(max_score, score)
        
        return max_score
    
    def _normalize_value(self, value: float) -> float:
        """値を0-1に正規化"""
        if value > 1.0:
            return value / 10.0
        return np.clip(value, 0, 1)
    
    def _triangular_membership(self, x: float, a: float, b: float, c: float) -> float:
        """三角型メンバーシップ関数"""
        if x <= a or x >= c:
            return 0.0
        elif abs(x - b) < 1e-6:
            return 1.0
        elif x < b:
            return (x - a) / (b - a) if (b - a) > 0 else 0.0
        else:
            return (c - x) / (c - b) if (c - b) > 0 else 0.0