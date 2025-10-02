# core/matching/field_matcher.py
"""
分野マッチングモジュール
複数の興味分野を統合して適合度を計算
"""

from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class FieldInterest:
    """分野への興味情報"""
    field_id: str
    interest_level: float      # 1-10
    experience_level: float    # 1-10
    importance_level: float    # 1-10
    
    def calculate_score(self) -> float:
        """
        分野興味スコアを計算
        
        Returns:
            0.0 ～ 1.0 のスコア
        """
        # 加重平均（興味60%、重要度30%、経験10%）
        weighted_score = (
            self.interest_level * 0.6 +
            self.importance_level * 0.3 +
            self.experience_level * 0.1
        ) / 10
        
        return min(1.0, max(0.0, weighted_score))


class FieldMatcher:
    """分野マッチング計算クラス"""
    
    # 分野のカテゴリマッピング
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
    
    def calculate_field_matching_score(
        self, 
        student_interests: Dict[str, FieldInterest],
        lab_field_id: str
    ) -> float:
        """
        分野マッチングスコアを計算
        
        Args:
            student_interests: 学生の分野興味辞書
            lab_field_id: 研究室の分野ID
        
        Returns:
            0.0 ～ 1.0 のスコア
        """
        
        # 完全一致の場合
        if lab_field_id in student_interests:
            interest = student_interests[lab_field_id]
            return interest.calculate_score()
        
        # 同じカテゴリの場合（部分一致）
        lab_category = self.FIELD_CATEGORIES.get(lab_field_id, "unknown")
        
        related_scores = []
        for field_id, interest in student_interests.items():
            field_category = self.FIELD_CATEGORIES.get(field_id, "unknown")
            if field_category == lab_category and field_category != "unknown":
                # 同じカテゴリなら50%のスコアを与える
                related_scores.append(interest.calculate_score() * 0.5)
        
        if related_scores:
            return max(related_scores)
        
        # 無関係な分野の場合
        return 0.1  # 最低スコア
    
    def parse_field_interests(
        self, 
        field_interests_data: Dict
    ) -> Dict[str, FieldInterest]:
        """分野興味データをパース"""
        
        interests = {}
        
        for field_id, data in field_interests_data.items():
            if isinstance(data, dict):
                interests[field_id] = FieldInterest(
                    field_id=field_id,
                    interest_level=float(data.get("interest_level", 5)),
                    experience_level=float(data.get("experience_level", 5)),
                    importance_level=float(data.get("importance_level", 5))
                )
            else:
                # 単純な数値の場合
                interests[field_id] = FieldInterest(
                    field_id=field_id,
                    interest_level=float(data),
                    experience_level=5,
                    importance_level=5
                )
        
        return interests