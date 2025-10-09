# backend/core/matching/field_matcher_corrected.py
"""
分野マッチャー - 技術資料準拠版

減衰係数: 0.7（技術資料 3.6節）
不一致ペナルティ: 0.3（技術資料 3.6節）
"""

from typing import Dict, Any, Tuple
from dataclasses import dataclass


# 分野カテゴリ定義（チャットコンテキストより）
FIELD_CATEGORIES = {
    "technology_system": {
        "name": "テクノロジー・システム分野",
        "fields": {
            "ai_ml": "人工知能・機械学習",
            "image_processing": "画像・映像処理",
            "network_security": "ネットワーク・セキュリティ",
            "database_info_system": "データベース・情報システム",
            "embedded_iot": "組込み・IoT",
            "education_linguistics": "教育・言語学",
            "natural_science_math": "自然科学・数理",
            "tourism_regional_system": "観光情報・地域システム",
            "management_decision_support": "経営情報・意思決定支援",
            "audio_processing": "音声・音響情報処理",
            "system_operation_ethics": "システム運用・情報倫理"
        }
    },
    "creative": {
        "name": "クリエイティブ分野",
        "fields": {
            "web_design_ui_ux": "Webデザイン・UI/UX",
            "design_visual": "デザイン・視覚表現",
            "video_animation": "映像・アニメーション",
            "computer_music_sound_art": "コンピュータ音楽・サウンドアート"
        }
    },
    "entertainment": {
        "name": "エンターテイメント分野",
        "fields": {
            "game_dev_esports": "ゲーム開発・eスポーツ",
            "vr_ar_media_art": "VR/AR・メディアアート"
        }
    },
    "humanities_social_sports": {
        "name": "人文・社会・体育分野",
        "fields": {
            "philosophy_humanities": "哲学・人文・環境行動学",
            "sports_science": "スポーツ・体育科学"
        }
    }
}


@dataclass
class FieldMatchResult:
    """分野マッチング結果"""
    score: float
    match_type: str  # "exact", "category", "none"
    lab_field: str
    lab_field_name: str
    student_field: str = None
    student_interest_level: float = None
    category_name: str = None
    message: str = ""


class FieldMatcherCorrected:
    """
    分野マッチャー（技術資料準拠版）
    
    技術資料 3.6節の定義に完全準拠：
    - 完全一致: I/10
    - カテゴリ一致: I/10 × 0.7  ★ 減衰係数0.7
    - 不一致: 0.3  ★ 固定値0.3
    """
    
    # 技術資料のパラメータ
    CATEGORY_MATCH_DECAY = 0.7  # ★ 減衰係数（技術資料 3.6節）
    NO_MATCH_PENALTY = 0.3      # ★ 不一致固定値（技術資料 3.6節）
    
    def __init__(self):
        self.field_categories = FIELD_CATEGORIES
        self._build_field_to_category_map()
        print("✅ FieldMatcherCorrected 初期化完了")
        print(f"   - 減衰係数: {self.CATEGORY_MATCH_DECAY} （技術資料準拠）")
        print(f"   - 不一致値: {self.NO_MATCH_PENALTY} （技術資料準拠）")
    
    def _build_field_to_category_map(self):
        """フィールドIDからカテゴリへのマップを構築"""
        self.field_to_category = {}
        
        for category_id, category_data in self.field_categories.items():
            for field_id, field_name in category_data["fields"].items():
                self.field_to_category[field_id] = {
                    "category_id": category_id,
                    "category_name": category_data["name"],
                    "field_name": field_name
                }
    
    def calculate_field_match(
        self,
        student_field_interests: Dict[str, float],
        lab_field_id: str
    ) -> FieldMatchResult:
        """
        分野マッチングスコア計算（技術資料 3.6節）
        
        Args:
            student_field_interests: 学生の分野興味 {field_id: interest_level}
            lab_field_id: 研究室の分野ID
        
        Returns:
            FieldMatchResult: マッチング結果
        """
        
        # 研究室分野の情報取得
        lab_info = self.field_to_category.get(lab_field_id)
        if not lab_info:
            return FieldMatchResult(
                score=0.5,
                match_type="unknown",
                lab_field=lab_field_id,
                lab_field_name="不明",
                message="研究室の分野情報が不明"
            )
        
        lab_field_name = lab_info["field_name"]
        lab_category_id = lab_info["category_id"]
        lab_category_name = lab_info["category_name"]
        
        # 学生の興味が空の場合
        if not student_field_interests:
            return FieldMatchResult(
                score=0.5,
                match_type="no_interest",
                lab_field=lab_field_id,
                lab_field_name=lab_field_name,
                message="学生の興味分野が未設定"
            )
        
        # 1. 完全一致チェック
        if lab_field_id in student_field_interests:
            interest_level = student_field_interests[lab_field_id]
            score = interest_level / 10.0  # ★ 技術資料 3.6節
            
            return FieldMatchResult(
                score=score,
                match_type="exact",
                lab_field=lab_field_id,
                lab_field_name=lab_field_name,
                student_field=lab_field_id,
                student_interest_level=interest_level,
                category_name=lab_category_name,
                message=f"興味分野と完全一致（興味度: {interest_level}/10）"
            )
        
        # 2. カテゴリ一致チェック
        best_category_match = None
        best_category_score = 0.0
        
        for interest_field_id, interest_level in student_field_interests.items():
            interest_info = self.field_to_category.get(interest_field_id)
            if not interest_info:
                continue
            
            # 同じカテゴリに属するか
            if interest_info["category_id"] == lab_category_id:
                # ★ 減衰係数 0.7 を適用（技術資料 3.6節）
                category_score = (interest_level / 10.0) * self.CATEGORY_MATCH_DECAY
                
                if category_score > best_category_score:
                    best_category_score = category_score
                    best_category_match = {
                        "field_id": interest_field_id,
                        "field_name": interest_info["field_name"],
                        "interest_level": interest_level
                    }
        
        if best_category_match:
            return FieldMatchResult(
                score=best_category_score,
                match_type="category",
                lab_field=lab_field_id,
                lab_field_name=lab_field_name,
                student_field=best_category_match["field_id"],
                student_interest_level=best_category_match["interest_level"],
                category_name=lab_category_name,
                message=(
                    f"同カテゴリの関連分野に興味 "
                    f"（{best_category_match['field_name']}、"
                    f"興味度: {best_category_match['interest_level']}/10、"
                    f"減衰係数{self.CATEGORY_MATCH_DECAY}適用）"
                )
            )
        
        # 3. 不一致
        # ★ 固定値 0.3（技術資料 3.6節）
        return FieldMatchResult(
            score=self.NO_MATCH_PENALTY,
            match_type="none",
            lab_field=lab_field_id,
            lab_field_name=lab_field_name,
            category_name=lab_category_name,
            message=f"興味分野との関連なし（固定値{self.NO_MATCH_PENALTY}）"
        )
    
    def get_field_name(self, field_id: str) -> str:
        """フィールドIDから名前を取得"""
        info = self.field_to_category.get(field_id)
        return info["field_name"] if info else field_id
    
    def get_category_name(self, field_id: str) -> str:
        """フィールドIDからカテゴリ名を取得"""
        info = self.field_to_category.get(field_id)
        return info["category_name"] if info else "不明"
    
    def is_same_category(self, field1_id: str, field2_id: str) -> bool:
        """2つのフィールドが同じカテゴリに属するか"""
        info1 = self.field_to_category.get(field1_id)
        info2 = self.field_to_category.get(field2_id)
        
        if not info1 or not info2:
            return False
        
        return info1["category_id"] == info2["category_id"]


# テスト用
if __name__ == "__main__":
    print("=" * 60)
    print("FieldMatcherCorrected テスト")
    print("=" * 60)
    
    matcher = FieldMatcherCorrected()
    
    # テストケース1: 完全一致
    print("\n【テスト1: 完全一致】")
    result1 = matcher.calculate_field_match(
        student_field_interests={"ai_ml": 10},
        lab_field_id="ai_ml"
    )
    print(f"スコア: {result1.score:.3f} (期待値: 1.000)")
    print(f"タイプ: {result1.match_type} (期待: exact)")
    print(f"メッセージ: {result1.message}")
    
    # テストケース2: カテゴリ一致
    print("\n【テスト2: カテゴリ一致】")
    result2 = matcher.calculate_field_match(
        student_field_interests={"ai_ml": 10},
        lab_field_id="image_processing"  # 同じtechnology_systemカテゴリ
    )
    print(f"スコア: {result2.score:.3f} (期待値: 0.700)")
    print(f"タイプ: {result2.match_type} (期待: category)")
    print(f"メッセージ: {result2.message}")
    
    # テストケース3: 不一致
    print("\n【テスト3: 不一致】")
    result3 = matcher.calculate_field_match(
        student_field_interests={"ai_ml": 10},
        lab_field_id="web_design_ui_ux"  # 異なるcreativeカテゴリ
    )
    print(f"スコア: {result3.score:.3f} (期待値: 0.300)")
    print(f"タイプ: {result3.match_type} (期待: none)")
    print(f"メッセージ: {result3.message}")
    
    print("\n" + "=" * 60)
    print("✅ テスト完了")