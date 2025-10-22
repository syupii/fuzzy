# -*- coding: utf-8 -*-
"""
デフォルトパラメータ設定（27分野体系）
北海道情報大学 研究室選択支援システム
"""

from typing import Dict, List
from dataclasses import dataclass

# ==================== 評価基準定義 ====================

# 基本項目（5項目）
BASIC_CRITERIA = [
    "research_intensity",      # 研究強度
    "advisor_style",          # 指導スタイル
    "team_work",              # チームワーク
    "workload",               # ワークロード
    "theory_practice"         # 理論・実践バランス
]

# 拡張項目（5項目）
EXTENDED_CRITERIA = [
    "skill_development",      # スキル開発
    "lab_atmosphere",         # 研究室雰囲気
    "flexibility",            # 柔軟性
    "publication_opportunity",# 論文発表機会
    "research_field_match"    # 研究分野適合性（メタパラメータ）
]

# 特殊項目（2項目）
SPECIAL_CRITERIA = [
    "interdisciplinary",      # 学際性
    "communication_style"     # コミュニケーション
]

# 全評価基準（12項目）
ALL_CRITERIA = BASIC_CRITERIA + EXTENDED_CRITERIA + SPECIAL_CRITERIA

# 評価基準の日本語名
CRITERIA_NAMES = {
    "research_intensity": "研究強度",
    "advisor_style": "指導スタイル",
    "team_work": "チームワーク",
    "workload": "ワークロード",
    "theory_practice": "理論・実践バランス",
    "skill_development": "スキル開発",
    "lab_atmosphere": "研究室雰囲気",
    "flexibility": "柔軟性",
    "publication_opportunity": "論文発表機会",
    "research_field_match": "研究分野適合性",
    "interdisciplinary": "学際性",
    "communication_style": "コミュニケーション"
}

# ==================== 研究分野定義（27分野） ====================

# 分野の日本語名マッピング
FIELD_NAMES = {
    # 🔧 テクノロジー・システム分野（10分野）
    "ai_ml": "人工知能・機械学習",
    "image_processing": "画像処理・コンピュータビジョン",
    "cg_graphics": "3DCG・グラフィックス",
    "network_security": "ネットワーク・セキュリティ",
    "database_systems": "データベース・情報システム",
    "embedded_iot": "組込み・IoT・HCI",
    "software_dev": "ソフトウェア開発・アプリ開発",
    "audio_processing": "音声・音響情報処理",
    "data_science_math": "データサイエンス・統計数理",
    "natural_science": "自然科学・地球物理学",
    
    # 📚 教育・言語・文化分野（4分野）
    "japanese_education": "日本語教育・言語学",
    "korean_studies": "韓国語・韓国文化研究",
    "educational_tech": "教育工学・学習支援",
    "english_humanities": "英語・人文学",
    
    # 🌍 観光・地域分野（1分野）
    "tourism_regional": "観光情報・地域システム",
    
    # 🎨 デザイン分野（4分野）
    "web_design_uiux": "Webデザイン・UI/UX",
    "graphic_visual": "グラフィック・視覚デザイン",
    "illustration_art": "イラストレーション・アート",
    "design_thinking_marketing": "デザイン思考・マーケティング",
    
    # 🎬 映像・音楽分野（4分野）
    "video_film": "映像制作・映画",
    "animation": "アニメーション",
    "computer_music": "コンピュータ音楽・サウンドアート",
    "media_art": "メディアアート",
    
    # 🎮 ゲーム・エンタメ分野（3分野）
    "game_dev": "ゲーム開発",
    "esports": "eスポーツ",
    "vr_ar_metaverse": "VR/AR・メタバース",
    
    # 🏃 人文・社会・体育分野（1分野）
    "sports_science": "スポーツ科学・バイオメカニクス",
}

# カテゴリ別分野マッピング
FIELD_CATEGORIES = {
    "テクノロジー・システム": [
        "ai_ml", "image_processing", "cg_graphics",
        "network_security", "database_systems", "embedded_iot",
        "software_dev", "audio_processing", 
        "data_science_math", "natural_science"
    ],
    "教育・言語・文化": [
        "japanese_education", "korean_studies",
        "educational_tech", "english_humanities"
    ],
    "観光・地域": [
        "tourism_regional"
    ],
    "デザイン": [
        "web_design_uiux", "graphic_visual",
        "illustration_art", "design_thinking_marketing"
    ],
    "映像・音楽": [
        "video_film", "animation",
        "computer_music", "media_art"
    ],
    "ゲーム・エンタメ": [
        "game_dev", "esports", "vr_ar_metaverse"
    ],
    "人文・社会・体育": [
        "sports_science"
    ]
}

# ==================== デフォルトパラメータ ====================

@dataclass
class DefaultParams:
    """システムのデフォルトパラメータを管理するクラス"""
    
    # ファジィ推論パラメータ
    similarity_sigma: float = 0.25  # ガウス関数のσ（標準偏差）
    
    # 類似度ボーナス
    exact_match_bonus: float = 1.15  # 完全一致時のボーナス（15%増）
    approximate_match_bonus: float = 1.05  # 近似一致時のボーナス（5%増）
    
    # 優先度パラメータ
    priority_exponent: float = 1.5  # 優先度の指数（重み付けの強さ）
    high_priority_threshold: int = 8  # 高優先度の閾値
    high_priority_bonus_excellent: float = 0.10  # 高優先度項目の優秀スコアボーナス
    high_priority_bonus_good: float = 0.05  # 高優先度項目の良好スコアボーナス
    
    # 分岐閾値（ファジィ決定木用）
    high_threshold: float = 0.7  # 高評価の閾値
    medium_threshold: float = 0.4  # 中評価の閾値
    # low_threshold = 0.0〜medium_threshold
    
    # 推薦レベル閾値
    recommendation_thresholds: Dict[str, float] = None
    
    # 分野マッチング重み
    field_match_weight: float = 0.3  # 分野マッチングの重み
    criteria_match_weight: float = 0.7  # 基本項目マッチングの重み
    
    # 一貫性パラメータ
    consistency_weight: float = 0.3  # 一貫性の重み
    
    def __post_init__(self):
        if self.recommendation_thresholds is None:
            self.recommendation_thresholds = {
                "excellent": 0.85,
                "very_good": 0.75,
                "good": 0.65,
                "fair": 0.50,
                "poor": 0.00
            }

# デフォルト重み（各基準項目の重要度）
DEFAULT_WEIGHTS = {
    "research_intensity": 1.0,
    "advisor_style": 1.0,
    "team_work": 0.9,
    "workload": 0.85,
    "theory_practice": 0.95,
    "skill_development": 0.9,
    "lab_atmosphere": 0.85,
    "flexibility": 0.8,
    "publication_opportunity": 0.9,
    "research_field_match": 1.0,
    "interdisciplinary": 0.75,
    "communication_style": 0.8,
}

# デフォルト優先度（ユーザーが指定しない場合）
DEFAULT_PRIORITIES = {
    "research_intensity": 5,
    "advisor_style": 5,
    "team_work": 5,
    "workload": 5,
    "theory_practice": 5,
    "skill_development": 5,
    "lab_atmosphere": 5,
    "flexibility": 5,
    "publication_opportunity": 5,
    "research_field_match": 5,
    "interdisciplinary": 5,
    "communication_style": 5,
}

# 一貫性重み
consistency_weight = 0.3

# グローバルインスタンス
DEFAULT_PARAMS = DefaultParams()

# ==================== ヘルパー関数 ====================

def get_field_category(field_id: str) -> str:
    """分野IDからカテゴリを取得"""
    for category, fields in FIELD_CATEGORIES.items():
        if field_id in fields:
            return category
    return "unknown"

def get_field_name(field_id: str) -> str:
    """分野IDから日本語名を取得"""
    return FIELD_NAMES.get(field_id, field_id)

def is_same_category(field_id1: str, field_id2: str) -> bool:
    """2つの分野が同じカテゴリか判定"""
    cat1 = get_field_category(field_id1)
    cat2 = get_field_category(field_id2)
    return cat1 == cat2 and cat1 != "unknown"

def get_criterion_name(criterion: str) -> str:
    """評価基準の日本語名を取得"""
    return CRITERIA_NAMES.get(criterion, criterion)

# ==================== 設定サマリー ====================

def print_params_summary():
    """パラメータサマリーを表示"""
    print("=" * 60)
    print("27分野体系パラメータ設定")
    print("=" * 60)
    print(f"\n【類似度計算】")
    print(f"  σ (sigma): {DEFAULT_PARAMS.similarity_sigma}")
    print(f"  完全一致ボーナス: {DEFAULT_PARAMS.exact_match_bonus:.2f}x")
    print(f"  近似一致ボーナス: {DEFAULT_PARAMS.approximate_match_bonus:.2f}x")
    
    print(f"\n【優先度重み付け】")
    print(f"  指数: {DEFAULT_PARAMS.priority_exponent}")
    print(f"  高優先度閾値: {DEFAULT_PARAMS.high_priority_threshold}")
    print(f"  最大ボーナス: +{DEFAULT_PARAMS.high_priority_bonus_excellent:.2%}")
    
    print(f"\n【推薦レベル閾値】")
    for level, threshold in DEFAULT_PARAMS.recommendation_thresholds.items():
        print(f"  {level}: {threshold:.2f}")
    
    print(f"\n【評価基準】")
    print(f"  基本項目数: {len(BASIC_CRITERIA)}")
    print(f"  拡張項目数: {len(EXTENDED_CRITERIA)}")
    print(f"  特殊項目数: {len(SPECIAL_CRITERIA)}")
    print(f"  合計: {len(ALL_CRITERIA)}項目")
    
    print(f"\n【研究分野】")
    total_fields = 0
    for category, fields in FIELD_CATEGORIES.items():
        print(f"  {category}: {len(fields)}分野")
        total_fields += len(fields)
    print(f"  合計: {total_fields}分野")
    
    print("=" * 60)

if __name__ == "__main__":
    print_params_summary()