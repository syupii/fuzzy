# config/default_params.py
"""
デフォルトパラメータ定義
パターンAで使用する固定値（遺伝的アルゴリズムなし）
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DefaultParams:
    """システムのデフォルトパラメータ"""
    
    # ファジィ類似度計算
    similarity_sigma: float = 0.2           # ガウス類似度の広がり
    
    # 分野マッチング
    field_boost_factor: float = 1.5         # 分野スコアのブースト係数
    field_exact_match_bonus: float = 1.0    # 完全一致時のボーナス
    field_category_match_ratio: float = 0.7 # カテゴリ一致時の係数
    field_no_match_penalty: float = 0.3     # 不一致時の基本スコア
    
    # メンバーシップ関数
    membership_overlap: float = 0.3         # メンバーシップ重複度
    membership_low_threshold: float = 0.3   # Low判定の閾値
    membership_high_threshold: float = 0.7  # High判定の閾値
    
    # 決定木
    pruning_threshold: float = 0.01         # 枝刈り閾値
    max_tree_depth: int = 5                 # 最大深さ
    min_samples_leaf: int = 3               # リーフの最小サンプル数
    
    # 重み設定（12項目）
    default_weights: Dict[str, float] = None
    
    def __post_init__(self):
        """初期化後の処理"""
        if self.default_weights is None:
            self.default_weights = {
                "research_intensity": 1.2,      # 研究強度（重要）
                "advisor_style": 1.0,           # 指導スタイル
                "team_work": 0.9,               # チームワーク
                "workload": 1.0,                # ワークロード
                "theory_practice": 1.0,         # 理論・実践バランス
                "skill_development": 0.9,       # スキル開発
                "lab_atmosphere": 1.0,          # 研究室雰囲気
                "flexibility": 0.8,             # 柔軟性
                "publication_opportunity": 1.2, # 論文発表機会（重要）
                "interdisciplinary": 0.8,       # 学際性
                "communication_style": 0.9,     # コミュニケーション
                "innovation_focus": 1.0         # 革新性重視
            }


# グローバルインスタンス
DEFAULT_PARAMS = DefaultParams()


# 評価基準の定義（13項目）
EVALUATION_CRITERIA = [
    # 基本項目（5項目）
    "research_intensity",       # 研究強度
    "advisor_style",            # 指導スタイル
    "team_work",                # チームワーク
    "workload",                 # ワークロード
    "theory_practice",          # 理論・実践バランス
    
    # 拡張項目（5項目）
    "research_field_match",     # 研究分野適合性（比重指数）
    "skill_development",        # スキル開発
    "lab_atmosphere",           # 研究室雰囲気
    "flexibility",              # 柔軟性
    "publication_opportunity",  # 論文発表機会
    
    # 特殊項目（3項目）
    "interdisciplinary",        # 学際性
    "communication_style",      # コミュニケーション
    "innovation_focus"          # 革新性重視
]

# 基本12項目（research_field_matchを除く）
BASIC_CRITERIA = [
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
    "communication_style",
    "innovation_focus"
]


# 分野カテゴリマッピング（20分野）
FIELD_CATEGORIES = {
    # テクノロジー・システム（12分野）
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
    
    # クリエイティブ（4分野）
    "web_design": "creative",
    "design_visual": "creative",
    "video_animation": "creative",
    "computer_music": "creative",
    
    # エンターテイメント（2分野）
    "game_esports": "entertainment",
    "vr_ar_media": "entertainment",
    
    # 人文・社会・体育（2分野）
    "philosophy_humanities": "humanities",
    "sports_science": "humanities"
}


# 分野名マッピング
FIELD_NAMES = {
    # テクノロジー・システム
    "ai_ml": "人工知能・機械学習",
    "image_processing": "画像・映像処理",
    "network_security": "ネットワーク・セキュリティ",
    "database_systems": "データベース・情報システム",
    "embedded_iot": "組込み・IoT",
    "education_linguistics": "教育・言語学",
    "natural_science_math": "自然科学・数理",
    "tourism_regional": "観光情報・地域システム",
    "business_decision": "経営情報・意思決定支援",
    "audio_processing": "音声・音響情報処理",
    "system_ethics": "システム運用・情報倫理",
    
    # クリエイティブ
    "web_design": "Webデザイン・UI/UX",
    "design_visual": "デザイン・視覚表現",
    "video_animation": "映像・アニメーション",
    "computer_music": "コンピュータ音楽・サウンドアート",
    
    # エンターテイメント
    "game_esports": "ゲーム開発・eスポーツ",
    "vr_ar_media": "VR/AR・メディアアート",
    
    # 人文・社会・体育
    "philosophy_humanities": "哲学・人文・環境行動学",
    "sports_science": "スポーツ・体育科学"
}


def get_field_category(field_id: str) -> str:
    """分野IDからカテゴリを取得"""
    return FIELD_CATEGORIES.get(field_id, "unknown")


def get_field_name(field_id: str) -> str:
    """分野IDから日本語名を取得"""
    return FIELD_NAMES.get(field_id, field_id)


def is_same_category(field_id1: str, field_id2: str) -> bool:
    """2つの分野が同じカテゴリに属するか"""
    cat1 = get_field_category(field_id1)
    cat2 = get_field_category(field_id2)
    return cat1 == cat2 and cat1 != "unknown"