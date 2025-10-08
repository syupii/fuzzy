# config/default_params.py - 改善版

"""
デフォルトパラメータ設定（改善版）

主な変更点:
1. similarity_sigma: 0.2 → 0.25 (より緩やかな類似度計算)
2. priority_exponent: 1.5 → 1.8 (優先度の非線形性を強化)
3. 完全一致・近似一致ボーナスの追加
"""

# ==================== 評価基準 ====================

BASIC_CRITERIA = [
    "research_intensity",      # 研究強度
    "advisor_style",           # 指導スタイル
    "team_work",              # チームワーク
    "workload",               # ワークロード
    "theory_practice",        # 理論・実践バランス
    "research_field_match",   # 研究分野適合性（分野重視度）
    "skill_development",      # スキル開発
    "lab_atmosphere",         # 研究室雰囲気
    "flexibility",            # 柔軟性
    "publication_opportunity", # 論文発表機会
    "interdisciplinary",      # 学際性
    "communication_style"     # コミュニケーション
]

# 日本語名マッピング
CRITERIA_NAMES = {
    "research_intensity": "研究強度",
    "advisor_style": "指導スタイル",
    "team_work": "チームワーク",
    "workload": "ワークロード",
    "theory_practice": "理論・実践バランス",
    "research_field_match": "分野重視度",
    "skill_development": "スキル開発",
    "lab_atmosphere": "研究室雰囲気",
    "flexibility": "柔軟性",
    "publication_opportunity": "論文発表機会",
    "interdisciplinary": "学際性",
    "communication_style": "コミュニケーション"
}

# ==================== 研究分野 ====================

FIELD_CATEGORIES = {
    "technology": [
        "ai_ml", "image_processing", "network_security",
        "database", "iot", "education_linguistics",
        "natural_science", "tourism_systems",
        "business_systems", "audio_processing",
        "system_operations"
    ],
    "creative": [
        "web_design", "visual_design", "video_animation",
        "computer_music"
    ],
    "entertainment": [
        "game_dev", "vr_ar"
    ],
    "humanities_sports": [
        "philosophy", "sports_science"
    ]
}

FIELD_NAMES = {
    # テクノロジー・システム
    "ai_ml": "人工知能・機械学習",
    "image_processing": "画像・映像処理",
    "network_security": "ネットワーク・セキュリティ",
    "database": "データベース・情報システム",
    "iot": "組込み・IoT",
    "education_linguistics": "教育・言語学",
    "natural_science": "自然科学・数理",
    "tourism_systems": "観光情報・地域システム",
    "business_systems": "経営情報・意思決定支援",
    "audio_processing": "音声・音響情報処理",
    "system_operations": "システム運用・情報倫理",
    
    # クリエイティブ
    "web_design": "Webデザイン・UI/UX",
    "visual_design": "デザイン・視覚表現",
    "video_animation": "映像・アニメーション",
    "computer_music": "コンピュータ音楽・サウンドアート",
    
    # エンターテイメント
    "game_dev": "ゲーム開発・eスポーツ",
    "vr_ar": "VR/AR・メディアアート",
    
    # 人文・スポーツ
    "philosophy": "哲学・人文・環境行動学",
    "sports_science": "スポーツ・体育科学"
}

# ==================== パラメータクラス ====================

class DefaultParams:
    """
    デフォルトパラメータ（改善版）
    """
    
    # 基本重み（均等）
    default_weights = {criterion: 1.0 for criterion in BASIC_CRITERIA}
    
    # ★ 改善: 類似度計算パラメータ
    similarity_sigma = 0.25  # 0.2 → 0.25 (より緩やか)
    
    # ★ 改善: 一致ボーナス
    exact_match_bonus = 1.15      # 完全一致: +15%
    approximate_match_bonus = 1.08 # 近似一致: +8%
    exact_match_threshold = 0.05   # 完全一致の閾値
    approximate_match_threshold = 0.15  # 近似一致の閾値
    
    # ★ 改善: 優先度重み計算
    priority_exponent = 1.8  # 1.5 → 1.8 (より強い非線形性)
    
    # ★ 改善: 高優先度ボーナス
    high_priority_threshold = 7.0   # 高優先度の閾値
    high_priority_bonus_excellent = 0.15  # avg >= 0.85
    high_priority_bonus_good = 0.10       # avg >= 0.75
    high_priority_bonus_moderate = 0.05   # avg >= 0.65
    
    # 分野マッチングパラメータ
    field_exact_match_bonus = 1.0     # 完全一致時のスコア
    field_category_match_ratio = 0.6  # カテゴリ一致時の割合
    field_no_match_penalty = 0.3      # 不一致時のペナルティ
    
    # 推薦レベル閾値（改善版）
    recommendation_thresholds = {
        "strongly_recommended": 0.80,  # 0.85 → 0.80
        "recommended": 0.65,           # 0.70 → 0.65
        "consider": 0.45,              # 0.50 → 0.45
        "careful": 0.30                # 0.35 → 0.30
    }
    
    # 信頼度計算
    confidence_high_priority_weight = 0.7
    confidence_consistency_weight = 0.3

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
    print("改善版パラメータ設定")
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
    
    print(f"\n【研究分野】")
    for category, fields in FIELD_CATEGORIES.items():
        print(f"  {category}: {len(fields)}分野")
    
    print("=" * 60)

if __name__ == "__main__":
    print_params_summary()