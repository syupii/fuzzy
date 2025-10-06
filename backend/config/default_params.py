# config/default_params.py
"""
デフォルトパラメータ設定 v3.0
- 12項目評価基準
- 20研究分野（4カテゴリ）
- パターンB対応
"""

from dataclasses import dataclass
from typing import Dict, List


# ===== 評価基準（13項目） =====

BASIC_CRITERIA = [
    # 基本項目（5項目）
    "research_intensity",      # 研究強度
    "advisor_style",           # 指導スタイル
    "team_work",              # チームワーク
    "workload",               # ワークロード
    "theory_practice",        # 理論・実践バランス
    
    # 拡張項目（5項目）
    "research_field_match",   # 分野重視度
    "skill_development",      # スキル開発
    "lab_atmosphere",         # 研究室雰囲気
    "flexibility",            # 柔軟性
    "publication_opportunity", # 論文発表機会
    
    # 特殊項目（2項目）
    "interdisciplinary",      # 学際性
    "communication_style",    # コミュニケーション
]

# 評価基準の詳細情報
CRITERIA_DETAILS = {
    # 基本項目
    "research_intensity": {
        "name": "研究強度",
        "description": "研究にどれだけ集中的に取り組みたいか",
        "range": "1（軽い研究）～ 10（集中研究）",
        "category": "basic"
    },
    "advisor_style": {
        "name": "指導スタイル",
        "description": "教授からの指導の受け方の好み",
        "range": "1（厳格指導）～ 10（自由指導）",
        "category": "basic"
    },
    "team_work": {
        "name": "チームワーク",
        "description": "研究での他者との協働の程度",
        "range": "1（個人研究）～ 10（チーム研究）",
        "category": "basic"
    },
    "workload": {
        "name": "ワークロード",
        "description": "研究活動の忙しさに対する許容度",
        "range": "1（軽い負荷）～ 10（重い負荷）",
        "category": "basic"
    },
    "theory_practice": {
        "name": "理論・実践バランス",
        "description": "理論研究と実践的研究のバランス",
        "range": "1（理論重視）～ 10（実践重視）",
        "category": "basic"
    },
    
    # 拡張項目
    "research_field_match": {
        "name": "分野重視度",
        "description": "分野マッチングと基本項目のどちらに比重を置くか",
        "range": "1（基本項目重視）～ 10（分野重視）",
        "category": "extended"
    },
    "skill_development": {
        "name": "スキル開発",
        "description": "専門性と汎用性のバランス",
        "range": "1（専門特化）～ 10（幅広いスキル）",
        "category": "extended"
    },
    "lab_atmosphere": {
        "name": "研究室雰囲気",
        "description": "研究室の全体的な雰囲気",
        "range": "1（静寂集中）～ 10（活発議論）",
        "category": "extended"
    },
    "flexibility": {
        "name": "柔軟性",
        "description": "研究時間の自由度",
        "range": "1（固定スケジュール）～ 10（柔軟スケジュール）",
        "category": "extended"
    },
    "publication_opportunity": {
        "name": "論文発表機会",
        "description": "研究成果の論文化機会",
        "range": "1（少ない機会）～ 10（豊富な機会）",
        "category": "extended"
    },
    
    # 特殊項目
    "interdisciplinary": {
        "name": "学際性",
        "description": "他分野との連携の程度",
        "range": "1（単一分野）～ 10（学際連携）",
        "category": "special"
    },
    "communication_style": {
        "name": "コミュニケーション",
        "description": "研究室での交流スタイル",
        "range": "1（少人数密接）～ 10（オープン交流）",
        "category": "special"
    },
}


# ===== デフォルト重み =====

DEFAULT_WEIGHTS = {
    # 基本項目（重み: 1.2）
    "research_intensity": 1.2,
    "advisor_style": 1.2,
    "team_work": 1.2,
    "workload": 1.2,
    "theory_practice": 1.2,
    
    # 拡張項目（重み: 1.0）
    "research_field_match": 1.0,
    "skill_development": 1.0,
    "lab_atmosphere": 1.0,
    "flexibility": 1.0,
    "publication_opportunity": 1.0,
    
    # 特殊項目（重み: 0.8）
    "interdisciplinary": 0.8,
    "communication_style": 0.8,
}


# ===== 研究分野定義（20分野） =====

FIELD_NAMES = {
    # テクノロジー・システム分野（12分野）
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
    "medical_healthcare": "医療情報・ヘルスケア",
    
    # クリエイティブ分野（4分野）
    "web_design": "Webデザイン・UI/UX",
    "design_visual": "デザイン・視覚表現",
    "video_animation": "映像・アニメーション",
    "computer_music": "コンピュータ音楽・サウンドアート",
    
    # エンターテイメント分野（2分野）
    "game_esports": "ゲーム開発・eスポーツ",
    "vr_ar_media": "VR/AR・メディアアート",
    
    # 人文・社会・体育分野（2分野）
    "philosophy_humanities": "哲学・人文・環境行動学",
    "sports_science": "スポーツ・体育科学",
}

# カテゴリマッピング
FIELD_CATEGORIES = {
    "テクノロジー・システム": [
        "ai_ml", "image_processing", "network_security",
        "database_systems", "embedded_iot", "education_linguistics",
        "natural_science_math", "tourism_regional", "business_decision",
        "audio_processing", "system_ethics", "medical_healthcare"
    ],
    "クリエイティブ": [
        "web_design", "design_visual", "video_animation", "computer_music"
    ],
    "エンターテイメント": [
        "game_esports", "vr_ar_media"
    ],
    "人文・社会・体育": [
        "philosophy_humanities", "sports_science"
    ]
}

# 分野詳細情報
FIELD_DETAILS = {
    # テクノロジー・システム
    "ai_ml": {
        "name": "人工知能・機械学習",
        "category": "テクノロジー・システム",
        "description": "AI、機械学習、ディープラーニング、自然言語処理の研究",
        "faculty_count": 7,
        "keywords": ["AI", "機械学習", "ディープラーニング", "NLP"]
    },
    "image_processing": {
        "name": "画像・映像処理",
        "category": "テクノロジー・システム",
        "description": "コンピュータビジョン、画像認識、医用画像処理の研究",
        "faculty_count": 6,
        "keywords": ["画像処理", "CV", "パターン認識"]
    },
    "network_security": {
        "name": "ネットワーク・セキュリティ",
        "category": "テクノロジー・システム",
        "description": "ネットワーク技術、情報セキュリティ、暗号化の研究",
        "faculty_count": 3,
        "keywords": ["ネットワーク", "セキュリティ", "暗号"]
    },
    "database_systems": {
        "name": "データベース・情報システム",
        "category": "テクノロジー・システム",
        "description": "データベース技術、情報システム、ビッグデータ処理の研究",
        "faculty_count": 3,
        "keywords": ["データベース", "情報システム", "ビッグデータ"]
    },
    "embedded_iot": {
        "name": "組込み・IoT",
        "category": "テクノロジー・システム",
        "description": "組込みシステム、IoT、ユビキタスコンピューティングの研究",
        "faculty_count": 2,
        "keywords": ["組込み", "IoT", "ユビキタス"]
    },
    "education_linguistics": {
        "name": "教育・言語学",
        "category": "テクノロジー・システム",
        "description": "教育工学、言語処理、eラーニングシステムの研究",
        "faculty_count": 5,
        "keywords": ["教育工学", "言語学", "eラーニング"]
    },
    "natural_science_math": {
        "name": "自然科学・数理",
        "category": "テクノロジー・システム",
        "description": "数理科学、シミュレーション、科学計算の研究",
        "faculty_count": 6,
        "keywords": ["数理科学", "シミュレーション", "科学計算"]
    },
    "tourism_regional": {
        "name": "観光情報・地域システム",
        "category": "テクノロジー・システム",
        "description": "観光情報学、地域活性化、GISの研究",
        "faculty_count": 2,
        "keywords": ["観光情報", "地域システム", "GIS"]
    },
    "business_decision": {
        "name": "経営情報・意思決定支援",
        "category": "テクノロジー・システム",
        "description": "経営情報システム、意思決定支援、データ分析の研究",
        "faculty_count": 3,
        "keywords": ["経営情報", "意思決定", "データ分析"]
    },
    "audio_processing": {
        "name": "音声・音響情報処理",
        "category": "テクノロジー・システム",
        "description": "音声認識、音響信号処理、音楽情報処理の研究",
        "faculty_count": 2,
        "keywords": ["音声処理", "音響", "音楽情報"]
    },
    "system_ethics": {
        "name": "システム運用・情報倫理",
        "category": "テクノロジー・システム",
        "description": "システム管理、情報倫理、ICT社会論の研究",
        "faculty_count": 3,
        "keywords": ["システム運用", "情報倫理", "ICT"]
    },
    "medical_healthcare": {
        "name": "医療情報・ヘルスケア",
        "category": "テクノロジー・システム",
        "description": "医療情報システム、ヘルスケアIT、遠隔医療の研究",
        "faculty_count": 2,
        "keywords": ["医療情報", "ヘルスケア", "遠隔医療"]
    },
    
    # クリエイティブ
    "web_design": {
        "name": "Webデザイン・UI/UX",
        "category": "クリエイティブ",
        "description": "Webデザイン、ユーザインタフェース、UX設計の研究",
        "faculty_count": 4,
        "keywords": ["Webデザイン", "UI/UX", "インタラクション"]
    },
    "design_visual": {
        "name": "デザイン・視覚表現",
        "category": "クリエイティブ",
        "description": "グラフィックデザイン、視覚デザイン、ブランディングの研究",
        "faculty_count": 4,
        "keywords": ["デザイン", "視覚表現", "グラフィック"]
    },
    "video_animation": {
        "name": "映像・アニメーション",
        "category": "クリエイティブ",
        "description": "映像制作、アニメーション表現、メディアアートの研究",
        "faculty_count": 2,
        "keywords": ["映像", "アニメーション", "メディアアート"]
    },
    "computer_music": {
        "name": "コンピュータ音楽・サウンドアート",
        "category": "クリエイティブ",
        "description": "コンピュータ音楽、サウンドデザイン、音響芸術の研究",
        "faculty_count": 2,
        "keywords": ["コンピュータ音楽", "サウンドアート", "音響芸術"]
    },
    
    # エンターテイメント
    "game_esports": {
        "name": "ゲーム開発・eスポーツ",
        "category": "エンターテイメント",
        "description": "ゲーム開発、ゲームデザイン、eスポーツ産業の研究",
        "faculty_count": 2,
        "keywords": ["ゲーム開発", "eスポーツ", "ゲームデザイン"]
    },
    "vr_ar_media": {
        "name": "VR/AR・メディアアート",
        "category": "エンターテイメント",
        "description": "仮想現実、拡張現実、インタラクティブアートの研究",
        "faculty_count": 2,
        "keywords": ["VR", "AR", "メディアアート"]
    },
    
    # 人文・社会・体育
    "philosophy_humanities": {
        "name": "哲学・人文・環境行動学",
        "category": "人文・社会・体育",
        "description": "哲学、人文科学、環境行動学の研究",
        "faculty_count": 2,
        "keywords": ["哲学", "人文学", "環境行動学"]
    },
    "sports_science": {
        "name": "スポーツ・体育科学",
        "category": "人文・社会・体育",
        "description": "スポーツ科学、体育学、健康科学の研究",
        "faculty_count": 2,
        "keywords": ["スポーツ科学", "体育学", "健康科学"]
    },
}


# ===== パターンB設定 =====

# 優先度閾値
HIGH_PRIORITY_THRESHOLD = 8.0   # 高優先度（3分岐）
MID_PRIORITY_THRESHOLD = 5.0    # 中優先度（2分岐）

# 分岐設定
BRANCH_CONFIG = {
    "high_priority": {
        "branches": 3,
        "split_points": [0.3, 0.7],
        "labels": ["低", "中", "高"]
    },
    "mid_priority": {
        "branches": 2,
        "split_points": [0.5],
        "labels": ["低", "高"]
    }
}


# ===== デフォルトパラメータクラス =====

@dataclass
class DefaultParams:
    """デフォルトパラメータ"""
    
    # 評価基準
    criteria: List[str] = None
    default_weights: Dict[str, float] = None
    
    # ファジィパラメータ
    similarity_sigma: float = 0.2
    
    # 優先度閾値（パターンB）
    high_priority_threshold: float = HIGH_PRIORITY_THRESHOLD
    mid_priority_threshold: float = MID_PRIORITY_THRESHOLD
    
    # 分野マッチング
    exact_match_weight: float = 1.0
    category_match_weight: float = 0.7
    no_match_weight: float = 0.3
    
    def __post_init__(self):
        if self.criteria is None:
            self.criteria = BASIC_CRITERIA
        if self.default_weights is None:
            self.default_weights = DEFAULT_WEIGHTS


# グローバルインスタンス
DEFAULT_PARAMS = DefaultParams()


# ===== ヘルパー関数 =====

def get_field_name(field_id: str) -> str:
    """分野IDから分野名を取得"""
    return FIELD_NAMES.get(field_id, "不明な分野")


def get_field_category(field_id: str) -> str:
    """分野IDからカテゴリを取得"""
    for category, fields in FIELD_CATEGORIES.items():
        if field_id in fields:
            return category
    return "不明なカテゴリ"


def is_same_category(field_id1: str, field_id2: str) -> bool:
    """2つの分野が同じカテゴリか判定"""
    cat1 = get_field_category(field_id1)
    cat2 = get_field_category(field_id2)
    return cat1 == cat2 and cat1 != "不明なカテゴリ"


def get_all_fields() -> List[Dict[str, any]]:
    """全分野情報を取得"""
    fields = []
    for field_id, details in FIELD_DETAILS.items():
        fields.append({
            "id": field_id,
            **details
        })
    return fields


def get_fields_by_category(category: str) -> List[str]:
    """カテゴリから分野IDリストを取得"""
    return FIELD_CATEGORIES.get(category, [])


# ===== 情報表示 =====

if __name__ == "__main__":
    print("="*60)
    print("デフォルトパラメータ v3.0")
    print("="*60)
    
    print(f"\n📊 評価基準: {len(BASIC_CRITERIA)}項目")
    for i, criterion in enumerate(BASIC_CRITERIA, 1):
        detail = CRITERIA_DETAILS[criterion]
        print(f"  {i:2d}. {detail['name']} ({detail['category']})")
    
    print(f"\n🏛️ 研究分野: {len(FIELD_NAMES)}分野")
    for category, fields in FIELD_CATEGORIES.items():
        print(f"\n  【{category}】({len(fields)}分野)")
        for field_id in fields:
            name = FIELD_NAMES[field_id]
            count = FIELD_DETAILS[field_id]['faculty_count']
            print(f"    - {name} ({count}名)")
    
    print(f"\n⚙️ パターンB設定")
    print(f"  - 高優先度閾値: {HIGH_PRIORITY_THRESHOLD} (3分岐)")
    print(f"  - 中優先度閾値: {MID_PRIORITY_THRESHOLD} (2分岐)")
    
    print(f"\n💡 research_field_match の意味")
    print(f"  - 1-10の値で「分野」と「基本項目」の比重を決定")
    print(f"  - 1: 基本項目を重視（分野10%・基本項目90%）")
    print(f"  - 5: 中間（分野50%・基本項目50%）")
    print(f"  - 10: 分野を重視（分野100%・基本項目0%）")
    print("="*60)