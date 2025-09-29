"""
システム設定ファイル
13項目完全対応版
"""

from typing import List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class EvaluationCriteria:
    """評価基準定義"""
    
    # 基本5項目（必須）
    BASIC_CRITERIA: List[str] = field(default_factory=lambda: [
        "research_intensity",  # 研究強度
        "advisor_style",       # 指導スタイル
        "team_work",          # チームワーク
        "workload",           # ワークロード
        "theory_practice"     # 理論・実践バランス
    ])
    
    # 拡張5項目
    EXTENDED_CRITERIA: List[str] = field(default_factory=lambda: [
        "research_field_match",      # 研究分野適合性
        "skill_development",         # スキル開発
        "lab_atmosphere",           # 研究室雰囲気
        "flexibility",              # 柔軟性
        "publication_opportunity"   # 論文発表機会
    ])
    
    # 特殊3項目
    SPECIAL_CRITERIA: List[str] = field(default_factory=lambda: [
        "interdisciplinary",      # 学際性
        "communication_style",    # コミュニケーション
        "innovation_risk"        # 革新性
    ])
    
    @property
    def all_criteria(self) -> List[str]:
        """全評価基準"""
        return self.BASIC_CRITERIA + self.EXTENDED_CRITERIA + self.SPECIAL_CRITERIA
    
    @property
    def criteria_descriptions(self) -> Dict[str, str]:
        """評価基準の説明"""
        return {
            # 基本項目
            "research_intensity": "研究強度: 研究にどれだけ集中的に取り組みたいか（1=軽い 〜 10=集中的）",
            "advisor_style": "指導スタイル: 教授からの指導の受け方の好み（1=厳格 〜 10=自由）",
            "team_work": "チームワーク: 研究での他者との協働の程度（1=個人 〜 10=チーム）",
            "workload": "ワークロード: 研究活動の忙しさに対する許容度（1=軽い 〜 10=重い）",
            "theory_practice": "理論・実践: 理論研究と実践的研究のバランス（1=理論 〜 10=実践）",
            
            # 拡張項目
            "research_field_match": "研究分野適合性: 自分の興味と研究室の分野の一致度（1=広い 〜 10=専門特化）",
            "skill_development": "スキル開発: 専門性と汎用性のバランス（1=専門特化 〜 10=幅広い）",
            "lab_atmosphere": "研究室雰囲気: 研究室の全体的な雰囲気（1=静寂集中 〜 10=活発議論）",
            "flexibility": "柔軟性: 研究時間の自由度（1=固定スケジュール 〜 10=柔軟）",
            "publication_opportunity": "論文発表機会: 研究成果の論文化機会（1=少ない 〜 10=豊富）",
            
            # 特殊項目
            "interdisciplinary": "学際性: 他分野との連携の程度（1=単一分野 〜 10=学際連携）",
            "communication_style": "コミュニケーション: 研究室での交流スタイル（1=少人数密接 〜 10=オープン）",
            "innovation_risk": "革新性: 新しい試みへの挑戦度（1=保守的 〜 10=革新的）"
        }
    
    @property
    def criteria_ranges(self) -> Dict[str, tuple]:
        """評価基準の範囲"""
        return {criterion: (0.0, 1.0) for criterion in self.all_criteria}
    
    @property
    def importance_weights(self) -> Dict[str, float]:
        """項目の重要度（合計=1.0）"""
        return {
            # 基本項目（40%）
            "research_intensity": 0.12,
            "advisor_style": 0.08,
            "team_work": 0.08,
            "workload": 0.06,
            "theory_practice": 0.06,
            
            # 拡張項目（35%）
            "research_field_match": 0.10,
            "skill_development": 0.07,
            "lab_atmosphere": 0.07,
            "flexibility": 0.06,
            "publication_opportunity": 0.05,
            
            # 特殊項目（25%）
            "interdisciplinary": 0.09,
            "communication_style": 0.08,
            "innovation_risk": 0.08
        }


@dataclass
class ResearchCategories:
    """研究分野カテゴリ定義"""
    
    CATEGORIES: Dict[str, List[str]] = field(default_factory=lambda: {
        "テクノロジー・システム": [
            "人工知能・機械学習",
            "画像・映像処理",
            "ネットワーク・セキュリティ",
            "データベース・情報システム",
            "組込み・IoT",
            "教育・言語学",
            "自然科学・数理",
            "観光情報・地域システム",
            "経営情報・意思決定支援",
            "音声・音響情報処理",
            "システム運用・情報倫理"
        ],
        "クリエイティブ": [
            "Webデザイン・UI/UX",
            "デザイン・視覚表現",
            "映像・アニメーション",
            "コンピュータ音楽・サウンドアート"
        ],
        "エンターテイメント": [
            "ゲーム開発・eスポーツ",
            "VR/AR・メディアアート"
        ],
        "人文・社会・体育": [
            "哲学・人文・環境行動学",
            "スポーツ・体育科学"
        ]
    })
    
    @property
    def all_fields(self) -> List[str]:
        """全研究分野"""
        all_fields = []
        for fields in self.CATEGORIES.values():
            all_fields.extend(fields)
        return all_fields
    
    def get_category_for_field(self, field: str) -> str:
        """分野からカテゴリを取得"""
        for category, fields in self.CATEGORIES.items():
            if field in fields:
                return category
        return "未分類"


@dataclass
class FuzzyTreeConfig:
    """ファジィ決定木設定"""
    
    # 決定木パラメータ
    max_depth: int = 3
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    
    # ファジィパラメータ
    membership_type: str = "triangular"  # triangular, gaussian, trapezoidal
    fuzzy_sets_per_variable: int = 3  # 3, 5, 7
    
    # Level 1分類に使用する特徴
    level1_feature: str = "research_intensity"
    
    # Level 2分類マップ
    level2_features: Dict[str, str] = field(default_factory=lambda: {
        "high": "team_work",
        "medium": "flexibility",
        "low": "lab_atmosphere"
    })
    
    # クラスタ定義
    cluster_definitions: Dict[str, Dict] = field(default_factory=lambda: {
        "high_team_oriented": {
            "description": "高研究強度・チーム志向",
            "characteristics": ["高い研究意欲", "チーム研究", "協力的"]
        },
        "high_individual_focused": {
            "description": "高研究強度・個人志向",
            "characteristics": ["高い研究意欲", "個人研究", "独立的"]
        },
        "medium_flexible_style": {
            "description": "中研究強度・柔軟志向",
            "characteristics": ["バランス型", "柔軟なスタイル", "適応力"]
        },
        "medium_structured_style": {
            "description": "中研究強度・構造志向",
            "characteristics": ["バランス型", "計画的", "規律的"]
        },
        "low_active_atmosphere": {
            "description": "軽研究負荷・活発志向",
            "characteristics": ["軽い負荷", "活発な雰囲気", "社交的"]
        },
        "low_quiet_atmosphere": {
            "description": "軽研究負荷・静寂志向",
            "characteristics": ["軽い負荷", "静かな雰囲気", "集中型"]
        }
    })


@dataclass
class GeneticAlgorithmConfig:
    """遺伝的アルゴリズム設定"""
    
    # 基本パラメータ
    population_size: int = 50
    generations: int = 100
    elite_size: int = 5
    
    # 遺伝的操作パラメータ
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    
    # 選択方法
    selection_method: str = "tournament"  # tournament, roulette, rank
    tournament_size: int = 5
    
    # 適合度関数パラメータ
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        "compatibility": 0.6,    # 適合度
        "diversity": 0.2,        # 多様性
        "balance": 0.2          # 研究室間のバランス
    })


@dataclass
class SystemSettings:
    """システム全体設定"""
    
    # アプリケーション情報
    app_name: str = "研究室選択支援システム with Genetic Fuzzy Decision Tree"
    version: str = "3.0.0"
    api_version: str = "v3"
    
    # サーバー設定
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # コンポーネント設定
    evaluation_criteria: EvaluationCriteria = field(default_factory=EvaluationCriteria)
    research_categories: ResearchCategories = field(default_factory=ResearchCategories)
    fuzzy_tree_config: FuzzyTreeConfig = field(default_factory=FuzzyTreeConfig)
    genetic_config: GeneticAlgorithmConfig = field(default_factory=GeneticAlgorithmConfig)
    
    # 機能フラグ
    enable_fuzzy_inference: bool = True
    enable_genetic_optimization: bool = True
    enable_multi_level_tree: bool = True
    enable_priority_weighting: bool = True
    
    # ログ設定
    log_level: str = "INFO"
    log_file: str = "system.log"
    
    # キャッシュ設定
    enable_cache: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # 秒
    
    @property
    def core_features(self) -> List[str]:
        """コア機能（基本5項目）"""
        return self.evaluation_criteria.BASIC_CRITERIA
    
    @property
    def all_features(self) -> List[str]:
        """全機能（13項目）"""
        return self.evaluation_criteria.all_criteria
    
    def get_feature_weight(self, feature: str) -> float:
        """特徴の重要度を取得"""
        return self.evaluation_criteria.importance_weights.get(feature, 0.0)
    
    def validate_profile(self, profile: Dict[str, float]) -> tuple[bool, List[str]]:
        """プロファイルを検証
        
        Returns:
            (is_valid, missing_fields)
        """
        missing = []
        
        # 基本項目は必須
        for criterion in self.evaluation_criteria.BASIC_CRITERIA:
            if criterion not in profile:
                missing.append(criterion)
        
        is_valid = len(missing) == 0
        
        return is_valid, missing
    
    def get_system_info(self) -> Dict[str, Any]:
        """システム情報を取得"""
        return {
            "name": self.app_name,
            "version": self.version,
            "api_version": self.api_version,
            "features": {
                "evaluation_criteria": len(self.all_features),
                "fuzzy_inference": self.enable_fuzzy_inference,
                "genetic_optimization": self.enable_genetic_optimization,
                "multi_level_tree": self.enable_multi_level_tree,
                "priority_weighting": self.enable_priority_weighting
            },
            "research_fields": len(self.research_categories.all_fields),
            "categories": list(self.research_categories.CATEGORIES.keys())
        }


# グローバル設定インスタンス
settings = SystemSettings()


# 使用例
if __name__ == "__main__":
    print("=" * 70)
    print("システム設定情報")
    print("=" * 70)
    
    print(f"\n📌 アプリケーション: {settings.app_name}")
    print(f"📌 バージョン: {settings.version}")
    
    print(f"\n📊 評価基準:")
    print(f"  基本項目: {len(settings.evaluation_criteria.BASIC_CRITERIA)}項目")
    print(f"  拡張項目: {len(settings.evaluation_criteria.EXTENDED_CRITERIA)}項目")
    print(f"  特殊項目: {len(settings.evaluation_criteria.SPECIAL_CRITERIA)}項目")
    print(f"  合計: {len(settings.all_features)}項目")
    
    print(f"\n🔬 研究分野:")
    for category, fields in settings.research_categories.CATEGORIES.items():
        print(f"  {category}: {len(fields)}分野")
    
    print(f"\n🌳 ファジィ決定木設定:")
    print(f"  最大深さ: {settings.fuzzy_tree_config.max_depth}")
    print(f"  メンバーシップタイプ: {settings.fuzzy_tree_config.membership_type}")
    print(f"  クラスタ数: {len(settings.fuzzy_tree_config.cluster_definitions)}")
    
    print(f"\n🧬 遺伝的アルゴリズム設定:")
    print(f"  集団サイズ: {settings.genetic_config.population_size}")
    print(f"  世代数: {settings.genetic_config.generations}")
    print(f"  交叉率: {settings.genetic_config.crossover_rate}")
    print(f"  突然変異率: {settings.genetic_config.mutation_rate}")
    
    print("\n" + "=" * 70)