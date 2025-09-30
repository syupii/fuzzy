# config/settings.py - 完全版システム設定

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
TEMP_DIR = DATA_DIR / "temp"

# ディレクトリ作成
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)


# ===============================
# 評価基準定義
# ===============================

class EvaluationCriteria:
    """評価基準クラス"""
    
    # 基本5項目
    BASIC_CRITERIA = [
        "research_intensity",
        "advisor_style",
        "team_work",
        "workload",
        "theory_practice"
    ]
    
    # 拡張5項目
    EXTENDED_CRITERIA = [
        "research_field_match",
        "skill_development",
        "lab_atmosphere",
        "flexibility",
        "publication_opportunity"
    ]
    
    # 特殊3項目
    SPECIAL_CRITERIA = [
        "interdisciplinary",
        "communication_style",
        "innovation_risk"
    ]
    
    # 全項目
    ALL_CRITERIA = BASIC_CRITERIA + EXTENDED_CRITERIA + SPECIAL_CRITERIA
    
    # 日本語名マッピング
    CRITERIA_NAMES_JP = {
        "research_intensity": "研究強度",
        "advisor_style": "指導スタイル",
        "team_work": "チームワーク",
        "workload": "ワークロード",
        "theory_practice": "理論・実践バランス",
        "research_field_match": "研究分野適合性",
        "skill_development": "スキル開発",
        "lab_atmosphere": "研究室雰囲気",
        "flexibility": "柔軟性",
        "publication_opportunity": "論文発表機会",
        "interdisciplinary": "学際性",
        "communication_style": "コミュニケーション",
        "innovation_risk": "革新性とリスク"
    }
    
    # 説明
    CRITERIA_DESCRIPTIONS = {
        "research_intensity": "研究にどれだけ集中的に取り組みたいか（1:軽い研究 ～ 10:集中研究）",
        "advisor_style": "教授からの指導の受け方の好み（1:厳格指導 ～ 10:自由指導）",
        "team_work": "研究での他者との協働の程度（1:個人研究 ～ 10:チーム研究）",
        "workload": "研究活動の忙しさに対する許容度（1:軽い負荷 ～ 10:重い負荷）",
        "theory_practice": "理論研究と実践的研究のバランス（1:理論重視 ～ 10:実践重視）",
        "research_field_match": "自分の興味と研究室の分野の一致度（1:広い分野 ～ 10:専門特化）",
        "skill_development": "専門性と汎用性のバランス（1:専門特化 ～ 10:幅広いスキル）",
        "lab_atmosphere": "研究室の全体的な雰囲気（1:静寂集中 ～ 10:活発議論）",
        "flexibility": "研究時間の自由度（1:固定スケジュール ～ 10:柔軟スケジュール）",
        "publication_opportunity": "研究成果の論文化機会（1:少ない機会 ～ 10:豊富な機会）",
        "interdisciplinary": "他分野との連携の程度（1:単一分野 ～ 10:学際連携）",
        "communication_style": "研究室での交流スタイル（1:少人数密接 ～ 10:オープン交流）",
        "innovation_risk": "革新的な研究への挑戦度（1:安定志向 ～ 10:挑戦志向）"
    }


# ===============================
# 研究分野定義
# ===============================

class ResearchFields:
    """研究分野クラス"""
    
    # テクノロジー・システム（11分野）
    TECHNOLOGY_FIELDS = [
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
    ]
    
    # クリエイティブ（4分野）
    CREATIVE_FIELDS = [
        "Webデザイン・UI/UX",
        "デザイン・視覚表現",
        "映像・アニメーション",
        "コンピュータ音楽・サウンドアート"
    ]
    
    # エンターテイメント（2分野）
    ENTERTAINMENT_FIELDS = [
        "ゲーム開発・eスポーツ",
        "VR/AR・メディアアート"
    ]
    
    # 人文・社会・体育（2分野）
    HUMANITIES_FIELDS = [
        "哲学・人文・環境行動学",
        "スポーツ・体育科学"
    ]
    
    # 全分野
    ALL_FIELDS = (
        TECHNOLOGY_FIELDS + 
        CREATIVE_FIELDS + 
        ENTERTAINMENT_FIELDS + 
        HUMANITIES_FIELDS
    )
    
    # カテゴリマッピング
    CATEGORIES = {
        "テクノロジー・システム": TECHNOLOGY_FIELDS,
        "クリエイティブ": CREATIVE_FIELDS,
        "エンターテイメント": ENTERTAINMENT_FIELDS,
        "人文・社会・体育": HUMANITIES_FIELDS
    }


# ===============================
# ファジィ推論設定
# ===============================

@dataclass
class FuzzyConfig:
    """ファジィ推論設定"""
    
    # メンバーシップ関数設定
    membership_type: str = "triangular"  # "triangular", "gaussian", "trapezoidal"
    linguistic_terms: int = 3  # 言語値の数（3, 5, 7）
    membership_overlap: float = 0.3  # オーバーラップ率
    
    # 推論設定
    inference_method: str = "mamdani"  # "mamdani", "sugeno"
    defuzzification_method: str = "centroid"  # "centroid", "max", "mean"
    
    # ルール設定
    rule_weight_threshold: float = 0.1  # ルール重み閾値
    min_rule_confidence: float = 0.2  # 最小ルール信頼度
    
    # 計算精度
    universe_resolution: int = 100  # 論理領域の分解能


# ===============================
# 遺伝的アルゴリズム設定
# ===============================

@dataclass
class GeneticConfig:
    """遺伝的アルゴリズム設定"""
    
    # 基本パラメータ
    population_size: int = 30
    generations: int = 50
    elite_size: int = 3
    
    # 遺伝的操作
    crossover_rate: float = 0.8
    mutation_rate: float = 0.15
    mutation_strength: float = 0.1
    tournament_size: int = 3
    
    # 選択・交叉・変異手法
    selection_method: str = "tournament"  # "tournament", "roulette", "rank"
    crossover_method: str = "uniform"  # "uniform", "single_point", "two_point"
    mutation_method: str = "gaussian"  # "gaussian", "uniform", "adaptive"
    
    # 収束判定
    early_stopping: bool = True
    patience: int = 15
    min_improvement: float = 1e-6
    convergence_threshold: float = 1e-4
    
    # 適応的パラメータ
    adaptive_parameters: bool = True
    adaptive_mutation_range: tuple = (0.05, 0.25)
    adaptive_crossover_range: tuple = (0.6, 0.9)
    
    # 多様性維持
    diversity_preservation: bool = True
    min_diversity: float = 0.1
    
    # 並列化
    parallel_evaluation: bool = False
    num_processes: int = 4


# ===============================
# ファジィ決定木設定
# ===============================

@dataclass
class DecisionTreeConfig:
    """ファジィ決定木設定"""
    
    # 木構造パラメータ
    max_depth: int = 8
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Optional[int] = None
    
    # ファジィ関連
    fuzzy_threshold: float = 0.1
    linguistic_terms: int = 3
    
    # 分岐基準
    split_criterion: str = "fuzzy_gain"  # "fuzzy_gain", "gini", "entropy"
    min_impurity_decrease: float = 1e-7
    
    # 枝刈り
    pruning_enabled: bool = True
    min_confidence_threshold: float = 0.1
    
    # ルール生成
    rule_extraction: bool = True
    max_rules_per_path: int = 10


# ===============================
# システム設定
# ===============================

@dataclass
class SystemSettings:
    """システム全体設定"""
    
    # アプリケーション情報
    app_name: str = "研究室選択支援システム"
    version: str = "3.0.0"
    api_version: str = "v1"
    description: str = "遺伝的アルゴリズムを用いたファジィ決定木による研究室マッチングシステム"
    
    # サーバー設定
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    reload: bool = True
    
    # データベース設定
    database_path: Path = DATA_DIR / "labs_database.json"
    backup_enabled: bool = True
    backup_dir: Path = DATA_DIR / "backups"
    
    # ログ設定
    log_level: str = "INFO"
    log_file: Path = DATA_DIR / "system.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # キャッシュ設定
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1時間
    cache_maxsize: int = 1000
    
    # API設定
    api_rate_limit: int = 100  # requests per minute
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    
    # 評価基準・研究分野
    evaluation_criteria: EvaluationCriteria = field(default_factory=EvaluationCriteria)
    research_fields: ResearchFields = field(default_factory=ResearchFields)
    
    # モジュール設定
    fuzzy_config: FuzzyConfig = field(default_factory=FuzzyConfig)
    genetic_config: GeneticConfig = field(default_factory=GeneticConfig)
    tree_config: DecisionTreeConfig = field(default_factory=DecisionTreeConfig)
    
    # 機能フラグ
    enable_fuzzy_inference: bool = True
    enable_genetic_optimization: bool = True
    enable_decision_tree: bool = True
    enable_priority_weighting: bool = True
    enable_field_matching: bool = True
    
    # パフォーマンス設定
    max_concurrent_evaluations: int = 10
    evaluation_timeout: int = 60  # 秒
    optimization_timeout: int = 300  # 5分
    
    @property
    def all_criteria(self) -> List[str]:
        """全評価基準を取得"""
        return self.evaluation_criteria.ALL_CRITERIA
    
    @property
    def all_fields(self) -> List[str]:
        """全研究分野を取得"""
        return self.research_fields.ALL_FIELDS
    
    def get_criteria_info(self, criterion: str) -> Dict[str, str]:
        """評価基準の詳細情報を取得"""
        return {
            "name": criterion,
            "name_jp": self.evaluation_criteria.CRITERIA_NAMES_JP.get(criterion, criterion),
            "description": self.evaluation_criteria.CRITERIA_DESCRIPTIONS.get(criterion, ""),
            "range": "1-10"
        }
    
    def validate_student_profile(self, profile: Dict[str, Any]) -> tuple[bool, List[str]]:
        """学生プロファイルを検証"""
        missing = []
        
        for criterion in self.all_criteria:
            if criterion not in profile:
                missing.append(criterion)
            elif not isinstance(profile[criterion], (int, float)):
                missing.append(f"{criterion} (invalid type)")
            elif not (0 <= profile[criterion] <= 10):
                missing.append(f"{criterion} (out of range)")
        
        return len(missing) == 0, missing
    
    def get_system_info(self) -> Dict[str, Any]:
        """システム情報を取得"""
        return {
            "name": self.app_name,
            "version": self.version,
            "api_version": self.api_version,
            "description": self.description,
            "features": {
                "evaluation_criteria": len(self.all_criteria),
                "research_fields": len(self.all_fields),
                "fuzzy_inference": self.enable_fuzzy_inference,
                "genetic_optimization": self.enable_genetic_optimization,
                "decision_tree": self.enable_decision_tree,
                "priority_weighting": self.enable_priority_weighting,
                "field_matching": self.enable_field_matching
            },
            "performance": {
                "max_concurrent_evaluations": self.max_concurrent_evaluations,
                "evaluation_timeout": self.evaluation_timeout,
                "optimization_timeout": self.optimization_timeout
            }
        }


# ===============================
# グローバル設定インスタンス
# ===============================

settings = SystemSettings()


# ===============================
# 環境変数からの設定上書き
# ===============================

def load_from_env():
    """環境変数から設定を読み込み"""
    
    # サーバー設定
    settings.host = os.getenv("APP_HOST", settings.host)
    settings.port = int(os.getenv("APP_PORT", str(settings.port)))
    settings.debug = os.getenv("APP_DEBUG", "true").lower() == "true"
    
    # データベース設定
    db_path = os.getenv("DATABASE_PATH")
    if db_path:
        settings.database_path = Path(db_path)
    
    # ログ設定
    settings.log_level = os.getenv("LOG_LEVEL", settings.log_level)
    
    # 機能フラグ
    settings.enable_fuzzy_inference = os.getenv(
        "ENABLE_FUZZY", "true"
    ).lower() == "true"
    settings.enable_genetic_optimization = os.getenv(
        "ENABLE_GENETIC", "true"
    ).lower() == "true"


# 環境変数読み込み実行
load_from_env()


# ===============================
# テスト・デバッグ用
# ===============================

if __name__ == "__main__":
    print("=" * 70)
    print("システム設定情報")
    print("=" * 70)
    
    info = settings.get_system_info()
    
    print(f"\n📌 アプリケーション:")
    print(f"  名前: {info['name']}")
    print(f"  バージョン: {info['version']}")
    print(f"  説明: {info['description']}")
    
    print(f"\n📊 機能:")
    for key, value in info['features'].items():
        print(f"  {key}: {value}")
    
    print(f"\n⚙️  評価基準:")
    print(f"  基本項目: {len(settings.evaluation_criteria.BASIC_CRITERIA)}項目")
    print(f"  拡張項目: {len(settings.evaluation_criteria.EXTENDED_CRITERIA)}項目")
    print(f"  特殊項目: {len(settings.evaluation_criteria.SPECIAL_CRITERIA)}項目")
    print(f"  合計: {len(settings.all_criteria)}項目")
    
    print(f"\n🔬 研究分野:")
    for category, fields in settings.research_fields.CATEGORIES.items():
        print(f"  {category}: {len(fields)}分野")
    
    print(f"\n🧬 遺伝的アルゴリズム:")
    print(f"  集団サイズ: {settings.genetic_config.population_size}")
    print(f"  世代数: {settings.genetic_config.generations}")
    print(f"  交叉率: {settings.genetic_config.crossover_rate}")
    print(f"  変異率: {settings.genetic_config.mutation_rate}")
    
    print("\n" + "=" * 70)