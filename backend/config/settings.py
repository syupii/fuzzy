# config/settings.py
"""システム設定"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Settings:
    """システム設定クラス"""
    
    # アプリケーション設定
    app_name: str = "Lab Matching System with Genetic Fuzzy Decision Tree"
    api_version: str = "v1"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # 評価基準（13項目）
    evaluation_criteria: List[str] = field(default_factory=lambda: [
        "research_intensity",
        "advisor_style",
        "team_work",
        "workload",
        "theory_practice",
        "research_field_match",
        "skill_development",
        "lab_atmosphere",
        "flexibility",
        "publication_opportunity",
        "interdisciplinary",
        "communication_style"
    ])
    
    # レガシー互換用（5項目）
    core_features: List[str] = field(default_factory=lambda: [
        "research_intensity",
        "advisor_style",
        "team_work",
        "workload",
        "theory_practice"
    ])
    
    # 遺伝的アルゴリズム設定
    ga_population_size: int = 20
    ga_generations: int = 30
    ga_mutation_rate: float = 0.1
    ga_crossover_rate: float = 0.8
    ga_elitism_rate: float = 0.1
    
    # 決定木設定
    max_tree_depth: int = 8  # 5 → 8に変更
    min_samples_split: int = 6
    min_samples_leaf: int = 3
    
    # ファジィ設定
    fuzzy_linguistic_terms: int = 3  # Low, Medium, High
    fuzzy_membership_overlap: float = 0.3
    
    # 分野設定
    enable_field_matching: bool = True
    field_weight_in_optimization: float = 0.85
    
    # データパス
    training_data_path: str = "data/training_data.json"
    optimized_weights_path: str = "data/optimized_weights.npy"


# グローバル設定インスタンス
settings = Settings()