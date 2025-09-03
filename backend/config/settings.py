"""
設定管理 - config/settings.py (pydantic不要版)
遺伝的アルゴリズムを用いたファジィ決定木システムの設定
"""

import os
from typing import List, Dict, Any


class Settings:
    """システム設定クラス（pydantic不要版）"""
    
    def __init__(self):
        # API設定
        self.app_name = "Lab Matching System with Genetic Fuzzy Decision Tree"
        self.api_version = "v1"
        self.host = "0.0.0.0"
        self.port = 8000
        self.debug = False
        
        # 遺伝的アルゴリズム設定
        self.ga_population_size = 30
        self.ga_generations = 20
        self.ga_mutation_rate = 0.1
        self.ga_crossover_rate = 0.8
        self.ga_elite_size = 3
        self.ga_tournament_size = 3
        
        # ファジィ決定木設定
        self.max_tree_depth = 5
        self.min_samples_split = 10
        self.min_samples_leaf = 5
        self.num_fuzzy_sets = 3  # Low, Medium, High
        
        # メンバーシップ関数設定
        self.membership_types = ["triangular", "gaussian", "trapezoidal"]
        self.default_membership_type = "triangular"
        
        # 最適化設定
        self.fitness_weights = {
            "accuracy": 0.6,
            "complexity": 0.2,
            "interpretability": 0.2
        }
        self.convergence_threshold = 0.001
        self.max_stagnant_generations = 5
        
        # データ設定
        self.data_dir = "./data"
        self.model_dir = "./data/models"
        self.temp_dir = "./data/temp"
        
        # 研究室マッチング設定
        self.max_labs_to_evaluate = 50
        self.min_compatibility_threshold = 0.3
        
        # 特徴量設定
        self.core_features = [
            "research_intensity",
            "advisor_style", 
            "team_work",
            "workload",
            "theory_practice"
        ]
        
        self.extended_features = [
            "research_field_match",
            "skill_development",
            "learning_pace",
            "difficulty_preference",
            "communication_style"
        ]
        
        # ログ設定
        self.log_level = "INFO"
        self.log_file = "./logs/system.log"
        
        # 環境変数から設定を上書き
        self._load_from_env()
    
    def _load_from_env(self):
        """環境変数から設定を読み込み"""
        
        # 環境変数のプレフィックス
        prefix = "GENETIC_FUZZY_"
        
        # 数値設定の環境変数チェック
        numeric_settings = {
            f"{prefix}GA_POPULATION_SIZE": "ga_population_size",
            f"{prefix}GA_GENERATIONS": "ga_generations", 
            f"{prefix}MAX_TREE_DEPTH": "max_tree_depth",
            f"{prefix}MIN_SAMPLES_SPLIT": "min_samples_split",
            f"{prefix}MIN_SAMPLES_LEAF": "min_samples_leaf",
            f"{prefix}PORT": "port"
        }
        
        for env_var, attr_name in numeric_settings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                try:
                    setattr(self, attr_name, int(env_value))
                except ValueError:
                    pass
        
        # 浮動小数点設定の環境変数チェック
        float_settings = {
            f"{prefix}GA_MUTATION_RATE": "ga_mutation_rate",
            f"{prefix}GA_CROSSOVER_RATE": "ga_crossover_rate",
            f"{prefix}CONVERGENCE_THRESHOLD": "convergence_threshold"
        }
        
        for env_var, attr_name in float_settings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                try:
                    setattr(self, attr_name, float(env_value))
                except ValueError:
                    pass
        
        # 文字列設定の環境変数チェック
        string_settings = {
            f"{prefix}DATA_DIR": "data_dir",
            f"{prefix}MODEL_DIR": "model_dir",
            f"{prefix}LOG_LEVEL": "log_level"
        }
        
        for env_var, attr_name in string_settings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                setattr(self, attr_name, env_value)
        
        # ブール設定の環境変数チェック
        bool_settings = {
            f"{prefix}DEBUG": "debug"
        }
        
        for env_var, attr_name in bool_settings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                setattr(self, attr_name, env_value.lower() in ('true', '1', 'yes', 'on'))
    
    def to_dict(self) -> Dict[str, Any]:
        """設定を辞書形式で取得"""
        
        return {
            # API設定
            'app_name': self.app_name,
            'api_version': self.api_version,
            'host': self.host,
            'port': self.port,
            'debug': self.debug,
            
            # 遺伝的アルゴリズム設定
            'ga_population_size': self.ga_population_size,
            'ga_generations': self.ga_generations,
            'ga_mutation_rate': self.ga_mutation_rate,
            'ga_crossover_rate': self.ga_crossover_rate,
            'ga_elite_size': self.ga_elite_size,
            'ga_tournament_size': self.ga_tournament_size,
            
            # ファジィ決定木設定
            'max_tree_depth': self.max_tree_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'num_fuzzy_sets': self.num_fuzzy_sets,
            
            # その他の設定
            'membership_types': self.membership_types,
            'default_membership_type': self.default_membership_type,
            'fitness_weights': self.fitness_weights,
            'core_features': self.core_features,
            'extended_features': self.extended_features
        }
    
    def __repr__(self):
        return f"Settings(population_size={self.ga_population_size}, generations={self.ga_generations}, max_depth={self.max_tree_depth})"


# グローバル設定インスタンス
settings = Settings()