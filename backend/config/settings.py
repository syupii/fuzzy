from pydantic import BaseSettings
from typing import List, Dict, Any

class Settings(BaseSettings):
    # API設定
    app_name: str = "Lab Matching System"
    api_version: str = "v1"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # 遺伝的アルゴリズム設定
    ga_population_size: int = 50
    ga_generations: int = 30
    ga_mutation_rate: float = 0.1
    ga_crossover_rate: float = 0.8
    
    # ファジィ決定木設定
    max_tree_depth: int = 6
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    
    # データベース設定
    data_dir: str = "./data"
    model_dir: str = "./data/models"
    
    class Config:
        env_file = ".env"