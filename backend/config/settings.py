# config/settings.py - システム設定管理

import os
from typing import Dict, List, Any, Optional
from pydantic import BaseSettings, Field
from enum import Enum

class Environment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"

class Settings(BaseSettings):
    """システム設定クラス"""
    
    # アプリケーション基本設定
    app_name: str = "研究室選択支援システム"
    api_version: str = "v1"
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    
    # サーバー設定
    host: str = "0.0.0.0"
    port: int = 8000
    
    # 遺伝的アルゴリズム設定
    ga_population_size: int = 30
    ga_generations: int = 50
    ga_mutation_rate: float = 0.1
    ga_crossover_rate: float = 0.8
    ga_elite_size: int = 5
    
    # ファジィ決定木設定
    max_tree_depth: int = 8
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    fuzzy_threshold: float = 0.1
    
    # 評価基準（13項目）
    evaluation_criteria: List[str] = [
        # 基本項目（5項目）
        "research_intensity",      # 研究強度
        "advisor_style",           # 指導スタイル
        "team_work",               # チームワーク
        "workload",                # ワークロード
        "theory_practice",         # 理論・実践バランス
        
        # 拡張項目（5項目）
        "research_field_match",    # 研究分野適合性
        "skill_development",       # スキル開発
        "lab_atmosphere",          # 研究室雰囲気
        "flexibility",             # 柔軟性
        "publication_opportunity", # 論文発表機会
        
        # 特殊項目（3項目）
        "interdisciplinary",       # 学際性
        "communication_style",     # コミュニケーション
        "innovation_risk"          # 革新性・リスク許容度
    ]
    
    # 研究分野（11分野）
    research_fields: List[str] = [
        # テクノロジー・システム分野（5分野）
        "ai_machine_learning",         # 人工知能・機械学習
        "image_video_processing",      # 画像・映像処理
        "network_security",            # コンピュータネットワーク・セキュリティ
        "database_information_systems", # データベース・情報システム
        "embedded_iot",                # 組込み・IoT
        
        # クリエイティブ分野（4分野）
        "web_design_ui_ux",           # Webデザイン・UI/UX
        "design_visual_expression",   # デザイン・視覚表現
        "video_animation",            # 映像・アニメーション
        "computer_music_sound_art",   # コンピュータ音楽・サウンドアート
        
        # エンターテイメント分野（2分野）
        "game_development_esports",   # ゲーム開発・eスポーツ
        "vr_ar_media_art"            # VR/AR・メディアアート
    ]
    
    # 研究分野カテゴリ
    field_categories: Dict[str, List[str]] = {
        "technology_systems": [
            "ai_machine_learning",
            "image_video_processing", 
            "network_security",
            "database_information_systems",
            "embedded_iot"
        ],
        "creative": [
            "web_design_ui_ux",
            "design_visual_expression",
            "video_animation",
            "computer_music_sound_art"
        ],
        "entertainment": [
            "game_development_esports",
            "vr_ar_media_art"
        ]
    }
    
    # 基本特徴量（必須）
    core_features: List[str] = [
        "research_intensity",
        "advisor_style", 
        "team_work",
        "workload",
        "theory_practice"
    ]
    
    # 教員情報
    faculty_data: Dict[str, Any] = {
        "ai_machine_learning": [
            {"name": "伊藤雅彦", "name_en": "Masahiko ITOH", 
             "specialties": ["情報可視化", "ユーザインタフェース", "データ工学"]},
            {"name": "内山敏雄", "name_en": "Toshio UCHIYAMA",
             "specialties": ["データ解析", "機械学習", "レコメンド", "テキストマイニング"]},
            {"name": "小野亮太", "name_en": "Ryota ONO",
             "specialties": ["人工知能", "情報工学", "マルチエージェントシステム", "情報推薦", "観光情報"]},
            {"name": "齋藤健司", "name_en": "Kenji SAITO",
             "specialties": ["人工知能", "教育システム", "仮想環境"]},
            {"name": "谷口文武", "name_en": "Fumitake TANIGUCHI",
             "specialties": ["機械学習", "パターン認識"]},
            {"name": "辻準平", "name_en": "Junpei TSUJI",
             "specialties": ["社会シミュレーション", "マルチエージェントシステム", "IoT"]},
            {"name": "山北貴典", "name_en": "Takanori YAMAKITA",
             "specialties": ["データベース技術"]}
        ],
        "image_video_processing": [
            {"name": "森圭佑", "name_en": "Keisuke MORI",
             "specialties": ["情報計測", "音声・画像情報処理", "医用情報処理", "ゲームプログラミング", "組み込み機器"]},
            {"name": "向田茂", "name_en": "Shigeru MUKAIDA",
             "specialties": ["画像処理", "顔学", "認知心理学", "VR/AR", "3DCG", "メディアアート"]},
            {"name": "高井奈美", "name_en": "Nami TAKAI",
             "specialties": ["コンピュータグラフィックス", "画像処理", "Webデザイン"]},
            {"name": "藤原孝行", "name_en": "Takayuki FUJIWARA",
             "specialties": ["コンピュータビジョン", "コンピュータグラフィックス"]},
            {"name": "越野一博", "name_en": "Kazuhiro KOSHINO",
             "specialties": ["医用画像工学", "数理統計学", "人工知能画像解析処理"]},
            {"name": "上杉正人", "name_en": "Masahito UESUGI",
             "specialties": ["医療情報システム開発", "医療言語処理", "画像処理"]}
        ]
        # 他の分野も同様に定義...
    }
    
    # ファイルパス設定
    data_dir: str = "./data"
    models_dir: str = "./data/models"
    temp_dir: str = "./data/temp"
    logs_dir: str = "./logs"
    
    # ログ設定
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # セキュリティ設定
    cors_origins: List[str] = ["*"]
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]
    
    # データベース設定（将来用）
    database_url: Optional[str] = None
    
    # 最適化設定
    optimization_timeout: int = 300  # 5分
    max_concurrent_optimizations: int = 5
    
    # 実験設定
    random_seed: int = 42
    test_mode: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

# グローバル設定インスタンス
settings = Settings()

# 設定検証
def validate_settings():
    """設定値の検証"""
    
    errors = []
    
    # 必須ディレクトリの存在確認
    required_dirs = [settings.data_dir, settings.models_dir, settings.temp_dir]
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
            except Exception as e:
                errors.append(f"ディレクトリ作成失敗: {dir_path} - {e}")
    
    # パラメータ範囲チェック
    if not (0 < settings.ga_mutation_rate < 1):
        errors.append(f"変異率が範囲外: {settings.ga_mutation_rate}")
    
    if not (0 < settings.ga_crossover_rate < 1):
        errors.append(f"交叉率が範囲外: {settings.ga_crossover_rate}")
    
    if settings.ga_population_size < 2:
        errors.append(f"集団サイズが小さすぎ: {settings.ga_population_size}")
    
    if settings.max_tree_depth < 1:
        errors.append(f"木の深度が小さすぎ: {settings.max_tree_depth}")
    
    # エラーがある場合は例外発生
    if errors:
        raise ValueError("設定エラー: " + "; ".join(errors))
    
    return True

# 設定初期化時に検証実行
if __name__ != "__main__":
    try:
        validate_settings()
    except Exception as e:
        print(f"⚠️ 設定検証エラー: {e}")