# config/settings.py - システム設定管理（修正版）

import os
from typing import Dict, List, Any, Optional
from enum import Enum

# Pydantic v2対応: BaseSettingsの代替
try:
    from pydantic_settings import BaseSettings
    PYDANTIC_SETTINGS_AVAILABLE = True
except ImportError:
    try:
        from pydantic import BaseSettings
        PYDANTIC_SETTINGS_AVAILABLE = True
    except ImportError:
        PYDANTIC_SETTINGS_AVAILABLE = False
        BaseSettings = object

class Environment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"

if PYDANTIC_SETTINGS_AVAILABLE:
    class Settings(BaseSettings):
        """システム設定クラス（Pydantic使用）"""
        
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
        
        @property
        def core_features(self) -> List[str]:
            """コア機能（基本5項目）"""
            return self.evaluation_criteria[:5]
        
        # 研究分野（11分野）
        research_fields: List[str] = [
            # テクノロジー・システム分野（5分野）
            "ai_machine_learning",        # 人工知能・機械学習
            "image_video_processing",     # 画像・映像処理
            "network_security",           # ネットワーク・セキュリティ
            "database_information_systems", # データベース・情報システム
            "embedded_iot",               # 組込み・IoT
            
            # クリエイティブ分野（4分野）
            "web_design_ui_ux",          # Webデザイン・UI/UX
            "design_visual_expression",  # デザイン・視覚表現
            "video_animation",           # 映像・アニメーション
            "computer_music_sound_art",  # コンピュータ音楽・サウンドアート
            
            # エンターテイメント分野（2分野）
            "game_development_esports",  # ゲーム開発・eスポーツ
            "vr_ar_media_art"           # VR/AR・メディアアート
        ]
        
        # 分野カテゴリ
        field_categories: Dict[str, List[str]] = {
            "テクノロジー・システム": [
                "ai_machine_learning", "image_video_processing", 
                "network_security", "database_information_systems", "embedded_iot"
            ],
            "クリエイティブ": [
                "web_design_ui_ux", "design_visual_expression",
                "video_animation", "computer_music_sound_art"
            ],
            "エンターテイメント": [
                "game_development_esports", "vr_ar_media_art"
            ]
        }
        
        class Config:
            env_file = ".env"
            case_sensitive = False

else:
    # Pydantic が利用できない場合のフォールバック
    class Settings:
        """システム設定クラス（フォールバック版）"""
        
        def __init__(self):
            # アプリケーション基本設定
            self.app_name = "研究室選択支援システム"
            self.api_version = "v1"
            self.environment = Environment.DEVELOPMENT
            self.debug = True
            
            # サーバー設定
            self.host = "0.0.0.0"
            self.port = 8000
            
            # 遺伝的アルゴリズム設定
            self.ga_population_size = 30
            self.ga_generations = 50
            self.ga_mutation_rate = 0.1
            self.ga_crossover_rate = 0.8
            self.ga_elite_size = 5
            
            # ファジィ決定木設定
            self.max_tree_depth = 8
            self.min_samples_split = 2
            self.min_samples_leaf = 1
            self.fuzzy_threshold = 0.1
            
            # 評価基準（13項目）
            self.evaluation_criteria = [
                # 基本項目（5項目）
                "research_intensity", "advisor_style", "team_work", 
                "workload", "theory_practice",
                
                # 拡張項目（5項目）
                "research_field_match", "skill_development", "lab_atmosphere",
                "flexibility", "publication_opportunity",
                
                # 特殊項目（3項目）
                "interdisciplinary", "communication_style", "innovation_risk"
            ]
            
            # 研究分野（11分野）
            self.research_fields = [
                "ai_machine_learning", "image_video_processing", 
                "network_security", "database_information_systems", "embedded_iot",
                "web_design_ui_ux", "design_visual_expression",
                "video_animation", "computer_music_sound_art",
                "game_development_esports", "vr_ar_media_art"
            ]
            
            # 分野カテゴリ
            self.field_categories = {
                "テクノロジー・システム": [
                    "ai_machine_learning", "image_video_processing", 
                    "network_security", "database_information_systems", "embedded_iot"
                ],
                "クリエイティブ": [
                    "web_design_ui_ux", "design_visual_expression",
                    "video_animation", "computer_music_sound_art"
                ],
                "エンターテイメント": [
                    "game_development_esports", "vr_ar_media_art"
                ]
            }
        
        @property
        def core_features(self) -> List[str]:
            """コア機能（基本5項目）"""
            return self.evaluation_criteria[:5]

# シングルトンパターンでインスタンス作成
settings = Settings()

# 研究室データテンプレート
LABORATORY_TEMPLATE = {
    "basic_info": {
        "id": "",
        "name": "", 
        "advisor": "",
        "description": ""
    },
    "characteristics": {
        "research_intensity": 5.0,
        "advisor_style": 5.0,
        "team_work": 5.0,
        "workload": 5.0,
        "theory_practice": 5.0,
        "research_field_match": 5.0,
        "skill_development": 5.0,
        "lab_atmosphere": 5.0,
        "flexibility": 5.0,
        "publication_opportunity": 5.0,
        "interdisciplinary": 5.0,
        "communication_style": 5.0,
        "innovation_risk": 5.0
    },
    "metadata": {
        "fields": [],
        "publications": 0,
        "funding": "中",
        "equipment": "",
        "graduate_employment": ""
    }
}

def validate_settings() -> List[str]:
    """設定値の検証"""
    errors = []
    
    try:
        # 基本設定の検証
        if settings.port < 1 or settings.port > 65535:
            errors.append("ポート番号が無効です")
        
        if settings.ga_population_size < 2:
            errors.append("遺伝的アルゴリズムの集団サイズが小さすぎます")
        
        if len(settings.evaluation_criteria) != 13:
            errors.append(f"評価基準数が13個ではありません（現在: {len(settings.evaluation_criteria)}個）")
        
        if len(settings.research_fields) != 11:
            errors.append(f"研究分野数が11個ではありません（現在: {len(settings.research_fields)}個）")
        
    except Exception as e:
        errors.append(f"設定検証中にエラー: {str(e)}")
    
    return errors

def get_settings_info() -> Dict[str, Any]:
    """設定情報の取得"""
    return {
        "pydantic_available": PYDANTIC_SETTINGS_AVAILABLE,
        "app_name": settings.app_name,
        "version": settings.api_version,
        "environment": settings.environment,
        "evaluation_criteria_count": len(settings.evaluation_criteria),
        "research_fields_count": len(settings.research_fields),
        "genetic_algorithm": {
            "population_size": settings.ga_population_size,
            "generations": settings.ga_generations,
            "mutation_rate": settings.ga_mutation_rate,
            "crossover_rate": settings.ga_crossover_rate
        }
    }

if __name__ == "__main__":
    # 設定テスト
    print("🔧 設定テスト開始")
    
    errors = validate_settings()
    if errors:
        print("❌ 設定エラー:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✅ 設定検証完了")
    
    info = get_settings_info()
    print(f"📊 設定情報: {info}")
    
    print("✅ 設定テスト完了")