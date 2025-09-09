# config/settings.py - 拡張された研究分野設定

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
        
        # 拡張された研究分野（20分野）
        research_fields: List[str] = [
            # テクノロジー・システム分野（12分野）
            "ai_machine_learning",            # 人工知能・機械学習
            "image_video_processing",         # 画像・映像処理
            "network_security",               # ネットワーク・セキュリティ
            "database_information_systems",   # データベース・情報システム
            "embedded_iot",                   # 組込み・IoT
            "education_linguistics",          # 教育・言語学
            "natural_science_mathematics",    # 自然科学・数理
            "medical_informatics",            # 医療情報・ヘルスケア
            "tourism_regional_systems",       # 観光情報・地域システム
            "business_information_systems",   # 経営情報・意思決定支援
            "audio_sound_processing",         # 音声・音響情報処理
            "system_operations_ethics",       # システム運用・情報倫理
            
            # クリエイティブ分野（4分野）
            "web_design_ui_ux",              # Webデザイン・UI/UX
            "design_visual_expression",      # デザイン・視覚表現
            "video_animation",               # 映像・アニメーション
            "computer_music_sound_art",      # コンピュータ音楽・サウンドアート
            
            # エンターテイメント分野（2分野）
            "game_development_esports",      # ゲーム開発・eスポーツ
            "vr_ar_media_art",              # VR/AR・メディアアート
            
            # 人文・社会・体育分野（2分野）
            "philosophy_humanities",         # 哲学・人文・環境行動学
            "sports_exercise_science"        # スポーツ・体育科学
        ]
        
        # 拡張された分野カテゴリ
        field_categories: Dict[str, List[str]] = {
            "テクノロジー・システム": [
                "ai_machine_learning", "image_video_processing", 
                "network_security", "database_information_systems", "embedded_iot",
                "education_linguistics", "natural_science_mathematics", 
                "medical_informatics", "tourism_regional_systems", 
                "business_information_systems", "audio_sound_processing", 
                "system_operations_ethics"
            ],
            "クリエイティブ": [
                "web_design_ui_ux", "design_visual_expression",
                "video_animation", "computer_music_sound_art"
            ],
            "エンターテイメント": [
                "game_development_esports", "vr_ar_media_art"
            ],
            "人文・社会・体育": [
                "philosophy_humanities", "sports_exercise_science"
            ]
        }
        
        # 詳細な研究分野情報
        research_fields_detail: Dict[str, Dict[str, Any]] = {
            # テクノロジー・システム分野
            "ai_machine_learning": {
                "name": "人工知能・機械学習",
                "description": "データ解析、機械学習、深層学習、自然言語処理など",
                "faculty": [
                    {"name": "伊藤雅彦", "specialties": ["情報可視化", "ユーザインタフェース", "データ工学"]},
                    {"name": "内山敏雄", "specialties": ["データ解析", "機械学習", "レコメンド", "テキストマイニング"]},
                    {"name": "小野亮太", "specialties": ["人工知能", "情報工学", "マルチエージェントシステム", "情報推薦"]},
                    {"name": "齋藤健司", "specialties": ["人工知能", "教育システム", "仮想環境"]},
                    {"name": "谷口文武", "specialties": ["機械学習", "パターン認識"]},
                    {"name": "辻準平", "specialties": ["社会シミュレーション", "マルチエージェントシステム", "IoT"]},
                    {"name": "山北貴典", "specialties": ["データベース技術"]}
                ],
                "faculty_count": 7,
                "keywords": ["AI", "機械学習", "データ分析", "深層学習"]
            },
            "image_video_processing": {
                "name": "画像・映像処理",
                "description": "コンピュータビジョン、画像認識、医用画像工学など",
                "faculty": [
                    {"name": "森圭佑", "specialties": ["情報計測", "音声・画像情報処理", "医用情報処理", "ゲームプログラミング"]},
                    {"name": "向田茂", "specialties": ["画像処理", "顔学", "認知心理学", "VR/AR", "3DCG"]},
                    {"name": "高井奈美", "specialties": ["コンピュータグラフィックス", "画像処理", "Webデザイン"]},
                    {"name": "藤原孝行", "specialties": ["コンピュータビジョン", "コンピュータグラフィックス"]},
                    {"name": "越野一博", "specialties": ["医用画像工学", "数理統計学", "人工知能画像解析処理"]},
                    {"name": "上杉正人", "specialties": ["医療情報システム開発", "医療言語処理", "画像処理"]}
                ],
                "faculty_count": 6,
                "keywords": ["画像処理", "コンピュータビジョン", "CG", "映像解析"]
            },
            "education_linguistics": {
                "name": "教育・言語学",
                "description": "日本語教育、多言語教育、教育システム、語学教育など",
                "faculty": [
                    {"name": "飯嶋美知子", "specialties": ["日本語教育学", "日中対照言語学"]},
                    {"name": "金銀珠", "specialties": ["日韓対照言語学", "日本語教育", "韓国語教育", "複言語教育"]},
                    {"name": "田中英夫", "specialties": ["国際経営論", "国際関係論", "中国語教育"]},
                    {"name": "齋藤一", "specialties": ["観光情報学", "教育工学"]},
                    {"name": "近澤潤", "specialties": ["発想法", "デザイン思考", "イノベーション教育"]}
                ],
                "faculty_count": 5,
                "keywords": ["日本語教育", "多言語", "教育システム", "語学"]
            },
            "natural_science_mathematics": {
                "name": "自然科学・数理",
                "description": "宇宙科学、地球科学、統計解析、数値計算、気象現象など",
                "faculty": [
                    {"name": "柿並義宏", "specialties": ["宇宙科学", "地球惑星科学", "大気科学", "動物行動学"]},
                    {"name": "甫喜本司", "specialties": ["データ解析法", "統計数理", "時間的・空間的な現象の予測方法"]},
                    {"name": "松井伸也", "specialties": ["非線形現象の解析", "流体現象", "気象現象", "反応拡散系"]},
                    {"name": "新井山亮", "specialties": ["社会情報工学", "光・波動電子工学", "数値解析"]},
                    {"name": "佐々木洋平", "specialties": ["地球流体力学", "惑星科学", "応用数学", "数値計算"]},
                    {"name": "湯村翼", "specialties": ["地球惑星科学", "ヒューマンコンピュータインタラクション"]}
                ],
                "faculty_count": 6,
                "keywords": ["宇宙科学", "地球科学", "統計解析", "数値計算"]
            },
            "medical_informatics": {
                "name": "医療情報・ヘルスケア",
                "description": "医用画像工学、医療情報システム、医療データ解析など",
                "faculty": [
                    {"name": "越野一博", "specialties": ["医用画像工学", "数理統計学", "人工知能画像解析処理"]},
                    {"name": "上杉正人", "specialties": ["医療情報システム開発", "医療言語処理", "画像処理"]}
                ],
                "faculty_count": 2,
                "keywords": ["医療IT", "医用画像", "ヘルスケア", "医療データ"]
            },
            "philosophy_humanities": {
                "name": "哲学・人文・環境行動学",
                "description": "哲学、倫理学、芸術学、環境行動学、地域コミュニティなど",
                "faculty": [
                    {"name": "三浦洋", "specialties": ["哲学", "倫理学", "芸術学"]},
                    {"name": "隼田尚彦", "specialties": ["環境行動学", "地域コミュニティ", "建築計画学"]}
                ],
                "faculty_count": 2,
                "keywords": ["哲学", "倫理学", "環境行動", "地域研究"]
            },
            "sports_exercise_science": {
                "name": "スポーツ・体育科学",
                "description": "スポーツバイオメカニクス、トレーニング科学、体育実践など",
                "faculty": [
                    {"name": "綿谷貴志", "specialties": ["スポーツバイオメカニクス", "トレーニング科学"]},
                    {"name": "織田哲", "specialties": ["体育"]}
                ],
                "faculty_count": 2,
                "keywords": ["スポーツ科学", "バイオメカニクス", "体育", "運動解析"]
            }
            # 他の分野の詳細も同様に定義...
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
                "research_intensity", "advisor_style", "team_work", 
                "workload", "theory_practice",
                "research_field_match", "skill_development", "lab_atmosphere",
                "flexibility", "publication_opportunity",
                "interdisciplinary", "communication_style", "innovation_risk"
            ]
            
            # 拡張された研究分野（20分野）
            self.research_fields = [
                # テクノロジー・システム分野
                "ai_machine_learning", "image_video_processing", 
                "network_security", "database_information_systems", "embedded_iot",
                "education_linguistics", "natural_science_mathematics", 
                "medical_informatics", "tourism_regional_systems", 
                "business_information_systems", "audio_sound_processing", 
                "system_operations_ethics",
                
                # クリエイティブ分野
                "web_design_ui_ux", "design_visual_expression",
                "video_animation", "computer_music_sound_art",
                
                # エンターテイメント分野
                "game_development_esports", "vr_ar_media_art",
                
                # 人文・社会・体育分野
                "philosophy_humanities", "sports_exercise_science"
            ]
            
            # 分野カテゴリ
            self.field_categories = {
                "テクノロジー・システム": [
                    "ai_machine_learning", "image_video_processing", 
                    "network_security", "database_information_systems", "embedded_iot",
                    "education_linguistics", "natural_science_mathematics", 
                    "medical_informatics", "tourism_regional_systems", 
                    "business_information_systems", "audio_sound_processing", 
                    "system_operations_ethics"
                ],
                "クリエイティブ": [
                    "web_design_ui_ux", "design_visual_expression",
                    "video_animation", "computer_music_sound_art"
                ],
                "エンターテイメント": [
                    "game_development_esports", "vr_ar_media_art"
                ],
                "人文・社会・体育": [
                    "philosophy_humanities", "sports_exercise_science"
                ]
            }
        
        @property
        def core_features(self):
            return self.evaluation_criteria[:5]

# シングルトンパターンでインスタンス作成
settings = Settings()

# 研究室データテンプレート（拡張版）
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
    "research_fields": [],  # 複数の分野に対応
    "metadata": {
        "faculty_count": 1,
        "student_count": 0,
        "recent_publications": 0,
        "funding_level": "中",
        "equipment_rating": 5
    }
}