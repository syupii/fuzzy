# config/settings.py - 27分野対応設定ファイル

import os
from typing import Dict, List

class Settings:
    """27分野対応システム設定"""
    
    def __init__(self):
        # 基本ディレクトリ設定
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, 'data')
        
        # 評価基準（13項目）
        self.evaluation_criteria = [
            "research_intensity", "advisor_style", "team_work", "workload",
            "theory_practice", "research_field_match", "skill_development",
            "lab_atmosphere", "flexibility", "publication_opportunity",
            "interdisciplinary", "communication_style", "innovation_risk"
        ]
        
        # 27研究分野定義
        self.research_fields = {
            # テクノロジー・システム分野（12分野）
            "ai_machine_learning": {
                "name": "人工知能・機械学習",
                "category": "テクノロジー・システム",
                "faculty": ["伊藤正彦", "谷口文威"],
                "difficulty": "intermediate",
                "tech_focus": 9, "creativity_focus": 4, "theory_practice": 6
            },
            "education_vr_systems": {
                "name": "教育システム・仮想環境",
                "category": "テクノロジー・システム",
                "faculty": ["齋藤健司"],
                "difficulty": "intermediate",
                "tech_focus": 8, "creativity_focus": 6, "theory_practice": 7
            },
            "social_simulation": {
                "name": "社会シミュレーション・マルチエージェント",
                "category": "テクノロジー・システム",
                "faculty": ["辻順平"],
                "difficulty": "advanced",
                "tech_focus": 9, "creativity_focus": 5, "theory_practice": 5
            },
            "database_technology": {
                "name": "データベース技術",
                "category": "テクノロジー・システム",
                "faculty": ["山北隆典"],
                "difficulty": "intermediate",
                "tech_focus": 8, "creativity_focus": 3, "theory_practice": 7
            },
            "image_computer_vision": {
                "name": "画像処理・コンピュータビジョン",
                "category": "テクノロジー・システム",
                "faculty": ["向田茂", "藤原孝幸"],
                "difficulty": "advanced",
                "tech_focus": 9, "creativity_focus": 6, "theory_practice": 6
            },
            "medical_audio_processing": {
                "name": "医用情報処理・音声処理",
                "category": "テクノロジー・システム",
                "faculty": ["守啓祐"],
                "difficulty": "advanced",
                "tech_focus": 9, "creativity_focus": 4, "theory_practice": 8
            },
            "network_security": {
                "name": "ネットワーク・セキュリティ",
                "category": "テクノロジー・システム",
                "faculty": ["佐々木洋平"],
                "difficulty": "advanced",
                "tech_focus": 9, "creativity_focus": 3, "theory_practice": 7
            },
            "data_analysis_statistics": {
                "name": "データ解析・統計数理",
                "category": "テクノロジー・システム",
                "faculty": ["甫喜本司"],
                "difficulty": "intermediate",
                "tech_focus": 8, "creativity_focus": 4, "theory_practice": 5
            },
            "social_info_engineering": {
                "name": "社会情報工学・数値計算",
                "category": "テクノロジー・システム",
                "faculty": ["新井山亮"],
                "difficulty": "advanced",
                "tech_focus": 9, "creativity_focus": 3, "theory_practice": 4
            },
            "ubiquitous_iot_hci": {
                "name": "ユビキタス・IoT・HCI",
                "category": "テクノロジー・システム",
                "faculty": ["湯村翼"],
                "difficulty": "intermediate",
                "tech_focus": 8, "creativity_focus": 6, "theory_practice": 8
            },
            "game_programming": {
                "name": "ゲームプログラミング",
                "category": "テクノロジー・システム",
                "faculty": ["森川悟"],
                "difficulty": "intermediate",
                "tech_focus": 8, "creativity_focus": 8, "theory_practice": 8
            },
            "computer_audio_systems": {
                "name": "コンピュータシステム・音響処理",
                "category": "テクノロジー・システム",
                "faculty": ["広奥暢"],
                "difficulty": "intermediate",
                "tech_focus": 8, "creativity_focus": 5, "theory_practice": 7
            },
            
            # クリエイティブ・デザイン分野（5分野）
            "web_design_branding": {
                "name": "Webデザイン・ブランディング",
                "category": "クリエイティブ・デザイン",
                "faculty": ["杉澤愛美"],
                "difficulty": "beginner",
                "tech_focus": 6, "creativity_focus": 9, "theory_practice": 8
            },
            "ux_ui_design_thinking": {
                "name": "UX/UI・デザイン思考",
                "category": "クリエイティブ・デザイン",
                "faculty": ["安田光孝", "近澤潤"],
                "difficulty": "intermediate",
                "tech_focus": 7, "creativity_focus": 9, "theory_practice": 9
            },
            "visual_design_kansei": {
                "name": "視覚デザイン・感性工学",
                "category": "クリエイティブ・デザイン",
                "faculty": ["坂本牧葉"],
                "difficulty": "intermediate",
                "tech_focus": 6, "creativity_focus": 9, "theory_practice": 7
            },
            "illustration_art_management": {
                "name": "イラストレーション・アートマネジメント",
                "category": "クリエイティブ・デザイン",
                "faculty": ["伊藤マーティ"],
                "difficulty": "beginner",
                "tech_focus": 4, "creativity_focus": 10, "theory_practice": 8
            },
            "video_animation": {
                "name": "映像制作・アニメーション",
                "category": "クリエイティブ・デザイン",
                "faculty": ["大島慶太郎", "島田英二"],
                "difficulty": "intermediate",
                "tech_focus": 6, "creativity_focus": 10, "theory_practice": 8
            },
            
            # メディア・エンターテイメント分野（3分野）
            "computer_music_sound": {
                "name": "コンピュータ音楽・サウンドアート",
                "category": "メディア・エンターテイメント",
                "faculty": ["平山晴花"],
                "difficulty": "advanced",
                "tech_focus": 7, "creativity_focus": 10, "theory_practice": 6
            },
            "esports_metaverse": {
                "name": "eスポーツ・メタバース",
                "category": "メディア・エンターテイメント",
                "faculty": ["河原大"],
                "difficulty": "intermediate",
                "tech_focus": 7, "creativity_focus": 8, "theory_practice": 8
            },
            "vr_ar_media_architecture": {
                "name": "VR/AR・メディアアート・建築",
                "category": "メディア・エンターテイメント",
                "faculty": ["向田茂", "隼田尚彦"],
                "difficulty": "advanced",
                "tech_focus": 8, "creativity_focus": 9, "theory_practice": 7
            },
            
            # 人文・社会・自然科学分野（7分野）
            "japanese_linguistics": {
                "name": "日本語教育・言語学",
                "category": "人文・社会・自然科学",
                "faculty": ["飯嶋美知子", "金銀珠"],
                "difficulty": "intermediate",
                "tech_focus": 3, "creativity_focus": 6, "theory_practice": 6
            },
            "tourism_education": {
                "name": "観光情報学・教育工学",
                "category": "人文・社会・自然科学",
                "faculty": ["斎藤一"],
                "difficulty": "intermediate",
                "tech_focus": 6, "creativity_focus": 5, "theory_practice": 7
            },
            "international_business": {
                "name": "国際経営・中国語教育",
                "category": "人文・社会・自然科学",
                "faculty": ["田中英夫"],
                "difficulty": "intermediate",
                "tech_focus": 2, "creativity_focus": 5, "theory_practice": 6
            },
            "space_earth_science": {
                "name": "宇宙・地球惑星科学",
                "category": "人文・社会・自然科学",
                "faculty": ["柿並義宏"],
                "difficulty": "advanced",
                "tech_focus": 7, "creativity_focus": 4, "theory_practice": 4
            },
            "mathematical_physics": {
                "name": "数理物理・非線形現象",
                "category": "人文・社会・自然科学",
                "faculty": ["松井伸也"],
                "difficulty": "advanced",
                "tech_focus": 8, "creativity_focus": 3, "theory_practice": 3
            },
            "philosophy_ethics_arts": {
                "name": "哲学・倫理学・芸術学",
                "category": "人文・社会・自然科学",
                "faculty": ["三浦洋"],
                "difficulty": "advanced",
                "tech_focus": 2, "creativity_focus": 8, "theory_practice": 4
            },
            "sports_biomechanics": {
                "name": "スポーツ科学・バイオメカニクス",
                "category": "人文・社会・自然科学",
                "faculty": ["織田哲", "綿谷貴志"],
                "difficulty": "intermediate",
                "tech_focus": 6, "creativity_focus": 4, "theory_practice": 9
            }
        }
        
        # 分野カテゴリ
        self.field_categories = {
            "テクノロジー・システム": [
                "ai_machine_learning", "education_vr_systems", "social_simulation",
                "database_technology", "image_computer_vision", "medical_audio_processing",
                "network_security", "data_analysis_statistics", "social_info_engineering",
                "ubiquitous_iot_hci", "game_programming", "computer_audio_systems"
            ],
            "クリエイティブ・デザイン": [
                "web_design_branding", "ux_ui_design_thinking", "visual_design_kansei",
                "illustration_art_management", "video_animation"
            ],
            "メディア・エンターテイメント": [
                "computer_music_sound", "esports_metaverse", "vr_ar_media_architecture"
            ],
            "人文・社会・自然科学": [
                "japanese_linguistics", "tourism_education", "international_business",
                "space_earth_science", "mathematical_physics", "philosophy_ethics_arts",
                "sports_biomechanics"
            ]
        }
        
        # 遺伝的アルゴリズム設定
        self.ga_population_size = 30
        self.ga_generations = 50
        self.ga_mutation_rate = 0.1
        self.ga_crossover_rate = 0.8
        
        # ファジィ決定木設定
        self.max_tree_depth = 8
        self.min_samples_split = 3

# グローバル設定インスタンス
settings = Settings()