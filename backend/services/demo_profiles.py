# backend/services/demo_profiles.py
"""
デモプロファイル生成サービス

多様な学生タイプのサンプルプロファイルを提供
"""

from typing import Dict, List, Any


class DemoProfileService:
    """デモプロファイル生成サービス"""
    
    # デモプロファイルデータ定義
    DEMO_PROFILES = {
        "研究集中型（AI研究者志向）": {
            "description": "研究に集中し、論文発表を重視するAI研究者タイプ",
            "characteristics": [
                "研究強度と論文発表機会を最重視",
                "理論と実践のバランス型",
                "AI・機械学習分野に特化"
            ],
            "profile": {
                # 基本評価項目
                "research_intensity": 9,
                "advisor_style": 6,
                "team_work": 7,
                "workload": 8,
                "theory_practice": 6,
                "research_field_match": 9,
                "skill_development": 8,
                "lab_atmosphere": 7,
                "flexibility": 6,
                "publication_opportunity": 10,
                "interdisciplinary": 7,
                "communication_style": 6,
                
                # 優先度
                "research_intensity_priority": 10,
                "advisor_style_priority": 7,
                "team_work_priority": 8,
                "workload_priority": 7,
                "theory_practice_priority": 6,
                "research_field_match_priority": 10,
                "skill_development_priority": 7,
                "lab_atmosphere_priority": 6,
                "flexibility_priority": 5,
                "publication_opportunity_priority": 10,
                "interdisciplinary_priority": 6,
                "communication_style_priority": 5,
                
                # 研究分野興味
                "field_interests": [
                    {"field_id": "ai_ml", "interest_level": 10},
                    {"field_id": "image_processing", "interest_level": 8},
                    {"field_id": "data_analysis", "interest_level": 7}
                ]
            }
        },
        
        "クリエイティブ型（デザイナー志向）": {
            "description": "柔軟性とチームワークを重視するデザイナータイプ",
            "characteristics": [
                "柔軟性とチームワークを最重視",
                "実践重視の研究スタイル",
                "デザイン・視覚表現分野に興味"
            ],
            "profile": {
                "research_intensity": 6,
                "advisor_style": 8,
                "team_work": 9,
                "workload": 6,
                "theory_practice": 8,
                "research_field_match": 8,
                "skill_development": 9,
                "lab_atmosphere": 9,
                "flexibility": 9,
                "publication_opportunity": 5,
                "interdisciplinary": 8,
                "communication_style": 9,
                
                "research_intensity_priority": 6,
                "advisor_style_priority": 7,
                "team_work_priority": 9,
                "workload_priority": 5,
                "theory_practice_priority": 7,
                "research_field_match_priority": 9,
                "skill_development_priority": 9,
                "lab_atmosphere_priority": 8,
                "flexibility_priority": 9,
                "publication_opportunity_priority": 4,
                "interdisciplinary_priority": 7,
                "communication_style_priority": 8,
                
                "field_interests": [
                    {"field_id": "web_design", "interest_level": 10},
                    {"field_id": "design_visual", "interest_level": 9},
                    {"field_id": "video_animation", "interest_level": 7}
                ]
            }
        },
        
        "バランス型（オールラウンダー）": {
            "description": "すべての項目をバランスよく重視するオールラウンダータイプ",
            "characteristics": [
                "すべての項目を均等に評価",
                "幅広い分野に興味",
                "適応力の高いタイプ"
            ],
            "profile": {
                "research_intensity": 6,
                "advisor_style": 6,
                "team_work": 6,
                "workload": 6,
                "theory_practice": 6,
                "research_field_match": 6,
                "skill_development": 7,
                "lab_atmosphere": 6,
                "flexibility": 7,
                "publication_opportunity": 6,
                "interdisciplinary": 6,
                "communication_style": 6,
                
                "research_intensity_priority": 6,
                "advisor_style_priority": 6,
                "team_work_priority": 6,
                "workload_priority": 6,
                "theory_practice_priority": 6,
                "research_field_match_priority": 6,
                "skill_development_priority": 7,
                "lab_atmosphere_priority": 6,
                "flexibility_priority": 7,
                "publication_opportunity_priority": 6,
                "interdisciplinary_priority": 6,
                "communication_style_priority": 6,
                
                "field_interests": [
                    {"field_id": "ai_ml", "interest_level": 7},
                    {"field_id": "web_design", "interest_level": 6},
                    {"field_id": "database_systems", "interest_level": 6}
                ]
            }
        },
        
        "実践志向型（エンジニア志向）": {
            "description": "実践とスキル開発を重視するエンジニアタイプ",
            "characteristics": [
                "実践重視（理論・実践バランス9）",
                "スキル開発を最重視",
                "IoT・セキュリティ分野に興味"
            ],
            "profile": {
                "research_intensity": 7,
                "advisor_style": 7,
                "team_work": 8,
                "workload": 7,
                "theory_practice": 9,
                "research_field_match": 7,
                "skill_development": 9,
                "lab_atmosphere": 7,
                "flexibility": 8,
                "publication_opportunity": 6,
                "interdisciplinary": 7,
                "communication_style": 7,
                
                "research_intensity_priority": 7,
                "advisor_style_priority": 6,
                "team_work_priority": 8,
                "workload_priority": 7,
                "theory_practice_priority": 9,
                "research_field_match_priority": 7,
                "skill_development_priority": 10,
                "lab_atmosphere_priority": 6,
                "flexibility_priority": 7,
                "publication_opportunity_priority": 5,
                "interdisciplinary_priority": 6,
                "communication_style_priority": 7,
                
                "field_interests": [
                    {"field_id": "embedded_iot", "interest_level": 9},
                    {"field_id": "network_security", "interest_level": 8},
                    {"field_id": "database_systems", "interest_level": 7}
                ]
            }
        },
        
        "理論志向型（研究者志向）": {
            "description": "理論研究と論文発表を最重視する研究者タイプ",
            "characteristics": [
                "研究強度と論文発表機会が最高値",
                "理論重視（理論・実践バランス3）",
                "個人研究を好む"
            ],
            "profile": {
                "research_intensity": 10,
                "advisor_style": 5,
                "team_work": 5,
                "workload": 9,
                "theory_practice": 3,
                "research_field_match": 9,
                "skill_development": 6,
                "lab_atmosphere": 5,
                "flexibility": 5,
                "publication_opportunity": 10,
                "interdisciplinary": 8,
                "communication_style": 5,
                
                "research_intensity_priority": 10,
                "advisor_style_priority": 6,
                "team_work_priority": 5,
                "workload_priority": 8,
                "theory_practice_priority": 8,
                "research_field_match_priority": 9,
                "skill_development_priority": 6,
                "lab_atmosphere_priority": 5,
                "flexibility_priority": 4,
                "publication_opportunity_priority": 10,
                "interdisciplinary_priority": 7,
                "communication_style_priority": 4,
                
                "field_interests": [
                    {"field_id": "ai_ml", "interest_level": 9},
                    {"field_id": "natural_science_math", "interest_level": 10},
                    {"field_id": "image_processing", "interest_level": 7}
                ]
            }
        },
        
        "チームワーク重視型（協調型）": {
            "description": "チームワークとコミュニケーションを最重視する協調タイプ",
            "characteristics": [
                "チームワークとコミュニケーションが最高値",
                "研究室の雰囲気を重視",
                "学際的な研究に興味"
            ],
            "profile": {
                "research_intensity": 7,
                "advisor_style": 7,
                "team_work": 10,
                "workload": 7,
                "theory_practice": 7,
                "research_field_match": 7,
                "skill_development": 8,
                "lab_atmosphere": 9,
                "flexibility": 7,
                "publication_opportunity": 7,
                "interdisciplinary": 8,
                "communication_style": 10,
                
                "research_intensity_priority": 7,
                "advisor_style_priority": 7,
                "team_work_priority": 10,
                "workload_priority": 6,
                "theory_practice_priority": 6,
                "research_field_match_priority": 7,
                "skill_development_priority": 7,
                "lab_atmosphere_priority": 9,
                "flexibility_priority": 6,
                "publication_opportunity_priority": 7,
                "interdisciplinary_priority": 8,
                "communication_style_priority": 10,
                
                "field_interests": [
                    {"field_id": "game_esports", "interest_level": 8},
                    {"field_id": "web_design", "interest_level": 7},
                    {"field_id": "business_decision", "interest_level": 7}
                ]
            }
        },
        
        "独立研究型（自由探求型）": {
            "description": "柔軟性と自由な指導スタイルを重視する独立研究タイプ",
            "characteristics": [
                "柔軟性と自由指導を最重視",
                "個人研究を好む（チームワーク4）",
                "学際的研究に興味"
            ],
            "profile": {
                "research_intensity": 8,
                "advisor_style": 9,
                "team_work": 4,
                "workload": 7,
                "theory_practice": 6,
                "research_field_match": 8,
                "skill_development": 7,
                "lab_atmosphere": 6,
                "flexibility": 10,
                "publication_opportunity": 8,
                "interdisciplinary": 9,
                "communication_style": 5,
                
                "research_intensity_priority": 8,
                "advisor_style_priority": 10,
                "team_work_priority": 3,
                "workload_priority": 6,
                "theory_practice_priority": 6,
                "research_field_match_priority": 8,
                "skill_development_priority": 7,
                "lab_atmosphere_priority": 5,
                "flexibility_priority": 10,
                "publication_opportunity_priority": 8,
                "interdisciplinary_priority": 8,
                "communication_style_priority": 4,
                
                "field_interests": [
                    {"field_id": "ai_ml", "interest_level": 8},
                    {"field_id": "philosophy_humanities", "interest_level": 7},
                    {"field_id": "audio_processing", "interest_level": 8}
                ]
            }
        },
        
        "ゲーム開発型（ゲームクリエイター志向）": {
            "description": "ゲーム開発に特化したクリエイタータイプ",
            "characteristics": [
                "分野重視度が最高値",
                "チームワークとスキル開発重視",
                "ゲーム・VR/AR分野に特化"
            ],
            "profile": {
                "research_intensity": 7,
                "advisor_style": 7,
                "team_work": 9,
                "workload": 8,
                "theory_practice": 8,
                "research_field_match": 9,
                "skill_development": 9,
                "lab_atmosphere": 8,
                "flexibility": 7,
                "publication_opportunity": 5,
                "interdisciplinary": 8,
                "communication_style": 8,
                
                "research_intensity_priority": 7,
                "advisor_style_priority": 6,
                "team_work_priority": 9,
                "workload_priority": 7,
                "theory_practice_priority": 8,
                "research_field_match_priority": 10,
                "skill_development_priority": 9,
                "lab_atmosphere_priority": 7,
                "flexibility_priority": 7,
                "publication_opportunity_priority": 4,
                "interdisciplinary_priority": 7,
                "communication_style_priority": 7,
                
                "field_interests": [
                    {"field_id": "game_esports", "interest_level": 10},
                    {"field_id": "vr_ar_media", "interest_level": 9},
                    {"field_id": "ai_ml", "interest_level": 6}
                ]
            }
        },
        
        "教育・言語学型（教育者志向）": {
            "description": "教育と学際性を重視する教育者タイプ",
            "characteristics": [
                "学際性とコミュニケーション重視",
                "教育・言語学分野に特化",
                "バランスの取れた研究スタイル"
            ],
            "profile": {
                "research_intensity": 7,
                "advisor_style": 6,
                "team_work": 8,
                "workload": 6,
                "theory_practice": 6,
                "research_field_match": 8,
                "skill_development": 7,
                "lab_atmosphere": 8,
                "flexibility": 7,
                "publication_opportunity": 7,
                "interdisciplinary": 9,
                "communication_style": 9,
                
                "research_intensity_priority": 7,
                "advisor_style_priority": 6,
                "team_work_priority": 8,
                "workload_priority": 5,
                "theory_practice_priority": 6,
                "research_field_match_priority": 9,
                "skill_development_priority": 7,
                "lab_atmosphere_priority": 7,
                "flexibility_priority": 7,
                "publication_opportunity_priority": 7,
                "interdisciplinary_priority": 9,
                "communication_style_priority": 9,
                
                "field_interests": [
                    {"field_id": "education_linguistics", "interest_level": 9},
                    {"field_id": "tourism_regional", "interest_level": 7},
                    {"field_id": "philosophy_humanities", "interest_level": 6}
                ]
            }
        },
        
        "スポーツ科学型（スポーツ研究者志向）": {
            "description": "スポーツ科学と実践研究を重視するタイプ",
            "characteristics": [
                "分野重視度が最高値",
                "実践重視の研究スタイル",
                "スポーツ科学・医療情報に興味"
            ],
            "profile": {
                "research_intensity": 7,
                "advisor_style": 6,
                "team_work": 8,
                "workload": 7,
                "theory_practice": 8,
                "research_field_match": 9,
                "skill_development": 8,
                "lab_atmosphere": 8,
                "flexibility": 7,
                "publication_opportunity": 7,
                "interdisciplinary": 8,
                "communication_style": 8,
                
                "research_intensity_priority": 7,
                "advisor_style_priority": 6,
                "team_work_priority": 8,
                "workload_priority": 6,
                "theory_practice_priority": 8,
                "research_field_match_priority": 10,
                "skill_development_priority": 8,
                "lab_atmosphere_priority": 7,
                "flexibility_priority": 7,
                "publication_opportunity_priority": 7,
                "interdisciplinary_priority": 8,
                "communication_style_priority": 7,
                
                "field_interests": [
                    {"field_id": "sports_science", "interest_level": 10},
                    {"field_id": "natural_science_math", "interest_level": 7},
                    {"field_id": "medical_healthcare", "interest_level": 6}
                ]
            }
        }
    }
    
    @classmethod
    def get_all_profiles(cls) -> Dict[str, Any]:
        """
        すべてのデモプロファイルを取得
        
        Returns:
            Dict[str, Any]: デモプロファイル一覧
        """
        return {
            profile_name: {
                "description": profile_data["description"],
                "characteristics": profile_data["characteristics"]
            }
            for profile_name, profile_data in cls.DEMO_PROFILES.items()
        }
    
    @classmethod
    def get_profile(cls, profile_name: str) -> Dict[str, Any]:
        """
        指定されたデモプロファイルを取得
        
        Args:
            profile_name: プロファイル名
            
        Returns:
            Dict[str, Any]: プロファイルデータ
            
        Raises:
            KeyError: 指定されたプロファイルが存在しない場合
        """
        if profile_name not in cls.DEMO_PROFILES:
            raise KeyError(f"プロファイル '{profile_name}' が見つかりません")
        
        return cls.DEMO_PROFILES[profile_name]["profile"]
    
    @classmethod
    def get_profile_names(cls) -> List[str]:
        """
        デモプロファイル名の一覧を取得
        
        Returns:
            List[str]: プロファイル名のリスト
        """
        return list(cls.DEMO_PROFILES.keys())
    
    @classmethod
    def get_profile_with_metadata(cls, profile_name: str) -> Dict[str, Any]:
        """
        メタデータを含むプロファイルを取得
        
        Args:
            profile_name: プロファイル名
            
        Returns:
            Dict[str, Any]: メタデータを含むプロファイルデータ
        """
        if profile_name not in cls.DEMO_PROFILES:
            raise KeyError(f"プロファイル '{profile_name}' が見つかりません")
        
        profile_data = cls.DEMO_PROFILES[profile_name]
        
        return {
            "name": profile_name,
            "description": profile_data["description"],
            "characteristics": profile_data["characteristics"],
            "profile": profile_data["profile"]
        }


# 使用例
if __name__ == "__main__":
    # デモプロファイル一覧を取得
    print("📋 デモプロファイル一覧:")
    profiles = DemoProfileService.get_all_profiles()
    for name, info in profiles.items():
        print(f"\n{name}")
        print(f"  説明: {info['description']}")
        print(f"  特徴:")
        for char in info['characteristics']:
            print(f"    - {char}")
    
    # 特定のプロファイルを取得
    print("\n\n🎯 研究集中型プロファイル取得:")
    profile = DemoProfileService.get_profile("研究集中型（AI研究者志向）")
    print(f"  研究強度: {profile['research_intensity']}")
    print(f"  論文発表機会: {profile['publication_opportunity']}")
    print(f"  分野興味: {profile['field_interests']}")