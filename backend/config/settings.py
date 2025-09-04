"""
設定管理 - config/settings.py (13項目拡張版)
遺伝的アルゴリズムを用いたファジィ決定木システムの設定
"""

import os
from typing import List, Dict, Any


class Settings:
    """システム設定クラス（13項目拡張版）"""
    
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
        
        # ✨ 13項目評価基準設定
        self.all_features = [
            # 基本項目（5項目）
            "research_intensity",      # 研究強度
            "advisor_style",          # 指導スタイル
            "team_work",              # チームワーク
            "workload",               # ワークロード
            "theory_practice",        # 理論・実践バランス
            
            # 拡張項目（5項目）
            "research_field_match",   # 研究分野適合性
            "skill_development",      # スキル開発
            "lab_atmosphere",         # 研究室雰囲気
            "flexibility",            # 柔軟性
            "publication_opportunity", # 論文発表機会
            
            # 特殊項目（3項目）
            "interdisciplinary",      # 学際性
            "communication_style",    # コミュニケーション
            "innovation_risk"         # 革新性・リスク許容度
        ]
        
        # 後方互換性のため
        self.core_features = self.all_features[:5]
        self.extended_features = self.all_features[5:]
        
        # ✨ 11研究分野設定
        self.research_fields = [
            # テクノロジー・システム分野（5分野）
            "人工知能・機械学習",
            "画像・映像処理", 
            "コンピュータネットワーク・セキュリティ",
            "データベース・情報システム",
            "組込み・IoT",
            
            # クリエイティブ分野（4分野）
            "Webデザイン・UI/UX",
            "デザイン・視覚表現",
            "映像・アニメーション", 
            "コンピュータ音楽・サウンドアート",
            
            # エンターテイメント分野（2分野）
            "ゲーム開発・eスポーツ",
            "VR/AR・メディアアート"
        ]
        
        # 分野カテゴリマッピング
        self.field_categories = {
            "テクノロジー・システム": [
                "人工知能・機械学習",
                "画像・映像処理", 
                "コンピュータネットワーク・セキュリティ",
                "データベース・情報システム",
                "組込み・IoT"
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
            ]
        }
        
        # 教員マッピング（分野 → 教員リスト）
        self.field_faculty_mapping = {
            "人工知能・機械学習": [
                "伊藤雅彦", "内山敏雄", "小野亮太", "齋藤健司", 
                "谷口文武", "辻準平", "山北貴典"
            ],
            "画像・映像処理": [
                "森圭佑", "向田茂", "高井奈美", "藤原孝行", 
                "越野一博", "上杉正人"
            ],
            "コンピュータネットワーク・セキュリティ": [
                "尾崎宏和", "中島潤", "佐々木洋平"
            ],
            "データベース・情報システム": [
                "山北貴典", "坂田圭司", "向原強"
            ],
            "組込み・IoT": [
                "田鎖次郎", "湯村翼"
            ],
            "Webデザイン・UI/UX": [
                "杉沢愛美", "坂本牧葉", "高井奈美", "安田光孝"
            ],
            "デザイン・視覚表現": [
                "坂本牧葉", "大嶋宏一", "Marty M. ITO", "安田光孝"
            ],
            "映像・アニメーション": [
                "大嶋宏一", "島田映二"
            ],
            "コンピュータ音楽・サウンドアート": [
                "平山遙香", "廣奥透"
            ],
            "ゲーム開発・eスポーツ": [
                "森川悟", "川原勝"
            ],
            "VR/AR・メディアアート": [
                "向田茂", "波田彰"
            ]
        }
        
        # 特徴量重み設定（研究に基づく）
        self.feature_weights = {
            # 最重要項目（研究で証明済み）
            "research_field_match": 0.15,     # 研究分野適合性が最重要
            "advisor_style": 0.12,            # 指導スタイルが重要
            "research_intensity": 0.10,       # 研究強度
            
            # 重要項目
            "lab_atmosphere": 0.09,           # 研究室雰囲気
            "publication_opportunity": 0.09,  # 論文機会
            "skill_development": 0.08,        # スキル開発
            
            # 標準項目
            "team_work": 0.07,               # チームワーク
            "flexibility": 0.07,             # 柔軟性
            "workload": 0.06,                # ワークロード
            "theory_practice": 0.06,         # 理論・実践
            
            # 補助項目
            "communication_style": 0.05,     # コミュニケーション
            "interdisciplinary": 0.04,       # 学際性
            "innovation_risk": 0.04          # 革新性・リスク
        }
        
        # 環境変数からの設定読み込み
        self._load_from_env()
    
    def _load_from_env(self):
        """環境変数から設定を読み込み"""
        
        # ポート設定
        if 'LAB_MATCHING_PORT' in os.environ:
            self.port = int(os.environ['LAB_MATCHING_PORT'])
        
        # デバッグモード
        if 'LAB_MATCHING_DEBUG' in os.environ:
            self.debug = os.environ['LAB_MATCHING_DEBUG'].lower() in ('true', '1', 'yes')
        
        # GA設定
        if 'GA_POPULATION_SIZE' in os.environ:
            self.ga_population_size = int(os.environ['GA_POPULATION_SIZE'])
        
        if 'GA_GENERATIONS' in os.environ:
            self.ga_generations = int(os.environ['GA_GENERATIONS'])
    
    def get_feature_weight(self, feature: str) -> float:
        """特徴量の重みを取得"""
        return self.feature_weights.get(feature, 0.05)  # デフォルト重み
    
    def get_field_category(self, field: str) -> str:
        """分野のカテゴリを取得"""
        for category, fields in self.field_categories.items():
            if field in fields:
                return category
        return "その他"
    
    def get_category_fields(self, category: str) -> List[str]:
        """カテゴリの分野リストを取得"""
        return self.field_categories.get(category, [])
    
    def validate_student_profile(self, profile: Dict[str, Any]) -> bool:
        """学生プロフィールの妥当性チェック"""
        
        # 必須項目チェック
        for feature in self.all_features:
            if feature not in profile:
                return False
            value = profile[feature]
            if not isinstance(value, (int, float)) or not (1 <= value <= 10):
                return False
        
        return True
    
    def validate_field_interests(self, interests: Dict[str, Any]) -> bool:
        """研究分野興味度の妥当性チェック"""
        
        for field in self.research_fields:
            if field not in interests:
                return False
            value = interests[field]
            if not isinstance(value, (int, float)) or not (1 <= value <= 10):
                return False
        
        return True
    
    def get_evaluation_summary(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """評価プロフィールのサマリー生成"""
        
        if not self.validate_student_profile(profile):
            return {"error": "Invalid profile"}
        
        feature_values = [profile[f] for f in self.all_features]
        
        return {
            "total_features": len(self.all_features),
            "average_importance": sum(feature_values) / len(feature_values),
            "max_importance": max(feature_values),
            "min_importance": min(feature_values),
            "high_priority_features": [
                f for f in self.all_features 
                if profile[f] >= 8.0
            ],
            "low_priority_features": [
                f for f in self.all_features 
                if profile[f] <= 4.0
            ],
            "feature_distribution": {
                "high (8-10)": len([v for v in feature_values if v >= 8]),
                "medium (5-7)": len([v for v in feature_values if 5 <= v < 8]),
                "low (1-4)": len([v for v in feature_values if v < 5])
            }
        }
    
    def get_field_summary(self, interests: Dict[str, Any]) -> Dict[str, Any]:
        """研究分野興味度のサマリー生成"""
        
        if not self.validate_field_interests(interests):
            return {"error": "Invalid field interests"}
        
        # カテゴリ別集計
        category_averages = {}
        for category, fields in self.field_categories.items():
            category_interests = [interests[field] for field in fields if field in interests]
            if category_interests:
                category_averages[category] = sum(category_interests) / len(category_interests)
        
        # 全体統計
        all_interests = list(interests.values())
        
        return {
            "total_fields": len(self.research_fields),
            "average_interest": sum(all_interests) / len(all_interests),
            "max_interest": max(all_interests),
            "min_interest": min(all_interests),
            "primary_category": max(category_averages.items(), key=lambda x: x[1])[0] if category_averages else "なし",
            "category_averages": category_averages,
            "top_interests": sorted(interests.items(), key=lambda x: x[1], reverse=True)[:3],
            "interest_distribution": {
                "strong (8-10)": len([v for v in all_interests if v >= 8]),
                "moderate (5-7)": len([v for v in all_interests if 5 <= v < 8]),
                "weak (1-4)": len([v for v in all_interests if v < 5])
            },
            "diversity_score": len([v for v in all_interests if v >= 6]) / len(all_interests)
        }
    
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
            
            # メンバーシップ関数設定
            'membership_types': self.membership_types,
            'default_membership_type': self.default_membership_type,
            
            # 最適化設定
            'fitness_weights': self.fitness_weights,
            'convergence_threshold': self.convergence_threshold,
            'max_stagnant_generations': self.max_stagnant_generations,
            
            # データ設定
            'data_dir': self.data_dir,
            'model_dir': self.model_dir,
            'temp_dir': self.temp_dir,
            
            # マッチング設定
            'max_labs_to_evaluate': self.max_labs_to_evaluate,
            'min_compatibility_threshold': self.min_compatibility_threshold,
            
            # 特徴量設定
            'all_features': self.all_features,
            'core_features': self.core_features,
            'extended_features': self.extended_features,
            'feature_weights': self.feature_weights,
            
            # 研究分野設定
            'research_fields': self.research_fields,
            'field_categories': self.field_categories,
            'field_statistics': self.field_statistics
        }
    
    def export_config(self, filepath: str = None) -> str:
        """設定をJSONファイルにエクスポート"""
        
        import json
        from datetime import datetime
        
        config_data = self.to_dict()
        config_data['export_timestamp'] = datetime.now().isoformat()
        config_data['version'] = '3.0.0'
        
        if filepath is None:
            filepath = os.path.join(self.data_dir, 'system_config.json')
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def load_config(self, filepath: str) -> bool:
        """JSONファイルから設定を読み込み"""
        
        try:
            import json
            
            with open(filepath, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 基本設定の更新
            for key in ['ga_population_size', 'ga_generations', 'max_tree_depth']:
                if key in config_data:
                    setattr(self, key, config_data[key])
            
            # 重み設定の更新
            if 'feature_weights' in config_data:
                self.feature_weights.update(config_data['feature_weights'])
            
            return True
            
        except Exception as e:
            print(f"設定読み込みエラー: {e}")
            return False
    
    def __repr__(self):
        return (f"Settings(features={len(self.all_features)}, "
                f"fields={len(self.research_fields)}, "
                f"population={self.ga_population_size}, "
                f"generations={self.ga_generations})")


# グローバル設定インスタンス
settings = Settings()

# 設定検証
if __name__ == "__main__":
    print("🔧 システム設定検証")
    print(f"📊 評価基準: {len(settings.all_features)}項目")
    print(f"🎯 研究分野: {len(settings.research_fields)}分野")
    print(f"🧬 遺伝的アルゴリズム: 集団{settings.ga_population_size}個体 × {settings.ga_generations}世代")
    print(f"🌳 ファジィ決定木: 最大深度{settings.max_tree_depth}")
    print("✅ 設定検証完了")