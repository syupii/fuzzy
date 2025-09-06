
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path

class LabDatabase:
    """研究室データベース管理クラス"""
    
    def __init__(self, database_path: str = None):
        if database_path is None:
            # デフォルトパス設定
            self.database_path = Path(__file__).parent.parent / "data" / "labs_database.json"
        else:
            self.database_path = Path(database_path)
        
        self.labs_data = []
        self.metadata = {}
        self._load_database()
    
    def _load_database(self):
        """データベースファイルを読み込み"""
        try:
            if self.database_path.exists():
                with open(self.database_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.labs_data = data.get('labs', [])
                    self.metadata = {
                        'version': data.get('version', '1.0.0'),
                        'last_updated': data.get('last_updated', ''),
                        'total_labs': len(self.labs_data)
                    }
                print(f"✅ 研究室データベース読み込み完了: {len(self.labs_data)}件")
            else:
                print(f"⚠️ データベースファイルが見つかりません: {self.database_path}")
                self._create_default_database()
        except Exception as e:
            print(f"❌ データベース読み込みエラー: {e}")
            self._create_default_database()
    
    def _create_default_database(self):
        """デフォルトデータベースを作成"""
        print("📝 デフォルトデータベースを作成中...")
        
        # デフォルトのサンプルデータ
        default_labs = [
            {
                "id": "lab_sample",
                "name": "サンプル研究室",
                "professor": "サンプル教授",
                "research_area": "人工知能・機械学習",
                "specialization": "サンプル専門分野",
                "research_fields": ["人工知能・機械学習"],
                "description": "サンプル研究室の説明",
                "features": {
                    "research_intensity": 7.0,
                    "advisor_style": 7.0,
                    "team_work": 7.0,
                    "workload": 7.0,
                    "theory_practice": 7.0,
                    "research_field_match": 7.0,
                    "skill_development": 7.0,
                    "learning_pace": 7.0,
                    "difficulty_preference": 7.0,
                    "lab_atmosphere": 7.0,
                    "communication_style": 7.0,
                    "meeting_frequency": 7.0,
                    "flexibility": 7.0,
                    "evening_weekend_work": 7.0,
                    "innovation_risk": 7.0,
                    "methodology_preference": 7.0,
                    "interdisciplinary": 7.0,
                    "publication_opportunity": 7.0,
                    "financial_support": 7.0,
                    "lab_hierarchy": 7.0,
                    "core_time_flexibility": 7.0
                },
                "metadata": {
                    "faculty_count": 1,
                    "student_count": 5,
                    "recent_publications": 10,
                    "funding_level": "中",
                    "equipment_rating": 7
                }
            }
        ]
        
        self.labs_data = default_labs
        self.save_database()
    
    def get_all_labs(self) -> List[Dict[str, Any]]:
        """全研究室データを取得"""
        return self.labs_data
    
    def get_lab_by_id(self, lab_id: str) -> Optional[Dict[str, Any]]:
        """IDで研究室を検索"""
        for lab in self.labs_data:
            if lab.get('id') == lab_id:
                return lab
        return None
    
    def get_labs_by_field(self, research_field: str) -> List[Dict[str, Any]]:
        """研究分野で研究室を検索"""
        matching_labs = []
        for lab in self.labs_data:
            if research_field in lab.get('research_fields', []):
                matching_labs.append(lab)
        return matching_labs
    
    def add_lab(self, lab_data: Dict[str, Any]) -> bool:
        """新しい研究室を追加"""
        try:
            # IDの重複チェック
            if self.get_lab_by_id(lab_data.get('id')):
                print(f"⚠️ 既存のIDです: {lab_data.get('id')}")
                return False
            
            # データ検証
            if self._validate_lab_data(lab_data):
                self.labs_data.append(lab_data)
                self.save_database()
                print(f"✅ 研究室を追加しました: {lab_data.get('name')}")
                return True
            else:
                print(f"❌ 無効なデータです: {lab_data.get('name')}")
                return False
        except Exception as e:
            print(f"❌ 研究室追加エラー: {e}")
            return False
    
    def update_lab(self, lab_id: str, lab_data: Dict[str, Any]) -> bool:
        """研究室データを更新"""
        try:
            for i, lab in enumerate(self.labs_data):
                if lab.get('id') == lab_id:
                    if self._validate_lab_data(lab_data):
                        self.labs_data[i] = lab_data
                        self.save_database()
                        print(f"✅ 研究室を更新しました: {lab_data.get('name')}")
                        return True
                    else:
                        print(f"❌ 無効なデータです: {lab_data.get('name')}")
                        return False
            
            print(f"⚠️ 研究室が見つかりません: {lab_id}")
            return False
        except Exception as e:
            print(f"❌ 研究室更新エラー: {e}")
            return False
    
    def delete_lab(self, lab_id: str) -> bool:
        """研究室を削除"""
        try:
            for i, lab in enumerate(self.labs_data):
                if lab.get('id') == lab_id:
                    deleted_lab = self.labs_data.pop(i)
                    self.save_database()
                    print(f"✅ 研究室を削除しました: {deleted_lab.get('name')}")
                    return True
            
            print(f"⚠️ 研究室が見つかりません: {lab_id}")
            return False
        except Exception as e:
            print(f"❌ 研究室削除エラー: {e}")
            return False
    
    def save_database(self):
        """データベースをファイルに保存"""
        try:
            # ディレクトリ作成
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
            
            # メタデータ更新
            from datetime import datetime
            
            database_content = {
                "version": "1.0.0",
                "last_updated": datetime.now().isoformat(),
                "description": "研究室選択支援システム用研究室データベース",
                "total_labs": len(self.labs_data),
                "labs": self.labs_data
            }
            
            with open(self.database_path, 'w', encoding='utf-8') as f:
                json.dump(database_content, f, ensure_ascii=False, indent=2)
            
            print(f"💾 データベースを保存しました: {self.database_path}")
        except Exception as e:
            print(f"❌ データベース保存エラー: {e}")
    
    def _validate_lab_data(self, lab_data: Dict[str, Any]) -> bool:
        """研究室データの妥当性検証"""
        required_fields = ['id', 'name', 'professor', 'research_area', 'features']
        
        # 必須フィールドチェック
        for field in required_fields:
            if field not in lab_data:
                print(f"❌ 必須フィールドが不足: {field}")
                return False
        
        # featuresの21項目チェック
        required_features = [
            'research_intensity', 'advisor_style', 'team_work', 'workload', 
            'theory_practice', 'research_field_match', 'skill_development',
            'learning_pace', 'difficulty_preference', 'lab_atmosphere',
            'communication_style', 'meeting_frequency', 'flexibility',
            'evening_weekend_work', 'innovation_risk', 'methodology_preference',
            'interdisciplinary', 'publication_opportunity', 'financial_support',
            'lab_hierarchy', 'core_time_flexibility'
        ]
        
        features = lab_data.get('features', {})
        for feature in required_features:
            if feature not in features:
                print(f"❌ 必須特徴量が不足: {feature}")
                return False
            
            value = features[feature]
            if not isinstance(value, (int, float)) or not (1.0 <= value <= 10.0):
                print(f"❌ 無効な特徴量値: {feature} = {value}")
                return False
        
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """データベース統計情報を取得"""
        if not self.labs_data:
            return {"total_labs": 0}
        
        # 分野別統計
        field_distribution = {}
        for lab in self.labs_data:
            for field in lab.get('research_fields', []):
                field_distribution[field] = field_distribution.get(field, 0) + 1
        
        # 学生数統計
        total_students = sum(lab.get('metadata', {}).get('student_count', 0) 
                           for lab in self.labs_data)
        
        return {
            "total_labs": len(self.labs_data),
            "total_students": total_students,
            "field_distribution": field_distribution,
            "avg_students_per_lab": total_students / len(self.labs_data) if self.labs_data else 0,
            "database_version": self.metadata.get('version', '1.0.0'),
            "last_updated": self.metadata.get('last_updated', '')
        }