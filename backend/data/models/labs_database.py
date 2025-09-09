# backend/data/models/labs_database.py - 研究室データベース管理クラス（更新版）
# labs_database.json v2.0対応

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

class LabDatabase:
    """研究室データベース管理クラス v2.0"""
    
    def __init__(self, database_path: str = None):
        if database_path is None:
            # デフォルトパス設定 - labs_database.json
            self.database_path = Path(__file__).parent.parent / "labs_database.json"
        else:
            self.database_path = Path(database_path)
        
        self.labs_data = []
        self.metadata = {}
        self.evaluation_criteria = []
        self.research_fields = []
        self._load_database()
    
    def _load_database(self):
        """データベースファイルを読み込み"""
        try:
            if self.database_path.exists():
                with open(self.database_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.labs_data = data.get('labs', [])
                    self.evaluation_criteria = data.get('evaluation_criteria', {})
                    self.research_fields = data.get('research_fields', [])
                    self.metadata = {
                        'version': data.get('version', '2.0.0'),
                        'last_updated': data.get('last_updated', ''),
                        'total_labs': len(self.labs_data),
                        'description': data.get('description', ''),
                        'total_fields': len(self.research_fields)
                    }
                print(f"✅ 研究室データベース v{self.metadata['version']} 読み込み完了: {len(self.labs_data)}件")
                print(f"📊 研究分野数: {len(self.research_fields)}")
            else:
                print(f"⚠️ データベースファイルが見つかりません: {self.database_path}")
                self._create_default_database()
        except Exception as e:
            print(f"❌ データベース読み込みエラー: {e}")
            self._create_default_database()
    
    def _create_default_database(self):
        """デフォルトデータベースを作成"""
        print("📝 デフォルトデータベース v2.0 を作成中...")
        
        # デフォルトの評価基準
        default_criteria = {
            "basic": [
                "research_intensity",
                "advisor_style", 
                "team_work",
                "workload",
                "theory_practice"
            ],
            "extended": [
                "research_field_match",
                "skill_development",
                "lab_atmosphere",
                "flexibility",
                "publication_opportunity"
            ],
            "special": [
                "interdisciplinary",
                "communication_style",
                "innovation_risk"
            ]
        }
        
        # デフォルトの研究分野
        default_fields = [
            "人工知能・機械学習",
            "画像・映像処理",
            "コンピュータネットワーク・セキュリティ",
            "データベース・情報システム",
            "組込み・IoT",
            "Webデザイン・UI/UX",
            "デザイン・視覚表現",
            "映像・アニメーション",
            "コンピュータ音楽・サウンドアート",
            "ゲーム開発・eスポーツ",
            "VR/AR・メディアアート"
        ]
        
        # デフォルトのサンプルデータ
        default_labs = [
            {
                "id": "lab_sample_001",
                "name": "サンプル人工知能研究室",
                "professor": "サンプル教授",
                "research_area": "人工知能・機械学習",
                "specialization": "機械学習、深層学習、自然言語処理",
                "research_fields": ["人工知能・機械学習"],
                "description": "人工知能と機械学習の研究を行うサンプル研究室です。深層学習や自然言語処理の最新技術に取り組んでいます。",
                "features": {
                    "research_intensity": 8.0,
                    "advisor_style": 7.0,
                    "team_work": 8.0,
                    "workload": 7.5,
                    "theory_practice": 6.5,
                    "research_field_match": 8.5,
                    "skill_development": 8.0,
                    "lab_atmosphere": 7.5,
                    "flexibility": 7.0,
                    "publication_opportunity": 8.0,
                    "interdisciplinary": 7.0,
                    "communication_style": 7.0,
                    "innovation_risk": 7.5
                },
                "metadata": {
                    "faculty_count": 1,
                    "student_count": 8,
                    "recent_publications": 15,
                    "funding_level": "高",
                    "equipment_rating": 9
                }
            }
        ]
        
        self.labs_data = default_labs
        self.evaluation_criteria = default_criteria
        self.research_fields = default_fields
        self.metadata = {
            'version': '2.0.0',
            'last_updated': datetime.now().isoformat(),
            'total_labs': len(default_labs),
            'description': '研究室選択支援システム用研究室データベース v2.0',
            'total_fields': len(default_fields)
        }
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
            lab_fields = lab.get('research_fields', [])
            if research_field in lab_fields or lab.get('research_area') == research_field:
                matching_labs.append(lab)
        return matching_labs
    
    def get_labs_by_professor(self, professor_name: str) -> List[Dict[str, Any]]:
        """教授名で研究室を検索"""
        matching_labs = []
        for lab in self.labs_data:
            if lab.get('professor', '').lower() == professor_name.lower():
                matching_labs.append(lab)
        return matching_labs
    
    def search_labs(self, query: str) -> List[Dict[str, Any]]:
        """研究室を検索（名前、教授、説明、専門分野で検索）"""
        query_lower = query.lower()
        matching_labs = []
        
        for lab in self.labs_data:
            # 検索対象のフィールド
            search_fields = [
                lab.get('name', ''),
                lab.get('professor', ''),
                lab.get('description', ''),
                lab.get('specialization', ''),
                lab.get('research_area', '')
            ]
            
            # 研究分野も検索対象に含める
            research_fields = lab.get('research_fields', [])
            search_fields.extend(research_fields)
            
            # いずれかのフィールドにクエリが含まれているかチェック
            for field in search_fields:
                if query_lower in str(field).lower():
                    matching_labs.append(lab)
                    break
        
        return matching_labs
    
    def get_research_fields_stats(self) -> Dict[str, Any]:
        """研究分野の統計情報を取得"""
        field_stats = {}
        
        for lab in self.labs_data:
            research_area = lab.get('research_area', 'その他')
            if research_area not in field_stats:
                field_stats[research_area] = {
                    'name': research_area,
                    'lab_count': 0,
                    'professors': [],
                    'avg_funding_level': [],
                    'avg_equipment_rating': []
                }
            
            field_stats[research_area]['lab_count'] += 1
            
            # 教授の追加
            prof = lab.get('professor')
            if prof and prof not in field_stats[research_area]['professors']:
                field_stats[research_area]['professors'].append(prof)
            
            # メタデータの統計
            metadata = lab.get('metadata', {})
            funding = metadata.get('funding_level')
            if funding:
                field_stats[research_area]['avg_funding_level'].append(funding)
            
            equipment = metadata.get('equipment_rating')
            if equipment:
                field_stats[research_area]['avg_equipment_rating'].append(equipment)
        
        # 平均値の計算
        for field, stats in field_stats.items():
            if stats['avg_equipment_rating']:
                stats['avg_equipment_rating'] = sum(stats['avg_equipment_rating']) / len(stats['avg_equipment_rating'])
            else:
                stats['avg_equipment_rating'] = None
        
        return field_stats
    
    def add_lab(self, lab_data: Dict[str, Any]) -> bool:
        """新しい研究室を追加"""
        try:
            # データ妥当性検証
            if not self._validate_lab_data(lab_data):
                return False
            
            # ID重複チェック
            existing_lab = self.get_lab_by_id(lab_data['id'])
            if existing_lab:
                print(f"❌ 重複するID: {lab_data['id']}")
                return False
            
            self.labs_data.append(lab_data)
            self.metadata['total_labs'] = len(self.labs_data)
            self.save_database()
            print(f"✅ 研究室を追加しました: {lab_data.get('name')}")
            return True
            
        except Exception as e:
            print(f"❌ 研究室追加エラー: {e}")
            return False
    
    def update_lab(self, lab_id: str, updated_data: Dict[str, Any]) -> bool:
        """研究室情報を更新"""
        try:
            for i, lab in enumerate(self.labs_data):
                if lab.get('id') == lab_id:
                    # 既存データをマージ
                    self.labs_data[i].update(updated_data)
                    self.save_database()
                    print(f"✅ 研究室を更新しました: {self.labs_data[i].get('name')}")
                    return True
            
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
                    self.metadata['total_labs'] = len(self.labs_data)
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
            database_content = {
                "version": "2.0.0",
                "last_updated": datetime.now().isoformat(),
                "description": "遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム用データベース",
                "total_labs": len(self.labs_data),
                "evaluation_criteria": self.evaluation_criteria,
                "research_fields": self.research_fields,
                "labs": self.labs_data
            }
            
            with open(self.database_path, 'w', encoding='utf-8') as f:
                json.dump(database_content, f, ensure_ascii=False, indent=2)
            
            print(f"💾 データベース v2.0 を保存しました: {self.database_path}")
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
        
        # features の13項目チェック
        required_features = [
            'research_intensity', 'advisor_style', 'team_work', 'workload', 
            'theory_practice', 'research_field_match', 'skill_development',
            'lab_atmosphere', 'flexibility', 'publication_opportunity',
            'interdisciplinary', 'communication_style', 'innovation_risk'
        ]
        
        features = lab_data.get('features', {})
        for feature in required_features:
            if feature not in features:
                print(f"⚠️ 推奨機能フィールドが不足: {feature}")
                # 警告のみで処理続行
        
        # 値の範囲チェック（1-10）
        for feature, value in features.items():
            if not isinstance(value, (int, float)) or not (1 <= value <= 10):
                print(f"❌ 無効な機能値: {feature} = {value} (1-10の範囲で指定してください)")
                return False
        
        print(f"✅ 研究室データ検証完了: {lab_data.get('name')}")
        return True
    
    def export_to_csv(self, output_path: str = None) -> bool:
        """CSVファイルにエクスポート"""
        try:
            import pandas as pd
            
            if output_path is None:
                output_path = self.database_path.parent / "labs_export.csv"
            
            # フラット化されたデータ作成
            flat_data = []
            for lab in self.labs_data:
                flat_lab = {
                    'id': lab.get('id'),
                    'name': lab.get('name'),
                    'professor': lab.get('professor'),
                    'research_area': lab.get('research_area'),
                    'specialization': lab.get('specialization', ''),
                    'description': lab.get('description', ''),
                    'research_fields': ', '.join(lab.get('research_fields', []))
                }
                
                # features を展開
                features = lab.get('features', {})
                for feature, value in features.items():
                    flat_lab[f'feature_{feature}'] = value
                
                # metadata を展開
                metadata = lab.get('metadata', {})
                for meta_key, meta_value in metadata.items():
                    flat_lab[f'meta_{meta_key}'] = meta_value
                
                flat_data.append(flat_lab)
            
            df = pd.DataFrame(flat_data)
            df.to_csv(output_path, index=False, encoding='utf-8')
            print(f"✅ CSVエクスポート完了: {output_path}")
            return True
            
        except ImportError:
            print("❌ pandas が必要です: pip install pandas")
            return False
        except Exception as e:
            print(f"❌ CSVエクスポートエラー: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """データベース情報を取得"""
        return {
            "version": self.metadata.get('version', '2.0.0'),
            "last_updated": self.metadata.get('last_updated'),
            "total_labs": len(self.labs_data),
            "total_fields": len(self.research_fields),
            "evaluation_criteria_count": sum(len(criteria) for criteria in self.evaluation_criteria.values()),
            "database_path": str(self.database_path),
            "research_fields": self.research_fields,
            "evaluation_criteria": self.evaluation_criteria
        }

# 使用例とテスト用のメイン関数
if __name__ == "__main__":
    # データベースのテスト
    print("🧪 LabDatabase v2.0 テスト開始...")
    
    db = LabDatabase()
    
    # データベース情報表示
    info = db.get_database_info()
    print(f"📊 データベース情報:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 研究分野統計
    field_stats = db.get_research_fields_stats()
    print(f"\n📈 研究分野統計:")
    for field, stats in field_stats.items():
        print(f"  {field}: {stats['lab_count']}研究室, {len(stats['professors'])}教員")
    
    # 検索テスト
    ai_labs = db.get_labs_by_field("人工知能・機械学習")
    print(f"\n🔍 人工知能・機械学習分野: {len(ai_labs)}件")
    
    print("✅ テスト完了")