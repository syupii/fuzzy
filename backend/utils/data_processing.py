# utils/data_processing.py - データ前処理ユーティリティ

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import re
import json
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging

from models.schemas import StudentProfile, Laboratory, FieldInterest, EvaluationCriteria
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class DataQualityReport:
    """データ品質レポート"""
    total_records: int
    valid_records: int
    missing_fields: Dict[str, int]
    invalid_values: Dict[str, int]
    duplicates: int
    outliers: Dict[str, int]
    recommendations: List[str]

class DataProcessor:
    """データ前処理メインクラス"""
    
    def __init__(self):
        self.validation_rules = self._initialize_validation_rules()
        self.processing_stats = {
            "processed_students": 0,
            "processed_labs": 0,
            "cleaning_operations": 0,
            "validation_errors": 0
        }
    
    def process_student_data(self, raw_data: List[Dict[str, Any]]) -> List[StudentProfile]:
        """学生データの前処理"""
        
        logger.info(f"学生データ前処理開始: {len(raw_data)}件")
        
        processed_students = []
        
        for i, student_data in enumerate(raw_data):
            try:
                # データクリーニング
                cleaned_data = self._clean_student_data(student_data)
                
                # バリデーション
                if not self._validate_student_data(cleaned_data):
                    logger.warning(f"学生データ{i}がバリデーションに失敗")
                    continue
                
                # StudentProfile作成
                student_profile = self._create_student_profile(cleaned_data)
                processed_students.append(student_profile)
                
                self.processing_stats["processed_students"] += 1
                
            except Exception as e:
                logger.error(f"学生データ{i}の処理エラー: {str(e)}")
                self.processing_stats["validation_errors"] += 1
        
        logger.info(f"学生データ前処理完了: {len(processed_students)}件成功")
        
        return processed_students
    
    def process_lab_data(self, raw_data: List[Dict[str, Any]]) -> List[Laboratory]:
        """研究室データの前処理"""
        
        logger.info(f"研究室データ前処理開始: {len(raw_data)}件")
        
        processed_labs = []
        
        for i, lab_data in enumerate(raw_data):
            try:
                # データクリーニング
                cleaned_data = self._clean_lab_data(lab_data)
                
                # バリデーション
                if not self._validate_lab_data(cleaned_data):
                    logger.warning(f"研究室データ{i}がバリデーションに失敗")
                    continue
                
                # Laboratory作成
                laboratory = self._create_laboratory(cleaned_data)
                processed_labs.append(laboratory)
                
                self.processing_stats["processed_labs"] += 1
                
            except Exception as e:
                logger.error(f"研究室データ{i}の処理エラー: {str(e)}")
                self.processing_stats["validation_errors"] += 1
        
        logger.info(f"研究室データ前処理完了: {len(processed_labs)}件成功")
        
        return processed_labs
    
    def _clean_student_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """学生データクリーニング"""
        
        cleaned = data.copy()
        self.processing_stats["cleaning_operations"] += 1
        
        # 文字列の正規化
        if "student_id" in cleaned:
            cleaned["student_id"] = str(cleaned["student_id"]).strip()
        
        # 評価基準の正規化（1-10範囲）
        criteria_fields = [
            "research_intensity", "advisor_style", "team_work", "workload",
            "theory_practice", "research_field_match", "skill_development",
            "lab_atmosphere", "flexibility", "publication_opportunity",
            "interdisciplinary", "communication_style", "innovation_risk"
        ]
        
        for field in criteria_fields:
            if field in cleaned:
                cleaned[field] = self._normalize_score(cleaned[field])
        
        # 分野興味データの正規化
        if "field_interests" in cleaned:
            cleaned["field_interests"] = self._clean_field_interests(cleaned["field_interests"])
        
        return cleaned
    
    def _clean_lab_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """研究室データクリーニング"""
        
        cleaned = data.copy()
        self.processing_stats["cleaning_operations"] += 1
        
        # ID正規化
        if "id" in cleaned:
            cleaned["id"] = str(cleaned["id"]).strip()
        
        # 名前正規化
        for field in ["name", "professor", "research_area", "specialization"]:
            if field in cleaned:
                cleaned[field] = str(cleaned[field]).strip()
        
        # 研究分野リストの正規化
        if "research_fields" in cleaned:
            if isinstance(cleaned["research_fields"], str):
                # カンマ区切り文字列の場合
                cleaned["research_fields"] = [
                    field.strip() for field in cleaned["research_fields"].split(",")
                ]
            elif not isinstance(cleaned["research_fields"], list):
                cleaned["research_fields"] = []
        
        # 特徴量の正規化
        if "features" in cleaned:
            cleaned["features"] = self._clean_lab_features(cleaned["features"])
        
        return cleaned
    
    def _clean_field_interests(self, interests: Any) -> List[Dict[str, Any]]:
        """分野興味データのクリーニング"""
        
        if not interests:
            return []
        
        cleaned_interests = []
        
        # 辞書リストの場合
        if isinstance(interests, list):
            for interest in interests:
                if isinstance(interest, dict):
                    cleaned_interest = {
                        "field_id": str(interest.get("field_id", "")).strip(),
                        "interest_level": self._normalize_score(interest.get("interest_level", 5)),
                        "experience_level": self._normalize_score(interest.get("experience_level", 1)),
                        "importance_level": self._normalize_score(interest.get("importance_level", 5))
                    }
                    
                    # 有効な分野IDのみ
                    if cleaned_interest["field_id"] in settings.research_fields:
                        cleaned_interests.append(cleaned_interest)
        
        # 辞書の場合（分野ID: スコア形式）
        elif isinstance(interests, dict):
            for field_id, score in interests.items():
                if field_id in settings.research_fields:
                    cleaned_interests.append({
                        "field_id": field_id,
                        "interest_level": self._normalize_score(score),
                        "experience_level": 1,  # デフォルト
                        "importance_level": 5   # デフォルト
                    })
        
        return cleaned_interests
    
    def _clean_lab_features(self, features: Any) -> Dict[str, float]:
        """研究室特徴データのクリーニング"""
        
        if not isinstance(features, dict):
            return {}
        
        cleaned_features = {}
        
        for criterion in settings.evaluation_criteria:
            if criterion in features:
                cleaned_features[criterion] = self._normalize_score(features[criterion])
            else:
                # デフォルト値設定
                cleaned_features[criterion] = 6.0  # 中間値
        
        return cleaned_features
    
    def _normalize_score(self, value: Any, min_val: int = 1, max_val: int = 10) -> int:
        """スコアの正規化（1-10範囲）"""
        
        try:
            # 数値変換
            if isinstance(value, str):
                value = float(value)
            
            # 範囲調整
            normalized = max(min_val, min(max_val, float(value)))
            return int(round(normalized))
            
        except (ValueError, TypeError):
            return 5  # デフォルト値
    
    def _validate_student_data(self, data: Dict[str, Any]) -> bool:
        """学生データのバリデーション"""
        
        # 必須フィールドチェック
        required_fields = ["student_id"]
        for field in required_fields:
            if field not in data or not data[field]:
                return False
        
        # 分野興味データの検証
        if "field_interests" in data:
            interests = data["field_interests"]
            if not isinstance(interests, list) or len(interests) == 0:
                return False
            
            # 有効な分野IDが含まれているかチェック
            valid_fields = 0
            for interest in interests:
                if isinstance(interest, dict) and interest.get("field_id") in settings.research_fields:
                    valid_fields += 1
            
            if valid_fields == 0:
                return False
        
        return True
    
    def _validate_lab_data(self, data: Dict[str, Any]) -> bool:
        """研究室データのバリデーション"""
        
        # 必須フィールドチェック
        required_fields = ["id", "name", "professor"]
        for field in required_fields:
            if field not in data or not data[field]:
                return False
        
        # 研究分野の検証
        if "research_fields" in data:
            research_fields = data["research_fields"]
            if not isinstance(research_fields, list) or len(research_fields) == 0:
                return False
            
            # 有効な分野IDが含まれているかチェック
            valid_fields = 0
            for field_id in research_fields:
                if field_id in settings.research_fields:
                    valid_fields += 1
            
            if valid_fields == 0:
                return False
        
        return True
    
    def _create_student_profile(self, data: Dict[str, Any]) -> StudentProfile:
        """StudentProfileオブジェクト作成"""
        
        # 評価基準作成
        criteria_data = {}
        for criterion in settings.evaluation_criteria:
            criteria_data[criterion] = data.get(criterion, 5)
        
        evaluation_criteria = EvaluationCriteria(**criteria_data)
        
        # 分野興味作成
        field_interests = []
        for interest_data in data.get("field_interests", []):
            field_interest = FieldInterest(**interest_data)
            field_interests.append(field_interest)
        
        return StudentProfile(
            student_id=data["student_id"],
            evaluation_criteria=evaluation_criteria,
            field_interests=field_interests
        )
    
    def _create_laboratory(self, data: Dict[str, Any]) -> Laboratory:
        """Laboratoryオブジェクト作成"""
        
        from models.schemas import LabFeatures
        
        # 研究室特徴作成
        features_data = data.get("features", {})
        lab_features = LabFeatures(**features_data)
        
        return Laboratory(
            id=data["id"],
            name=data["name"],
            professor=data["professor"],
            research_area=data.get("research_area", ""),
            specialization=data.get("specialization", ""),
            research_fields=data.get("research_fields", []),
            description=data.get("description", ""),
            features=lab_features
        )
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """バリデーションルールの初期化"""
        
        return {
            "score_range": (1, 10),
            "required_student_fields": ["student_id"],
            "required_lab_fields": ["id", "name", "professor"],
            "valid_field_ids": set(settings.research_fields.keys()),
            "max_field_interests": 10,
            "min_field_interests": 1
        }
    
    def analyze_data_quality(self, raw_data: List[Dict[str, Any]], 
                           data_type: str = "student") -> DataQualityReport:
        """データ品質分析"""
        
        total_records = len(raw_data)
        valid_records = 0
        missing_fields = defaultdict(int)
        invalid_values = defaultdict(int)
        duplicates = 0
        outliers = defaultdict(int)
        recommendations = []
        
        # 重複チェック用
        seen_ids = set()
        
        for record in raw_data:
            is_valid = True
            
            # ID重複チェック
            record_id = record.get("id" if data_type == "lab" else "student_id")
            if record_id in seen_ids:
                duplicates += 1
            else:
                seen_ids.add(record_id)
            
            # 必須フィールドチェック
            required_fields = (
                self.validation_rules["required_lab_fields"] if data_type == "lab"
                else self.validation_rules["required_student_fields"]
            )
            
            for field in required_fields:
                if field not in record or not record[field]:
                    missing_fields[field] += 1
                    is_valid = False
            
            # スコア範囲チェック
            if data_type == "student":
                for criterion in settings.evaluation_criteria:
                    if criterion in record:
                        value = record[criterion]
                        try:
                            num_value = float(value)
                            if num_value < 1 or num_value > 10:
                                outliers[criterion] += 1
                        except (ValueError, TypeError):
                            invalid_values[criterion] += 1
                            is_valid = False
            
            if is_valid:
                valid_records += 1
        
        # 推奨事項生成
        if duplicates > 0:
            recommendations.append(f"{duplicates}件の重複レコードがあります。除去を検討してください。")
        
        if missing_fields:
            top_missing = sorted(missing_fields.items(), key=lambda x: x[1], reverse=True)[:3]
            recommendations.append(f"主な欠損フィールド: {', '.join([f'{field}({count}件)' for field, count in top_missing])}")
        
        if invalid_values:
            recommendations.append("無効な値が含まれています。データ型の確認が必要です。")
        
        quality_rate = valid_records / total_records if total_records > 0 else 0
        if quality_rate < 0.8:
            recommendations.append("データ品質が低いです。クリーニング処理を強化してください。")
        
        return DataQualityReport(
            total_records=total_records,
            valid_records=valid_records,
            missing_fields=dict(missing_fields),
            invalid_values=dict(invalid_values),
            duplicates=duplicates,
            outliers=dict(outliers),
            recommendations=recommendations
        )
    
    def export_processed_data(self, students: List[StudentProfile], 
                            labs: List[Laboratory], filepath: str) -> None:
        """処理済みデータのエクスポート"""
        
        export_data = {
            "metadata": {
                "export_timestamp": datetime.now().isoformat(),
                "student_count": len(students),
                "lab_count": len(labs),
                "processing_stats": self.processing_stats
            },
            "students": [self._student_to_dict(student) for student in students],
            "laboratories": [self._lab_to_dict(lab) for lab in labs]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"処理済みデータをエクスポートしました: {filepath}")
    
    def _student_to_dict(self, student: StudentProfile) -> Dict[str, Any]:
        """StudentProfileを辞書に変換"""
        
        return {
            "student_id": student.student_id,
            "evaluation_criteria": student.evaluation_criteria.dict(),
            "field_interests": [
                {
                    "field_id": fi.field_id,
                    "interest_level": fi.interest_level,
                    "experience_level": fi.experience_level,
                    "importance_level": fi.importance_level
                }
                for fi in student.field_interests
            ]
        }
    
    def _lab_to_dict(self, lab: Laboratory) -> Dict[str, Any]:
        """Laboratoryを辞書に変換"""
        
        return {
            "id": lab.id,
            "name": lab.name,
            "professor": lab.professor,
            "research_area": lab.research_area,
            "specialization": lab.specialization,
            "research_fields": lab.research_fields,
            "description": lab.description,
            "features": lab.features.dict()
        }
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """処理統計の取得"""
        
        return {
            "processing_stats": self.processing_stats.copy(),
            "validation_rules": {
                "score_range": self.validation_rules["score_range"],
                "valid_field_count": len(self.validation_rules["valid_field_ids"]),
                "required_fields": {
                    "student": self.validation_rules["required_student_fields"],
                    "lab": self.validation_rules["required_lab_fields"]
                }
            }
        }

class DataGenerator:
    """テスト用データ生成クラス"""
    
    def __init__(self):
        self.student_name_samples = [
            "田中太郎", "佐藤花子", "鈴木一郎", "高橋美咲", "渡辺健太",
            "伊藤麻衣", "山本大輔", "中村沙織", "小林隆志", "加藤愛美"
        ]
    
    def generate_sample_students(self, count: int = 50) -> List[Dict[str, Any]]:
        """サンプル学生データ生成"""
        
        students = []
        
        for i in range(count):
            # ランダムな評価基準
            criteria = {}
            for criterion in settings.evaluation_criteria:
                criteria[criterion] = np.random.randint(1, 11)
            
            # ランダムな分野選択（1-5分野）
            selected_fields = np.random.choice(
                list(settings.research_fields.keys()),
                size=np.random.randint(1, 6),
                replace=False
            )
            
            field_interests = []
            for field_id in selected_fields:
                field_interests.append({
                    "field_id": field_id,
                    "interest_level": np.random.randint(5, 11),
                    "experience_level": np.random.randint(1, 8),
                    "importance_level": np.random.randint(4, 11)
                })
            
            student = {
                "student_id": f"student_{i+1:03d}",
                "field_interests": field_interests,
                **criteria
            }
            
            students.append(student)
        
        return students
    
    def generate_sample_labs(self, count: int = 20) -> List[Dict[str, Any]]:
        """サンプル研究室データ生成"""
        
        labs = []
        professors = ["田中教授", "佐藤教授", "鈴木教授", "高橋教授", "渡辺教授"]
        
        for i in range(count):
            # ランダムな研究分野（1-3分野）
            selected_fields = np.random.choice(
                list(settings.research_fields.keys()),
                size=np.random.randint(1, 4),
                replace=False
            )
            
            # ランダムな特徴量
            features = {}
            for criterion in settings.evaluation_criteria:
                features[criterion] = round(np.random.uniform(4.0, 9.0), 1)
            
            lab = {
                "id": f"lab_{i+1:03d}",
                "name": f"{np.random.choice(selected_fields).replace('_', ' ').title()}研究室",
                "professor": np.random.choice(professors),
                "research_area": settings.research_fields[selected_fields[0]]["name"],
                "specialization": f"{settings.research_fields[selected_fields[0]]['name']}の専門研究",
                "research_fields": list(selected_fields),
                "description": f"{settings.research_fields[selected_fields[0]]['name']}に関する研究を行っています",
                "features": features
            }
            
            labs.append(lab)
        
        return labs

# ユーティリティ関数

def load_data_from_csv(filepath: str, data_type: str = "student") -> List[Dict[str, Any]]:
    """CSVファイルからデータを読み込み"""
    
    try:
        df = pd.read_csv(filepath)
        
        # DataFrameを辞書のリストに変換
        data = df.to_dict('records')
        
        logger.info(f"CSVファイル読み込み完了: {filepath} ({len(data)}件)")
        
        return data
        
    except Exception as e:
        logger.error(f"CSVファイル読み込みエラー: {str(e)}")
        return []

def save_data_to_csv(data: List[Dict[str, Any]], filepath: str) -> None:
    """データをCSVファイルに保存"""
    
    try:
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        
        logger.info(f"CSVファイル保存完了: {filepath}")
        
    except Exception as e:
        logger.error(f"CSVファイル保存エラー: {str(e)}")

def validate_field_mappings() -> Dict[str, Any]:
    """分野マッピングの妥当性チェック"""
    
    validation_result = {
        "valid": True,
        "issues": [],
        "field_count": len(settings.research_fields),
        "category_distribution": {}
    }
    
    # カテゴリ分布
    category_counts = defaultdict(int)
    for field_info in settings.research_fields.values():
        category = field_info.get("category", "未分類")
        category_counts[category] += 1
    
    validation_result["category_distribution"] = dict(category_counts)
    
    # 教員マッピングチェック
    for field_id, field_info in settings.research_fields.items():
        faculty = field_info.get("faculty", [])
        if not faculty:
            validation_result["issues"].append(f"分野 {field_id} に担当教員が設定されていません")
            validation_result["valid"] = False
    
    return validation_result