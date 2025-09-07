# utils/data_processing.py - データ前処理ユーティリティ

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
from pathlib import Path

from models.schemas import (
    StudentProfile, Laboratory, EvaluationCriteria, 
    FieldInterest, ResearchFieldEnum
)
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class DataValidationResult:
    """データ検証結果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    processed_count: int
    skipped_count: int

@dataclass
class DataStatistics:
    """データ統計情報"""
    total_records: int
    valid_records: int
    invalid_records: int
    missing_values: Dict[str, int]
    value_ranges: Dict[str, Tuple[float, float]]
    categorical_distributions: Dict[str, Dict[str, int]]

class DataValidator:
    """データ検証クラス"""
    
    def __init__(self):
        self.validation_rules = self._setup_validation_rules()
    
    def _setup_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """検証ルールの設定"""
        
        return {
            "evaluation_criteria": {
                "required_fields": [
                    "research_intensity", "advisor_style", "team_work", 
                    "workload", "theory_practice"
                ],
                "value_range": (1.0, 10.0),
                "type": float
            },
            "field_interests": {
                "min_count": 1,
                "max_count": 5,
                "required_fields": ["field", "interest_level", "priority"]
            },
            "student_profile": {
                "required_fields": ["student_id", "evaluation_criteria", "field_interests"],
                "student_id_pattern": r"^[a-zA-Z0-9_-]+$"
            },
            "laboratory": {
                "required_fields": ["lab_id", "faculty", "research_field", "characteristics"],
                "lab_id_pattern": r"^[a-zA-Z0-9_-]+$"
            }
        }
    
    def validate_student_profile(self, profile_data: Dict[str, Any]) -> DataValidationResult:
        """学生プロフィールの検証"""
        
        errors = []
        warnings = []
        
        try:
            # 必須フィールドチェック
            required_fields = self.validation_rules["student_profile"]["required_fields"]
            for field in required_fields:
                if field not in profile_data:
                    errors.append(f"必須フィールドが不足: {field}")
            
            # 学生ID形式チェック
            if "student_id" in profile_data:
                pattern = self.validation_rules["student_profile"]["student_id_pattern"]
                if not re.match(pattern, profile_data["student_id"]):
                    errors.append("学生IDの形式が無効です")
            
            # 評価基準の検証
            if "evaluation_criteria" in profile_data:
                criteria_result = self._validate_evaluation_criteria(
                    profile_data["evaluation_criteria"]
                )
                errors.extend(criteria_result.errors)
                warnings.extend(criteria_result.warnings)
            
            # 分野興味の検証
            if "field_interests" in profile_data:
                interests_result = self._validate_field_interests(
                    profile_data["field_interests"]
                )
                errors.extend(interests_result.errors)
                warnings.extend(interests_result.warnings)
            
            return DataValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                processed_count=1 if len(errors) == 0 else 0,
                skipped_count=1 if len(errors) > 0 else 0
            )
            
        except Exception as e:
            errors.append(f"検証中にエラー: {str(e)}")
            return DataValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                processed_count=0,
                skipped_count=1
            )
    
    def _validate_evaluation_criteria(self, criteria_data: Dict[str, Any]) -> DataValidationResult:
        """評価基準の検証"""
        
        errors = []
        warnings = []
        
        rules = self.validation_rules["evaluation_criteria"]
        required_fields = rules["required_fields"]
        value_range = rules["value_range"]
        
        # 必須フィールドチェック
        for field in required_fields:
            if field not in criteria_data:
                errors.append(f"必須評価基準が不足: {field}")
            elif criteria_data[field] is None:
                errors.append(f"必須評価基準が未設定: {field}")
        
        # 値の範囲チェック
        for field, value in criteria_data.items():
            if value is not None:
                if not isinstance(value, (int, float)):
                    errors.append(f"{field}は数値である必要があります")
                elif not (value_range[0] <= value <= value_range[1]):
                    errors.append(f"{field}は{value_range[0]}-{value_range[1]}の範囲で指定してください")
        
        # 拡張項目の推奨チェック
        extended_fields = settings.evaluation_criteria[5:10]
        missing_extended = [f for f in extended_fields if criteria_data.get(f) is None]
        
        if len(missing_extended) > 3:
            warnings.append(f"拡張評価項目の設定を推奨します: {', '.join(missing_extended[:3])}")
        
        return DataValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            processed_count=0,
            skipped_count=0
        )
    
    def _validate_field_interests(self, interests_data: List[Dict[str, Any]]) -> DataValidationResult:
        """分野興味の検証"""
        
        errors = []
        warnings = []
        
        rules = self.validation_rules["field_interests"]
        
        # 件数チェック
        if len(interests_data) < rules["min_count"]:
            errors.append(f"最低{rules['min_count']}つの研究分野への興味が必要です")
        elif len(interests_data) > rules["max_count"]:
            warnings.append(f"研究分野への興味は{rules['max_count']}つまでを推奨します")
        
        # 各興味の検証
        priorities = []
        for i, interest in enumerate(interests_data):
            # 必須フィールド
            for field in rules["required_fields"]:
                if field not in interest:
                    errors.append(f"分野興味{i+1}に必須フィールドが不足: {field}")
            
            # 興味レベルの範囲
            if "interest_level" in interest:
                level = interest["interest_level"]
                if not isinstance(level, (int, float)) or not (1 <= level <= 10):
                    errors.append(f"分野興味{i+1}の興味レベルは1-10の範囲で指定してください")
            
            # 優先順位の重複チェック
            if "priority" in interest:
                priority = interest["priority"]
                if priority in priorities:
                    errors.append(f"優先順位{priority}が重複しています")
                priorities.append(priority)
            
            # 研究分野の有効性
            if "field" in interest:
                field_value = interest["field"]
                valid_fields = [f.value for f in ResearchFieldEnum]
                if field_value not in valid_fields:
                    errors.append(f"無効な研究分野: {field_value}")
        
        return DataValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            processed_count=0,
            skipped_count=0
        )

class DataCleaner:
    """データクリーニングクラス"""
    
    def clean_student_profile_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """学生プロフィールデータのクリーニング"""
        
        cleaned_data = raw_data.copy()
        
        try:
            # 学生IDの正規化
            if "student_id" in cleaned_data:
                cleaned_data["student_id"] = str(cleaned_data["student_id"]).strip()
            
            # 評価基準のクリーニング
            if "evaluation_criteria" in cleaned_data:
                cleaned_data["evaluation_criteria"] = self._clean_evaluation_criteria(
                    cleaned_data["evaluation_criteria"]
                )
            
            # 分野興味のクリーニング
            if "field_interests" in cleaned_data:
                cleaned_data["field_interests"] = self._clean_field_interests(
                    cleaned_data["field_interests"]
                )
            
            # 不要なフィールドの削除
            cleaned_data = self._remove_extra_fields(cleaned_data, "student_profile")
            
            return cleaned_data
            
        except Exception as e:
            logger.error(f"学生プロフィールクリーニングエラー: {e}")
            return raw_data
    
    def _clean_evaluation_criteria(self, criteria_data: Dict[str, Any]) -> Dict[str, Any]:
        """評価基準のクリーニング"""
        
        cleaned_criteria = {}
        
        for criterion in settings.evaluation_criteria:
            value = criteria_data.get(criterion)
            
            if value is not None:
                # 数値変換
                try:
                    float_value = float(value)
                    # 範囲制限
                    cleaned_value = max(1.0, min(10.0, float_value))
                    # 小数点以下1桁に丸め
                    cleaned_criteria[criterion] = round(cleaned_value, 1)
                except (ValueError, TypeError):
                    logger.warning(f"無効な評価基準値: {criterion}={value}")
                    cleaned_criteria[criterion] = None
            else:
                cleaned_criteria[criterion] = None
        
        return cleaned_criteria
    
    def _clean_field_interests(self, interests_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分野興味のクリーニング"""
        
        cleaned_interests = []
        
        for interest in interests_data:
            cleaned_interest = {}
            
            # 研究分野の正規化
            if "field" in interest:
                field_value = str(interest["field"]).lower().replace("-", "_")
                valid_fields = [f.value for f in ResearchFieldEnum]
                
                if field_value in valid_fields:
                    cleaned_interest["field"] = field_value
                else:
                    # 類似フィールドの検索
                    similar_field = self._find_similar_field(field_value, valid_fields)
                    if similar_field:
                        cleaned_interest["field"] = similar_field
                        logger.info(f"分野名を正規化: {field_value} -> {similar_field}")
                    else:
                        logger.warning(f"無効な研究分野をスキップ: {field_value}")
                        continue
            
            # 興味レベルの正規化
            if "interest_level" in interest:
                try:
                    level = float(interest["interest_level"])
                    cleaned_interest["interest_level"] = max(1.0, min(10.0, level))
                except (ValueError, TypeError):
                    cleaned_interest["interest_level"] = 5.0  # デフォルト値
            
            # 優先順位の正規化
            if "priority" in interest:
                try:
                    priority = int(interest["priority"])
                    cleaned_interest["priority"] = max(1, priority)
                except (ValueError, TypeError):
                    cleaned_interest["priority"] = len(cleaned_interests) + 1
            
            cleaned_interests.append(cleaned_interest)
        
        # 優先順位の重複解決
        cleaned_interests = self._resolve_priority_conflicts(cleaned_interests)
        
        return cleaned_interests
    
    def _find_similar_field(self, field_value: str, valid_fields: List[str]) -> Optional[str]:
        """類似分野名の検索"""
        
        # 簡易的な類似度計算
        field_value_normalized = field_value.lower().replace("_", "").replace("-", "")
        
        best_match = None
        best_score = 0
        
        for valid_field in valid_fields:
            valid_normalized = valid_field.lower().replace("_", "").replace("-", "")
            
            # 部分一致スコア
            score = 0
            if field_value_normalized in valid_normalized:
                score = len(field_value_normalized) / len(valid_normalized)
            elif valid_normalized in field_value_normalized:
                score = len(valid_normalized) / len(field_value_normalized)
            
            if score > best_score and score > 0.5:  # 50%以上の一致
                best_score = score
                best_match = valid_field
        
        return best_match
    
    def _resolve_priority_conflicts(self, interests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """優先順位の重複解決"""
        
        # 優先順位でソート
        sorted_interests = sorted(interests, key=lambda x: x.get("priority", 999))
        
        # 優先順位を再割り当て
        for i, interest in enumerate(sorted_interests):
            interest["priority"] = i + 1
        
        return sorted_interests
    
    def _remove_extra_fields(self, data: Dict[str, Any], data_type: str) -> Dict[str, Any]:
        """不要フィールドの削除"""
        
        # 許可されたフィールドの定義
        allowed_fields = {
            "student_profile": [
                "student_id", "evaluation_criteria", "field_interests",
                "grade", "gpa", "preferred_lab_size", "time_availability"
            ],
            "laboratory": [
                "lab_id", "faculty", "research_field", "characteristics",
                "lab_name", "description", "recent_achievements", 
                "required_skills", "lab_environment"
            ]
        }
        
        if data_type not in allowed_fields:
            return data
        
        return {
            key: value for key, value in data.items()
            if key in allowed_fields[data_type]
        }

class DataConverter:
    """データ変換クラス"""
    
    def csv_to_student_profiles(self, csv_file_path: str) -> List[StudentProfile]:
        """CSVファイルから学生プロフィールリストに変換"""
        
        try:
            df = pd.read_csv(csv_file_path)
            profiles = []
            
            validator = DataValidator()
            cleaner = DataCleaner()
            
            for index, row in df.iterrows():
                try:
                    # 行データを辞書に変換
                    raw_data = row.to_dict()
                    
                    # 評価基準の抽出
                    criteria_data = {}
                    for criterion in settings.evaluation_criteria:
                        if criterion in raw_data:
                            criteria_data[criterion] = raw_data[criterion]
                    
                    # 分野興味の抽出（複数列想定）
                    interests_data = []
                    for i in range(1, 6):  # 最大5つの分野
                        field_col = f"field_{i}"
                        level_col = f"interest_level_{i}"
                        priority_col = f"priority_{i}"
                        
                        if field_col in raw_data and pd.notna(raw_data[field_col]):
                            interest = {
                                "field": raw_data[field_col],
                                "interest_level": raw_data.get(level_col, 5.0),
                                "priority": raw_data.get(priority_col, i)
                            }
                            interests_data.append(interest)
                    
                    # プロフィールデータの構築
                    profile_data = {
                        "student_id": raw_data.get("student_id", f"student_{index}"),
                        "evaluation_criteria": criteria_data,
                        "field_interests": interests_data
                    }
                    
                    # データクリーニング
                    cleaned_data = cleaner.clean_student_profile_data(profile_data)
                    
                    # 検証
                    validation_result = validator.validate_student_profile(cleaned_data)
                    
                    if validation_result.is_valid:
                        # Pydanticモデルに変換
                        profile = self._dict_to_student_profile(cleaned_data)
                        profiles.append(profile)
                    else:
                        logger.warning(f"行{index}をスキップ: {validation_result.errors}")
                
                except Exception as e:
                    logger.error(f"行{index}の処理エラー: {e}")
                    continue
            
            logger.info(f"CSV変換完了: {len(profiles)}件のプロフィール")
            return profiles
            
        except Exception as e:
            logger.error(f"CSV読み込みエラー: {e}")
            return []
    
    def _dict_to_student_profile(self, data: Dict[str, Any]) -> StudentProfile:
        """辞書からStudentProfileに変換"""
        
        from models.schemas import EvaluationCriteria, FieldInterest
        
        # 評価基準の変換
        criteria_data = data["evaluation_criteria"]
        evaluation_criteria = EvaluationCriteria(**criteria_data)
        
        # 分野興味の変換
        field_interests = []
        for interest_data in data["field_interests"]:
            field_interest = FieldInterest(
                field=ResearchFieldEnum(interest_data["field"]),
                interest_level=interest_data["interest_level"],
                priority=interest_data["priority"]
            )
            field_interests.append(field_interest)
        
        # プロフィールの作成
        return StudentProfile(
            student_id=data["student_id"],
            evaluation_criteria=evaluation_criteria,
            field_interests=field_interests,
            grade=data.get("grade"),
            gpa=data.get("gpa"),
            preferred_lab_size=data.get("preferred_lab_size"),
            time_availability=data.get("time_availability")
        )
    
    def json_to_laboratories(self, json_file_path: str) -> List[Laboratory]:
        """JSONファイルから研究室リストに変換"""
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            laboratories = []
            
            for lab_data in data.get("laboratories", []):
                try:
                    laboratory = self._dict_to_laboratory(lab_data)
                    laboratories.append(laboratory)
                except Exception as e:
                    logger.error(f"研究室データ変換エラー: {e}")
                    continue
            
            logger.info(f"JSON変換完了: {len(laboratories)}件の研究室")
            return laboratories
            
        except Exception as e:
            logger.error(f"JSON読み込みエラー: {e}")
            return []
    
    def _dict_to_laboratory(self, data: Dict[str, Any]) -> Laboratory:
        """辞書からLaboratoryに変換"""
        
        from models.schemas import Faculty, EvaluationCriteria
        
        # 教員情報の変換
        faculty_data = data["faculty"]
        faculty = Faculty(
            name=faculty_data["name"],
            name_en=faculty_data.get("name_en"),
            title=faculty_data.get("title"),
            specialties=faculty_data["specialties"]
        )
        
        # 研究室特性の変換
        characteristics_data = data["characteristics"]
        characteristics = EvaluationCriteria(**characteristics_data)
        
        # 研究室の作成
        return Laboratory(
            lab_id=data["lab_id"],
            faculty=faculty,
            research_field=ResearchFieldEnum(data["research_field"]),
            characteristics=characteristics,
            lab_name=data.get("lab_name"),
            description=data.get("description"),
            recent_achievements=data.get("recent_achievements"),
            required_skills=data.get("required_skills"),
            lab_environment=data.get("lab_environment"),
            current_students=data.get("current_students"),
            graduation_rate=data.get("graduation_rate"),
            job_placement_rate=data.get("job_placement_rate")
        )

class DataAnalyzer:
    """データ分析クラス"""
    
    def analyze_student_profiles(self, profiles: List[StudentProfile]) -> DataStatistics:
        """学生プロフィールの統計分析"""
        
        if not profiles:
            return DataStatistics(
                total_records=0, valid_records=0, invalid_records=0,
                missing_values={}, value_ranges={}, categorical_distributions={}
            )
        
        # 基本統計
        total_records = len(profiles)
        valid_records = len([p for p in profiles if p.evaluation_criteria])
        invalid_records = total_records - valid_records
        
        # 評価基準の統計
        criteria_values = defaultdict(list)
        missing_values = defaultdict(int)
        
        for profile in profiles:
            criteria_dict = profile.evaluation_criteria.dict()
            for criterion, value in criteria_dict.items():
                if value is not None:
                    criteria_values[criterion].append(value)
                else:
                    missing_values[criterion] += 1
        
        # 値の範囲計算
        value_ranges = {}
        for criterion, values in criteria_values.items():
            if values:
                value_ranges[criterion] = (min(values), max(values))
        
        # カテゴリカル分布（研究分野）
        field_distribution = defaultdict(int)
        for profile in profiles:
            for interest in profile.field_interests:
                field_distribution[interest.field.value] += 1
        
        categorical_distributions = {
            "research_fields": dict(field_distribution)
        }
        
        return DataStatistics(
            total_records=total_records,
            valid_records=valid_records,
            invalid_records=invalid_records,
            missing_values=dict(missing_values),
            value_ranges=value_ranges,
            categorical_distributions=categorical_distributions
        )
    
    def generate_summary_report(self, profiles: List[StudentProfile], 
                               laboratories: List[Laboratory]) -> Dict[str, Any]:
        """サマリーレポートの生成"""
        
        student_stats = self.analyze_student_profiles(profiles)
        
        # 研究室統計
        lab_field_distribution = defaultdict(int)
        for lab in laboratories:
            lab_field_distribution[lab.research_field.value] += 1
        
        # 適合性分析（簡易版）
        field_matches = 0
        total_combinations = len(profiles) * len(laboratories)
        
        for profile in profiles:
            student_fields = {interest.field.value for interest in profile.field_interests}
            for lab in laboratories:
                if lab.research_field.value in student_fields:
                    field_matches += 1
        
        field_match_rate = field_matches / total_combinations if total_combinations > 0 else 0
        
        return {
            "summary": {
                "total_students": len(profiles),
                "total_laboratories": len(laboratories),
                "field_match_rate": field_match_rate,
                "report_generated": datetime.now().isoformat()
            },
            "student_statistics": {
                "valid_profiles": student_stats.valid_records,
                "invalid_profiles": student_stats.invalid_records,
                "missing_data_summary": student_stats.missing_values,
                "criteria_ranges": student_stats.value_ranges,
                "field_interests": student_stats.categorical_distributions.get("research_fields", {})
            },
            "laboratory_statistics": {
                "field_distribution": dict(lab_field_distribution),
                "average_faculty_specialties": np.mean([len(lab.faculty.specialties) for lab in laboratories]) if laboratories else 0
            },
            "compatibility_analysis": {
                "potential_matches": field_matches,
                "total_combinations": total_combinations,
                "field_match_rate": field_match_rate
            }
        }

# 使用例とテスト
def test_data_processing():
    """データ処理のテスト"""
    
    print("📊 データ処理ユーティリティテスト開始")
    
    # バリデータのテスト
    validator = DataValidator()
    
    test_profile_data = {
        "student_id": "test_student_001",
        "evaluation_criteria": {
            "research_intensity": 8.0,
            "advisor_style": 7.0,
            "team_work": 6.0,
            "workload": 7.0,
            "theory_practice": 8.0
        },
        "field_interests": [
            {
                "field": "ai_machine_learning",
                "interest_level": 9.0,
                "priority": 1
            }
        ]
    }
    
    validation_result = validator.validate_student_profile(test_profile_data)
    print(f"✅ バリデーション結果: {'有効' if validation_result.is_valid else '無効'}")
    
    if validation_result.errors:
        print(f"   エラー: {validation_result.errors}")
    if validation_result.warnings:
        print(f"   警告: {validation_result.warnings}")
    
    # クリーナーのテスト
    cleaner = DataCleaner()
    cleaned_data = cleaner.clean_student_profile_data(test_profile_data)
    print(f"✅ データクリーニング完了")
    
    # アナライザーのテスト
    analyzer = DataAnalyzer()
    
    # テスト用プロフィールの作成
    from models.schemas import StudentProfile, EvaluationCriteria, FieldInterest
    
    test_profiles = [
        StudentProfile(
            student_id="test_001",
            evaluation_criteria=EvaluationCriteria(
                research_intensity=8.0, advisor_style=7.0, team_work=6.0,
                workload=7.0, theory_practice=8.0
            ),
            field_interests=[
                FieldInterest(field=ResearchFieldEnum.AI_MACHINE_LEARNING, interest_level=9.0, priority=1)
            ]
        )
    ]
    
    stats = analyzer.analyze_student_profiles(test_profiles)
    print(f"📈 統計分析完了:")
    print(f"   総レコード数: {stats.total_records}")
    print(f"   有効レコード数: {stats.valid_records}")
    
    print("✅ データ処理ユーティリティテスト完了")

if __name__ == "__main__":
    from collections import defaultdict
    test_data_processing()