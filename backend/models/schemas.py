# models/schemas.py
"""
データスキーマ定義（パターンA版）
遺伝的アルゴリズム関連のスキーマを除去
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


# ==================== 基本スキーマ ====================

class StudentProfileBase(BaseModel):
    """学生プロファイル基本スキーマ"""
    
    # 基本12項目（1-10スケール）
    research_intensity: float = Field(..., ge=1, le=10, description="研究強度")
    advisor_style: float = Field(..., ge=1, le=10, description="指導スタイル")
    team_work: float = Field(..., ge=1, le=10, description="チームワーク")
    workload: float = Field(..., ge=1, le=10, description="ワークロード")
    theory_practice: float = Field(..., ge=1, le=10, description="理論・実践バランス")
    skill_development: float = Field(..., ge=1, le=10, description="スキル開発")
    lab_atmosphere: float = Field(..., ge=1, le=10, description="研究室雰囲気")
    flexibility: float = Field(..., ge=1, le=10, description="柔軟性")
    publication_opportunity: float = Field(..., ge=1, le=10, description="論文発表機会")
    interdisciplinary: float = Field(..., ge=1, le=10, description="学際性")
    communication_style: float = Field(..., ge=1, le=10, description="コミュニケーション")
    innovation_focus: float = Field(..., ge=1, le=10, description="革新性重視")
    
    # 優先度（1-10スケール）
    research_intensity_priority: Optional[float] = Field(5.0, ge=1, le=10)
    advisor_style_priority: Optional[float] = Field(5.0, ge=1, le=10)
    team_work_priority: Optional[float] = Field(5.0, ge=1, le=10)
    workload_priority: Optional[float] = Field(5.0, ge=1, le=10)
    theory_practice_priority: Optional[float] = Field(5.0, ge=1, le=10)
    skill_development_priority: Optional[float] = Field(5.0, ge=1, le=10)
    lab_atmosphere_priority: Optional[float] = Field(5.0, ge=1, le=10)
    flexibility_priority: Optional[float] = Field(5.0, ge=1, le=10)
    publication_opportunity_priority: Optional[float] = Field(5.0, ge=1, le=10)
    interdisciplinary_priority: Optional[float] = Field(5.0, ge=1, le=10)
    communication_style_priority: Optional[float] = Field(5.0, ge=1, le=10)
    innovation_focus_priority: Optional[float] = Field(5.0, ge=1, le=10)
    
    # 分野重視度
    research_field_match: float = Field(..., ge=1, le=10, description="分野重視度")
    
    # 分野興味度
    field_interests: Dict[str, float] = Field(..., description="分野興味度")
    
    @validator("field_interests")
    def validate_field_interests(cls, v):
        """分野興味度の検証"""
        for field_id, interest in v.items():
            if not (1 <= interest <= 10):
                raise ValueError(f"Interest level for {field_id} must be between 1 and 10")
        return v


class StudentProfile(StudentProfileBase):
    """学生プロファイル完全版"""
    
    student_id: Optional[str] = Field(None, description="学生ID")
    name: Optional[str] = Field(None, description="学生名")
    grade: Optional[int] = Field(None, ge=1, le=4, description="学年")
    created_at: Optional[datetime] = Field(None, description="作成日時")


class LaboratoryBase(BaseModel):
    """研究室基本スキーマ"""
    
    lab_id: str = Field(..., description="研究室ID")
    name: str = Field(..., description="研究室名")
    professor: str = Field(..., description="教授名")
    field_id: str = Field(..., description="分野ID")
    description: Optional[str] = Field(None, description="研究室説明")
    
    # 基本12項目
    research_intensity: float = Field(..., ge=0, le=10)
    advisor_style: float = Field(..., ge=0, le=10)
    team_work: float = Field(..., ge=0, le=10)
    workload: float = Field(..., ge=0, le=10)
    theory_practice: float = Field(..., ge=0, le=10)
    skill_development: float = Field(..., ge=0, le=10)
    lab_atmosphere: float = Field(..., ge=0, le=10)
    flexibility: float = Field(..., ge=0, le=10)
    publication_opportunity: float = Field(..., ge=0, le=10)
    interdisciplinary: float = Field(..., ge=0, le=10)
    communication_style: float = Field(..., ge=0, le=10)
    innovation_focus: float = Field(..., ge=0, le=10)
    
    # 追加情報
    students_count: Optional[int] = Field(None, ge=0, description="学生数")
    equipment: Optional[str] = Field(None, description="設備")
    funding: Optional[str] = Field(None, description="資金状況")


class Laboratory(LaboratoryBase):
    """研究室完全版"""
    
    field_name: Optional[str] = Field(None, description="分野名")
    created_at: Optional[datetime] = Field(None, description="作成日時")
    updated_at: Optional[datetime] = Field(None, description="更新日時")


# ==================== 評価結果スキーマ ====================

class CompatibilityScore(BaseModel):
    """適合度スコア"""
    
    overall_compatibility: float = Field(..., ge=0, le=1, description="総合適合度")
    basic_score: float = Field(..., ge=0, le=1, description="基本項目スコア")
    field_score: float = Field(..., ge=0, le=1, description="分野スコア")
    field_weight_alpha: float = Field(..., ge=0, le=1, description="分野比重")
    basic_weight_beta: float = Field(..., ge=0, le=1, description="基本項目比重")


class CriteriaScore(BaseModel):
    """項目別スコア"""
    
    criterion: str = Field(..., description="評価項目名")
    score: float = Field(..., ge=0, le=1, description="スコア")
    description: Optional[str] = Field(None, description="説明")


class FieldMatchDetail(BaseModel):
    """分野マッチ詳細"""
    
    match_type: str = Field(..., description="マッチタイプ (exact/category/none)")
    lab_field: str = Field(..., description="研究室分野ID")
    lab_field_name: Optional[str] = Field(None, description="研究室分野名")
    message: str = Field(..., description="説明メッセージ")
    interest_level: Optional[float] = Field(None, description="興味レベル")
    related_count: Optional[int] = Field(None, description="関連分野数")


class LabEvaluationResult(BaseModel):
    """研究室評価結果"""
    
    lab_id: str
    lab_name: str
    professor: str
    field_id: str
    field_name: str
    
    # スコア
    overall_compatibility: float
    basic_score: float
    field_score: float
    field_weight: float
    basic_weight: float
    
    # 詳細
    criteria_scores: Dict[str, float]
    field_detail: Dict[str, Any]
    tree_layers: List[str]
    
    # 説明
    explanation: str
    recommendation: str
    
    # 研究室情報
    students_count: Optional[int] = None
    equipment: Optional[str] = None
    funding: Optional[str] = None


class EvaluationResponse(BaseModel):
    """評価レスポンス"""
    
    student_profile: Dict[str, Any]
    evaluation_results: List[LabEvaluationResult]
    total_labs_evaluated: int
    evaluation_timestamp: float
    system_info: Dict[str, Any]


class ExplanationResponse(BaseModel):
    """説明レスポンス"""
    
    lab_id: str
    lab_name: str
    overall_compatibility: float
    recommendation: str
    explanation: str
    
    score_breakdown: Dict[str, float]
    strengths: List[CriteriaScore]
    concerns: List[CriteriaScore]
    field_analysis: Dict[str, Any]
    decision_tree_layers: List[str]


# ==================== システム情報スキーマ ====================

class SystemInfo(BaseModel):
    """システム情報"""
    
    version: str
    pattern: str
    status: str
    uptime_seconds: float
    
    modules: Dict[str, bool]
    database: Dict[str, Any]
    features: Dict[str, bool]


class HealthResponse(BaseModel):
    """ヘルスチェックレスポンス"""
    
    status: str
    version: str
    pattern: str
    timestamp: float
    uptime_seconds: float
    system_initialized: bool
    
    modules: Dict[str, bool]
    database: Dict[str, Any]
    features: Dict[str, bool]


# ==================== 評価基準・分野スキーマ ====================

class CriterionInfo(BaseModel):
    """評価基準情報"""
    
    id: str
    name: str
    description: str
    range: str
    importance: str


class CriteriaResponse(BaseModel):
    """評価基準レスポンス"""
    
    criteria: List[CriterionInfo]
    total_count: int
    basic_count: int
    has_field_match: bool


class FieldInfo(BaseModel):
    """分野情報"""
    
    id: str
    name: str


class FieldsResponse(BaseModel):
    """分野レスポンス"""
    
    fields: List[FieldInfo]
    total_count: int


class LabsResponse(BaseModel):
    """研究室一覧レスポンス"""
    
    labs: List[Laboratory]
    total_count: int
    last_updated: float


# ==================== エラーレスポンス ====================

class ErrorResponse(BaseModel):
    """エラーレスポンス"""
    
    error: str
    detail: str
    status_code: int
    timestamp: float


# ==================== バリデーションヘルパー ====================

def validate_student_profile(profile: Dict[str, Any]) -> bool:
    """学生プロファイルの簡易検証"""
    
    required_fields = [
        "research_field_match",
        "field_interests"
    ]
    
    for field in required_fields:
        if field not in profile:
            return False
    
    return True


def normalize_profile_values(profile: Dict[str, Any]) -> Dict[str, Any]:
    """プロファイル値の正規化"""
    
    normalized = profile.copy()
    
    # 1-10スケールを0-1に正規化（必要な場合）
    for key, value in normalized.items():
        if isinstance(value, (int, float)) and value > 1:
            # 優先度やfield_interestsは維持
            if not key.endswith("_priority") and key not in ["research_field_match"]:
                if key in profile.get("field_interests", {}):
                    continue
                # その他の項目は正規化しない（1-10のまま維持）
    
    return normalized


# ==================== 使用例 ====================

if __name__ == "__main__":
    # 学生プロファイル例
    student = StudentProfile(
        student_id="S001",
        name="山田太郎",
        grade=3,
        
        research_intensity=9,
        advisor_style=7,
        team_work=5,
        workload=8,
        theory_practice=6,
        skill_development=7,
        lab_atmosphere=6,
        flexibility=5,
        publication_opportunity=9,
        interdisciplinary=4,
        communication_style=6,
        innovation_focus=8,
        
        research_intensity_priority=10,
        publication_opportunity_priority=10,
        
        research_field_match=7,
        
        field_interests={
            "ai_ml": 10,
            "image_processing": 7
        },
        
        created_at=datetime.now()
    )
    
    print("✅ 学生プロファイル作成成功")
    print(f"学生ID: {student.student_id}")
    print(f"研究強度: {student.research_intensity}")
    print(f"分野重視度: {student.research_field_match}")
    print(f"興味分野: {list(student.field_interests.keys())}")