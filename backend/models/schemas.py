# models/schemas.py - Pydantic スキーマ定義

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
from enum import Enum

class DifficultyLevel(str, Enum):
    """難易度レベル"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"

class FieldCategory(str, Enum):
    """分野カテゴリ"""
    TECHNOLOGY = "テクノロジー・システム"
    CREATIVE = "クリエイティブ・デザイン"
    ENTERTAINMENT = "メディア・エンターテイメント"
    HUMANITIES = "人文・社会・自然科学"

class FieldInterest(BaseModel):
    """分野別興味度モデル"""
    field_id: str = Field(..., description="研究分野ID")
    interest_level: int = Field(..., ge=1, le=10, description="興味度 (1-10)")
    experience_level: int = Field(..., ge=1, le=10, description="経験レベル (1-10)")
    importance_level: int = Field(..., ge=1, le=10, description="重要度 (1-10)")
    
    @validator('field_id')
    def validate_field_id(cls, v):
        from config.settings import settings
        if v not in settings.research_fields:
            raise ValueError(f"無効な研究分野ID: {v}")
        return v

class EvaluationCriteria(BaseModel):
    """評価基準モデル（13項目）"""
    research_intensity: int = Field(..., ge=1, le=10, description="研究強度")
    advisor_style: int = Field(..., ge=1, le=10, description="指導スタイル")
    team_work: int = Field(..., ge=1, le=10, description="チームワーク")
    workload: int = Field(..., ge=1, le=10, description="ワークロード")
    theory_practice: int = Field(..., ge=1, le=10, description="理論・実践バランス")
    research_field_match: int = Field(..., ge=1, le=10, description="研究分野適合性")
    skill_development: int = Field(..., ge=1, le=10, description="スキル開発")
    lab_atmosphere: int = Field(..., ge=1, le=10, description="研究室雰囲気")
    flexibility: int = Field(..., ge=1, le=10, description="柔軟性")
    publication_opportunity: int = Field(..., ge=1, le=10, description="論文発表機会")
    interdisciplinary: int = Field(..., ge=1, le=10, description="学際性")
    communication_style: int = Field(..., ge=1, le=10, description="コミュニケーション")
    innovation_risk: int = Field(..., ge=1, le=10, description="革新性・リスク許容度")

class StudentProfile(BaseModel):
    """学生プロフィール"""
    student_id: str = Field(..., description="学生ID")
    evaluation_criteria: EvaluationCriteria = Field(..., description="評価基準")
    field_interests: List[FieldInterest] = Field(..., description="分野別興味度リスト")
    
    @validator('field_interests')
    def validate_field_interests(cls, v):
        if len(v) == 0:
            raise ValueError("少なくとも1つの研究分野を選択してください")
        return v

class LabFeatures(BaseModel):
    """研究室特徴モデル"""
    research_intensity: float = Field(..., ge=0, le=10)
    advisor_style: float = Field(..., ge=0, le=10)
    team_work: float = Field(..., ge=0, le=10)
    workload: float = Field(..., ge=0, le=10)
    theory_practice: float = Field(..., ge=0, le=10)
    research_field_match: float = Field(..., ge=0, le=10)
    skill_development: float = Field(..., ge=0, le=10)
    lab_atmosphere: float = Field(..., ge=0, le=10)
    flexibility: float = Field(..., ge=0, le=10)
    publication_opportunity: float = Field(..., ge=0, le=10)
    interdisciplinary: float = Field(..., ge=0, le=10)
    communication_style: float = Field(..., ge=0, le=10)
    innovation_risk: float = Field(..., ge=0, le=10)

class Laboratory(BaseModel):
    """研究室モデル"""
    id: str = Field(..., description="研究室ID")
    name: str = Field(..., description="研究室名")
    professor: str = Field(..., description="教授名")
    research_area: str = Field(..., description="研究分野")
    specialization: str = Field(..., description="専門分野")
    research_fields: List[str] = Field(..., description="対応研究分野IDリスト")
    description: Optional[str] = Field(None, description="研究室説明")
    features: LabFeatures = Field(..., description="研究室特徴")

class CompatibilityScore(BaseModel):
    """適合性スコア"""
    overall_score: float = Field(..., ge=0, le=10, description="総合スコア")
    field_compatibility: float = Field(..., ge=0, le=1, description="分野適合性")
    criteria_compatibility: float = Field(..., ge=0, le=1, description="基準適合性")
    detailed_scores: Dict[str, float] = Field(..., description="詳細スコア")

class LabResult(BaseModel):
    """研究室マッチング結果"""
    lab: Laboratory = Field(..., description="研究室情報")
    compatibility: CompatibilityScore = Field(..., description="適合性スコア")
    ranking_position: int = Field(..., description="ランキング順位")
    recommendations: List[str] = Field(default=[], description="推奨事項")

class EvaluationSummary(BaseModel):
    """評価サマリー"""
    total_labs: int = Field(..., description="総研究室数")
    avg_compatibility: float = Field(..., description="平均適合度")
    best_match_score: float = Field(..., description="最高適合度")
    selected_fields_count: int = Field(..., description="選択分野数")
    field_distribution: Dict[str, int] = Field(..., description="分野分布")

class EvaluationResponse(BaseModel):
    """評価結果レスポンス"""
    results: List[LabResult] = Field(..., description="マッチング結果リスト")
    summary: EvaluationSummary = Field(..., description="評価サマリー")
    optimization_info: Dict[str, float] = Field(..., description="最適化情報")

class FieldInfoResponse(BaseModel):
    """分野情報レスポンス"""
    field_id: str = Field(..., description="分野ID")
    name: str = Field(..., description="分野名")
    category: str = Field(..., description="カテゴリ")
    faculty: List[str] = Field(..., description="担当教員リスト")
    difficulty: DifficultyLevel = Field(..., description="難易度")
    characteristics: Dict[str, int] = Field(..., description="分野特徴")

class SystemStatus(BaseModel):
    """システム状態"""
    status: str = Field(..., description="システム状態")
    total_fields: int = Field(..., description="総分野数")
    total_labs: int = Field(..., description="総研究室数")
    last_updated: Optional[str] = Field(None, description="最終更新日時")