# models/schemas.py - Pydantic スキーマ定義

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum

class ResearchFieldEnum(str, Enum):
    """研究分野列挙型"""
    AI_MACHINE_LEARNING = "ai_machine_learning"
    IMAGE_VIDEO_PROCESSING = "image_video_processing"
    NETWORK_SECURITY = "network_security"
    DATABASE_INFORMATION_SYSTEMS = "database_information_systems"
    EMBEDDED_IOT = "embedded_iot"
    WEB_DESIGN_UI_UX = "web_design_ui_ux"
    DESIGN_VISUAL_EXPRESSION = "design_visual_expression"
    VIDEO_ANIMATION = "video_animation"
    COMPUTER_MUSIC_SOUND_ART = "computer_music_sound_art"
    GAME_DEVELOPMENT_ESPORTS = "game_development_esports"
    VR_AR_MEDIA_ART = "vr_ar_media_art"

class EvaluationCriteria(BaseModel):
    """評価基準（13項目）"""
    
    # 基本項目（5項目）
    research_intensity: float = Field(..., ge=1, le=10, description="研究強度（1-10）")
    advisor_style: float = Field(..., ge=1, le=10, description="指導スタイル（1-10）")
    team_work: float = Field(..., ge=1, le=10, description="チームワーク（1-10）")
    workload: float = Field(..., ge=1, le=10, description="ワークロード（1-10）")
    theory_practice: float = Field(..., ge=1, le=10, description="理論・実践バランス（1-10）")
    
    # 拡張項目（5項目）
    research_field_match: Optional[float] = Field(None, ge=1, le=10, description="研究分野適合性（1-10）")
    skill_development: Optional[float] = Field(None, ge=1, le=10, description="スキル開発（1-10）")
    lab_atmosphere: Optional[float] = Field(None, ge=1, le=10, description="研究室雰囲気（1-10）")
    flexibility: Optional[float] = Field(None, ge=1, le=10, description="柔軟性（1-10）")
    publication_opportunity: Optional[float] = Field(None, ge=1, le=10, description="論文発表機会（1-10）")
    
    # 特殊項目（3項目）
    interdisciplinary: Optional[float] = Field(None, ge=1, le=10, description="学際性（1-10）")
    communication_style: Optional[float] = Field(None, ge=1, le=10, description="コミュニケーション（1-10）")
    innovation_risk: Optional[float] = Field(None, ge=1, le=10, description="革新性・リスク許容度（1-10）")
    
    @validator('research_intensity', 'advisor_style', 'team_work', 'workload', 'theory_practice')
    def validate_required_criteria(cls, v):
        if v is None:
            raise ValueError('基本評価項目は必須です')
        return v

class FieldInterest(BaseModel):
    """研究分野への興味"""
    field: ResearchFieldEnum = Field(..., description="研究分野")
    interest_level: float = Field(..., ge=1, le=10, description="興味レベル（1-10）")
    priority: int = Field(..., ge=1, description="優先順位")
    
    @validator('interest_level')
    def validate_interest_level(cls, v):
        if not 1 <= v <= 10:
            raise ValueError('興味レベルは1-10の範囲で指定してください')
        return v

class StudentProfile(BaseModel):
    """学生プロフィール"""
    student_id: str = Field(..., description="学生ID")
    evaluation_criteria: EvaluationCriteria = Field(..., description="評価基準")
    field_interests: List[FieldInterest] = Field(..., description="研究分野への興味")
    
    # 追加情報（オプション）
    grade: Optional[int] = Field(None, ge=1, le=4, description="学年")
    gpa: Optional[float] = Field(None, ge=0.0, le=4.0, description="GPA")
    preferred_lab_size: Optional[str] = Field(None, description="希望研究室サイズ")
    time_availability: Optional[float] = Field(None, ge=1, le=10, description="時間的余裕")
    
    # メタデータ
    created_at: datetime = Field(default_factory=datetime.now, description="作成日時")
    updated_at: Optional[datetime] = Field(None, description="更新日時")
    
    @validator('field_interests')
    def validate_field_interests(cls, v):
        if not v:
            raise ValueError('最低1つの研究分野への興味を指定してください')
        
        # 優先順位の重複チェック
        priorities = [interest.priority for interest in v]
        if len(priorities) != len(set(priorities)):
            raise ValueError('優先順位に重複があります')
        
        return v

class Faculty(BaseModel):
    """教員情報"""
    name: str = Field(..., description="教員名")
    name_en: Optional[str] = Field(None, description="英語名")
    title: Optional[str] = Field(None, description="役職")
    specialties: List[str] = Field(..., description="専門分野")
    lab_capacity: Optional[int] = Field(None, description="研究室定員")
    research_style: Optional[str] = Field(None, description="研究スタイル")

class Laboratory(BaseModel):
    """研究室情報"""
    lab_id: str = Field(..., description="研究室ID")
    faculty: Faculty = Field(..., description="指導教員")
    research_field: ResearchFieldEnum = Field(..., description="研究分野")
    
    # 研究室特性（evaluation_criteriaに対応）
    characteristics: EvaluationCriteria = Field(..., description="研究室特性")
    
    # 追加情報
    lab_name: Optional[str] = Field(None, description="研究室名")
    description: Optional[str] = Field(None, description="研究室説明")
    recent_achievements: Optional[List[str]] = Field(None, description="最近の成果")
    required_skills: Optional[List[str]] = Field(None, description="必要スキル")
    lab_environment: Optional[str] = Field(None, description="研究環境")
    
    # 統計情報
    current_students: Optional[int] = Field(None, description="現在の学生数")
    graduation_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="卒業率")
    job_placement_rate: Optional[float] = Field(None, ge=0.0, le=1.0, description="就職率")

class CompatibilityScore(BaseModel):
    """適合性スコア"""
    overall_score: float = Field(..., ge=0.0, le=1.0, description="総合適合性スコア")
    criteria_scores: Dict[str, float] = Field(..., description="各基準の適合性スコア")
    field_match_score: float = Field(..., ge=0.0, le=1.0, description="分野適合性スコア")
    confidence: float = Field(..., ge=0.0, le=1.0, description="信頼度")

class LabResult(BaseModel):
    """研究室マッチング結果"""
    laboratory: Laboratory = Field(..., description="研究室情報")
    compatibility_score: CompatibilityScore = Field(..., description="適合性スコア")
    ranking: int = Field(..., description="ランキング")
    reasons: List[str] = Field(..., description="推薦理由")
    concerns: Optional[List[str]] = Field(None, description="懸念点")

class EvaluationResponse(BaseModel):
    """評価レスポンス"""
    student_profile: StudentProfile = Field(..., description="学生プロフィール")
    lab_results: List[LabResult] = Field(..., description="研究室マッチング結果")
    
    # 処理情報
    processing_time: float = Field(..., description="処理時間（秒）")
    algorithm_version: str = Field(..., description="アルゴリズムバージョン")
    total_labs_evaluated: int = Field(..., description="評価対象研究室数")
    
    # 統計情報
    score_distribution: Dict[str, float] = Field(..., description="スコア分布統計")
    recommendation_confidence: float = Field(..., ge=0.0, le=1.0, description="推薦信頼度")
    
    # メタデータ
    evaluation_id: str = Field(..., description="評価ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="評価日時")

class OptimizationRequest(BaseModel):
    """最適化リクエスト"""
    student_profiles: List[StudentProfile] = Field(..., description="学生プロフィール群")
    target_labs: List[Laboratory] = Field(..., description="対象研究室群")
    
    # 最適化パラメータ
    population_size: Optional[int] = Field(30, ge=10, le=100, description="集団サイズ")
    generations: Optional[int] = Field(50, ge=10, le=200, description="世代数")
    mutation_rate: Optional[float] = Field(0.1, ge=0.01, le=0.5, description="変異率")
    crossover_rate: Optional[float] = Field(0.8, ge=0.1, le=1.0, description="交叉率")
    
    # 実行設定
    timeout_seconds: Optional[int] = Field(300, ge=30, le=1800, description="タイムアウト（秒）")
    verbose: Optional[bool] = Field(False, description="詳細ログ出力")

class OptimizationResult(BaseModel):
    """最適化結果"""
    request_id: str = Field(..., description="リクエストID")
    best_weights: Dict[str, float] = Field(..., description="最適重み")
    best_fitness: float = Field(..., description="最適適応度")
    
    # 進化過程
    generation_history: List[Dict[str, Any]] = Field(..., description="世代別履歴")
    convergence_generation: int = Field(..., description="収束世代")
    
    # 実行情報
    execution_time: float = Field(..., description="実行時間（秒）")
    total_evaluations: int = Field(..., description="総評価回数")
    success: bool = Field(..., description="成功フラグ")
    
    # メタデータ
    algorithm_config: Dict[str, Any] = Field(..., description="アルゴリズム設定")
    timestamp: datetime = Field(default_factory=datetime.now, description="実行日時")

class SystemStatus(BaseModel):
    """システム状態"""
    status: str = Field(..., description="システム状態")
    version: str = Field(..., description="バージョン")
    uptime: float = Field(..., description="稼働時間（秒）")
    
    # モジュール状態
    modules: Dict[str, bool] = Field(..., description="モジュール利用可能性")
    
    # 統計情報
    total_evaluations: int = Field(..., description="累計評価回数")
    active_optimizations: int = Field(..., description="実行中最適化数")
    
    # リソース情報
    memory_usage: Optional[float] = Field(None, description="メモリ使用量")
    cpu_usage: Optional[float] = Field(None, description="CPU使用率")

# バリデーション関数
def validate_student_profile(profile: StudentProfile) -> List[str]:
    """学生プロフィールの詳細バリデーション"""
    
    errors = []
    
    # 基本評価基準の完全性チェック
    required_criteria = ['research_intensity', 'advisor_style', 'team_work', 'workload', 'theory_practice']
    criteria_dict = profile.evaluation_criteria.dict()
    
    for criterion in required_criteria:
        value = criteria_dict.get(criterion)
        if value is None or not (1 <= value <= 10):
            errors.append(f"基本評価基準 '{criterion}' が無効です")
    
    # 研究分野興味の妥当性チェック
    if not profile.field_interests:
        errors.append("最低1つの研究分野への興味が必要です")
    
    # 興味レベルの分布チェック
    interest_levels = [interest.interest_level for interest in profile.field_interests]
    if interest_levels and max(interest_levels) < 5:
        errors.append("最低1つの分野に5以上の興味レベルが必要です")
    
    return errors

def validate_laboratory(lab: Laboratory) -> List[str]:
    """研究室情報の詳細バリデーション"""
    
    errors = []
    
    # 特性値の妥当性チェック
    characteristics_dict = lab.characteristics.dict()
    for key, value in characteristics_dict.items():
        if value is not None and not (1 <= value <= 10):
            errors.append(f"研究室特性 '{key}' が範囲外です: {value}")
    
    # 教員情報の完全性チェック
    if not lab.faculty.name:
        errors.append("教員名が必要です")
    
    if not lab.faculty.specialties:
        errors.append("教員の専門分野が必要です")
    
    return errors

# エクスポート用リスト
__all__ = [
    'ResearchFieldEnum',
    'EvaluationCriteria',
    'FieldInterest', 
    'StudentProfile',
    'Faculty',
    'Laboratory',
    'CompatibilityScore',
    'LabResult',
    'EvaluationResponse',
    'OptimizationRequest',
    'OptimizationResult',
    'SystemStatus',
    'validate_student_profile',
    'validate_laboratory'
]