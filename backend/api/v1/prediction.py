# api/v1/prediction.py - 予測APIエンドポイント

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
import logging

from models.schemas import (
    StudentProfile, EvaluationResponse, FieldInfoResponse, 
    SystemStatus, FieldInterest, EvaluationCriteria
)
from services.lab_matching import LabMatchingService
from config.settings import settings

# ルーター設定
router = APIRouter(prefix="/prediction", tags=["prediction"])

# ロガー設定
logger = logging.getLogger(__name__)

# サービス依存性
def get_matching_service() -> LabMatchingService:
    """マッチングサービスの依存性注入"""
    return LabMatchingService()

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_lab_matching(
    student_profile: StudentProfile,
    matching_service: LabMatchingService = Depends(get_matching_service)
) -> EvaluationResponse:
    """
    学生プロフィールに基づいて研究室マッチングを実行
    
    Args:
        student_profile: 学生の評価基準と分野興味
        matching_service: マッチングサービス
    
    Returns:
        EvaluationResponse: マッチング結果
    """
    
    try:
        logger.info(f"研究室マッチング開始: 学生ID={student_profile.student_id}")
        
        # 入力検証
        if not student_profile.field_interests:
            raise HTTPException(
                status_code=400,
                detail="少なくとも1つの研究分野を選択してください"
            )
        
        # 分野IDの検証
        valid_field_ids = set(settings.research_fields.keys())
        selected_field_ids = {fi.field_id for fi in student_profile.field_interests}
        
        invalid_fields = selected_field_ids - valid_field_ids
        if invalid_fields:
            raise HTTPException(
                status_code=400,
                detail=f"無効な研究分野ID: {list(invalid_fields)}"
            )
        
        # マッチング実行
        result = matching_service.find_best_matches(student_profile)
        
        logger.info(f"マッチング完了: {len(result.results)}件の研究室を評価")
        
        return result
        
    except ValueError as e:
        logger.error(f"入力値エラー: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"マッチング処理エラー: {str(e)}")
        raise HTTPException(status_code=500, detail="マッチング処理中にエラーが発生しました")

@router.get("/fields", response_model=List[FieldInfoResponse])
async def get_research_fields() -> List[FieldInfoResponse]:
    """
    利用可能な研究分野の一覧を取得
    
    Returns:
        List[FieldInfoResponse]: 研究分野情報のリスト
    """
    
    try:
        field_responses = []
        
        for field_id, field_info in settings.research_fields.items():
            field_response = FieldInfoResponse(
                field_id=field_id,
                name=field_info["name"],
                category=field_info["category"],
                faculty=field_info["faculty"],
                difficulty=field_info["difficulty"],
                characteristics={
                    "tech_focus": field_info["tech_focus"],
                    "creativity_focus": field_info["creativity_focus"],
                    "theory_practice": field_info["theory_practice"]
                }
            )
            field_responses.append(field_response)
        
        return field_responses
        
    except Exception as e:
        logger.error(f"分野情報取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail="分野情報の取得に失敗しました")

@router.get("/fields/{field_id}", response_model=FieldInfoResponse)
async def get_field_info(field_id: str) -> FieldInfoResponse:
    """
    特定の研究分野の詳細情報を取得
    
    Args:
        field_id: 研究分野ID
    
    Returns:
        FieldInfoResponse: 研究分野の詳細情報
    """
    
    try:
        if field_id not in settings.research_fields:
            raise HTTPException(
                status_code=404,
                detail=f"研究分野が見つかりません: {field_id}"
            )
        
        field_info = settings.research_fields[field_id]
        
        return FieldInfoResponse(
            field_id=field_id,
            name=field_info["name"],
            category=field_info["category"],
            faculty=field_info["faculty"],
            difficulty=field_info["difficulty"],
            characteristics={
                "tech_focus": field_info["tech_focus"],
                "creativity_focus": field_info["creativity_focus"],
                "theory_practice": field_info["theory_practice"]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分野詳細情報取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail="分野詳細情報の取得に失敗しました")

@router.get("/fields/category/{category}", response_model=List[FieldInfoResponse])
async def get_fields_by_category(category: str) -> List[FieldInfoResponse]:
    """
    カテゴリ別の研究分野一覧を取得
    
    Args:
        category: 分野カテゴリ
    
    Returns:
        List[FieldInfoResponse]: カテゴリ内の研究分野リスト
    """
    
    try:
        if category not in settings.field_categories:
            raise HTTPException(
                status_code=404,
                detail=f"カテゴリが見つかりません: {category}"
            )
        
        category_field_ids = settings.field_categories[category]
        field_responses = []
        
        for field_id in category_field_ids:
            field_info = settings.research_fields[field_id]
            field_response = FieldInfoResponse(
                field_id=field_id,
                name=field_info["name"],
                category=field_info["category"],
                faculty=field_info["faculty"],
                difficulty=field_info["difficulty"],
                characteristics={
                    "tech_focus": field_info["tech_focus"],
                    "creativity_focus": field_info["creativity_focus"],
                    "theory_practice": field_info["theory_practice"]
                }
            )
            field_responses.append(field_response)
        
        return field_responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"カテゴリ別分野取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail="カテゴリ別分野の取得に失敗しました")

@router.post("/quick-evaluation")
async def quick_evaluation(
    field_interests: List[FieldInterest],
    matching_service: LabMatchingService = Depends(get_matching_service)
) -> Dict[str, Any]:
    """
    簡易評価（評価基準をデフォルト値で実行）
    
    Args:
        field_interests: 分野興味のリスト
        matching_service: マッチングサービス
    
    Returns:
        Dict[str, Any]: 簡易評価結果
    """
    
    try:
        if not field_interests:
            raise HTTPException(
                status_code=400,
                detail="少なくとも1つの研究分野を選択してください"
            )
        
        # デフォルト評価基準（全て中間値）
        default_criteria = EvaluationCriteria(
            research_intensity=6,
            advisor_style=6,
            team_work=6,
            workload=6,
            theory_practice=6,
            research_field_match=8,  # 分野適合性は高めに設定
            skill_development=7,
            lab_atmosphere=6,
            flexibility=6,
            publication_opportunity=6,
            interdisciplinary=6,
            communication_style=6,
            innovation_risk=6
        )
        
        # 簡易プロフィール作成
        quick_profile = StudentProfile(
            student_id="quick_eval",
            evaluation_criteria=default_criteria,
            field_interests=field_interests
        )
        
        # マッチング実行
        result = matching_service.find_best_matches(quick_profile)
        
        # 簡易結果を返却
        return {
            "total_labs": len(result.results),
            "top_3_labs": [
                {
                    "lab_name": r.lab.name,
                    "professor": r.lab.professor,
                    "score": r.compatibility.overall_score,
                    "research_area": r.lab.research_area
                }
                for r in result.results[:3]
            ],
            "avg_compatibility": result.summary.avg_compatibility,
            "selected_fields": len(field_interests)
        }
        
    except ValueError as e:
        logger.error(f"簡易評価入力値エラー: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"簡易評価エラー: {str(e)}")
        raise HTTPException(status_code=500, detail="簡易評価中にエラーが発生しました")

@router.get("/status", response_model=SystemStatus)
async def get_system_status() -> SystemStatus:
    """
    システム状態を取得
    
    Returns:
        SystemStatus: システムの現在状態
    """
    
    try:
        from datetime import datetime
        
        return SystemStatus(
            status="active",
            total_fields=len(settings.research_fields),
            total_labs=5,  # サンプルデータの数
            last_updated=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"システム状態取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail="システム状態の取得に失敗しました")

@router.get("/categories")
async def get_field_categories() -> Dict[str, List[str]]:
    """
    研究分野のカテゴリ一覧を取得
    
    Returns:
        Dict[str, List[str]]: カテゴリとその分野IDのマッピング
    """
    
    try:
        return settings.field_categories
        
    except Exception as e:
        logger.error(f"カテゴリ取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail="カテゴリ情報の取得に失敗しました")

@router.get("/evaluation-criteria")
async def get_evaluation_criteria() -> Dict[str, str]:
    """
    評価基準の一覧と説明を取得
    
    Returns:
        Dict[str, str]: 評価基準とその説明
    """
    
    try:
        criteria_descriptions = {
            "research_intensity": "研究にどれだけ集中的に取り組みたいか (1:軽い研究 〜 10:集中研究)",
            "advisor_style": "教授からの指導の受け方の好み (1:厳格指導 〜 10:自由指導)",
            "team_work": "研究での他者との協働の程度 (1:個人研究 〜 10:チーム研究)",
            "workload": "研究活動の忙しさに対する許容度 (1:軽い負荷 〜 10:重い負荷)",
            "theory_practice": "理論研究と実践的研究のバランス (1:理論重視 〜 10:実践重視)",
            "research_field_match": "自分の興味と研究室の分野の一致度 (1:広い分野 〜 10:専門特化)",
            "skill_development": "専門性と汎用性のバランス (1:専門特化 〜 10:幅広いスキル)",
            "lab_atmosphere": "研究室の全体的な雰囲気 (1:静寂集中 〜 10:活発議論)",
            "flexibility": "研究時間の自由度 (1:固定スケジュール 〜 10:柔軟スケジュール)",
            "publication_opportunity": "研究成果の論文化機会 (1:少ない機会 〜 10:豊富な機会)",
            "interdisciplinary": "他分野との連携の程度 (1:単一分野 〜 10:学際連携)",
            "communication_style": "研究室での交流スタイル (1:少人数密接 〜 10:オープン交流)",
            "innovation_risk": "新しい手法への挑戦度 (1:安全手法 〜 10:革新手法)"
        }
        
        return criteria_descriptions
        
    except Exception as e:
        logger.error(f"評価基準取得エラー: {str(e)}")
        raise HTTPException(status_code=500, detail="評価基準情報の取得に失敗しました")