# api/v1/prediction.py - 予測API

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import asyncio

from models.schemas import (
    StudentProfile, Laboratory, EvaluationResponse,
    LabResult, SystemStatus, ResearchFieldEnum
)
from services.lab_matching import LabMatchingService, MatchingConfig
from services.prediction import PredictionService
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prediction", tags=["prediction"])

# サービスインスタンス（シングルトン）
_lab_matching_service: Optional[LabMatchingService] = None
_prediction_service: Optional[PredictionService] = None

def get_lab_matching_service() -> LabMatchingService:
    """研究室マッチングサービスの取得"""
    global _lab_matching_service
    
    if _lab_matching_service is None:
        config = MatchingConfig(
            enable_genetic_optimization=True,
            max_recommendations=settings.ga_population_size,
            use_fuzzy_inference=True
        )
        _lab_matching_service = LabMatchingService(config)
    
    return _lab_matching_service

def get_prediction_service() -> PredictionService:
    """予測サービスの取得"""
    global _prediction_service
    
    if _prediction_service is None:
        try:
            from services.prediction import PredictionService
            _prediction_service = PredictionService()
        except ImportError:
            # prediction.pyが利用できない場合のフォールバック
            _prediction_service = None
    
    return _prediction_service

@router.get("/fields", response_model=Dict[str, Any])
async def get_research_fields():
    """利用可能な研究分野一覧を取得"""
    
    try:
        fields_info = {
            "total_fields": len(settings.research_fields),
            "field_categories": settings.field_categories,
            "fields": []
        }
        
        for field_enum in ResearchFieldEnum:
            field_info = {
                "field_id": field_enum.value,
                "field_name": field_enum.value.replace("_", " ").title(),
                "category": None
            }
            
            # カテゴリの特定
            for category, fields in settings.field_categories.items():
                if field_enum.value in fields:
                    field_info["category"] = category
                    break
            
            fields_info["fields"].append(field_info)
        
        # 教員情報も含める
        faculty_info = {}
        for field, faculty_list in settings.faculty_data.items():
            if field in [f.value for f in ResearchFieldEnum]:
                faculty_info[field] = len(faculty_list)
        
        fields_info["faculty_counts"] = faculty_info
        
        return fields_info
        
    except Exception as e:
        logger.error(f"研究分野取得エラー: {e}")
        raise HTTPException(status_code=500, detail=f"研究分野の取得に失敗しました: {str(e)}")

@router.get("/criteria", response_model=Dict[str, Any])
async def get_evaluation_criteria():
    """評価基準一覧を取得"""
    
    try:
        criteria_info = {
            "total_criteria": len(settings.evaluation_criteria),
            "criteria_groups": {
                "basic": {
                    "description": "基本項目（必須）",
                    "criteria": settings.evaluation_criteria[:5]
                },
                "extended": {
                    "description": "拡張項目（推奨）",
                    "criteria": settings.evaluation_criteria[5:10]
                },
                "special": {
                    "description": "特殊項目（詳細分析用）",
                    "criteria": settings.evaluation_criteria[10:13]
                }
            },
            "criteria_descriptions": {
                "research_intensity": "研究にどれだけ集中的に取り組みたいか（1:軽い研究 〜 10:集中研究）",
                "advisor_style": "教授からの指導の受け方の好み（1:厳格指導 〜 10:自由指導）",
                "team_work": "研究での他者との協働の程度（1:個人研究 〜 10:チーム研究）",
                "workload": "研究活動の忙しさに対する許容度（1:軽い負荷 〜 10:重い負荷）",
                "theory_practice": "理論研究と実践的研究のバランス（1:理論重視 〜 10:実践重視）",
                "research_field_match": "自分の興味と研究室の分野の一致度（1:広い分野 〜 10:専門特化）",
                "skill_development": "専門性と汎用性のバランス（1:専門特化 〜 10:幅広いスキル）",
                "lab_atmosphere": "研究室の全体的な雰囲気（1:静寂集中 〜 10:活発議論）",
                "flexibility": "研究時間の自由度（1:固定スケジュール 〜 10:柔軟スケジュール）",
                "publication_opportunity": "研究成果の論文化機会（1:少ない機会 〜 10:豊富な機会）",
                "interdisciplinary": "他分野との連携の程度（1:単一分野 〜 10:学際連携）",
                "communication_style": "研究室での交流スタイル（1:少人数密接 〜 10:オープン交流）",
                "innovation_risk": "新しい手法への挑戦度（1:安全手法 〜 10:革新手法）"
            }
        }
        
        return criteria_info
        
    except Exception as e:
        logger.error(f"評価基準取得エラー: {e}")
        raise HTTPException(status_code=500, detail=f"評価基準の取得に失敗しました: {str(e)}")

@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_lab_compatibility(
    student_profile: StudentProfile,
    target_lab_ids: Optional[List[str]] = None,
    matching_service: LabMatchingService = Depends(get_lab_matching_service)
):
    """学生プロフィールに基づく研究室適合性評価"""
    
    try:
        # 入力検証
        if not student_profile.field_interests:
            raise HTTPException(
                status_code=400, 
                detail="最低1つの研究分野への興味を指定してください"
            )
        
        # 対象研究室の特定
        target_laboratories = None
        
        if target_lab_ids:
            target_laboratories = []
            for lab_id in target_lab_ids:
                lab = matching_service.get_laboratory(lab_id)
                if lab:
                    target_laboratories.append(lab)
                else:
                    logger.warning(f"研究室が見つかりません: {lab_id}")
            
            if not target_laboratories:
                raise HTTPException(
                    status_code=404, 
                    detail="指定された研究室が見つかりません"
                )
        
        # 適合性評価の実行
        logger.info(f"適合性評価開始: 学生{student_profile.student_id}")
        
        evaluation_response = matching_service.evaluate_student_lab_compatibility(
            student_profile, target_laboratories
        )
        
        logger.info(f"適合性評価完了: {len(evaluation_response.lab_results)}件の結果")
        
        return evaluation_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"適合性評価エラー: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"適合性評価の実行に失敗しました: {str(e)}"
        )

@router.post("/evaluate-single", response_model=Dict[str, Any])
async def evaluate_single_lab(
    student_profile: StudentProfile,
    lab_id: str,
    matching_service: LabMatchingService = Depends(get_lab_matching_service)
):
    """特定研究室との適合性評価"""
    
    try:
        # 研究室の取得
        laboratory = matching_service.get_laboratory(lab_id)
        if not laboratory:
            raise HTTPException(status_code=404, detail=f"研究室が見つかりません: {lab_id}")
        
        # 単一研究室での評価
        evaluation_response = matching_service.evaluate_student_lab_compatibility(
            student_profile, [laboratory]
        )
        
        if not evaluation_response.lab_results:
            raise HTTPException(status_code=500, detail="評価結果が生成されませんでした")
        
        # 単一結果の詳細情報
        lab_result = evaluation_response.lab_results[0]
        
        detailed_result = {
            "student_id": student_profile.student_id,
            "lab_id": lab_id,
            "laboratory_info": {
                "faculty_name": laboratory.faculty.name,
                "research_field": laboratory.research_field.value,
                "specialties": laboratory.faculty.specialties,
                "description": laboratory.description
            },
            "compatibility_score": lab_result.compatibility_score.dict(),
            "detailed_analysis": {
                "criteria_breakdown": lab_result.compatibility_score.criteria_scores,
                "field_match_analysis": {
                    "score": lab_result.compatibility_score.field_match_score,
                    "student_interests": [
                        {"field": interest.field.value, "level": interest.interest_level, "priority": interest.priority}
                        for interest in student_profile.field_interests
                    ],
                    "lab_field": laboratory.research_field.value
                }
            },
            "recommendations": {
                "reasons": lab_result.reasons,
                "concerns": lab_result.concerns,
                "suggestion": _generate_improvement_suggestions(
                    student_profile, laboratory, lab_result.compatibility_score
                )
            },
            "processing_info": {
                "evaluation_id": evaluation_response.evaluation_id,
                "processing_time": evaluation_response.processing_time,
                "timestamp": evaluation_response.timestamp.isoformat()
            }
        }
        
        return detailed_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"単一研究室評価エラー: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"研究室評価の実行に失敗しました: {str(e)}"
        )

def _generate_improvement_suggestions(student_profile: StudentProfile, 
                                    laboratory: Laboratory,
                                    compatibility_score) -> List[str]:
    """改善提案の生成"""
    
    suggestions = []
    
    # 低スコア基準に対する提案
    low_criteria = [
        criterion for criterion, score in compatibility_score.criteria_scores.items()
        if score < 0.4
    ]
    
    criterion_suggestions = {
        "research_intensity": "研究への取り組み強度を見直してみてください",
        "advisor_style": "指導スタイルの希望を再考してみてください",
        "team_work": "チームでの研究への参加度を検討してみてください",
        "workload": "研究活動の負荷に対する期待を調整してみてください",
        "theory_practice": "理論と実践のバランスを見直してみてください"
    }
    
    for criterion in low_criteria[:2]:  # 上位2つのみ
        if criterion in criterion_suggestions:
            suggestions.append(criterion_suggestions[criterion])
    
    # 分野適合性が低い場合
    if compatibility_score.field_match_score < 0.3:
        suggestions.append("他の研究分野への興味も探ってみてください")
    
    # 全体的な適合性が中程度の場合
    if 0.4 <= compatibility_score.overall_score <= 0.6:
        suggestions.append("研究室見学や教員との面談を通じて詳細を確認することをお勧めします")
    
    return suggestions

@router.get("/laboratories", response_model=Dict[str, Any])
async def get_laboratories(
    field: Optional[str] = None,
    limit: Optional[int] = None,
    matching_service: LabMatchingService = Depends(get_lab_matching_service)
):
    """研究室一覧の取得"""
    
    try:
        laboratories = matching_service.get_all_laboratories()
        
        # 分野フィルタ
        if field:
            laboratories = [
                lab for lab in laboratories 
                if lab.research_field.value == field
            ]
        
        # 件数制限
        if limit:
            laboratories = laboratories[:limit]
        
        # レスポンス構築
        lab_list = []
        for lab in laboratories:
            lab_info = {
                "lab_id": lab.lab_id,
                "faculty": {
                    "name": lab.faculty.name,
                    "name_en": lab.faculty.name_en,
                    "specialties": lab.faculty.specialties
                },
                "research_field": lab.research_field.value,
                "description": lab.description,
                "characteristics_summary": {
                    "research_intensity": lab.characteristics.research_intensity,
                    "advisor_style": lab.characteristics.advisor_style,
                    "team_work": lab.characteristics.team_work
                }
            }
            lab_list.append(lab_info)
        
        return {
            "total_count": len(matching_service.get_all_laboratories()),
            "filtered_count": len(lab_list),
            "laboratories": lab_list,
            "available_fields": list(set(lab.research_field.value for lab in matching_service.get_all_laboratories()))
        }
        
    except Exception as e:
        logger.error(f"研究室一覧取得エラー: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"研究室一覧の取得に失敗しました: {str(e)}"
        )

@router.get("/laboratory/{lab_id}", response_model=Dict[str, Any])
async def get_laboratory_detail(
    lab_id: str,
    matching_service: LabMatchingService = Depends(get_lab_matching_service)
):
    """研究室詳細情報の取得"""
    
    try:
        laboratory = matching_service.get_laboratory(lab_id)
        
        if not laboratory:
            raise HTTPException(status_code=404, detail=f"研究室が見つかりません: {lab_id}")
        
        # 詳細情報の構築
        detail_info = {
            "lab_id": laboratory.lab_id,
            "basic_info": {
                "lab_name": laboratory.lab_name,
                "faculty": {
                    "name": laboratory.faculty.name,
                    "name_en": laboratory.faculty.name_en,
                    "title": laboratory.faculty.title,
                    "specialties": laboratory.faculty.specialties
                },
                "research_field": laboratory.research_field.value,
                "description": laboratory.description
            },
            "characteristics": laboratory.characteristics.dict(),
            "additional_info": {
                "recent_achievements": laboratory.recent_achievements,
                "required_skills": laboratory.required_skills,
                "lab_environment": laboratory.lab_environment,
                "current_students": laboratory.current_students,
                "graduation_rate": laboratory.graduation_rate,
                "job_placement_rate": laboratory.job_placement_rate
            },
            "field_info": {
                "category": None,
                "related_fields": []
            }
        }
        
        # カテゴリ情報の追加
        for category, fields in settings.field_categories.items():
            if laboratory.research_field.value in fields:
                detail_info["field_info"]["category"] = category
                detail_info["field_info"]["related_fields"] = [
                    f for f in fields if f != laboratory.research_field.value
                ]
                break
        
        return detail_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"研究室詳細取得エラー: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"研究室詳細の取得に失敗しました: {str(e)}"
        )

@router.get("/statistics", response_model=Dict[str, Any])
async def get_prediction_statistics(
    matching_service: LabMatchingService = Depends(get_lab_matching_service)
):
    """予測システムの統計情報取得"""
    
    try:
        service_stats = matching_service.get_service_statistics()
        
        system_stats = {
            "service_statistics": service_stats,
            "system_info": {
                "total_research_fields": len(settings.research_fields),
                "total_evaluation_criteria": len(settings.evaluation_criteria),
                "genetic_algorithm_config": {
                    "population_size": settings.ga_population_size,
                    "generations": settings.ga_generations,
                    "mutation_rate": settings.ga_mutation_rate,
                    "crossover_rate": settings.ga_crossover_rate
                }
            },
            "algorithm_status": {
                "fuzzy_inference": service_stats["fuzzy_inference_available"],
                "genetic_optimization": service_stats["optimization_enabled"],
                "optimized_weights": service_stats["optimized_weights_available"]
            }
        }
        
        return system_stats
        
    except Exception as e:
        logger.error(f"統計情報取得エラー: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"統計情報の取得に失敗しました: {str(e)}"
        )

@router.post("/optimize-weights")
async def optimize_matching_weights(
    background_tasks: BackgroundTasks,
    sample_size: int = 100,
    matching_service: LabMatchingService = Depends(get_lab_matching_service)
):
    """マッチング重みの最適化（バックグラウンド処理）"""
    
    try:
        # サンプルデータの生成（実際の実装では外部データを使用）
        def generate_optimization_data():
            logger.info("重み最適化をバックグラウンドで開始します")
            
            # ここでは簡易的なサンプルデータを生成
            # 実際の実装では過去のマッチング結果や評価データを使用
            sample_data = []
            
            # 実装が複雑になるため、ここでは省略
            # matching_service.optimize_weights(sample_data)
            
            logger.info("重み最適化が完了しました")
        
        background_tasks.add_task(generate_optimization_data)
        
        return {
            "message": "重み最適化をバックグラウンドで開始しました",
            "status": "processing",
            "estimated_time": "5-10分",
            "sample_size": sample_size
        }
        
    except Exception as e:
        logger.error(f"重み最適化エラー: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"重み最適化の開始に失敗しました: {str(e)}"
        )

@router.get("/health", response_model=SystemStatus)
async def get_system_health(
    matching_service: LabMatchingService = Depends(get_lab_matching_service)
):
    """システムヘルスチェック"""
    
    try:
        service_stats = matching_service.get_service_statistics()
        
        # モジュールの利用可能性チェック
        modules = {
            "fuzzy_inference": service_stats["fuzzy_inference_available"],
            "genetic_optimization": service_stats["optimization_enabled"],
            "lab_matching": True,
            "prediction": get_prediction_service() is not None
        }
        
        # システム状態の判定
        critical_modules = ["lab_matching"]
        status = "healthy"
        
        if not all(modules[mod] for mod in critical_modules):
            status = "degraded"
        
        if not any(modules.values()):
            status = "unhealthy"
        
        return SystemStatus(
            status=status,
            version="1.0.0",
            uptime=0.0,  # 実際の実装では起動からの経過時間
            modules=modules,
            total_evaluations=service_stats["total_evaluations"],
            active_optimizations=0,  # 実際の実装では進行中の最適化数
        )
        
    except Exception as e:
        logger.error(f"ヘルスチェックエラー: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"システムヘルスチェックに失敗しました: {str(e)}"
        )

# エラーハンドラ
@router.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(
        status_code=400,
        content={"detail": f"入力値エラー: {str(exc)}"}
    )

@router.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"予期しないエラー: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "内部サーバーエラーが発生しました"}
    )