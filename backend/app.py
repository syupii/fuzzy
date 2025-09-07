# app.py - FastAPI メインアプリケーション

import logging
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from api.v1.prediction import router as prediction_router
from config.settings import settings

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションライフサイクル管理"""
    
    # 起動時の処理
    logger.info("🚀 研究室選択支援システム起動中...")
    logger.info(f"📊 対応研究分野数: {len(settings.research_fields)}")
    logger.info(f"📋 評価基準数: {len(settings.evaluation_criteria)}")
    logger.info(f"🧬 遺伝的アルゴリズム設定: 集団{settings.ga_population_size}, 世代{settings.ga_generations}")
    
    yield
    
    # 終了時の処理
    logger.info("🛑 研究室選択支援システム終了")

# FastAPIアプリケーション作成
app = FastAPI(
    title="道都大学情報メディア学部 研究室選択支援システム",
    description="""
    ## 🎯 遺伝的アルゴリズムを用いたファジィ決定木による研究室選択支援システム
    
    ### 主な機能
    - **27研究分野対応**: 詳細な分野分類による精密マッチング
    - **13項目評価**: 多面的な学生プロフィール分析
    - **ファジィ推論**: 曖昧な評価を効果的に処理
    - **遺伝的アルゴリズム**: 最適な重み組み合わせの自動探索
    
    ### API使用方法
    1. `/prediction/fields` で利用可能な研究分野を確認
    2. 学生プロフィールを作成（分野興味 + 評価基準）
    3. `/prediction/evaluate` でマッチング実行
    4. 結果の適合度スコアと推奨事項を確認
    
    ### 対応分野カテゴリ
    - **テクノロジー・システム分野** (12分野)
    - **クリエイティブ・デザイン分野** (5分野)
    - **メディア・エンターテイメント分野** (3分野)
    - **人文・社会・自然科学分野** (7分野)
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に制限する
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# APIルーター登録
app.include_router(prediction_router, prefix="/api/v1")

# グローバル例外ハンドラー
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """グローバル例外ハンドラー"""
    logger.error(f"予期しないエラー: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "内部サーバーエラーが発生しました"}
    )

# ルートエンドポイント
@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "message": "道都大学情報メディア学部 研究室選択支援システム",
        "version": "2.0.0",
        "features": [
            "27研究分野対応",
            "ファジィ決定木",
            "遺伝的アルゴリズム最適化",
            "多面的評価システム"
        ],
        "docs": "/docs",
        "api_base": "/api/v1"
    }

@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    return {
        "status": "healthy",
        "system": "active",
        "fields_loaded": len(settings.research_fields),
        "criteria_loaded": len(settings.evaluation_criteria)
    }

@app.get("/api/v1/system-info")
async def get_system_info():
    """システム情報エンドポイント"""
    return {
        "research_fields": {
            "total": len(settings.research_fields),
            "categories": list(settings.field_categories.keys()),
            "category_counts": {
                category: len(fields) 
                for category, fields in settings.field_categories.items()
            }
        },
        "evaluation_criteria": {
            "total": len(settings.evaluation_criteria),
            "criteria": settings.evaluation_criteria
        },
        "algorithm_config": {
            "genetic_algorithm": {
                "population_size": settings.ga_population_size,
                "generations": settings.ga_generations,
                "mutation_rate": settings.ga_mutation_rate,
                "crossover_rate": settings.ga_crossover_rate
            },
            "fuzzy_decision_tree": {
                "max_depth": settings.max_tree_depth,
                "min_samples_split": settings.min_samples_split
            }
        }
    }

# 開発用テストエンドポイント
@app.post("/api/v1/test/sample-evaluation")
async def test_sample_evaluation():
    """テスト用サンプル評価"""
    
    from models.schemas import StudentProfile, EvaluationCriteria, FieldInterest
    from services.lab_matching import LabMatchingService
    
    try:
        # サンプル学生プロフィール
        sample_student = StudentProfile(
            student_id="test_student_001",
            evaluation_criteria=EvaluationCriteria(
                research_intensity=8,
                advisor_style=7,
                team_work=6,
                workload=7,
                theory_practice=8,
                research_field_match=9,
                skill_development=8,
                lab_atmosphere=7,
                flexibility=6,
                publication_opportunity=8,
                interdisciplinary=6,
                communication_style=7,
                innovation_risk=8
            ),
            field_interests=[
                FieldInterest(
                    field_id="ai_machine_learning",
                    interest_level=9,
                    experience_level=6,
                    importance_level=10
                ),
                FieldInterest(
                    field_id="game_programming",
                    interest_level=7,
                    experience_level=5,
                    importance_level=7
                ),
                FieldInterest(
                    field_id="web_design_branding",
                    interest_level=6,
                    experience_level=4,
                    importance_level=5
                )
            ]
        )
        
        # マッチング実行
        matching_service = LabMatchingService()
        result = matching_service.find_best_matches(sample_student)
        
        # 簡略化した結果を返却
        return {
            "test_status": "success",
            "student_profile": {
                "selected_fields": len(sample_student.field_interests),
                "field_names": [
                    settings.research_fields[fi.field_id]["name"]
                    for fi in sample_student.field_interests
                ]
            },
            "results": {
                "total_labs": len(result.results),
                "top_3_matches": [
                    {
                        "rank": i + 1,
                        "lab_name": r.lab.name,
                        "professor": r.lab.professor,
                        "overall_score": r.compatibility.overall_score,
                        "field_compatibility": r.compatibility.field_compatibility,
                        "criteria_compatibility": r.compatibility.criteria_compatibility
                    }
                    for i, r in enumerate(result.results[:3])
                ],
                "summary": {
                    "avg_compatibility": result.summary.avg_compatibility,
                    "best_match_score": result.summary.best_match_score,
                    "optimization_fitness": result.optimization_info["final_fitness"]
                }
            }
        }
        
    except Exception as e:
        logger.error(f"テスト評価エラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"テスト評価に失敗しました: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    print("🎯 研究室選択支援システム起動")
    print("=" * 50)
    print(f"📊 対応研究分野: {len(settings.research_fields)}分野")
    print(f"📋 評価基準: {len(settings.evaluation_criteria)}項目")
    print(f"🧬 GA設定: 集団{settings.ga_population_size} × {settings.ga_generations}世代")
    print("=" * 50)
    
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )