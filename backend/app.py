# backend/app.py - 完全版（技術資料準拠）
"""
研究室選択支援システム バックエンドAPI

技術資料「ファジィ決定木を用いた研究室マッチングアルゴリズムの提案（最終版）」
に完全準拠した実装を使用
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# プロジェクトルート設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# マッチャーインポート
try:
    from core.matching.fuzzy_multipath_matcher import (
        FuzzyMultiPathMatcher, 
        FuzzyPath,
        CompatibilityResult as FuzzyCompatibilityResult
    )
    FUZZY_MATCHER_AVAILABLE = True
    print("✅ FuzzyMultiPathMatcher インポート成功")
except ImportError as e:
    FUZZY_MATCHER_AVAILABLE = False
    print(f"❌ FuzzyMultiPathMatcher インポート失敗: {e}")

try:
    from core.matching.simple_matcher import SimpleMatcher, CompatibilityResult
    SIMPLE_MATCHER_AVAILABLE = True
    print("✅ SimpleMatcher インポート成功")
except ImportError as e:
    SIMPLE_MATCHER_AVAILABLE = False
    print(f"❌ SimpleMatcher インポート失敗: {e}")

try:
    from core.matching.field_matcher_corrected import (
        FieldMatcherCorrected,
        FieldMatchResult
    )
    FIELD_MATCHER_AVAILABLE = True
    print("✅ FieldMatcherCorrected インポート成功")
except ImportError as e:
    FIELD_MATCHER_AVAILABLE = False
    print(f"❌ FieldMatcherCorrected インポート失敗: {e}")


# ==================== FastAPI アプリケーション ====================

app = FastAPI(
    title="研究室選択支援システム",
    description="ファジィ決定木による高精度マッチングシステム",
    version="3.0.0"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に制限すること
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== データモデル ====================

class StudentProfile(BaseModel):
    """学生プロファイル"""
    # 基本13項目（1-10）
    research_intensity: float = Field(5.0, ge=1, le=10)
    advisor_style: float = Field(5.0, ge=1, le=10)
    team_work: float = Field(5.0, ge=1, le=10)
    workload: float = Field(5.0, ge=1, le=10)
    theory_practice: float = Field(5.0, ge=1, le=10)
    research_field_match: float = Field(5.0, ge=1, le=10)
    skill_development: float = Field(5.0, ge=1, le=10)
    lab_atmosphere: float = Field(5.0, ge=1, le=10)
    flexibility: float = Field(5.0, ge=1, le=10)
    publication_opportunity: float = Field(5.0, ge=1, le=10)
    interdisciplinary: float = Field(5.0, ge=1, le=10)
    communication_style: float = Field(5.0, ge=1, le=10)
    
    # 優先度（1-10）
    research_intensity_priority: float = Field(5.0, ge=1, le=10)
    advisor_style_priority: float = Field(5.0, ge=1, le=10)
    team_work_priority: float = Field(5.0, ge=1, le=10)
    workload_priority: float = Field(5.0, ge=1, le=10)
    theory_practice_priority: float = Field(5.0, ge=1, le=10)
    research_field_match_priority: float = Field(5.0, ge=1, le=10)
    skill_development_priority: float = Field(5.0, ge=1, le=10)
    lab_atmosphere_priority: float = Field(5.0, ge=1, le=10)
    flexibility_priority: float = Field(5.0, ge=1, le=10)
    publication_opportunity_priority: float = Field(5.0, ge=1, le=10)
    interdisciplinary_priority: float = Field(5.0, ge=1, le=10)
    communication_style_priority: float = Field(5.0, ge=1, le=10)
    
    # 分野興味（field_id: interest_level）
    field_interests: Dict[str, float] = Field(default_factory=dict)


class EvaluationRequest(BaseModel):
    """評価リクエスト"""
    student: StudentProfile
    matcher_type: Optional[str] = "multipath"  # "multipath" or "simple"


class LabEvaluationResponse(BaseModel):
    """研究室評価レスポンス"""
    lab_id: str
    lab_name: str
    professor: str
    field_id: str
    field_name: Optional[str] = None
    
    # スコア
    total_compatibility: float
    basic_score: float
    field_score: float
    
    # 詳細
    num_fuzzy_paths: Optional[int] = None
    explanation: str
    recommendation: str
    
    # 研究室情報
    description: Optional[str] = None
    students_count: Optional[int] = None


# ==================== グローバル状態 ====================

system_state = {
    "initialized": False,
    "matcher_multipath": None,
    "matcher_simple": None,
    "field_matcher": None,
    "lab_database": [],
    "evaluation_count": 0,
    "startup_time": None
}

# 12項目評価基準
EVALUATION_CRITERIA = [
    "research_intensity",
    "advisor_style", 
    "team_work",
    "workload",
    "theory_practice",
    "research_field_match",
    "skill_development",
    "lab_atmosphere",
    "flexibility",
    "publication_opportunity",
    "interdisciplinary",
    "communication_style"
]


# ==================== データベース読み込み ====================

def load_labs_database() -> List[Dict[str, Any]]:
    """研究室データベース読み込み"""
    
    # パス候補
    db_paths = [
        project_root / "data" / "labs_database.json",
        Path("data/labs_database.json"),
        Path("backend/data/labs_database.json"),
        project_root.parent / "data" / "labs_database.json"
    ]
    
    for db_path in db_paths:
        if db_path.exists():
            try:
                with open(db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    labs = data.get("labs", [])
                    if labs:
                        print(f"✅ 研究室データ読み込み: {db_path} ({len(labs)}件)")
                        return labs
            except Exception as e:
                print(f"⚠️ {db_path} 読み込みエラー: {e}")
                continue
    
    print("⚠️ labs_database.json が見つかりません。サンプルデータを使用します。")
    return get_sample_labs()


def get_sample_labs() -> List[Dict[str, Any]]:
    """サンプル研究室データ"""
    return [
        {
            "id": "ai_lab",
            "name": "AI研究室",
            "professor": "山田教授",
            "description": "深層学習と機械学習の応用研究",
            "field_id": "ai_ml",
            "research_intensity": 9,
            "advisor_style": 7,
            "team_work": 8,
            "workload": 8,
            "theory_practice": 6,
            "skill_development": 8,
            "lab_atmosphere": 7,
            "flexibility": 6,
            "publication_opportunity": 9,
            "interdisciplinary": 5,
            "communication_style": 7,
            "students_count": 15
        },
        {
            "id": "image_lab",
            "name": "画像処理研究室",
            "professor": "佐藤教授",
            "description": "コンピュータビジョンと画像認識",
            "field_id": "image_processing",
            "research_intensity": 8,
            "advisor_style": 6,
            "team_work": 7,
            "workload": 7,
            "theory_practice": 7,
            "skill_development": 7,
            "lab_atmosphere": 6,
            "flexibility": 7,
            "publication_opportunity": 7,
            "interdisciplinary": 6,
            "communication_style": 6,
            "students_count": 12
        },
        {
            "id": "network_lab",
            "name": "ネットワークセキュリティ研究室",
            "professor": "鈴木教授",
            "description": "ネットワークセキュリティと暗号技術",
            "field_id": "network_security",
            "research_intensity": 7,
            "advisor_style": 8,
            "team_work": 6,
            "workload": 6,
            "theory_practice": 5,
            "skill_development": 6,
            "lab_atmosphere": 7,
            "flexibility": 8,
            "publication_opportunity": 6,
            "interdisciplinary": 4,
            "communication_style": 7,
            "students_count": 10
        },
        {
            "id": "web_design_lab",
            "name": "Webデザイン研究室",
            "professor": "田中教授",
            "description": "UI/UXとWebデザインの研究",
            "field_id": "web_design_ui_ux",
            "research_intensity": 6,
            "advisor_style": 9,
            "team_work": 8,
            "workload": 5,
            "theory_practice": 8,
            "skill_development": 7,
            "lab_atmosphere": 8,
            "flexibility": 9,
            "publication_opportunity": 5,
            "interdisciplinary": 7,
            "communication_style": 8,
            "students_count": 14
        },
        {
            "id": "game_lab",
            "name": "ゲーム開発研究室",
            "professor": "高橋教授",
            "description": "ゲーム開発とeスポーツ",
            "field_id": "game_dev_esports",
            "research_intensity": 7,
            "advisor_style": 8,
            "team_work": 9,
            "workload": 7,
            "theory_practice": 9,
            "skill_development": 8,
            "lab_atmosphere": 9,
            "flexibility": 7,
            "publication_opportunity": 4,
            "interdisciplinary": 6,
            "communication_style": 9,
            "students_count": 16
        }
    ]


# ==================== 起動処理 ====================

@app.on_event("startup")
async def startup_event():
    """起動時初期化"""
    
    print("\n" + "="*70)
    print("🚀 研究室選択支援システム 起動")
    print("="*70)
    
    system_state["startup_time"] = datetime.now()
    
    # 研究室データベース読み込み
    system_state["lab_database"] = load_labs_database()
    print(f"📚 研究室数: {len(system_state['lab_database'])}")
    
    # FuzzyMultiPathMatcher 初期化
    if FUZZY_MATCHER_AVAILABLE:
        try:
            system_state["matcher_multipath"] = FuzzyMultiPathMatcher()
            print("✅ FuzzyMultiPathMatcher 初期化完了（技術資料準拠版）")
        except Exception as e:
            print(f"❌ FuzzyMultiPathMatcher 初期化失敗: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️ FuzzyMultiPathMatcher が利用できません")
    
    # SimpleMatcher 初期化（バックアップ）
    if SIMPLE_MATCHER_AVAILABLE:
        try:
            system_state["matcher_simple"] = SimpleMatcher()
            print("✅ SimpleMatcher 初期化完了（改善版）")
        except Exception as e:
            print(f"❌ SimpleMatcher 初期化失敗: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️ SimpleMatcher が利用できません")
    
    # FieldMatcherCorrected 初期化
    if FIELD_MATCHER_AVAILABLE:
        try:
            system_state["field_matcher"] = FieldMatcherCorrected()
            print("✅ FieldMatcherCorrected 初期化完了")
        except Exception as e:
            print(f"❌ FieldMatcherCorrected 初期化失敗: {e}")
            import traceback
            traceback.print_exc()
    
    system_state["initialized"] = True
    
    print("="*70)
    print("✅ システム初期化完了")
    print(f"🌐 APIエンドポイント: http://localhost:8000")
    print(f"📖 ドキュメント: http://localhost:8000/docs")
    print("="*70 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    """シャットダウン処理"""
    print("\n" + "="*70)
    print("🛑 システムシャットダウン")
    print(f"📊 総評価回数: {system_state['evaluation_count']}")
    print("="*70 + "\n")


# ==================== APIエンドポイント ====================

@app.get("/")
async def root():
    """ルート"""
    return {
        "message": "研究室選択支援システム API",
        "version": "3.0.0",
        "status": "running",
        "matcher_available": {
            "multipath": FUZZY_MATCHER_AVAILABLE,
            "simple": SIMPLE_MATCHER_AVAILABLE
        },
        "endpoints": {
            "health": "/api/health",
            "labs": "/api/labs",
            "evaluate": "/api/evaluate",
            "stats": "/api/stats"
        }
    }


@app.get("/api/health")
async def health_check():
    """ヘルスチェック"""
    
    if not system_state.get("initialized"):
        raise HTTPException(status_code=503, detail="システムが初期化されていません")
    
    return {
        "status": "healthy",
        "initialized": system_state["initialized"],
        "lab_count": len(system_state["lab_database"]),
        "evaluation_count": system_state["evaluation_count"],
        "matcher_status": {
            "multipath": system_state["matcher_multipath"] is not None,
            "simple": system_state["matcher_simple"] is not None,
            "field": system_state["field_matcher"] is not None
        },
        "uptime_seconds": (
            datetime.now() - system_state["startup_time"]
        ).total_seconds() if system_state["startup_time"] else 0
    }


@app.get("/api/labs")
async def get_labs(
    field_id: Optional[str] = Query(None, description="分野IDでフィルタ")
):
    """研究室一覧取得"""
    
    labs = system_state["lab_database"]
    
    # フィルタリング
    if field_id:
        labs = [lab for lab in labs if lab.get("field_id") == field_id]
    
    return {
        "total": len(labs),
        "labs": labs
    }


@app.get("/api/labs/{lab_id}")
async def get_lab_detail(lab_id: str):
    """研究室詳細取得"""
    
    labs = system_state["lab_database"]
    lab = next((l for l in labs if l.get("id") == lab_id), None)
    
    if not lab:
        raise HTTPException(status_code=404, detail=f"研究室ID '{lab_id}' が見つかりません")
    
    return {"lab": lab}


@app.post("/api/evaluate")
async def evaluate_labs(request: EvaluationRequest):
    """
    研究室評価API
    
    学生プロファイルに基づき、全研究室との適合度を計算
    """
    
    if not system_state.get("initialized"):
        raise HTTPException(status_code=503, detail="システムが初期化されていません")
    
    # マッチャータイプ選択
    matcher_type = request.matcher_type or "multipath"
    
    if matcher_type == "multipath":
        matcher = system_state.get("matcher_multipath")
        if not matcher:
            # フォールバック
            matcher = system_state.get("matcher_simple")
            if not matcher:
                raise HTTPException(
                    status_code=500, 
                    detail="マッチャーが利用できません"
                )
            matcher_type = "simple"
    else:
        matcher = system_state.get("matcher_simple")
        if not matcher:
            raise HTTPException(
                status_code=500,
                detail="SimpleMatcherが利用できません"
            )
    
    # 学生プロファイルを辞書に変換
    student_dict = request.student.dict()
    
    # 全研究室を評価
    results = []
    labs = system_state["lab_database"]
    
    start_time = time.time()
    
    for lab in labs:
        try:
            # 適合度計算
            result = matcher.calculate_compatibility(student_dict, lab)
            
            # フィールドマッチャーで分野名取得
            field_name = lab.get("field_id", "不明")
            if system_state.get("field_matcher"):
                field_name = system_state["field_matcher"].get_field_name(
                    lab.get("field_id", "")
                )
            
            # パス数（FuzzyMultiPathMatcherの場合のみ）
            num_paths = None
            if hasattr(result, 'fuzzy_paths'):
                num_paths = len(result.fuzzy_paths)
            
            results.append({
                "lab_id": lab.get("id"),
                "lab_name": lab.get("name"),
                "professor": lab.get("professor", ""),
                "field_id": lab.get("field_id", ""),
                "field_name": field_name,
                "total_compatibility": result.total_compatibility,
                "basic_score": result.basic_score,
                "field_score": result.field_score,
                "num_fuzzy_paths": num_paths,
                "explanation": result.explanation,
                "recommendation": result.recommendation,
                "description": lab.get("description"),
                "students_count": lab.get("students_count")
            })
            
        except Exception as e:
            print(f"⚠️ 研究室 {lab.get('id')} の評価エラー: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # スコア降順でソート
    results.sort(key=lambda x: x["total_compatibility"], reverse=True)
    
    processing_time = time.time() - start_time
    
    # 統計更新
    system_state["evaluation_count"] += 1
    
    return {
        "total_labs": len(results),
        "results": results,
        "matcher_type": matcher_type,
        "processing_time_ms": processing_time * 1000,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/evaluate/{lab_id}")
async def evaluate_single_lab(lab_id: str, request: Dict[str, Any]):
    """
    単一研究室の詳細評価
    
    より詳細な情報を返す
    """
    
    if not system_state.get("initialized"):
        raise HTTPException(status_code=503, detail="システムが初期化されていません")
    
    # 研究室取得
    labs = system_state["lab_database"]
    lab = next((l for l in labs if l.get("id") == lab_id), None)
    
    if not lab:
        raise HTTPException(status_code=404, detail=f"研究室ID '{lab_id}' が見つかりません")
    
    # マッチャー取得（デフォルト: multipath）
    matcher = system_state.get("matcher_multipath") or system_state.get("matcher_simple")
    if not matcher:
        raise HTTPException(status_code=500, detail="マッチャーが利用できません")
    
    # 学生プロファイル
    student = request.get("student", {})
    
    # 適合度計算
    result = matcher.calculate_compatibility(student, lab)
    
    # 詳細情報構築
    response = {
        "lab": lab,
        "compatibility": {
            "total": result.total_compatibility,
            "basic": result.basic_score,
            "field": result.field_score
        },
        "explanation": result.explanation,
        "recommendation": result.recommendation
    }
    
    # FuzzyMultiPathMatcherの場合、パス情報も含める
    if hasattr(result, 'fuzzy_paths'):
        response["fuzzy_paths"] = [
            {
                "path_id": path.path_id,
                "membership": path.total_membership,
                "score": path.score,
                "layers": [
                    {"criterion": layer[0], "label": layer[1], "membership": layer[2]}
                    for layer in path.layers
                ]
            }
            for path in result.fuzzy_paths[:5]  # 上位5パスのみ
        ]
    
    # 項目別スコア
    if hasattr(result, 'criteria_scores'):
        response["criteria_scores"] = result.criteria_scores
    
    return response


@app.get("/api/stats")
async def get_statistics():
    """システム統計情報"""
    
    labs = system_state["lab_database"]
    
    # 分野別集計
    field_distribution = {}
    for lab in labs:
        field_id = lab.get("field_id", "unknown")
        field_distribution[field_id] = field_distribution.get(field_id, 0) + 1
    
    return {
        "total_labs": len(labs),
        "total_evaluations": system_state["evaluation_count"],
        "field_distribution": field_distribution,
        "system_info": {
            "version": "3.0.0",
            "matcher_type": "FuzzyMultiPathMatcher (技術資料準拠版)",
            "initialized": system_state["initialized"],
            "uptime_seconds": (
                datetime.now() - system_state["startup_time"]
            ).total_seconds() if system_state["startup_time"] else 0
        }
    }


@app.get("/api/criteria")
async def get_criteria_info():
    """評価基準情報取得"""
    
    criteria_info = {
        "research_intensity": {
            "name": "研究強度",
            "description": "研究にどれだけ集中的に取り組みたいか",
            "range": [1, 10]
        },
        "advisor_style": {
            "name": "指導スタイル",
            "description": "教授からの指導の受け方の好み",
            "range": [1, 10]
        },
        "team_work": {
            "name": "チームワーク",
            "description": "研究での他者との協働の程度",
            "range": [1, 10]
        },
        "workload": {
            "name": "ワークロード",
            "description": "研究活動の忙しさに対する許容度",
            "range": [1, 10]
        },
        "theory_practice": {
            "name": "理論・実践バランス",
            "description": "理論研究と実践的研究のバランス",
            "range": [1, 10]
        },
        "research_field_match": {
            "name": "研究分野重視度",
            "description": "分野の一致をどれだけ重視するか",
            "range": [1, 10]
        },
        "skill_development": {
            "name": "スキル開発",
            "description": "専門性と汎用性のバランス",
            "range": [1, 10]
        },
        "lab_atmosphere": {
            "name": "研究室雰囲気",
            "description": "研究室の全体的な雰囲気",
            "range": [1, 10]
        },
        "flexibility": {
            "name": "柔軟性",
            "description": "研究時間の自由度",
            "range": [1, 10]
        },
        "publication_opportunity": {
            "name": "論文発表機会",
            "description": "研究成果の論文化機会",
            "range": [1, 10]
        },
        "interdisciplinary": {
            "name": "学際性",
            "description": "他分野との連携の程度",
            "range": [1, 10]
        },
        "communication_style": {
            "name": "コミュニケーション",
            "description": "研究室での交流スタイル",
            "range": [1, 10]
        }
    }
    
    return {
        "total": len(criteria_info),
        "criteria": criteria_info
    }


# ==================== エラーハンドラー ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTPエラーハンドラー"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """一般エラーハンドラー"""
    print(f"❌ エラー発生: {exc}")
    import traceback
    traceback.print_exc()
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "内部サーバーエラー",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# ==================== メイン実行 ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 研究室選択支援システム 起動準備")
    print("="*70)
    print(f"📂 プロジェクトルート: {project_root}")
    print(f"🐍 Python バージョン: {sys.version}")
    print("="*70 + "\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 開発時のみ
        log_level="info"
    )