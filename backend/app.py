#!/usr/bin/env python3
# app.py
"""
研究室選択支援システム v3.0 - パターンB
- 12項目評価基準
- 20研究分野対応
- 適応的ファジィ決定木
"""

import os
import sys
import json
import time
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Dict, List, Any, Optional

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 設定のインポート
try:
    from config.settings import settings
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False
    class FallbackSettings:
        host = "0.0.0.0"
        port = 8000
        debug = True
    settings = FallbackSettings()

# マッチャーのインポート
try:
    from core.matching.simple_matcher import SimpleMatcher, CompatibilityResult
    MATCHER_AVAILABLE = True
except ImportError as e:
    MATCHER_AVAILABLE = False
    print(f"⚠️ マッチャーが利用できません: {e}")

# デフォルトパラメータのインポート
try:
    from config.default_params import (
        DEFAULT_PARAMS, BASIC_CRITERIA, CRITERIA_DETAILS,
        FIELD_NAMES, FIELD_CATEGORIES, FIELD_DETAILS,
        get_field_name, get_field_category, get_all_fields
    )
    PARAMS_AVAILABLE = True
except ImportError as e:
    PARAMS_AVAILABLE = False
    print(f"⚠️ デフォルトパラメータが利用できません: {e}")

# FastAPIアプリケーション初期化
app = FastAPI(
    title="研究室選択支援システム v3.0（パターンB）",
    description="12項目評価 + 20研究分野対応 + 適応的ファジィ決定木による研究室推薦システム",
    version="3.0.0-PatternB",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# 静的ファイル配信
if os.path.exists("../frontend/build"):
    app.mount("/static", StaticFiles(directory="../frontend/build/static"), name="static")

# グローバル変数（システム状態）
system_state = {
    "initialized": False,
    "matcher": None,
    "lab_database": [],
    "evaluation_count": 0
}

# サンプル研究室データ（12項目 + 20分野対応）
SAMPLE_LABS = [
    {
        "id": "lab_ai",
        "name": "人工知能研究室",
        "professor": "田中教授",
        "field_id": "ai_ml",
        "category": "テクノロジー・システム",
        "description": "機械学習とディープラーニングの最先端研究",
        # 基本5項目
        "research_intensity": 9.0,
        "advisor_style": 7.0,
        "team_work": 8.0,
        "workload": 8.5,
        "theory_practice": 6.0,
        # 拡張5項目
        "research_field_match": 9.0,
        "skill_development": 8.0,
        "lab_atmosphere": 7.5,
        "flexibility": 6.5,
        "publication_opportunity": 9.5,
        # 特殊2項目
        "interdisciplinary": 7.0,
        "communication_style": 7.5,
    },
    {
        "id": "lab_image",
        "name": "画像処理研究室",
        "professor": "佐藤教授",
        "field_id": "image_processing",
        "category": "テクノロジー・システム",
        "description": "コンピュータビジョンとパターン認識の研究",
        # 基本5項目
        "research_intensity": 8.5,
        "advisor_style": 6.5,
        "team_work": 7.5,
        "workload": 8.0,
        "theory_practice": 6.5,
        # 拡張5項目
        "research_field_match": 8.0,
        "skill_development": 8.0,
        "lab_atmosphere": 7.0,
        "flexibility": 7.0,
        "publication_opportunity": 8.5,
        # 特殊2項目
        "interdisciplinary": 6.5,
        "communication_style": 6.5,
    },
    {
        "id": "lab_web",
        "name": "Webデザイン研究室",
        "professor": "鈴木教授",
        "field_id": "web_design",
        "category": "クリエイティブ",
        "description": "Webデザインとユーザー体験の研究",
        # 基本5項目
        "research_intensity": 6.5,
        "advisor_style": 8.0,
        "team_work": 9.0,
        "workload": 6.0,
        "theory_practice": 8.0,
        # 拡張5項目
        "research_field_match": 7.5,
        "skill_development": 8.5,
        "lab_atmosphere": 8.5,
        "flexibility": 8.0,
        "publication_opportunity": 6.0,
        # 特殊2項目
        "interdisciplinary": 8.0,
        "communication_style": 9.0,
    },
    {
        "id": "lab_game",
        "name": "ゲーム開発研究室",
        "professor": "高橋教授",
        "field_id": "game_esports",
        "category": "エンターテイメント",
        "description": "ゲーム開発とeスポーツの研究",
        # 基本5項目
        "research_intensity": 7.5,
        "advisor_style": 7.5,
        "team_work": 9.0,
        "workload": 7.0,
        "theory_practice": 8.5,
        # 拡張5項目
        "research_field_match": 7.0,
        "skill_development": 9.0,
        "lab_atmosphere": 9.0,
        "flexibility": 7.5,
        "publication_opportunity": 6.5,
        # 特殊2項目
        "interdisciplinary": 7.5,
        "communication_style": 8.5,
    },
    {
        "id": "lab_vr",
        "name": "VR/AR研究室",
        "professor": "山田教授",
        "field_id": "vr_ar_media",
        "category": "エンターテイメント",
        "description": "仮想現実とメディアアートの研究",
        # 基本5項目
        "research_intensity": 8.0,
        "advisor_style": 7.0,
        "team_work": 8.5,
        "workload": 7.5,
        "theory_practice": 7.5,
        # 拡張5項目
        "research_field_match": 7.5,
        "skill_development": 8.5,
        "lab_atmosphere": 8.0,
        "flexibility": 7.0,
        "publication_opportunity": 7.5,
        # 特殊2項目
        "interdisciplinary": 9.0,
        "communication_style": 8.0,
    },
]


def initialize_system():
    """システム初期化"""
    try:
        print("\n" + "="*60)
        print("研究室選択支援システム v3.0 初期化")
        print("="*60)
        
        # マッチャー初期化
        if MATCHER_AVAILABLE and PARAMS_AVAILABLE:
            system_state["matcher"] = SimpleMatcher()
            system_state["lab_database"] = SAMPLE_LABS
            system_state["initialized"] = True
            
            print(f"\n✅ システム初期化完了")
            print(f"   - パターン: B (適応的決定木)")
            print(f"   - 評価項目: {len(BASIC_CRITERIA)}項目")
            print(f"   - 研究分野: {len(FIELD_NAMES)}分野")
            print(f"   - 研究室数: {len(SAMPLE_LABS)}件")
        else:
            print(f"\n⚠️ 初期化失敗: 必要なモジュールが不足")
            system_state["initialized"] = False
        
        print("="*60 + "\n")
        
    except Exception as e:
        import traceback
        print(f"❌ システム初期化エラー: {e}")
        print(traceback.format_exc())
        system_state["initialized"] = False


# システム初期化
initialize_system()


# ===== APIエンドポイント =====

@app.get("/")
async def read_root():
    """ルートエンドポイント"""
    if os.path.exists("../frontend/build/index.html"):
        return FileResponse("../frontend/build/index.html")
    else:
        return {
            "message": "研究室選択支援システム v3.0（パターンB）",
            "version": "3.0.0-PatternB",
            "status": "running",
            "features": {
                "criteria_count": 12,
                "fields_count": 20,
                "pattern": "B (Adaptive Fuzzy Decision Tree)"
            },
            "endpoints": {
                "health": "/health",
                "labs": "/api/labs",
                "criteria": "/api/criteria",
                "fields": "/api/fields",
                "evaluate": "/api/evaluate",
                "docs": "/docs"
            }
        }


@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    
    lab_count = len(system_state.get("lab_database", []))
    
    return {
        "status": "healthy" if system_state["initialized"] else "unhealthy",
        "version": "3.0.0-PatternB",
        "timestamp": time.time(),
        "system_initialized": system_state["initialized"],
        "modules": {
            "matcher": MATCHER_AVAILABLE,
            "params": PARAMS_AVAILABLE,
            "settings": SETTINGS_AVAILABLE
        },
        "database": {
            "status": "OK" if lab_count > 0 else "Empty",
            "lab_count": lab_count,
            "evaluation_count": system_state["evaluation_count"]
        },
        "configuration": {
            "pattern": "B (Adaptive)",
            "criteria_count": len(BASIC_CRITERIA) if PARAMS_AVAILABLE else 0,
            "fields_count": len(FIELD_NAMES) if PARAMS_AVAILABLE else 0,
            "priority_thresholds": {
                "high": 8.0,
                "mid": 5.0
            }
        }
    }


@app.get("/api/labs")
async def get_labs():
    """研究室一覧取得"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {
        "labs": system_state["lab_database"],
        "total_count": len(system_state["lab_database"]),
        "last_updated": time.time()
    }


@app.get("/api/labs/{lab_id}")
async def get_lab_detail(lab_id: str):
    """特定研究室の詳細取得"""
    
    lab = next((lab for lab in system_state["lab_database"] if lab["id"] == lab_id), None)
    
    if not lab:
        raise HTTPException(status_code=404, detail="Lab not found")
    
    return lab


@app.get("/api/criteria")
async def get_criteria():
    """評価基準一覧取得（13項目）"""
    
    if not PARAMS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Parameters not available")
    
    criteria_list = []
    for criterion in BASIC_CRITERIA:
        detail = CRITERIA_DETAILS.get(criterion, {})
        criteria_list.append({
            "id": criterion,
            "name": detail.get("name", criterion),
            "description": detail.get("description", ""),
            "range": detail.get("range", "1-10"),
            "category": detail.get("category", "basic")
        })
    
    return {
        "criteria": criteria_list,
        "total_count": len(criteria_list)
    }


@app.get("/api/fields")
async def get_fields():
    """研究分野一覧取得（20分野）"""
    
    if not PARAMS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Parameters not available")
    
    fields = get_all_fields()
    
    return {
        "fields": fields,
        "total_count": len(fields),
        "categories": list(FIELD_CATEGORIES.keys())
    }


def _validate_student_profile(profile: Dict[str, Any]) -> bool:
    """学生プロファイルのバリデーション"""
    
    # 必須項目チェック（12項目）
    for criterion in BASIC_CRITERIA:
        if criterion not in profile:
            return False
        
        value = profile[criterion]
        if not isinstance(value, (int, float)) or value < 1 or value > 10:
            return False
    
    return True


@app.post("/api/evaluate")
async def evaluate_labs(profile: Dict[str, Any]):
    """
    研究室適合度評価（12項目 + 20分野対応）
    
    リクエストボディ:
    {
        // 基本5項目
        "research_intensity": 9,
        "advisor_style": 7,
        "team_work": 5,
        "workload": 8,
        "theory_practice": 6,
        
        // 拡張5項目
        "research_field_match": 7,  // 分野重視度（1=基本項目重視, 10=分野重視）
        "skill_development": 7,
        "lab_atmosphere": 6,
        "flexibility": 5,
        "publication_opportunity": 9,
        
        // 特殊2項目
        "interdisciplinary": 4,
        "communication_style": 6,
        
        // 優先度（オプション）
        "research_intensity_priority": 10,
        "publication_opportunity_priority": 10,
        // ...
        
        // 分野興味
        "field_interests": {
            "ai_ml": 10,
            "image_processing": 7
        }
    }
    """
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # バリデーション
    if not _validate_student_profile(profile):
        raise HTTPException(
            status_code=400,
            detail="Invalid profile: All 12 criteria required with values 1-10"
        )
    
    try:
        matcher = system_state["matcher"]
        labs = system_state["lab_database"]
        
        # 一括評価
        results = matcher.batch_calculate(profile, labs)
        
        # レスポンス整形
        evaluation_results = []
        total_score = 0.0
        
        for lab, result in results:
            evaluation_results.append({
                "lab_id": lab["id"],
                "lab_name": lab["name"],
                "professor": lab.get("professor", ""),
                "field_id": lab["field_id"],
                "field_name": get_field_name(lab["field_id"]),
                "category": lab.get("category", ""),
                
                # スコア
                "overall_compatibility": result.total_compatibility,
                "basic_score": result.basic_score,
                "field_score": result.field_score,
                "field_weight_alpha": result.field_weight_alpha,
                "basic_weight_beta": result.basic_weight_beta,
                
                # 推薦
                "recommendation": result.recommendation,
                "explanation": result.explanation,
                
                # パターンB固有情報
                "tree_path": result.tree_path,
                "tree_layers": result.tree_layers,
                "leaf_criteria": result.leaf_criteria,
                
                # 詳細
                "criteria_scores": result.criteria_scores,
                "field_detail": result.field_detail,
            })
            
            total_score += result.total_compatibility
        
        # 統計情報
        avg_score = total_score / len(results) if results else 0
        best_match = evaluation_results[0] if evaluation_results else None
        
        # カウント更新
        system_state["evaluation_count"] += 1
        
        return {
            "evaluation_results": evaluation_results,
            "summary": {
                "total_labs": len(evaluation_results),
                "avg_score": avg_score,
                "best_match": best_match["lab_name"] if best_match else None,
                "evaluation_count": system_state["evaluation_count"]
            },
            "system_info": {
                "pattern": "B",
                "version": "3.0.0",
                "criteria_count": 12,
                "fields_count": 20,
                "timestamp": time.time()
            }
        }
    
    except Exception as e:
        import traceback
        print(f"❌ 評価エラー: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")


@app.get("/api/test/sample-profile")
async def get_sample_profile():
    """サンプルプロファイル取得"""
    
    return {
        # 基本5項目
        "research_intensity": 7,
        "advisor_style": 6,
        "team_work": 7,
        "workload": 6,
        "theory_practice": 7,
        
        # 拡張5項目
        "research_field_match": 8,  # 分野重視度
        "skill_development": 7,
        "lab_atmosphere": 7,
        "flexibility": 7,
        "publication_opportunity": 8,
        
        # 特殊2項目
        "interdisciplinary": 6,
        "communication_style": 6,
        
        # 優先度（一部）
        "research_intensity_priority": 9,
        "publication_opportunity_priority": 8,
        
        # 分野興味
        "field_interests": {
            "ai_ml": 9,
            "image_processing": 7,
            "web_design": 6
        }
    }


# ===== サーバー起動 =====

if __name__ == "__main__":
    print("\n🚀 研究室選択支援システム v3.0 起動中...")
    print(f"   パターン: B (適応的ファジィ決定木)")
    print(f"   評価項目: 12項目")
    print(f"   研究分野: 20分野")
    print(f"   ポート: {settings.port}\n")
    
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info"
    )