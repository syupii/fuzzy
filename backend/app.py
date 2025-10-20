#!/usr/bin/env python3
"""
研究室選択支援システム - FastAPI バックエンド（修正版）
評価が0になる問題を修正
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# プロジェクトルート設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# --- ルーターのインポート ---
from api.v1.demo import router as demo_router

# --- マッチャーインポート ---
try:
    from core.matching.fuzzy_multipath_matcher import (
        FuzzyMultiPathMatcher, 
        CompatibilityResult as FuzzyCompatibilityResult
    )
    FUZZY_MATCHER_AVAILABLE = True
    print("✅ FuzzyMultiPathMatcher インポート成功")
except ImportError as e:
    FUZZY_MATCHER_AVAILABLE = False
    print(f"⚠️ FuzzyMultiPathMatcher インポート失敗: {e}")


# FastAPIアプリケーションのインスタンスを先に作成
app = FastAPI(
    title="研究室選択支援システム",
    description="ファジィ決定木による高精度マッチングシステム",
    version="3.1.0-fixed"
)

# --- CORS設定 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルーターの登録をインスタンス作成後に移動
app.include_router(demo_router)
print("✅ デモプロファイルAPI登録完了: /api/demo/*")


# --- グローバル状態 ---
system_state = {
    "initialized": False,
    "matcher_multipath": None,
    "lab_database": [],
    "evaluation_count": 0,
    "startup_time": None
}

# 12項目評価基準
EVALUATION_CRITERIA = [
    "research_intensity", "advisor_style", "team_work",
    "workload", "theory_practice", "research_field_match",
    "skill_development", "lab_atmosphere", "flexibility",
    "publication_opportunity", "interdisciplinary", "communication_style"
]


# ==================== データベース読み込み ====================

def load_labs_database() -> List[Dict[str, Any]]:
    """研究室データベース読み込み"""
    
    db_paths = [
        project_root / "data" / "labs_database.json",
        Path("data/labs_database.json"),
        Path("backend/data/labs_database.json"),
    ]
    
    for db_path in db_paths:
        if db_path.exists():
            try:
                with open(db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    labs = data.get("labs", [])
                    normalized_labs = normalize_lab_data(labs)
                    
                    if normalized_labs:
                        print(f"✅ 研究室データ読み込み: {db_path} ({len(normalized_labs)}件)")
                        return normalized_labs
            except Exception as e:
                print(f"⚠️ {db_path} 読み込みエラー: {e}")
                continue
    
    print("⚠️ labs_database.json が見つかりません")
    return []


def normalize_lab_data(labs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """研究室データを正規化"""
    normalized = []
    
    for lab in labs:
        normalized_lab = lab.copy()
        
        if "features" in lab:
            features = lab["features"]
            for criterion in EVALUATION_CRITERIA:
                if criterion in features:
                    value = features[criterion]
                    normalized_lab[criterion] = (value - 1) / 9.0 if value > 1 else 0.0
        
        if "field_id" not in normalized_lab:
            research_area = normalized_lab.get("research_area", "")
            normalized_lab["field_id"] = research_area.lower().replace("・", "_")
        
        normalized.append(normalized_lab)
    
    return normalized


# ==================== システム初期化 ====================

def initialize_system():
    """システム初期化"""
    print("\n🚀 システム初期化開始...")
    try:
        system_state["lab_database"] = load_labs_database()
        if not system_state["lab_database"]:
            print("❌ 研究室データがありません")
            return False
        
        if FUZZY_MATCHER_AVAILABLE:
            system_state["matcher_multipath"] = FuzzyMultiPathMatcher()
            print("✅ FuzzyMultiPathMatcher 初期化完了")
        else:
            print("⚠️ マッチャーが利用できません")
            return False
        
        system_state["initialized"] = True
        system_state["startup_time"] = datetime.now()
        print(f"✅ システム初期化完了: {len(system_state['lab_database'])}研究室")
        return True
    except Exception as e:
        print(f"❌ 初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

# 起動時初期化
initialize_system()


# ==================== API エンドポイント ====================

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return { "message": "研究室選択支援システム API", "version": "3.1.0-fixed" }


@app.get("/api/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy" if system_state["initialized"] else "unhealthy",
        "lab_count": len(system_state["lab_database"]),
        "evaluation_count": system_state["evaluation_count"],
    }


@app.get("/api/labs")
async def get_labs(field_id: Optional[str] = Query(None)):
    """研究室一覧取得"""
    labs = system_state["lab_database"]
    if field_id:
        labs = [lab for lab in labs if lab.get("field_id") == field_id]
    return { "total": len(labs), "labs": labs }


def normalize_student_profile(student: Dict[str, Any]) -> Dict[str, Any]:
    """学生プロファイルを正規化 (1-10スケールを0-1スケールに変換)"""
    normalized = {}
    for criterion in EVALUATION_CRITERIA:
        value = student.get(criterion, 5.0)
        normalized[criterion] = (value - 1) / 9.0 if value > 1 else 0.0
        
        priority_key = f"{criterion}_priority"
        normalized[priority_key] = student.get(priority_key, 5.0)
    
    # ★★★ 修正: field_interests は正規化せず、1-10のまま辞書形式に変換 ★★★
    field_interests_list = student.get("field_interests", [])
    normalized["field_interests"] = {
        item["field_id"]: item.get("interest_level", 5)  # ← 正規化を削除
        for item in field_interests_list if "field_id" in item
    }
    
    # デバッグ出力
    print(f"\n{'='*70}")
    print(f"【デバッグ】学生プロファイル正規化")
    print(f"{'='*70}")
    print(f"正規化前 field_interests: {field_interests_list}")
    print(f"正規化後 field_interests: {normalized['field_interests']}")
    print(f"{'='*70}\n")
    
    return normalized


@app.post("/api/evaluate")
async def evaluate_labs(request: Dict[str, Any] = Body(...)):
    """研究室評価API"""
    if not system_state.get("initialized"):
        raise HTTPException(status_code=503, detail="システムが初期化されていません")
    
    student = request.get("student_profile")
    if not student:
        raise HTTPException(status_code=400, detail="学生プロファイルが見つかりません")
    
    normalized_student = normalize_student_profile(student)
    matcher = system_state.get("matcher_multipath")
    if not matcher:
        raise HTTPException(status_code=500, detail="評価マッチャーが利用できません")

    labs_to_evaluate = system_state["lab_database"]
    results = []
    start_time = time.time()
    
    for lab in labs_to_evaluate:
        try:
            result = matcher.calculate_compatibility(normalized_student, lab)
            lab_result = {
                "lab_id": lab.get("id"),
                "lab_name": lab.get("name"),
                "professor_name": lab.get("professor"),
                "research_area": lab.get("research_area"),
                "category": lab.get("category"),
                "final_score": result.total_compatibility,
                "basic_score": result.basic_score,
                "field_score": result.field_score,
                "feature_scores": result.criteria_scores,
                "explanation": result.explanation,
                "recommendation": result.recommendation,
                "priority_analysis": getattr(result, 'priority_analysis', None),
                "confidence": getattr(result, 'confidence', 0.85),
            }
            results.append(lab_result)
        except Exception as e:
            print(f"⚠️ {lab.get('name')} の評価エラー: {e}")
            continue
            
    processing_time = time.time() - start_time
    results.sort(key=lambda x: x["final_score"], reverse=True)
    system_state["evaluation_count"] += 1
    
    summary = {}
    if results:
        summary = {
            "total_labs": len(results),
            "high_compatibility_count": len([r for r in results if r["final_score"] >= 0.7]),
            "avg_score": sum(r["final_score"] for r in results) / len(results),
            "best_match_lab": results[0]["lab_name"],
            "best_match_score": results[0]["final_score"]
        }

    return {
        "evaluation_results": results,
        "summary": summary,
        "system_info": {
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
    }


# ==================== サーバー起動 ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("研究室選択支援システム バックエンド")
    print("="*60)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)