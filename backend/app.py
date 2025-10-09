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

# マッチャーインポート
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

# FastAPIアプリケーション
app = FastAPI(
    title="研究室選択支援システム",
    description="ファジィ決定木による高精度マッチングシステム",
    version="3.1.0-fixed"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバル状態
system_state = {
    "initialized": False,
    "matcher_multipath": None,
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
                    
                    # ★ データ構造の正規化 ★
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
    """
    研究室データを正規化
    
    ★★★ 重要な修正点 ★★★
    features内の評価基準をトップレベルに展開
    """
    normalized = []
    
    for lab in labs:
        normalized_lab = lab.copy()
        
        # featuresがある場合、トップレベルに展開
        if "features" in lab:
            features = lab["features"]
            for criterion in EVALUATION_CRITERIA:
                if criterion in features:
                    # 1-10スケールを0-1スケールに正規化
                    value = features[criterion]
                    normalized_lab[criterion] = value / 10.0
        
        # field_idがない場合、research_areaから生成
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
        # 研究室データベース読み込み
        system_state["lab_database"] = load_labs_database()
        
        if not system_state["lab_database"]:
            print("❌ 研究室データがありません")
            return False
        
        # マッチャー初期化
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
    return {
        "message": "研究室選択支援システム API",
        "version": "3.1.0-fixed",
        "status": "running" if system_state["initialized"] else "initializing",
        "endpoints": {
            "health": "/api/health",
            "labs": "/api/labs",
            "evaluate": "/api/evaluate",
            "docs": "/docs"
        }
    }


@app.get("/api/health")
async def health_check():
    """ヘルスチェック"""
    
    return {
        "status": "healthy" if system_state["initialized"] else "unhealthy",
        "initialized": system_state["initialized"],
        "lab_count": len(system_state["lab_database"]),
        "evaluation_count": system_state["evaluation_count"],
        "matcher_status": {
            "multipath": system_state["matcher_multipath"] is not None
        },
        "uptime_seconds": (
            (datetime.now() - system_state["startup_time"]).total_seconds() 
            if system_state["startup_time"] else 0
        )
    }


@app.get("/api/labs")
async def get_labs(field_id: Optional[str] = Query(None)):
    """研究室一覧取得"""
    
    labs = system_state["lab_database"]
    
    if field_id:
        labs = [lab for lab in labs if lab.get("field_id") == field_id]
    
    return {
        "total": len(labs),
        "labs": labs
    }


@app.post("/api/evaluate")
async def evaluate_labs(request: Dict[str, Any] = Body(...)):
    """
    研究室評価API
    
    ★★★ 修正点 ★★★
    1. リクエスト形式の柔軟な解析
    2. 値の正規化処理の追加
    3. デバッグログの追加
    """
    
    if not system_state.get("initialized"):
        raise HTTPException(status_code=503, detail="システムが初期化されていません")
    
    print(f"\n📥 評価リクエスト受信")
    print(f"  リクエストキー: {list(request.keys())}")
    
    # ★ リクエスト形式の柔軟な判定 ★
    student = None
    
    if "student_profile" in request:
        # 形式1: {"student_profile": {...}}  ← フロントエンドからの形式
        student = request.get("student_profile", {})
    elif "student" in request:
        # 形式2: {"student": {...}}
        student = request.get("student", {})
    else:
        # 形式3: 直接プロファイル
        student = request
    
    if not student:
        raise HTTPException(status_code=400, detail="学生プロファイルがありません")
    
    print(f"  学生プロファイル取得: {list(student.keys())[:5]}...")
    
    # ★ 値の正規化（1-10 → 0-1） ★
    normalized_student = normalize_student_profile(student)
    
    # マッチャー取得
    matcher = system_state.get("matcher_multipath")
    if not matcher:
        raise HTTPException(status_code=500, detail="マッチャーが利用できません")
    
    # 評価実行
    labs = system_state["lab_database"]
    results = []
    
    print(f"  評価開始: {len(labs)}研究室")
    
    start_time = time.time()
    
    for lab in labs:
        try:
            # ★ calculate_compatibilityを呼び出し ★
            result = matcher.calculate_compatibility(normalized_student, lab)
            
            lab_result = {
                "lab_id": lab.get("id"),
                "lab_name": lab.get("name"),
                "professor": lab.get("professor"),
                "research_area": lab.get("research_area"),
                "field_id": lab.get("field_id"),
                
                # スコア（0-1スケール）
                "total_compatibility": result.total_compatibility,
                "basic_score": result.basic_score,
                "field_score": result.field_score,
                
                # 重み
                "field_weight_alpha": result.field_weight_alpha,
                "basic_weight_beta": result.basic_weight_beta,
                
                # 詳細
                "criteria_scores": result.criteria_scores,
                "explanation": result.explanation,
                "recommendation": result.recommendation,
                
                # フロントエンド互換用（パーセント表示）
                "overall_compatibility": result.total_compatibility,  # 0-1
                "final_score": result.total_compatibility  # 0-1
            }
            
            results.append(lab_result)
            
        except Exception as e:
            print(f"  ⚠️ {lab.get('name')} の評価エラー: {e}")
            continue
    
    processing_time = time.time() - start_time
    
    # スコアでソート
    results.sort(key=lambda x: x["total_compatibility"], reverse=True)
    
    system_state["evaluation_count"] += 1
    
    print(f"  ✅ 評価完了: {len(results)}件 ({processing_time:.2f}秒)")
    print(f"  最高スコア: {results[0]['total_compatibility']:.3f} ({results[0]['lab_name']})")
    
    # サマリー情報
    high_compatibility = [r for r in results if r["total_compatibility"] >= 0.7]
    avg_score = sum(r["total_compatibility"] for r in results) / len(results) if results else 0
    
    return {
        "evaluation_results": results,
        "summary": {
            "total_labs": len(results),
            "high_compatibility_count": len(high_compatibility),
            "avg_score": avg_score,
            "best_match_lab": results[0]["lab_name"] if results else None,
            "best_match_score": results[0]["total_compatibility"] if results else 0
        },
        "system_info": {
            "matcher_type": "fuzzy_multipath",
            "processing_time_ms": processing_time * 1000,
            "timestamp": datetime.now().isoformat()
        }
    }


def normalize_student_profile(student: Dict[str, Any]) -> Dict[str, Any]:
    """
    学生プロファイルを正規化
    
    ★★★ 重要な修正 ★★★
    1-10スケールの入力を0-1スケールに変換
    """
    normalized = {}
    
    for criterion in EVALUATION_CRITERIA:
        if criterion in student:
            value = student[criterion]
            # 1-10 → 0-1 に正規化
            normalized[criterion] = value / 10.0
        else:
            normalized[criterion] = 0.5  # デフォルト値
        
        # 優先度も正規化
        priority_key = f"{criterion}_priority"
        if priority_key in student:
            normalized[priority_key] = student[priority_key]
        else:
            normalized[priority_key] = 5.0  # デフォルト
    
    # 分野興味
    normalized["field_interests"] = student.get("field_interests", {})
    
    return normalized


# ==================== サーバー起動 ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("研究室選択支援システム バックエンド")
    print("="*60)
    print(f"バージョン: 3.1.0-fixed")
    print(f"ポート: 8000")
    print(f"API ドキュメント: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )