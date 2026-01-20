#!/usr/bin/env python3
"""
研究室選択支援システム - FastAPI バックエンド（詳細情報追加版）
評価が0になる問題を修正 + field_interestsのデータ形式を修正 + 研究室詳細情報を追加
★★★ 説明生成機能追加版 ★★★
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
    version="3.1.3-with-explanations"
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

# 12項目評価基準（research_field_matchは除外）
EVALUATION_CRITERIA = [
    "research_intensity", "advisor_style", "team_work",
    "workload", "theory_practice", "skill_development", 
    "lab_atmosphere", "flexibility", "publication_opportunity", 
    "interdisciplinary", "communication_style"
]

# research_field_matchは分野マッチングの比重を決めるメタパラメータ
META_PARAMETERS = ["research_field_match"]


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
    return { "message": "研究室選択支援システム API", "version": "3.1.3-with-explanations" }


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
    
    # 基本11項目を正規化
    for criterion in EVALUATION_CRITERIA:
        value = student.get(criterion, 5.0)
        normalized[criterion] = (value - 1) / 9.0 if value > 1 else 0.0
        
        priority_key = f"{criterion}_priority"
        normalized[priority_key] = student.get(priority_key, 5.0)
    
    # ★★★ メタパラメータ（research_field_match）を正規化 ★★★
    # これは研究室の特性ではなく、基本スコアと分野スコアの比重を決める
    for meta_param in META_PARAMETERS:
        value = student.get(meta_param, 5.0)
        normalized[meta_param] = (value - 1) / 9.0 if value > 1 else 0.0
        
        priority_key = f"{meta_param}_priority"
        normalized[priority_key] = student.get(priority_key, 5.0)
    
    # ★★★ 修正: field_interests をオブジェクト形式と配列形式の両方に対応 ★★★
    field_interests_data = student.get("field_interests", {})
    
    if isinstance(field_interests_data, dict):
        # すでにオブジェクト形式 {"ai_ml": 10, "image_processing": 8}
        normalized["field_interests"] = field_interests_data
    elif isinstance(field_interests_data, list):
        # 配列形式 [{"field_id": "ai_ml", "interest_level": 10}]
        normalized["field_interests"] = {
            item["field_id"]: item.get("interest_level", 5)
            for item in field_interests_data if "field_id" in item
        }
    else:
        normalized["field_interests"] = {}
    
    # デバッグ出力
    print(f"\n{'='*70}")
    print(f"【デバッグ】学生プロファイル正規化")
    print(f"{'='*70}")
    print(f"受信データ型: {type(field_interests_data)}")
    print(f"正規化前 field_interests: {field_interests_data}")
    print(f"正規化後 field_interests: {normalized['field_interests']}")
    print(f"{'='*70}\n")
    
    return normalized


@app.post("/api/evaluate")
async def evaluate_labs(student_profile: Dict[str, Any] = Body(...)):
    """研究室評価API"""
    if not system_state.get("initialized"):
        raise HTTPException(status_code=503, detail="システムが初期化されていません")
    
    # ★★★ 修正: リクエストボディ全体を学生プロファイルとして扱う ★★★
    # 後方互換性のため、student_profileキーがある場合も対応
    if "student_profile" in student_profile:
        student = student_profile["student_profile"]
    else:
        student = student_profile
    
    # デバッグ: 受信したリクエストを出力
    print(f"\n{'='*70}")
    print(f"【デバッグ】受信したリクエスト")
    print(f"{'='*70}")
    print(f"field_interests型: {type(student.get('field_interests'))}")
    print(f"field_interests内容: {student.get('field_interests')}")
    print(f"基本項目例 - research_intensity: {student.get('research_intensity')}")
    print(f"優先度例 - research_intensity_priority: {student.get('research_intensity_priority')}")
    print(f"{'='*70}\n")
    
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
            
            # ★★★ 研究室の正規化されたfeaturesを作成 ★★★
            normalized_features = {}
            if "features" in lab:
                for criterion in EVALUATION_CRITERIA:
                    if criterion in lab["features"]:
                        value = lab["features"][criterion]
                        # 1-10スケールを0-1スケールに正規化
                        normalized_features[criterion] = (value - 1) / 9.0 if value > 1 else 0.0
            
            # ★★★ 詳細情報を含むレスポンスを構築 ★★★
            lab_result = {
                "lab_id": lab.get("id"),
                "lab_name": lab.get("name"),
                "professor": lab.get("professor", lab.get("professor_name", "")),
                "field_id": lab.get("field_id"),
                "field_name": lab.get("research_area", ""),
                "category": lab.get("category", ""),
                
                # スコア情報
                "overall_compatibility": result.total_compatibility,
                "basic_score": result.basic_score,
                "field_score": result.field_score,
                "field_weight": getattr(result, 'field_weight_alpha', 0.5),
                "basic_weight": getattr(result, 'basic_weight_beta', 0.5),
                
                # 詳細スコア
                "criteria_scores": result.criteria_scores,
                "feature_scores": result.criteria_scores,  # 互換性のため
                "field_detail": getattr(result, 'field_detail', {}),
                
                # 決定木情報
                "tree_path": getattr(result, 'tree_path', ''),
                "tree_layers": getattr(result, 'tree_layers', []),
                "leaf_criteria": getattr(result, 'leaf_criteria', []),
                "fuzzy_paths": getattr(result, 'fuzzy_paths', []),
                
                # ★★★ 推薦情報（3種類の説明を含む） ★★★
                "explanation": result.explanation,                        # 従来版
                "explanation_detailed": getattr(result, 'explanation_detailed', ''),  # 詳細版
                "explanation_short": getattr(result, 'explanation_short', ''),        # 短縮版
                "recommendation": result.recommendation,
                "confidence": getattr(result, 'confidence', 0.85),
                
                # ★★★ 研究室詳細情報（フロントエンド表示用） ★★★
                "description": lab.get("description", ""),
                "specialization": lab.get("specialization", lab.get("research_area", "")),
                "research_fields": lab.get("research_fields", lab.get("keywords", [])),
                
                # ★★★ 研究室の特徴値（正規化済み0-1スケール）- 比較表示用 ★★★
                "features": normalized_features
            }
            results.append(lab_result)
            
        except Exception as e:
            print(f"⚠️ {lab.get('name')} の評価エラー: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    processing_time = time.time() - start_time
    
    # ソート
    results.sort(key=lambda x: x["overall_compatibility"], reverse=True)
    
    system_state["evaluation_count"] += 1
    
    # サマリー情報
    summary = {}
    if results:
        summary = {
            "total_labs": len(results),
            "high_compatibility_count": len([r for r in results if r["overall_compatibility"] >= 0.7]),
            "avg_score": sum(r["overall_compatibility"] for r in results) / len(results),
            "best_match_lab": results[0]["lab_name"],
            "best_match_score": results[0]["overall_compatibility"]
        }

    return {
        "evaluation_results": results,
        "student_profile": normalized_student,  # ★★★ 学生プロファイルを追加 ★★★
        "summary": summary,
        "system_info": {
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "matcher_type": "FuzzyMultiPathMatcher"
        }
    }


# ==================== サーバー起動 ====================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("研究室選択支援システム バックエンド")
    print("="*60)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)