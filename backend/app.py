# backend/app.py - 完全動作版（12項目対応）

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# プロジェクトルート設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# SimpleMatcher インポート
try:
    from core.matching.simple_matcher import SimpleMatcher, CompatibilityResult
    MATCHER_AVAILABLE = True
    print("✅ SimpleMatcher インポート成功")
except ImportError as e:
    MATCHER_AVAILABLE = False
    print(f"❌ SimpleMatcher インポート失敗: {e}")

# FastAPI アプリケーション
app = FastAPI(
    title="研究室選択支援システム",
    description="12項目評価による研究室マッチングシステム",
    version="2.0.0"
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
    "matcher": None,
    "lab_database": [],
    "evaluation_count": 0
}

# 12項目評価基準
EVALUATION_CRITERIA = [
    "research_intensity", "advisor_style", "team_work", 
    "workload", "theory_practice", "research_field_match",
    "skill_development", "lab_atmosphere", "flexibility",
    "publication_opportunity", "interdisciplinary", "communication_style"
]


def load_labs_database() -> List[Dict[str, Any]]:
    """研究室データベース読み込み"""
    
    # パス候補
    db_paths = [
        project_root / "data" / "labs_database.json",
        Path("data/labs_database.json"),
        Path("backend/data/labs_database.json")
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
    
    print("❌ labs_database.json が見つかりません")
    return []


@app.on_event("startup")
async def startup_event():
    """起動時初期化"""
    
    print("\n" + "="*60)
    print("🚀 研究室選択支援システム 起動")
    print("="*60)
    
    # データベース読み込み
    system_state["lab_database"] = load_labs_database()
    
    if not system_state["lab_database"]:
        print("⚠️ 研究室データがありません")
    else:
        print(f"📚 研究室数: {len(system_state['lab_database'])}")
    
    # SimpleMatcher 初期化
    if MATCHER_AVAILABLE:
        try:
            system_state["matcher"] = SimpleMatcher()
            print("✅ SimpleMatcher 初期化完了")
        except Exception as e:
            print(f"❌ SimpleMatcher 初期化失敗: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ SimpleMatcher が利用できません")
    
    system_state["initialized"] = True
    print("✅ システム初期化完了")
    print("="*60 + "\n")


@app.get("/")
async def root():
    """ルート"""
    return {
        "message": "研究室選択支援システム",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "health": "/api/health",
            "labs": "/api/labs",
            "evaluate": "/api/evaluate"
        }
    }


@app.get("/api/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy" if system_state["initialized"] else "initializing",
        "matcher_available": MATCHER_AVAILABLE and system_state["matcher"] is not None,
        "labs_loaded": len(system_state["lab_database"]),
        "evaluation_count": system_state["evaluation_count"]
    }


@app.get("/api/labs")
async def get_labs():
    """研究室一覧"""
    return {
        "labs": system_state["lab_database"],
        "total_count": len(system_state["lab_database"])
    }


@app.post("/api/evaluate")
async def evaluate_compatibility(request_data: Dict[str, Any]):
    """適合度評価"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="システム未初期化")
    
    if not MATCHER_AVAILABLE or not system_state["matcher"]:
        raise HTTPException(status_code=503, detail="SimpleMatcher が利用できません")
    
    if not system_state["lab_database"]:
        raise HTTPException(status_code=503, detail="研究室データがありません")
    
    try:
        print("\n" + "="*60)
        print("📥 評価リクエスト受信")
        
        # student_profile 抽出
        student_profile = request_data.get("student_profile")
        if not student_profile:
            raise HTTPException(status_code=400, detail="student_profile が必要です")
        
        # 12項目検証
        missing = [c for c in EVALUATION_CRITERIA if c not in student_profile]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"不足している項目: {', '.join(missing)}"
            )
        
        print(f"✅ 12項目検証完了")
        
        # マッチング実行
        matcher = system_state["matcher"]
        results = []
        
        for lab in system_state["lab_database"]:
            try:
                # lab の features を取得
                lab_features = lab.get("features", {})
                
                # calculate_compatibility 実行
                match_result = matcher.calculate_compatibility(
                    student=student_profile,
                    lab=lab_features
                )
                
                # 結果構築
                result = {
                    "lab_id": lab["id"],
                    "lab_name": lab["name"],
                    "professor": lab.get("professor", ""),
                    "research_area": lab.get("research_area", ""),
                    "field_id": lab.get("field_id", ""),
                    
                    "overall_compatibility": float(match_result.total_compatibility),
                    "basic_score": float(match_result.basic_score),
                    "field_score": float(match_result.field_score),
                    "field_weight": float(match_result.field_weight_alpha),
                    "basic_weight": float(match_result.basic_weight_beta),
                    
                    "criteria_scores": match_result.criteria_scores,
                    "explanation": match_result.explanation,
                    "recommendation": match_result.recommendation,
                    "confidence": min(1.0, match_result.total_compatibility + 0.1)
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"⚠️ 研究室 {lab.get('id', '?')} 評価エラー: {e}")
                continue
        
        # スコア順ソート
        results.sort(key=lambda x: x["overall_compatibility"], reverse=True)
        
        print(f"✅ 評価完了: {len(results)}研究室")
        
        # 統計計算
        scores = [r["overall_compatibility"] for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        high_count = sum(1 for s in scores if s >= 0.7)
        
        # レスポンス
        response = {
            "student_profile": student_profile,
            "lab_results": results,
            "evaluation_results": results,
            "total_labs_evaluated": len(results),
            "evaluation_timestamp": time.time(),
            
            "summary": {
                "total_labs": len(results),
                "avg_score": avg_score,
                "avg_compatibility": avg_score,
                "best_match_lab": results[0]["lab_name"] if results else None,
                "best_match_score": results[0]["overall_compatibility"] if results else 0,
                "high_compatibility_count": high_count
            },
            
            "metadata": {
                "evaluation_method": "simple_matcher_12_criteria",
                "criteria_count": 12
            },
            
            "system_info": {
                "pattern": "A",
                "matcher_type": "simple",
                "evaluation_count": system_state["evaluation_count"] + 1
            }
        }
        
        system_state["evaluation_count"] += 1
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"\n❌ 評価エラー:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"評価エラー: {str(e)}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 サーバー起動")
    print("📍 URL: http://localhost:8000")
    print("📚 ドキュメント: http://localhost:8000/docs")
    print("="*60 + "\n")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )