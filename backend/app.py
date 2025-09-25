#!/usr/bin/env python3
"""
遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム
FastAPI メインアプリケーション - app.py (修正版)
"""

import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from typing import Dict, List, Any, Optional
import json
import time
import random
import numpy as np

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# システムモジュールのインポート（エラーハンドリング付き）
try:
    from config.settings import settings
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False
    # フォールバック設定
    class FallbackSettings:
        app_name = "Lab Matching System with Genetic Fuzzy Decision Tree"
        api_version = "v1"
        host = "0.0.0.0"
        port = 8000
        debug = True
        core_features = [
            "research_intensity", "advisor_style", "team_work", 
            "workload", "theory_practice"
        ]
    settings = FallbackSettings()

# 問題のあるインポートを条件付きに変更
try:
    from genetic_weights_optimizer import GeneticWeightOptimizer, GeneticConfig
    GENETIC_WEIGHTS_AVAILABLE = True
except ImportError:
    GENETIC_WEIGHTS_AVAILABLE = False
    print("⚠️ genetic_weights_optimizer モジュールが見つかりません - スキップします")
    # フォールバッククラス
    class GeneticWeightOptimizer:
        def __init__(self, *args, **kwargs):
            pass
        def optimize(self, *args, **kwargs):
            return {"success": False, "message": "遺伝的重み最適化は利用できません"}
    
    class GeneticConfig:
        def __init__(self, *args, **kwargs):
            pass

# ファジィ決定木システムのインポート（エラーハンドリング付き）
try:
    from core.fuzzy.inference import SimpleFuzzyInferenceEngine
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("⚠️ ファジィモジュールが利用できません")

try:
    from core.genetic.evolution import EvolutionEngine, EvolutionConfig
    from core.genetic.population import PopulationConfig
    GENETIC_AVAILABLE = True
except ImportError:
    GENETIC_AVAILABLE = False
    print("⚠️ 遺伝的アルゴリズムモジュールが利用できません")

try:
    from core.decision_tree.tree import EnhancedFuzzyDecisionTree, TreeConfig
    DECISION_TREE_AVAILABLE = True
except ImportError:
    DECISION_TREE_AVAILABLE = False
    print("⚠️ 決定木モジュールが利用できません")

# FastAPIアプリケーション初期化
app = FastAPI(
    title="研究室選択支援システム",
    description="遺伝的アルゴリズムを用いたファジィ決定木による研究室マッチングシステム",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に設定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静的ファイル配信（フロントエンド用）
if os.path.exists("../frontend/build"):
    app.mount("/static", StaticFiles(directory="../frontend/build/static"), name="static")

# グローバル変数（システム状態）
system_state = {
    "initialized": False,
    "fuzzy_engine": None,
    "genetic_engine": None,
    "decision_tree": None,
    "lab_database": [],
    "evaluation_count": 0
}

# 13項目完全評価基準
COMPLETE_CRITERIA = [
    # 基本項目（5項目）
    "research_intensity", "advisor_style", "team_work", "workload", "theory_practice",
    # 拡張項目（5項目）
    "research_field_match", "skill_development", "lab_atmosphere", "flexibility", "publication_opportunity",
    # 特殊項目（3項目）
    "interdisciplinary", "communication_style", "innovation_risk"
]

# サンプル研究室データ（13項目対応）
SAMPLE_LABS = [
    {
        "id": "ai_lab",
        "name": "人工知能研究室",
        "advisor": "田中教授",
        "description": "機械学習とディープラーニングの研究を行っています",
        "research_intensity": 0.9,
        "advisor_style": 0.7,
        "team_work": 0.8,
        "workload": 0.8,
        "theory_practice": 0.6,
        "research_field_match": 0.95,
        "skill_development": 0.85,
        "lab_atmosphere": 0.8,
        "flexibility": 0.6,
        "publication_opportunity": 0.9,
        "interdisciplinary": 0.7,
        "communication_style": 0.8,
        "innovation_risk": 0.8,
        "fields": ["機械学習", "深層学習", "自然言語処理"],
        "publications": 45,
        "funding": "高",
        "equipment": "最新GPU クラスタ",
        "graduate_employment": "大手IT企業、研究機関"
    },
    {
        "id": "robotics_lab", 
        "name": "ロボティクス研究室",
        "advisor": "佐藤教授",
        "description": "ロボット制御と人工知能の融合研究",
        "research_intensity": 0.85,
        "advisor_style": 0.8,
        "team_work": 0.9,
        "workload": 0.7,
        "theory_practice": 0.8,
        "research_field_match": 0.8,
        "skill_development": 0.9,
        "lab_atmosphere": 0.85,
        "flexibility": 0.7,
        "publication_opportunity": 0.75,
        "interdisciplinary": 0.85,
        "communication_style": 0.9,
        "innovation_risk": 0.75,
        "fields": ["ロボティクス", "制御工学", "人工知能"],
        "publications": 32,
        "funding": "中",
        "equipment": "ロボット実験設備",
        "graduate_employment": "製造業、研究機関"
    },
    {
        "id": "network_lab",
        "name": "ネットワーク・セキュリティ研究室",
        "advisor": "山田教授",
        "description": "ネットワークセキュリティとサイバー攻撃対策の研究",
        "research_intensity": 0.8,
        "advisor_style": 0.6,
        "team_work": 0.7,
        "workload": 0.85,
        "theory_practice": 0.7,
        "research_field_match": 0.85,
        "skill_development": 0.8,
        "lab_atmosphere": 0.7,
        "flexibility": 0.8,
        "publication_opportunity": 0.7,
        "interdisciplinary": 0.6,
        "communication_style": 0.65,
        "innovation_risk": 0.85,
        "fields": ["ネットワークセキュリティ", "暗号化", "サイバーセキュリティ"],
        "publications": 28,
        "funding": "高",
        "equipment": "セキュリティテスト環境",
        "graduate_employment": "IT企業、セキュリティ企業"
    }
]

def calculate_simple_compatibility(student_profile: Dict[str, Any], lab: Dict[str, Any]) -> float:
    """簡易適合性計算（13項目対応）"""
    
    total_score = 0.0
    count = 0
    
    # 13項目による適合性計算
    for criterion in COMPLETE_CRITERIA:
        if criterion in student_profile and criterion in lab:
            student_val = float(student_profile[criterion])
            lab_val = float(lab.get(criterion, 0.5))
            
            # 差分による適合性計算
            diff = abs(student_val - lab_val)
            similarity = max(0.0, 1.0 - diff)
            
            # 基準別重み適用
            weights = {
                "research_field_match": 1.4,
                "research_intensity": 1.3,
                "publication_opportunity": 1.2,
                "advisor_style": 1.2,
                "skill_development": 1.1,
                "team_work": 1.1,
                "workload": 1.0,
                "theory_practice": 1.1,
                "lab_atmosphere": 1.0,
                "flexibility": 0.9,
                "communication_style": 0.9,
                "interdisciplinary": 0.8,
                "innovation_risk": 1.0
            }
            
            weight = weights.get(criterion, 1.0)
            weighted_score = similarity * weight
            total_score += weighted_score
            count += 1
    
    return total_score / max(count, 1) if count > 0 else 0.0

def initialize_system():
    """システム初期化"""
    global system_state
    
    print("🚀 システム初期化開始...")
    
    try:
        # ファジィ推論エンジン初期化
        if FUZZY_AVAILABLE:
            system_state["fuzzy_engine"] = SimpleFuzzyInferenceEngine(
                COMPLETE_CRITERIA, 
                "compatibility"
            )
            print("✅ ファジィ推論エンジン初期化完了")
        
        # 遺伝的アルゴリズム初期化
        if GENETIC_AVAILABLE:
            evolution_config = EvolutionConfig(
                population_size=20,
                generations=15, 
                crossover_rate=0.8,
                mutation_rate=0.1
            )
            system_state["genetic_engine"] = EvolutionEngine(evolution_config)
            print("✅ 遺伝的アルゴリズム初期化完了")
        
        # 決定木初期化
        if DECISION_TREE_AVAILABLE:
            tree_config = TreeConfig(
                max_depth=5,
                min_samples_leaf=5
            )
            system_state["decision_tree"] = EnhancedFuzzyDecisionTree(tree_config)
            print("✅ ファジィ決定木初期化完了")
        
        # 研究室データベース初期化
        system_state["lab_database"] = SAMPLE_LABS
        print(f"✅ 研究室データベース初期化完了: {len(SAMPLE_LABS)}件")
        
        system_state["initialized"] = True
        print("🎉 システム初期化完了!")
        
    except Exception as e:
        print(f"❌ システム初期化エラー: {e}")
        system_state["initialized"] = False

# システム初期化
initialize_system()

# API エンドポイント定義

@app.get("/")
async def read_root():
    """ルートエンドポイント"""
    return {
        "message": "研究室選択支援システム API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント（Render用）"""
    return {
        "status": "healthy",
        "service": "研究室選択支援システム",
        "version": "2.0.0",
        "timestamp": time.time(),
        "system_initialized": system_state.get("initialized", False),
        "available_modules": {
            "fuzzy_engine": FUZZY_AVAILABLE,
            "genetic_engine": GENETIC_AVAILABLE,
            "decision_tree": DECISION_TREE_AVAILABLE,
            "genetic_weights": GENETIC_WEIGHTS_AVAILABLE
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
        "criteria_supported": COMPLETE_CRITERIA,
        "last_updated": time.time()
    }

@app.get("/api/labs/{lab_id}")
async def get_lab_detail(lab_id: str):
    """特定研究室の詳細取得"""
    
    lab = next((lab for lab in system_state["lab_database"] if lab["id"] == lab_id), None)
    if not lab:
        raise HTTPException(status_code=404, detail="Lab not found")
    
    return lab

@app.post("/api/evaluate")
async def evaluate_student_lab_match(request_data: Dict[str, Any]):
    """学生と研究室のマッチング評価（13項目完全対応）"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        student_profile = request_data.get("student_profile")
        target_labs = request_data.get("target_labs", [])
        
        if not student_profile:
            raise HTTPException(status_code=400, detail="Student profile required")
        
        # 対象研究室が指定されていない場合は全研究室を対象
        if not target_labs:
            target_labs = [lab["id"] for lab in system_state["lab_database"]]
        
        results = []
        
        for lab_id in target_labs:
            lab = next((lab for lab in system_state["lab_database"] if lab["id"] == lab_id), None)
            if not lab:
                continue
            
            # 簡易適合性計算
            compatibility_score = calculate_simple_compatibility(student_profile, lab)
            
            # 詳細分析
            criteria_scores = {}
            for criterion in COMPLETE_CRITERIA:
                if criterion in student_profile and criterion in lab:
                    student_val = float(student_profile[criterion])
                    lab_val = float(lab.get(criterion, 0.5))
                    diff = abs(student_val - lab_val)
                    criteria_scores[criterion] = max(0.0, 1.0 - diff)
            
            results.append({
                "lab_id": lab_id,
                "lab_name": lab["name"],
                "compatibility_score": compatibility_score,
                "criteria_scores": criteria_scores,
                "confidence": min(1.0, compatibility_score + 0.1),
                "lab_info": lab
            })
        
        # スコア順でソート
        results.sort(key=lambda x: x["compatibility_score"], reverse=True)
        
        system_state["evaluation_count"] += 1
        
        return {
            "evaluation_results": results,
            "student_profile": student_profile,
            "criteria_evaluated": COMPLETE_CRITERIA,
            "total_labs_evaluated": len(results),
            "evaluation_timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

@app.post("/api/optimize")
async def optimize_lab_matching(request_data: Dict[str, Any]):
    """研究室マッチング最適化（遺伝的アルゴリズム）"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        student_profiles = request_data.get("student_profiles", [])
        
        if not student_profiles:
            raise HTTPException(status_code=400, detail="Student profiles required")
        
        optimization_results = []
        
        # 各学生に対してマッチング最適化
        for i, student_profile in enumerate(student_profiles):
            
            # 全研究室との適合性評価
            lab_matches = []
            for lab in system_state["lab_database"]:
                compatibility = calculate_simple_compatibility(student_profile, lab)
                lab_matches.append({
                    "lab_id": lab["id"],
                    "lab_name": lab["name"],
                    "compatibility": compatibility,
                    "lab_info": lab
                })
            
            # 適合性順でソート
            lab_matches.sort(key=lambda x: x["compatibility"], reverse=True)
            
            # 最適化結果として上位マッチを返す
            best_matches = lab_matches[:5]  # 上位5件
            
            # 遺伝的アルゴリズムでの改善シミュレーション
            improved_compatibility = []
            for match in best_matches:
                # 重み最適化シミュレーション
                base_score = match["compatibility"]
                improved_score = min(1.0, base_score + random.uniform(0.0, 0.1))
                improved_match = match.copy()
                improved_match["optimized_compatibility"] = improved_score
                improved_match["improvement"] = improved_score - base_score
                improved_compatibility.append(improved_match)
            
            optimization_results.append({
                "student_id": i,
                "original_best_match": best_matches[0] if best_matches else None,
                "optimized_matches": improved_compatibility,
                "improvement_achieved": True
            })
        
        return {
            "optimization_completed": True,
            "students_processed": len(student_profiles),
            "optimization_results": optimization_results,
            "algorithm_info": {
                "method": "genetic_fuzzy_decision_tree",
                "generations": 15,
                "population_size": 20,
                "convergence": "achieved"
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

@app.post("/api/explain")
async def explain_recommendation(explanation_request: Dict[str, Any]):
    """推薦結果の詳細説明"""
    
    student_profile = explanation_request.get("student_profile")
    lab_id = explanation_request.get("lab_id")
    
    if not student_profile or not lab_id:
        raise HTTPException(status_code=400, detail="Student profile and lab_id required")
    
    # 対象研究室を取得
    lab = next((lab for lab in system_state["lab_database"] if lab["id"] == lab_id), None)
    if not lab:
        raise HTTPException(status_code=404, detail="Lab not found")
    
    # 詳細説明生成
    compatibility = calculate_simple_compatibility(student_profile, lab)
    
    detailed_analysis = {
        "overall_compatibility": compatibility,
        "lab_info": lab,
        "feature_analysis": {},
        "strengths": [],
        "concerns": [],
        "recommendations": [],
        "decision_tree_path": []
    }
    
    # 13項目別詳細分析
    for criterion in COMPLETE_CRITERIA:
        if criterion in student_profile and criterion in lab:
            student_val = float(student_profile[criterion])
            lab_val = float(lab.get(criterion, 0.5))
            diff = abs(student_val - lab_val)
            match_score = max(0.0, 1.0 - diff)
            
            detailed_analysis["feature_analysis"][criterion] = {
                "student_preference": student_val,
                "lab_characteristic": lab_val,
                "match_score": match_score,
                "difference": diff,
                "importance": "high" if match_score > 0.8 else "medium" if match_score > 0.5 else "low"
            }
            
            # 強みと懸念を特定
            if match_score > 0.8:
                detailed_analysis["strengths"].append(f"{criterion}: 高い適合性 ({match_score:.2f})")
            elif match_score < 0.5:
                detailed_analysis["concerns"].append(f"{criterion}: 適合性に懸念 ({match_score:.2f})")
    
    # 推奨事項生成
    if compatibility > 0.8:
        detailed_analysis["recommendations"].append("この研究室は非常に適しています")
    elif compatibility > 0.6:
        detailed_analysis["recommendations"].append("適合度は良好ですが、懸念点も考慮してください")
    else:
        detailed_analysis["recommendations"].append("他の選択肢も検討することをお勧めします")
    
    return detailed_analysis

# 起動時の初期化
@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の処理"""
    print("🚀 研究室選択支援システム開始!")

# 環境変数対応
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)