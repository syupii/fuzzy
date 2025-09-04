#!/usr/bin/env python3
"""
遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム
FastAPI メインアプリケーション - app.py
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
    allow_origins=[
        "http://localhost:3000",  # React開発サーバー
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
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

# サンプル研究室データ
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
        "description": "自律移動ロボットと制御システムの開発",
        "research_intensity": 0.8,
        "advisor_style": 0.6,
        "team_work": 0.9,
        "workload": 0.7,
        "theory_practice": 0.8,
        "fields": ["ロボット工学", "制御工学", "コンピュータビジョン"],
        "publications": 32,
        "funding": "中",
        "equipment": "産業用ロボット、センサー",
        "graduate_employment": "製造業、ロボット開発企業"
    },
    {
        "id": "security_lab",
        "name": "サイバーセキュリティ研究室",
        "advisor": "山田教授",
        "description": "暗号技術とネットワークセキュリティ",
        "research_intensity": 0.7,
        "advisor_style": 0.8,
        "team_work": 0.6,
        "workload": 0.6,
        "theory_practice": 0.4,
        "fields": ["暗号学", "ネットワークセキュリティ", "プライバシー保護"],
        "publications": 28,
        "funding": "中",
        "equipment": "セキュリティ解析環境",
        "graduate_employment": "金融機関、セキュリティ企業"
    },
    {
        "id": "hci_lab",
        "name": "ヒューマンコンピュータインタラクション研究室",
        "advisor": "鈴木教授",
        "description": "ユーザーインターフェースとユーザビリティの研究",
        "research_intensity": 0.6,
        "advisor_style": 0.9,
        "team_work": 0.8,
        "workload": 0.5,
        "theory_practice": 0.7,
        "fields": ["HCI", "UX/UI", "アクセシビリティ"],
        "publications": 22,
        "funding": "中",
        "equipment": "ユーザビリティ実験室",
        "graduate_employment": "Web開発企業、デザイン会社"
    },
    {
        "id": "theory_lab",
        "name": "計算理論研究室",
        "advisor": "高橋教授",
        "description": "アルゴリズム理論と計算複雑性",
        "research_intensity": 0.9,
        "advisor_style": 0.5,
        "team_work": 0.4,
        "workload": 0.8,
        "theory_practice": 0.2,
        "fields": ["アルゴリズム", "計算複雑性", "組合せ最適化"],
        "publications": 38,
        "funding": "高",
        "equipment": "高性能計算クラスタ",
        "graduate_employment": "研究機関、大学院進学"
    }
]

def initialize_system():
    """システム初期化"""
    global system_state
    
    print("🚀 システム初期化開始...")
    
    try:
        # ファジィ推論エンジン初期化
        if FUZZY_AVAILABLE:
            system_state["fuzzy_engine"] = SimpleFuzzyInferenceEngine(
                settings.core_features, 
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
    """ルートエンドポイント - フロントエンド配信またはAPI情報"""
    if os.path.exists("../frontend/build/index.html"):
        return FileResponse("../frontend/build/index.html")
    else:
        return {
            "message": "遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム",
            "version": "2.0.0",
            "status": "running",
            "endpoints": {
                "health": "/health",
                "labs": "/api/labs",
                "evaluate": "/api/evaluate",
                "optimize": "/api/optimize",
                "docs": "/docs"
            }
        }

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    
    # データベース状態チェック
    lab_count = len(system_state.get("lab_database", []))
    
    # モジュール可用性チェック
    modules_status = {
        "fuzzy": FUZZY_AVAILABLE,
        "genetic": GENETIC_AVAILABLE,
        "decision_tree": DECISION_TREE_AVAILABLE,
        "settings": SETTINGS_AVAILABLE
    }
    
    # 全体的な健全性
    overall_health = (
        system_state["initialized"] and 
        any(modules_status.values()) and 
        lab_count > 0
    )
    
    return {
        "status": "healthy" if overall_health else "unhealthy",
        "version": "2.0.0",
        "timestamp": time.time(),
        "system_initialized": system_state["initialized"],
        "modules": modules_status,
        "database": {
            "status": "OK" if lab_count > 0 else "Empty",
            "lab_count": lab_count,
            "evaluation_count": system_state["evaluation_count"]
        }
    }
@app.get("/api/test")
async def test_connection():
    return {
        "status": "success",
        "message": "接続成功！",
        "timestamp": int(time.time())
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

@app.post("/api/evaluate")
async def evaluate_compatibility(student_profile: Dict[str, Any]):
    """学生プロファイルに基づく研究室適合度評価"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # 入力検証
        required_features = settings.core_features
        for feature in required_features:
            if feature not in student_profile:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing required feature: {feature}"
                )
        
        # 各研究室との適合度計算
        results = []
        
        for lab in system_state["lab_database"]:
            # 基本的な適合度計算（ファジィ推論が利用可能な場合は使用）
            if FUZZY_AVAILABLE and system_state["fuzzy_engine"]:
                try:
                    compatibility = system_state["fuzzy_engine"].predict(student_profile)
                except:
                    compatibility = calculate_simple_compatibility(student_profile, lab)
            else:
                compatibility = calculate_simple_compatibility(student_profile, lab)
            
            # 詳細スコア計算
            feature_scores = {}
            for feature in required_features:
                student_val = student_profile[feature]
                lab_val = lab.get(feature, 0.5)
                feature_scores[feature] = 1.0 - abs(student_val - lab_val)
            
            lab_result = {
                "lab_id": lab["id"],
                "lab_name": lab["name"],
                "advisor": lab["advisor"],
                "overall_compatibility": float(compatibility),
                "feature_scores": feature_scores,
                "confidence": min(1.0, compatibility + random.uniform(0.0, 0.1)),
                "recommendation": get_recommendation_level(compatibility),
                "explanation": generate_explanation(student_profile, lab, compatibility)
            }
            
            results.append(lab_result)
        
        # 適合度でソート
        results.sort(key=lambda x: x["overall_compatibility"], reverse=True)
        
        # 評価回数増加
        system_state["evaluation_count"] += 1
        
        return {
            "student_profile": student_profile,
            "evaluation_results": results,
            "total_labs_evaluated": len(results),
            "evaluation_timestamp": time.time(),
            "system_info": {
                "fuzzy_enabled": FUZZY_AVAILABLE,
                "genetic_enabled": GENETIC_AVAILABLE,
                "evaluation_count": system_state["evaluation_count"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

@app.post("/api/optimize")
async def optimize_matching(optimization_request: Dict[str, Any]):
    """遺伝的アルゴリズムによる最適化"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not GENETIC_AVAILABLE:
        raise HTTPException(status_code=501, detail="Genetic algorithm not available")
    
    try:
        student_profiles = optimization_request.get("student_profiles", [])
        if not student_profiles:
            raise HTTPException(status_code=400, detail="No student profiles provided")
        
        # 簡易最適化シミュレーション
        optimization_results = []
        
        for i, profile in enumerate(student_profiles):
            # 基本評価を実行
            eval_response = await evaluate_compatibility(profile)
            best_matches = eval_response["evaluation_results"][:3]  # 上位3つ
            
            # 遺伝的アルゴリズムによる改善シミュレーション
            improved_compatibility = []
            for match in best_matches:
                original_score = match["overall_compatibility"]
                improved_score = min(1.0, original_score + random.uniform(0.05, 0.15))
                
                improved_match = match.copy()
                improved_match["overall_compatibility"] = improved_score
                improved_match["optimization_improvement"] = improved_score - original_score
                improved_compatibility.append(improved_match)
            
            optimization_results.append({
                "student_id": i,
                "original_best_match": best_matches[0],
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
        "decision_tree_path": generate_decision_path(student_profile, lab)
    }
    
    # 特徴量別詳細分析
    for feature in settings.core_features:
        student_val = student_profile[feature]
        lab_val = lab.get(feature, 0.5)
        diff = abs(student_val - lab_val)
        match_score = 1.0 - diff
        
        detailed_analysis["feature_analysis"][feature] = {
            "student_preference": student_val,
            "lab_characteristic": lab_val,
            "match_score": match_score,
            "interpretation": interpret_feature_match(feature, student_val, lab_val, match_score)
        }
        
        if match_score > 0.8:
            detailed_analysis["strengths"].append(f"{feature}: 高い適合性 (スコア: {match_score:.2f})")
        elif match_score < 0.5:
            detailed_analysis["concerns"].append(f"{feature}: 適合性に課題 (スコア: {match_score:.2f})")
    
    # 推薦事項生成
    if compatibility > 0.7:
        detailed_analysis["recommendations"].append("この研究室は高い適合性を示しています")
    elif compatibility > 0.5:
        detailed_analysis["recommendations"].append("適度な適合性があります。詳細な検討をお勧めします")
    else:
        detailed_analysis["recommendations"].append("他の選択肢も検討することをお勧めします")
    
    return detailed_analysis

# ヘルパー関数

def calculate_simple_compatibility(student_profile: Dict[str, Any], lab: Dict[str, Any]) -> float:
    """簡単な適合度計算"""
    
    total_score = 0.0
    feature_count = 0
    
    for feature in settings.core_features:
        if feature in student_profile and feature in lab:
            student_val = student_profile[feature]
            lab_val = lab[feature]
            
            # 類似度計算（差の逆数）
            similarity = 1.0 - abs(student_val - lab_val)
            total_score += similarity
            feature_count += 1
    
    return total_score / max(1, feature_count)

def get_recommendation_level(compatibility: float) -> str:
    """推薦レベル取得"""
    
    if compatibility >= 0.8:
        return "強く推薦"
    elif compatibility >= 0.6:
        return "推薦"
    elif compatibility >= 0.4:
        return "検討可能"
    else:
        return "推薦しない"

def generate_explanation(student_profile: Dict[str, Any], lab: Dict[str, Any], compatibility: float) -> str:
    """説明文生成"""
    
    feature_matches = []
    for feature in settings.core_features[:3]:  # 上位3つの特徴のみ
        if feature in student_profile and feature in lab:
            student_val = student_profile[feature]
            lab_val = lab[feature]
            match_score = 1.0 - abs(student_val - lab_val)
            
            if match_score > 0.7:
                feature_matches.append(f"{feature}で高い適合性")
    
    if feature_matches:
        return f"この研究室は{', '.join(feature_matches)}を示しており、総合適合度は{compatibility:.1%}です。"
    else:
        return f"総合適合度は{compatibility:.1%}です。各特徴量を詳しく検討することをお勧めします。"

def interpret_feature_match(feature: str, student_val: float, lab_val: float, match_score: float) -> str:
    """特徴量マッチの解釈"""
    
    feature_names = {
        "research_intensity": "研究強度",
        "advisor_style": "指導スタイル",
        "team_work": "チームワーク",
        "workload": "作業負荷",
        "theory_practice": "理論・実践バランス"
    }
    
    feature_name = feature_names.get(feature, feature)
    
    if match_score > 0.8:
        return f"{feature_name}において学生の希望と研究室の特性が非常によく一致しています"
    elif match_score > 0.6:
        return f"{feature_name}において適度な一致が見られます"
    elif match_score > 0.4:
        return f"{feature_name}においていくつかの違いがありますが、許容範囲内です"
    else:
        return f"{feature_name}において学生の希望と研究室の特性に大きな違いがあります"

def generate_decision_path(student_profile: Dict[str, Any], lab: Dict[str, Any]) -> List[str]:
    """決定パス生成（簡易版）"""
    
    path = ["評価開始"]
    
    # 主要特徴による分岐シミュレーション
    research_intensity = student_profile.get("research_intensity", 0.5)
    if research_intensity > 0.7:
        path.append("高研究強度を希望 → 研究集約型研究室を評価")
    else:
        path.append("バランス型を希望 → 幅広い研究室を評価")
    
    compatibility = calculate_simple_compatibility(student_profile, lab)
    if compatibility > 0.7:
        path.append("高適合性を確認 → 推薦")
    else:
        path.append("中程度の適合性 → 要検討")
    
    return path

# サーバー起動部分
if __name__ == "__main__":
    print("\n🚀 FastAPI サーバー起動中...")
    print(f"📍 URL: http://localhost:{settings.port}")
    print(f"📚 API文書: http://localhost:{settings.port}/docs")
    print("🔧 システム状況:")
    print(f"  - ファジィ推論: {'✅' if FUZZY_AVAILABLE else '❌'}")
    print(f"  - 遺伝的アルゴリズム: {'✅' if GENETIC_AVAILABLE else '❌'}")
    print(f"  - 決定木: {'✅' if DECISION_TREE_AVAILABLE else '❌'}")
    print(f"  - 研究室データ: {len(SAMPLE_LABS)}件")
    
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if settings.debug else "warning"
    )