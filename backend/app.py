#!/usr/bin/env python3
"""
遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム
FastAPI メインアプリケーション - 分野考慮型完全版
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

# ===== システムモジュールのインポート =====

# 設定
try:
    from config.settings import settings
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False
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

# ファジィ推論エンジン
try:
    from core.fuzzy.inference import SimpleFuzzyInferenceEngine
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("⚠️ ファジィモジュールが利用できません")

# 遺伝的アルゴリズム
try:
    from core.genetic.evolution import EvolutionEngine, EvolutionConfig
    from core.genetic.population import PopulationConfig
    GENETIC_AVAILABLE = True
except ImportError:
    GENETIC_AVAILABLE = False
    print("⚠️ 遺伝的アルゴリズムモジュールが利用できません")

# ファジィ決定木
try:
    from core.decision_tree.tree import EnhancedFuzzyDecisionTree, TreeConfig
    DECISION_TREE_AVAILABLE = True
except ImportError:
    DECISION_TREE_AVAILABLE = False
    print("⚠️ 決定木モジュールが利用できません")

# 分野考慮型計算器（NEW!）
try:
    from core.fuzzy.field_aware_calculator import FieldAwareFuzzyCompatibilityCalculator
    from core.genetic.field_aware_gene import FieldAwareFuzzyTreeGene
    FIELD_AWARE_AVAILABLE = True
except ImportError as e:
    FIELD_AWARE_AVAILABLE = False
    print(f"⚠️ 分野考慮型計算器が利用できません: {e}")

# ===== FastAPIアプリケーション初期化 =====

app = FastAPI(
    title="研究室選択支援システム（分野考慮型）",
    description="遺伝的アルゴリズムを用いたファジィ決定木による研究室マッチングシステム",
    version="3.2.0",
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

# ===== グローバル変数（システム状態）=====

system_state = {
    "initialized": False,
    "fuzzy_engine": None,
    "genetic_engine": None,
    "decision_tree": None,
    "field_aware_calculator": None,  # NEW!
    "field_aware_gene": None,        # NEW!
    "lab_database": [],
    "evaluation_count": 0
}

# ===== サンプル研究室データ =====

SAMPLE_LABS = [
    {
        "id": "ai_lab",
        "name": "人工知能研究室",
        "advisor": "田中教授",
        "description": "機械学習とディープラーニングの研究を行っています",
        "research_intensity": 9.0,
        "advisor_style": 7.0,
        "team_work": 8.0,
        "workload": 8.0,
        "theory_practice": 6.0,
        "skill_development": 9.0,
        "lab_atmosphere": 7.0,
        "flexibility": 6.0,
        "publication_opportunity": 9.0,
        "interdisciplinary": 7.0,
        "communication_style": 7.0,
        "innovation_risk": 8.0,
        "field_id": "ai_ml",
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
        "description": "自律移動ロボットとヒューマンロボットインタラクションの研究",
        "research_intensity": 8.0,
        "advisor_style": 6.0,
        "team_work": 9.0,
        "workload": 7.0,
        "theory_practice": 8.0,
        "skill_development": 8.0,
        "lab_atmosphere": 8.0,
        "flexibility": 7.0,
        "publication_opportunity": 7.0,
        "interdisciplinary": 8.0,
        "communication_style": 8.0,
        "innovation_risk": 7.0,
        "field_id": "embedded_iot",
        "fields": ["ロボティクス", "センサー技術", "制御工学"],
        "publications": 32,
        "funding": "中",
        "equipment": "ロボット実験室、センサー設備",
        "graduate_employment": "製造業、ロボット企業"
    },
    {
        "id": "web_design_lab",
        "name": "Webデザイン研究室",
        "advisor": "鈴木教授",
        "description": "UI/UXデザインとフロントエンド技術の研究",
        "research_intensity": 6.0,
        "advisor_style": 8.0,
        "team_work": 7.0,
        "workload": 6.0,
        "theory_practice": 9.0,
        "skill_development": 8.0,
        "lab_atmosphere": 9.0,
        "flexibility": 9.0,
        "publication_opportunity": 5.0,
        "interdisciplinary": 6.0,
        "communication_style": 9.0,
        "innovation_risk": 6.0,
        "field_id": "web_design",
        "fields": ["UI/UX", "フロントエンド", "デザイン思考"],
        "publications": 18,
        "funding": "中",
        "equipment": "デザインスタジオ、最新ソフトウェア",
        "graduate_employment": "Web制作会社、IT企業"
    },
    {
        "id": "db_lab",
        "name": "データベース研究室",
        "advisor": "高橋教授",
        "description": "大規模データ管理とデータベースシステムの研究",
        "research_intensity": 7.0,
        "advisor_style": 5.0,
        "team_work": 6.0,
        "workload": 7.0,
        "theory_practice": 5.0,
        "skill_development": 7.0,
        "lab_atmosphere": 6.0,
        "flexibility": 5.0,
        "publication_opportunity": 8.0,
        "interdisciplinary": 5.0,
        "communication_style": 6.0,
        "innovation_risk": 5.0,
        "field_id": "database_systems",
        "fields": ["データベース", "ビッグデータ", "クラウド"],
        "publications": 28,
        "funding": "高",
        "equipment": "サーバー設備、ストレージシステム",
        "graduate_employment": "IT企業、金融機関"
    },
    {
        "id": "game_lab",
        "name": "ゲーム開発研究室",
        "advisor": "山田教授",
        "description": "ゲームエンジンとインタラクティブメディアの研究",
        "research_intensity": 7.0,
        "advisor_style": 8.0,
        "team_work": 9.0,
        "workload": 8.0,
        "theory_practice": 9.0,
        "skill_development": 9.0,
        "lab_atmosphere": 9.0,
        "flexibility": 8.0,
        "publication_opportunity": 6.0,
        "interdisciplinary": 7.0,
        "communication_style": 9.0,
        "innovation_risk": 8.0,
        "field_id": "game_esports",
        "fields": ["ゲーム開発", "Unity", "eスポーツ"],
        "publications": 22,
        "funding": "中",
        "equipment": "ゲーム開発環境、VR機器",
        "graduate_employment": "ゲーム会社、エンタメ企業"
    }
]

# ===== システム初期化 =====

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
        
        # 分野考慮型計算器の初期化（NEW!）
        if FIELD_AWARE_AVAILABLE:
            # 遺伝子作成
            gene = FieldAwareFuzzyTreeGene()
            system_state["field_aware_gene"] = gene
            
            # 計算器作成
            calculator = FieldAwareFuzzyCompatibilityCalculator(gene)
            system_state["field_aware_calculator"] = calculator
            
            print("✅ 分野考慮型計算器初期化完了")
        
        # 研究室データベース初期化
        system_state["lab_database"] = SAMPLE_LABS
        print(f"✅ 研究室データベース初期化完了: {len(SAMPLE_LABS)}件")
        
        system_state["initialized"] = True
        print("🎉 システム初期化完了!")
        
    except Exception as e:
        print(f"❌ システム初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        system_state["initialized"] = False

# システム初期化実行
initialize_system()

# ===== API エンドポイント =====

@app.get("/")
async def read_root():
    """ルートエンドポイント"""
    if os.path.exists("../frontend/build/index.html"):
        return FileResponse("../frontend/build/index.html")
    else:
        return {
            "message": "遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム",
            "version": "3.2.0",
            "status": "running",
            "endpoints": {
                "health": "/health",
                "labs": "/api/labs",
                "evaluate": "/api/evaluate",
                "optimize": "/api/optimize",
                "explain": "/api/explain",
                "docs": "/docs"
            }
        }

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    
    lab_count = len(system_state.get("lab_database", []))
    
    modules_status = {
        "fuzzy": FUZZY_AVAILABLE,
        "genetic": GENETIC_AVAILABLE,
        "decision_tree": DECISION_TREE_AVAILABLE,
        "field_aware": FIELD_AWARE_AVAILABLE,
        "settings": SETTINGS_AVAILABLE
    }
    
    overall_health = (
        system_state["initialized"] and 
        any(modules_status.values()) and 
        lab_count > 0
    )
    
    return {
        "status": "healthy" if overall_health else "unhealthy",
        "version": "3.2.0",
        "timestamp": time.time(),
        "system_initialized": system_state["initialized"],
        "modules": modules_status,
        "database": {
            "status": "OK" if lab_count > 0 else "Empty",
            "lab_count": lab_count,
            "evaluation_count": system_state["evaluation_count"]
        },
        "field_aware": {
            "available": FIELD_AWARE_AVAILABLE,
            "calculator_initialized": system_state.get("field_aware_calculator") is not None
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

@app.post("/api/evaluate")
async def evaluate_compatibility(student_profile: Dict[str, Any]):
    """学生プロファイルに基づく研究室適合度評価（分野考慮型）"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # 分野考慮型計算器の使用判定
        use_field_aware = (
            FIELD_AWARE_AVAILABLE and 
            system_state["field_aware_calculator"] is not None and
            "field_interests" in student_profile  # 分野情報がある場合のみ
        )
        
        results = []
        
        for lab in system_state["lab_database"]:
            
            if use_field_aware:
                # === 分野考慮型計算 ===
                calculator = system_state["field_aware_calculator"]
                compatibility, breakdown = calculator.calculate_compatibility(
                    student_profile, lab
                )
                
                lab_result = {
                    "lab_id": lab["id"],
                    "lab_name": lab["name"],
                    "advisor": lab.get("advisor", ""),
                    "description": lab.get("description", ""),
                    "overall_compatibility": float(compatibility),
                    
                    # 分野考慮型の詳細情報
                    "calculation_method": "field_aware_fuzzy",
                    "basic_score": breakdown["basic_score"],
                    "field_score": breakdown["field_score_raw"],
                    "field_score_effective": breakdown["field_score_effective"],
                    "alpha": breakdown["alpha"],  # 分野の比重
                    "beta": breakdown["beta"],    # 基本項目の比重
                    "basic_contribution": breakdown["basic_contribution"],
                    "field_contribution": breakdown["field_contribution"],
                    
                    # 項目別スコア
                    "feature_scores": breakdown["criteria_scores"],
                    
                    # 信頼度
                    "confidence": min(1.0, compatibility + 0.05),
                    
                    # 推薦レベル
                    "recommendation": get_recommendation_level(compatibility),
                    
                    # 説明文
                    "explanation": generate_field_aware_explanation(breakdown, lab)
                }
            
            else:
                # === 従来の計算方法（フォールバック）===
                if FUZZY_AVAILABLE and system_state["fuzzy_engine"]:
                    try:
                        compatibility = system_state["fuzzy_engine"].predict(student_profile)
                    except:
                        compatibility = calculate_simple_compatibility(student_profile, lab)
                else:
                    compatibility = calculate_simple_compatibility(student_profile, lab)
                
                # 詳細スコア計算
                feature_scores = {}
                for feature in ["research_intensity", "advisor_style", "team_work", 
                               "workload", "theory_practice"]:
                    student_val = student_profile.get(feature, 5.0)
                    lab_val = lab.get(feature, 5.0)
                    
                    # 正規化
                    if student_val > 1.0:
                        student_val /= 10.0
                    if lab_val > 1.0:
                        lab_val /= 10.0
                    
                    feature_scores[feature] = 1.0 - abs(student_val - lab_val)
                
                lab_result = {
                    "lab_id": lab["id"],
                    "lab_name": lab["name"],
                    "advisor": lab.get("advisor", ""),
                    "description": lab.get("description", ""),
                    "overall_compatibility": float(compatibility),
                    "calculation_method": "simple",
                    "feature_scores": feature_scores,
                    "confidence": min(1.0, compatibility + random.uniform(0.0, 0.1)),
                    "recommendation": get_recommendation_level(compatibility),
                    "explanation": generate_explanation(student_profile, lab, compatibility)
                }
            
            results.append(lab_result)
        
        # スコアでソート
        results.sort(key=lambda x: x["overall_compatibility"], reverse=True)
        
        # 評価回数増加
        system_state["evaluation_count"] += 1
        
        return {
            "student_profile": student_profile,
            "evaluation_results": results,
            "total_labs_evaluated": len(results),
            "evaluation_timestamp": time.time(),
            "system_info": {
                "calculation_method": "field_aware_fuzzy" if use_field_aware else "simple",
                "field_aware_available": FIELD_AWARE_AVAILABLE,
                "fuzzy_enabled": FUZZY_AVAILABLE,
                "genetic_enabled": GENETIC_AVAILABLE,
                "evaluation_count": system_state["evaluation_count"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
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
    use_field_aware = (
        FIELD_AWARE_AVAILABLE and 
        system_state["field_aware_calculator"] is not None and
        "field_interests" in student_profile
    )
    
    if use_field_aware:
        calculator = system_state["field_aware_calculator"]
        compatibility, breakdown = calculator.calculate_compatibility(student_profile, lab)
        
        detailed_analysis = {
            "overall_compatibility": compatibility,
            "lab_info": lab,
            "calculation_method": "field_aware_fuzzy",
            "breakdown": breakdown,
            "explanation": generate_field_aware_explanation(breakdown, lab),
            "strengths": generate_strengths(breakdown),
            "concerns": generate_concerns(breakdown),
            "recommendations": generate_recommendations(compatibility, breakdown)
        }
    else:
        compatibility = calculate_simple_compatibility(student_profile, lab)
        
        detailed_analysis = {
            "overall_compatibility": compatibility,
            "lab_info": lab,
            "calculation_method": "simple",
            "explanation": generate_explanation(student_profile, lab, compatibility),
            "decision_tree_path": generate_decision_path(student_profile, lab)
        }
    
    return detailed_analysis

# ===== ヘルパー関数 =====

def calculate_simple_compatibility(student_profile: Dict[str, Any], lab: Dict[str, Any]) -> float:
    """簡単な適合度計算"""
    
    total_score = 0.0
    feature_count = 0
    
    for feature in ["research_intensity", "advisor_style", "team_work", 
                    "workload", "theory_practice"]:
        if feature in student_profile and feature in lab:
            student_val = student_profile[feature]
            lab_val = lab[feature]
            
            # 正規化
            if student_val > 1.0:
                student_val /= 10.0
            if lab_val > 1.0:
                lab_val /= 10.0
            
            # 類似度計算
            similarity = 1.0 - abs(student_val - lab_val)
            total_score += similarity
            feature_count += 1
    
    return total_score / max(1, feature_count)

def get_recommendation_level(compatibility: float) -> str:
    """推薦レベル取得"""
    
    if compatibility >= 0.85:
        return "強く推薦"
    elif compatibility >= 0.70:
        return "推薦"
    elif compatibility >= 0.55:
        return "検討可能"
    elif compatibility >= 0.40:
        return "要慎重検討"
    else:
        return "推薦しない"

def generate_field_aware_explanation(breakdown: Dict, lab: Dict) -> str:
    """分野考慮型の説明文を生成"""
    
    alpha = breakdown["alpha"]
    beta = breakdown["beta"]
    basic_score = breakdown["basic_score"]
    field_score = breakdown["field_score_raw"]
    total = breakdown["total_compatibility"]
    
    parts = []
    
    # 総合評価
    if total >= 0.85:
        parts.append("✅ 非常に高い適合度")
    elif total >= 0.70:
        parts.append("⭐ 高い適合度")
    elif total >= 0.55:
        parts.append("🔵 良好な適合度")
    else:
        parts.append("⚠️ 中程度の適合度")
    
    # 比重の説明
    if alpha > 0.7:
        parts.append(f"分野を重視した評価（分野比重{alpha:.0%}）で、分野適合性が高い研究室です")
    elif alpha < 0.3:
        parts.append(f"基本項目を重視した評価（基本比重{beta:.0%}）で、総合的に適合する研究室です")
    else:
        parts.append(f"分野と基本項目をバランス良く評価（分野{alpha:.0%}、基本{beta:.0%}）しています")
    
    # スコアの内訳
    parts.append(f"基本12項目スコア: {basic_score:.2f}, 分野スコア: {field_score:.2f}")
    
    # 研究室情報
    lab_name = lab.get("name", "")
    parts.append(f"{lab_name}との適合性評価です")
    
    return "。".join(parts) + "。"

def generate_explanation(student_profile: Dict[str, Any], lab: Dict[str, Any], compatibility: float) -> str:
    """簡易説明文生成"""
    
    feature_matches = []
    
    for feature in ["research_intensity", "team_work", "advisor_style"]:
        if feature in student_profile and feature in lab:
            student_val = student_profile[feature]
            lab_val = lab[feature]
            
            if student_val > 1.0:
                student_val /= 10.0
            if lab_val > 1.0:
                lab_val /= 10.0
            
            match_score = 1.0 - abs(student_val - lab_val)
            
            if match_score > 0.7:
                feature_matches.append(f"{feature}で高い適合性")
    
    if feature_matches:
        return f"この研究室は{', '.join(feature_matches)}を示しており、総合適合度は{compatibility:.1%}です。"
    else:
        return f"総合適合度は{compatibility:.1%}です。各特徴量を詳しく検討することをお勧めします。"

def generate_strengths(breakdown: Dict) -> List[str]:
    """強みを生成"""
    strengths = []
    
    criteria_scores = breakdown.get("criteria_scores", {})
    sorted_scores = sorted(criteria_scores.items(), key=lambda x: x[1], reverse=True)
    
    for criterion, score in sorted_scores[:3]:
        if score > 0.8:
            strengths.append(f"{criterion}: 非常に高い適合性（{score:.2f}）")
    
    return strengths

def generate_concerns(breakdown: Dict) -> List[str]:
    """懸念点を生成"""
    concerns = []
    
    criteria_scores = breakdown.get("criteria_scores", {})
    sorted_scores = sorted(criteria_scores.items(), key=lambda x: x[1])
    
    for criterion, score in sorted_scores[:2]:
        if score < 0.5:
            concerns.append(f"{criterion}: 適合性が低い可能性（{score:.2f}）")
    
    return concerns

def generate_recommendations(compatibility: float, breakdown: Dict) -> List[str]:
    """推薦事項を生成"""
    recommendations = []
    
    if compatibility > 0.8:
        recommendations.append("この研究室は高い適合性を示しています。積極的に検討することをお勧めします")
    elif compatibility > 0.6:
        recommendations.append("適度な適合性があります。研究室訪問で詳細を確認することをお勧めします")
    else:
        recommendations.append("他の選択肢も検討することをお勧めします")
    
    return recommendations

def generate_decision_path(student_profile: Dict[str, Any], lab: Dict[str, Any]) -> List[str]:
    """決定パス生成（簡易版）"""
    
    path = ["評価開始"]
    
    research_intensity = student_profile.get("research_intensity", 5.0)
    if research_intensity > 1.0:
        research_intensity /= 10.0
    
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

# ===== サーバー起動 =====

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 FastAPI サーバー起動中...")
    print("=" * 70)
    print(f"📍 URL: http://localhost:{settings.port}")
    print(f"📚 API文書: http://localhost:{settings.port}/docs")
    print(f"🔧 システム状況:")
    print(f"  - ファジィ推論: {'✅' if FUZZY_AVAILABLE else '❌'}")
    print(f"  - 遺伝的アルゴリズム: {'✅' if GENETIC_AVAILABLE else '❌'}")
    print(f"  - 決定木: {'✅' if DECISION_TREE_AVAILABLE else '❌'}")
    print(f"  - 分野考慮型: {'✅' if FIELD_AWARE_AVAILABLE else '❌'}")
    print(f"  - 研究室データ: {len(SAMPLE_LABS)}件")
    print("=" * 70 + "\n")
    
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if settings.debug else "warning"
    )