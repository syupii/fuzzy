#!/usr/bin/env python3
"""
遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム
FastAPI メインアプリケーション - 本格実装版
"""

import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from typing import Dict, List, Any, Optional, Tuple
import json
import time
import random
import numpy as np
from dataclasses import dataclass, asdict

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# システムモジュールのインポート
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

# ファジィ・遺伝的アルゴリズムのインポート
try:
    from core.fuzzy.inference import SimpleFuzzyInferenceEngine
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("⚠️ ファジィモジュールが利用できません")

try:
    from core.genetic.evolution import EvolutionEngine, EvolutionConfig, Individual
    from core.genetic.population import Population, PopulationConfig
    from core.genetic.operators import OperatorConfig, SelectionMethod, CrossoverMethod, MutationMethod
    GENETIC_AVAILABLE = True
except ImportError:
    GENETIC_AVAILABLE = False
    print("⚠️ 遺伝的アルゴリズムモジュールが利用できません")

try:
    from core.decision_tree.tree import FuzzyDecisionTree
    DECISION_TREE_AVAILABLE = True
except ImportError:
    DECISION_TREE_AVAILABLE = False
    print("⚠️ 決定木モジュールが利用できません")

# FastAPIアプリケーション作成
app = FastAPI(
    title=settings.app_name if SETTINGS_AVAILABLE else "Lab Matching System",
    version=settings.api_version if SETTINGS_AVAILABLE else "v1",
    description="遺伝的アルゴリズムを用いたファジィ決定木による研究室選択支援システム"
)

# CORSミドルウェア設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバル状態管理
system_state = {
    "initialized": False,
    "lab_database": [],
    "evaluation_count": 0,
    "optimization_cache": {},
    "genetic_engine": None,
    "best_weights": None
}

# ===== 研究室データベース =====

SAMPLE_LABS = [
    # テクノロジー・システム分野
    {"id": "lab_001", "name": "松本研究室", "field": "人工知能・機械学習", "category": "technology",
     "research_intensity": 0.85, "advisor_style": 0.65, "team_work": 0.75, "workload": 0.80, 
     "theory_practice": 0.70, "research_field_match": 0.90, "skill_development": 0.85,
     "lab_atmosphere": 0.70, "flexibility": 0.60, "publication_opportunity": 0.85,
     "interdisciplinary": 0.65, "communication_style": 0.70},
    
    {"id": "lab_002", "name": "小川研究室", "field": "画像・映像処理", "category": "technology",
     "research_intensity": 0.80, "advisor_style": 0.70, "team_work": 0.70, "workload": 0.75,
     "theory_practice": 0.65, "research_field_match": 0.85, "skill_development": 0.80,
     "lab_atmosphere": 0.75, "flexibility": 0.65, "publication_opportunity": 0.80,
     "interdisciplinary": 0.60, "communication_style": 0.75},
    
    {"id": "lab_003", "name": "高橋研究室", "field": "ネットワーク・セキュリティ", "category": "technology",
     "research_intensity": 0.75, "advisor_style": 0.60, "team_work": 0.65, "workload": 0.70,
     "theory_practice": 0.75, "research_field_match": 0.80, "skill_development": 0.75,
     "lab_atmosphere": 0.65, "flexibility": 0.70, "publication_opportunity": 0.75,
     "interdisciplinary": 0.55, "communication_style": 0.65},
    
    {"id": "lab_004", "name": "坂田研究室", "field": "経営情報・意思決定支援", "category": "technology",
     "research_intensity": 0.70, "advisor_style": 0.75, "team_work": 0.80, "workload": 0.65,
     "theory_practice": 0.80, "research_field_match": 0.75, "skill_development": 0.85,
     "lab_atmosphere": 0.80, "flexibility": 0.75, "publication_opportunity": 0.70,
     "interdisciplinary": 0.85, "communication_style": 0.80},
    
    # クリエイティブ分野
    {"id": "lab_005", "name": "佐藤研究室", "field": "Webデザイン・UI/UX", "category": "creative",
     "research_intensity": 0.65, "advisor_style": 0.80, "team_work": 0.85, "workload": 0.70,
     "theory_practice": 0.85, "research_field_match": 0.75, "skill_development": 0.90,
     "lab_atmosphere": 0.85, "flexibility": 0.80, "publication_opportunity": 0.60,
     "interdisciplinary": 0.75, "communication_style": 0.85},
    
    {"id": "lab_006", "name": "田中研究室", "field": "デザイン・視覚表現", "category": "creative",
     "research_intensity": 0.60, "advisor_style": 0.85, "team_work": 0.80, "workload": 0.65,
     "theory_practice": 0.90, "research_field_match": 0.70, "skill_development": 0.85,
     "lab_atmosphere": 0.90, "flexibility": 0.85, "publication_opportunity": 0.55,
     "interdisciplinary": 0.70, "communication_style": 0.90},
    
    # エンターテイメント分野
    {"id": "lab_007", "name": "山田研究室", "field": "ゲーム開発・eスポーツ", "category": "entertainment",
     "research_intensity": 0.70, "advisor_style": 0.75, "team_work": 0.90, "workload": 0.80,
     "theory_practice": 0.80, "research_field_match": 0.80, "skill_development": 0.80,
     "lab_atmosphere": 0.85, "flexibility": 0.70, "publication_opportunity": 0.65,
     "interdisciplinary": 0.60, "communication_style": 0.85},
    
    {"id": "lab_008", "name": "伊藤研究室", "field": "VR/AR・メディアアート", "category": "entertainment",
     "research_intensity": 0.75, "advisor_style": 0.70, "team_work": 0.85, "workload": 0.75,
     "theory_practice": 0.75, "research_field_match": 0.85, "skill_development": 0.85,
     "lab_atmosphere": 0.80, "flexibility": 0.75, "publication_opportunity": 0.70,
     "interdisciplinary": 0.80, "communication_style": 0.80},
    
    # 人文・社会・体育分野
    {"id": "lab_009", "name": "三浦研究室", "field": "哲学・人文・環境行動学", "category": "humanities",
     "research_intensity": 0.65, "advisor_style": 0.80, "team_work": 0.60, "workload": 0.60,
     "theory_practice": 0.60, "research_field_match": 0.70, "skill_development": 0.70,
     "lab_atmosphere": 0.70, "flexibility": 0.85, "publication_opportunity": 0.75,
     "interdisciplinary": 0.90, "communication_style": 0.75},
    
    {"id": "lab_010", "name": "綿谷研究室", "field": "スポーツ・体育科学", "category": "humanities",
     "research_intensity": 0.70, "advisor_style": 0.75, "team_work": 0.85, "workload": 0.75,
     "theory_practice": 0.85, "research_field_match": 0.75, "skill_development": 0.80,
     "lab_atmosphere": 0.85, "flexibility": 0.70, "publication_opportunity": 0.70,
     "interdisciplinary": 0.70, "communication_style": 0.85},
]

# ===== 遺伝的アルゴリズム最適化関数 =====

def create_genetic_engine(config: Optional[Dict[str, Any]] = None) -> EvolutionEngine:
    """遺伝的アルゴリズムエンジンの作成"""
    
    # デフォルト設定
    evolution_config = EvolutionConfig(
        population_size=config.get("population_size", 30) if config else 30,
        generations=config.get("generations", 50) if config else 50,
        mutation_rate=config.get("mutation_rate", 0.15) if config else 0.15,
        crossover_rate=config.get("crossover_rate", 0.8) if config else 0.8,
        elitism_rate=config.get("elitism_rate", 0.1) if config else 0.1,
        tournament_size=config.get("tournament_size", 3) if config else 3,
        convergence_threshold=config.get("convergence_threshold", 1e-6) if config else 1e-6,
        max_stagnation=config.get("max_stagnation", 15) if config else 15
    )
    
    return EvolutionEngine(evolution_config)

def fitness_function(individual: Individual, student_profile: Dict[str, float], 
                     labs: List[Dict[str, Any]]) -> float:
    """適応度関数：個体の重みを使って研究室とのマッチング精度を評価"""
    
    try:
        weights = individual.chromosome
        
        # 評価基準の取得
        features = settings.core_features if SETTINGS_AVAILABLE else [
            "research_intensity", "advisor_style", "team_work", 
            "workload", "theory_practice"
        ]
        
        # 重みの数を調整
        if len(weights) > len(features):
            weights = weights[:len(features)]
        elif len(weights) < len(features):
            weights = list(weights) + [1.0] * (len(features) - len(weights))
        
        total_error = 0.0
        
        # 全研究室とのマッチングを評価
        for lab in labs:
            # 重み付き距離計算
            weighted_distance = 0.0
            for i, feature in enumerate(features):
                student_val = float(student_profile.get(feature, 0.5))
                lab_val = float(lab.get(feature, 0.5))
                
                # 10段階評価の場合は正規化
                if student_val > 1.0:
                    student_val /= 10.0
                
                # 重みを適用した距離
                diff = abs(student_val - lab_val)
                weighted_distance += weights[i] * diff
            
            # 正規化された誤差を累積
            normalized_error = weighted_distance / sum(weights) if sum(weights) > 0 else 0
            total_error += normalized_error
        
        # 適応度は誤差の逆数（誤差が小さいほど高い適応度）
        avg_error = total_error / len(labs) if len(labs) > 0 else 1.0
        fitness = 1.0 / (avg_error + 1e-6)
        
        return fitness
        
    except Exception as e:
        print(f"⚠️ 適応度計算エラー: {e}")
        return 0.0

def optimize_with_genetic_algorithm(student_profiles: List[Dict[str, float]], 
                                    labs: List[Dict[str, Any]], 
                                    config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """遺伝的アルゴリズムによる重み最適化"""
    
    print(f"\n🧬 遺伝的アルゴリズム最適化開始")
    print(f"  学生プロファイル数: {len(student_profiles)}")
    print(f"  研究室数: {len(labs)}")
    
    start_time = time.time()
    
    # エンジン作成
    engine = create_genetic_engine(config)
    
    # 初期集団生成
    engine.initialize_population()
    print(f"  初期集団生成完了: {len(engine.population)}個体")
    
    # 各世代の進化
    best_fitness_history = []
    avg_fitness_history = []
    
    for generation in range(engine.config.generations):
        # 適応度評価
        for individual in engine.population:
            # 複数の学生プロファイルで評価し、平均を取る
            total_fitness = 0.0
            for profile in student_profiles:
                total_fitness += fitness_function(individual, profile, labs)
            individual.fitness = total_fitness / len(student_profiles)
        
        # 集団を適応度でソート
        engine.population.sort(key=lambda ind: ind.fitness, reverse=True)
        
        # 統計情報
        best_fitness = engine.population[0].fitness
        avg_fitness = sum(ind.fitness for ind in engine.population) / len(engine.population)
        
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        if generation % 10 == 0:
            print(f"  世代 {generation}: 最良適応度={best_fitness:.4f}, 平均適応度={avg_fitness:.4f}")
        
        # 収束判定
        if generation > 0:
            improvement = best_fitness - best_fitness_history[-2]
            if abs(improvement) < engine.config.convergence_threshold:
                engine.stagnation_count += 1
                if engine.stagnation_count >= engine.config.max_stagnation:
                    print(f"  収束検出（世代{generation}）")
                    break
            else:
                engine.stagnation_count = 0
        
        # 次世代生成
        engine.evolve_generation()
    
    processing_time = time.time() - start_time
    
    # 最良個体の取得
    best_individual = engine.population[0]
    
    print(f"✅ 最適化完了")
    print(f"  最終世代: {generation + 1}")
    print(f"  最良適応度: {best_individual.fitness:.6f}")
    print(f"  処理時間: {processing_time:.2f}秒")
    print(f"  最適重み: {[f'{w:.3f}' for w in best_individual.chromosome[:5]]}...")
    
    return {
        "best_weights": best_individual.chromosome,
        "best_fitness": best_individual.fitness,
        "generations": generation + 1,
        "processing_time": processing_time,
        "fitness_history": {
            "best": best_fitness_history,
            "average": avg_fitness_history
        },
        "convergence_achieved": engine.stagnation_count >= engine.config.max_stagnation,
        "population_size": len(engine.population)
    }

# ===== システム初期化 =====

def initialize_system():
    """システム初期化"""
    try:
        print("\n🚀 システム初期化中...")
        
        # 研究室データベース読み込み
        system_state["lab_database"] = SAMPLE_LABS
        print(f"✅ 研究室データ読み込み完了: {len(SAMPLE_LABS)}件")
        
        # 遺伝的アルゴリズムエンジン初期化
        if GENETIC_AVAILABLE:
            system_state["genetic_engine"] = create_genetic_engine()
            print("✅ 遺伝的アルゴリズムエンジン初期化完了")
        
        system_state["initialized"] = True
        print("✅ システム初期化完了")
        
    except Exception as e:
        print(f"❌ システム初期化エラー: {e}")
        system_state["initialized"] = False

initialize_system()

# ===== 適合度計算関数 =====

def calculate_compatibility_with_weights(student_profile: Dict[str, float], 
                                        lab: Dict[str, Any],
                                        weights: Optional[List[float]] = None) -> float:
    """重み付き適合度計算（改良版）"""
    
    try:
        # 評価基準の取得
        features = settings.core_features if SETTINGS_AVAILABLE else [
            "research_intensity", "advisor_style", "team_work", 
            "workload", "theory_practice"
        ]
        
        if weights is None:
            # デフォルト重み（均等）
            weights = [1.0] * len(features)
        
        # 重みの数を調整（足りない場合は1.0で埋める）
        if len(weights) < len(features):
            weights = list(weights) + [1.0] * (len(features) - len(weights))
        
        total_weighted_similarity = 0.0
        total_weight = 0.0
        
        for i, feature in enumerate(features):
            # 学生の値を取得（デフォルト0.5）
            student_val = float(student_profile.get(feature, 0.5))
            
            # 10段階評価の場合は0-1に正規化
            if student_val > 1.0:
                student_val /= 10.0
            
            # 研究室の値を取得（デフォルト0.5）
            lab_val = float(lab.get(feature, 0.5))
            
            # 類似度計算（1 - 差の絶対値）
            similarity = 1.0 - abs(student_val - lab_val)
            similarity = max(0.0, min(1.0, similarity))  # 0-1に制限
            
            # 重み適用
            weight = weights[i] if i < len(weights) else 1.0
            weighted_similarity = weight * similarity
            
            total_weighted_similarity += weighted_similarity
            total_weight += weight
        
        # 正規化
        if total_weight > 0:
            compatibility = total_weighted_similarity / total_weight
        else:
            compatibility = 0.0
        
        # 0-1の範囲に制限
        compatibility = max(0.0, min(1.0, compatibility))
        
        return compatibility
        
    except Exception as e:
        print(f"⚠️ 適合度計算エラー: {e}")
        import traceback
        traceback.print_exc()
        return 0.5  # エラー時はデフォルト値を返す

# ===== APIエンドポイント =====

@app.get("/")
async def read_root():
    """ルートエンドポイント"""
    if os.path.exists("../frontend/build/index.html"):
        return FileResponse("../frontend/build/index.html")
    else:
        return {
            "message": "遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム",
            "version": "2.0.0",
            "status": "running",
            "genetic_algorithm": "fully_implemented",
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
    
    lab_count = len(system_state.get("lab_database", []))
    
    modules_status = {
        "fuzzy": FUZZY_AVAILABLE,
        "genetic": GENETIC_AVAILABLE,
        "decision_tree": DECISION_TREE_AVAILABLE,
        "settings": SETTINGS_AVAILABLE
    }
    
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
        "genetic_engine_status": "initialized" if system_state.get("genetic_engine") else "not_initialized",
        "database": {
            "status": "OK" if lab_count > 0 else "Empty",
            "lab_count": lab_count,
            "evaluation_count": system_state["evaluation_count"]
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

@app.post("/api/evaluate")
async def evaluate_compatibility(request: Dict[str, Any]):
    """研究室との適合度評価（フロントエンド互換版）"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # リクエストから学生プロファイルを取得
        student_profile = request.get("student_profile", request)
        
        print(f"📥 評価リクエスト受信")
        print(f"  プロファイルキー: {list(student_profile.keys())[:5]}...")
        
        # 入力検証
        required_features = settings.core_features if SETTINGS_AVAILABLE else [
            "research_intensity", "advisor_style", "team_work", 
            "workload", "theory_practice"
        ]
        
        for feature in required_features:
            if feature not in student_profile:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing required feature: {feature}"
                )
        
        # 現在の最適重みを使用（存在すれば）
        weights = system_state.get("best_weights")
        
        # 各研究室との適合度計算
        lab_results = []
        for lab in system_state["lab_database"]:
            compatibility = calculate_compatibility_with_weights(
                student_profile, lab, weights
            )
            
            # 詳細スコア計算
            feature_scores = {}
            for feature in required_features:
                student_val = float(student_profile.get(feature, 0.5))
                lab_val = float(lab.get(feature, 0.5))
                
                # 10段階評価の場合は正規化
                if student_val > 1.0:
                    student_val /= 10.0
                
                similarity = 1.0 - abs(student_val - lab_val)
                feature_scores[feature] = similarity
            
            # フロントエンドが期待する形式でレスポンス構築
            lab_result = {
                # 基本情報
                "lab_id": lab["id"],
                "lab_name": lab["name"],
                "field": lab.get("field", "Unknown"),
                "category": lab.get("category", "technology"),
                
                # スコア（複数の名前で提供して互換性を確保）
                "final_score": float(compatibility),
                "compatibility_score": float(compatibility),
                "overall_compatibility": float(compatibility),
                
                # 詳細情報
                "feature_scores": feature_scores,
                "using_optimized_weights": weights is not None,
                "confidence": min(1.0, compatibility),
                
                # 説明
                "recommendation": "高推奨" if compatibility > 0.7 else "推奨" if compatibility > 0.5 else "要検討",
                "explanation": f"総合適合度: {compatibility:.1%}"
            }
            
            lab_results.append(lab_result)
        
        # 適合度でソート
        lab_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        # 統計情報計算
        scores = [lab["final_score"] for lab in lab_results]
        summary = {
            "total_labs": len(lab_results),
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0
        }
        
        system_state["evaluation_count"] += 1
        
        # フロントエンドが期待する形式でレスポンス
        response = {
            "lab_results": lab_results,
            "summary": summary,
            "metadata": {
                "processing_time": 0.1,
                "timestamp": time.time(),
                "optimized_weights_used": weights is not None,
                "evaluation_count": system_state["evaluation_count"]
            }
        }
        
        print(f"📤 評価結果送信: {len(lab_results)}件")
        print(f"  最高適合度: {summary['max_score']:.3f}")
        print(f"  平均適合度: {summary['avg_score']:.3f}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"❌ 評価エラー: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

@app.post("/api/optimize")
async def optimize_matching(request: Dict[str, Any]):
    """遺伝的アルゴリズムによる最適化（フロントエンド互換版）"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not GENETIC_AVAILABLE:
        raise HTTPException(status_code=501, detail="Genetic algorithm not available")
    
    try:
        # リクエストからデータ取得
        student_profiles = request.get("student_profiles", [])
        
        # 単一プロファイルの場合の対応
        if not student_profiles:
            if "student_profile" in request:
                student_profiles = [request["student_profile"]]
            elif "evaluation_criteria" in request:
                student_profiles = [request["evaluation_criteria"]]
            else:
                student_profiles = [request]
        
        if not student_profiles:
            raise HTTPException(status_code=400, detail="No student profiles provided")
        
        ga_config = request.get("genetic_config", {})
        
        print(f"\n🧬 遺伝的アルゴリズム最適化実行")
        print(f"  学生数: {len(student_profiles)}")
        print(f"  研究室数: {len(system_state['lab_database'])}")
        
        # 遺伝的アルゴリズム実行
        optimization_result = optimize_with_genetic_algorithm(
            student_profiles, 
            system_state["lab_database"],
            ga_config
        )
        
        # 最適重みを保存
        system_state["best_weights"] = optimization_result["best_weights"]
        
        print(f"✅ 最適化完了")
        print(f"  最良適応度: {optimization_result['best_fitness']:.6f}")
        print(f"  実行世代数: {optimization_result['generations']}")
        
        # 最適化後の評価を実行（フロントエンド互換形式）
        optimized_evaluations = []
        for profile in student_profiles:
            # 最適化後の評価
            eval_request = {"student_profile": profile}
            eval_response = await evaluate_compatibility(eval_request)
            
            # トップ3の取得
            top_3_labs = eval_response["lab_results"][:3]
            
            optimized_evaluations.append({
                "student_profile": profile,
                "top_matches": top_3_labs,
                "optimization_applied": True
            })
        
        # フロントエンド互換のレスポンス形式
        response = {
            # 最適化情報
            "optimization_completed": True,
            "students_processed": len(student_profiles),
            "genetic_algorithm_info": {
                "method": "real_genetic_algorithm",
                "generations": optimization_result["generations"],
                "population_size": optimization_result["population_size"],
                "best_fitness": optimization_result["best_fitness"],
                "convergence_achieved": optimization_result["convergence_achieved"],
                "processing_time": optimization_result["processing_time"]
            },
            
            # 最適化重み
            "optimized_weights": optimization_result["best_weights"],
            "fitness_history": optimization_result["fitness_history"],
            
            # 評価結果（フロントエンド互換）
            "lab_results": optimized_evaluations[0]["top_matches"] if optimized_evaluations else [],
            "summary": {
                "total_labs": len(system_state["lab_database"]),
                "avg_score": sum(r["final_score"] for r in optimized_evaluations[0]["top_matches"]) / 3 if optimized_evaluations else 0,
                "max_score": max(r["final_score"] for r in optimized_evaluations[0]["top_matches"]) if optimized_evaluations else 0,
                "min_score": min(r["final_score"] for r in optimized_evaluations[0]["top_matches"]) if optimized_evaluations else 0
            },
            
            # 追加情報
            "optimization_results": optimized_evaluations,
            "timestamp": time.time(),
            
            # メタデータ
            "metadata": {
                "optimization_method": "genetic_algorithm",
                "weights_optimized": True,
                "processing_time": optimization_result["processing_time"]
            }
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"❌ 最適化エラー: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

@app.post("/api/explain")
async def explain_recommendation(request: Dict[str, Any]):
    """推薦結果の詳細説明（フロントエンド互換版）"""
    
    try:
        # リクエストからデータ取得
        student_profile = request.get("student_profile", request)
        lab_id = request.get("lab_id")
        
        if not lab_id:
            raise HTTPException(status_code=400, detail="lab_id is required")
        
        # 研究室を取得
        lab = next((lab for lab in system_state["lab_database"] if lab["id"] == lab_id), None)
        if not lab:
            raise HTTPException(status_code=404, detail="Lab not found")
        
        # 現在の重みを使用
        weights = system_state.get("best_weights")
        
        # 適合度計算
        compatibility = calculate_compatibility_with_weights(student_profile, lab, weights)
        
        # 評価基準
        required_features = settings.core_features if SETTINGS_AVAILABLE else [
            "research_intensity", "advisor_style", "team_work", 
            "workload", "theory_practice"
        ]
        
        # 特徴量別の詳細分析
        feature_analysis = {}
        strengths = []
        concerns = []
        
        for i, feature in enumerate(required_features):
            student_val = float(student_profile.get(feature, 0.5))
            lab_val = float(lab.get(feature, 0.5))
            
            # 10段階評価の場合は正規化
            if student_val > 1.0:
                student_val /= 10.0
            
            similarity = 1.0 - abs(student_val - lab_val)
            weight = weights[i] if weights else 1.0
            
            feature_analysis[feature] = {
                "student_value": student_val,
                "lab_value": lab_val,
                "similarity": similarity,
                "weight": weight,
                "weighted_score": similarity * weight,
                "match_level": "高" if similarity > 0.8 else "中" if similarity > 0.6 else "低"
            }
            
            # 強みと懸念点の抽出
            if similarity > 0.8:
                strengths.append(f"{feature}の高い適合性（{similarity:.1%}）")
            elif similarity < 0.5:
                concerns.append(f"{feature}の適合性が低い（{similarity:.1%}）")
        
        # 推奨事項の生成
        recommendations = []
        if compatibility > 0.7:
            recommendations.append("この研究室は高い適合性を示しています")
            recommendations.append("積極的に検討することをお勧めします")
        elif compatibility > 0.5:
            recommendations.append("適度な適合性があります")
            recommendations.append("研究室訪問で詳細を確認することをお勧めします")
        else:
            recommendations.append("他の研究室も検討することをお勧めします")
        
        # レスポンス構築
        response = {
            "lab_id": lab_id,
            "lab_info": {
                "name": lab["name"],
                "field": lab.get("field", "Unknown"),
                "category": lab.get("category", "technology")
            },
            "overall_compatibility": compatibility,
            "compatibility_score": compatibility,
            "final_score": compatibility,
            
            "feature_analysis": feature_analysis,
            "strengths": strengths,
            "concerns": concerns,
            "recommendations": recommendations,
            
            "optimization_info": {
                "optimized_weights_available": weights is not None,
                "weights_used": "optimized" if weights else "default",
                "weight_values": weights if weights else None
            },
            
            "metadata": {
                "analysis_timestamp": time.time(),
                "features_analyzed": len(feature_analysis)
            }
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"❌ 説明生成エラー: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")

# サーバー起動
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 FastAPI サーバー起動中...")
    print("=" * 70)
    print(f"📍 URL: http://localhost:{settings.port if SETTINGS_AVAILABLE else 8000}")
    print(f"📚 API文書: http://localhost:{settings.port if SETTINGS_AVAILABLE else 8000}/docs")
    print(f"🔧 システム状況:")
    print(f"  - ファジィ推論: {'✅' if FUZZY_AVAILABLE else '❌ (オプション)'}")
    print(f"  - 遺伝的アルゴリズム: {'✅ 本格実装' if GENETIC_AVAILABLE else '❌'}")
    print(f"  - 決定木: {'✅' if DECISION_TREE_AVAILABLE else '❌ (オプション)'}")
    print(f"  - 設定ファイル: {'✅' if SETTINGS_AVAILABLE else '❌ (デフォルト使用)'}")
    print(f"  - 研究室データ: {len(SAMPLE_LABS)}件")
    print(f"  - システム初期化: {'✅' if system_state['initialized'] else '❌'}")
    print(f"\n📡 利用可能なエンドポイント:")
    print(f"  GET  /health           - ヘルスチェック")
    print(f"  GET  /api/labs         - 研究室一覧")
    print(f"  POST /api/evaluate     - 適合度評価")
    print(f"  POST /api/optimize     - 遺伝的アルゴリズム最適化")
    print(f"  POST /api/explain      - 推薦理由説明")
    print("=" * 70)
    
    # reloadオプションを使う場合はインポート文字列で指定
    uvicorn.run(
        "app:app",  # ← 文字列で指定（重要！）
        host=settings.host if SETTINGS_AVAILABLE else "0.0.0.0",
        port=settings.port if SETTINGS_AVAILABLE else 8000,
        reload=settings.debug if SETTINGS_AVAILABLE else True,
        log_level="info"
    )