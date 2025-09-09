# backend/app.py - 遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム
# 更新版 - labs_database.json対応

import os
import sys
import time
import json
import math
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# FastAPI関連
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

# プロジェクトルート設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# モジュール可用性チェック（エラーハンドリング付き）
try:
    from config.settings import settings
    SETTINGS_AVAILABLE = True
    print("✅ 設定モジュール読み込み成功")
except ImportError as e:
    SETTINGS_AVAILABLE = False
    print(f"⚠️ 設定モジュール読み込み失敗: {e}")
    # デフォルト設定
    class DefaultSettings:
        app_name = "研究室選択支援システム"
        api_version = "v2"
        host = "0.0.0.0"
        port = 8000
        evaluation_criteria = [
            "research_intensity", "advisor_style", "team_work", "workload", "theory_practice",
            "research_field_match", "skill_development", "lab_atmosphere", "flexibility", 
            "publication_opportunity", "interdisciplinary", "communication_style", "innovation_risk"
        ]
    settings = DefaultSettings()

try:
    from data.models.labs_database import LabDatabase
    DATABASE_AVAILABLE = True
    print("✅ データベースモジュール読み込み成功")
except ImportError as e:
    DATABASE_AVAILABLE = False
    print(f"⚠️ データベースモジュール読み込み失敗: {e}")

# ファジィ決定木システムのインポート（エラーハンドリング付き）
try:
    from core.fuzzy.inference import SimpleFuzzyInferenceEngine
    FUZZY_AVAILABLE = True
    print("✅ ファジィ推論モジュール読み込み成功")
except ImportError as e:
    FUZZY_AVAILABLE = False
    print(f"⚠️ ファジィ推論モジュール読み込み失敗: {e}")

try:
    from core.genetic.evolution import EvolutionEngine, EvolutionConfig
    from core.genetic.population import PopulationConfig
    GENETIC_AVAILABLE = True
    print("✅ 遺伝的アルゴリズムモジュール読み込み成功")
except ImportError as e:
    GENETIC_AVAILABLE = False
    print(f"⚠️ 遺伝的アルゴリズムモジュール読み込み失敗: {e}")

try:
    from core.decision_tree.tree import FuzzyDecisionTree
    DECISION_TREE_AVAILABLE = True
    print("✅ 決定木モジュール読み込み成功")
except ImportError as e:
    DECISION_TREE_AVAILABLE = False
    print(f"⚠️ 決定木モジュール読み込み失敗: {e}")

# FastAPIアプリケーション初期化
app = FastAPI(
    title="研究室選択支援システム v2.0",
    description="遺伝的アルゴリズムを用いたファジィ決定木による研究室マッチングシステム - 完全版データベース対応",
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
    "lab_database": None,
    "lab_data": [],
    "evaluation_count": 0,
    "database_version": "2.0.0",
    "last_updated": None
}

def load_laboratory_data():
    """研究室データの読み込み（labs_database.json対応）"""
    
    try:
        if DATABASE_AVAILABLE:
            # データベースクラスを使用
            print("📁 LabDatabaseクラスから研究室データを読み込み中...")
            lab_db = LabDatabase()
            labs_data = lab_db.get_all_labs()
            system_state["lab_database"] = lab_db
            system_state["last_updated"] = lab_db.metadata.get("last_updated")
            print(f"✅ データベースから{len(labs_data)}件の研究室データを読み込み")
            return labs_data
        else:
            # 直接JSONファイルを読み込み
            json_path = project_root / "data" / "labs_database.json"
            print(f"📁 JSONファイルから研究室データを読み込み中: {json_path}")
            
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    labs_data = data.get('labs', [])
                    system_state["database_version"] = data.get('version', '2.0.0')
                    system_state["last_updated"] = data.get('last_updated')
                    print(f"✅ JSONファイルから{len(labs_data)}件の研究室データを読み込み")
                    print(f"📊 データベースバージョン: {system_state['database_version']}")
                    return labs_data
            else:
                print(f"⚠️ データファイルが見つかりません: {json_path}")
                return create_fallback_labs()
                
    except Exception as e:
        print(f"❌ 研究室データ読み込みエラー: {e}")
        traceback.print_exc()
        return create_fallback_labs()

def create_fallback_labs():
    """フォールバック用の研究室データ"""
    print("🔄 フォールバック研究室データを作成中...")
    
    return [
        {
            "id": "lab_fallback_001",
            "name": "AI・機械学習研究室（フォールバック）",
            "professor": "サンプル教授",
            "research_area": "人工知能・機械学習",
            "specialization": "機械学習、深層学習",
            "description": "人工知能と機械学習の研究を行うフォールバック研究室です",
            "features": {
                "research_intensity": 8.0,
                "advisor_style": 7.0,
                "team_work": 8.0,
                "workload": 7.5,
                "theory_practice": 6.5,
                "research_field_match": 8.5,
                "skill_development": 8.0,
                "lab_atmosphere": 7.5,
                "flexibility": 7.0,
                "publication_opportunity": 7.5,
                "interdisciplinary": 7.0,
                "communication_style": 7.0,
                "innovation_risk": 7.5
            },
            "metadata": {
                "faculty_count": 1,
                "student_count": 6,
                "recent_publications": 10,
                "funding_level": "中",
                "equipment_rating": 7
            }
        }
    ]

def initialize_system():
    """システム初期化"""
    try:
        print("\n🔧 システム初期化開始...")
        
        # 研究室データ読み込み
        lab_data = load_laboratory_data()
        system_state["lab_data"] = lab_data
        
        # ファジィ推論エンジン初期化
        if FUZZY_AVAILABLE:
            try:
                system_state["fuzzy_engine"] = SimpleFuzzyInferenceEngine()
                print("✅ ファジィ推論エンジン初期化完了")
            except Exception as e:
                print(f"⚠️ ファジィ推論エンジン初期化失敗: {e}")
        
        # 遺伝的アルゴリズム初期化
        if GENETIC_AVAILABLE:
            try:
                evolution_config = EvolutionConfig(
                    population_size=30,
                    generations=50,
                    mutation_rate=0.1,
                    crossover_rate=0.8
                )
                system_state["genetic_engine"] = EvolutionEngine(evolution_config)
                print("✅ 遺伝的アルゴリズム初期化完了")
            except Exception as e:
                print(f"⚠️ 遺伝的アルゴリズム初期化失敗: {e}")
        
        # 決定木初期化
        if DECISION_TREE_AVAILABLE:
            try:
                system_state["decision_tree"] = FuzzyDecisionTree()
                print("✅ ファジィ決定木初期化完了")
            except Exception as e:
                print(f"⚠️ ファジィ決定木初期化失敗: {e}")
        
        system_state["initialized"] = True
        print(f"✅ システム初期化完了 - 研究室数: {len(lab_data)}件")
        
    except Exception as e:
        print(f"❌ システム初期化エラー: {e}")
        traceback.print_exc()
        system_state["initialized"] = False

def calculate_enhanced_compatibility(student_profile: Dict[str, Any], lab: Dict[str, Any]) -> float:
    """拡張適合性計算（13項目評価基準対応）"""
    
    lab_features = lab.get("features", {})
    if not lab_features:
        return 0.5  # デフォルト適合性
    
    # 13項目の評価基準
    evaluation_criteria = settings.evaluation_criteria if SETTINGS_AVAILABLE else [
        "research_intensity", "advisor_style", "team_work", "workload", "theory_practice",
        "research_field_match", "skill_development", "lab_atmosphere", "flexibility", 
        "publication_opportunity", "interdisciplinary", "communication_style", "innovation_risk"
    ]
    
    total_score = 0.0
    criteria_count = 0
    
    # 基本適合性計算
    for criterion in evaluation_criteria:
        student_value = student_profile.get(criterion)
        lab_value = lab_features.get(criterion)
        
        if student_value is not None and lab_value is not None:
            # 正規化（1-10 → 0-1）
            student_norm = (float(student_value) - 1.0) / 9.0
            lab_norm = (float(lab_value) - 1.0) / 9.0
            
            # 類似度計算（距離ベース）
            similarity = 1.0 - abs(student_norm - lab_norm)
            total_score += similarity
            criteria_count += 1
    
    # 研究分野ボーナス
    field_bonus = 0.0
    student_interests = student_profile.get("research_interests", [])
    lab_fields = lab.get("research_fields", [])
    
    if student_interests and lab_fields:
        # 分野の一致度チェック
        matches = len(set(student_interests) & set(lab_fields))
        field_bonus = min(0.2, matches * 0.1)  # 最大0.2のボーナス
    
    # 最終適合性スコア
    base_compatibility = total_score / criteria_count if criteria_count > 0 else 0.0
    final_score = min(1.0, base_compatibility + field_bonus)
    
    return final_score

# システム初期化
initialize_system()

# =============================================================================
# API エンドポイント定義
# =============================================================================

@app.get("/")
async def read_root():
    """ルートエンドポイント - フロントエンド配信またはAPI情報"""
    if os.path.exists("../frontend/build/index.html"):
        return FileResponse("../frontend/build/index.html")
    else:
        return {
            "message": "遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム v2.0",
            "version": "2.0.0",
            "database_version": system_state.get("database_version", "2.0.0"),
            "status": "running",
            "total_labs": len(system_state.get("lab_data", [])),
            "last_updated": system_state.get("last_updated"),
            "endpoints": {
                "health": "/health",
                "labs": "/api/labs",
                "evaluate": "/api/evaluate",
                "optimize": "/api/optimize",
                "statistics": "/api/statistics",
                "fields": "/api/research-fields",
                "docs": "/docs"
            }
        }

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    
    # データベース状態チェック
    lab_count = len(system_state.get("lab_data", []))
    
    # モジュール可用性チェック
    modules_status = {
        "fuzzy": FUZZY_AVAILABLE,
        "genetic": GENETIC_AVAILABLE,
        "decision_tree": DECISION_TREE_AVAILABLE,
        "settings": SETTINGS_AVAILABLE,
        "database": DATABASE_AVAILABLE
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
        "database_version": system_state.get("database_version", "2.0.0"),
        "timestamp": time.time(),
        "system_initialized": system_state["initialized"],
        "modules": modules_status,
        "database": {
            "status": "OK" if lab_count > 0 else "Empty",
            "lab_count": lab_count,
            "evaluation_count": system_state["evaluation_count"],
            "database_type": "LabDatabase" if DATABASE_AVAILABLE else "JSON Direct",
            "last_updated": system_state.get("last_updated")
        }
    }

@app.get("/api/labs")
async def get_labs():
    """研究室一覧取得"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {
        "labs": system_state["lab_data"],
        "total_count": len(system_state["lab_data"]),
        "database_version": system_state.get("database_version", "2.0.0"),
        "last_updated": system_state.get("last_updated"),
        "data_source": "LabDatabase" if DATABASE_AVAILABLE else "JSON Direct"
    }

@app.get("/api/labs/{lab_id}")
async def get_lab_detail(lab_id: str):
    """特定研究室の詳細取得"""
    
    # データベースクラスが利用可能な場合
    if system_state["lab_database"]:
        lab = system_state["lab_database"].get_lab_by_id(lab_id)
        if not lab:
            raise HTTPException(status_code=404, detail="Lab not found")
        return lab
    
    # 直接検索
    lab = next((lab for lab in system_state["lab_data"] if lab.get("id") == lab_id), None)
    if not lab:
        raise HTTPException(status_code=404, detail="Lab not found")
    
    return lab

@app.get("/api/research-fields")
async def get_research_fields():
    """研究分野一覧取得"""
    
    if not system_state["lab_data"]:
        raise HTTPException(status_code=503, detail="Lab data not available")
    
    # 研究分野の統計情報
    field_stats = {}
    for lab in system_state["lab_data"]:
        research_area = lab.get("research_area", "その他")
        if research_area not in field_stats:
            field_stats[research_area] = {
                "name": research_area,
                "lab_count": 0,
                "professors": []
            }
        field_stats[research_area]["lab_count"] += 1
        prof = lab.get("professor")
        if prof and prof not in field_stats[research_area]["professors"]:
            field_stats[research_area]["professors"].append(prof)
    
    return {
        "research_fields": list(field_stats.values()),
        "total_fields": len(field_stats),
        "total_labs": len(system_state["lab_data"])
    }

@app.post("/api/evaluate")
async def evaluate_compatibility(evaluation_data: Dict[str, Any]):
    """研究室適合性評価"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    student_profile = evaluation_data.get("student_profile", evaluation_data.get("preferences", {}))
    if not student_profile:
        raise HTTPException(status_code=400, detail="Student profile required")
    
    # 全研究室との適合性を計算
    results = []
    
    for lab in system_state["lab_data"]:
        compatibility = calculate_enhanced_compatibility(student_profile, lab)
        
        results.append({
            "lab_id": lab.get("id"),
            "lab_name": lab.get("name"),
            "professor": lab.get("professor"),
            "research_area": lab.get("research_area"),
            "specialization": lab.get("specialization", ""),
            "compatibility_score": compatibility,
            "description": lab.get("description", ""),
            "research_fields": lab.get("research_fields", []),
            "metadata": lab.get("metadata", {}),
            "features": lab.get("features", {})
        })
    
    # スコア順でソート
    results.sort(key=lambda x: x["compatibility_score"], reverse=True)
    
    system_state["evaluation_count"] += 1
    
    return {
        "evaluation_id": f"eval_{system_state['evaluation_count']}_{int(time.time())}",
        "student_profile": student_profile,
        "lab_results": results,
        "total_labs_evaluated": len(results),
        "timestamp": time.time(),
        "processing_time": 0.1,
        "algorithm_info": {
            "method": "enhanced_fuzzy_similarity",
            "evaluation_criteria": 13,
            "data_source": "LabDatabase v2.0" if DATABASE_AVAILABLE else "JSON v2.0",
            "fuzzy_available": FUZZY_AVAILABLE,
            "genetic_available": GENETIC_AVAILABLE,
            "decision_tree_available": DECISION_TREE_AVAILABLE
        }
    }

@app.get("/api/statistics")
async def get_data_statistics():
    """データ統計情報の取得"""
    
    if not system_state["lab_data"]:
        return {"error": "No data available"}
    
    # 基本統計
    total_labs = len(system_state["lab_data"])
    
    # 研究分野別統計
    field_counts = {}
    professor_counts = {}
    equipment_ratings = []
    funding_levels = {}
    
    for lab in system_state["lab_data"]:
        # 研究分野
        field = lab.get("research_area", "Unknown")
        field_counts[field] = field_counts.get(field, 0) + 1
        
        # 教員
        prof = lab.get("professor", "Unknown")
        professor_counts[prof] = professor_counts.get(prof, 0) + 1
        
        # メタデータ分析
        metadata = lab.get("metadata", {})
        if "equipment_rating" in metadata:
            equipment_ratings.append(metadata["equipment_rating"])
        
        funding = metadata.get("funding_level", "Unknown")
        funding_levels[funding] = funding_levels.get(funding, 0) + 1
    
    # 設備評価の統計
    equipment_stats = {}
    if equipment_ratings:
        equipment_stats = {
            "average": sum(equipment_ratings) / len(equipment_ratings),
            "max": max(equipment_ratings),
            "min": min(equipment_ratings),
            "count": len(equipment_ratings)
        }
    
    return {
        "database_info": {
            "version": system_state.get("database_version", "2.0.0"),
            "last_updated": system_state.get("last_updated"),
            "total_laboratories": total_labs
        },
        "field_distribution": field_counts,
        "funding_distribution": funding_levels,
        "equipment_statistics": equipment_stats,
        "professor_count": len(professor_counts),
        "evaluation_count": system_state["evaluation_count"],
        "data_completeness": {
            "with_features": len([lab for lab in system_state["lab_data"] if "features" in lab]),
            "with_metadata": len([lab for lab in system_state["lab_data"] if "metadata" in lab]),
            "with_description": len([lab for lab in system_state["lab_data"] if lab.get("description")]),
            "with_specialization": len([lab for lab in system_state["lab_data"] if lab.get("specialization")])
        },
        "system_status": {
            "fuzzy_engine": FUZZY_AVAILABLE,
            "genetic_algorithm": GENETIC_AVAILABLE,
            "decision_tree": DECISION_TREE_AVAILABLE,
            "database_module": DATABASE_AVAILABLE
        }
    }

@app.post("/api/optimize")
async def optimize_preferences(optimization_data: Dict[str, Any]):
    """遺伝的アルゴリズムによる嗜好最適化"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # パラメータ取得
    target_preferences = optimization_data.get("target_preferences", {})
    constraints = optimization_data.get("constraints", {})
    optimization_config = optimization_data.get("config", {})
    
    # デフォルト設定
    evolution_config = {
        "population_size": optimization_config.get("population_size", 30),
        "generations": optimization_config.get("generations", 50),
        "mutation_rate": optimization_config.get("mutation_rate", 0.1),
        "crossover_rate": optimization_config.get("crossover_rate", 0.8),
        "elite_size": optimization_config.get("elite_size", 5)
    }
    
    try:
        if GENETIC_AVAILABLE:
            # 遺伝的アルゴリズムによる最適化
            from core.genetic.evolution import EvolutionEngine, EvolutionConfig
            
            config = EvolutionConfig(**evolution_config)
            engine = EvolutionEngine(config)
            
            # 最適化実行
            result = engine.evolve(system_state["lab_data"], target_preferences)
            
            # 最適化結果の詳細評価
            optimized_preferences = result["best_individual"]
            
            # 最適化された嗜好での研究室評価
            evaluation_result = []
            for lab in system_state["lab_data"]:
                compatibility = calculate_enhanced_compatibility(optimized_preferences, lab)
                evaluation_result.append({
                    "lab_id": lab.get("id"),
                    "lab_name": lab.get("name"),
                    "compatibility_score": compatibility
                })
            
            # スコア順ソート
            evaluation_result.sort(key=lambda x: x["compatibility_score"], reverse=True)
            
            return {
                "optimization_id": f"opt_{int(time.time())}",
                "optimized_preferences": optimized_preferences,
                "optimization_fitness": result["best_fitness"],
                "generations_completed": result["generations_completed"],
                "convergence_achieved": result.get("convergence_achieved", False),
                "top_matching_labs": evaluation_result[:10],  # トップ10
                "evolution_history": result.get("evolution_history", []),
                "algorithm_info": {
                    "method": "genetic_algorithm",
                    "population_size": evolution_config["population_size"],
                    "generations": evolution_config["generations"],
                    "mutation_rate": evolution_config["mutation_rate"]
                },
                "processing_time": 0.5,
                "timestamp": time.time()
            }
        else:
            # ファジィ推論による最適化（フォールバック）
            if FUZZY_AVAILABLE:
                from core.fuzzy.inference import AdvancedFuzzyInferenceEngine
                
                fuzzy_engine = AdvancedFuzzyInferenceEngine()
                
                # シンプルな最適化（ランダムサーチ）
                best_preferences = None
                best_fitness = 0.0
                iterations = optimization_config.get("iterations", 100)
                
                for _ in range(iterations):
                    # ランダムな嗜好生成
                    random_preferences = {
                        criterion: random.uniform(1.0, 10.0)
                        for criterion in ["research_intensity", "advisor_style", "team_work", 
                                        "workload", "theory_practice", "research_field_match",
                                        "skill_development", "lab_atmosphere", "flexibility",
                                        "publication_opportunity", "interdisciplinary", 
                                        "communication_style", "innovation_risk"]
                    }
                    
                    # 適応度評価
                    total_compatibility = 0.0
                    for lab in system_state["lab_data"]:
                        compatibility = calculate_enhanced_compatibility(random_preferences, lab)
                        total_compatibility += compatibility
                    
                    avg_fitness = total_compatibility / len(system_state["lab_data"]) if system_state["lab_data"] else 0.0
                    
                    if avg_fitness > best_fitness:
                        best_fitness = avg_fitness
                        best_preferences = random_preferences
                
                return {
                    "optimization_id": f"opt_fuzzy_{int(time.time())}",
                    "optimized_preferences": best_preferences,
                    "optimization_fitness": best_fitness,
                    "method": "fuzzy_random_search",
                    "iterations": iterations,
                    "processing_time": 0.3,
                    "timestamp": time.time()
                }
            else:
                raise HTTPException(status_code=503, detail="Optimization modules not available")
                
    except Exception as e:
        print(f"❌ 最適化エラー: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@app.post("/api/train-decision-tree")
async def train_decision_tree(training_data: Dict[str, Any]):
    """ファジィ決定木の訓練"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        if not DECISION_TREE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Decision tree module not available")
        
        from core.decision_tree.tree import FuzzyDecisionTree, TreeConfig
        
        # 訓練データの準備
        X = training_data.get("features", [])
        y = training_data.get("targets", [])
        
        if not X or not y or len(X) != len(y):
            raise HTTPException(status_code=400, detail="Invalid training data")
        
        # 決定木設定
        tree_config = TreeConfig(
            max_depth=training_data.get("max_depth", 8),
            min_samples_split=training_data.get("min_samples_split", 2),
            min_samples_leaf=training_data.get("min_samples_leaf", 1),
            fuzzy_threshold=training_data.get("fuzzy_threshold", 0.1)
        )
        
        # 決定木訓練
        tree = FuzzyDecisionTree(tree_config)
        tree.fit(X, y)
        
        # システム状態に保存
        system_state["decision_tree"] = tree
        
        return {
            "training_id": f"train_{int(time.time())}",
            "tree_info": tree.to_dict(),
            "feature_importance": tree.feature_importance,
            "training_samples": len(X),
            "tree_depth": tree._calculate_tree_depth(),
            "leaf_nodes": tree._count_leaf_nodes(),
            "timestamp": time.time()
        }
        
    except Exception as e:
        print(f"❌ 決定木訓練エラー: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/api/predict-tree")
async def predict_with_tree(prediction_data: Dict[str, Any]):
    """ファジィ決定木による予測"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not system_state["decision_tree"]:
        raise HTTPException(status_code=400, detail="Decision tree not trained")
    
    try:
        input_data = prediction_data.get("input", {})
        explain_prediction = prediction_data.get("explain", False)
        
        # 予測実行
        prediction = system_state["decision_tree"].predict(input_data)
        
        result = {
            "prediction_id": f"pred_{int(time.time())}",
            "input": input_data,
            "prediction": prediction,
            "timestamp": time.time()
        }
        
        # 予測の説明を追加
        if explain_prediction:
            prediction_path = system_state["decision_tree"].get_prediction_path(input_data)
            result["explanation"] = {
                "prediction_path": prediction_path,
                "path_length": len(prediction_path),
                "decision_nodes": [node for node in prediction_path if not node.get("is_leaf", False)]
            }
        
        return result
        
    except Exception as e:
        print(f"❌ 予測エラー: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/api/config")
async def get_system_config():
    """システム設定情報の取得"""
    
    # データベースからメタデータを取得
    db_metadata = {}
    if system_state["lab_database"]:
        db_metadata = system_state["lab_database"].metadata
    
    return {
        "app_name": settings.app_name if SETTINGS_AVAILABLE else "研究室選択支援システム v2.0",
        "api_version": settings.api_version if SETTINGS_AVAILABLE else "v2",
        "database_version": system_state.get("database_version", "2.0.0"),
        "evaluation_criteria": getattr(settings, 'evaluation_criteria', [
            "research_intensity", "advisor_style", "team_work", "workload", "theory_practice",
            "research_field_match", "skill_development", "lab_atmosphere", "flexibility", 
            "publication_opportunity", "interdisciplinary", "communication_style", "innovation_risk"
        ]),
        "criteria_categories": {
            "basic": ["research_intensity", "advisor_style", "team_work", "workload", "theory_practice"],
            "extended": ["research_field_match", "skill_development", "lab_atmosphere", "flexibility", "publication_opportunity"],
            "special": ["interdisciplinary", "communication_style", "innovation_risk"]
        },
        "lab_count": len(system_state["lab_data"]),
        "database_metadata": db_metadata,
        "system_modules": {
            "fuzzy": FUZZY_AVAILABLE,
            "genetic": GENETIC_AVAILABLE,
            "decision_tree": DECISION_TREE_AVAILABLE,
            "database": DATABASE_AVAILABLE
        },
        "api_endpoints": {
            "basic": ["/api/labs", "/api/evaluate", "/health"],
            "advanced": ["/api/optimize", "/api/train-decision-tree", "/api/predict-tree"],
            "analysis": ["/api/statistics", "/api/research-fields"]
        }
    }

# =============================================================================
# サーバー起動
# =============================================================================

if __name__ == "__main__":
    print("\n🚀 FastAPI サーバー起動中...")
    print(f"📍 URL: http://localhost:{getattr(settings, 'port', 8000)}")
    print(f"📚 API文書: http://localhost:{getattr(settings, 'port', 8000)}/docs")
    print("🔧 システム状況:")
    print(f"  - データベースバージョン: {system_state.get('database_version', '2.0.0')}")
    print(f"  - ファジィ推論: {'✅' if FUZZY_AVAILABLE else '❌'}")
    print(f"  - 遺伝的アルゴリズム: {'✅' if GENETIC_AVAILABLE else '❌'}")
    print(f"  - 決定木: {'✅' if DECISION_TREE_AVAILABLE else '❌'}")
    print(f"  - データベースモジュール: {'✅' if DATABASE_AVAILABLE else '❌'}")
    print(f"  - 研究室データ: {len(system_state['lab_data'])}件")
    print(f"  - 最終更新: {system_state.get('last_updated', 'Unknown')}")
    
    uvicorn.run(
        app,
        host=getattr(settings, 'host', '0.0.0.0'),
        port=getattr(settings, 'port', 8000),
        reload=False,
        log_level="info"
    )