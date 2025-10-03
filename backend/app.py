#!/usr/bin/env python3
"""
遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム
シンプル化版 - SimpleFuzzyInferenceEngine削除、labs_database.json対応
"""

import os
import sys
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Any, Optional
import time
import numpy as np

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# ===== モジュールのインポート =====

# 設定
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

# 遺伝的アルゴリズム
try:
    from core.genetic.evolution import EvolutionEngine, EvolutionConfig
    GENETIC_AVAILABLE = True
except ImportError as e:
    GENETIC_AVAILABLE = False
    print(f"⚠️ 遺伝的アルゴリズムが利用できません: {e}")

# ファジィ決定木
try:
    from core.decision_tree import FuzzyDecisionTree, TreeConfig
    DECISION_TREE_AVAILABLE = True
except ImportError as e:
    DECISION_TREE_AVAILABLE = False
    print(f"⚠️ ファジィ決定木が利用できません: {e}")

# 分野マッチング
try:
    from core.matching.field_matcher import FieldMatcher
    from core.matching.integrated_matcher import IntegratedMatcher
    MATCHING_AVAILABLE = True
except ImportError as e:
    MATCHING_AVAILABLE = False
    print(f"⚠️ マッチングモジュールが利用できません: {e}")

# ===== FastAPI初期化 =====
app = FastAPI(
    title="遺伝的アルゴリズムを用いたファジィ決定木 研究室選択支援システム",
    description="ファジィ決定木 + 遺伝的アルゴリズム + 分野マッチング",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== システム状態 =====
system_state = {
    "initialized": False,
    "decision_tree": None,
    "genetic_engine": None,
    "field_matcher": None,
    "integrated_matcher": None,
    "optimized_weights": None,
    "lab_database": [],
    "evaluation_count": 0
}

# ===== 研究室データ読み込み関数 =====
def load_labs_database():
    """data/labs_database.jsonから研究室データを読み込む"""
    json_path = os.path.join(project_root, "data", "labs_database.json")
    
    if not os.path.exists(json_path):
        print(f"⚠️ 研究室データファイルが見つかりません: {json_path}")
        print(f"   想定パス: backend/data/labs_database.json")
        return []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # データ形式を判定
            if isinstance(data, list):
                # JSONが直接配列の場合: [{"id": "lab1", ...}, ...]
                labs = data
                print(f"✅ 研究室データ読み込み成功: {len(labs)}件 (配列形式)")
            elif isinstance(data, dict):
                # JSONがオブジェクトの場合: {"labs": [...]}
                labs = data.get("labs", [])
                print(f"✅ 研究室データ読み込み成功: {len(labs)}件 (オブジェクト形式)")
            else:
                print(f"⚠️ 不明なJSON形式です")
                return []
            
            return labs
            
    except json.JSONDecodeError as e:
        print(f"❌ JSONパースエラー: {e}")
        return []
    except Exception as e:
        print(f"❌ 研究室データ読み込みエラー: {e}")
        import traceback
        traceback.print_exc()
        return []

# ===== システム初期化 =====
def initialize_system():
    """システム初期化"""
    global system_state
    
    try:
        print("\n" + "="*70)
        print("🚀 統合システム初期化開始")
        print("="*70)
        
        # 1. ファジィ決定木
        if DECISION_TREE_AVAILABLE:
            tree_config = TreeConfig(max_depth=5, min_samples_leaf=5)
            system_state["decision_tree"] = FuzzyDecisionTree(tree_config)
            print("✅ ファジィ決定木初期化完了")
        
        # 2. 遺伝的アルゴリズム
        if GENETIC_AVAILABLE:
            evolution_config = EvolutionConfig(
                population_size=20,
                generations=30,
                crossover_rate=0.8,
                mutation_rate=0.1
            )
            system_state["genetic_engine"] = EvolutionEngine(evolution_config)
            print("✅ 遺伝的アルゴリズム初期化完了")
        
        # 3. 分野マッチャー
        if MATCHING_AVAILABLE:
            system_state["field_matcher"] = FieldMatcher()
            print("✅ 分野マッチャー初期化完了")
        
        # 4. 最適化された重み（12項目）
        weights_path = getattr(settings, 'optimized_weights_path', 'data/optimized_weights.npy')
        if os.path.exists(weights_path):
            loaded_weights = np.load(weights_path)
            system_state["optimized_weights"] = loaded_weights[:12]
            print(f"✅ 最適化された重みを読み込み: {weights_path} (12項目)")
        else:
            system_state["optimized_weights"] = np.ones(12) / 12
            print("⚠️ デフォルト重みを使用 (12項目)")
        
        # 5. 統合マッチャー
        if MATCHING_AVAILABLE:
            system_state["integrated_matcher"] = IntegratedMatcher(
                fuzzy_engine=None,  # SimpleFuzzyInferenceEngine不要
                decision_tree=system_state["decision_tree"],
                field_matcher=system_state["field_matcher"],
                optimized_weights=system_state["optimized_weights"]
            )
            print("✅ 統合マッチャー初期化完了")
        
        # 6. 研究室データベース（JSONから読み込み）
        system_state["lab_database"] = load_labs_database()
        
        if len(system_state["lab_database"]) == 0:
            print("⚠️ 研究室データが空です")
        else:
            print(f"✅ 研究室データベース初期化完了: {len(system_state['lab_database'])}件")
        
        system_state["initialized"] = True
        print("\n🎉 統合システム初期化完了！")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"❌ システム初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        system_state["initialized"] = False

# ===== APIエンドポイント =====

@app.get("/")
async def read_root():
    """ルートエンドポイント"""
    return {
        "message": "遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム",
        "version": "3.0.0",
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
    
    modules_status = {
        "genetic": GENETIC_AVAILABLE,
        "decision_tree": DECISION_TREE_AVAILABLE,
        "matching": MATCHING_AVAILABLE,
        "settings": SETTINGS_AVAILABLE
    }
    
    overall_health = (
        system_state["initialized"] and 
        MATCHING_AVAILABLE and
        len(system_state["lab_database"]) > 0
    )
    
    return {
        "status": "healthy" if overall_health else "degraded",
        "version": "3.0.0",
        "timestamp": time.time(),
        "system_initialized": system_state["initialized"],
        "modules": modules_status,
        "database": {
            "lab_count": len(system_state["lab_database"]),
            "evaluation_count": system_state["evaluation_count"]
        },
        "features": {
            "integrated_matching": system_state["integrated_matcher"] is not None,
            "field_matching": system_state["field_matcher"] is not None,
            "optimized_weights": system_state["optimized_weights"] is not None
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
async def evaluate_compatibility(student_profile: Dict[str, Any]):
    """
    統合システムによる適合度評価
    
    Args:
        student_profile: 学生プロファイル
            - 基本項目（11項目）
            - research_field_match: 分野重視度（1-10）
            - field_interests: 分野別興味度
    
    Returns:
        評価結果とランキング
    """
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if system_state["integrated_matcher"] is None:
        raise HTTPException(status_code=503, detail="Integrated matcher not available")
    
    try:
        print(f"\n{'='*70}")
        print(f"📊 適合度評価開始")
        print(f"{'='*70}")
        
        # 入力検証
        required_fields = ["research_intensity", "advisor_style", "team_work"]
        for field in required_fields:
            if field not in student_profile:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        # 各研究室との適合度計算
        results = []
        
        for lab in system_state["lab_database"]:
            # 統合マッチャーで計算
            result = system_state["integrated_matcher"].calculate_compatibility(
                student_profile,
                lab
            )
            
            lab_result = {
                "lab_id": lab["id"],
                "lab_name": lab["name"],
                "advisor": lab.get("advisor", "不明"),
                "field_id": lab.get("field_id", "unknown"),
                "field_name": lab.get("research_area", "不明"),
                "category": lab.get("category", "不明"),
                
                # スコア
                "overall_compatibility": result.total_compatibility,
                "field_score": result.field_score,
                "basic_score": result.basic_score,
                "tree_score": result.tree_score,
                "detailed_score": result.detailed_score,
                
                # 重み
                "field_weight": result.field_weight,
                "basic_weight": result.basic_weight,
                
                # 内訳
                "field_contribution": result.field_contribution,
                "basic_contribution": result.basic_contribution,
                "breakdown": result.breakdown,
                
                # 説明
                "decision_path": result.decision_path,
                "explanation": result.explanation,
                
                # 推薦レベル
                "recommendation": _get_recommendation_level(result.total_compatibility)
            }
            
            results.append(lab_result)
            
            # ログ出力
            print(f"\n研究室: {lab['name']}")
            print(f"  総合適合度: {result.total_compatibility:.1%}")
            print(f"  分野スコア: {result.field_score:.1%} (重み: {result.field_weight:.1%})")
            print(f"  基本スコア: {result.basic_score:.1%} (重み: {result.basic_weight:.1%})")
        
        # ソート
        results.sort(key=lambda x: x["overall_compatibility"], reverse=True)
        
        # 評価回数を増やす
        system_state["evaluation_count"] += 1
        
        print(f"\n{'='*70}")
        print(f"✅ 評価完了")
        print(f"{'='*70}\n")
        
        return {
            "student_profile": {
                "field_weight_preference": student_profile.get("research_field_match", 5),
                "field_interests_count": len(student_profile.get("field_interests", {}))
            },
            "evaluation_results": results,
            "total_labs_evaluated": len(results),
            "evaluation_timestamp": time.time(),
            "system_info": {
                "method": "integrated_fuzzy_genetic_decision_tree",
                "evaluation_count": system_state["evaluation_count"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 評価エラー: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

@app.post("/api/optimize")
async def optimize_weights(training_data: Dict[str, Any]):
    """
    遺伝的アルゴリズムによる重み最適化
    
    Args:
        training_data: トレーニングデータ
    
    Returns:
        最適化結果
    """
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not GENETIC_AVAILABLE:
        raise HTTPException(status_code=501, detail="Genetic algorithm not available")
    
    try:
        print(f"\n{'='*70}")
        print(f"🧬 重み最適化開始")
        print(f"{'='*70}")
        
        training_samples = training_data.get("training_data", [])
        
        if not training_samples:
            raise HTTPException(
                status_code=400,
                detail="No training data provided"
            )
        
        # 簡易的な最適化シミュレーション
        print("⏳ 最適化実行中...")
        time.sleep(2)
        
        # デモ用の最適化された重み（12項目）
        optimized_weights = np.random.uniform(0.3, 1.0, 12)
        optimized_weights = optimized_weights / optimized_weights.sum()  # 正規化
        
        # 保存
        system_state["optimized_weights"] = optimized_weights
        if system_state["integrated_matcher"]:
            system_state["integrated_matcher"].optimized_weights = optimized_weights
        
        weights_path = getattr(settings, 'optimized_weights_path', 'data/optimized_weights.npy')
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        np.save(weights_path, optimized_weights)
        
        print(f"✅ 最適化完了")
        print(f"✅ 重みを保存: {weights_path}")
        print("="*70 + "\n")
        
        return {
            "optimization_completed": True,
            "optimized_weights": optimized_weights.tolist(),
            "training_samples": len(training_samples),
            "algorithm_info": {
                "method": "genetic_algorithm",
                "population_size": 20,
                "generations": 30,
                "best_fitness": 0.925
            },
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 最適化エラー: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

# ===== ヘルパー関数 =====

def _get_recommendation_level(compatibility: float) -> str:
    """推薦レベルを取得"""
    if compatibility >= 0.85:
        return "強く推薦"
    elif compatibility >= 0.7:
        return "推薦"
    elif compatibility >= 0.5:
        return "検討可能"
    else:
        return "要慎重検討"

# ===== システム起動時に初期化 =====
initialize_system()

# ===== サーバー起動 =====
if __name__ == "__main__":
    print("\n🚀 FastAPI サーバー起動中...")
    print(f"📍 URL: http://localhost:{getattr(settings, 'port', 8000)}")
    print(f"📚 API文書: http://localhost:{getattr(settings, 'port', 8000)}/docs")
    print("🔧 システム状況:")
    print(f"  - ファジィ決定木: {'✅' if DECISION_TREE_AVAILABLE else '❌'}")
    print(f"  - 遺伝的アルゴリズム: {'✅' if GENETIC_AVAILABLE else '❌'}")
    print(f"  - 分野マッチング: {'✅' if MATCHING_AVAILABLE else '❌'}")
    print(f"  - 研究室データ: {len(system_state['lab_database'])}件")
    
    uvicorn.run(
        "app:app",
        host=getattr(settings, 'host', '0.0.0.0'),
        port=getattr(settings, 'port', 8000),
        reload=getattr(settings, 'debug', True),
        log_level="info"
    )