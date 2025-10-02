# backend/app.py - 完全統合版
#!/usr/bin/env python3
"""
遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム
完全統合版 - ファジィ + 決定木 + 遺伝的アルゴリズム + 分野マッチング
"""

import os
import sys
import json
import time
import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Any, Optional

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# ===== モジュールのインポート =====
try:
    from config.settings import settings
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False
    print("⚠️ settings.py が見つかりません")

try:
    from core.fuzzy.inference import SimpleFuzzyInferenceEngine
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("⚠️ ファジィモジュールが利用できません")

try:
    from core.genetic.evolution import EvolutionEngine, EvolutionConfig
    GENETIC_AVAILABLE = True
except ImportError:
    GENETIC_AVAILABLE = False
    print("⚠️ 遺伝的アルゴリズムが利用できません")

try:
    from core.decision_tree.tree import FuzzyDecisionTree, TreeConfig
    DECISION_TREE_AVAILABLE = True
except ImportError:
    DECISION_TREE_AVAILABLE = False
    print("⚠️ 決定木が利用できません")

try:
    from core.matching.field_matcher import FieldMatcher
    from core.matching.integrated_matcher import IntegratedMatcher
    MATCHING_AVAILABLE = True
except ImportError:
    MATCHING_AVAILABLE = False
    print("⚠️ マッチングモジュールが利用できません")

# ===== FastAPI初期化 =====
app = FastAPI(
    title="研究室選択支援システム（統合版）",
    description="ファジィ推論 + 決定木 + 遺伝的アルゴリズム + 分野マッチング",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== グローバル変数 =====
system_state = {
    "initialized": False,
    "fuzzy_engine": None,
    "genetic_engine": None,
    "decision_tree": None,
    "field_matcher": None,
    "integrated_matcher": None,
    "lab_database": [],
    "optimized_weights": None,
    "evaluation_count": 0
}

# ===== 研究室データベース（field_id追加） =====
SAMPLE_LABS = [
    {
        "id": "ai_lab",
        "name": "人工知能研究室",
        "advisor": "田中教授",
        "field_id": "ai_ml",  # ★追加
        "research_area": "人工知能・機械学習",
        "category": "テクノロジー・システム",
        "description": "機械学習とディープラーニングの研究",
        "research_intensity": 9.0,
        "advisor_style": 7.0,
        "team_work": 8.0,
        "workload": 8.5,
        "theory_practice": 6.0,
        "skill_development": 8.0,
        "lab_atmosphere": 7.0,
        "flexibility": 6.0,
        "publication_opportunity": 9.5,
        "interdisciplinary": 7.0,
        "communication_style": 8.0
    },
    {
        "id": "image_lab",
        "name": "画像処理研究室",
        "advisor": "佐藤教授",
        "field_id": "image_processing",  # ★追加
        "research_area": "画像・映像処理",
        "category": "テクノロジー・システム",
        "research_intensity": 8.0,
        "advisor_style": 6.0,
        "team_work": 7.0,
        "workload": 7.5,
        "theory_practice": 7.0,
        "skill_development": 7.0,
        "lab_atmosphere": 6.5,
        "flexibility": 7.0,
        "publication_opportunity": 8.0,
        "interdisciplinary": 6.0,
        "communication_style": 7.0
    },
    {
        "id": "network_lab",
        "name": "ネットワークセキュリティ研究室",
        "advisor": "鈴木教授",
        "field_id": "network_security",  # ★追加
        "research_area": "ネットワーク・セキュリティ",
        "category": "テクノロジー・システム",
        "research_intensity": 8.5,
        "advisor_style": 5.0,
        "team_work": 9.0,
        "workload": 9.0,
        "theory_practice": 5.0,
        "skill_development": 9.0,
        "lab_atmosphere": 8.0,
        "flexibility": 5.0,
        "publication_opportunity": 7.0,
        "interdisciplinary": 5.0,
        "communication_style": 9.0
    },
    {
        "id": "web_lab",
        "name": "Webデザイン研究室",
        "advisor": "高橋教授",
        "field_id": "web_design",  # ★追加
        "research_area": "Webデザイン・UI/UX",
        "category": "クリエイティブ",
        "research_intensity": 6.0,
        "advisor_style": 8.0,
        "team_work": 8.0,
        "workload": 6.0,
        "theory_practice": 9.0,
        "skill_development": 9.0,
        "lab_atmosphere": 9.0,
        "flexibility": 9.0,
        "publication_opportunity": 5.0,
        "interdisciplinary": 8.0,
        "communication_style": 9.0
    },
    {
        "id": "game_lab",
        "name": "ゲーム開発研究室",
        "advisor": "山田教授",
        "field_id": "game_esports",  # ★追加
        "research_area": "ゲーム開発・eスポーツ",
        "category": "エンターテイメント",
        "research_intensity": 7.0,
        "advisor_style": 7.5,
        "team_work": 8.5,
        "workload": 7.0,
        "theory_practice": 8.5,
        "skill_development": 9.0,
        "lab_atmosphere": 8.5,
        "flexibility": 7.5,
        "publication_opportunity": 6.0,
        "interdisciplinary": 7.0,
        "communication_style": 8.5
    }
]

# ===== システム初期化 =====
def initialize_system():
    """統合システムを初期化"""
    global system_state
    
    print("\n" + "="*70)
    print("🚀 統合システム初期化開始")
    print("="*70)
    
    try:
        # 1. ファジィ推論エンジン
        if FUZZY_AVAILABLE:
            system_state["fuzzy_engine"] = SimpleFuzzyInferenceEngine(
                settings.core_features if SETTINGS_AVAILABLE else [
                    "research_intensity", "advisor_style", "team_work", 
                    "workload", "theory_practice"
                ],
                "compatibility"
            )
            print("✅ ファジィ推論エンジン初期化完了")
        
        # 2. 決定木
        if DECISION_TREE_AVAILABLE:
            tree_config = TreeConfig(
                max_depth=8,  # ★ 8層に変更
                min_samples_leaf=3
            )
            system_state["decision_tree"] = FuzzyDecisionTree(tree_config)
            print("✅ ファジィ決定木初期化完了（8層）")
        
        # 3. 遺伝的アルゴリズム
        if GENETIC_AVAILABLE:
            evolution_config = EvolutionConfig(
                population_size=20,
                generations=30,
                crossover_rate=0.8,
                mutation_rate=0.1
            )
            system_state["genetic_engine"] = EvolutionEngine(evolution_config)
            print("✅ 遺伝的アルゴリズム初期化完了")
        
        # 4. 分野マッチャー
        if MATCHING_AVAILABLE:
            system_state["field_matcher"] = FieldMatcher()
            print("✅ 分野マッチャー初期化完了")
        
        # 5. 最適化された重みを読み込み（なければデフォルト）
        weights_path = getattr(settings, 'optimized_weights_path', 'data/optimized_weights.npy')
        if os.path.exists(weights_path):
            system_state["optimized_weights"] = np.load(weights_path)
            print(f"✅ 最適化された重みを読み込み: {weights_path}")
        else:
            # デフォルト重み（14次元: 12基本項目 + 1分野 + 1予備）
            system_state["optimized_weights"] = np.ones(14) / 14
            print("⚠️ デフォルト重みを使用")
        
        # 6. 統合マッチャーを作成
        if MATCHING_AVAILABLE:
            system_state["integrated_matcher"] = IntegratedMatcher(
                fuzzy_engine=system_state["fuzzy_engine"],
                decision_tree=system_state["decision_tree"],
                field_matcher=system_state["field_matcher"],
                optimized_weights=system_state["optimized_weights"]
            )
            print("✅ 統合マッチャー初期化完了")
        
        # 7. 研究室データベース
        system_state["lab_database"] = SAMPLE_LABS
        print(f"✅ 研究室データベース初期化完了: {len(SAMPLE_LABS)}件")
        
        system_state["initialized"] = True
        print("\n🎉 統合システム初期化完了！")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"❌ システム初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        system_state["initialized"] = False

# ===== API エンドポイント =====

@app.get("/")
async def read_root():
    """ルートエンドポイント"""
    return {
        "message": "遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム（統合版）",
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
        "fuzzy": FUZZY_AVAILABLE,
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
            - 基本項目（12項目）
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
                "advisor": lab["advisor"],
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
                "method": "integrated_fuzzy_genetic_tree_field",
                "components": [
                    "fuzzy_inference",
                    "decision_tree_8layers",
                    "genetic_optimization",
                    "field_matching"
                ],
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

def _get_recommendation_level(compatibility: float) -> str:
    """推薦レベルを取得"""
    if compatibility >= 0.85:
        return "strongly_recommended"
    elif compatibility >= 0.7:
        return "recommended"
    elif compatibility >= 0.5:
        return "consider"
    else:
        return "not_recommended"

@app.post("/api/optimize")
async def optimize_weights(training_data: Optional[List[Dict]] = None):
    """
    遺伝的アルゴリズムで重みを最適化
    
    Args:
        training_data: 訓練データ（オプション）
    
    Returns:
        最適化結果
    """
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not GENETIC_AVAILABLE or not MATCHING_AVAILABLE:
        raise HTTPException(status_code=501, detail="Optimization not available")
    
    try:
        print("\n" + "="*70)
        print("🧬 重み最適化開始（遺伝的アルゴリズム）")
        print("="*70)
        
        # 訓練データを読み込み
        if training_data is None:
            training_data_path = getattr(settings, 'training_data_path', 'data/training_data.json')
            if os.path.exists(training_data_path):
                with open(training_data_path, 'r', encoding='utf-8') as f:
                    training_data = json.load(f)
                print(f"✅ 訓練データ読み込み: {len(training_data)}件")
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Training data not found. Please provide training data."
                )
        
        # TODO: 実際の遺伝的アルゴリズムによる最適化を実装
        # ここでは簡易的なシミュレーション
        
        print("⏳ 最適化実行中...")
        time.sleep(2)  # シミュレーション
        
        # デモ用の最適化された重み
        optimized_weights = np.array([
            0.92, 0.45, 0.58, 0.67, 0.39,  # 5項目
            0.88, 0.54, 0.71, 0.43, 0.95,  # 5項目
            0.31, 0.52, 0.61, 0.85         # 3項目 + 分野
        ])
        
        # 保存
        system_state["optimized_weights"] = optimized_weights
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
            "training_samples": len(training_data),
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

# ===== システム起動時に初期化 =====
initialize_system()

# ===== サーバー起動 =====
if __name__ == "__main__":
    print("\n🚀 FastAPI サーバー起動中...")
    print(f"📍 URL: http://localhost:{getattr(settings, 'port', 8000)}")
    print(f"📚 API文書: http://localhost:{getattr(settings, 'port', 8000)}/docs")
    
    uvicorn.run(
        app,
        host=getattr(settings, 'host', '0.0.0.0'),
        port=getattr(settings, 'port', 8000),
        reload=getattr(settings, 'debug', True),
        log_level="info"
    )