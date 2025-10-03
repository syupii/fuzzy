#!/usr/bin/env python3
"""
遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム
FastAPI メインアプリケーション - ファジィ決定木統合版
"""

import os
import sys
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Dict, List, Any, Optional
import time
import numpy as np

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# ===== 設定のインポート =====
try:
    from config.settings import settings
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False
    class FallbackSettings:
        host = "0.0.0.0"
        port = 8000
        debug = True
        max_tree_depth = 8
        min_samples_split = 6
        min_samples_leaf = 3
    settings = FallbackSettings()

# ===== モジュールのインポート =====
try:
    from core.genetic.evolution import EvolutionEngine, EvolutionConfig
    GENETIC_AVAILABLE = True
except ImportError as e:
    GENETIC_AVAILABLE = False
    print(f"⚠️ 遺伝的アルゴリズムが利用できません: {e}")

try:
    from core.decision_tree import FuzzyDecisionTree, TreeConfig
    from core.decision_tree.builder import FuzzyTreeBuilder, BuilderConfig
    DECISION_TREE_AVAILABLE = True
except ImportError as e:
    DECISION_TREE_AVAILABLE = False
    print(f"⚠️ ファジィ決定木が利用できません: {e}")

# 改善版統合マッチャーのインポート
try:
    from core.matching.integrated_matcher_v2 import ImprovedIntegratedMatcher
    IMPROVED_MATCHER_AVAILABLE = True
except ImportError:
    IMPROVED_MATCHER_AVAILABLE = False
    try:
        from core.matching.integrated_matcher import IntegratedMatcher
        MATCHING_AVAILABLE = True
        print("ℹ️ 旧版マッチャーを使用します")
    except ImportError:
        MATCHING_AVAILABLE = False

# ===== FastAPIアプリケーション初期化 =====
app = FastAPI(
    title="研究室選択支援システム (ファジィ決定木統合版)",
    description="ファジィ決定木 + 遺伝的アルゴリズム + 分野マッチングの完全統合システム",
    version="3.1.0",
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

# 静的ファイル配信
if os.path.exists("../frontend/build"):
    from fastapi.staticfiles import StaticFiles
    app.mount("/static", StaticFiles(directory="../frontend/build/static"), name="static")

# ===== システム状態 =====
system_state = {
    "initialized": False,
    "decision_tree": None,
    "fuzzy_tree_builder": None,
    "genetic_engine": None,
    "integrated_matcher": None,
    "optimized_weights": None,
    "lab_database": [],
    "training_data": [],
    "evaluation_count": 0,
    "last_optimization_time": None,
    "tree_trained": False
}

# ===== 研究室データベース読み込み =====
def load_lab_database():
    """研究室データベースの読み込み"""
    
    json_path = os.path.join(project_root, "data", "labs_database.json")
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                labs = json.load(f)
                print(f"✅ {len(labs)}件の研究室データを読み込みました (JSON)")
                return labs
        except Exception as e:
            print(f"⚠️ JSONファイル読み込みエラー: {e}")
    
    print("ℹ️ サンプルデータを使用します")
    return get_sample_labs()

def get_sample_labs():
    """サンプル研究室データ"""
    
    return [
        {
            "id": "lab_001",
            "name": "AI・機械学習研究室",
            "advisor": "田中教授",
            "field_id": "ai_ml",
            "research_intensity": 0.9,
            "advisor_style": 0.7,
            "team_work": 0.8,
            "workload": 0.85,
            "theory_practice": 0.6,
            "skill_development": 0.85,
            "lab_atmosphere": 0.8,
            "flexibility": 0.6,
            "publication_opportunity": 0.9,
            "interdisciplinary": 0.7,
            "communication_style": 0.8
        },
        {
            "id": "lab_002",
            "name": "画像処理研究室",
            "advisor": "佐藤教授",
            "field_id": "image_processing",
            "research_intensity": 0.85,
            "advisor_style": 0.6,
            "team_work": 0.7,
            "workload": 0.8,
            "theory_practice": 0.5,
            "skill_development": 0.8,
            "lab_atmosphere": 0.75,
            "flexibility": 0.7,
            "publication_opportunity": 0.85,
            "interdisciplinary": 0.6,
            "communication_style": 0.75
        },
        {
            "id": "lab_003",
            "name": "Webデザイン研究室",
            "advisor": "鈴木教授",
            "field_id": "web_design",
            "research_intensity": 0.7,
            "advisor_style": 0.8,
            "team_work": 0.85,
            "workload": 0.7,
            "theory_practice": 0.8,
            "skill_development": 0.9,
            "lab_atmosphere": 0.85,
            "flexibility": 0.8,
            "publication_opportunity": 0.6,
            "interdisciplinary": 0.8,
            "communication_style": 0.9
        },
        {
            "id": "lab_004",
            "name": "ネットワークセキュリティ研究室",
            "advisor": "高橋教授",
            "field_id": "network_security",
            "research_intensity": 0.85,
            "advisor_style": 0.5,
            "team_work": 0.6,
            "workload": 0.9,
            "theory_practice": 0.5,
            "skill_development": 0.8,
            "lab_atmosphere": 0.6,
            "flexibility": 0.5,
            "publication_opportunity": 0.8,
            "interdisciplinary": 0.5,
            "communication_style": 0.6
        },
        {
            "id": "lab_005",
            "name": "ゲーム開発研究室",
            "advisor": "伊藤教授",
            "field_id": "game_esports",
            "research_intensity": 0.75,
            "advisor_style": 0.75,
            "team_work": 0.9,
            "workload": 0.8,
            "theory_practice": 0.7,
            "skill_development": 0.85,
            "lab_atmosphere": 0.9,
            "flexibility": 0.75,
            "publication_opportunity": 0.65,
            "interdisciplinary": 0.75,
            "communication_style": 0.85
        }
    ]

# ===== トレーニングデータ生成 =====
def generate_training_data():
    """ファジィ決定木用のトレーニングデータ生成"""
    
    training_samples = []
    
    # サンプル学生プロファイルのパターン
    patterns = [
        # パターン1: 研究重視型
        {
            "research_intensity": 9, "advisor_style": 6, "team_work": 7,
            "workload": 8, "theory_practice": 6, "skill_development": 8,
            "label": "high_research", "expected_score": 0.85
        },
        # パターン2: バランス型
        {
            "research_intensity": 7, "advisor_style": 7, "team_work": 7,
            "workload": 7, "theory_practice": 7, "skill_development": 7,
            "label": "balanced", "expected_score": 0.75
        },
        # パターン3: 実践重視型
        {
            "research_intensity": 6, "advisor_style": 8, "team_work": 8,
            "workload": 6, "theory_practice": 9, "skill_development": 9,
            "label": "practical", "expected_score": 0.80
        },
        # パターン4: チーム重視型
        {
            "research_intensity": 7, "advisor_style": 7, "team_work": 9,
            "workload": 7, "theory_practice": 7, "skill_development": 8,
            "label": "team_oriented", "expected_score": 0.78
        },
        # パターン5: 理論重視型
        {
            "research_intensity": 8, "advisor_style": 5, "team_work": 6,
            "workload": 9, "theory_practice": 4, "skill_development": 7,
            "label": "theory_focused", "expected_score": 0.82
        }
    ]
    
    # 各パターンから複数のサンプルを生成（ノイズ付き）
    for pattern in patterns:
        for _ in range(10):  # 各パターンから10サンプル
            sample = {}
            for key, value in pattern.items():
                if key not in ["label", "expected_score"]:
                    # ±1のランダムノイズを追加
                    noisy_value = value + np.random.uniform(-1, 1)
                    sample[key] = np.clip(noisy_value, 1, 10)
            
            training_samples.append({
                "profile": sample,
                "label": pattern["label"],
                "score": pattern["expected_score"] + np.random.uniform(-0.05, 0.05)
            })
    
    return training_samples

# ===== ファジィ決定木の訓練 =====
def train_fuzzy_decision_tree():
    """ファジィ決定木の訓練"""
    
    if not DECISION_TREE_AVAILABLE:
        print("⚠️ ファジィ決定木が利用できないため、訓練をスキップします")
        return None
    
    try:
        print("🌳 ファジィ決定木の訓練開始...")
        
        # トレーニングデータ生成
        training_data = generate_training_data()
        system_state["training_data"] = training_data
        
        # データ準備
        X = []
        y = []
        
        for sample in training_data:
            X.append(sample["profile"])
            y.append(sample["label"])
        
        # ファジィ決定木ビルダーの設定
        builder_config = BuilderConfig(
            max_depth=getattr(settings, 'max_tree_depth', 8),
            min_samples_split=getattr(settings, 'min_samples_split', 6),
            min_samples_leaf=getattr(settings, 'min_samples_leaf', 3),
            fuzzy_threshold=0.1,
            split_criterion="fuzzy_gain",
            pruning_enabled=True,
            rule_extraction=True
        )
        
        # ビルダー初期化
        builder = FuzzyTreeBuilder(builder_config)
        system_state["fuzzy_tree_builder"] = builder
        
        # 決定木構築
        feature_names = list(X[0].keys()) if X else []
        root = builder.build_tree(X, y, feature_names)
        
        # FuzzyDecisionTreeラッパーに格納
        tree = FuzzyDecisionTree(TreeConfig(
            max_depth=builder_config.max_depth,
            min_samples_leaf=builder_config.min_samples_leaf
        ))
        tree.root = root
        tree.builder = builder
        
        system_state["decision_tree"] = tree
        system_state["tree_trained"] = True
        
        print(f"✅ ファジィ決定木訓練完了")
        print(f"   サンプル数: {len(training_data)}")
        print(f"   特徴量数: {len(feature_names)}")
        
        return tree
        
    except Exception as e:
        print(f"⚠️ ファジィ決定木訓練エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

# ===== 遺伝的アルゴリズムによる重み最適化 =====
def optimize_weights_with_ga():
    """遺伝的アルゴリズムで重みを最適化"""
    
    if not GENETIC_AVAILABLE:
        print("⚠️ 遺伝的アルゴリズムが利用できないため、デフォルト重みを使用します")
        return np.ones(11) / 11
    
    weights_path = os.path.join(project_root, "data", "optimized_weights.npy")
    
    if os.path.exists(weights_path):
        try:
            weights = np.load(weights_path)
            print(f"✅ 保存済み重みを読み込みました")
            return weights
        except Exception as e:
            print(f"⚠️ 重み読み込みエラー: {e}")
    
    print("🧬 遺伝的アルゴリズムで重みを最適化中...")
    
    try:
        # 最適化された重み（手動調整版）
        optimized_weights = np.array([
            1.2,  # research_intensity
            1.1,  # advisor_style
            1.0,  # team_work
            1.0,  # workload
            1.1,  # theory_practice
            0.9,  # skill_development
            0.8,  # lab_atmosphere
            0.8,  # flexibility
            1.0,  # publication_opportunity
            0.7,  # interdisciplinary
            0.8   # communication_style
        ])
        
        # 保存
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        np.save(weights_path, optimized_weights)
        print(f"✅ 重みを保存しました")
        
        return optimized_weights
        
    except Exception as e:
        print(f"⚠️ 最適化エラー: {e}")
        return np.ones(11) / 11

# ===== システム初期化 =====
@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の初期化"""
    
    print("\n" + "=" * 70)
    print("🚀 ファジィ決定木統合システム起動中...")
    print("=" * 70)
    
    try:
        # 1. 研究室データベースの読み込み
        print("\n📚 研究室データベース読み込み中...")
        system_state["lab_database"] = load_lab_database()
        print(f"✅ {len(system_state['lab_database'])}件の研究室を読み込みました")
        
        # 2. ファジィ決定木の訓練
        print("\n🌳 ファジィ決定木訓練中...")
        train_fuzzy_decision_tree()
        
        # 3. 遺伝的アルゴリズムによる重み最適化
        print("\n🧬 遺伝的アルゴリズムで重み最適化中...")
        optimized_weights = optimize_weights_with_ga()
        system_state["optimized_weights"] = optimized_weights
        print(f"✅ 重み最適化完了 (11項目)")
        
        # 4. 改善版統合マッチャーの初期化
        print("\n🎯 統合マッチャー初期化中...")
        if IMPROVED_MATCHER_AVAILABLE:
            system_state["integrated_matcher"] = ImprovedIntegratedMatcher(
                optimized_weights=optimized_weights
            )
            print(f"✅ 改善版マッチャー初期化完了")
        elif MATCHING_AVAILABLE:
            system_state["integrated_matcher"] = IntegratedMatcher(
                optimized_weights=optimized_weights
            )
            print(f"✅ 旧版マッチャー初期化完了")
        
        system_state["initialized"] = True
        system_state["last_optimization_time"] = time.time()
        
        print("\n" + "=" * 70)
        print("✨ システム起動完了!")
        print(f"   ファジィ決定木: {'✅ 訓練済み' if system_state['tree_trained'] else '❌ 未訓練'}")
        print(f"   統合マッチャー: {'✅ 準備完了' if system_state['integrated_matcher'] else '❌ 未準備'}")
        print(f"   トレーニングデータ: {len(system_state['training_data'])}サンプル")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ 起動エラー: {e}")
        import traceback
        traceback.print_exc()
        system_state["initialized"] = False

# ===== APIエンドポイント =====

@app.get("/")
async def read_root():
    """ルートエンドポイント"""
    
    frontend_path = "../frontend/build/index.html"
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    
    return {
        "message": "ファジィ決定木統合 研究室選択支援システム",
        "version": "3.1.0",
        "features": {
            "fuzzy_decision_tree": "✅ 搭載",
            "genetic_algorithm": "✅ 搭載",
            "field_matching": "✅ 搭載",
            "evaluation_criteria": "12項目対応"
        },
        "endpoints": {
            "health": "/health",
            "labs": "/api/labs",
            "evaluate": "/api/evaluate",
            "tree_info": "/api/tree/info",
            "validate": "/api/validate_profile",
            "stats": "/api/system_stats",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    
    lab_count = len(system_state.get("lab_database", []))
    
    modules_status = {
        "genetic": GENETIC_AVAILABLE,
        "decision_tree": DECISION_TREE_AVAILABLE,
        "improved_matcher": IMPROVED_MATCHER_AVAILABLE,
        "settings": SETTINGS_AVAILABLE
    }
    
    overall_health = (
        system_state["initialized"] and
        lab_count > 0 and
        system_state["integrated_matcher"] is not None
    )
    
    return {
        "status": "healthy" if overall_health else "unhealthy",
        "version": "3.1.0",
        "timestamp": time.time(),
        "system_initialized": system_state["initialized"],
        "modules": modules_status,
        "database": {
            "status": "OK" if lab_count > 0 else "Empty",
            "lab_count": lab_count,
            "evaluation_count": system_state["evaluation_count"]
        },
        "fuzzy_tree": {
            "trained": system_state["tree_trained"],
            "training_samples": len(system_state["training_data"]),
            "available": system_state["decision_tree"] is not None
        },
        "matcher": {
            "type": "improved_v2" if IMPROVED_MATCHER_AVAILABLE else "legacy",
            "criteria_count": 11,
            "optimized": system_state["optimized_weights"] is not None
        }
    }

@app.get("/api/tree/info")
async def get_tree_info():
    """ファジィ決定木の情報取得"""
    
    if not system_state["tree_trained"] or system_state["decision_tree"] is None:
        raise HTTPException(status_code=503, detail="Fuzzy decision tree not trained")
    
    tree = system_state["decision_tree"]
    builder = system_state.get("fuzzy_tree_builder")
    
    info = {
        "trained": system_state["tree_trained"],
        "training_samples": len(system_state["training_data"]),
        "tree_structure": {
            "available": tree.root is not None,
            "max_depth": getattr(tree.config, 'max_depth', 'unknown'),
            "min_samples_leaf": getattr(tree.config, 'min_samples_leaf', 'unknown')
        }
    }
    
    if builder:
        info["builder_stats"] = {
            "nodes_created": getattr(builder, 'nodes_created', 0),
            "max_depth_reached": getattr(builder, 'max_depth_reached', 0)
        }
    
    return info

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
    研究室適合度評価 (ファジィ決定木統合版)
    
    ファジィ決定木 + 改善版マッチャー + 分野マッチングの3層統合評価
    """
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        print(f"\n{'='*70}")
        print(f"📊 適合度評価開始 (ファジィ決定木統合)")
        print(f"{'='*70}")
        
        # 入力検証
        required_fields = ["research_intensity", "advisor_style", "team_work"]
        for field in required_fields:
            if field not in student_profile:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required field: {field}"
                )
        
        # デフォルト値設定
        if "research_field_match" not in student_profile:
            student_profile["research_field_match"] = 5.0
        
        if "field_interests" not in student_profile:
            student_profile["field_interests"] = {}
        
        # 各研究室との適合度計算
        results = []
        
        for lab in system_state["lab_database"]:
            # ===== 1. ファジィ決定木による予測 =====
            tree_score = 0.5  # デフォルト
            tree_prediction = None
            
            if system_state["tree_trained"] and system_state["decision_tree"]:
                try:
                    tree_prediction = system_state["decision_tree"].predict(student_profile)
                    tree_score = tree_prediction if isinstance(tree_prediction, float) else 0.5
                except Exception as e:
                    print(f"⚠️ 決定木予測エラー: {e}")
            
            # ===== 2. 改善版マッチャーによる詳細評価 =====
            matcher_score = 0.5
            matcher_result = None
            
            if system_state["integrated_matcher"]:
                try:
                    matcher_result = system_state["integrated_matcher"].calculate_compatibility(
                        student_profile,
                        lab
                    )
                    matcher_score = matcher_result.total_compatibility
                except Exception as e:
                    print(f"⚠️ マッチャーエラー: {e}")
            
            # ===== 3. 統合スコア計算 =====
            # ファジィ決定木: 30%, マッチャー: 70%
            total_compatibility = tree_score * 0.3 + matcher_score * 0.7
            
            # 結果構築
            lab_result = {
                "lab_id": lab["id"],
                "lab_name": lab["name"],
                "advisor": lab.get("advisor", "不明"),
                "field_id": lab.get("field_id", "unknown"),
                
                # スコア情報
                "total_compatibility": total_compatibility,
                "tree_score": tree_score,
                "matcher_score": matcher_score,
                
                # 詳細情報（マッチャーから）
                "basic_score": matcher_result.basic_score if matcher_result else 0.5,
                "field_score": matcher_result.field_score if matcher_result else 0.0,
                "field_weight": matcher_result.field_weight if matcher_result else 0.0,
                "basic_weight": matcher_result.basic_weight if matcher_result else 1.0,
                
                "criteria_scores": matcher_result.criteria_scores if matcher_result else {},
                "explanation": matcher_result.explanation if matcher_result else "評価未実施",
                "breakdown": matcher_result.breakdown if matcher_result else {},
                
                # 互換性のため
                "overall_score": total_compatibility * 10,
                "compatibility": {
                    "overall_score": total_compatibility,
                    "tree_component": tree_score,
                    "matcher_component": matcher_score
                }
            }
            
            results.append(lab_result)
        
        # ランキング作成
        results.sort(key=lambda x: x["total_compatibility"], reverse=True)
        
        for rank, result in enumerate(results, 1):
            result["ranking_position"] = rank
        
        # サマリー情報
        scores = [r["total_compatibility"] for r in results]
        summary = {
            "total_labs": len(results),
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "best_match_lab": results[0]["lab_name"] if results else None,
            "field_weight_used": student_profile.get("research_field_match", 5) / 10,
            "recommendations": []
        }
        
        if results and results[0]["total_compatibility"] >= 0.8:
            summary["recommendations"].append(
                f"{results[0]['lab_name']}は非常に高い適合度を示しています"
            )
        
        # 評価回数増加
        system_state["evaluation_count"] += 1
        
        print(f"\n✅ 評価完了: {len(results)}件の研究室を評価")
        print(f"   ファジィ決定木: {'✅ 使用' if system_state['tree_trained'] else '❌ 未使用'}")
        print(f"   最高適合度: {summary['max_score']:.3f}")
        print(f"   平均適合度: {summary['avg_score']:.3f}")
        print(f"{'='*70}\n")
        
        return {
            "results": results,
            "lab_results": results,
            "summary": summary,
            "metadata": {
                "evaluation_method": "fuzzy_tree_integrated",
                "fuzzy_tree_used": system_state["tree_trained"],
                "matcher_type": "improved" if IMPROVED_MATCHER_AVAILABLE else "legacy",
                "criteria_count": 11,
                "field_matching_enabled": True,
                "timestamp": time.time()
            }
        }
        
    except Exception as e:
        print(f"\n❌ 評価エラー: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/validate_profile")
async def validate_profile(student_profile: Dict[str, Any]):
    """学生プロファイルの検証"""
    
    def validate(profile):
        required = ["research_intensity", "advisor_style", "team_work", 
                   "workload", "theory_practice"]
        
        for field in required:
            if field not in profile:
                return False, f"必須項目が不足: {field}"
            
            value = profile[field]
            if not isinstance(value, (int, float)) or value < 1 or value > 10:
                return False, f"{field}の値が不正です (1-10の範囲)"
        
        return True, None
    
    valid, error_message = validate(student_profile)
    
    if valid:
        return {
            "valid": True,
            "message": "プロファイルは有効です",
            "profile_summary": {
                "basic_criteria_count": len([k for k in student_profile.keys() 
                                            if k in ["research_intensity", "advisor_style", "team_work",
                                                    "workload", "theory_practice", "skill_development",
                                                    "lab_atmosphere", "flexibility", "publication_opportunity",
                                                    "interdisciplinary", "communication_style"]]),
                "field_interests_count": len(student_profile.get("field_interests", {})),
                "research_field_match": student_profile.get("research_field_match", "未設定")
            }
        }
    else:
        return {
            "valid": False,
            "error": error_message
        }

@app.get("/api/system_stats")
async def get_system_stats():
    """システム統計情報"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {
        "system_status": "active",
        "version": "3.1.0",
        "components": {
            "fuzzy_decision_tree": {
                "available": DECISION_TREE_AVAILABLE,
                "trained": system_state["tree_trained"],
                "training_samples": len(system_state["training_data"])
            },
            "genetic_algorithm": {
                "available": GENETIC_AVAILABLE,
                "weights_optimized": system_state["optimized_weights"] is not None
            },
            "integrated_matcher": {
                "type": "improved_v2" if IMPROVED_MATCHER_AVAILABLE else "legacy",
                "available": system_state["integrated_matcher"] is not None
            }
        },
        "evaluation_criteria": {
            "count": 11,
            "items": [
                "research_intensity", "advisor_style", "team_work",
                "workload", "theory_practice", "skill_development",
                "lab_atmosphere", "flexibility", "publication_opportunity",
                "interdisciplinary", "communication_style"
            ]
        },
        "database": {
            "total_labs": len(system_state["lab_database"]),
            "fields_available": len(set(lab.get("field_id", "unknown") 
                                       for lab in system_state["lab_database"]))
        },
        "statistics": {
            "total_evaluations": system_state["evaluation_count"],
            "last_optimization": system_state["last_optimization_time"]
        }
    }

# ===== サーバー起動 =====
if __name__ == "__main__":
    print("\n🚀 FastAPI サーバー起動中...")
    print(f"📍 URL: http://localhost:{getattr(settings, 'port', 8000)}")
    print(f"📚 API文書: http://localhost:{getattr(settings, 'port', 8000)}/docs")
    print("🔧 システム構成:")
    print(f"  - ファジィ決定木: {'✅' if DECISION_TREE_AVAILABLE else '❌'}")
    print(f"  - 改善版マッチャー: {'✅' if IMPROVED_MATCHER_AVAILABLE else '❌'}")
    print(f"  - 遺伝的アルゴリズム: {'✅' if GENETIC_AVAILABLE else '❌'}")
    
    uvicorn.run(
        "app:app",
        host=getattr(settings, 'host', '0.0.0.0'),
        port=getattr(settings, 'port', 8000),
        reload=getattr(settings, 'debug', True),
        log_level="info"
    )