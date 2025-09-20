#!/usr/bin/env python3
"""
遺伝的アルゴリズム統合版 研究室選択支援システム
app.py - 重み係数を遺伝的アルゴリズムで最適化
"""

import os
import sys
import time
import json
import random
import math
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# FastAPI関連のインポート
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# NumPy可用性チェック
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("⚠️ NumPy が見つかりません。基本機能のみ使用します。")

# 遺伝的アルゴリズムモジュールをインポート
# （前回作成したGeneticWeightOptimizerクラスを使用）
from genetic_weights_optimizer import GeneticWeightOptimizer, GeneticConfig

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# === FastAPIアプリケーション初期化 ===
app = FastAPI(
    title="研究室選択支援システム - 遺伝的アルゴリズム統合版",
    description="重み係数を遺伝的アルゴリズムで最適化する研究室マッチングシステム",
    version="4.2.0",
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
    app.mount("/static", StaticFiles(directory="../frontend/build/static"), name="static")

# === 評価基準定義（13項目） ===
EVALUATION_CRITERIA = [
    "research_intensity", "advisor_style", "team_work", "workload", "theory_practice",
    "research_field_match", "skill_development", "lab_atmosphere", "flexibility", 
    "publication_opportunity", "interdisciplinary", "communication_style", "innovation_risk"
]

# === 研究分野定義（20分野）===
RESEARCH_FIELDS = [
    "人工知能・機械学習", "画像・映像処理", "ネットワーク・セキュリティ", 
    "データベース・情報システム", "組込み・IoT", "教育・言語学", 
    "自然科学・数理", "医療情報・ヘルスケア", "観光情報・地域システム", 
    "経営情報・意思決定支援", "音声・音響情報処理", "システム運用・情報倫理",
    "Webデザイン・UI/UX", "デザイン・視覚表現", "映像・アニメーション", 
    "コンピュータ音楽・サウンドアート", "ゲーム開発・eスポーツ", "VR/AR・メディアアート",
    "哲学・人文・環境行動学", "スポーツ・体育科学"
]

# === システム状態管理 ===
system_state = {
    "initialized": False,
    "evaluation_count": 0,
    "api_calls": 0,
    "optimization_status": "ready",  # ready, running, completed
    "optimization_progress": 0,
    "current_weights": None,
    "last_updated": None,
    "lab_count": 0
}

# === 重み係数管理 ===
class WeightManager:
    """重み係数管理クラス"""
    
    def __init__(self):
        self.current_weights = self._get_default_weights()
        self.optimization_history = []
        self.is_optimized = False
        
    def _get_default_weights(self) -> Dict[str, float]:
        """デフォルトの重み係数（均等分散）"""
        return {criterion: 1.0/13 for criterion in EVALUATION_CRITERIA}
    
    def update_weights(self, new_weights: Dict[str, float]):
        """重み係数を更新"""
        self.current_weights = new_weights
        self.is_optimized = True
        system_state["current_weights"] = new_weights
        system_state["last_updated"] = datetime.now().isoformat()
    
    def get_weights(self) -> Dict[str, float]:
        """現在の重み係数を取得"""
        return self.current_weights
    
    def reset_to_default(self):
        """デフォルトに戻す"""
        self.current_weights = self._get_default_weights()
        self.is_optimized = False
        system_state["current_weights"] = self.current_weights

# グローバル重み管理インスタンス
weight_manager = WeightManager()

# === 研究室データベース ===
SAMPLE_LABS = []

def load_labs_database():
    """研究室データベース読み込み"""
    global SAMPLE_LABS
    
    database_path = project_root / "data" / "labs_database.json"
    
    try:
        if database_path.exists():
            with open(database_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                SAMPLE_LABS = data.get("labs", [])
                system_state["lab_count"] = len(SAMPLE_LABS)
                print(f"✅ データベース読み込み成功: {len(SAMPLE_LABS)}件の研究室")
        else:
            print("⚠️ データベースファイルが見つかりません。サンプルデータを使用します。")
            SAMPLE_LABS = create_sample_data()
            system_state["lab_count"] = len(SAMPLE_LABS)
    except Exception as e:
        print(f"❌ データベース読み込みエラー: {e}")
        SAMPLE_LABS = create_sample_data()
        system_state["lab_count"] = len(SAMPLE_LABS)

def create_sample_data():
    """サンプル研究室データ生成"""
    return [
        {
            "id": "lab_sample",
            "name": "サンプル研究室",
            "professor": "サンプル教授",
            "research_area": "人工知能・機械学習",
            "specialization": "機械学習、データサイエンス",
            "features": {
                "research_intensity": 8.0,
                "advisor_style": 7.0,
                "team_work": 6.0,
                "workload": 8.0,
                "theory_practice": 7.0,
                "research_field_match": 9.0,
                "skill_development": 8.0,
                "lab_atmosphere": 7.0,
                "flexibility": 6.0,
                "publication_opportunity": 8.0,
                "interdisciplinary": 5.0,
                "communication_style": 6.0,
                "innovation_risk": 7.0
            }
        }
    ]

# === 適合度計算（遺伝的アルゴリズム対応版） ===
def calculate_compatibility_with_genetic_weights(student_profile: Dict, lab: Dict) -> Tuple[float, Dict[str, Any]]:
    """遺伝的アルゴリズム最適化重みによる適合度計算"""
    
    current_weights = weight_manager.get_weights()
    total_score = 0.0
    total_weight = 0.0
    criteria_details = {}
    
    for criterion in EVALUATION_CRITERIA:
        student_val = float(student_profile.get(criterion, 5.0))
        lab_val = float(lab["features"].get(criterion, 5.0))
        
        # 正規化
        if student_val > 1.0:
            student_val /= 10.0
        if lab_val > 1.0:
            lab_val /= 10.0
        
        # 適合度計算
        diff = abs(student_val - lab_val)
        compatibility = 1.0 - diff
        
        # 重み適用（遺伝的アルゴリズムで最適化された重み）
        weight = current_weights.get(criterion, 1.0/13)
        weighted_score = compatibility * weight
        
        total_score += weighted_score
        total_weight += weight
        
        # 詳細情報保存
        criteria_details[criterion] = {
            "student_value": student_val,
            "lab_value": lab_val,
            "compatibility": compatibility,
            "weight": weight,
            "weighted_score": weighted_score,
            "is_genetic_optimized": weight_manager.is_optimized
        }
    
    final_compatibility = total_score / total_weight if total_weight > 0 else 0.0
    
    explanation = {
        "final_compatibility": final_compatibility,
        "is_weights_optimized": weight_manager.is_optimized,
        "optimization_method": "genetic_algorithm" if weight_manager.is_optimized else "default_uniform",
        "criteria_details": criteria_details,
        "weight_distribution": current_weights
    }
    
    return final_compatibility, explanation

# === APIモデル定義 ===
class EvaluationRequest(BaseModel):
    """評価リクエスト"""
    research_intensity: float
    advisor_style: float
    team_work: float
    workload: float
    theory_practice: float
    research_field_match: float
    skill_development: float
    lab_atmosphere: float
    flexibility: float
    publication_opportunity: float
    interdisciplinary: float
    communication_style: float
    innovation_risk: float

class OptimizationRequest(BaseModel):
    """重み最適化リクエスト"""
    population_size: Optional[int] = 30
    generations: Optional[int] = 50
    crossover_rate: Optional[float] = 0.8
    mutation_rate: Optional[float] = 0.1
    use_training_data: Optional[bool] = True

# === 最適化関連のグローバル変数 ===
current_optimization_task = None

# === APIエンドポイント ===

@app.get("/")
async def read_root():
    """ルートエンドポイント"""
    if os.path.exists("../frontend/build/index.html"):
        return FileResponse("../frontend/build/index.html")
    else:
        return {
            "message": "遺伝的アルゴリズム統合版 研究室選択支援システム",
            "version": "4.2.0",
            "status": "running",
            "features": ["遺伝的アルゴリズム重み最適化", "詳細説明", "リアルタイム進化"],
            "endpoints": {
                "health": "/health",
                "labs": "/api/labs", 
                "evaluate": "/api/evaluate",
                "optimize": "/api/optimize",
                "weights": "/api/weights",
                "optimization_status": "/api/optimization/status",
                "docs": "/docs"
            }
        }

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy",
        "version": "4.2.0",
        "timestamp": time.time(),
        "system_initialized": system_state["initialized"],
        "lab_count": len(SAMPLE_LABS),
        "evaluation_count": system_state["evaluation_count"],
        "optimization_status": system_state["optimization_status"],
        "weights_optimized": weight_manager.is_optimized
    }

@app.get("/api/labs")
async def get_labs():
    """研究室一覧取得"""
    return {
        "labs": SAMPLE_LABS,
        "total_count": len(SAMPLE_LABS),
        "categories": get_lab_categories(),
        "source": "genetic_database"
    }

def get_lab_categories():
    """研究室のカテゴリ別統計"""
    categories = {}
    for lab in SAMPLE_LABS:
        area = lab.get("research_area", "その他")
        categories[area] = categories.get(area, 0) + 1
    return categories

@app.get("/api/fields")
async def get_research_fields():
    """研究分野一覧取得"""
    return {
        "fields": RESEARCH_FIELDS,
        "total_count": len(RESEARCH_FIELDS),
        "categories": {
            "テクノロジー・システム": 12,
            "クリエイティブ": 4,
            "エンターテイメント": 2,
            "人文・社会・体育": 2
        }
    }

@app.post("/api/evaluate")
async def evaluate_compatibility(request: EvaluationRequest):
    """研究室適合度評価（遺伝的アルゴリズム重み使用）"""
    
    try:
        student_profile = request.dict()
        results = []
        
        for lab in SAMPLE_LABS:
            # 遺伝的アルゴリズム最適化重みで適合度計算
            compatibility, explanation = calculate_compatibility_with_genetic_weights(student_profile, lab)
            
            results.append({
                "lab": lab,
                "compatibility": compatibility,
                "explanation": explanation
            })
        
        # 適合度でソート
        results.sort(key=lambda x: x["compatibility"], reverse=True)
        
        system_state["evaluation_count"] += 1
        system_state["api_calls"] += 1
        
        return {
            "results": results,
            "total_evaluated": len(results),
            "evaluation_summary": {
                "best_match": results[0]["lab"]["name"] if results else None,
                "best_score": results[0]["compatibility"] if results else 0,
                "evaluation_time": datetime.now().isoformat(),
                "criteria_used": len(EVALUATION_CRITERIA),
                "weights_method": "genetic_algorithm" if weight_manager.is_optimized else "default_uniform"
            },
            "weight_info": {
                "is_optimized": weight_manager.is_optimized,
                "current_weights": weight_manager.get_weights()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"評価エラー: {str(e)}")

@app.post("/api/optimize")
async def optimize_weights(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """重み係数の遺伝的アルゴリズム最適化"""
    
    global current_optimization_task
    
    if system_state["optimization_status"] == "running":
        raise HTTPException(status_code=400, detail="最適化が既に実行中です")
    
    # 最適化タスクをバックグラウンドで実行
    background_tasks.add_task(run_optimization, request)
    
    system_state["optimization_status"] = "running"
    system_state["optimization_progress"] = 0
    
    return {
        "message": "重み最適化を開始しました",
        "config": request.dict(),
        "estimated_time": f"{request.generations * request.population_size * 0.001:.1f}秒",
        "status_endpoint": "/api/optimization/status"
    }

async def run_optimization(request: OptimizationRequest):
    """最適化実行（バックグラウンドタスク）"""
    
    try:
        # 設定
        config = GeneticConfig(
            population_size=request.population_size,
            generations=request.generations,
            crossover_rate=request.crossover_rate,
            mutation_rate=request.mutation_rate
        )
        
        # 最適化実行
        optimizer = GeneticWeightOptimizer(config)
        
        # 進捗更新用のコールバックを設定（簡単な実装）
        def update_progress(generation: int):
            progress = (generation / request.generations) * 100
            system_state["optimization_progress"] = progress
        
        # 最適化実行
        results = optimizer.evolve()
        optimized_weights = optimizer.get_optimized_weights()
        
        # 重み更新
        weight_manager.update_weights(optimized_weights)
        
        # 状態更新
        system_state["optimization_status"] = "completed"
        system_state["optimization_progress"] = 100
        
        print("🎯 重み最適化完了！")
        print(f"最良適応度: {optimizer.best_individual.fitness:.4f}")
        
    except Exception as e:
        print(f"❌ 最適化エラー: {e}")
        system_state["optimization_status"] = "error"
        system_state["optimization_progress"] = 0

@app.get("/api/optimization/status")
async def get_optimization_status():
    """最適化状態取得"""
    return {
        "status": system_state["optimization_status"],
        "progress": system_state["optimization_progress"],
        "weights_optimized": weight_manager.is_optimized,
        "current_weights": weight_manager.get_weights() if weight_manager.is_optimized else None,
        "last_updated": system_state.get("last_updated")
    }

@app.get("/api/weights")
async def get_current_weights():
    """現在の重み係数取得"""
    return {
        "weights": weight_manager.get_weights(),
        "is_optimized": weight_manager.is_optimized,
        "optimization_method": "genetic_algorithm" if weight_manager.is_optimized else "default_uniform",
        "criteria_count": len(EVALUATION_CRITERIA)
    }

@app.post("/api/weights/reset")
async def reset_weights():
    """重み係数をデフォルトにリセット"""
    weight_manager.reset_to_default()
    system_state["optimization_status"] = "ready"
    system_state["optimization_progress"] = 0
    
    return {
        "message": "重み係数をデフォルトにリセットしました",
        "weights": weight_manager.get_weights()
    }

@app.get("/api/system")
async def get_system_info():
    """システム情報取得"""
    return {
        "system_state": system_state,
        "sample_labs_count": len(SAMPLE_LABS),
        "criteria_count": len(EVALUATION_CRITERIA),
        "research_fields_count": len(RESEARCH_FIELDS),
        "current_weights": weight_manager.get_weights(),
        "version": "4.2.0 - 遺伝的アルゴリズム統合版",
        "features": [
            "遺伝的アルゴリズム重み最適化",
            "リアルタイム進化プロセス",
            "詳細な結果説明",
            "適応度履歴追跡",
            "13項目高精度評価"
        ]
    }

# === システム初期化 ===
def initialize_system():
    """システム初期化"""
    try:
        print("🔧 システム初期化中...")
        load_labs_database()
        system_state["initialized"] = True
        system_state["last_updated"] = datetime.now()
        print(f"✅ システム初期化完了 - 研究室数: {len(SAMPLE_LABS)}件")
    except Exception as e:
        print(f"❌ システム初期化エラー: {e}")
        system_state["initialized"] = False

# システム初期化実行
initialize_system()

# === サーバー起動 ===
def start_server(host: str = "0.0.0.0", port: int = 8000) -> bool:
    """サーバー起動"""
    
    print("\n" + "=" * 80)
    print("🧬🌳 遺伝的アルゴリズム × ファジィ決定木 研究室マッチングシステム v4.2.0")
    print("【遺伝的アルゴリズム統合版】")
    print("=" * 80)
    print(f"🚀 サーバー起動中...")
    print(f"📍 URL: http://localhost:{port}")
    print(f"📚 API文書: http://localhost:{port}/docs")
    print(f"🔧 システム状況:")
    print(f"   - FastAPI: ✅")
    print(f"   - NumPy: {'✅' if HAS_NUMPY else '❌ (オプション)'}")
    print(f"   - 研究室データ: {len(SAMPLE_LABS)}件")
    print(f"   - 評価基準: {len(EVALUATION_CRITERIA)}項目")
    print(f"   - 研究分野: {len(RESEARCH_FIELDS)}分野")
    print(f"   - 遺伝的アルゴリズム: ✅ 統合済み")
    print(f"   - 重み最適化: ✅ 利用可能")
    print("=" * 80)
    print("\n🌟 新機能:")
    print("   - 重み係数の自動最適化")
    print("   - リアルタイム進化プロセス表示")
    print("   - 適応度履歴の追跡")
    print("   - カスタム最適化設定")
    print("=" * 80)
    
    return True

if __name__ == "__main__":
    start_server()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )