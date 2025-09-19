#!/usr/bin/env python3
"""
遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム
接続問題修正版 FastAPI メインアプリケーション - app.py
"""

import os
import sys
import time
import json
import math
import random
import traceback
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# 基本ライブラリ（エラーハンドリング付き）
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("⚠️ numpy が利用できません。基本機能で代替します。")

# FastAPI関連（エラーハンドリング付き）
try:
    from fastapi import FastAPI, HTTPException, Request, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    HAS_FASTAPI = True
    print("✅ FastAPI モジュール正常にロード")
except ImportError as e:
    print(f"❌ FastAPI インポートエラー: {e}")
    print("💡 解決方法: pip install fastapi uvicorn")
    HAS_FASTAPI = False
    sys.exit(1)

# プロジェクトルート設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# システム状態管理（強化版）
system_state = {
    "initialized": False,
    "lab_data": [],
    "evaluation_count": 0,
    "last_updated": None,
    "database_version": "3.0.0",
    "server_start_time": datetime.now(),
    "api_calls": 0,
    "error_count": 0,
    "startup_errors": []
}

# 完全な18分野対応の研究分野データ
RESEARCH_FIELDS_DATA = [
    # テクノロジー・システム分野（12分野）
    {"id": "ai_ml", "name": "人工知能・機械学習", "category": "テクノロジー・システム", "faculty_count": 7},
    {"id": "image_processing", "name": "画像・映像処理", "category": "テクノロジー・システム", "faculty_count": 6},
    {"id": "network_security", "name": "ネットワーク・セキュリティ", "category": "テクノロジー・システム", "faculty_count": 3},
    {"id": "database_systems", "name": "データベース・情報システム", "category": "テクノロジー・システム", "faculty_count": 3},
    {"id": "embedded_iot", "name": "組込み・IoT", "category": "テクノロジー・システム", "faculty_count": 2},
    {"id": "education_linguistics", "name": "教育・言語学", "category": "テクノロジー・システム", "faculty_count": 5},
    {"id": "natural_science_math", "name": "自然科学・数理", "category": "テクノロジー・システム", "faculty_count": 6},
    {"id": "medical_healthcare", "name": "医療情報・ヘルスケア", "category": "テクノロジー・システム", "faculty_count": 2},
    {"id": "tourism_regional", "name": "観光情報・地域システム", "category": "テクノロジー・システム", "faculty_count": 2},
    {"id": "business_decision", "name": "経営情報・意思決定支援", "category": "テクノロジー・システム", "faculty_count": 3},
    {"id": "audio_processing", "name": "音声・音響情報処理", "category": "テクノロジー・システム", "faculty_count": 2},
    {"id": "system_ethics", "name": "システム運用・情報倫理", "category": "テクノロジー・システム", "faculty_count": 3},
    
    # クリエイティブ分野（4分野）
    {"id": "web_ui_ux", "name": "Webデザイン・UI/UX", "category": "クリエイティブ", "faculty_count": 4},
    {"id": "design_visual", "name": "デザイン・視覚表現", "category": "クリエイティブ", "faculty_count": 4},
    {"id": "video_animation", "name": "映像・アニメーション", "category": "クリエイティブ", "faculty_count": 2},
    {"id": "computer_music", "name": "コンピュータ音楽・サウンドアート", "category": "クリエイティブ", "faculty_count": 2},
    
    # エンターテイメント分野（2分野）
    {"id": "game_esports", "name": "ゲーム開発・eスポーツ", "category": "エンターテイメント", "faculty_count": 2},
    {"id": "vr_ar_media", "name": "VR/AR・メディアアート", "category": "エンターテイメント", "faculty_count": 2},
    
    # 人文・社会・体育分野（2分野）
    {"id": "philosophy_humanities", "name": "哲学・人文・環境行動学", "category": "人文・社会・体育", "faculty_count": 2},
    {"id": "sports_science", "name": "スポーツ・体育科学", "category": "人文・社会・体育", "faculty_count": 2},
]

# 完全な13項目評価基準
COMPLETE_EVALUATION_CRITERIA = [
    # 基本項目（5項目）
    "research_intensity", "advisor_style", "team_work", "workload", "theory_practice",
    # 拡張項目（5項目）
    "research_field_match", "skill_development", "lab_atmosphere", "flexibility", "publication_opportunity",
    # 特殊項目（3項目）
    "interdisciplinary", "communication_style", "innovation_risk"
]

# 基準別デフォルト重み（13項目完全対応）
DEFAULT_CRITERIA_WEIGHTS = {
    # 基本項目：標準〜高重み
    "research_intensity": 1.2,
    "advisor_style": 1.1,
    "team_work": 1.0,
    "workload": 1.0,
    "theory_practice": 1.1,
    
    # 拡張項目：中〜高重み
    "research_field_match": 1.4,  # 最重要
    "skill_development": 0.9,
    "lab_atmosphere": 0.8,
    "flexibility": 0.8,
    "publication_opportunity": 1.0,
    
    # 特殊項目：調整重み
    "interdisciplinary": 0.7,
    "communication_style": 0.8,
    "innovation_risk": 0.9
}

# Pydantic モデル定義
class StudentProfile(BaseModel):
    """学生プロフィール（13項目完全対応）"""
    # 基本項目
    research_intensity: float = Field(ge=1, le=10, description="研究強度 (1-10)")
    advisor_style: float = Field(ge=1, le=10, description="指導スタイル (1-10)")
    team_work: float = Field(ge=1, le=10, description="チームワーク (1-10)")
    workload: float = Field(ge=1, le=10, description="ワークロード (1-10)")
    theory_practice: float = Field(ge=1, le=10, description="理論・実践バランス (1-10)")
    
    # 拡張項目
    research_field_match: float = Field(ge=1, le=10, description="研究分野適合性 (1-10)")
    skill_development: float = Field(ge=1, le=10, description="スキル開発 (1-10)")
    lab_atmosphere: float = Field(ge=1, le=10, description="研究室雰囲気 (1-10)")
    flexibility: float = Field(ge=1, le=10, description="柔軟性 (1-10)")
    publication_opportunity: float = Field(ge=1, le=10, description="論文発表機会 (1-10)")
    
    # 特殊項目
    interdisciplinary: float = Field(ge=1, le=10, description="学際性 (1-10)")
    communication_style: float = Field(ge=1, le=10, description="コミュニケーション (1-10)")
    innovation_risk: float = Field(ge=1, le=10, description="革新性・リスク許容度 (1-10)")
    
    # メタデータ
    student_id: Optional[str] = None
    timestamp: Optional[str] = None
    preferred_fields: List[str] = []

class LabData(BaseModel):
    """研究室データ（13項目完全対応）"""
    id: str
    name: str
    advisor: str
    description: str
    field_category: str
    research_fields: List[str]
    
    # 13項目の研究室特性値
    research_intensity: float = Field(ge=1, le=10)
    advisor_style: float = Field(ge=1, le=10)
    team_work: float = Field(ge=1, le=10)
    workload: float = Field(ge=1, le=10)
    theory_practice: float = Field(ge=1, le=10)
    research_field_match: float = Field(ge=1, le=10)
    skill_development: float = Field(ge=1, le=10)
    lab_atmosphere: float = Field(ge=1, le=10)
    flexibility: float = Field(ge=1, le=10)
    publication_opportunity: float = Field(ge=1, le=10)
    interdisciplinary: float = Field(ge=1, le=10)
    communication_style: float = Field(ge=1, le=10)
    innovation_risk: float = Field(ge=1, le=10)
    
    # メタデータ
    faculty_count: int = 1
    graduate_employment: Optional[str] = None
    equipment: Optional[str] = None
    funding_level: Optional[str] = None

class MatchingResult(BaseModel):
    """マッチング結果"""
    lab_id: str
    lab_name: str
    compatibility_score: float
    detailed_scores: Dict[str, float]
    explanation: str
    recommendation_level: str
    field_match: bool
    timestamp: str

class OptimizationRequest(BaseModel):
    """最適化リクエスト"""
    student_profile: StudentProfile
    population_size: int = Field(default=30, ge=10, le=100)
    generations: int = Field(default=20, ge=5, le=50)
    mutation_rate: float = Field(default=0.1, ge=0.01, le=0.5)
    crossover_rate: float = Field(default=0.8, ge=0.1, le=1.0)
    custom_weights: Optional[Dict[str, float]] = None

# 拡張された研究室データ（18分野対応・13項目完全対応）
EXTENDED_LAB_DATA = [
    # AI・機械学習分野
    {
        "id": "ai_lab_01", "name": "知的システム研究室", "advisor": "伊藤雅彦教授", 
        "description": "情報可視化、ユーザインタフェース、データ工学の研究",
        "field_category": "テクノロジー・システム", "research_fields": ["人工知能・機械学習"],
        # 13項目の基準値（1-10スケール）
        "research_intensity": 8.5, "advisor_style": 7.0, "team_work": 8.0, "workload": 8.0, "theory_practice": 6.0,
        "research_field_match": 9.0, "skill_development": 8.5, "lab_atmosphere": 7.5, "flexibility": 6.5, 
        "publication_opportunity": 8.0, "interdisciplinary": 7.0, "communication_style": 8.0, "innovation_risk": 8.5,
        "faculty_count": 1, "graduate_employment": "大手IT企業、研究機関", "equipment": "高性能GPU、可視化システム"
    },
    {
        "id": "ai_lab_02", "name": "機械学習・データ解析研究室", "advisor": "内山敏雄教授", 
        "description": "データ解析、機械学習、レコメンド、テキストマイニング",
        "field_category": "テクノロジー・システム", "research_fields": ["人工知能・機械学習"],
        # 13項目の基準値
        "research_intensity": 9.0, "advisor_style": 6.5, "team_work": 7.5, "workload": 8.5, "theory_practice": 7.0,
        "research_field_match": 9.5, "skill_development": 9.0, "lab_atmosphere": 8.0, "flexibility": 7.0, 
        "publication_opportunity": 8.5, "interdisciplinary": 8.0, "communication_style": 7.5, "innovation_risk": 8.0,
        "faculty_count": 1, "graduate_employment": "データサイエンティスト、AI研究者", "equipment": "機械学習クラスタ、大容量ストレージ"
    },
    
    # 画像・映像処理分野
    {
        "id": "image_lab_01", "name": "コンピュータビジョン研究室", "advisor": "藤原孝行教授", 
        "description": "コンピュータビジョン、コンピュータグラフィックス",
        "field_category": "テクノロジー・システム", "research_fields": ["画像・映像処理"],
        # 13項目の基準値
        "research_intensity": 8.0, "advisor_style": 7.5, "team_work": 7.0, "workload": 7.5, "theory_practice": 8.0,
        "research_field_match": 8.5, "skill_development": 8.0, "lab_atmosphere": 7.0, "flexibility": 7.5, 
        "publication_opportunity": 7.5, "interdisciplinary": 6.5, "communication_style": 7.0, "innovation_risk": 8.0,
        "faculty_count": 1, "graduate_employment": "映像制作会社、VR/AR企業", "equipment": "高解像度カメラ、画像処理サーバー"
    },
    
    # 医療情報・ヘルスケア分野
    {
        "id": "medical_lab_01", "name": "医用画像工学研究室", "advisor": "越野一博教授", 
        "description": "医用画像工学、数理統計学、人工知能画像解析処理",
        "field_category": "テクノロジー・システム", "research_fields": ["医療情報・ヘルスケア", "画像・映像処理"],
        # 13項目の基準値
        "research_intensity": 8.5, "advisor_style": 6.0, "team_work": 6.5, "workload": 8.0, "theory_practice": 7.5,
        "research_field_match": 9.0, "skill_development": 8.5, "lab_atmosphere": 6.0, "flexibility": 6.0, 
        "publication_opportunity": 8.5, "interdisciplinary": 9.5, "communication_style": 6.5, "innovation_risk": 8.5,
        "faculty_count": 1, "graduate_employment": "医療機器メーカー、病院情報システム部門", "equipment": "医用画像解析システム、統計解析ソフト"
    },
    
    # Webデザイン・UI/UX分野
    {
        "id": "design_lab_01", "name": "UXデザイン研究室", "advisor": "安田光孝教授", 
        "description": "UX/UIデザイン、コンテンツプロデュース、デザイン思考",
        "field_category": "クリエイティブ", "research_fields": ["Webデザイン・UI/UX"],
        # 13項目の基準値
        "research_intensity": 7.0, "advisor_style": 8.5, "team_work": 9.0, "workload": 7.0, "theory_practice": 8.5,
        "research_field_match": 8.0, "skill_development": 9.0, "lab_atmosphere": 9.5, "flexibility": 9.0, 
        "publication_opportunity": 6.5, "interdisciplinary": 8.5, "communication_style": 9.5, "innovation_risk": 8.0,
        "faculty_count": 1, "graduate_employment": "デザイン会社、スタートアップ", "equipment": "デザインワークステーション、プロトタイピングツール"
    },
    
    # ゲーム開発・eスポーツ分野
    {
        "id": "game_lab_01", "name": "ゲーム・eスポーツ研究室", "advisor": "川原勝教授", 
        "description": "eスポーツ、メタバース、教育学",
        "field_category": "エンターテイメント", "research_fields": ["ゲーム開発・eスポーツ"],
        # 13項目の基準値
        "research_intensity": 7.5, "advisor_style": 8.0, "team_work": 8.5, "workload": 7.5, "theory_practice": 8.0,
        "research_field_match": 8.5, "skill_development": 8.0, "lab_atmosphere": 9.0, "flexibility": 8.5, 
        "publication_opportunity": 7.0, "interdisciplinary": 8.0, "communication_style": 9.0, "innovation_risk": 9.0,
        "faculty_count": 1, "graduate_employment": "ゲーム会社、eスポーツ関連企業", "equipment": "ゲーミングPC、VRセットアップ"
    },
    
    # 哲学・人文・環境行動学分野
    {
        "id": "humanities_lab_01", "name": "環境行動学研究室", "advisor": "波田彰教授", 
        "description": "環境行動学、地域コミュニティ、建築計画学、環境認知",
        "field_category": "人文・社会・体育", "research_fields": ["哲学・人文・環境行動学"],
        # 13項目の基準値
        "research_intensity": 6.5, "advisor_style": 7.5, "team_work": 7.0, "workload": 6.0, "theory_practice": 5.5,
        "research_field_match": 8.0, "skill_development": 7.0, "lab_atmosphere": 8.0, "flexibility": 8.0, 
        "publication_opportunity": 7.0, "interdisciplinary": 9.0, "communication_style": 8.5, "innovation_risk": 7.0,
        "faculty_count": 1, "graduate_employment": "都市計画関連、建築事務所", "equipment": "フィールドワーク機材、測定機器"
    }
]

# FastAPIアプリケーション初期化（シンプル版）
app = FastAPI(
    title="研究室選択支援システム v3.0 (完全18分野・13項目対応版)",
    description="遺伝的アルゴリズムを用いたファジィ決定木による研究室マッチングシステム - 完全対応版",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# システム初期化（起動時実行）
def initialize_system():
    """システム初期化"""
    try:
        print("\n🚀 研究室選択支援システム v3.0 起動中...")
        
        # システム状態初期化
        system_state["lab_data"] = EXTENDED_LAB_DATA
        system_state["initialized"] = True
        system_state["last_updated"] = datetime.now().isoformat()
        
        print(f"✅ システム初期化完了")
        print(f"📊 研究室データ: {len(EXTENDED_LAB_DATA)}件")
        print(f"🔧 評価基準: {len(COMPLETE_EVALUATION_CRITERIA)}項目")
        print(f"🏛️ 研究分野: {len(RESEARCH_FIELDS_DATA)}分野")
        
        return True
        
    except Exception as e:
        print(f"❌ システム初期化エラー: {e}")
        traceback.print_exc()
        system_state["initialized"] = False
        return False

# CORS設定（強化版）
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8080", 
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "http://localhost:5173",  # Vite
        "http://127.0.0.1:5173",
        "*"  # 開発用：本番では削除
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# グローバル例外ハンドラー
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """グローバル例外ハンドラー"""
    system_state["error_count"] += 1
    
    error_detail = {
        "error": str(exc),
        "type": type(exc).__name__,
        "path": str(request.url),
        "method": request.method,
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"❌ API エラー: {error_detail}")
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "内部サーバーエラーが発生しました",
            "error_info": error_detail,
            "suggestion": "サーバーログを確認してください"
        }
    )

# ミドルウェア：リクエスト追跡（改善版）
@app.middleware("http")
async def enhanced_request_middleware(request: Request, call_next):
    """強化されたリクエスト追跡ミドルウェア"""
    start_time = time.time()
    system_state["api_calls"] += 1
    
    # リクエスト情報のログ
    print(f"📥 {request.method} {request.url.path} - 開始")
    
    try:
        response = await call_next(request)
        
        # 処理時間計算
        process_time = time.time() - start_time
        
        # レスポンスヘッダー追加
        response.headers["X-Process-Time"] = f"{process_time:.3f}"
        response.headers["X-System-Status"] = "operational" if system_state["initialized"] else "initializing"
        response.headers["X-API-Version"] = "3.0.0"
        response.headers["Access-Control-Expose-Headers"] = "X-Process-Time,X-System-Status,X-API-Version"
        
        print(f"📤 {request.method} {request.url.path} - 完了 ({process_time:.3f}s)")
        
        return response
        
    except Exception as e:
        system_state["error_count"] += 1
        process_time = time.time() - start_time
        
        print(f"💥 {request.method} {request.url.path} - エラー ({process_time:.3f}s): {e}")
        
        # エラーレスポンス
        return JSONResponse(
            status_code=500,
            content={
                "detail": f"リクエスト処理でエラーが発生しました: {str(e)}",
                "path": str(request.url.path),
                "method": request.method,
                "process_time": process_time
            }
        )

# ===== 核心アルゴリズム実装 =====

def calculate_fuzzy_membership(value: float, low: float, medium: float, high: float) -> Dict[str, float]:
    """ファジィメンバーシップ関数計算"""
    membership = {"low": 0.0, "medium": 0.0, "high": 0.0}
    
    # 三角形メンバーシップ関数
    if value <= low:
        membership["low"] = 1.0
    elif value <= medium:
        membership["low"] = (medium - value) / (medium - low)
        membership["medium"] = (value - low) / (medium - low)
    elif value <= high:
        membership["medium"] = (high - value) / (high - medium)
        membership["high"] = (value - medium) / (high - medium)
    else:
        membership["high"] = 1.0
    
    return membership

def genetic_algorithm_optimization(
    student_profile: Dict[str, float], 
    lab_data: List[Dict], 
    population_size: int = 30, 
    generations: int = 20,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.8,
    custom_weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """遺伝的アルゴリズムによる重み最適化"""
    
    print(f"🧬 遺伝的アルゴリズム開始: 集団={population_size}, 世代={generations}")
    
    # 重みの初期化
    weights = custom_weights if custom_weights else DEFAULT_CRITERIA_WEIGHTS.copy()
    
    def evaluate_fitness(weights_vector: List[float]) -> float:
        """適応度関数"""
        temp_weights = {
            criterion: weights_vector[i] 
            for i, criterion in enumerate(COMPLETE_EVALUATION_CRITERIA)
        }
        
        total_fitness = 0.0
        sample_labs = lab_data[:min(5, len(lab_data))]  # サンプルの研究室で評価
        
        for lab in sample_labs:
            try:
                compatibility = calculate_enhanced_compatibility(student_profile, lab, temp_weights)
                total_fitness += compatibility["overall_score"]
            except Exception as e:
                print(f"⚠️ 適応度評価エラー: {e}")
                continue
        
        return total_fitness / len(sample_labs) if sample_labs else 0.0
    
    # 単純な遺伝的アルゴリズム実装
    initial_weights = list(weights.values())
    best_fitness = evaluate_fitness(initial_weights)
    best_weights = initial_weights.copy()
    
    print(f"📊 初期適応度: {best_fitness:.3f}")
    
    for generation in range(generations):
        # ランダム変異
        mutated_weights = [
            max(0.1, min(2.0, w + (random.random() - 0.5) * mutation_rate))
            for w in best_weights
        ]
        
        fitness = evaluate_fitness(mutated_weights)
        if fitness > best_fitness:
            best_fitness = fitness
            best_weights = mutated_weights
            print(f"📈 世代{generation}: 改善 適応度={best_fitness:.3f}")
    
    # 最適化された重みを辞書に変換
    optimized_weights = {
        criterion: best_weights[i]
        for i, criterion in enumerate(COMPLETE_EVALUATION_CRITERIA)
    }
    
    initial_fitness = evaluate_fitness(initial_weights)
    improvement = best_fitness > initial_fitness
    
    print(f"✅ 最適化完了: 初期={initial_fitness:.3f} -> 最終={best_fitness:.3f}, 改善={improvement}")
    
    return {
        "optimized_weights": optimized_weights,
        "fitness_score": best_fitness,
        "initial_fitness": initial_fitness,
        "generations": generations,
        "improvement": improvement,
        "improvement_percentage": ((best_fitness - initial_fitness) / max(initial_fitness, 0.001)) * 100
    }

    return {
        "optimized_weights": optimized_weights,
        "fitness_score": best_fitness,
        "initial_fitness": initial_fitness,
        "generations": generations,
        "improvement": improvement,
        "improvement_percentage": ((best_fitness - initial_fitness) / max(initial_fitness, 0.001)) * 100
    }

def generate_detailed_explanation(compatibility_result: Dict[str, Any]) -> str:
    """詳細な説明生成（修正版）"""
    
    try:
        score = compatibility_result.get("overall_score", 0.0)
        recommendation = compatibility_result.get("recommendation_level", "不明")
        criteria_count = compatibility_result.get("total_criteria_evaluated", 0)
        
        explanation_parts = [
            f"総合適合度: {score:.1%} ({recommendation})"
        ]
        
        if criteria_count > 0:
            explanation_parts.append(f"評価基準数: {criteria_count}/13項目")
        
        # 高スコア基準の特定
        criteria_scores = compatibility_result.get("criteria_scores", {})
        if criteria_scores:
            high_score_criteria = [
                criterion for criterion, data in criteria_scores.items()
                if isinstance(data, dict) and data.get("weighted_score", 0) > 0.8
            ]
            
            if high_score_criteria:
                explanation_parts.append(f"特に適合: {', '.join(high_score_criteria[:2])}")
        
        # 分野マッチ
        if compatibility_result.get("field_match"):
            explanation_parts.append("研究分野が一致")
        
        # データ完全性
        completeness = compatibility_result.get("data_completeness", 0)
        if completeness < 1.0:
            explanation_parts.append(f"データ完全性: {completeness:.1%}")
        
        return "。".join(explanation_parts) + "。"
        
    except Exception as e:
        print(f"⚠️ 説明生成エラー: {e}")
        return f"適合度: {compatibility_result.get('overall_score', 0):.1%}"
    """詳細な説明生成（修正版）"""
    
    try:
        score = compatibility_result.get("overall_score", 0.0)
        recommendation = compatibility_result.get("recommendation_level", "不明")
        criteria_count = compatibility_result.get("total_criteria_evaluated", 0)
        
        explanation_parts = [
            f"総合適合度: {score:.1%} ({recommendation})"
        ]
        
        if criteria_count > 0:
            explanation_parts.append(f"評価基準数: {criteria_count}/13項目")
        
        # 高スコア基準の特定
        criteria_scores = compatibility_result.get("criteria_scores", {})
        if criteria_scores:
            high_score_criteria = [
                criterion for criterion, data in criteria_scores.items()
                if isinstance(data, dict) and data.get("weighted_score", 0) > 0.8
            ]
            
            if high_score_criteria:
                explanation_parts.append(f"特に適合: {', '.join(high_score_criteria[:2])}")
        
        # 分野マッチ
        if compatibility_result.get("field_match"):
            explanation_parts.append("研究分野が一致")
        
        # データ完全性
        completeness = compatibility_result.get("data_completeness", 0)
        if completeness < 1.0:
            explanation_parts.append(f"データ完全性: {completeness:.1%}")
        
        return "。".join(explanation_parts) + "。"
        
    except Exception as e:
        print(f"⚠️ 説明生成エラー: {e}")
        return f"適合度: {compatibility_result.get('overall_score', 0):.1%}"

def calculate_enhanced_compatibility(
    student_profile: Dict[str, float], 
    lab_data: Dict[str, Any],
    custom_weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """13項目完全対応の適合度計算（修正版）"""
    
    print(f"🧮 適合度計算開始: 研究室={lab_data.get('name', 'Unknown')}")
    
    weights = custom_weights if custom_weights else DEFAULT_CRITERIA_WEIGHTS
    
    # 各基準の適合度計算
    criteria_scores = {}
    total_weighted_score = 0.0
    total_weights = 0.0
    successful_calculations = 0
    
    for criterion in COMPLETE_EVALUATION_CRITERIA:
        try:
            if criterion in student_profile:
                student_value = float(student_profile[criterion])
                
                # 研究室側の値を取得（デフォルト値による補完）
                lab_value = float(lab_data.get(criterion, 5.5))  # 中間値をデフォルト
                
                # 類似度計算（差分の逆数ベース）
                max_diff = 9.0  # 最大差分（1-10の範囲）
                diff = abs(student_value - lab_value)
                
                # 類似度スコア計算（0-1の範囲）
                if diff == 0:
                    similarity_score = 1.0  # 完全一致
                else:
                    # 差分が小さいほど高スコア
                    similarity_score = max(0.0, 1.0 - (diff / max_diff))
                
                # 重み適用
                weight = weights.get(criterion, 1.0)
                weighted_score = similarity_score * weight
                
                criteria_scores[criterion] = {
                    "student_value": student_value,
                    "lab_value": lab_value,
                    "difference": diff,
                    "similarity_score": similarity_score,
                    "weight": weight,
                    "weighted_score": weighted_score
                }
                
                total_weighted_score += weighted_score
                total_weights += weight
                successful_calculations += 1
                
                print(f"  ✅ {criterion}: 学生={student_value}, 研究室={lab_value}, 類似度={similarity_score:.3f}, 重み適用後={weighted_score:.3f}")
            else:
                print(f"  ⚠️ {criterion}: 学生プロフィールに不足")
        except Exception as e:
            print(f"  ❌ {criterion}: 計算エラー - {e}")
            continue
    
    # 全体スコア計算
    if total_weights > 0 and successful_calculations > 0:
        overall_score = total_weighted_score / total_weights
    else:
        print(f"  ⚠️ 重み合計が0またはに成功計算なし: total_weights={total_weights}, successful={successful_calculations}")
        overall_score = 0.0
    
    # 研究分野マッチボーナス
    field_match = False
    field_bonus = 0.0
    
    if hasattr(student_profile, 'preferred_fields') and student_profile.preferred_fields:
        lab_fields = lab_data.get("research_fields", [])
        for preferred_field in student_profile.preferred_fields:
            if preferred_field in lab_fields:
                field_match = True
                field_bonus = 0.1  # 10%のボーナス
                break
    
    # 最終スコア計算
    final_score = min(1.0, overall_score + field_bonus)
    
    # 推薦レベル決定
    if final_score >= 0.8:
        recommendation_level = "強く推薦"
    elif final_score >= 0.65:
        recommendation_level = "推薦"
    elif final_score >= 0.5:
        recommendation_level = "要検討"
    else:
        recommendation_level = "不適合"
    
    print(f"📊 計算結果: 基本適合度={overall_score:.3f}, 分野ボーナス={field_bonus:.3f}, 最終スコア={final_score:.3f}")
    
    return {
        "lab_id": lab_data.get("id"),
        "lab_name": lab_data.get("name"),
        "overall_score": final_score,
        "base_score": overall_score,
        "field_bonus": field_bonus,
        "criteria_scores": criteria_scores,
        "field_match": field_match,
        "recommendation_level": recommendation_level,
        "total_criteria_evaluated": successful_calculations,
        "data_completeness": successful_calculations / len(COMPLETE_EVALUATION_CRITERIA),
        "calculation_details": {
            "total_weighted_score": total_weighted_score,
            "total_weights": total_weights,
            "successful_calculations": successful_calculations
        }
    }

def generate_detailed_explanation(compatibility_result: Dict[str, Any]) -> str:
    """詳細な説明生成"""
    
    score = compatibility_result["overall_score"]
    criteria_scores = compatibility_result["criteria_scores"]
    
    explanation_parts = [
        f"総合適合度: {score:.1%} ({compatibility_result['recommendation_level']})"
    ]
    
    # 高スコア基準
    high_score_criteria = [
        criterion for criterion, data in criteria_scores.items()
        if data["weighted_score"] > 0.8
    ]
    
    if high_score_criteria:
        explanation_parts.append(
            f"特に適合性が高い項目: {', '.join(high_score_criteria[:3])}"
        )
    
    # 改善提案
    low_score_criteria = [
        criterion for criterion, data in criteria_scores.items()
        if data["weighted_score"] < 0.4
    ]
    
    if low_score_criteria:
        explanation_parts.append(
            f"検討が必要な項目: {', '.join(low_score_criteria[:2])}"
        )
    
    return "。".join(explanation_parts) + "。"

# ===== デバッグ・診断エンドポイント =====

# ===== デバッグ・診断エンドポイント =====

@app.get("/api/debug/endpoints", response_class=JSONResponse)
async def list_endpoints():
    """利用可能なAPIエンドポイント一覧"""
    endpoints = [
        # システム基本エンドポイント
        {"path": "/", "method": "GET", "description": "システム情報", "category": "system"},
        {"path": "/health", "method": "GET", "description": "ヘルスチェック", "category": "system"},
        
        # フロントエンド互換エンドポイント（優先度高）
        {"path": "/api/evaluate", "method": "POST", "description": "研究室マッチング（互換）", "category": "frontend", "priority": "high"},
        {"path": "/api/optimize", "method": "POST", "description": "重み最適化（互換）", "category": "frontend", "priority": "high"},
        {"path": "/api/labs", "method": "GET", "description": "研究室一覧（互換）", "category": "frontend", "priority": "high"},
        {"path": "/api/fields", "method": "GET", "description": "研究分野一覧（互換）", "category": "frontend", "priority": "high"},
        {"path": "/api/status", "method": "GET", "description": "システム状態（互換）", "category": "frontend", "priority": "high"},
        {"path": "/api/evaluation-criteria", "method": "GET", "description": "評価基準（互換）", "category": "frontend", "priority": "high"},
        {"path": "/api/categories", "method": "GET", "description": "研究分野カテゴリ", "category": "frontend"},
        {"path": "/api/labs/{lab_id}", "method": "GET", "description": "研究室詳細", "category": "frontend"},
        {"path": "/api/fields/{field_id}", "method": "GET", "description": "研究分野詳細", "category": "frontend"},
        
        # メインAPI（v1）
        {"path": "/api/v1/research-fields", "method": "GET", "description": "研究分野一覧", "category": "data"},
        {"path": "/api/v1/evaluation-criteria", "method": "GET", "description": "評価基準一覧", "category": "data"},
        {"path": "/api/v1/labs", "method": "GET", "description": "研究室一覧", "category": "data"},
        {"path": "/api/v1/match", "method": "POST", "description": "研究室マッチング", "category": "function"},
        {"path": "/api/v1/optimize", "method": "POST", "description": "重み最適化", "category": "function"},
        {"path": "/api/v1/stats", "method": "GET", "description": "システム統計", "category": "system"},
        {"path": "/api/v1/config", "method": "GET", "description": "システム設定", "category": "system"},
        {"path": "/api/v1/sample-profile", "method": "GET", "description": "サンプルプロフィール", "category": "helper"},
        {"path": "/api/v1/validate-profile", "method": "POST", "description": "プロフィール検証", "category": "helper"},
        {"path": "/api/v1/system-info", "method": "GET", "description": "詳細システム情報", "category": "system"},
        {"path": "/api/v1/test-match", "method": "POST", "description": "マッチングテスト", "category": "test"},
        
        # エイリアス・ショートカット
        {"path": "/api/v1/fields", "method": "GET", "description": "研究分野一覧（短縮パス）", "category": "alias"},
        {"path": "/api/v1/criteria", "method": "GET", "description": "評価基準一覧（短縮パス）", "category": "alias"},
        {"path": "/api/v1/evaluate", "method": "POST", "description": "互換性評価（マッチングのエイリアス）", "category": "alias"},
        {"path": "/api/v1/status", "method": "GET", "description": "システム状態取得（ヘルスチェックのエイリアス）", "category": "alias"},
        
        # デバッグ・診断
        {"path": "/api/debug/endpoints", "method": "GET", "description": "エンドポイント一覧", "category": "debug"},
        {"path": "/api/debug/test", "method": "GET", "description": "接続テスト", "category": "debug"},
        {"path": "/api/debug/test-evaluate", "method": "POST", "description": "/api/evaluate テスト", "category": "debug"},
        {"path": "/api/debug/lab-data-check", "method": "GET", "description": "研究室データ確認", "category": "debug"},
        
        # API文書
        {"path": "/docs", "method": "GET", "description": "API文書（Swagger）", "category": "docs"},
        {"path": "/redoc", "method": "GET", "description": "ReDoc API文書", "category": "docs"}
    ]
    
    # カテゴリ別に分類
    by_category = {}
    for endpoint in endpoints:
        category = endpoint["category"]
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(endpoint)
    
    return {
        "available_endpoints": endpoints,
        "by_category": by_category,
        "total_count": len(endpoints),
        "base_url": "http://localhost:8000",
        "categories": list(by_category.keys()),
        "summary": {
            "system": len(by_category.get("system", [])),
            "data": len(by_category.get("data", [])),
            "function": len(by_category.get("function", [])),
            "frontend": len(by_category.get("frontend", [])),
            "debug": len(by_category.get("debug", [])),
            "docs": len(by_category.get("docs", []))
        },
        "important_frontend_endpoints": [
            "POST /api/evaluate - メインマッチング機能",
            "GET /api/labs - 研究室一覧取得", 
            "GET /api/fields - 研究分野一覧取得",
            "GET /api/status - システム状態確認"
        ],
        "debug_info": {
            "lab_data_count": len(system_state["lab_data"]),
            "criteria_count": len(COMPLETE_EVALUATION_CRITERIA),
            "system_initialized": system_state["initialized"]
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/debug/test", response_class=JSONResponse)
async def debug_test():
    """接続・動作テスト用エンドポイント"""
    return {
        "status": "success",
        "message": "APIエンドポイントは正常に動作しています",
        "server_time": datetime.now().isoformat(),
        "system_status": "operational" if system_state["initialized"] else "initializing",
        "test_data": {
            "number": 12345,
            "boolean": True,
            "array": [1, 2, 3, 4, 5],
            "object": {"key": "value", "nested": {"test": "success"}}
        },
        "available_endpoints": {
            "evaluate": "POST /api/evaluate",
            "labs": "GET /api/labs", 
            "fields": "GET /api/fields",
            "status": "GET /api/status"
        }
    }

@app.post("/api/debug/test-evaluate")
async def debug_test_evaluate():
    """POST /api/evaluate エンドポイントのテスト"""
    try:
        print("🧪 /api/debug/test-evaluate: テスト開始")
        
        # テスト用のサンプルプロフィール
        test_profile_data = {
            "research_intensity": 8.0,
            "advisor_style": 7.0,
            "team_work": 6.0,
            "workload": 7.0,
            "theory_practice": 8.0,
            "research_field_match": 9.0,
            "skill_development": 7.0,
            "lab_atmosphere": 8.0,
            "flexibility": 6.0,
            "publication_opportunity": 8.0,
            "interdisciplinary": 5.0,
            "communication_style": 7.0,
            "innovation_risk": 8.0
        }
        
        # 内部的に /api/evaluate を呼び出し
        # MockRequestを作成
        from fastapi import Request
        
        class MockRequest:
            async def json(self):
                return test_profile_data
        
        mock_request = MockRequest()
        result = await evaluate_compatibility_frontend(mock_request)
        
        return {
            "test_status": "success",
            "message": "/api/evaluate エンドポイントは正常に動作しています",
            "test_profile": test_profile_data,
            "result_preview": {
                "total_labs": len(result.get("lab_results", [])) if hasattr(result, 'get') else 0,
                "endpoint_accessible": True
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ /api/debug/test-evaluate エラー: {e}")
        return {
            "test_status": "error",
            "message": f"/api/evaluate エンドポイントのテストに失敗しました: {str(e)}",
            "error_details": str(e),
            "timestamp": datetime.now().isoformat()
        }

# OPTIONS プリフライトリクエスト対応
@app.options("/{path:path}")
async def options_handler(path: str):
    """CORS プリフライトリクエスト対応"""
    print(f"🔄 OPTIONS リクエスト: {path}")
    return JSONResponse(
        content={"message": "OK"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Max-Age": "86400"
        }
    )

# ===== API エンドポイント =====

@app.get("/", response_class=JSONResponse)
async def root():
    """ルートエンドポイント - システム状態確認（強化版）"""
    try:
        return {
            "message": "研究室選択支援システム v3.0",
            "status": "operational" if system_state["initialized"] else "initializing",
            "version": "3.0.0",
            "timestamp": datetime.now().isoformat(),
            "features": {
                "research_fields": len(RESEARCH_FIELDS_DATA),
                "evaluation_criteria": len(COMPLETE_EVALUATION_CRITERIA),
                "lab_database": len(system_state["lab_data"]),
                "genetic_algorithm": True,
                "fuzzy_inference": True,
                "decision_tree": True
            },
            "stats": {
                "api_calls": system_state["api_calls"],
                "error_count": system_state["error_count"],
                "uptime_seconds": int((datetime.now() - system_state["server_start_time"]).total_seconds())
            },
            "api_endpoints": {
                "health": "/health",
                "docs": "/docs",
                "research_fields": "/api/v1/research-fields",
                "evaluation_criteria": "/api/v1/evaluation-criteria",
                "labs": "/api/v1/labs",
                "match": "/api/v1/match",
                "optimize": "/api/v1/optimize",
                "stats": "/api/v1/stats",
                # フロントエンド互換エンドポイント
                "evaluate_compat": "/api/evaluate",
                "labs_compat": "/api/labs",
                "fields_compat": "/api/fields",
                "status_compat": "/api/status"
            }
        }
    except Exception as e:
        print(f"❌ ルートエンドポイントエラー: {e}")
        return {
            "message": "研究室選択支援システム v3.0",
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/health", response_class=JSONResponse)
async def health_check():
    """ヘルスチェック（強化版）"""
    try:
        health_status = {
            "status": "healthy" if system_state["initialized"] else "starting",
            "timestamp": datetime.now().isoformat(),
            "database_status": "connected" if len(system_state["lab_data"]) > 0 else "empty",
            "system_info": {
                "has_numpy": HAS_NUMPY,
                "has_fastapi": HAS_FASTAPI,
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform
            },
            "services": {
                "api": True,
                "fuzzy_engine": True,
                "genetic_algorithm": True,
                "lab_database": len(system_state["lab_data"]) > 0
            },
            "performance": {
                "total_api_calls": system_state["api_calls"],
                "error_count": system_state["error_count"],
                "error_rate": system_state["error_count"] / max(1, system_state["api_calls"])
            }
        }
        
        # 簡易パフォーマンステスト
        start_time = time.time()
        test_calculation = sum(i * i for i in range(1000))
        calculation_time = time.time() - start_time
        health_status["performance"]["calculation_test_ms"] = round(calculation_time * 1000, 2)
        
        return health_status
        
    except Exception as e:
        print(f"❌ ヘルスチェックエラー: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

# ===== フロントエンド互換性エンドポイント（最初に配置） =====

@app.post("/api/evaluate")
async def evaluate_compatibility_frontend(request: Request):
    """フロントエンド互換性エンドポイント - /api/evaluate（完全修正版）"""
    try:
        print(f"🎯 /api/evaluate エンドポイント実行開始")
        
        # システム初期化チェック
        if not system_state["initialized"]:
            print("⚠️ システム初期化実行中...")
            if not initialize_system():
                raise HTTPException(
                    status_code=503,
                    detail="システムの初期化に失敗しました。サーバーを再起動してください。"
                )
        
        # リクエストボディを取得
        body = await request.json()
        print(f"📥 リクエスト受信: {type(body)} - {len(str(body))}文字")
        
        # リクエスト形式の正規化
        profile_data = None
        if "student_profile" in body:
            profile_data = body["student_profile"]
            print("📋 形式: student_profile")
        elif "preferences" in body:
            profile_data = body["preferences"]
            print("📋 形式: preferences")
        elif "evaluation_criteria" in body:
            profile_data = body["evaluation_criteria"]
            print("📋 形式: evaluation_criteria")
        else:
            profile_data = body
            print("📋 形式: 直接プロフィール")
        
        if not profile_data:
            raise HTTPException(
                status_code=400,
                detail="プロフィールデータが見つかりません"
            )
        
        print(f"📊 プロフィールデータ項目数: {len(profile_data)}")
        
        # 13項目の必須フィールド確認と補完
        missing_fields = [field for field in COMPLETE_EVALUATION_CRITERIA if field not in profile_data]
        if missing_fields:
            print(f"⚠️ 不足フィールド: {missing_fields}")
            for field in missing_fields:
                profile_data[field] = 5.0  # デフォルト値で補完
            print(f"✅ デフォルト値で補完完了")
        
        # StudentProfile検証
        for criterion in COMPLETE_EVALUATION_CRITERIA:
            value = profile_data.get(criterion, 5.0)
            if not isinstance(value, (int, float)) or not (1 <= value <= 10):
                print(f"⚠️ 無効な値を修正: {criterion} = {value} -> 5.0")
                profile_data[criterion] = 5.0
        
        print(f"✅ プロフィール検証完了: {len(profile_data)}項目")
        
        # マッチング実行
        print(f"🔄 マッチング処理開始...")
        print(f"📊 利用可能研究室数: {len(system_state['lab_data'])}")
        
        if not system_state["lab_data"]:
            raise HTTPException(
                status_code=500,
                detail="研究室データが初期化されていません"
            )
        
        # 各研究室との適合度計算（修正版）
        results = []
        calculation_errors = []
        
        for i, lab in enumerate(system_state["lab_data"]):
            try:
                print(f"\n--- 研究室 {i+1}: {lab.get('name', 'Unknown')} ---")
                
                # 新しい適合度計算関数を使用
                compatibility = calculate_enhanced_compatibility(profile_data, lab)
                
                result = {
                    "lab_id": compatibility["lab_id"],
                    "lab_name": compatibility["lab_name"],
                    "compatibility_score": compatibility["overall_score"],
                    "detailed_scores": {
                        criterion: data["weighted_score"] if isinstance(data, dict) else data
                        for criterion, data in compatibility["criteria_scores"].items()
                    },
                    "explanation": generate_detailed_explanation(compatibility),
                    "recommendation_level": compatibility["recommendation_level"],
                    "field_match": compatibility["field_match"],
                    "timestamp": datetime.now().isoformat(),
                    # 追加の詳細情報
                    "advisor": lab.get("advisor", "不明"),
                    "description": lab.get("description", ""),
                    "field_category": lab.get("field_category", ""),
                    "base_score": compatibility.get("base_score", compatibility["overall_score"]),
                    "data_completeness": compatibility.get("data_completeness", 1.0)
                }
                results.append(result)
                print(f"✅ 計算成功: {compatibility['lab_name']} = {compatibility['overall_score']:.3f}")
                
            except Exception as e:
                error_msg = f"研究室 {lab.get('name', f'ID:{i}')} の計算エラー: {str(e)}"
                calculation_errors.append(error_msg)
                print(f"❌ {error_msg}")
                continue
        
        if not results:
            raise HTTPException(
                status_code=500,
                detail="すべての研究室の計算に失敗しました。システム管理者に連絡してください。"
            )
        
        # スコア順でソート
        results.sort(key=lambda x: x["compatibility_score"], reverse=True)
        
        # 統計計算
        scores = [r["compatibility_score"] for r in results]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        
        # フロントエンド互換形式でレスポンス
        frontend_response = {
            "lab_results": results,
            "results": results,  # 両方の形式をサポート
            "summary": {
                "total_labs": len(results),
                "avg_score": avg_score,
                "max_score": max_score,
                "min_score": min_score,
                "high_compatibility_count": len([r for r in results if r["compatibility_score"] >= 0.8]),
                "medium_compatibility_count": len([r for r in results if 0.6 <= r["compatibility_score"] < 0.8]),
                "low_compatibility_count": len([r for r in results if r["compatibility_score"] < 0.6])
            },
            "metadata": {
                "processing_time": time.time() - time.time(),
                "evaluation_count": system_state["evaluation_count"] + 1,
                "timestamp": datetime.now().isoformat(),
                "endpoint": "/api/evaluate",
                "calculation_method": "enhanced_compatibility_v3",
                "criteria_used": len(COMPLETE_EVALUATION_CRITERIA)
            }
        }
        
        # 評価回数更新
        system_state["evaluation_count"] += 1
        
        if calculation_errors:
            frontend_response["warnings"] = {
                "calculation_errors": calculation_errors,
                "message": "一部の研究室で計算エラーが発生しましたが、他の研究室の結果を表示しています。"
            }
        
        print(f"📤 /api/evaluate レスポンス送信: {len(results)}件")
        print(f"📊 適合度統計: 平均={avg_score:.3f}, 最高={max_score:.3f}, 最低={min_score:.3f}")
        
        return JSONResponse(content=frontend_response)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ /api/evaluate 予期しないエラー: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "message": "評価処理で予期しないエラーが発生しました",
                "error": str(e),
                "endpoint": "/api/evaluate",
                "suggestion": "システム管理者に連絡してください"
            }
        )

@app.post("/api/optimize")
async def optimize_frontend(request: Request):
    """フロントエンド互換性エンドポイント - /api/optimize"""
    try:
        print(f"🧬 /api/optimize エンドポイント実行開始")
        
        body = await request.json()
        print(f"📥 /api/optimize リクエスト受信")
        
        # リクエスト形式の正規化
        if "student_profile" in body:
            profile_data = body["student_profile"]
            optimization_params = body.get("optimization_params", {})
        elif "evaluation_criteria" in body:
            profile_data = body["evaluation_criteria"]
            optimization_params = body
        else:
            profile_data = body
            optimization_params = {}
        
        # 不足フィールドを補完
        for field in COMPLETE_EVALUATION_CRITERIA:
            if field not in profile_data:
                profile_data[field] = 5.0
        
        # OptimizationRequestに変換
        try:
            student_profile = StudentProfile(**profile_data)
            optimization_request = OptimizationRequest(
                student_profile=student_profile,
                population_size=optimization_params.get("population_size", 30),
                generations=optimization_params.get("generations", 20),
                mutation_rate=optimization_params.get("mutation_rate", 0.1),
                crossover_rate=optimization_params.get("crossover_rate", 0.8),
                custom_weights=optimization_params.get("custom_weights")
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"最適化リクエストの形式が無効です: {str(e)}"
            )
        
        # 最適化実行
        result = await optimize_weights(optimization_request)
        
        # フロントエンド互換形式でレスポンス
        frontend_response = {
            "optimization_result": result["optimization_result"],
            "lab_results": result["sample_results"],
            "results": result["sample_results"],
            "metadata": {
                "processing_time": result["processing_time_seconds"],
                "parameters_used": result["parameters_used"],
                "timestamp": result["timestamp"],
                "endpoint": "/api/optimize"
            }
        }
        
        print(f"📤 /api/optimize レスポンス送信")
        return JSONResponse(content=frontend_response)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ /api/optimize エラー: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"最適化処理でエラーが発生しました: {str(e)}"
        )

@app.get("/api/labs")
async def get_labs_frontend():
    """フロントエンド互換性エンドポイント - /api/labs"""
    try:
        result = await get_labs()
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/fields")
async def get_fields_frontend():
    """フロントエンド互換性エンドポイント - /api/fields"""
    try:
        result = await get_research_fields()
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_status_frontend():
    """フロントエンド互換性エンドポイント - /api/status"""
    try:
        result = await health_check()
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/labs/{lab_id}")
async def get_lab_detail_frontend(lab_id: str):
    """特定研究室の詳細取得"""
    try:
        # 研究室データから指定IDを検索
        lab = next((lab for lab in system_state["lab_data"] if lab["id"] == lab_id), None)
        
        if not lab:
            raise HTTPException(
                status_code=404,
                detail=f"研究室ID '{lab_id}' が見つかりません"
            )
        
        return JSONResponse(content={
            "lab": lab,
            "timestamp": datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"研究室詳細の取得に失敗しました: {str(e)}"
        )

@app.get("/api/fields/{field_id}")
async def get_field_detail_frontend(field_id: str):
    """特定研究分野の詳細取得"""
    try:
        field = next((field for field in RESEARCH_FIELDS_DATA if field["id"] == field_id), None)
        
        if not field:
            raise HTTPException(
                status_code=404,
                detail=f"研究分野ID '{field_id}' が見つかりません"
            )
        
        return JSONResponse(content={
            "field": field,
            "timestamp": datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"研究分野詳細の取得に失敗しました: {str(e)}"
        )

@app.get("/api/categories")
async def get_categories_frontend():
    """研究分野カテゴリ一覧取得"""
    try:
        categories = {}
        for field in RESEARCH_FIELDS_DATA:
            category = field["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(field)
        
        return JSONResponse(content={
            "categories": categories,
            "category_names": list(categories.keys()),
            "total_categories": len(categories),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"カテゴリ一覧の取得に失敗しました: {str(e)}"
        )

@app.get("/api/evaluation-criteria")
async def get_evaluation_criteria_frontend():
    """フロントエンド互換性エンドポイント - /api/evaluation-criteria"""
    try:
        result = await get_evaluation_criteria()
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/research-fields", response_class=JSONResponse)
async def get_research_fields():
    """研究分野一覧取得"""
    return {
        "fields": RESEARCH_FIELDS_DATA,
        "total_count": len(RESEARCH_FIELDS_DATA),
        "categories": list(set(field["category"] for field in RESEARCH_FIELDS_DATA))
    }

@app.get("/api/v1/evaluation-criteria", response_class=JSONResponse)
async def get_evaluation_criteria():
    """評価基準一覧取得"""
    return {
        "criteria": COMPLETE_EVALUATION_CRITERIA,
        "total_count": len(COMPLETE_EVALUATION_CRITERIA),
        "categories": {
            "basic": COMPLETE_EVALUATION_CRITERIA[:5],
            "extended": COMPLETE_EVALUATION_CRITERIA[5:10],
            "special": COMPLETE_EVALUATION_CRITERIA[10:]
        },
        "default_weights": DEFAULT_CRITERIA_WEIGHTS
    }

@app.get("/api/v1/labs", response_class=JSONResponse)
async def get_labs():
    """研究室一覧取得"""
    return {
        "labs": system_state["lab_data"],
        "total_count": len(system_state["lab_data"]),
        "last_updated": system_state["last_updated"]
    }

@app.post("/api/v1/match", response_class=JSONResponse)
async def match_labs(student_profile: StudentProfile):
    """研究室マッチング実行（強化版）"""
    try:
        print(f"🎯 マッチングリクエスト受信: {student_profile.student_id or 'anonymous'}")
        
        # システム初期化チェック
        if not system_state["initialized"]:
            print("⚠️ システムが初期化されていません。初期化を実行...")
            if not initialize_system():
                raise HTTPException(
                    status_code=503, 
                    detail="システムの初期化に失敗しました。サーバーを再起動してください。"
                )
        
        system_state["evaluation_count"] += 1
        start_time = time.time()
        
        # 学生プロフィールを辞書に変換
        profile_dict = student_profile.dict()
        print(f"📝 プロフィール検証: {len(profile_dict)}項目")
        
        # 入力値検証
        for criterion in COMPLETE_EVALUATION_CRITERIA:
            if criterion not in profile_dict:
                raise HTTPException(
                    status_code=400,
                    detail=f"必須フィールドが不足しています: {criterion}"
                )
            
            value = profile_dict[criterion]
            if not isinstance(value, (int, float)) or not (1 <= value <= 10):
                raise HTTPException(
                    status_code=400,
                    detail=f"無効な値: {criterion} = {value} (1-10の範囲で指定してください)"
                )
        
        # 全研究室との適合度計算
        results = []
        calculation_errors = []
        
        for i, lab in enumerate(system_state["lab_data"]):
            try:
                compatibility = calculate_enhanced_compatibility(profile_dict, lab)
                
                result = MatchingResult(
                    lab_id=compatibility["lab_id"],
                    lab_name=compatibility["lab_name"],
                    compatibility_score=compatibility["overall_score"],
                    detailed_scores={
                        criterion: data["weighted_score"]
                        for criterion, data in compatibility["criteria_scores"].items()
                    },
                    explanation=generate_detailed_explanation(compatibility),
                    recommendation_level=compatibility["recommendation_level"],
                    field_match=compatibility["field_match"],
                    timestamp=datetime.now().isoformat()
                )
                results.append(result)
                
            except Exception as e:
                error_msg = f"研究室 {lab.get('name', f'ID:{i}')} の計算エラー: {str(e)}"
                calculation_errors.append(error_msg)
                print(f"⚠️ {error_msg}")
                continue
        
        if not results:
            raise HTTPException(
                status_code=500,
                detail="マッチング計算でエラーが発生しました。すべての研究室の計算に失敗しました。"
            )
        
        # スコア順でソート
        results.sort(key=lambda x: x.compatibility_score, reverse=True)
        
        processing_time = time.time() - start_time
        
        response_data = {
            "results": [result.dict() for result in results],
            "total_evaluated": len(results),
            "processing_time_seconds": round(processing_time, 3),
            "evaluation_count": system_state["evaluation_count"],
            "timestamp": datetime.now().isoformat(),
            "quality_metrics": {
                "successful_calculations": len(results),
                "failed_calculations": len(calculation_errors),
                "success_rate": len(results) / len(system_state["lab_data"]),
                "average_compatibility": sum(r.compatibility_score for r in results) / len(results)
            }
        }
        
        if calculation_errors:
            response_data["warnings"] = {
                "calculation_errors": calculation_errors,
                "message": "一部の研究室で計算エラーが発生しましたが、他の研究室の結果を表示しています。"
            }
        
        print(f"✅ マッチング完了: {len(results)}件, {processing_time:.3f}秒")
        return response_data
        
    except HTTPException:
        # HTTPExceptionはそのまま再発生
        raise
    except Exception as e:
        print(f"❌ マッチング処理で予期しないエラー: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail={
                "message": "マッチング処理で予期しないエラーが発生しました",
                "error": str(e),
                "suggestion": "入力データを確認し、再度お試しください"
            }
        )

@app.post("/api/v1/optimize", response_class=JSONResponse)
async def optimize_weights(request: OptimizationRequest):
    """遺伝的アルゴリズムによる重み最適化（強化版）"""
    try:
        print(f"🧬 最適化リクエスト受信: 集団サイズ={request.population_size}, 世代数={request.generations}")
        
        # システム初期化チェック
        if not system_state["initialized"]:
            initialize_system()
        
        start_time = time.time()
        
        # パラメータ検証
        if not (10 <= request.population_size <= 100):
            raise HTTPException(status_code=400, detail="population_size は 10-100 の範囲で指定してください")
        
        if not (5 <= request.generations <= 50):
            raise HTTPException(status_code=400, detail="generations は 5-50 の範囲で指定してください")
        
        if not (0.01 <= request.mutation_rate <= 0.5):
            raise HTTPException(status_code=400, detail="mutation_rate は 0.01-0.5 の範囲で指定してください")
        
        # 最適化実行
        try:
            optimization_result = genetic_algorithm_optimization(
                student_profile=request.student_profile.dict(),
                lab_data=system_state["lab_data"],
                population_size=request.population_size,
                generations=request.generations,
                mutation_rate=request.mutation_rate,
                crossover_rate=request.crossover_rate,
                custom_weights=request.custom_weights
            )
        except Exception as e:
            print(f"❌ 遺伝的アルゴリズム実行エラー: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"遺伝的アルゴリズムの実行でエラーが発生しました: {str(e)}"
            )
        
        # 最適化された重みでマッチング実行
        optimized_profile = request.student_profile.dict()
        results = []
        
        try:
            sample_labs = system_state["lab_data"][:min(10, len(system_state["lab_data"]))]
            
            for lab in sample_labs:
                compatibility = calculate_enhanced_compatibility(
                    optimized_profile, 
                    lab, 
                    optimization_result["optimized_weights"]
                )
                results.append({
                    "lab_id": compatibility["lab_id"],
                    "lab_name": compatibility["lab_name"],
                    "compatibility_score": compatibility["overall_score"],
                    "recommendation_level": compatibility["recommendation_level"]
                })
            
            results.sort(key=lambda x: x["compatibility_score"], reverse=True)
            
        except Exception as e:
            print(f"❌ 最適化後マッチング計算エラー: {e}")
            # 最適化結果は返すが、サンプル結果は空にする
            results = []
        
        processing_time = time.time() - start_time
        
        response_data = {
            "optimization_result": optimization_result,
            "sample_results": results[:5],
            "processing_time_seconds": round(processing_time, 3),
            "parameters_used": {
                "population_size": request.population_size,
                "generations": request.generations,
                "mutation_rate": request.mutation_rate,
                "crossover_rate": request.crossover_rate
            },
            "quality_metrics": {
                "improvement_achieved": optimization_result.get("improvement", False),
                "final_fitness": optimization_result.get("fitness_score", 0),
                "sample_labs_evaluated": len(results)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"✅ 最適化完了: {processing_time:.3f}秒, 改善={optimization_result.get('improvement', False)}")
        return response_data
        
    except HTTPException:
        # HTTPExceptionはそのまま再発生
        raise
    except Exception as e:
        print(f"❌ 最適化処理で予期しないエラー: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail={
                "message": "最適化処理で予期しないエラーが発生しました",
                "error": str(e),
                "suggestion": "パラメータを確認し、再度お試しください"
            }
        )

# ===== メインAPI v1エンドポイント =====

@app.get("/api/v1/fields", response_class=JSONResponse)
async def get_fields_alias():
    """研究分野一覧取得（短縮パス）"""
    return await get_research_fields()

@app.get("/api/v1/criteria", response_class=JSONResponse) 
async def get_criteria_alias():
    """評価基準一覧取得（短縮パス）"""
    return await get_evaluation_criteria()

@app.post("/api/v1/evaluate", response_class=JSONResponse)
async def evaluate_compatibility(student_profile: StudentProfile):
    """互換性評価（マッチングのエイリアス）"""
    return await match_labs(student_profile)

@app.get("/api/v1/status", response_class=JSONResponse)
async def get_status():
    """システム状態取得（ヘルスチェックのエイリアス）"""
    return await health_check()

@app.post("/api/v1/test-match", response_class=JSONResponse)
async def test_matching():
    """マッチングテスト用エンドポイント（詳細ログ付き）"""
    try:
        print("\n🧪 === マッチングテスト開始 ===")
        
        # テスト用プロフィール
        test_profile_data = {
            "research_intensity": 8.0,
            "advisor_style": 7.0,
            "team_work": 6.0,
            "workload": 7.0,
            "theory_practice": 8.0,
            "research_field_match": 9.0,
            "skill_development": 7.0,
            "lab_atmosphere": 8.0,
            "flexibility": 6.0,
            "publication_opportunity": 8.0,
            "interdisciplinary": 5.0,
            "communication_style": 7.0,
            "innovation_risk": 8.0,
            "student_id": "test_user",
            "preferred_fields": ["人工知能・機械学習"]
        }
        
        print(f"📝 テストプロフィール: {test_profile_data}")
        
        # StudentProfileに変換
        test_profile = StudentProfile(**test_profile_data)
        print(f"✅ プロフィール変換成功")
        
        # システム初期化チェック
        if not system_state["initialized"]:
            print("⚠️ システム初期化実行中...")
            initialize_system()
        
        print(f"📊 利用可能研究室数: {len(system_state['lab_data'])}")
        
        # 各研究室との適合度を個別にテスト
        test_results = []
        for i, lab in enumerate(system_state["lab_data"][:3]):  # 最初の3件をテスト
            print(f"\n--- 研究室 {i+1}: {lab['name']} ---")
            
            try:
                compatibility = calculate_enhanced_compatibility(test_profile_data, lab)
                test_results.append({
                    "lab_name": compatibility["lab_name"],
                    "lab_id": compatibility["lab_id"],
                    "compatibility_score": compatibility["overall_score"],
                    "base_score": compatibility.get("base_score", 0),
                    "criteria_count": compatibility["total_criteria_evaluated"],
                    "recommendation": compatibility["recommendation_level"],
                    "calculation_success": True
                })
                print(f"✅ 計算成功: スコア={compatibility['overall_score']:.3f}")
            except Exception as e:
                print(f"❌ 計算エラー: {e}")
                test_results.append({
                    "lab_name": lab.get("name", "Unknown"),
                    "lab_id": lab.get("id", "unknown"),
                    "compatibility_score": 0.0,
                    "error": str(e),
                    "calculation_success": False
                })
        
        # 正常なマッチング処理も実行
        try:
            print(f"\n🔄 フルマッチング実行...")
            full_result = await match_labs(test_profile)
            full_success = True
            full_count = full_result.get("total_evaluated", 0)
            print(f"✅ フルマッチング成功: {full_count}件評価")
        except Exception as e:
            print(f"❌ フルマッチングエラー: {e}")
            full_success = False
            full_count = 0
            full_result = {"error": str(e)}
        
        return {
            "test_status": "completed",
            "message": "マッチングテストが完了しました",
            "test_profile": test_profile_data,
            "individual_tests": test_results,
            "full_matching": {
                "success": full_success,
                "total_evaluated": full_count,
                "sample_results": full_result.get("results", [])[:3] if full_success else []
            },
            "system_info": {
                "initialized": system_state["initialized"],
                "lab_count": len(system_state["lab_data"]),
                "criteria_count": len(COMPLETE_EVALUATION_CRITERIA)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        traceback.print_exc()
        return {
            "test_status": "error",
            "message": f"マッチングテストに失敗しました: {str(e)}",
            "error_details": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/debug/lab-data-check", response_class=JSONResponse)
async def check_lab_data():
    """研究室データの構造確認"""
    try:
        if not system_state["lab_data"]:
            return {"error": "研究室データが初期化されていません"}
        
        sample_lab = system_state["lab_data"][0]
        
        # 13項目の存在チェック
        criteria_status = {}
        for criterion in COMPLETE_EVALUATION_CRITERIA:
            criteria_status[criterion] = {
                "exists": criterion in sample_lab,
                "value": sample_lab.get(criterion, "不明"),
                "type": type(sample_lab.get(criterion, None)).__name__
            }
        
        return {
            "total_labs": len(system_state["lab_data"]),
            "sample_lab": {
                "id": sample_lab.get("id"),
                "name": sample_lab.get("name"),
                "all_fields": list(sample_lab.keys())
            },
            "criteria_status": criteria_status,
            "missing_criteria": [c for c in COMPLETE_EVALUATION_CRITERIA if c not in sample_lab],
            "extra_fields": [k for k in sample_lab.keys() if k not in COMPLETE_EVALUATION_CRITERIA],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "error": f"データ確認でエラーが発生しました: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/v1/config", response_class=JSONResponse)
async def get_system_config():
    """システム設定情報取得"""
    try:
        return {
            "system_config": {
                "version": "3.0.0",
                "api_version": "v1",
                "max_request_size": "10MB",
                "timeout_seconds": 30,
                "supported_methods": ["GET", "POST", "OPTIONS"],
                "cors_enabled": True
            },
            "evaluation_config": {
                "criteria_count": len(COMPLETE_EVALUATION_CRITERIA),
                "value_range": {"min": 1, "max": 10},
                "required_fields": COMPLETE_EVALUATION_CRITERIA
            },
            "lab_config": {
                "total_labs": len(system_state["lab_data"]),
                "field_count": len(RESEARCH_FIELDS_DATA),
                "matching_algorithm": "fuzzy_genetic"
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"設定情報の取得に失敗しました: {str(e)}")

@app.get("/api/v1/sample-profile", response_class=JSONResponse)
async def get_sample_profile():
    """サンプルプロフィール取得"""
    try:
        sample_profiles = [
            {
                "name": "AI研究志向",
                "description": "人工知能・機械学習分野に強い関心",
                "profile": {
                    "research_intensity": 9.0,
                    "advisor_style": 6.0,
                    "team_work": 7.0,
                    "workload": 8.0,
                    "theory_practice": 7.0,
                    "research_field_match": 10.0,
                    "skill_development": 8.0,
                    "lab_atmosphere": 7.0,
                    "flexibility": 6.0,
                    "publication_opportunity": 9.0,
                    "interdisciplinary": 6.0,
                    "communication_style": 7.0,
                    "innovation_risk": 9.0
                }
            },
            {
                "name": "バランス型",
                "description": "全体的にバランスの取れた志向",
                "profile": {
                    "research_intensity": 6.0,
                    "advisor_style": 6.0,
                    "team_work": 7.0,
                    "workload": 6.0,
                    "theory_practice": 6.0,
                    "research_field_match": 7.0,
                    "skill_development": 7.0,
                    "lab_atmosphere": 7.0,
                    "flexibility": 7.0,
                    "publication_opportunity": 6.0,
                    "interdisciplinary": 6.0,
                    "communication_style": 7.0,
                    "innovation_risk": 6.0
                }
            },
            {
                "name": "デザイン重視",
                "description": "クリエイティブ・デザイン分野志向",
                "profile": {
                    "research_intensity": 6.0,
                    "advisor_style": 8.0,
                    "team_work": 9.0,
                    "workload": 5.0,
                    "theory_practice": 8.0,
                    "research_field_match": 8.0,
                    "skill_development": 9.0,
                    "lab_atmosphere": 9.0,
                    "flexibility": 9.0,
                    "publication_opportunity": 5.0,
                    "interdisciplinary": 8.0,
                    "communication_style": 9.0,
                    "innovation_risk": 8.0
                }
            }
        ]
        
        return {
            "sample_profiles": sample_profiles,
            "usage": "これらのプロフィールはテスト・デモ用です",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"サンプルプロフィールの取得に失敗しました: {str(e)}")

@app.post("/api/v1/validate-profile", response_class=JSONResponse)
async def validate_student_profile(profile_data: dict):
    """学生プロフィールの検証"""
    try:
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        # 必須フィールドチェック
        for criterion in COMPLETE_EVALUATION_CRITERIA:
            if criterion not in profile_data:
                validation_results["is_valid"] = False
                validation_results["errors"].append(f"必須フィールドが不足: {criterion}")
            else:
                value = profile_data[criterion]
                if not isinstance(value, (int, float)):
                    validation_results["is_valid"] = False
                    validation_results["errors"].append(f"無効なデータ型: {criterion} ({type(value).__name__})")
                elif not (1 <= value <= 10):
                    validation_results["is_valid"] = False
                    validation_results["errors"].append(f"値の範囲外: {criterion} = {value} (1-10が必要)")
                elif value == 5.0:
                    validation_results["warnings"].append(f"中間値: {criterion} = {value} (より具体的な値を推奨)")
        
        # バランスチェック
        if validation_results["is_valid"]:
            values = [profile_data[c] for c in COMPLETE_EVALUATION_CRITERIA if c in profile_data]
            if values:
                mean_val = sum(values) / len(values)
                if mean_val < 3.0:
                    validation_results["suggestions"].append("全体的に低い値です。より積極的な評価を検討してください")
                elif mean_val > 8.0:
                    validation_results["suggestions"].append("全体的に高い値です。より現実的な評価を検討してください")
                
                # 分散チェック
                variance = sum((v - mean_val) ** 2 for v in values) / len(values)
                if variance < 0.5:
                    validation_results["suggestions"].append("値のばらつきが少ないです。特徴をより明確にすることを推奨します")
        
        return {
            "validation": validation_results,
            "profile_summary": {
                "field_count": len([k for k in profile_data.keys() if k in COMPLETE_EVALUATION_CRITERIA]),
                "required_field_count": len(COMPLETE_EVALUATION_CRITERIA),
                "completion_rate": len([k for k in profile_data.keys() if k in COMPLETE_EVALUATION_CRITERIA]) / len(COMPLETE_EVALUATION_CRITERIA)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"プロフィール検証に失敗しました: {str(e)}")

@app.get("/api/v1/system-info", response_class=JSONResponse)
async def get_detailed_system_info():
    """詳細システム情報"""
    try:
        import platform
        import psutil
        
        system_info = {
            "application": {
                "name": "研究室選択支援システム",
                "version": "3.0.0",
                "api_version": "v1",
                "description": "遺伝的アルゴリズムを用いたファジィ決定木による研究室マッチングシステム"
            },
            "server": {
                "platform": platform.platform(),
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "architecture": platform.architecture()[0],
                "processor": platform.processor() or "Unknown"
            },
            "features": {
                "numpy_available": HAS_NUMPY,
                "fastapi_available": HAS_FASTAPI,
                "fuzzy_inference": True,
                "genetic_algorithm": True,
                "decision_tree": True,
                "cors_enabled": True
            },
            "data": {
                "research_fields": len(RESEARCH_FIELDS_DATA),
                "evaluation_criteria": len(COMPLETE_EVALUATION_CRITERIA),
                "lab_database": len(system_state["lab_data"]),
                "algorithm_components": ["fuzzy_membership", "genetic_optimization", "decision_tree_classification"]
            },
            "runtime": {
                "uptime_seconds": int((datetime.now() - system_state["server_start_time"]).total_seconds()),
                "total_requests": system_state["api_calls"],
                "error_count": system_state["error_count"],
                "evaluation_count": system_state["evaluation_count"]
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return system_info
        
    except Exception as e:
        # psutilが利用できない場合の基本情報
        return {
            "application": {
                "name": "研究室選択支援システム",
                "version": "3.0.0",
                "status": "operational"
            },
            "error": f"詳細情報の取得に失敗: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
    """システム統計情報"""
    return {
        "system_info": {
            "version": "3.0.0",
            "initialized": system_state["initialized"],
            "uptime_seconds": int((datetime.now() - system_state["server_start_time"]).total_seconds()),
            "database_version": system_state["database_version"]
        },
        "usage_stats": {
            "total_api_calls": system_state["api_calls"],
            "evaluation_count": system_state["evaluation_count"],
            "error_count": system_state["error_count"],
            "success_rate": (system_state["api_calls"] - system_state["error_count"]) / max(1, system_state["api_calls"])
        },
        "data_stats": {
            "research_fields": len(RESEARCH_FIELDS_DATA),
            "evaluation_criteria": len(COMPLETE_EVALUATION_CRITERIA),
            "lab_database_size": len(system_state["lab_data"]),
            "last_updated": system_state["last_updated"]
        }
    }

# サーバー起動設定（改善版）
def start_server():
    """サーバー起動関数"""
    
    # システム初期化
    print("🔧 システム初期化中...")
    if not initialize_system():
        print("❌ システム初期化に失敗しました")
        return False
    
    # ポート設定
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"\n🚀 研究室選択支援システム v3.0 サーバー起動...")
    print(f"📍 URL: http://localhost:{port}")
    print(f"📚 API文書: http://localhost:{port}/docs")
    print(f"🔧 ヘルスチェック: http://localhost:{port}/health")
    print(f"🎯 研究分野: {len(RESEARCH_FIELDS_DATA)}分野")
    print(f"📊 評価基準: {len(COMPLETE_EVALUATION_CRITERIA)}項目")
    print(f"🏛️ 研究室データ: {len(EXTENDED_LAB_DATA)}件")
    
    print("\n📋 利用可能なAPIエンドポイント:")
    endpoints = [
        ("GET", "/", "システム情報"),
        ("GET", "/health", "ヘルスチェック"),
        
        # メインAPIエンドポイント（v1）
        ("GET", "/api/v1/research-fields", "研究分野一覧"),
        ("GET", "/api/v1/evaluation-criteria", "評価基準一覧"),
        ("GET", "/api/v1/labs", "研究室一覧"),
        ("POST", "/api/v1/match", "研究室マッチング"),
        ("POST", "/api/v1/optimize", "重み最適化"),
        ("GET", "/api/v1/stats", "システム統計"),
        ("GET", "/api/v1/config", "システム設定"),
        ("GET", "/api/v1/sample-profile", "サンプルプロフィール"),
        ("POST", "/api/v1/validate-profile", "プロフィール検証"),
        
        # フロントエンド互換性エンドポイント
        ("POST", "/api/evaluate", "研究室マッチング（互換）"),
        ("POST", "/api/optimize", "重み最適化（互換）"),
        ("GET", "/api/labs", "研究室一覧（互換）"),
        ("GET", "/api/labs/{lab_id}", "研究室詳細"),
        ("GET", "/api/fields", "研究分野一覧（互換）"),
        ("GET", "/api/fields/{field_id}", "研究分野詳細"),
        ("GET", "/api/categories", "研究分野カテゴリ"),
        ("GET", "/api/evaluation-criteria", "評価基準（互換）"),
        ("GET", "/api/status", "システム状態（互換）"),
        
        # デバッグ・診断エンドポイント
        ("GET", "/api/debug/endpoints", "エンドポイント一覧"),
        ("GET", "/api/debug/test", "接続テスト"),
        ("POST", "/api/v1/test-match", "マッチングテスト"),
        ("GET", "/api/v1/system-info", "詳細システム情報"),
        
        # API文書
        ("GET", "/docs", "API文書（Swagger）"),
        ("GET", "/redoc", "API文書（ReDoc）")
    ]
    
    for method, path, description in endpoints:
        print(f"   {method:4} {path:30} - {description}")
    
    print("\n💡 フロントエンドから接続テスト:")
    print(f"   # 基本接続テスト")
    print(f"   curl http://localhost:{port}/health")
    print(f"   curl http://localhost:{port}/api/debug/test")
    print(f"   curl http://localhost:{port}/api/debug/endpoints")
    
    print(f"\n   # データ取得テスト")
    print(f"   curl http://localhost:{port}/api/fields")
    print(f"   curl http://localhost:{port}/api/labs")
    print(f"   curl http://localhost:{port}/api/evaluation-criteria")
    
    print(f"\n   # フロントエンド互換エンドポイントテスト")
    print(f"   curl http://localhost:{port}/api/status")
    print(f"   curl http://localhost:{port}/api/categories")
    
    print("\n💡 マッチングテスト:")
    print(f"   # 簡易テスト")
    print(f"   curl -X POST http://localhost:{port}/api/v1/test-match")
    
    print(f"   # 完全マッチングテスト（JSON）")
    sample_json = '''{
  "research_intensity": 8.0,
  "advisor_style": 7.0,
  "team_work": 6.0,
  "workload": 7.0,
  "theory_practice": 8.0,
  "research_field_match": 9.0,
  "skill_development": 7.0,
  "lab_atmosphere": 8.0,
  "flexibility": 6.0,
  "publication_opportunity": 8.0,
  "interdisciplinary": 5.0,
  "communication_style": 7.0,
  "innovation_risk": 8.0
}'''
    print(f"   curl -X POST http://localhost:{port}/api/evaluate \\")
    print(f"        -H 'Content-Type: application/json' \\")
    print(f"        -d '{sample_json}'")
    
    print(f"\n🔧 トラブルシューティング:")
    print(f"   - エンドポイント一覧: http://localhost:{port}/api/debug/endpoints")
    print(f"   - 接続テスト: http://localhost:{port}/api/debug/test")
    print(f"   - システム詳細: http://localhost:{port}/api/v1/system-info")
    print(f"   - API文書: http://localhost:{port}/docs")
    
    print(f"\n📝 フロントエンド開発者向け:")
    print(f"   - ベースURL: http://localhost:{port}")
    print(f"   - メインマッチングAPI: POST /api/evaluate")
    print(f"   - 研究室一覧: GET /api/labs")
    print(f"   - 研究分野一覧: GET /api/fields")
    print(f"   - システム状態: GET /api/status")
    
    print("\n🛑 停止するには Ctrl+C を押してください")
    print("=" * 80)
    
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=False,  # 本番環境では無効
            log_level="info",
            access_log=True,
            server_header=False,
            date_header=False
        )
        
    except KeyboardInterrupt:
        print("\n🛑 サーバーを停止しています...")
        system_state["initialized"] = False
        print("✅ サーバー停止完了")
        
    except Exception as e:
        print(f"❌ サーバー起動エラー: {e}")
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    try:
        # 基本的な環境チェック
        print("🔍 環境チェック中...")
        
        if not HAS_FASTAPI:
            print("❌ FastAPI が利用できません")
            print("💡 解決方法: pip install fastapi uvicorn")
            sys.exit(1)
        
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        print(f"✅ FastAPI 利用可能")
        print(f"✅ NumPy {'利用可能' if HAS_NUMPY else '利用不可（オプション）'}")
        
        # サーバー起動
        success = start_server()
        
        if not success:
            print("❌ サーバー起動に失敗しました")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 プロセスが中断されました")
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        traceback.print_exc()
        sys.exit(1)