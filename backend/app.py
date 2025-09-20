# backend/app.py - 遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム（完全版）

import json
import os
import time
import traceback
import random
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# FastAPI関連
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ===== システム設定 =====

# 13項目完全評価基準
COMPLETE_EVALUATION_CRITERIA = [
    # 基本項目（5項目）
    "research_intensity", "advisor_style", "team_work", "workload", "theory_practice",
    # 拡張項目（5項目）
    "research_field_match", "skill_development", "lab_atmosphere", "flexibility", "publication_opportunity",
    # 特殊項目（3項目）
    "interdisciplinary", "communication_style", "innovation_risk"
]

# 基準別重要度重み（13項目完全対応）
DEFAULT_CRITERIA_WEIGHTS = {
    # 基本項目：高重要度
    "research_intensity": 1.3,
    "advisor_style": 1.2,
    "team_work": 1.1,
    "workload": 1.0,
    "theory_practice": 1.1,
    # 拡張項目：中〜高重要度
    "research_field_match": 1.4,  # 最重要
    "skill_development": 1.1,
    "lab_atmosphere": 1.0,
    "flexibility": 0.9,
    "publication_opportunity": 1.2,
    # 特殊項目：可変重要度
    "interdisciplinary": 0.8,
    "communication_style": 0.9,
    "innovation_risk": 1.0
}

# ===== モジュール可用性チェック =====
try:
    from config.settings import Settings
    settings = Settings()
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False
    print("⚠️ 設定モジュールが利用できません - デフォルト設定を使用")
    
    # デフォルト設定
    @dataclass
    class DefaultSettings:
        host: str = "0.0.0.0"
        port: int = 8000
        debug: bool = True
        core_features: List[str] = None
        
        def __post_init__(self):
            if self.core_features is None:
                self.core_features = COMPLETE_EVALUATION_CRITERIA[:5]  # 基本項目のみ
    
    settings = DefaultSettings()

try:
    from core.fuzzy.inference import SimpleFuzzyInferenceEngine
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("⚠️ ファジィ推論モジュールが利用できません")

try:
    from core.genetic.evolution import EvolutionEngine, EvolutionConfig
    GENETIC_AVAILABLE = True
    print("✅ 遺伝的アルゴリズムモジュールを読み込み完了")
except ImportError as e:
    GENETIC_AVAILABLE = False
    print(f"❌ 遺伝的アルゴリズムモジュールが利用できません: {e}")

try:
    from core.decision_tree.tree import EnhancedFuzzyDecisionTree, TreeConfig
    DECISION_TREE_AVAILABLE = True
except ImportError:
    DECISION_TREE_AVAILABLE = False
    print("⚠️ 決定木モジュールが利用できません")

# ===== FastAPIアプリケーション初期化 =====
app = FastAPI(
    title="研究室選択支援システム",
    description="遺伝的アルゴリズムを用いたファジィ決定木による研究室マッチングシステム",
    version="3.0.0",
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

# ===== システム状態管理 =====
system_state = {
    "initialized": False,
    "fuzzy_engine": None,
    "genetic_engine": None,
    "decision_tree": None,
    "lab_database": [],
    "lab_data": [],  # エイリアス
    "evaluation_count": 0,
    "optimization_count": 0,
    "api_calls": 0,
    "error_count": 0
}

# ===== データ管理関数 =====

def load_labs_from_json(json_path: str = None) -> List[Dict[str, Any]]:
    """labs_database.jsonから研究室データを読み込む"""
    if json_path is None:
        # デフォルトパス設定
        current_dir = Path(__file__).parent
        possible_paths = [
            current_dir / "data" / "labs_database.json",
            current_dir / "labs_database.json",
            current_dir.parent / "labs_database.json",
            current_dir / "backend" / "data" / "labs_database.json"
        ]
        
        json_path = None
        for path in possible_paths:
            if path.exists():
                json_path = path
                break
    
    try:
        if json_path is None:
            print(f"⚠️ labs_database.jsonファイルが見つかりません")
            print(f"📂 確認した場所:")
            for path in possible_paths:
                print(f"   - {path} {'(存在)' if path.exists() else '(不存在)'}")
            return create_fallback_labs_data()
        
        print(f"📂 研究室データ読み込み: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        labs_data = data.get('labs', [])
        
        # データ構造を正規化（featuresを最上位に移動）
        normalized_labs = []
        for lab in labs_data:
            normalized_lab = {
                "id": lab.get("id", "unknown"),
                "name": lab.get("name", "不明な研究室"),
                "professor": lab.get("professor", "不明"),
                "research_area": lab.get("research_area", "情報工学"),
                "specialization": lab.get("specialization", ""),
                "description": lab.get("description", ""),
                "research_fields": lab.get("research_fields", [])
            }
            
            # featuresオブジェクトの内容を最上位に移動
            features = lab.get("features", {})
            for criterion in COMPLETE_EVALUATION_CRITERIA:
                normalized_lab[criterion] = features.get(criterion, 5.5)
            
            # メタデータも保持
            normalized_lab["metadata"] = lab.get("metadata", {})
            
            normalized_labs.append(normalized_lab)
        
        print(f"✅ 研究室データ読み込み完了: {len(normalized_labs)}件")
        print(f"📊 データベースバージョン: {data.get('version', 'unknown')}")
        
        return normalized_labs
        
    except Exception as e:
        print(f"❌ JSONファイル読み込みエラー: {e}")
        print(f"🔄 フォールバックデータを使用します")
        return create_fallback_labs_data()

def create_fallback_labs_data() -> List[Dict[str, Any]]:
    """フォールバック用の研究室データ作成"""
    print("📝 フォールバック研究室データを作成中...")
    
    fallback_labs = [
        {
            "id": "fallback_ai_lab",
            "name": "AI・機械学習研究室（フォールバック）",
            "professor": "システム教授",
            "research_area": "人工知能・機械学習",
            "specialization": "機械学習、深層学習",
            "description": "人工知能と機械学習の研究（システム生成データ）",
            "research_fields": ["人工知能・機械学習"],
            # 13項目の評価基準
            "research_intensity": 8.0,
            "advisor_style": 7.0,
            "team_work": 8.0,
            "workload": 7.5,
            "theory_practice": 6.5,
            "research_field_match": 8.5,
            "skill_development": 8.0,
            "lab_atmosphere": 7.5,
            "flexibility": 7.0,
            "publication_opportunity": 8.0,
            "interdisciplinary": 7.0,
            "communication_style": 7.5,
            "innovation_risk": 8.0,
            "metadata": {
                "faculty_count": 1,
                "student_count": 8,
                "recent_publications": 10,
                "funding_level": "中",
                "equipment_rating": 8
            }
        },
        {
            "id": "fallback_design_lab",
            "name": "デザイン研究室（フォールバック）",
            "professor": "デザイン教授",
            "research_area": "Webデザイン・UI/UX",
            "specialization": "Webデザイン、UI/UXデザイン",
            "description": "Webデザインとユーザビリティの研究（システム生成データ）",
            "research_fields": ["Webデザイン・UI/UX"],
            # 13項目の評価基準
            "research_intensity": 6.5,
            "advisor_style": 8.5,
            "team_work": 8.5,
            "workload": 6.0,
            "theory_practice": 8.0,
            "research_field_match": 8.0,
            "skill_development": 8.5,
            "lab_atmosphere": 9.0,
            "flexibility": 9.0,
            "publication_opportunity": 6.5,
            "interdisciplinary": 8.0,
            "communication_style": 9.0,
            "innovation_risk": 7.5,
            "metadata": {
                "faculty_count": 1,
                "student_count": 12,
                "recent_publications": 5,
                "funding_level": "中",
                "equipment_rating": 7
            }
        },
        {
            "id": "fallback_game_lab",
            "name": "ゲーム開発研究室（フォールバック）",
            "professor": "ゲーム教授",
            "research_area": "ゲーム開発・eスポーツ",
            "specialization": "ゲームプログラミング、eスポーツ",
            "description": "ゲーム開発とeスポーツの研究（システム生成データ）",
            "research_fields": ["ゲーム開発・eスポーツ"],
            # 13項目の評価基準
            "research_intensity": 7.0,
            "advisor_style": 8.0,
            "team_work": 8.5,
            "workload": 7.0,
            "theory_practice": 8.5,
            "research_field_match": 8.5,
            "skill_development": 9.0,
            "lab_atmosphere": 9.0,
            "flexibility": 8.0,
            "publication_opportunity": 6.0,
            "interdisciplinary": 7.0,
            "communication_style": 8.5,
            "innovation_risk": 8.0,
            "metadata": {
                "faculty_count": 1,
                "student_count": 15,
                "recent_publications": 4,
                "funding_level": "中",
                "equipment_rating": 8
            }
        },
        {
            "id": "fallback_security_lab",
            "name": "セキュリティ研究室（フォールバック）",
            "professor": "セキュリティ教授",
            "research_area": "ネットワーク・セキュリティ",
            "specialization": "サイバーセキュリティ、暗号技術",
            "description": "ネットワークセキュリティと暗号技術の研究（システム生成データ）",
            "research_fields": ["ネットワーク・セキュリティ"],
            # 13項目の評価基準
            "research_intensity": 7.5,
            "advisor_style": 6.5,
            "team_work": 6.0,
            "workload": 7.5,
            "theory_practice": 5.5,
            "research_field_match": 8.0,
            "skill_development": 7.5,
            "lab_atmosphere": 6.5,
            "flexibility": 6.0,
            "publication_opportunity": 7.5,
            "interdisciplinary": 6.0,
            "communication_style": 6.5,
            "innovation_risk": 7.0,
            "metadata": {
                "faculty_count": 1,
                "student_count": 8,
                "recent_publications": 8,
                "funding_level": "中",
                "equipment_rating": 7
            }
        },
        {
            "id": "fallback_theory_lab",
            "name": "理論研究室（フォールバック）",
            "professor": "理論教授",
            "research_area": "自然科学・数理",
            "specialization": "計算理論、アルゴリズム",
            "description": "計算理論とアルゴリズムの研究（システム生成データ）",
            "research_fields": ["自然科学・数理"],
            # 13項目の評価基準
            "research_intensity": 9.0,
            "advisor_style": 5.5,
            "team_work": 5.0,
            "workload": 8.5,
            "theory_practice": 3.0,
            "research_field_match": 8.5,
            "skill_development": 7.0,
            "lab_atmosphere": 6.0,
            "flexibility": 5.5,
            "publication_opportunity": 9.0,
            "interdisciplinary": 5.0,
            "communication_style": 5.5,
            "innovation_risk": 7.0,
            "metadata": {
                "faculty_count": 1,
                "student_count": 5,
                "recent_publications": 15,
                "funding_level": "高",
                "equipment_rating": 6
            }
        }
    ]
    
    print(f"✅ フォールバックデータ作成完了: {len(fallback_labs)}件")
    return fallback_labs

def initialize_system():
    """システム初期化（JSON読み込み対応版）"""
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
                population_size=30,
                generations=50, 
                crossover_rate=0.8,
                mutation_rate=0.1,
                elitism_rate=0.1,
                tournament_size=3
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
        
        # 研究室データベース初期化（JSON読み込み）
        labs_data = load_labs_from_json()
        system_state["lab_database"] = labs_data
        system_state["lab_data"] = labs_data  # エイリアス
        
        print(f"✅ 研究室データベース初期化完了: {len(labs_data)}件")
        
        # データ検証
        if labs_data:
            sample_lab = labs_data[0]
            available_criteria = [c for c in COMPLETE_EVALUATION_CRITERIA if c in sample_lab]
            print(f"📊 利用可能な評価基準: {len(available_criteria)}/{len(COMPLETE_EVALUATION_CRITERIA)}項目")
            
            if len(available_criteria) < 10:
                print(f"⚠️ 評価基準が不足しています")
                missing_criteria = [c for c in COMPLETE_EVALUATION_CRITERIA if c not in sample_lab]
                print(f"   不足項目: {missing_criteria}")
        
        system_state["initialized"] = True
        print("🎉 システム初期化完了!")
        
    except Exception as e:
        print(f"❌ システム初期化エラー: {e}")
        traceback.print_exc()
        system_state["initialized"] = False

# ===== 適合度計算関数 =====

def calculate_enhanced_compatibility_from_json(
    student_profile: Dict[str, float], 
    lab_data: Dict[str, Any],
    custom_weights: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """JSONデータ対応の適合度計算"""
    
    print(f"🧮 適合度計算開始: 研究室={lab_data.get('name', 'Unknown')}")
    
    # デフォルト重みを使用
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
                
                # 研究室側の値を取得（JSONデータから直接）
                lab_value = float(lab_data.get(criterion, 5.5))
                
                # 類似度計算（改良版）
                actual_diff = abs(student_value - lab_value)
                
                # より寛容な類似度計算
                if actual_diff <= 1.0:
                    similarity_score = 1.0
                elif actual_diff <= 2.0:
                    similarity_score = 0.9
                elif actual_diff <= 3.0:
                    similarity_score = 0.7
                elif actual_diff <= 4.0:
                    similarity_score = 0.5
                else:
                    similarity_score = max(0.1, 0.5 - (actual_diff - 4.0) * 0.1)
                
                # 重み適用
                weight = weights.get(criterion, 1.0)
                weighted_score = similarity_score * weight
                
                criteria_scores[criterion] = {
                    "student_value": student_value,
                    "lab_value": lab_value,
                    "difference": actual_diff,
                    "similarity_score": similarity_score,
                    "weight": weight,
                    "weighted_score": weighted_score
                }
                
                total_weighted_score += weighted_score
                total_weights += weight
                successful_calculations += 1
                
                print(f"  ✅ {criterion}: 学生={student_value}, 研究室={lab_value}, 類似度={similarity_score:.3f}")
                
        except Exception as e:
            print(f"  ❌ {criterion}: 計算エラー - {e}")
            continue
    
    # 総合スコア計算
    if total_weights > 0:
        final_score = total_weighted_score / total_weights
    else:
        final_score = 0.5  # フォールバック
    
    # 推薦レベル決定
    if final_score >= 0.8:
        recommendation_level = "強く推薦"
    elif final_score >= 0.6:
        recommendation_level = "推薦"
    elif final_score >= 0.4:
        recommendation_level = "検討可能"
    else:
        recommendation_level = "推薦しない"
    
    result = {
        "lab_id": lab_data.get("id", "unknown"),
        "lab_name": lab_data.get("name", "不明な研究室"),
        "overall_score": final_score,
        "criteria_scores": criteria_scores,
        "recommendation_level": recommendation_level,
        "field_match": True,
        "data_completeness": successful_calculations / len(COMPLETE_EVALUATION_CRITERIA),
        "total_criteria_evaluated": successful_calculations,
        "total_weighted_score": total_weighted_score,
        "total_weights": total_weights
    }
    
    print(f"📊 計算完了: 総合スコア={final_score:.3f}, 評価基準={successful_calculations}/{len(COMPLETE_EVALUATION_CRITERIA)}")
    
    return result

def generate_detailed_explanation(compatibility_result: Dict[str, Any]) -> str:
    """詳細な説明生成"""
    
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

# ===== ミドルウェア設定 =====

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

# ===== APIエンドポイント =====

@app.get("/")
async def read_root():
    """ルートエンドポイント - フロントエンド配信またはAPI情報"""
    if os.path.exists("../frontend/build/index.html"):
        return FileResponse("../frontend/build/index.html")
    else:
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
        lab_count > 0
    )
    
    return {
        "status": "healthy" if overall_health else "unhealthy",
        "version": "3.0.0",
        "timestamp": time.time(),
        "system_initialized": system_state["initialized"],
        "modules": modules_status,
        "database": {
            "status": "OK" if lab_count > 0 else "Empty",
            "lab_count": lab_count,
            "evaluation_count": system_state["evaluation_count"]
        },
        "statistics": {
            "api_calls": system_state["api_calls"],
            "error_count": system_state["error_count"],
            "optimization_count": system_state["optimization_count"]
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

@app.get("/api/fields")
async def get_research_fields():
    """研究分野一覧取得"""
    
    research_fields = [
        "人工知能・機械学習",
        "画像・映像処理",
        "ネットワーク・セキュリティ",
        "データベース・情報システム",
        "組込み・IoT",
        "教育・言語学",
        "自然科学・数理",
        "医療情報・ヘルスケア",
        "観光情報・地域システム",
        "経営情報・意思決定支援",
        "音声・音響情報処理",
        "システム運用・情報倫理",
        "Webデザイン・UI/UX",
        "デザイン・視覚表現",
        "映像・アニメーション",
        "コンピュータ音楽・サウンドアート",
        "ゲーム開発・eスポーツ",
        "VR/AR・メディアアート",
        "哲学・人文・環境行動学",
        "スポーツ・体育科学"
    ]
    
    return {
        "research_fields": research_fields,
        "total_count": len(research_fields),
        "categories": {
            "テクノロジー・システム": 12,
            "クリエイティブ": 4,
            "エンターテイメント": 2,
            "人文・社会・体育": 2
        }
    }

@app.get("/api/evaluation-criteria")
async def get_evaluation_criteria():
    """評価基準一覧取得"""
    
    criteria_details = {
        "research_intensity": {
            "name": "研究強度",
            "description": "研究にどれだけ集中的に取り組みたいか",
            "range": "1(軽い研究) ～ 10(集中研究)"
        },
        "advisor_style": {
            "name": "指導スタイル",
            "description": "教授からの指導の受け方の好み",
            "range": "1(厳格指導) ～ 10(自由指導)"
        },
        "team_work": {
            "name": "チームワーク",
            "description": "研究での他者との協働の程度",
            "range": "1(個人研究) ～ 10(チーム研究)"
        },
        "workload": {
            "name": "ワークロード",
            "description": "研究活動の忙しさに対する許容度",
            "range": "1(軽い負荷) ～ 10(重い負荷)"
        },
        "theory_practice": {
            "name": "理論・実践バランス",
            "description": "理論研究と実践的研究のバランス",
            "range": "1(理論重視) ～ 10(実践重視)"
        },
        "research_field_match": {
            "name": "研究分野適合性",
            "description": "自分の興味と研究室の分野の一致度",
            "range": "1(広い分野) ～ 10(専門特化)"
        },
        "skill_development": {
            "name": "スキル開発",
            "description": "専門性と汎用性のバランス",
            "range": "1(専門特化) ～ 10(幅広いスキル)"
        },
        "lab_atmosphere": {
            "name": "研究室雰囲気",
            "description": "研究室の全体的な雰囲気",
            "range": "1(静寂集中) ～ 10(活発議論)"
        },
        "flexibility": {
            "name": "柔軟性",
            "description": "研究時間の自由度",
            "range": "1(固定スケジュール) ～ 10(柔軟スケジュール)"
        },
        "publication_opportunity": {
            "name": "論文発表機会",
            "description": "研究成果の論文化機会",
            "range": "1(少ない機会) ～ 10(豊富な機会)"
        },
        "interdisciplinary": {
            "name": "学際性",
            "description": "他分野との連携の程度",
            "range": "1(単一分野) ～ 10(学際連携)"
        },
        "communication_style": {
            "name": "コミュニケーション",
            "description": "研究室での交流スタイル",
            "range": "1(少人数密接) ～ 10(オープン交流)"
        },
        "innovation_risk": {
            "name": "革新性・リスク許容度",
            "description": "新しい手法への挑戦度",
            "range": "1(安全手法) ～ 10(革新手法)"
        }
    }
    
    return {
        "evaluation_criteria": COMPLETE_EVALUATION_CRITERIA,
        "criteria_details": criteria_details,
        "total_count": len(COMPLETE_EVALUATION_CRITERIA),
        "categories": {
            "basic": COMPLETE_EVALUATION_CRITERIA[:5],
            "extended": COMPLETE_EVALUATION_CRITERIA[5:10],
            "special": COMPLETE_EVALUATION_CRITERIA[10:]
        }
    }

@app.post("/api/evaluate")
async def evaluate_compatibility_frontend(request: Request):
    """フロントエンド互換性エンドポイント - /api/evaluate（JSON対応版）"""
    try:
        print(f"🎯 /api/evaluate エンドポイント実行開始")
        
        if not system_state["initialized"]:
            raise HTTPException(status_code=503, detail="システムが初期化されていません")
        
        body = await request.json()
        print(f"📥 /api/evaluate リクエスト受信: {len(str(body))}文字")
        
        # リクエスト形式の正規化
        if "student_profile" in body:
            profile_data = body["student_profile"]
        elif "evaluation_criteria" in body:
            profile_data = body["evaluation_criteria"]
        else:
            profile_data = body
        
        print(f"📊 プロフィールデータ解析: {len(profile_data)}項目")
        
        # 必須フィールドの確認と補完
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
        
        # 研究室データベースを取得
        lab_database = system_state.get("lab_database", [])
        if not lab_database:
            # フォールバックデータを再読み込み
            print("🔄 研究室データが空です。再読み込みを試行...")
            lab_database = load_labs_from_json()
            system_state["lab_database"] = lab_database
            system_state["lab_data"] = lab_database
        
        if not lab_database:
            raise HTTPException(
                status_code=500,
                detail="研究室データが初期化されていません。labs_database.jsonファイルを確認してください。"
            )
        
        print(f"📊 利用可能研究室数: {len(lab_database)}")
        
        # 各研究室との適合度計算
        results = []
        calculation_errors = []
        
        for i, lab in enumerate(lab_database):
            try:
                print(f"\n--- 研究室 {i+1}: {lab.get('name', 'Unknown')} ---")
                
                # JSON対応の適合度計算関数を使用
                compatibility = calculate_enhanced_compatibility_from_json(profile_data, lab)
                
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
                    "field_match": compatibility.get("field_match", True),
                    "timestamp": datetime.now().isoformat(),
                    # 追加の詳細情報
                    "advisor": lab.get("professor", "不明"),
                    "description": lab.get("description", ""),
                    "research_area": lab.get("research_area", ""),
                    "specialization": lab.get("specialization", ""),
                    "research_fields": lab.get("research_fields", []),
                    "base_score": compatibility.get("base_score", compatibility["overall_score"]),
                    "data_completeness": compatibility.get("data_completeness", 1.0),
                    "metadata": lab.get("metadata", {})
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
                detail="すべての研究室の計算に失敗しました。研究室データの形式を確認してください。"
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
                "processing_time": 0.1,  # 仮の処理時間
                "evaluation_count": system_state.get("evaluation_count", 0) + 1,
                "timestamp": datetime.now().isoformat(),
                "endpoint": "/api/evaluate",
                "calculation_method": "json_enhanced_compatibility_v1",
                "criteria_used": len(COMPLETE_EVALUATION_CRITERIA),
                "data_source": "labs_database.json"
            }
        }
        
        # 評価回数更新
        system_state["evaluation_count"] = system_state.get("evaluation_count", 0) + 1
        
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
                "suggestion": "labs_database.jsonファイルの存在と形式を確認してください"
            }
        )

@app.post("/api/optimize")
async def optimize_frontend(request: Request):
    """遺伝的アルゴリズムによる重み最適化エンドポイント"""
    try:
        print(f"🧬 /api/optimize エンドポイント実行開始")
        
        if not system_state["initialized"]:
            raise HTTPException(status_code=503, detail="システムが初期化されていません")
        
        if not GENETIC_AVAILABLE:
            raise HTTPException(status_code=503, detail="遺伝的アルゴリズムが利用できません")
        
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
        
        # 研究室データベースを取得
        lab_database = system_state.get("lab_database", [])
        if not lab_database:
            raise HTTPException(status_code=500, detail="研究室データが利用できません")
        
        print(f"🧬 遺伝的アルゴリズムによる重み最適化開始...")
        
        # 最適化パラメータの設定
        evolution_config = EvolutionConfig(
            population_size=optimization_params.get("population_size", 30),
            generations=optimization_params.get("generations", 25),
            crossover_rate=optimization_params.get("crossover_rate", 0.8),
            mutation_rate=optimization_params.get("mutation_rate", 0.1),
            elitism_rate=0.1,
            tournament_size=3,
            convergence_threshold=1e-4,
            max_stagnation=10
        )
        
        # 遺伝的アルゴリズムエンジンの作成
        genetic_engine = EvolutionEngine(evolution_config)
        
        # 最適化実行
        start_time = time.time()
        evolution_result = genetic_engine.optimize_weights_for_student(profile_data, lab_database)
        processing_time = time.time() - start_time
        
        # 最適化された重みを辞書形式に変換
        optimized_weights_list = evolution_result.best_individual.chromosome
        optimized_weights_dict = {
            COMPLETE_EVALUATION_CRITERIA[i]: optimized_weights_list[i] 
            for i in range(min(len(COMPLETE_EVALUATION_CRITERIA), len(optimized_weights_list)))
        }
        
        print(f"✅ 遺伝的アルゴリズム最適化完了:")
        print(f"   - 世代数: {evolution_result.generation + 1}")
        print(f"   - 最高適応度: {evolution_result.best_fitness:.4f}")
        print(f"   - 収束: {evolution_result.convergence_achieved}")
        print(f"   - 処理時間: {processing_time:.2f}秒")
        
        # 最適化された重みで研究室評価を実行
        optimized_results = []
        for lab in lab_database:
            compatibility = calculate_enhanced_compatibility_from_json(
                profile_data, lab, optimized_weights_dict
            )
            
            optimized_results.append({
                "lab_id": compatibility["lab_id"],
                "lab_name": compatibility["lab_name"],
                "compatibility_score": compatibility["overall_score"],
                "explanation": generate_detailed_explanation(compatibility),
                "recommendation_level": compatibility["recommendation_level"],
                "advisor": lab.get("professor", "不明"),
                "research_area": lab.get("research_area", ""),
                "detailed_scores": {
                    criterion: data["weighted_score"] if isinstance(data, dict) else data
                    for criterion, data in compatibility["criteria_scores"].items()
                },
                "metadata": lab.get("metadata", {})
            })
        
        optimized_results.sort(key=lambda x: x["compatibility_score"], reverse=True)
        
        # 改善度の計算
        original_scores = []
        optimized_scores = [r["compatibility_score"] for r in optimized_results]
        
        # デフォルト重みでの評価も実行（比較用）
        for lab in lab_database:
            compatibility = calculate_enhanced_compatibility_from_json(profile_data, lab, DEFAULT_CRITERIA_WEIGHTS)
            original_scores.append(compatibility["overall_score"])
        
        avg_original = sum(original_scores) / len(original_scores)
        avg_optimized = sum(optimized_scores) / len(optimized_scores)
        improvement = ((avg_optimized - avg_original) / max(avg_original, 0.001)) * 100
        
        # フロントエンド互換形式でレスポンス
        frontend_response = {
            "optimization_result": {
                "optimized_weights": optimized_weights_dict,
                "original_weights": DEFAULT_CRITERIA_WEIGHTS,
                "fitness_score": evolution_result.best_fitness,
                "generations_used": evolution_result.generation + 1,
                "convergence_achieved": evolution_result.convergence_achieved,
                "improvement_percentage": improvement,
                "avg_original_score": avg_original,
                "avg_optimized_score": avg_optimized,
                "optimization_method": "genetic_algorithm",
                "algorithm_params": {
                    "population_size": evolution_config.population_size,
                    "generations": evolution_config.generations,
                    "mutation_rate": evolution_config.mutation_rate,
                    "crossover_rate": evolution_config.crossover_rate
                }
            },
            "lab_results": optimized_results,
            "results": optimized_results,
            "metadata": {
                "processing_time": processing_time,
                "optimization_count": system_state.get("optimization_count", 0) + 1,
                "timestamp": datetime.now().isoformat(),
                "endpoint": "/api/optimize",
                "data_source": "labs_database.json"
            }
        }
        
        # 最適化回数更新
        system_state["optimization_count"] = system_state.get("optimization_count", 0) + 1
        
        print(f"📤 /api/optimize レスポンス送信")
        print(f"📈 改善度: {improvement:.2f}% (平均スコア: {avg_original:.3f} → {avg_optimized:.3f})")
        
        return JSONResponse(content=frontend_response)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ /api/optimize 予期しないエラー: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "message": "最適化処理で予期しないエラーが発生しました",
                "error": str(e),
                "endpoint": "/api/optimize",
                "suggestion": "遺伝的アルゴリズムモジュールとlabs_database.jsonファイルを確認してください"
            }
        )

# ===== デバッグ・診断エンドポイント =====

@app.get("/api/debug/lab-data-check")
async def debug_lab_data_check():
    """研究室データの詳細確認用デバッグエンドポイント"""
    try:
        lab_database = system_state.get("lab_database", [])
        
        if not lab_database:
            return {
                "status": "error",
                "message": "研究室データが読み込まれていません",
                "lab_count": 0,
                "suggestions": [
                    "labs_database.jsonファイルの存在を確認",
                    "ファイルの形式とエンコーディングを確認",
                    "バックエンドサーバーの再起動"
                ]
            }
        
        # データ分析
        sample_lab = lab_database[0] if lab_database else {}
        available_criteria = [c for c in COMPLETE_EVALUATION_CRITERIA if c in sample_lab]
        missing_criteria = [c for c in COMPLETE_EVALUATION_CRITERIA if c not in sample_lab]
        
        # 各研究室の評価基準完全性チェック
        completeness_stats = []
        for lab in lab_database:
            available = len([c for c in COMPLETE_EVALUATION_CRITERIA if c in lab])
            completeness = available / len(COMPLETE_EVALUATION_CRITERIA)
            completeness_stats.append({
                "name": lab.get("name", "不明"),
                "available_criteria": available,
                "completeness": completeness
            })
        
        return {
            "status": "success",
            "lab_count": len(lab_database),
            "data_analysis": {
                "total_criteria_expected": len(COMPLETE_EVALUATION_CRITERIA),
                "available_criteria_count": len(available_criteria),
                "missing_criteria_count": len(missing_criteria),
                "sample_lab_keys": list(sample_lab.keys()),
                "available_criteria": available_criteria,
                "missing_criteria": missing_criteria,
                "completeness_per_lab": completeness_stats
            },
            "labs_summary": [
                {
                    "id": lab.get("id"),
                    "name": lab.get("name"),
                    "professor": lab.get("professor"),
                    "research_area": lab.get("research_area"),
                    "criteria_count": len([c for c in COMPLETE_EVALUATION_CRITERIA if c in lab])
                }
                for lab in lab_database
            ],
            "system_state": {
                "initialized": system_state.get("initialized", False),
                "evaluation_count": system_state.get("evaluation_count", 0),
                "optimization_count": system_state.get("optimization_count", 0)
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"データ確認中にエラーが発生しました: {str(e)}",
            "error_details": str(e)
        }

@app.get("/api/debug/genetic-test")
async def debug_genetic_test():
    """遺伝的アルゴリズムのテスト用エンドポイント"""
    try:
        if not GENETIC_AVAILABLE:
            return {
                "status": "error",
                "message": "遺伝的アルゴリズムが利用できません",
                "genetic_available": False
            }
        
        # 簡単なテスト用適応度関数
        def test_fitness_function(weights):
            return sum(weights) / len(weights)
        
        # テスト用設定
        test_config = EvolutionConfig(
            population_size=10,
            generations=5,
            crossover_rate=0.8,
            mutation_rate=0.1
        )
        
        # テスト実行
        genetic_engine = EvolutionEngine(test_config)
        start_time = time.time()
        result = genetic_engine.evolve(test_fitness_function, verbose=False)
        processing_time = time.time() - start_time
        
        return {
            "status": "success",
            "message": "遺伝的アルゴリズムのテスト実行が成功しました",
            "genetic_available": True,
            "test_result": {
                "best_fitness": result.best_fitness,
                "generations_used": result.generation + 1,
                "convergence_achieved": result.convergence_achieved,
                "processing_time": processing_time,
                "chromosome_length": len(result.best_individual.chromosome)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"遺伝的アルゴリズムのテストでエラーが発生しました: {str(e)}",
            "error_details": str(e),
            "genetic_available": GENETIC_AVAILABLE
        }

@app.get("/api/debug/test")
async def debug_test():
    """接続・動作テスト用エンドポイント"""
    return {
        "status": "success",
        "message": "APIエンドポイントは正常に動作しています",
        "server_time": datetime.now().isoformat(),
        "system_status": "operational" if system_state["initialized"] else "initializing",
        "modules_available": {
            "fuzzy": FUZZY_AVAILABLE,
            "genetic": GENETIC_AVAILABLE,
            "decision_tree": DECISION_TREE_AVAILABLE,
            "settings": SETTINGS_AVAILABLE
        },
        "test_data": {
            "number": 12345,
            "boolean": True,
            "array": [1, 2, 3, 4, 5],
            "object": {"key": "value", "nested": {"test": "success"}}
        },
        "available_endpoints": {
            "evaluate": "POST /api/evaluate",
            "optimize": "POST /api/optimize",
            "labs": "GET /api/labs", 
            "fields": "GET /api/fields",
            "status": "GET /health"
        }
    }

# OPTIONS プリフライトリクエスト対応
@app.options("/{path:path}")
async def options_handler(path: str):
    """CORS プリフライトリクエスト対応"""
    print(f"🔄 OPTIONS リクエスト: {path}")
    return JSONResponse(
        content={"message": "CORS preflight handled"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization"
        }
    )

# ===== システム初期化実行 =====
initialize_system()

# ===== サーバー起動部分 =====
if __name__ == "__main__":
    print("\n🚀 FastAPI サーバー起動中...")
    print(f"📍 URL: http://localhost:{settings.port}")
    print(f"📚 API文書: http://localhost:{settings.port}/docs")
    print("🔧 システム状況:")
    print(f"  - ファジィ推論: {'✅' if FUZZY_AVAILABLE else '❌'}")
    print(f"  - 遺伝的アルゴリズム: {'✅' if GENETIC_AVAILABLE else '❌'}")
    print(f"  - 決定木: {'✅' if DECISION_TREE_AVAILABLE else '❌'}")
    print(f"  - 研究室データ: {len(system_state['lab_database'])}件")
    print(f"  - 評価基準: {len(COMPLETE_EVALUATION_CRITERIA)}項目")
    
    uvicorn.run(
    app,
    host=settings.host,
    port=settings.port,
    reload=False,  # reloadをFalseに変更
    log_level="info"
)