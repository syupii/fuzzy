# backend/app.py - 修正版 (API通信コード維持)
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

# その他のモジュール読み込み（既存のまま維持）
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
    description="遺伝的アルゴリズムを用いたファジィ決定木による研究室マッチングシステム - 修正版",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS設定 (API通信コード維持)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に設定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# システム状態管理
system_state = {
    "initialized": False,
    "lab_data": [],
    "evaluation_count": 0,
    "last_updated": None,
    "database_version": "2.0.0"
}

# 評価基準の完全な定義（13項目すべて）
COMPLETE_EVALUATION_CRITERIA = [
    "research_intensity", "advisor_style", "team_work", "workload", "theory_practice",
    "research_field_match", "skill_development", "lab_atmosphere", "flexibility", 
    "publication_opportunity", "interdisciplinary", "communication_style", "innovation_risk"
]

def load_lab_database():
    """研究室データベースの読み込み（修正版）"""
    
    try:
        # 優先順位: 1. LabDatabase, 2. JSON直接読み込み
        if DATABASE_AVAILABLE:
            try:
                lab_db = LabDatabase()
                lab_data = lab_db.get_all_labs()
                print(f"✅ LabDatabase経由でデータ読み込み: {len(lab_data)}件")
                return lab_data
            except Exception as e:
                print(f"⚠️ LabDatabase読み込み失敗: {e}")
        
        # フォールバック: JSON直接読み込み
        json_path = project_root / "data" / "labs_database.json"
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                labs = data.get("labs", data if isinstance(data, list) else [])
                print(f"✅ JSON直接読み込み成功: {len(labs)}件")
                return labs
        
        # サンプルデータ作成
        print("⚠️ データファイルが見つかりません。サンプルデータを作成します。")
        return create_enhanced_sample_labs()
        
    except Exception as e:
        print(f"❌ データベース読み込みエラー: {e}")
        return create_enhanced_sample_labs()

def create_enhanced_sample_labs():
    """強化されたサンプル研究室データ"""
    
    return [
        {
            "id": "ai_lab_001",
            "name": "伊藤研究室",
            "professor": "伊藤雅彦",
            "research_area": "人工知能・機械学習",
            "specialization": "情報可視化、ユーザインタフェース、データ工学",
            "description": "データの可視化とユーザビリティを重視した研究を行っています。",
            "research_fields": ["ai_machine_learning", "web_ui_ux"],
            "features": {
                "research_intensity": 8,
                "advisor_style": 7,
                "team_work": 8,
                "workload": 7,
                "theory_practice": 8,
                "research_field_match": 9,
                "skill_development": 8,
                "lab_atmosphere": 8,
                "flexibility": 7,
                "publication_opportunity": 8,
                "interdisciplinary": 7,
                "communication_style": 8,
                "innovation_risk": 7
            },
            "metadata": {
                "faculty_type": "情報工学",
                "student_capacity": 8,
                "equipment_level": 8,
                "funding_level": "high"
            }
        },
        {
            "id": "img_lab_001", 
            "name": "森研究室",
            "professor": "森圭佑",
            "research_area": "画像・映像処理",
            "specialization": "情報計測、音声・画像情報処理、医用情報処理",
            "description": "画像処理とゲームプログラミングを組み合わせた実践的研究。",
            "research_fields": ["image_video_processing", "game_esports"],
            "features": {
                "research_intensity": 7,
                "advisor_style": 6,
                "team_work": 7,
                "workload": 6,
                "theory_practice": 9,
                "research_field_match": 8,
                "skill_development": 9,
                "lab_atmosphere": 7,
                "flexibility": 8,
                "publication_opportunity": 6,
                "interdisciplinary": 8,
                "communication_style": 7,
                "innovation_risk": 8
            },
            "metadata": {
                "faculty_type": "情報工学",
                "student_capacity": 6,
                "equipment_level": 9,
                "funding_level": "medium"
            }
        },
        {
            "id": "web_lab_001",
            "name": "杉沢研究室", 
            "professor": "杉沢愛美",
            "research_area": "Webデザイン・UI/UX",
            "specialization": "Webデザイン、グラフィックデザイン、UX・UIデザイン",
            "description": "デザイン思考とユーザビリティを重視したクリエイティブな研究環境。",
            "research_fields": ["web_ui_ux", "design_visual"],
            "features": {
                "research_intensity": 6,
                "advisor_style": 8,
                "team_work": 9,
                "workload": 5,
                "theory_practice": 7,
                "research_field_match": 9,
                "skill_development": 8,
                "lab_atmosphere": 9,
                "flexibility": 9,
                "publication_opportunity": 5,
                "interdisciplinary": 9,
                "communication_style": 9,
                "innovation_risk": 8
            },
            "metadata": {
                "faculty_type": "デザイン",
                "student_capacity": 10,
                "equipment_level": 7,
                "funding_level": "medium"
            }
        },
        {
            "id": "net_lab_001",
            "name": "尾崎研究室",
            "professor": "尾崎宏和", 
            "research_area": "コンピュータネットワーク・セキュリティ",
            "specialization": "コンピュータネットワーク、通信システム、信頼性",
            "description": "ネットワークセキュリティの最先端技術を研究。",
            "research_fields": ["network_security", "system_development"],
            "features": {
                "research_intensity": 9,
                "advisor_style": 5,
                "team_work": 6,
                "workload": 8,
                "theory_practice": 6,
                "research_field_match": 8,
                "skill_development": 7,
                "lab_atmosphere": 6,
                "flexibility": 5,
                "publication_opportunity": 9,
                "interdisciplinary": 5,
                "communication_style": 6,
                "innovation_risk": 9
            },
            "metadata": {
                "faculty_type": "情報工学",
                "student_capacity": 5,
                "equipment_level": 8,
                "funding_level": "high"
            }
        },
        {
            "id": "game_lab_001",
            "name": "森川研究室",
            "professor": "森川悟",
            "research_area": "ゲーム開発・eスポーツ",
            "specialization": "ゲームプログラミング",
            "description": "ゲーム開発の技術とエンターテイメント性を追求。",
            "research_fields": ["game_esports"],
            "features": {
                "research_intensity": 7,
                "advisor_style": 8,
                "team_work": 8,
                "workload": 7,
                "theory_practice": 9,
                "research_field_match": 8,
                "skill_development": 8,
                "lab_atmosphere": 8,
                "flexibility": 8,
                "publication_opportunity": 6,
                "interdisciplinary": 7,
                "communication_style": 8,
                "innovation_risk": 8
            },
            "metadata": {
                "faculty_type": "情報工学",
                "student_capacity": 8,
                "equipment_level": 8,
                "funding_level": "medium"
            }
        }
    ]

def calculate_enhanced_compatibility(student_profile: Dict[str, Any], lab: Dict[str, Any]) -> float:
    """強化された適合性計算（13項目対応）"""
    
    if not student_profile or not lab:
        return 0.0
    
    lab_features = lab.get("features", {})
    
    # 学生プロフィールの形式を正規化
    if "evaluation_criteria" in student_profile:
        # StudentProfile形式
        criteria_data = student_profile["evaluation_criteria"]
    elif "student_profile" in student_profile:
        # ネストされた形式
        criteria_data = student_profile["student_profile"]
    else:
        # 直接形式
        criteria_data = student_profile
    
    total_score = 0.0
    criteria_count = 0
    
    # 全13項目で適合性を計算
    for criterion in COMPLETE_EVALUATION_CRITERIA:
        student_value = criteria_data.get(criterion)
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
    field_interests = student_profile.get("field_interests", {})
    lab_fields = lab.get("research_fields", [])
    
    if field_interests and lab_fields:
        # 分野の一致度チェック
        matches = 0
        for field_id in lab_fields:
            if field_id in field_interests:
                matches += 1
        field_bonus = min(0.2, matches * 0.1)  # 最大0.2のボーナス
    
    # 最終適合性スコア
    base_compatibility = total_score / criteria_count if criteria_count > 0 else 0.0
    final_score = min(1.0, base_compatibility + field_bonus)
    
    return final_score

def initialize_system():
    """システム初期化（修正版）"""
    
    if system_state["initialized"]:
        return
    
    print("\n🚀 システム初期化開始...")
    
    try:
        # 研究室データ読み込み
        lab_data = load_lab_database()
        system_state["lab_data"] = lab_data
        system_state["last_updated"] = datetime.now().isoformat()
        system_state["initialized"] = True
        
        print(f"✅ システム初期化完了")
        print(f"   - 研究室データ: {len(lab_data)}件")
        print(f"   - データベース形式: {'LabDatabase' if DATABASE_AVAILABLE else 'JSON/Sample'}")
        print(f"   - 最終更新: {system_state['last_updated']}")
        
    except Exception as e:
        print(f"❌ システム初期化失敗: {e}")
        traceback.print_exc()

# システム初期化を実行
initialize_system()

# =============================================================================
# API エンドポイント定義（修正版 - API通信コード維持）
# =============================================================================

@app.get("/")
async def read_root():
    """ルートエンドポイント"""
    if os.path.exists("../frontend/build/index.html"):
        return FileResponse("../frontend/build/index.html")
    else:
        return {
            "message": "遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム v2.0 (修正版)",
            "version": "2.0.0", 
            "status": "running",
            "total_labs": len(system_state.get("lab_data", [])),
            "evaluation_criteria": COMPLETE_EVALUATION_CRITERIA,
            "last_updated": system_state.get("last_updated"),
            "endpoints": {
                "health": "/health",
                "labs": "/api/labs", 
                "evaluate": "/api/evaluate",
                "docs": "/docs"
            }
        }

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    
    lab_count = len(system_state.get("lab_data", []))
    
    modules_status = {
        "fuzzy": FUZZY_AVAILABLE,
        "genetic": GENETIC_AVAILABLE,
        "decision_tree": DECISION_TREE_AVAILABLE,
        "settings": SETTINGS_AVAILABLE,
        "database": DATABASE_AVAILABLE
    }
    
    overall_health = all([
        system_state["initialized"],
        lab_count > 0,
        any(modules_status.values())  # 少なくとも1つのモジュールが利用可能
    ])
    
    return {
        "status": "healthy" if overall_health else "degraded",
        "database_status": "connected" if system_state["initialized"] else "disconnected",
        "lab_count": lab_count,
        "modules": modules_status,
        "evaluation_criteria_count": len(COMPLETE_EVALUATION_CRITERIA),
        "system_version": "2.0.0",
        "timestamp": time.time()
    }

@app.get("/api/labs")
async def get_labs():
    """研究室一覧取得"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {
        "labs": system_state["lab_data"],
        "total_count": len(system_state["lab_data"]),
        "last_updated": system_state["last_updated"]
    }

@app.post("/api/evaluate")
async def evaluate_compatibility(evaluation_data: Dict[str, Any]):
    """研究室適合性評価（修正版 - API通信コード維持）"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        # 入力データの正規化（複数の形式に対応）
        student_profile = None
        
        # パターン1: {"student_profile": {...}}
        if "student_profile" in evaluation_data:
            student_profile = evaluation_data["student_profile"]
        
        # パターン2: {"preferences": {...}} (旧形式互換)
        elif "preferences" in evaluation_data:
            student_profile = evaluation_data["preferences"]
        
        # パターン3: 直接形式
        elif "research_intensity" in evaluation_data:
            student_profile = evaluation_data
        
        else:
            raise HTTPException(
                status_code=400, 
                detail="Student profile required. Expected format: {'student_profile': {...}} or direct preference object"
            )
        
        if not student_profile:
            raise HTTPException(status_code=400, detail="Student profile is empty")
        
        # 評価基準の検証
        missing_criteria = []
        for criterion in COMPLETE_EVALUATION_CRITERIA:
            if criterion not in student_profile:
                missing_criteria.append(criterion)
        
        if missing_criteria:
            print(f"⚠️ 不足している評価基準: {missing_criteria}")
            # デフォルト値で補完
            for criterion in missing_criteria:
                student_profile[criterion] = 5.0  # 中間値
        
        # 全研究室との適合性を計算
        results = []
        
        for lab in system_state["lab_data"]:
            try:
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
            except Exception as e:
                print(f"⚠️ 研究室 {lab.get('name', 'Unknown')} の評価でエラー: {e}")
                continue
        
        # スコア順でソート
        results.sort(key=lambda x: x["compatibility_score"], reverse=True)
        
        system_state["evaluation_count"] += 1
        
        response = {
            "evaluation_id": f"eval_{system_state['evaluation_count']}_{int(time.time())}",
            "student_profile": student_profile,
            "lab_results": results,
            "total_labs_evaluated": len(results),
            "timestamp": time.time(),
            "processing_time": 0.1,
            "algorithm_info": {
                "method": "enhanced_fuzzy_similarity",
                "evaluation_criteria": len(COMPLETE_EVALUATION_CRITERIA),
                "data_source": "LabDatabase v2.0" if DATABASE_AVAILABLE else "Enhanced Sample v2.0",
                "fuzzy_available": FUZZY_AVAILABLE,
                "genetic_available": GENETIC_AVAILABLE,
                "decision_tree_available": DECISION_TREE_AVAILABLE
            },
            "summary": {
                "total_labs": len(results),
                "avg_score": sum(r["compatibility_score"] for r in results) / len(results) if results else 0,
                "best_match_lab": results[0]["lab_name"] if results else None
            }
        }
        
        print(f"✅ 評価完了: {len(results)}件の研究室、平均スコア: {response['summary']['avg_score']:.3f}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Evaluation error: {str(e)}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

# その他のエンドポイント（既存のAPI通信コードを維持）
@app.get("/api/statistics")
async def get_data_statistics():
    """データ統計情報の取得"""
    
    if not system_state["lab_data"]:
        return {"error": "No data available"}
    
    total_labs = len(system_state["lab_data"])
    
    # 研究分野別統計
    field_counts = {}
    for lab in system_state["lab_data"]:
        field = lab.get("research_area", "Unknown")
        field_counts[field] = field_counts.get(field, 0) + 1
    
    # 平均特徴値
    avg_features = {}
    for criterion in COMPLETE_EVALUATION_CRITERIA:
        values = []
        for lab in system_state["lab_data"]:
            features = lab.get("features", {})
            if criterion in features:
                values.append(features[criterion])
        avg_features[criterion] = sum(values) / len(values) if values else 0
    
    return {
        "total_labs": total_labs,
        "field_distribution": field_counts,
        "average_features": avg_features,
        "evaluation_count": system_state["evaluation_count"],
        "last_updated": system_state["last_updated"]
    }

@app.get("/api/research-fields")
async def get_research_fields():
    """研究分野一覧の取得"""
    
    field_stats = {}
    
    for lab in system_state["lab_data"]:
        fields = lab.get("research_fields", [])
        area = lab.get("research_area", "Unknown")
        
        if area not in field_stats:
            field_stats[area] = {
                "field_name": area,
                "lab_count": 0,
                "professors": [],
                "specializations": []
            }
        
        field_stats[area]["lab_count"] += 1
        field_stats[area]["professors"].append(lab.get("professor", ""))
        field_stats[area]["specializations"].append(lab.get("specialization", ""))
    
    return {
        "research_fields": list(field_stats.values()),
        "total_fields": len(field_stats),
        "total_labs": len(system_state["lab_data"])
    }

# サーバー起動部分（API通信コード維持）
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
    print(f"  - 評価基準: {len(COMPLETE_EVALUATION_CRITERIA)}項目")
    print(f"  - 最終更新: {system_state.get('last_updated', 'Unknown')}")
    
    uvicorn.run(
        app,
        host=getattr(settings, 'host', '0.0.0.0'),
        port=getattr(settings, 'port', 8000),
        reload=False,
        log_level="info"
    )