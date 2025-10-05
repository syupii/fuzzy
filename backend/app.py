#!/usr/bin/env python3
# app.py
"""
研究室選択支援システム - パターンA
デフォルトパラメータ + 動的決定木 + 分野マッチング
遺伝的アルゴリズムなし
"""

import os
import sys
import json 
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import Dict, List, Any, Optional
import time

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 設定のインポート
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

# シンプルマッチャーのインポート
try:
    from core.matching.simple_matcher import SimpleMatcher, CompatibilityResult
    MATCHER_AVAILABLE = True
except ImportError as e:
    MATCHER_AVAILABLE = False
    print(f"⚠️ マッチャーが利用できません: {e}")

# デフォルトパラメータのインポート
try:
    from config.default_params import (
        DEFAULT_PARAMS, EVALUATION_CRITERIA, 
        FIELD_NAMES, get_field_name
    )
    PARAMS_AVAILABLE = True
except ImportError as e:
    PARAMS_AVAILABLE = False
    print(f"⚠️ デフォルトパラメータが利用できません: {e}")

# FastAPIアプリケーション初期化
app = FastAPI(
    title="研究室選択支援システム（パターンA）",
    description="デフォルトパラメータ + 動的決定木 + 分野マッチングによる研究室推薦システム",
    version="2.0.0-PatternA",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS設定（より柔軟に）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では適切に設定
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# 静的ファイル配信
if os.path.exists("../frontend/build"):
    app.mount("/static", StaticFiles(directory="../frontend/build/static"), name="static")

# システム状態
system_state = {
    "initialized": False,
    "matcher": None,
    "lab_database": [],
    "evaluation_count": 0,
    "start_time": time.time()
}

# サンプル研究室データ
SAMPLE_LABS = [
    {
        "id": "ai_lab",
        "name": "人工知能研究室",
        "professor": "田中教授",
        "field_id": "ai_ml",
        "description": "機械学習とディープラーニングの研究",
        "research_intensity": 9,
        "advisor_style": 7,
        "team_work": 8,
        "workload": 8,
        "theory_practice": 6,
        "skill_development": 8,
        "lab_atmosphere": 7,
        "flexibility": 6,
        "publication_opportunity": 9,
        "interdisciplinary": 5,
        "communication_style": 7,
        "innovation_focus": 9,
        "students_count": 12,
        "equipment": "最新GPU クラスタ",
        "funding": "高"
    },
    {
        "id": "web_design_lab",
        "name": "Webデザイン研究室",
        "professor": "佐藤教授",
        "field_id": "web_design",
        "description": "UI/UXデザインとWebシステム開発",
        "research_intensity": 6,
        "advisor_style": 8,
        "team_work": 9,
        "workload": 6,
        "theory_practice": 8,
        "skill_development": 7,
        "lab_atmosphere": 9,
        "flexibility": 8,
        "publication_opportunity": 5,
        "interdisciplinary": 7,
        "communication_style": 9,
        "innovation_focus": 7,
        "students_count": 10,
        "equipment": "デザインツール充実",
        "funding": "中"
    },
    {
        "id": "robotics_lab",
        "name": "ロボティクス研究室",
        "professor": "鈴木教授",
        "field_id": "embedded_iot",
        "description": "ロボット工学と組込みシステム",
        "research_intensity": 8,
        "advisor_style": 6,
        "team_work": 9,
        "workload": 9,
        "theory_practice": 5,
        "skill_development": 9,
        "lab_atmosphere": 8,
        "flexibility": 5,
        "publication_opportunity": 7,
        "interdisciplinary": 8,
        "communication_style": 8,
        "innovation_focus": 8,
        "students_count": 15,
        "equipment": "ロボット実験設備",
        "funding": "高"
    },
    {
        "id": "game_lab",
        "name": "ゲーム開発研究室",
        "professor": "高橋教授",
        "field_id": "game_esports",
        "description": "ゲームデザインとeスポーツ研究",
        "research_intensity": 7,
        "advisor_style": 9,
        "team_work": 10,
        "workload": 7,
        "theory_practice": 7,
        "skill_development": 8,
        "lab_atmosphere": 10,
        "flexibility": 9,
        "publication_opportunity": 6,
        "interdisciplinary": 6,
        "communication_style": 10,
        "innovation_focus": 9,
        "students_count": 8,
        "equipment": "ゲーム開発環境",
        "funding": "中"
    },
    {
        "id": "db_lab",
        "name": "データベース研究室",
        "professor": "伊藤教授",
        "field_id": "database_systems",
        "description": "データベースシステムと情報検索",
        "research_intensity": 8,
        "advisor_style": 6,
        "team_work": 6,
        "workload": 7,
        "theory_practice": 4,
        "skill_development": 7,
        "lab_atmosphere": 6,
        "flexibility": 6,
        "publication_opportunity": 8,
        "interdisciplinary": 5,
        "communication_style": 6,
        "innovation_focus": 7,
        "students_count": 11,
        "equipment": "高性能サーバー",
        "funding": "中"
    }
]


def load_labs_from_json():
    """JSONファイルから研究室データを読み込む"""
    json_path = os.path.join(project_root, "data", "labs_database.json")
    
    # JSONファイルが存在するか確認
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                labs_data = json.load(f)
            print(f"✅ 研究室データ読み込み成功: {json_path}")
            
            # データが配列かオブジェクトか確認
            if isinstance(labs_data, dict) and "labs" in labs_data:
                return labs_data["labs"]
            elif isinstance(labs_data, list):
                return labs_data
            else:
                print(f"⚠️ 予期しないデータ形式: {type(labs_data)}")
                return SAMPLE_LABS
                
        except json.JSONDecodeError as e:
            print(f"❌ JSONパースエラー: {e}")
            return SAMPLE_LABS
        except Exception as e:
            print(f"❌ ファイル読み込みエラー: {e}")
            return SAMPLE_LABS
    else:
        print(f"⚠️ 研究室データファイルが見つかりません: {json_path}")
        print(f"   サンプルデータを使用します")
        return SAMPLE_LABS


def initialize_system():
    """システム初期化"""
    global system_state
    
    print("🚀 システム初期化開始...")
    
    try:
        # シンプルマッチャー初期化
        if MATCHER_AVAILABLE:
            system_state["matcher"] = SimpleMatcher()
            print("✅ シンプルマッチャー初期化完了")
        else:
            print("⚠️ マッチャーが利用できません")
        
        # 研究室データベース初期化（JSONから読み込み）
        labs_data = load_labs_from_json()
        system_state["lab_database"] = labs_data
        print(f"✅ 研究室データベース初期化完了: {len(labs_data)}件")
        
        # データの内容を確認
        if labs_data:
            first_lab = labs_data[0]
            print(f"   サンプル: {first_lab.get('name', 'N/A')} ({first_lab.get('field_id', 'N/A')})")
        
        system_state["initialized"] = True
        print("🎉 システム初期化完了!")
        
    except Exception as e:
        print(f"❌ システム初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        system_state["initialized"] = False


# ==================== API エンドポイント ====================

@app.get("/")
async def read_root():
    """ルートエンドポイント"""
    if os.path.exists("../frontend/build/index.html"):
        return FileResponse("../frontend/build/index.html")
    
    return {
        "message": "研究室選択支援システム（パターンA）",
        "version": "2.0.0-PatternA",
        "status": "running",
        "pattern": "A",
        "features": {
            "genetic_algorithm": False,
            "default_parameters": True,
            "dynamic_decision_tree": True,
            "field_matching": True,
            "criteria_count": 13
        },
        "endpoints": {
            "health": "/health",
            "labs": "/api/labs",
            "evaluate": "/api/evaluate",
            "criteria": "/api/criteria",
            "fields": "/api/fields",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    
    lab_count = len(system_state.get("lab_database", []))
    uptime = time.time() - system_state["start_time"]
    
    return {
        "status": "healthy" if system_state["initialized"] else "unhealthy",
        "version": "2.0.0-PatternA",
        "pattern": "A (遺伝的アルゴリズムなし)",
        "timestamp": time.time(),
        "uptime_seconds": uptime,
        "system_initialized": system_state["initialized"],
        "modules": {
            "matcher": MATCHER_AVAILABLE,
            "params": PARAMS_AVAILABLE,
            "settings": SETTINGS_AVAILABLE
        },
        "database": {
            "status": "OK" if lab_count > 0 else "Empty",
            "lab_count": lab_count,
            "evaluation_count": system_state["evaluation_count"]
        },
        "features": {
            "default_params": True,
            "dynamic_tree": True,
            "field_matching": True,
            "genetic_optimization": False
        }
    }


@app.get("/api/criteria")
async def get_criteria():
    """評価基準一覧取得"""
    
    if not PARAMS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Parameters not available")
    
    criteria_info = []
    
    for criterion in EVALUATION_CRITERIA:
        info = {
            "id": criterion,
            "name": criterion.replace("_", " ").title(),
            "description": _get_criterion_description(criterion),
            "range": "1-10",
            "importance": "high" if criterion in ["research_intensity", "publication_opportunity"] else "normal"
        }
        criteria_info.append(info)
    
    return {
        "criteria": criteria_info,
        "total_count": len(EVALUATION_CRITERIA),
        "basic_count": 12,
        "has_field_match": True
    }


@app.get("/api/fields")
async def get_fields():
    """分野一覧取得"""
    
    if not PARAMS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Parameters not available")
    
    fields_list = []
    
    for field_id, field_name in FIELD_NAMES.items():
        fields_list.append({
            "id": field_id,
            "name": field_name
        })
    
    return {
        "fields": fields_list,
        "total_count": len(FIELD_NAMES)
    }


@app.get("/api/labs")
async def get_labs():
    """研究室一覧取得"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    labs = system_state["lab_database"]
    
    # 分野名を追加
    for lab in labs:
        lab["field_name"] = get_field_name(lab.get("field_id", "unknown"))
    
    return {
        "labs": labs,
        "total_count": len(labs),
        "last_updated": time.time()
    }


@app.get("/api/labs/{lab_id}")
async def get_lab_detail(lab_id: str):
    """特定研究室の詳細取得"""
    
    labs = system_state["lab_database"]
    lab = next((lab for lab in labs if lab["id"] == lab_id), None)
    
    if not lab:
        raise HTTPException(status_code=404, detail="Lab not found")
    
    # 分野名を追加
    lab["field_name"] = get_field_name(lab.get("field_id", "unknown"))
    
    return lab


@app.post("/api/evaluate")
async def evaluate_compatibility(student_profile: Dict[str, Any]):
    """学生プロファイルに基づく研究室適合度評価"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not MATCHER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Matcher not available")
    
    try:
        # デバッグ: 受信データをログ出力
        print("\n" + "="*60)
        print("📥 受信した学生プロファイル:")
        print(f"データキー: {list(student_profile.keys())}")
        print(f"research_field_match: {student_profile.get('research_field_match', '未設定')}")
        print(f"field_interests: {student_profile.get('field_interests', '未設定')}")
        print("="*60 + "\n")
        
        # 入力検証（柔軟版）
        _validate_student_profile(student_profile)
        
        # マッチャー取得
        matcher = system_state["matcher"]
        
        # 各研究室との適合度計算
        results = []
        
        for lab in system_state["lab_database"]:
            # 適合度計算
            try:
                compatibility_result = matcher.calculate_compatibility(
                    student_profile, lab
                )
                
                # 結果整形
                lab_result = {
                    "lab_id": lab["id"],
                    "lab_name": lab["name"],
                    "professor": lab["professor"],
                    "field_id": lab["field_id"],
                    "field_name": get_field_name(lab["field_id"]),
                    
                    # スコア
                    "overall_compatibility": compatibility_result.total_compatibility,
                    "basic_score": compatibility_result.basic_score,
                    "field_score": compatibility_result.field_score,
                    
                    # 比重
                    "field_weight": compatibility_result.field_weight_alpha,
                    "basic_weight": compatibility_result.basic_weight_beta,
                    
                    # 詳細
                    "criteria_scores": compatibility_result.criteria_scores,
                    "field_detail": compatibility_result.field_detail,
                    "tree_layers": compatibility_result.tree_layers,
                    
                    # 説明
                    "explanation": compatibility_result.explanation,
                    "recommendation": compatibility_result.recommendation,
                    
                    # 研究室情報
                    "students_count": lab.get("students_count", 0),
                    "equipment": lab.get("equipment", ""),
                    "funding": lab.get("funding", "")
                }
                
                results.append(lab_result)
                
            except Exception as e:
                print(f"⚠️ 研究室 {lab['name']} の評価中にエラー: {e}")
                continue
        
        if not results:
            raise HTTPException(
                status_code=500,
                detail="すべての研究室の評価に失敗しました"
            )
        
        # 適合度でソート
        results.sort(key=lambda x: x["overall_compatibility"], reverse=True)
        
        # 評価回数増加
        system_state["evaluation_count"] += 1
        
        print(f"✅ 評価完了: {len(results)}研究室")
        
        return {
            "student_profile": student_profile,
            "evaluation_results": results,
            "total_labs_evaluated": len(results),
            "evaluation_timestamp": time.time(),
            "system_info": {
                "pattern": "A",
                "matcher_type": "simple",
                "uses_genetic_algorithm": False,
                "uses_default_params": True,
                "evaluation_count": system_state["evaluation_count"]
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"\n❌ 評価エラー:\n{error_detail}")
        raise HTTPException(
            status_code=500, 
            detail=f"Evaluation error: {str(e)}"
        )


@app.post("/api/explain/{lab_id}")
async def explain_recommendation(lab_id: str, student_profile: Dict[str, Any]):
    """推薦結果の詳細説明"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # 対象研究室を取得
    lab = next((lab for lab in system_state["lab_database"] if lab["id"] == lab_id), None)
    if not lab:
        raise HTTPException(status_code=404, detail="Lab not found")
    
    # 入力検証
    _validate_student_profile(student_profile)
    
    # 適合度計算
    matcher = system_state["matcher"]
    result = matcher.calculate_compatibility(student_profile, lab)
    
    # 詳細分析
    top_criteria = sorted(
        result.criteria_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    
    bottom_criteria = sorted(
        result.criteria_scores.items(),
        key=lambda x: x[1]
    )[:3]
    
    return {
        "lab_id": lab_id,
        "lab_name": lab["name"],
        "overall_compatibility": result.total_compatibility,
        "recommendation": result.recommendation,
        "explanation": result.explanation,
        
        "score_breakdown": {
            "basic_score": result.basic_score,
            "field_score": result.field_score,
            "field_weight": result.field_weight_alpha,
            "basic_weight": result.basic_weight_beta
        },
        
        "strengths": [
            {
                "criterion": criterion,
                "score": score,
                "description": _get_criterion_description(criterion)
            }
            for criterion, score in top_criteria
        ],
        
        "concerns": [
            {
                "criterion": criterion,
                "score": score,
                "description": _get_criterion_description(criterion)
            }
            for criterion, score in bottom_criteria
        ],
        
        "field_analysis": result.field_detail,
        "decision_tree_layers": result.tree_layers
    }


@app.post("/api/debug/echo")
async def debug_echo(data: Dict[str, Any]):
    """デバッグ用: 受信データをそのまま返す"""
    print("\n" + "="*60)
    print("🔍 デバッグ: 受信データ")
    print(f"キー: {list(data.keys())}")
    print(f"データ: {data}")
    print("="*60 + "\n")
    
    return {
        "received": data,
        "keys": list(data.keys()),
        "data_types": {k: str(type(v).__name__) for k, v in data.items()},
        "timestamp": time.time()
    }


@app.get("/api/test/sample-profile")
async def get_sample_profile():
    """テスト用: サンプル学生プロファイルを返す"""
    return {
        "research_intensity": 9,
        "advisor_style": 7,
        "team_work": 5,
        "workload": 8,
        "theory_practice": 6,
        "skill_development": 7,
        "lab_atmosphere": 6,
        "flexibility": 5,
        "publication_opportunity": 9,
        "interdisciplinary": 4,
        "communication_style": 6,
        "innovation_focus": 8,
        "research_field_match": 7,
        "field_interests": {
            "ai_ml": 10,
            "image_processing": 7
        }
    }


@app.get("/api/debug/labs")
async def debug_labs():
    """デバッグ用: 研究室データの詳細を返す"""
    labs = system_state.get("lab_database", [])
    
    if not labs:
        return {
            "status": "error",
            "message": "研究室データがありません",
            "lab_count": 0
        }
    
    # 最初の研究室の詳細構造を確認
    first_lab = labs[0] if labs else {}
    
    return {
        "status": "ok",
        "lab_count": len(labs),
        "first_lab_keys": list(first_lab.keys()) if first_lab else [],
        "first_lab_sample": first_lab,
        "all_lab_names": [lab.get("name", "N/A") for lab in labs],
        "all_lab_fields": [lab.get("field_id", "N/A") for lab in labs],
        "data_source": "labs_database.json" if os.path.exists(os.path.join(project_root, "data", "labs_database.json")) else "SAMPLE_LABS"
    }


# ==================== ヘルパー関数 ====================

def _validate_student_profile(profile: Dict[str, Any]):
    """学生プロファイルの検証（柔軟版）"""
    
    # research_field_matchのデフォルト値設定
    if "research_field_match" not in profile:
        profile["research_field_match"] = 5.0  # デフォルト：バランス型
        print(f"⚠️ research_field_match が未設定のため、デフォルト値 5.0 を使用")
    
    # field_interestsのデフォルト値設定
    if "field_interests" not in profile or not profile["field_interests"]:
        # 最初の研究室の分野をデフォルトとして使用
        if system_state["lab_database"]:
            first_lab_field = system_state["lab_database"][0].get("field_id", "ai_ml")
            profile["field_interests"] = {first_lab_field: 5.0}
            print(f"⚠️ field_interests が未設定のため、デフォルト値を使用: {first_lab_field}")
    
    # 基本12項目のデフォルト値設定
    from config.default_params import BASIC_CRITERIA
    for criterion in BASIC_CRITERIA:
        if criterion not in profile:
            profile[criterion] = 5.0  # デフォルト：中間値
    
    # 優先度のデフォルト値設定
    for criterion in BASIC_CRITERIA:
        priority_key = f"{criterion}_priority"
        if priority_key not in profile:
            profile[priority_key] = 5.0  # デフォルト：中間値
    
    # データ型の正規化
    try:
        # 数値型に変換
        profile["research_field_match"] = float(profile["research_field_match"])
        
        # field_interestsの正規化
        if isinstance(profile["field_interests"], dict):
            normalized_interests = {}
            for field_id, interest in profile["field_interests"].items():
                normalized_interests[str(field_id)] = float(interest)
            profile["field_interests"] = normalized_interests
        
        # 基本項目の正規化
        for criterion in BASIC_CRITERIA:
            if criterion in profile:
                profile[criterion] = float(profile[criterion])
            priority_key = f"{criterion}_priority"
            if priority_key in profile:
                profile[priority_key] = float(profile[priority_key])
        
    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"データ型エラー: {str(e)}"
        )
    
    # 値の範囲チェック（1-10）
    for key, value in profile.items():
        if key == "field_interests":
            continue
        if isinstance(value, (int, float)):
            if value < 1 or value > 10:
                print(f"⚠️ {key} の値が範囲外 ({value}) のため、5.0 に調整")
                profile[key] = 5.0


def _get_criterion_description(criterion: str) -> str:
    """評価基準の説明を取得"""
    
    descriptions = {
        "research_intensity": "研究にどれだけ集中的に取り組みたいか",
        "advisor_style": "指導教員からの指導スタイルの好み",
        "team_work": "他者との協働の程度",
        "workload": "研究活動の忙しさに対する許容度",
        "theory_practice": "理論研究と実践研究のバランス",
        "research_field_match": "分野と項目のどちらを重視するか",
        "skill_development": "専門性と汎用性のバランス",
        "lab_atmosphere": "研究室の全体的な雰囲気",
        "flexibility": "研究時間の自由度",
        "publication_opportunity": "研究成果の論文化機会",
        "interdisciplinary": "他分野との連携の程度",
        "communication_style": "研究室での交流スタイル",
        "innovation_focus": "革新的な研究への注力度"
    }
    
    return descriptions.get(criterion, "")


# ==================== サーバー起動 ====================

if __name__ == "__main__":
    # システム初期化
    initialize_system()
    
    print("\n🚀 FastAPI サーバー起動中...")
    print(f"📍 URL: http://localhost:{settings.port if SETTINGS_AVAILABLE else 8000}")
    print(f"📚 API文書: http://localhost:{settings.port if SETTINGS_AVAILABLE else 8000}/docs")
    print("🔧 システム情報:")
    print(f"  - パターン: A（遺伝的アルゴリズムなし）")
    print(f"  - マッチャー: {'✅' if MATCHER_AVAILABLE else '❌'}")
    print(f"  - デフォルトパラメータ: {'✅' if PARAMS_AVAILABLE else '❌'}")
    print(f"  - 研究室データ: {len(SAMPLE_LABS)}件")
    print("\n" + "="*60)
    print("サーバーを停止するには Ctrl+C を押してください")
    print("="*60 + "\n")
    
    # サーバー起動（ブロッキングモード）
    uvicorn.run(
        app,
        host=settings.host if SETTINGS_AVAILABLE else "0.0.0.0",
        port=settings.port if SETTINGS_AVAILABLE else 8000,
        log_level="info"
    )