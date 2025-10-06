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

# グローバル変数（システム状態）
system_state = {
    "initialized": False,
    "matcher": None,
    "lab_database": [],
    "evaluation_count": 0
}


def load_lab_database() -> List[Dict[str, Any]]:
    """研究室データベースの読み込み"""
    
    # データベースファイルのパス候補
    db_paths = [
        "data/labs_database.json",
        "backend/data/labs_database.json",
        "../data/labs_database.json"
    ]
    
    # JSONファイルから読み込みを試行
    for db_path in db_paths:
        if os.path.exists(db_path):
            try:
                with open(db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    labs = data.get('labs', [])
                    if labs:
                        print(f"✅ 研究室データベース読み込み成功: {db_path} ({len(labs)}件)")
                        return labs
            except Exception as e:
                print(f"⚠️ {db_path} の読み込みエラー: {e}")
    
    # フォールバック: サンプルデータ
    print("⚠️ データベースファイルが見つかりません。サンプルデータを使用します。")
    return get_sample_labs()


def get_sample_labs() -> List[Dict[str, Any]]:
    """サンプル研究室データ"""
    return [
        {
            "id": "lab_ai_ml",
            "name": "機械学習研究室",
            "professor": "田中教授",
            "research_area": "人工知能・機械学習",
            "specialization": "深層学習、強化学習",
            "research_fields": ["人工知能・機械学習"],
            "field_id": "ai_machine_learning",
            "description": "機械学習とディープラーニングの研究を行っています",
            "features": {
                "research_intensity": 9.0,
                "advisor_style": 7.0,
                "team_work": 8.0,
                "workload": 8.0,
                "theory_practice": 6.0,
                "research_field_match": 8.5,
                "skill_development": 8.5,
                "lab_atmosphere": 7.5,
                "flexibility": 6.5,
                "publication_opportunity": 9.0,
                "interdisciplinary": 7.0,
                "communication_style": 7.0
            },
            "metadata": {
                "faculty_count": 1,
                "student_count": 8,
                "recent_publications": 15,
                "funding_level": "高",
                "equipment_rating": 9
            }
        },
        {
            "id": "lab_image_processing",
            "name": "画像認識研究室",
            "professor": "佐藤教授",
            "research_area": "画像・映像処理",
            "specialization": "コンピュータビジョン、パターン認識",
            "research_fields": ["画像・映像処理"],
            "field_id": "image_video_processing",
            "description": "コンピュータビジョンとパターン認識の研究",
            "features": {
                "research_intensity": 8.5,
                "advisor_style": 6.5,
                "team_work": 7.5,
                "workload": 8.0,
                "theory_practice": 6.5,
                "research_field_match": 8.0,
                "skill_development": 8.0,
                "lab_atmosphere": 7.0,
                "flexibility": 7.0,
                "publication_opportunity": 8.5,
                "interdisciplinary": 6.5,
                "communication_style": 6.5
            },
            "metadata": {
                "faculty_count": 1,
                "student_count": 7,
                "recent_publications": 12,
                "funding_level": "高",
                "equipment_rating": 8
            }
        },
        {
            "id": "lab_web_design",
            "name": "Webデザイン研究室",
            "professor": "鈴木教授",
            "research_area": "Webデザイン・UI/UX",
            "specialization": "ユーザーインターフェース設計",
            "research_fields": ["Webデザイン・UI/UX"],
            "field_id": "web_ui_ux",
            "description": "Webデザインとユーザー体験の研究",
            "features": {
                "research_intensity": 6.5,
                "advisor_style": 8.0,
                "team_work": 9.0,
                "workload": 6.0,
                "theory_practice": 8.0,
                "research_field_match": 7.5,
                "skill_development": 8.5,
                "lab_atmosphere": 8.5,
                "flexibility": 8.5,
                "publication_opportunity": 6.0,
                "interdisciplinary": 8.0,
                "communication_style": 9.0
            },
            "metadata": {
                "faculty_count": 1,
                "student_count": 10,
                "recent_publications": 8,
                "funding_level": "中",
                "equipment_rating": 7
            }
        }
    ]


def normalize_lab_data(lab: Dict[str, Any]) -> Dict[str, Any]:
    """
    研究室データの正規化
    professor/advisor, features, field_idなどの整合性を確保
    """
    normalized = lab.copy()
    
    # professor/advisorフィールドの統一
    if 'advisor' in normalized and 'professor' not in normalized:
        normalized['professor'] = normalized['advisor']
    elif 'professor' in normalized and 'advisor' not in normalized:
        normalized['advisor'] = normalized['professor']
    
    # featuresフィールドの確保
    if 'features' not in normalized:
        normalized['features'] = {}
    
    # 基本12項目をfeaturesから直接アクセス可能に（後方互換性）
    features = normalized['features']
    for criterion in ['research_intensity', 'advisor_style', 'team_work', 'workload', 
                     'theory_practice', 'research_field_match', 'skill_development',
                     'lab_atmosphere', 'flexibility', 'publication_opportunity',
                     'interdisciplinary', 'communication_style']:
        if criterion in features and criterion not in normalized:
            normalized[criterion] = features[criterion]
        elif criterion in normalized and criterion not in features:
            features[criterion] = normalized[criterion]
    
    # field_idの確保
    if 'field_id' not in normalized:
        # research_fields[0]やresearch_areaから推測
        if 'research_fields' in normalized and normalized['research_fields']:
            # 分野名からfield_idを推測（簡易版）
            field_name = normalized['research_fields'][0]
            normalized['field_id'] = convert_field_name_to_id(field_name)
        elif 'research_area' in normalized:
            normalized['field_id'] = convert_field_name_to_id(normalized['research_area'])
        else:
            normalized['field_id'] = 'general'
    
    return normalized


def convert_field_name_to_id(field_name: str) -> str:
    """分野名からfield_idに変換（簡易版）"""
    mapping = {
        '人工知能・機械学習': 'ai_machine_learning',
        '画像・映像処理': 'image_video_processing',
        'ネットワーク・セキュリティ': 'network_security',
        'データベース・情報システム': 'database_systems',
        '組込み・IoT': 'embedded_iot',
        '教育・言語学': 'education_linguistics',
        '自然科学・数理': 'natural_science_math',
        '観光情報・地域システム': 'tourism_systems',
        '経営情報・意思決定支援': 'management_systems',
        '音声・音響情報処理': 'audio_processing',
        'システム運用・情報倫理': 'system_operations',
        'Webデザイン・UI/UX': 'web_ui_ux',
        'デザイン・視覚表現': 'design_visual',
        '映像・アニメーション': 'video_animation',
        'コンピュータ音楽・サウンドアート': 'computer_music',
        'ゲーム開発・eスポーツ': 'game_esports',
        'VR/AR・メディアアート': 'vr_ar_media_art',
        '哲学・人文・環境行動学': 'philosophy_humanities',
        'スポーツ・体育科学': 'sports_science'
    }
    return mapping.get(field_name, 'general')


def initialize_system():
    """システム初期化"""
    global system_state
    
    print("\n" + "="*60)
    print("🚀 研究室選択支援システム（パターンA）初期化開始")
    print("="*60)
    
    try:
        # 研究室データベース読み込み
        labs = load_lab_database()
        
        # データの正規化
        normalized_labs = [normalize_lab_data(lab) for lab in labs]
        system_state["lab_database"] = normalized_labs
        
        print(f"✅ 研究室データベース: {len(normalized_labs)}件")
        
        # マッチャー初期化
        if MATCHER_AVAILABLE:
            system_state["matcher"] = SimpleMatcher()
            print("✅ シンプルマッチャー初期化完了")
        else:
            print("⚠️ マッチャーが利用できません")
        
        system_state["initialized"] = True
        print("🎉 システム初期化完了!")
        print("="*60 + "\n")
        
    except Exception as e:
        import traceback
        print(f"❌ システム初期化エラー: {e}")
        print(traceback.format_exc())
        system_state["initialized"] = False


# システム初期化
initialize_system()


# ===== APIエンドポイント =====

@app.get("/")
async def read_root():
    """ルートエンドポイント"""
    if os.path.exists("../frontend/build/index.html"):
        return FileResponse("../frontend/build/index.html")
    else:
        return {
            "message": "研究室選択支援システム（パターンA）",
            "version": "2.0.0-PatternA",
            "status": "running",
            "endpoints": {
                "health": "/health",
                "labs": "/api/labs",
                "evaluate": "/api/evaluate",
                "explain": "/api/explain/{lab_id}",
                "docs": "/docs"
            }
        }


@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    
    lab_count = len(system_state.get("lab_database", []))
    
    return {
        "status": "healthy" if system_state["initialized"] else "unhealthy",
        "version": "2.0.0-PatternA",
        "timestamp": time.time(),
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
        "pattern": "A (Simple Matcher + Default Params + Dynamic Decision Tree)"
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


def _validate_student_profile(student_profile: Dict[str, Any]) -> None:
    """学生プロファイルの検証（旧形式：evaluation_criteria構造）"""
    
    # evaluation_criteriaの検証
    if "evaluation_criteria" not in student_profile:
        raise HTTPException(
            status_code=400,
            detail="evaluation_criteria is required"
        )
    
    criteria = student_profile["evaluation_criteria"]
    required_criteria = [
        "research_intensity", "advisor_style", "team_work", "workload", "theory_practice",
        "research_field_match", "skill_development", "lab_atmosphere", "flexibility",
        "publication_opportunity", "interdisciplinary", "communication_style"
    ]
    
    for criterion in required_criteria:
        if criterion not in criteria:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required criterion: {criterion}"
            )


def _validate_student_profile_frontend(student_profile: Dict[str, Any]) -> None:
    """学生プロファイルの検証（フロントエンド形式）"""
    
    required_criteria = [
        "research_intensity", "advisor_style", "team_work", "workload", "theory_practice",
        "research_field_match", "skill_development", "lab_atmosphere", "flexibility",
        "publication_opportunity", "interdisciplinary", "communication_style"
    ]
    
    for criterion in required_criteria:
        if criterion not in student_profile:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required criterion: {criterion}"
            )


@app.post("/api/evaluate")
async def evaluate_compatibility(request_data: Dict[str, Any]):
    """学生プロファイルに基づく研究室適合度評価"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not MATCHER_AVAILABLE or not system_state["matcher"]:
        raise HTTPException(status_code=503, detail="Matcher not available")
    
    try:
        print("\n" + "="*60)
        print("📥 受信したリクエストデータ:")
        print(f"データキー: {list(request_data.keys())}")
        
        # student_profileを取り出す
        student_profile = request_data.get("student_profile")
        if not student_profile:
            raise HTTPException(
                status_code=400,
                detail="student_profile is required in request body"
            )
        
        print(f"学生プロファイルキー: {list(student_profile.keys())}")
        
        # 入力検証（フロントエンドの構造に合わせる）
        _validate_student_profile_frontend(student_profile)
        
        # SimpleMatcher用に変換
        field_interests = student_profile.get("field_interests", {})
        priorities = student_profile.get("priorities", {})
        
        # research_field_matchの取得
        research_field_match = student_profile.get("research_field_match", 5.0)
        print(f"research_field_match: {research_field_match}")
        print(f"field_interests: {field_interests}")
        print(f"priorities: {priorities}")
        print("="*60)
        
        # マッチャー用学生データ作成
        matcher_student = {}
        
        # 基本12項目をコピー（直接アクセス）
        for criterion in ["research_intensity", "advisor_style", "team_work", "workload",
                         "theory_practice", "research_field_match", "skill_development",
                         "lab_atmosphere", "flexibility", "publication_opportunity",
                         "interdisciplinary", "communication_style"]:
            matcher_student[criterion] = student_profile.get(criterion, 5.0)
        
        # 分野興味をコピー
        matcher_student["field_interests"] = field_interests if field_interests else {}
        
        # 優先度を設定
        for criterion in ["research_intensity", "advisor_style", "team_work", "workload",
                         "theory_practice", "research_field_match", "skill_development",
                         "lab_atmosphere", "flexibility", "publication_opportunity",
                         "interdisciplinary", "communication_style"]:
            priority_key = f"{criterion}_priority"
            # prioritiesオブジェクトから取得、なければデフォルト値5
            if priorities and criterion in priorities:
                matcher_student[priority_key] = priorities[criterion]
            else:
                matcher_student[priority_key] = 5
        
        # 各研究室との適合度計算
        matcher = system_state["matcher"]
        results = []
        
        for lab in system_state["lab_database"]:
            try:
                # SimpleMatcher用に研究室データを変換
                matcher_lab = lab.copy()
                
                # featuresから基本項目を展開
                if 'features' in matcher_lab:
                    for k, v in matcher_lab['features'].items():
                        if k not in matcher_lab:
                            matcher_lab[k] = v
                
                # 適合度計算
                result = matcher.calculate_compatibility(matcher_student, matcher_lab)
                
                # 結果を辞書形式に変換
                lab_result = {
                    "lab_id": lab["id"],
                    "lab_name": lab["name"],
                    "name": lab["name"],  # 互換性のため
                    "professor": lab.get("professor", lab.get("advisor", "不明")),
                    "advisor": lab.get("professor", lab.get("advisor", "不明")),  # 互換性のため
                    "research_area": lab.get("research_area", ""),
                    "description": lab.get("description", ""),
                    "overall_compatibility": result.total_compatibility,
                    "final_score": result.total_compatibility,  # ResultsList.tsxが期待
                    "compatibility_score": result.total_compatibility,  # 互換性のため
                    "basic_score": result.basic_score,
                    "field_score": result.field_score,
                    "field_weight": result.field_weight_alpha,
                    "basic_weight": result.basic_weight_beta,
                    "criteria_scores": result.criteria_scores,
                    "feature_scores": result.criteria_scores,  # ResultsList.tsxが期待
                    "field_detail": result.field_detail,
                    "recommendation": result.recommendation,
                    "recommendation_level": result.recommendation,  # 互換性のため
                    "explanation": result.explanation,
                    "tree_layers": result.tree_layers,
                    "confidence": min(1.0, result.total_compatibility + 0.05)  # 信頼度
                }
                
                results.append(lab_result)
                
            except Exception as e:
                print(f"⚠️ 研究室 {lab.get('name', 'unknown')} の評価中にエラー: {e}")
                import traceback
                print(traceback.format_exc())
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
        
        # 統計計算
        scores = [r["overall_compatibility"] for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        high_compatibility_count = sum(1 for s in scores if s >= 0.7)
        
        # フロントエンドが期待する形式で返す
        return {
            "student_profile": request_data,
            "lab_results": results,  # フロントエンドはlab_resultsを期待
            "evaluation_results": results,  # 互換性のため両方返す
            "total_labs_evaluated": len(results),
            "evaluation_timestamp": time.time(),
            "summary": {
                "total_labs": len(results),
                "avg_score": avg_score,
                "avg_compatibility": avg_score,  # App.tsxが期待
                "best_match_lab": results[0]["lab_name"] if results else None,
                "high_compatibility_count": high_compatibility_count  # App.tsxが期待
            },
            "metadata": {
                "priorities_used": bool(priorities),
                "field_interests_count": len(field_interests),
                "processing_time": 0.1  # ダミー値
            },
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
async def explain_recommendation(lab_id: str, request_data: Dict[str, Any]):
    """推薦結果の詳細説明"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # student_profileを取り出す
    student_profile = request_data.get("student_profile")
    if not student_profile:
        raise HTTPException(
            status_code=400,
            detail="student_profile is required in request body"
        )
    
    # 対象研究室を取得
    lab = next((lab for lab in system_state["lab_database"] if lab["id"] == lab_id), None)
    if not lab:
        raise HTTPException(status_code=404, detail="Lab not found")
    
    # 入力検証
    _validate_student_profile_frontend(student_profile)
    
    # 適合度計算（/api/evaluateと同じロジック）
    field_interests = student_profile.get("field_interests", {})
    priorities = student_profile.get("priorities", {})
    
    matcher_student = {}
    for criterion in ["research_intensity", "advisor_style", "team_work", "workload",
                     "theory_practice", "research_field_match", "skill_development",
                     "lab_atmosphere", "flexibility", "publication_opportunity",
                     "interdisciplinary", "communication_style"]:
        matcher_student[criterion] = student_profile.get(criterion, 5.0)
        priority_key = f"{criterion}_priority"
        if priorities and criterion in priorities:
            matcher_student[priority_key] = priorities[criterion]
        else:
            matcher_student[priority_key] = 5
    
    matcher_student["field_interests"] = field_interests if field_interests else {}
    
    matcher_lab = lab.copy()
    if 'features' in matcher_lab:
        for k, v in matcher_lab['features'].items():
            if k not in matcher_lab:
                matcher_lab[k] = v
    
    matcher = system_state["matcher"]
    result = matcher.calculate_compatibility(matcher_student, matcher_lab)
    
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
        "professor": lab.get("professor", lab.get("advisor", "不明")),
        "overall_compatibility": result.total_compatibility,
        "basic_score": result.basic_score,
        "field_score": result.field_score,
        "field_weight": result.field_weight_alpha,
        "basic_weight": result.basic_weight_beta,
        "recommendation": result.recommendation,
        "explanation": result.explanation,
        "detailed_analysis": {
            "top_matching_criteria": [
                {
                    "criterion": criterion,
                    "score": score,
                    "interpretation": _interpret_criterion_score(criterion, score)
                }
                for criterion, score in top_criteria
            ],
            "areas_for_consideration": [
                {
                    "criterion": criterion,
                    "score": score,
                    "interpretation": _interpret_criterion_score(criterion, score)
                }
                for criterion, score in bottom_criteria
            ],
            "field_analysis": result.field_detail,
            "decision_tree_path": result.tree_layers
        }
    }


def _interpret_criterion_score(criterion: str, score: float) -> str:
    """評価項目スコアの解釈"""
    
    criterion_names = {
        "research_intensity": "研究強度",
        "advisor_style": "指導スタイル",
        "team_work": "チームワーク",
        "workload": "ワークロード",
        "theory_practice": "理論・実践バランス",
        "research_field_match": "研究分野適合性",
        "skill_development": "スキル開発",
        "lab_atmosphere": "研究室雰囲気",
        "flexibility": "柔軟性",
        "publication_opportunity": "論文発表機会",
        "interdisciplinary": "学際性",
        "communication_style": "コミュニケーション"
    }
    
    name = criterion_names.get(criterion, criterion)
    
    if score >= 0.8:
        return f"{name}が非常によく一致しています"
    elif score >= 0.6:
        return f"{name}が適度に一致しています"
    elif score >= 0.4:
        return f"{name}に若干の違いがあります"
    else:
        return f"{name}に大きな違いがあります"


# サーバー起動
if __name__ == "__main__":
    print("\n🚀 FastAPI サーバー起動中...")
    print(f"📍 URL: http://localhost:{settings.port}")
    print(f"📚 API文書: http://localhost:{settings.port}/docs")
    print("🔧 システム状況:")
    print(f"  - マッチャー: {'✅' if MATCHER_AVAILABLE else '❌'}")
    print(f"  - パラメータ: {'✅' if PARAMS_AVAILABLE else '❌'}")
    print(f"  - 研究室データ: {len(system_state['lab_database'])}件")
    print(f"  - パターン: A (シンプル + デフォルト + 動的決定木)")
    
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=False,  # python app.py で起動する場合はreloadを無効化
        log_level="info"
    )