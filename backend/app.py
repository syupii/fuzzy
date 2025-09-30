#!/usr/bin/env python3
"""
遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム
FastAPI メインアプリケーション - 完全統合版 v3.0
13項目評価基準 + 19分野対応 + labs_database.json連携
"""

import os
import sys
import json
import uvicorn
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import time
import numpy as np

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# ===============================
# 定数定義
# ===============================

# 13項目評価基準（完全版）
EVALUATION_CRITERIA = [
    # 基本5項目
    "research_intensity", "advisor_style", "team_work", 
    "workload", "theory_practice",
    # 拡張5項目
    "research_field_match", "skill_development", "lab_atmosphere",
    "flexibility", "publication_opportunity",
    # 特殊3項目
    "interdisciplinary", "communication_style", "innovation_risk"
]

# 評価基準の説明
CRITERIA_DESCRIPTIONS = {
    "research_intensity": "研究強度（1:軽い研究 ～ 10:集中研究）",
    "advisor_style": "指導スタイル（1:厳格指導 ～ 10:自由指導）",
    "team_work": "チームワーク（1:個人研究 ～ 10:チーム研究）",
    "workload": "ワークロード（1:軽い負荷 ～ 10:重い負荷）",
    "theory_practice": "理論・実践バランス（1:理論重視 ～ 10:実践重視）",
    "research_field_match": "研究分野適合性（1:広い分野 ～ 10:専門特化）",
    "skill_development": "スキル開発（1:専門特化 ～ 10:幅広いスキル）",
    "lab_atmosphere": "研究室雰囲気（1:静寂集中 ～ 10:活発議論）",
    "flexibility": "柔軟性（1:固定スケジュール ～ 10:柔軟スケジュール）",
    "publication_opportunity": "論文発表機会（1:少ない機会 ～ 10:豊富な機会）",
    "interdisciplinary": "学際性（1:単一分野 ～ 10:学際連携）",
    "communication_style": "コミュニケーション（1:少人数密接 ～ 10:オープン交流）",
    "innovation_risk": "革新性とリスク（1:安定志向 ～ 10:挑戦志向）"
}

# 19研究分野
RESEARCH_FIELDS = [
    # テクノロジー・システム（11分野）
    "人工知能・機械学習", "画像・映像処理", "ネットワーク・セキュリティ",
    "データベース・情報システム", "組込み・IoT", "教育・言語学",
    "自然科学・数理", "観光情報・地域システム", "経営情報・意思決定支援",
    "音声・音響情報処理", "システム運用・情報倫理",
    # クリエイティブ（4分野）
    "Webデザイン・UI/UX", "デザイン・視覚表現", "映像・アニメーション",
    "コンピュータ音楽・サウンドアート",
    # エンターテイメント（2分野）
    "ゲーム開発・eスポーツ", "VR/AR・メディアアート",
    # 人文・社会・体育（2分野）
    "哲学・人文・環境行動学", "スポーツ・体育科学"
]

# ===============================
# システムモジュールインポート
# ===============================

# 設定読み込み
try:
    from config.settings import settings
    SETTINGS_AVAILABLE = True
except ImportError:
    SETTINGS_AVAILABLE = False
    class FallbackSettings:
        app_name = "Lab Matching System"
        version = "3.0.0"
        host = "0.0.0.0"
        port = 8000
        debug = True
    settings = FallbackSettings()

# ファジィ推論
try:
    from core.fuzzy import SimpleFuzzyInferenceEngine
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("⚠️ ファジィ推論モジュールが利用できません")

# 遺伝的アルゴリズム
try:
    from core.genetic import EvolutionEngine, EvolutionConfig
    GENETIC_AVAILABLE = True
except ImportError:
    GENETIC_AVAILABLE = False
    print("⚠️ 遺伝的アルゴリズムモジュールが利用できません")

# 決定木
try:
    from core.decision_tree import FuzzyDecisionTree
    DECISION_TREE_AVAILABLE = True
except ImportError:
    DECISION_TREE_AVAILABLE = False
    print("⚠️ ファジィ決定木モジュールが利用できません")

# ===============================
# FastAPI アプリケーション初期化
# ===============================

app = FastAPI(
    title="研究室選択支援システム v3.0 完全版",
    description="遺伝的アルゴリズムを用いたファジィ決定木による研究室マッチングシステム（13項目+19分野対応）",
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

# ===============================
# グローバル状態管理
# ===============================

system_state = {
    "initialized": False,
    "fuzzy_engine": None,
    "genetic_engine": None,
    "decision_tree": None,
    "lab_database": [],
    "evaluation_count": 0,
    "optimization_count": 0,
    "last_updated": None
}

# ===============================
# データベース読み込み関数
# ===============================

def load_labs_database() -> List[Dict[str, Any]]:
    """labs_database.jsonから研究室データを読み込み"""
    
    # データベースファイルのパス候補
    possible_paths = [
        Path(project_root) / "data" / "labs_database.json",
        Path(project_root) / "labs_database.json",
        Path(project_root).parent / "data" / "labs_database.json",
    ]
    
    for db_path in possible_paths:
        if db_path.exists():
            try:
                with open(db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    labs = data.get('labs', [])
                    
                    print(f"✅ データベース読み込み成功: {db_path}")
                    print(f"   研究室数: {len(labs)}件")
                    print(f"   バージョン: {data.get('version', 'N/A')}")
                    
                    # データ検証
                    validated_labs = []
                    for lab in labs:
                        if validate_lab_data(lab):
                            validated_labs.append(lab)
                        else:
                            print(f"⚠️ 無効なデータをスキップ: {lab.get('name', 'Unknown')}")
                    
                    return validated_labs
                    
            except Exception as e:
                print(f"❌ データベース読み込みエラー ({db_path}): {e}")
                continue
    
    print("⚠️ labs_database.json が見つかりません。サンプルデータを使用します。")
    return create_sample_labs()

def validate_lab_data(lab: Dict[str, Any]) -> bool:
    """研究室データの検証"""
    
    required_fields = ['id', 'name', 'professor', 'features']
    
    # 必須フィールドチェック
    for field in required_fields:
        if field not in lab:
            return False
    
    # featuresの検証
    features = lab.get('features', {})
    for criterion in EVALUATION_CRITERIA:
        if criterion not in features:
            # デフォルト値を設定
            features[criterion] = 5.0
    
    return True

def create_sample_labs() -> List[Dict[str, Any]]:
    """サンプル研究室データ作成（フォールバック用）"""
    
    return [
        {
            "id": "sample_ai_lab",
            "name": "AI研究室（サンプル）",
            "professor": "サンプル教授",
            "research_area": "人工知能・機械学習",
            "research_fields": ["人工知能・機械学習"],
            "description": "サンプル研究室",
            "features": {
                "research_intensity": 8.0,
                "advisor_style": 7.0,
                "team_work": 7.5,
                "workload": 8.0,
                "theory_practice": 6.0,
                "research_field_match": 8.5,
                "skill_development": 8.0,
                "lab_atmosphere": 7.0,
                "flexibility": 6.0,
                "publication_opportunity": 8.5,
                "interdisciplinary": 7.0,
                "communication_style": 7.0,
                "innovation_risk": 7.5
            },
            "metadata": {
                "faculty_count": 1,
                "student_count": 8,
                "recent_publications": 15
            }
        }
    ]

# ===============================
# 適合度計算エンジン
# ===============================

class CompatibilityCalculator:
    """研究室適合度計算クラス"""
    
    def __init__(self, fuzzy_engine=None):
        self.fuzzy_engine = fuzzy_engine
    
    def calculate_compatibility(
        self, 
        student_profile: Dict[str, Any], 
        lab: Dict[str, Any],
        priorities: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        適合度を計算
        
        Args:
            student_profile: 学生プロファイル
            lab: 研究室データ
            priorities: 各基準の優先度（オプション）
            
        Returns:
            (総合適合度, 詳細スコア)
        """
        
        lab_features = lab.get('features', {})
        feature_scores = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        # デフォルト優先度
        if priorities is None:
            priorities = {criterion: 1.0 for criterion in EVALUATION_CRITERIA}
        
        # 各基準の適合度計算
        for criterion in EVALUATION_CRITERIA:
            student_val = student_profile.get(criterion, 5.0)
            lab_val = lab_features.get(criterion, 5.0)
            
            # 正規化（0-10 → 0-1）
            student_norm = student_val / 10.0
            lab_norm = lab_val / 10.0
            
            # 距離ベースの適合度（0-1, 1が最高）
            distance = abs(student_norm - lab_norm)
            match_score = 1.0 - distance
            
            # ファジィメンバーシップ適用（オプション）
            if self.fuzzy_engine:
                match_score = self._apply_fuzzy_membership(match_score)
            
            feature_scores[criterion] = match_score
            
            # 重み付き合計
            weight = priorities.get(criterion, 1.0)
            weighted_sum += match_score * weight
            total_weight += weight
        
        # 総合適合度
        overall_compatibility = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # 分野マッチングボーナス
        field_bonus = self._calculate_field_bonus(student_profile, lab)
        overall_compatibility = min(1.0, overall_compatibility + field_bonus)
        
        return overall_compatibility, feature_scores
    
    def _apply_fuzzy_membership(self, score: float) -> float:
        """ファジィメンバーシップ関数を適用"""
        
        # 高適合度を強調する非線形変換
        if score > 0.8:
            return 0.8 + (score - 0.8) * 2.0  # 0.8以上を強調
        elif score < 0.3:
            return score * 0.5  # 低スコアをペナルティ
        else:
            return score
    
    def _calculate_field_bonus(
        self, 
        student_profile: Dict[str, Any], 
        lab: Dict[str, Any]
    ) -> float:
        """研究分野マッチングボーナス計算"""
        
        student_interests = student_profile.get('field_interests', {})
        lab_fields = lab.get('research_fields', [])
        
        if not student_interests or not lab_fields:
            return 0.0
        
        max_interest = 0.0
        for field in lab_fields:
            interest_score = student_interests.get(field, 0.0) / 10.0
            max_interest = max(max_interest, interest_score)
        
        return max_interest * 0.1  # 最大10%のボーナス

# ===============================
# システム初期化
# ===============================

def initialize_system():
    """システム全体を初期化"""
    
    global system_state
    
    print("\n" + "="*70)
    print("🚀 研究室選択支援システム v3.0 初期化開始")
    print("="*70)
    
    try:
        # 研究室データベース読み込み
        print("\n📚 研究室データベース読み込み中...")
        system_state["lab_database"] = load_labs_database()
        print(f"✅ 読み込み完了: {len(system_state['lab_database'])}件")
        
        # ファジィ推論エンジン初期化
        if FUZZY_AVAILABLE:
            print("\n🔮 ファジィ推論エンジン初期化中...")
            system_state["fuzzy_engine"] = SimpleFuzzyInferenceEngine(
                EVALUATION_CRITERIA, 
                "compatibility"
            )
            print("✅ ファジィ推論エンジン初期化完了")
        
        # 遺伝的アルゴリズム初期化
        if GENETIC_AVAILABLE:
            print("\n🧬 遺伝的アルゴリズムエンジン初期化中...")
            evolution_config = EvolutionConfig(
                population_size=30,
                generations=50,
                elite_size=3,
                crossover_rate=0.8,
                mutation_rate=0.15
            )
            system_state["genetic_engine"] = EvolutionEngine(evolution_config)
            print("✅ 遺伝的アルゴリズム初期化完了")
        
        # 適合度計算エンジン初期化
        system_state["calculator"] = CompatibilityCalculator(
            fuzzy_engine=system_state.get("fuzzy_engine")
        )
        print("✅ 適合度計算エンジン初期化完了")
        
        system_state["initialized"] = True
        system_state["last_updated"] = datetime.now().isoformat()
        
        print("\n" + "="*70)
        print("🎉 システム初期化完了！")
        print("="*70)
        print(f"\n📊 システム情報:")
        print(f"  - 研究室数: {len(system_state['lab_database'])}件")
        print(f"  - 評価基準: {len(EVALUATION_CRITERIA)}項目")
        print(f"  - 研究分野: {len(RESEARCH_FIELDS)}分野")
        print(f"  - ファジィ推論: {'✅' if FUZZY_AVAILABLE else '❌'}")
        print(f"  - 遺伝的アルゴリズム: {'✅' if GENETIC_AVAILABLE else '❌'}")
        print(f"  - 決定木: {'✅' if DECISION_TREE_AVAILABLE else '❌'}")
        print()
        
    except Exception as e:
        print(f"\n❌ システム初期化エラー: {e}")
        import traceback
        traceback.print_exc()
        system_state["initialized"] = False

# システム初期化実行
initialize_system()

# ===============================
# APIエンドポイント
# ===============================

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "message": "研究室選択支援システム v3.0",
        "version": "3.0.0",
        "status": "running" if system_state["initialized"] else "error",
        "features": {
            "evaluation_criteria": f"{len(EVALUATION_CRITERIA)}項目対応",
            "research_fields": f"{len(RESEARCH_FIELDS)}分野対応",
            "labs_count": len(system_state["lab_database"]),
            "fuzzy_inference": FUZZY_AVAILABLE,
            "genetic_algorithm": GENETIC_AVAILABLE,
            "decision_tree": DECISION_TREE_AVAILABLE
        },
        "endpoints": {
            "health": "/health",
            "criteria": "/api/criteria",
            "fields": "/api/fields",
            "labs": "/api/labs",
            "evaluate": "/api/evaluate",
            "optimize": "/api/optimize",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    
    return {
        "status": "healthy" if system_state["initialized"] else "unhealthy",
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "initialized": system_state["initialized"],
            "labs_count": len(system_state["lab_database"]),
            "evaluation_criteria": len(EVALUATION_CRITERIA),
            "research_fields": len(RESEARCH_FIELDS),
            "evaluation_count": system_state["evaluation_count"],
            "optimization_count": system_state["optimization_count"],
            "last_updated": system_state.get("last_updated")
        },
        "modules": {
            "fuzzy": FUZZY_AVAILABLE,
            "genetic": GENETIC_AVAILABLE,
            "decision_tree": DECISION_TREE_AVAILABLE,
            "settings": SETTINGS_AVAILABLE
        }
    }

@app.get("/api/criteria")
async def get_criteria():
    """評価基準一覧取得"""
    
    criteria_list = []
    for criterion in EVALUATION_CRITERIA:
        criteria_list.append({
            "name": criterion,
            "description": CRITERIA_DESCRIPTIONS.get(criterion, ""),
            "range": "1-10"
        })
    
    return {
        "total_count": len(EVALUATION_CRITERIA),
        "categories": {
            "basic": EVALUATION_CRITERIA[:5],
            "extended": EVALUATION_CRITERIA[5:10],
            "special": EVALUATION_CRITERIA[10:]
        },
        "criteria": criteria_list
    }

@app.get("/api/fields")
async def get_fields():
    """研究分野一覧取得"""
    
    return {
        "total_count": len(RESEARCH_FIELDS),
        "fields": RESEARCH_FIELDS,
        "categories": {
            "technology": RESEARCH_FIELDS[:11],
            "creative": RESEARCH_FIELDS[11:15],
            "entertainment": RESEARCH_FIELDS[15:17],
            "humanities": RESEARCH_FIELDS[17:]
        }
    }

@app.get("/api/labs")
async def get_labs():
    """研究室一覧取得"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {
        "total_count": len(system_state["lab_database"]),
        "labs": system_state["lab_database"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/labs/{lab_id}")
async def get_lab_detail(lab_id: str):
    """特定研究室の詳細取得"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    lab = next((lab for lab in system_state["lab_database"] if lab["id"] == lab_id), None)
    
    if not lab:
        raise HTTPException(status_code=404, detail=f"Lab not found: {lab_id}")
    
    return lab

@app.post("/api/evaluate")
async def evaluate_compatibility(request: Dict[str, Any]):
    """
    研究室適合度評価
    
    Request Body:
    {
        "student_profile": {
            "research_intensity": 8.0,
            "advisor_style": 7.0,
            ... (13項目)
        },
        "priorities": {  // オプション
            "research_intensity": 0.9,
            "team_work": 0.7
        },
        "field_interests": {  // オプション
            "人工知能・機械学習": 9.0,
            "画像・映像処理": 7.0
        }
    }
    """
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        student_profile = request.get("student_profile", {})
        priorities = request.get("priorities")
        field_interests = request.get("field_interests")
        
        # 分野興味度をプロファイルに追加
        if field_interests:
            student_profile["field_interests"] = field_interests
        
        # 入力検証
        missing_criteria = [c for c in EVALUATION_CRITERIA if c not in student_profile]
        if missing_criteria:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required criteria: {', '.join(missing_criteria)}"
            )
        
        # 全研究室との適合度計算
        calculator = system_state["calculator"]
        results = []
        
        for lab in system_state["lab_database"]:
            compatibility, feature_scores = calculator.calculate_compatibility(
                student_profile, lab, priorities
            )
            
            results.append({
                "lab_id": lab["id"],
                "lab_name": lab["name"],
                "professor": lab["professor"],
                "research_area": lab.get("research_area", ""),
                "research_fields": lab.get("research_fields", []),
                "overall_compatibility": round(compatibility, 4),
                "feature_scores": {k: round(v, 4) for k, v in feature_scores.items()},
                "confidence": round(min(1.0, compatibility + 0.05), 4),
                "recommendation_level": get_recommendation_level(compatibility)
            })
        
        # 適合度でソート
        results.sort(key=lambda x: x["overall_compatibility"], reverse=True)
        
        # 評価カウント更新
        system_state["evaluation_count"] += 1
        
        return {
            "status": "success",
            "student_profile_summary": {
                "criteria_count": len(student_profile),
                "priorities_applied": priorities is not None,
                "field_interests_applied": field_interests is not None
            },
            "evaluation_results": results,
            "total_labs_evaluated": len(results),
            "top_matches": results[:5],
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "evaluation_count": system_state["evaluation_count"],
                "fuzzy_applied": FUZZY_AVAILABLE
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

@app.post("/api/optimize")
async def optimize_matching(request: Dict[str, Any]):
    """
    遺伝的アルゴリズムによる最適化
    
    Request Body:
    {
        "training_mode": "balanced",  // "balanced", "custom"
        "num_samples": 100,
        "generations": 50,
        "population_size": 30
    }
    """
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    if not GENETIC_AVAILABLE:
        raise HTTPException(
            status_code=501, 
            detail="Genetic algorithm module not available"
        )
    
    try:
        from utils.training_data import TrainingDataGenerator
        
        training_mode = request.get("training_mode", "balanced")
        num_samples = request.get("num_samples", 100)
        generations = request.get("generations", 50)
        population_size = request.get("population_size", 30)
        
        print(f"\n🧬 遺伝的アルゴリズム最適化開始...")
        print(f"  モード: {training_mode}")
        print(f"  サンプル数: {num_samples}")
        print(f"  世代数: {generations}")
        
        # 訓練データ生成
        if training_mode == "balanced":
            samples_per_type = num_samples // 6
            training_data = TrainingDataGenerator.generate_balanced_dataset(samples_per_type)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown training mode: {training_mode}")
        
        # 進化設定
        evolution_config = EvolutionConfig(
            generations=generations,
            population_size=population_size,
            elite_size=max(2, population_size // 10),
            verbose=True
        )
        
        # 進化エンジン実行
        engine = EvolutionEngine(evolution_config)
        best_individual, evolution_history = engine.optimize(training_data)
        optimal_tree = engine.get_optimized_decision_tree()
        
        # 結果保存
        system_state["optimized_tree"] = optimal_tree
        system_state["optimization_history"] = evolution_history
        system_state["optimization_count"] += 1
        
        return {
            "status": "success",
            "optimal_tree": {
                "level1_feature": optimal_tree["level1_feature"],
                "level2_features": optimal_tree["level2_features"],
                "fitness": round(optimal_tree["fitness"], 4)
            },
            "evolution_summary": {
                "total_generations": len(evolution_history),
                "final_fitness": round(optimal_tree["fitness"], 4),
                "convergence_generation": next(
                    (i for i, h in enumerate(evolution_history) 
                     if abs(h["best_fitness"] - optimal_tree["fitness"]) < 0.001),
                    len(evolution_history)
                )
            },
            "optimization_metadata": {
                "timestamp": datetime.now().isoformat(),
                "optimization_count": system_state["optimization_count"],
                "training_samples": len(training_data)
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

# ===============================
# ユーティリティ関数
# ===============================

def get_recommendation_level(compatibility: float) -> str:
    """適合度から推薦レベルを判定"""
    
    if compatibility >= 0.85:
        return "強く推薦"
    elif compatibility >= 0.75:
        return "推薦"
    elif compatibility >= 0.65:
        return "検討推奨"
    elif compatibility >= 0.50:
        return "要検討"
    else:
        return "適合度低"

# ===============================
# サーバー起動
# ===============================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 研究室選択支援システム v3.0 起動")
    print("="*70)
    print(f"\n📍 URL: http://localhost:{getattr(settings, 'port', 8000)}")
    print(f"📚 API文書: http://localhost:{getattr(settings, 'port', 8000)}/docs")
    print(f"📖 ReDoc: http://localhost:{getattr(settings, 'port', 8000)}/redoc")
    print("\n" + "="*70 + "\n")
    
    uvicorn.run(
        "app:app",  # インポート文字列として指定
        host=getattr(settings, 'host', '0.0.0.0'),
        port=getattr(settings, 'port', 8000),
        reload=getattr(settings, 'debug', True),
        log_level="info"
    )