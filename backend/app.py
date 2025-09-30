#!/usr/bin/env python3
"""
遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム
FastAPI メインアプリケーション - 完全版 v3.0
12項目評価基準 + 19分野興味度対応
"""

import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List, Any
import time
import numpy as np
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# ===============================
# 定数定義
# ===============================

# 12項目完全対応評価基準
COMPLETE_CRITERIA = [
    # 基本5項目
    "research_intensity", "advisor_style", "team_work", 
    "workload", "theory_practice",
    # 拡張5項目
    "research_field_match", "skill_development", "lab_atmosphere",
    "flexibility", "publication_opportunity",
    # 特殊2項目
    "interdisciplinary", "communication_style"
]

# 研究分野リスト（19分野）
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

# FastAPIアプリケーション初期化
app = FastAPI(
    title="研究室選択支援システム v3.0",
    description="遺伝的アルゴリズムを用いたファジィ決定木による研究室マッチングシステム（12項目+19分野対応版）",
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
# ファジィメンバーシップ関数
# ===============================

class FuzzyMembershipFunctions:
    """ファジィメンバーシップ関数"""
    
    @staticmethod
    def triangular(x: float, a: float, b: float, c: float) -> float:
        """三角型メンバーシップ関数"""
        if x <= a or x >= c:
            return 0.0
        elif x == b:
            return 1.0
        elif x < b:
            return (x - a) / (b - a)
        else:
            return (c - x) / (c - b)
    
    @staticmethod
    def gaussian(x: float, mean: float, sigma: float) -> float:
        """ガウス型メンバーシップ関数"""
        return np.exp(-0.5 * ((x - mean) / sigma) ** 2)

# ===============================
# 多階層ファジィ決定木
# ===============================

class MultiLevelFuzzyDecisionTree:
    """多階層ファジィ決定木（3レベル階層）"""
    
    def __init__(self):
        self.fuzzy = FuzzyMembershipFunctions()
    
    def classify(self, profile: Dict[str, float]) -> Dict[str, Any]:
        """学生プロファイルを分類"""
        research_intensity = profile.get("research_intensity", 0.5)
        
        # Level 1: research_intensityでファジィ分類
        low_degree = self.fuzzy.triangular(research_intensity, 0, 0, 0.5)
        medium_degree = self.fuzzy.triangular(research_intensity, 0.3, 0.5, 0.7)
        high_degree = self.fuzzy.triangular(research_intensity, 0.5, 1.0, 1.0)
        
        level1_memberships = {
            "low": low_degree,
            "medium": medium_degree,
            "high": high_degree
        }
        
        primary_branch = max(level1_memberships, key=level1_memberships.get)
        
        # Level 2: 細分化
        if primary_branch == "high":
            level2_feature = "team_work"
            level2_value = profile.get(level2_feature, 0.5)
            team_oriented = self.fuzzy.triangular(level2_value, 0.5, 0.7, 1.0)
            individual = self.fuzzy.triangular(level2_value, 0.0, 0.3, 0.7)
            cluster_memberships = {
                "team_oriented": team_oriented,
                "individual_focused": individual
            }
        elif primary_branch == "medium":
            level2_feature = "flexibility"
            level2_value = profile.get(level2_feature, 0.5)
            flexible = self.fuzzy.triangular(level2_value, 0.4, 0.6, 1.0)
            structured = self.fuzzy.triangular(level2_value, 0.0, 0.4, 0.6)
            cluster_memberships = {
                "flexible_style": flexible,
                "structured_style": structured
            }
        else:  # low
            level2_feature = "lab_atmosphere"
            level2_value = profile.get(level2_feature, 0.5)
            active = self.fuzzy.triangular(level2_value, 0.4, 0.6, 1.0)
            quiet = self.fuzzy.triangular(level2_value, 0.0, 0.4, 0.6)
            cluster_memberships = {
                "active_atmosphere": active,
                "quiet_atmosphere": quiet
            }
        
        final_cluster = max(cluster_memberships, key=cluster_memberships.get)
        
        return {
            "primary_cluster": f"{primary_branch}_{final_cluster}",
            "level1_branch": primary_branch,
            "level2_cluster": final_cluster,
            "level1_memberships": level1_memberships,
            "level2_memberships": cluster_memberships,
            "classification_path": [
                f"Level1: {primary_branch} (μ={level1_memberships[primary_branch]:.3f})",
                f"Level2: {final_cluster} (μ={cluster_memberships[final_cluster]:.3f})"
            ]
        }

# ===============================
# 拡張適合度計算エンジン
# ===============================

class EnhancedCompatibilityEngine:
    """12項目+19分野対応の拡張適合度計算エンジン"""
    
    def __init__(self):
        self.fuzzy = FuzzyMembershipFunctions()
        self.decision_tree = MultiLevelFuzzyDecisionTree()
        
        # 項目の重要度
        self.importance_weights = {
            # 基本項目（42%）
            "research_intensity": 0.13,
            "advisor_style": 0.09,
            "team_work": 0.09,
            "workload": 0.06,
            "theory_practice": 0.05,
            # 拡張項目（36%）
            "research_field_match": 0.11,
            "skill_development": 0.07,
            "lab_atmosphere": 0.07,
            "flexibility": 0.06,
            "publication_opportunity": 0.05,
            # 特殊項目（22%）
            "interdisciplinary": 0.11,
            "communication_style": 0.11
        }
    
    def calculate_compatibility(self, student: Dict[str, float], 
                               lab: Dict[str, float]) -> Dict[str, Any]:
        """12項目 + 分野興味度を考慮した詳細適合度計算"""
        feature_scores = {}
        weighted_sum = 0.0
        
        # 各項目の適合度計算
        for feature in COMPLETE_CRITERIA:
            student_val = student.get(feature, 0.5)
            lab_val = lab.get(feature, 0.5)
            diff = abs(student_val - lab_val)
            compatibility = self._fuzzy_similarity(diff)
            feature_scores[feature] = compatibility
            weighted_sum += compatibility * self.importance_weights[feature]
        
        # 分野興味度の処理
        field_match_score = 0.0
        field_interests = student.get("field_interests", {})
        lab_field = lab.get("field")
        
        if field_interests and lab_field:
            interest_level = field_interests.get(lab_field, 0.5)
            field_match_score = interest_level
            # 分野マッチングを総合スコアに反映（20%の重み）
            weighted_sum = weighted_sum * 0.8 + field_match_score * 0.2
        
        overall_score = weighted_sum
        cluster_info = self.decision_tree.classify(student)
        recommendation = self._determine_recommendation(overall_score, cluster_info)
        
        return {
            "overall_score": round(overall_score, 4),
            "weighted_score": round(weighted_sum, 4),
            "feature_scores": {k: round(v, 4) for k, v in feature_scores.items()},
            "field_match_score": round(field_match_score, 4),
            "cluster_info": cluster_info,
            "recommendation_level": recommendation["level"],
            "recommendation_confidence": recommendation["confidence"],
            "fuzzy_analysis": self._generate_fuzzy_analysis(feature_scores, cluster_info, field_match_score)
        }
    
    def _fuzzy_similarity(self, difference: float) -> float:
        """ファジィ類似度計算"""
        return float(np.exp(-0.5 * (difference / 0.2) ** 2))
    
    def _determine_recommendation(self, score: float, cluster_info: Dict) -> Dict[str, Any]:
        """推薦レベルを決定"""
        if score >= 0.85:
            base_level, base_confidence = "excellent", 0.95
        elif score >= 0.75:
            base_level, base_confidence = "very_good", 0.85
        elif score >= 0.65:
            base_level, base_confidence = "good", 0.75
        elif score >= 0.50:
            base_level, base_confidence = "fair", 0.60
        else:
            base_level, base_confidence = "not_recommended", 0.40
        
        level1_max = max(cluster_info["level1_memberships"].values())
        level2_max = max(cluster_info["level2_memberships"].values())
        adjusted_confidence = base_confidence * (0.7 + 0.3 * (level1_max + level2_max) / 2)
        
        return {"level": base_level, "confidence": round(adjusted_confidence, 4)}
    
    def _generate_fuzzy_analysis(self, feature_scores: Dict, 
                                 cluster_info: Dict, field_match: float) -> Dict[str, Any]:
        """ファジィ分析結果を生成"""
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "top_matching_features": [
                {"feature": f, "score": s} for f, s in sorted_features[:5]
            ],
            "improvement_areas": [
                {"feature": f, "score": s} for f, s in sorted_features[-3:]
            ],
            "field_match_score": round(field_match, 4),
            "cluster_interpretation": cluster_info.get("fuzzy_interpretation", ""),
            "classification_confidence": {
                "level1": max(cluster_info["level1_memberships"].values()),
                "level2": max(cluster_info["level2_memberships"].values())
            }
        }

# ===============================
# 研究室データベース（12項目対応）
# ===============================

ENHANCED_LAB_DATABASE = [
    {
        "id": "ai_ml_lab",
        "name": "人工知能・機械学習研究室",
        "advisor": "田中教授",
        "category": "テクノロジー・システム",
        "field": "人工知能・機械学習",
        "description": "深層学習とデータマイニングの先端研究",
        "research_intensity": 0.9, "advisor_style": 0.7, "team_work": 0.8,
        "workload": 0.85, "theory_practice": 0.6, "research_field_match": 0.9,
        "skill_development": 0.85, "lab_atmosphere": 0.8, "flexibility": 0.6,
        "publication_opportunity": 0.9, "interdisciplinary": 0.7, "communication_style": 0.8
    },
    {
        "id": "image_processing_lab",
        "name": "画像・映像処理研究室",
        "advisor": "佐藤教授",
        "category": "テクノロジー・システム",
        "field": "画像・映像処理",
        "description": "コンピュータビジョンとメディア処理",
        "research_intensity": 0.8, "advisor_style": 0.6, "team_work": 0.7,
        "workload": 0.75, "theory_practice": 0.7, "research_field_match": 0.85,
        "skill_development": 0.8, "lab_atmosphere": 0.75, "flexibility": 0.7,
        "publication_opportunity": 0.8, "interdisciplinary": 0.6, "communication_style": 0.7
    },
    {
        "id": "network_security_lab",
        "name": "ネットワーク・セキュリティ研究室",
        "advisor": "鈴木教授",
        "category": "テクノロジー・システム",
        "field": "ネットワーク・セキュリティ",
        "description": "サイバーセキュリティとネットワーク最適化",
        "research_intensity": 0.85, "advisor_style": 0.8, "team_work": 0.6,
        "workload": 0.8, "theory_practice": 0.5, "research_field_match": 0.8,
        "skill_development": 0.9, "lab_atmosphere": 0.6, "flexibility": 0.5,
        "publication_opportunity": 0.75, "interdisciplinary": 0.5, "communication_style": 0.6
    },
    {
        "id": "web_ui_ux_lab",
        "name": "Webデザイン・UI/UX研究室",
        "advisor": "高橋教授",
        "category": "クリエイティブ",
        "field": "Webデザイン・UI/UX",
        "description": "ユーザー体験設計とインタラクションデザイン",
        "research_intensity": 0.7, "advisor_style": 0.5, "team_work": 0.85,
        "workload": 0.7, "theory_practice": 0.85, "research_field_match": 0.8,
        "skill_development": 0.85, "lab_atmosphere": 0.9, "flexibility": 0.8,
        "publication_opportunity": 0.6, "interdisciplinary": 0.8, "communication_style": 0.9
    },
    {
        "id": "game_esports_lab",
        "name": "ゲーム開発・eスポーツ研究室",
        "advisor": "山田教授",
        "category": "エンターテイメント",
        "field": "ゲーム開発・eスポーツ",
        "description": "ゲームAIとeスポーツ科学",
        "research_intensity": 0.75, "advisor_style": 0.6, "team_work": 0.9,
        "workload": 0.8, "theory_practice": 0.8, "research_field_match": 0.85,
        "skill_development": 0.9, "lab_atmosphere": 0.95, "flexibility": 0.7,
        "publication_opportunity": 0.65, "interdisciplinary": 0.75, "communication_style": 0.95
    },
    {
        "id": "sports_science_lab",
        "name": "スポーツ・体育科学研究室",
        "advisor": "綿谷教授",
        "category": "人文・社会・体育",
        "field": "スポーツ・体育科学",
        "description": "スポーツパフォーマンスと健康科学",
        "research_intensity": 0.75, "advisor_style": 0.7, "team_work": 0.85,
        "workload": 0.75, "theory_practice": 0.85, "research_field_match": 0.8,
        "skill_development": 0.8, "lab_atmosphere": 0.9, "flexibility": 0.75,
        "publication_opportunity": 0.75, "interdisciplinary": 0.8, "communication_style": 0.85
    }
]

# ===============================
# システム状態管理
# ===============================

system_state = {
    "initialized": False,
    "compatibility_engine": None,
    "lab_database": [],
    "evaluation_count": 0,
    "version": "3.0.0",
    "optimized_tree": None,
    "optimization_history": []
}

def initialize_system():
    """システム初期化"""
    try:
        print("=" * 60)
        print("🚀 研究室選択支援システム v3.0 起動中...")
        print("=" * 60)
        
        system_state["compatibility_engine"] = EnhancedCompatibilityEngine()
        print("✅ 12項目対応適合度エンジン初期化完了")
        
        system_state["lab_database"] = ENHANCED_LAB_DATABASE
        print(f"✅ 研究室データベース読み込み完了: {len(ENHANCED_LAB_DATABASE)}研究室")
        
        print("✅ 多階層ファジィ決定木初期化完了")
        
        system_state["initialized"] = True
        print("=" * 60)
        print("🎉 システム起動完了！")
        print(f"📊 対応項目数: {len(COMPLETE_CRITERIA)}項目")
        print(f"🏫 研究分野数: {len(RESEARCH_FIELDS)}分野")
        print(f"🌳 決定木レベル: 3階層")
        print(f"🔬 研究室数: {len(ENHANCED_LAB_DATABASE)}研究室")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ システム初期化エラー: {e}")
        system_state["initialized"] = False

initialize_system()

# ===============================
# APIエンドポイント
# ===============================

@app.get("/")
async def read_root():
    """ルートエンドポイント"""
    return {
        "message": "遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム v3.0",
        "version": "3.0.0",
        "features": {
            "evaluation_criteria": f"{len(COMPLETE_CRITERIA)}項目対応",
            "research_fields": f"{len(RESEARCH_FIELDS)}分野対応",
            "decision_tree": "3階層ファジィ決定木",
            "fuzzy_logic": "三角型・ガウス型メンバーシップ関数",
            "genetic_algorithm": "完全実装",
            "labs_count": len(ENHANCED_LAB_DATABASE)
        },
        "status": "running"
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
            "evaluation_criteria": len(COMPLETE_CRITERIA),
            "research_fields": len(RESEARCH_FIELDS),
            "lab_count": len(system_state["lab_database"]),
            "evaluation_count": system_state["evaluation_count"],
            "features": {
                "fuzzy_membership": True,
                "multi_level_tree": True,
                "12_criteria": True,
                "19_fields": True,
                "genetic_algorithm": True
            }
        }
    }

@app.get("/api/criteria")
async def get_evaluation_criteria():
    """評価基準一覧取得（12項目 + 19分野）"""
    return {
        "total_count": len(COMPLETE_CRITERIA),
        "categories": {
            "basic": COMPLETE_CRITERIA[0:5],
            "extended": COMPLETE_CRITERIA[5:10],
            "special": COMPLETE_CRITERIA[10:12]
        },
        "all_criteria": COMPLETE_CRITERIA,
        "research_fields": {
            "total_count": len(RESEARCH_FIELDS),
            "all_fields": RESEARCH_FIELDS,
            "by_category": {
                "テクノロジー・システム": RESEARCH_FIELDS[0:11],
                "クリエイティブ": RESEARCH_FIELDS[11:15],
                "エンターテイメント": RESEARCH_FIELDS[15:17],
                "人文・社会・体育": RESEARCH_FIELDS[17:19]
            }
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
        "last_updated": datetime.now().isoformat()
    }

@app.post("/api/evaluate")
async def evaluate_compatibility(request_data: Dict[str, Any]):
    """学生プロファイルに基づく研究室適合度評価（12項目+19分野対応）"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        student_profile = request_data.get("student_profile", {})
        
        # 基本5項目の検証（必須）
        basic_criteria = COMPLETE_CRITERIA[0:5]
        missing_basic = [c for c in basic_criteria if c not in student_profile]
        
        if missing_basic:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required basic criteria: {missing_basic}"
            )
        
        # デフォルト値で補完
        for criterion in COMPLETE_CRITERIA:
            if criterion not in student_profile:
                student_profile[criterion] = 0.5
        
        # 各研究室との適合度計算
        engine = system_state["compatibility_engine"]
        results = []
        
        for lab in system_state["lab_database"]:
            compatibility_result = engine.calculate_compatibility(student_profile, lab)
            
            lab_result = {
                "lab_id": lab["id"],
                "lab_name": lab["name"],
                "advisor": lab["advisor"],
                "category": lab["category"],
                "field": lab["field"],
                "overall_compatibility": compatibility_result["overall_score"],
                "field_match_score": compatibility_result["field_match_score"],
                "feature_scores": compatibility_result["feature_scores"],
                "cluster_info": compatibility_result["cluster_info"],
                "recommendation_level": compatibility_result["recommendation_level"],
                "fuzzy_analysis": compatibility_result["fuzzy_analysis"]
            }
            
            results.append(lab_result)
        
        results.sort(key=lambda x: x["overall_compatibility"], reverse=True)
        system_state["evaluation_count"] += 1
        
        return {
            "student_profile": student_profile,
            "lab_results": results,
            "summary": {
                "total_labs": len(results),
                "top_match": results[0] if results else None,
                "excellent_matches": len([r for r in results if r["recommendation_level"] == "excellent"]),
                "evaluation_method": "12項目+19分野対応ファジィ決定木"
            },
            "metadata": {
                "evaluation_timestamp": datetime.now().isoformat(),
                "system_version": "3.0.0",
                "criteria_count": len(COMPLETE_CRITERIA),
                "field_count": len(RESEARCH_FIELDS)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

@app.post("/api/optimize")
async def optimize_matching(optimization_request: Dict[str, Any]):
    """遺伝的アルゴリズムによる決定木最適化"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        from utils.training_data import TrainingDataGenerator
        from core.genetic.evolution import EvolutionConfig, EvolutionEngine
        
        training_mode = optimization_request.get("training_mode", "balanced")
        num_samples = optimization_request.get("num_samples", 100)
        
        print(f"\n🧬 遺伝的アルゴリズム最適化開始...")
        
        # 訓練データ生成
        if training_mode == "balanced":
            samples_per_type = num_samples // 6
            training_data = TrainingDataGenerator.generate_balanced_dataset(samples_per_type)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown training mode: {training_mode}")
        
        # 進化設定
        evolution_config = EvolutionConfig(
            generations=optimization_request.get("generations", 50),
            population_size=optimization_request.get("population_size", 30),
            elite_size=optimization_request.get("elite_size", 3),
            verbose=True
        )
        
        # 進化エンジン実行
        engine = EvolutionEngine(evolution_config)
        best_individual, evolution_history = engine.optimize(training_data)
        optimal_tree = engine.get_optimized_decision_tree()
        
        system_state["optimized_tree"] = optimal_tree
        system_state["optimization_history"] = evolution_history
        
        return {
            "status": "success",
            "optimal_tree": {
                "level1_feature": optimal_tree["level1_feature"],
                "level2_features": optimal_tree["level2_features"],
                "fitness": optimal_tree["fitness"]
            },
            "evolution_summary": {
                "total_generations": len(evolution_history),
                "final_fitness": optimal_tree["fitness"]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

# ===============================
# サーバー起動
# ===============================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")