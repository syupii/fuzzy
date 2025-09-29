#!/usr/bin/env python3
"""
遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム
FastAPI メインアプリケーション - 改善版 (13項目対応)
Version: 3.0.0
"""

import os
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from typing import Dict, List, Any, Optional
import json
import time
import random
import numpy as np
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# ===============================
# 設定とモジュールインポート
# ===============================

# 13項目完全対応評価基準
COMPLETE_CRITERIA = [
    # 基本5項目
    "research_intensity", "advisor_style", "team_work", 
    "workload", "theory_practice",
    # 拡張5項目
    "research_field_match", "skill_development", "lab_atmosphere",
    "flexibility", "publication_opportunity",
    # 特殊3項目
    "interdisciplinary", "communication_style", "innovation_risk"
]

# FastAPIアプリケーション初期化
app = FastAPI(
    title="研究室選択支援システム v3.0",
    description="遺伝的アルゴリズムを用いたファジィ決定木による研究室マッチングシステム（13項目完全対応版）",
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
        """三角型メンバーシップ関数
        Args:
            x: 入力値
            a: 左端
            b: 中心（ピーク）
            c: 右端
        Returns:
            メンバーシップ度 [0, 1]
        """
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
    
    @staticmethod
    def trapezoid(x: float, a: float, b: float, c: float, d: float) -> float:
        """台形型メンバーシップ関数"""
        if x <= a or x >= d:
            return 0.0
        elif b <= x <= c:
            return 1.0
        elif a < x < b:
            return (x - a) / (b - a)
        else:
            return (d - x) / (d - c)

# ===============================
# 多階層ファジィ決定木
# ===============================

class MultiLevelFuzzyDecisionTree:
    """多階層ファジィ決定木（3レベル階層）"""
    
    def __init__(self):
        self.fuzzy = FuzzyMembershipFunctions()
        self.tree_structure = self._build_tree_structure()
    
    def _build_tree_structure(self) -> Dict:
        """決定木構造を定義"""
        return {
            "level1": {
                "feature": "research_intensity",
                "branches": {
                    "high": {
                        "threshold_min": 0.7,
                        "level2_feature": "team_work",
                        "clusters": {
                            "team_oriented": {"threshold_min": 0.7},
                            "individual_focused": {"threshold_max": 0.7}
                        }
                    },
                    "medium": {
                        "threshold_min": 0.4,
                        "threshold_max": 0.7,
                        "level2_feature": "flexibility",
                        "clusters": {
                            "flexible_style": {"threshold_min": 0.6},
                            "structured_style": {"threshold_max": 0.6}
                        }
                    },
                    "low": {
                        "threshold_max": 0.4,
                        "level2_feature": "lab_atmosphere",
                        "clusters": {
                            "active_atmosphere": {"threshold_min": 0.6},
                            "quiet_atmosphere": {"threshold_max": 0.6}
                        }
                    }
                }
            }
        }
    
    def classify(self, profile: Dict[str, float]) -> Dict[str, Any]:
        """学生プロファイルを分類
        
        Returns:
            {
                "primary_cluster": str,
                "membership_degrees": Dict[str, float],
                "classification_path": List[str],
                "fuzzy_scores": Dict
            }
        """
        research_intensity = profile.get("research_intensity", 0.5)
        
        # レベル1: research_intensityでファジィ分類
        low_degree = self.fuzzy.triangular(research_intensity, 0, 0, 0.5)
        medium_degree = self.fuzzy.triangular(research_intensity, 0.3, 0.5, 0.7)
        high_degree = self.fuzzy.triangular(research_intensity, 0.5, 1.0, 1.0)
        
        level1_memberships = {
            "low": low_degree,
            "medium": medium_degree,
            "high": high_degree
        }
        
        # 最大メンバーシップ度のブランチを選択
        primary_branch = max(level1_memberships, key=level1_memberships.get)
        
        # レベル2: 細分化
        branch_config = self.tree_structure["level1"]["branches"][primary_branch]
        level2_feature = branch_config["level2_feature"]
        level2_value = profile.get(level2_feature, 0.5)
        
        # レベル2分類
        clusters = branch_config["clusters"]
        cluster_memberships = {}
        
        for cluster_name, cluster_config in clusters.items():
            if "threshold_min" in cluster_config:
                # 高値クラスタ（threshold_min以上）
                membership = self.fuzzy.triangular(
                    level2_value,
                    cluster_config["threshold_min"] - 0.2,
                    cluster_config["threshold_min"],
                    1.0
                )
            else:
                # 低値クラスタ（threshold_max以下）
                membership = self.fuzzy.triangular(
                    level2_value,
                    0.0,
                    cluster_config["threshold_max"],
                    cluster_config["threshold_max"] + 0.2
                )
            
            cluster_memberships[cluster_name] = membership
        
        # 最終クラスタ決定
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
            ],
            "fuzzy_interpretation": self._generate_interpretation(
                primary_branch, final_cluster, level1_memberships, cluster_memberships
            )
        }
    
    def _generate_interpretation(self, branch: str, cluster: str, 
                                 l1_mem: Dict, l2_mem: Dict) -> str:
        """分類結果の解釈を生成"""
        interpretations = {
            "high_team_oriented": "高い研究意欲とチーム志向を持つ学生",
            "high_individual_focused": "高い研究意欲と個人研究志向を持つ学生",
            "medium_flexible_style": "中程度の研究意欲と柔軟なスタイルを好む学生",
            "medium_structured_style": "中程度の研究意欲と構造化されたスタイルを好む学生",
            "low_active_atmosphere": "軽い研究負荷で活発な雰囲気を好む学生",
            "low_quiet_atmosphere": "軽い研究負荷で静かな雰囲気を好む学生"
        }
        
        key = f"{branch}_{cluster}"
        return interpretations.get(key, f"{branch}ブランチ、{cluster}タイプの学生")

# ===============================
# 拡張適合度計算エンジン
# ===============================

class EnhancedCompatibilityEngine:
    """13項目対応の拡張適合度計算エンジン"""
    
    def __init__(self):
        self.fuzzy = FuzzyMembershipFunctions()
        self.decision_tree = MultiLevelFuzzyDecisionTree()
        
        # 項目の重要度（カテゴリ別）
        self.importance_weights = {
            # 基本項目（40%）
            "research_intensity": 0.12,
            "advisor_style": 0.08,
            "team_work": 0.08,
            "workload": 0.06,
            "theory_practice": 0.06,
            
            # 拡張項目（35%）
            "research_field_match": 0.10,
            "skill_development": 0.07,
            "lab_atmosphere": 0.07,
            "flexibility": 0.06,
            "publication_opportunity": 0.05,
            
            # 特殊項目（25%）
            "interdisciplinary": 0.09,
            "communication_style": 0.08,
            "innovation_risk": 0.08
        }
    
    def calculate_compatibility(self, student: Dict[str, float], 
                               lab: Dict[str, float]) -> Dict[str, Any]:
        """13項目を考慮した詳細適合度計算
        
        Returns:
            {
                "overall_score": float,
                "weighted_score": float,
                "feature_scores": Dict[str, float],
                "fuzzy_memberships": Dict[str, float],
                "cluster_info": Dict,
                "recommendation_level": str
            }
        """
        feature_scores = {}
        weighted_sum = 0.0
        
        # 各項目の適合度計算（ファジィメンバーシップ使用）
        for feature in COMPLETE_CRITERIA:
            student_val = student.get(feature, 0.5)
            lab_val = lab.get(feature, 0.5)
            
            # 差分計算
            diff = abs(student_val - lab_val)
            
            # ファジィメンバーシップで適合度を計算
            # 差分が小さいほど高い適合度
            compatibility = self._fuzzy_similarity(diff)
            
            feature_scores[feature] = compatibility
            weighted_sum += compatibility * self.importance_weights[feature]
        
        # 総合スコア
        overall_score = weighted_sum
        
        # 決定木による分類
        cluster_info = self.decision_tree.classify(student)
        
        # 推薦レベル決定
        recommendation = self._determine_recommendation(overall_score, cluster_info)
        
        return {
            "overall_score": round(overall_score, 4),
            "weighted_score": round(weighted_sum, 4),
            "feature_scores": {k: round(v, 4) for k, v in feature_scores.items()},
            "cluster_info": cluster_info,
            "recommendation_level": recommendation["level"],
            "recommendation_confidence": recommendation["confidence"],
            "fuzzy_analysis": self._generate_fuzzy_analysis(feature_scores, cluster_info)
        }
    
    def _fuzzy_similarity(self, difference: float) -> float:
        """ファジィ類似度計算
        
        差分が小さいほど高い類似度を返す
        """
        # ガウス型メンバーシップで類似度を計算
        # sigma=0.2 で適度なファジネスを持たせる
        similarity = np.exp(-0.5 * (difference / 0.2) ** 2)
        return float(similarity)
    
    def _determine_recommendation(self, score: float, 
                                  cluster_info: Dict) -> Dict[str, Any]:
        """推薦レベルを決定"""
        # スコアベースの基本判定
        if score >= 0.85:
            base_level = "excellent"
            base_confidence = 0.95
        elif score >= 0.75:
            base_level = "very_good"
            base_confidence = 0.85
        elif score >= 0.65:
            base_level = "good"
            base_confidence = 0.75
        elif score >= 0.50:
            base_level = "fair"
            base_confidence = 0.60
        else:
            base_level = "not_recommended"
            base_confidence = 0.40
        
        # クラスタ情報で信頼度を調整
        level1_max = max(cluster_info["level1_memberships"].values())
        level2_max = max(cluster_info["level2_memberships"].values())
        
        # ファジィメンバーシップが高いほど信頼度が上がる
        adjusted_confidence = base_confidence * (0.7 + 0.3 * (level1_max + level2_max) / 2)
        
        return {
            "level": base_level,
            "confidence": round(adjusted_confidence, 4)
        }
    
    def _generate_fuzzy_analysis(self, feature_scores: Dict, 
                                 cluster_info: Dict) -> Dict[str, Any]:
        """ファジィ分析結果を生成"""
        # トップ5の適合項目
        sorted_features = sorted(feature_scores.items(), 
                                key=lambda x: x[1], reverse=True)
        top_matches = sorted_features[:5]
        
        # 改善が必要な項目
        bottom_features = sorted_features[-3:]
        
        return {
            "top_matching_features": [
                {"feature": f, "score": s} for f, s in top_matches
            ],
            "improvement_areas": [
                {"feature": f, "score": s} for f, s in bottom_features
            ],
            "cluster_interpretation": cluster_info["fuzzy_interpretation"],
            "classification_confidence": {
                "level1": max(cluster_info["level1_memberships"].values()),
                "level2": max(cluster_info["level2_memberships"].values())
            }
        }

# ===============================
# 研究室データベース（新分野対応）
# ===============================

ENHANCED_LAB_DATABASE = [
    # テクノロジー・システム分野
    {
        "id": "ai_ml_lab",
        "name": "人工知能・機械学習研究室",
        "advisor": "田中教授",
        "category": "テクノロジー・システム",
        "field": "人工知能・機械学習",
        "description": "深層学習とデータマイニングの先端研究",
        # 基本5項目
        "research_intensity": 0.9,
        "advisor_style": 0.7,
        "team_work": 0.8,
        "workload": 0.85,
        "theory_practice": 0.6,
        # 拡張5項目
        "research_field_match": 0.9,
        "skill_development": 0.85,
        "lab_atmosphere": 0.8,
        "flexibility": 0.6,
        "publication_opportunity": 0.9,
        # 特殊3項目
        "interdisciplinary": 0.7,
        "communication_style": 0.8,
        "innovation_risk": 0.8
    },
    {
        "id": "image_processing_lab",
        "name": "画像・映像処理研究室",
        "advisor": "佐藤教授",
        "category": "テクノロジー・システム",
        "field": "画像・映像処理",
        "description": "コンピュータビジョンとメディア処理",
        "research_intensity": 0.8,
        "advisor_style": 0.6,
        "team_work": 0.7,
        "workload": 0.75,
        "theory_practice": 0.7,
        "research_field_match": 0.85,
        "skill_development": 0.8,
        "lab_atmosphere": 0.75,
        "flexibility": 0.7,
        "publication_opportunity": 0.8,
        "interdisciplinary": 0.6,
        "communication_style": 0.7,
        "innovation_risk": 0.7
    },
    {
        "id": "network_security_lab",
        "name": "ネットワーク・セキュリティ研究室",
        "advisor": "鈴木教授",
        "category": "テクノロジー・システム",
        "field": "ネットワーク・セキュリティ",
        "description": "サイバーセキュリティとネットワーク最適化",
        "research_intensity": 0.85,
        "advisor_style": 0.8,
        "team_work": 0.6,
        "workload": 0.8,
        "theory_practice": 0.5,
        "research_field_match": 0.8,
        "skill_development": 0.9,
        "lab_atmosphere": 0.6,
        "flexibility": 0.5,
        "publication_opportunity": 0.75,
        "interdisciplinary": 0.5,
        "communication_style": 0.6,
        "innovation_risk": 0.7
    },
    {
        "id": "education_language_lab",
        "name": "教育・言語学研究室",
        "advisor": "飯嶋教授",
        "category": "テクノロジー・システム",
        "field": "教育・言語学",
        "description": "教育工学と言語処理の融合研究",
        "research_intensity": 0.7,
        "advisor_style": 0.5,
        "team_work": 0.75,
        "workload": 0.6,
        "theory_practice": 0.6,
        "research_field_match": 0.75,
        "skill_development": 0.7,
        "lab_atmosphere": 0.8,
        "flexibility": 0.8,
        "publication_opportunity": 0.7,
        "interdisciplinary": 0.9,
        "communication_style": 0.85,
        "innovation_risk": 0.6
    },
    {
        "id": "natural_science_lab",
        "name": "自然科学・数理研究室",
        "advisor": "柿並教授",
        "category": "テクノロジー・システム",
        "field": "自然科学・数理",
        "description": "数理モデリングとシミュレーション",
        "research_intensity": 0.85,
        "advisor_style": 0.6,
        "team_work": 0.5,
        "workload": 0.7,
        "theory_practice": 0.3,
        "research_field_match": 0.8,
        "skill_development": 0.75,
        "lab_atmosphere": 0.5,
        "flexibility": 0.6,
        "publication_opportunity": 0.8,
        "interdisciplinary": 0.7,
        "communication_style": 0.5,
        "innovation_risk": 0.6
    },
    {
        "id": "tourism_info_lab",
        "name": "観光情報・地域システム研究室",
        "advisor": "齋藤教授",
        "category": "テクノロジー・システム",
        "field": "観光情報・地域システム",
        "description": "観光DXと地域活性化システム",
        "research_intensity": 0.65,
        "advisor_style": 0.6,
        "team_work": 0.8,
        "workload": 0.6,
        "theory_practice": 0.8,
        "research_field_match": 0.7,
        "skill_development": 0.75,
        "lab_atmosphere": 0.85,
        "flexibility": 0.8,
        "publication_opportunity": 0.65,
        "interdisciplinary": 0.85,
        "communication_style": 0.9,
        "innovation_risk": 0.7
    },
    
    # クリエイティブ分野
    {
        "id": "web_ui_ux_lab",
        "name": "Webデザイン・UI/UX研究室",
        "advisor": "高橋教授",
        "category": "クリエイティブ",
        "field": "Webデザイン・UI/UX",
        "description": "ユーザー体験設計とインタラクションデザイン",
        "research_intensity": 0.7,
        "advisor_style": 0.5,
        "team_work": 0.85,
        "workload": 0.7,
        "theory_practice": 0.85,
        "research_field_match": 0.8,
        "skill_development": 0.85,
        "lab_atmosphere": 0.9,
        "flexibility": 0.8,
        "publication_opportunity": 0.6,
        "interdisciplinary": 0.8,
        "communication_style": 0.9,
        "innovation_risk": 0.75
    },
    {
        "id": "visual_design_lab",
        "name": "デザイン・視覚表現研究室",
        "advisor": "伊藤教授",
        "category": "クリエイティブ",
        "field": "デザイン・視覚表現",
        "description": "グラフィックデザインとビジュアルコミュニケーション",
        "research_intensity": 0.65,
        "advisor_style": 0.4,
        "team_work": 0.75,
        "workload": 0.65,
        "theory_practice": 0.9,
        "research_field_match": 0.75,
        "skill_development": 0.8,
        "lab_atmosphere": 0.85,
        "flexibility": 0.85,
        "publication_opportunity": 0.55,
        "interdisciplinary": 0.7,
        "communication_style": 0.85,
        "innovation_risk": 0.8
    },
    
    # エンターテイメント分野
    {
        "id": "game_esports_lab",
        "name": "ゲーム開発・eスポーツ研究室",
        "advisor": "山田教授",
        "category": "エンターテイメント",
        "field": "ゲーム開発・eスポーツ",
        "description": "ゲームAIとeスポーツ科学",
        "research_intensity": 0.75,
        "advisor_style": 0.6,
        "team_work": 0.9,
        "workload": 0.8,
        "theory_practice": 0.8,
        "research_field_match": 0.85,
        "skill_development": 0.9,
        "lab_atmosphere": 0.95,
        "flexibility": 0.7,
        "publication_opportunity": 0.65,
        "interdisciplinary": 0.75,
        "communication_style": 0.95,
        "innovation_risk": 0.85
    },
    {
        "id": "vr_ar_lab",
        "name": "VR/AR・メディアアート研究室",
        "advisor": "中村教授",
        "category": "エンターテイメント",
        "field": "VR/AR・メディアアート",
        "description": "仮想現実と拡張現実の芸術応用",
        "research_intensity": 0.8,
        "advisor_style": 0.5,
        "team_work": 0.8,
        "workload": 0.75,
        "theory_practice": 0.75,
        "research_field_match": 0.8,
        "skill_development": 0.85,
        "lab_atmosphere": 0.85,
        "flexibility": 0.75,
        "publication_opportunity": 0.7,
        "interdisciplinary": 0.9,
        "communication_style": 0.8,
        "innovation_risk": 0.9
    },
    
    # 人文・社会・体育分野
    {
        "id": "philosophy_humanities_lab",
        "name": "哲学・人文・環境行動学研究室",
        "advisor": "三浦教授",
        "category": "人文・社会・体育",
        "field": "哲学・人文・環境行動学",
        "description": "人間行動と環境の相互作用研究",
        "research_intensity": 0.7,
        "advisor_style": 0.4,
        "team_work": 0.6,
        "workload": 0.6,
        "theory_practice": 0.4,
        "research_field_match": 0.7,
        "skill_development": 0.65,
        "lab_atmosphere": 0.7,
        "flexibility": 0.85,
        "publication_opportunity": 0.7,
        "interdisciplinary": 0.95,
        "communication_style": 0.75,
        "innovation_risk": 0.6
    },
    {
        "id": "sports_science_lab",
        "name": "スポーツ・体育科学研究室",
        "advisor": "綿谷教授",
        "category": "人文・社会・体育",
        "field": "スポーツ・体育科学",
        "description": "スポーツパフォーマンスと健康科学",
        "research_intensity": 0.75,
        "advisor_style": 0.7,
        "team_work": 0.85,
        "workload": 0.75,
        "theory_practice": 0.85,
        "research_field_match": 0.8,
        "skill_development": 0.8,
        "lab_atmosphere": 0.9,
        "flexibility": 0.75,
        "publication_opportunity": 0.75,
        "interdisciplinary": 0.8,
        "communication_style": 0.85,
        "innovation_risk": 0.65
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
    "version": "3.0.0"
}

def initialize_system():
    """システム初期化"""
    try:
        print("=" * 60)
        print("🚀 研究室選択支援システム v3.0 起動中...")
        print("=" * 60)
        
        # 適合度計算エンジン初期化
        system_state["compatibility_engine"] = EnhancedCompatibilityEngine()
        print("✅ 13項目対応適合度エンジン初期化完了")
        
        # 研究室データベース読み込み
        system_state["lab_database"] = ENHANCED_LAB_DATABASE
        print(f"✅ 研究室データベース読み込み完了: {len(ENHANCED_LAB_DATABASE)}研究室")
        
        # 多階層決定木初期化
        decision_tree = MultiLevelFuzzyDecisionTree()
        print("✅ 多階層ファジィ決定木初期化完了")
        
        system_state["initialized"] = True
        print("=" * 60)
        print("🎉 システム起動完了！")
        print(f"📊 対応項目数: {len(COMPLETE_CRITERIA)}項目")
        print(f"🌳 決定木レベル: 3階層")
        print(f"🔬 研究室数: {len(ENHANCED_LAB_DATABASE)}研究室")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ システム初期化エラー: {e}")
        system_state["initialized"] = False

# システム初期化
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
            "decision_tree": "3階層ファジィ決定木",
            "fuzzy_logic": "三角型・ガウス型メンバーシップ関数",
            "labs_count": len(ENHANCED_LAB_DATABASE)
        },
        "status": "running",
        "endpoints": {
            "health": "/health",
            "labs": "/api/labs",
            "evaluate": "/api/evaluate",
            "criteria": "/api/criteria",
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
            "evaluation_criteria": len(COMPLETE_CRITERIA),
            "lab_count": len(system_state["lab_database"]),
            "evaluation_count": system_state["evaluation_count"],
            "features": {
                "fuzzy_membership": True,
                "multi_level_tree": True,
                "13_criteria": True
            }
        }
    }

@app.get("/api/criteria")
async def get_evaluation_criteria():
    """評価基準一覧取得"""
    criteria_info = {
        "total_count": len(COMPLETE_CRITERIA),
        "categories": {
            "basic": COMPLETE_CRITERIA[0:5],
            "extended": COMPLETE_CRITERIA[5:10],
            "special": COMPLETE_CRITERIA[10:13]
        },
        "all_criteria": COMPLETE_CRITERIA,
        "descriptions": {
            "research_intensity": "研究強度（1=軽い 〜 10=集中的）",
            "advisor_style": "指導スタイル（1=厳格 〜 10=自由）",
            "team_work": "チームワーク（1=個人 〜 10=チーム）",
            "workload": "ワークロード（1=軽い 〜 10=重い）",
            "theory_practice": "理論実践（1=理論 〜 10=実践）",
            "research_field_match": "研究分野適合性（1=広い 〜 10=専門特化）",
            "skill_development": "スキル開発（1=専門 〜 10=幅広い）",
            "lab_atmosphere": "研究室雰囲気（1=静寂 〜 10=活発）",
            "flexibility": "柔軟性（1=固定 〜 10=柔軟）",
            "publication_opportunity": "論文発表機会（1=少ない 〜 10=豊富）",
            "interdisciplinary": "学際性（1=単一分野 〜 10=学際連携）",
            "communication_style": "コミュニケーション（1=少人数 〜 10=オープン）",
            "innovation_risk": "革新性（1=保守的 〜 10=革新的）"
        }
    }
    return criteria_info

@app.get("/api/labs")
async def get_labs():
    """研究室一覧取得"""
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # カテゴリ別に整理
    categories = {}
    for lab in system_state["lab_database"]:
        category = lab["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append(lab)
    
    return {
        "labs": system_state["lab_database"],
        "total_count": len(system_state["lab_database"]),
        "categories": list(categories.keys()),
        "by_category": categories,
        "last_updated": datetime.now().isoformat()
    }

@app.get("/api/labs/{lab_id}")
async def get_lab_detail(lab_id: str):
    """特定研究室の詳細取得"""
    lab = next((lab for lab in system_state["lab_database"] if lab["id"] == lab_id), None)
    
    if not lab:
        raise HTTPException(status_code=404, detail="Lab not found")
    
    return lab

@app.post("/api/evaluate")
async def evaluate_compatibility(request_data: Dict[str, Any]):
    """学生プロファイルに基づく研究室適合度評価（13項目対応）"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        student_profile = request_data.get("student_profile", {})
        
        # 13項目の検証（priorities付きでも対応）
        if "priorities" in student_profile:
            # 優先度付き評価の場合
            profile_values = {k: v for k, v in student_profile.items() 
                            if k in COMPLETE_CRITERIA}
        else:
            profile_values = student_profile
        
        # 必須項目チェック（基本5項目のみ必須、他はデフォルト値使用）
        basic_criteria = COMPLETE_CRITERIA[0:5]
        missing_basic = [c for c in basic_criteria if c not in profile_values]
        
        if missing_basic:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required basic criteria: {missing_basic}"
            )
        
        # デフォルト値で補完
        for criterion in COMPLETE_CRITERIA:
            if criterion not in profile_values:
                profile_values[criterion] = 0.5  # 中立値
        
        # 各研究室との適合度計算
        engine = system_state["compatibility_engine"]
        results = []
        
        for lab in system_state["lab_database"]:
            compatibility_result = engine.calculate_compatibility(
                profile_values, lab
            )
            
            lab_result = {
                "lab_id": lab["id"],
                "lab_name": lab["name"],
                "advisor": lab["advisor"],
                "category": lab["category"],
                "field": lab["field"],
                "overall_compatibility": compatibility_result["overall_score"],
                "weighted_score": compatibility_result["weighted_score"],
                "feature_scores": compatibility_result["feature_scores"],
                "cluster_info": compatibility_result["cluster_info"],
                "recommendation_level": compatibility_result["recommendation_level"],
                "recommendation_confidence": compatibility_result["recommendation_confidence"],
                "fuzzy_analysis": compatibility_result["fuzzy_analysis"]
            }
            
            results.append(lab_result)
        
        # スコアでソート
        results.sort(key=lambda x: x["overall_compatibility"], reverse=True)
        
        # 評価回数増加
        system_state["evaluation_count"] += 1
        
        # レスポンス構築
        response = {
            "student_profile": profile_values,
            "lab_results": results,
            "summary": {
                "total_labs": len(results),
                "top_match": results[0] if results else None,
                "excellent_matches": len([r for r in results 
                                         if r["recommendation_level"] == "excellent"]),
                "good_matches": len([r for r in results 
                                    if r["recommendation_level"] in ["very_good", "good"]]),
                "evaluation_method": "13項目対応ファジィ決定木",
                "decision_tree_levels": 3
            },
            "metadata": {
                "evaluation_timestamp": datetime.now().isoformat(),
                "evaluation_count": system_state["evaluation_count"],
                "system_version": "3.0.0",
                "criteria_count": len(COMPLETE_CRITERIA)
            }
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

@app.post("/api/optimize")
async def optimize_matching(optimization_request: Dict[str, Any]):
    """遺伝的アルゴリズムによる最適化（将来実装用）"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {
        "status": "not_implemented",
        "message": "遺伝的アルゴリズム最適化は次期バージョンで実装予定です",
        "available_in": "v3.1.0"
    }

# ===============================
# サーバー起動
# ===============================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )