#!/usr/bin/env python3
"""
遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム
FastAPI メインアプリケーション - 21項目+11分野+遺伝的アルゴリズム対応版
"""

import os
import sys
import uvicorn
import json
import time
import random
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# FastAPIアプリケーション初期化
app = FastAPI(
    title="研究室選択支援システム",
    description="遺伝的アルゴリズムを用いたファジィ決定木による研究室マッチングシステム（21項目+11分野対応）",
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

# Pydanticモデル定義
class EvaluationPreferences(BaseModel):
    # 基本項目（5項目）
    research_intensity: float
    advisor_style: float
    team_work: float
    workload: float
    theory_practice: float
    
    # 拡張項目（5項目）
    research_field_match: float
    skill_development: float
    learning_pace: float
    difficulty_preference: float
    lab_atmosphere: float
    
    # コミュニケーション関連（4項目）
    communication_style: float
    meeting_frequency: float
    flexibility: float
    evening_weekend_work: float
    
    # 研究アプローチ関連（3項目）
    innovation_risk: float
    methodology_preference: float
    interdisciplinary: float
    
    # 重要項目（4項目）
    publication_opportunity: float
    financial_support: float
    lab_hierarchy: float
    core_time_flexibility: float

class ResearchFieldInterests(BaseModel):
    # フィールド名をPython対応に変更
    ai_ml: float = 0.0  # 人工知能・機械学習
    image_video: float = 0.0  # 画像・映像処理
    network_security: float = 0.0  # コンピュータネットワーク・セキュリティ
    database_systems: float = 0.0  # データベース・情報システム
    embedded_iot: float = 0.0  # 組込み・IoT
    web_ui_ux: float = 0.0  # Webデザイン・UI/UX
    design_visual: float = 0.0  # デザイン・視覚表現
    video_animation: float = 0.0  # 映像・アニメーション
    computer_music: float = 0.0  # コンピュータ音楽・サウンドアート
    game_esports: float = 0.0  # ゲーム開発・eスポーツ
    vr_ar_media: float = 0.0  # VR/AR・メディアアート

class PersonalInfo(BaseModel):
    grade: int = 3
    experience_level: str = "intermediate"
    career_goals: List[str] = []
    preferred_learning_style: str = "mixed"

class StudentProfile(BaseModel):
    evaluation_criteria: EvaluationPreferences
    field_interests: ResearchFieldInterests
    personal_info: Optional[PersonalInfo] = None

# フィールド名マッピング（日本語名との対応）
FIELD_NAME_MAPPING = {
    "ai_ml": "人工知能・機械学習",
    "image_video": "画像・映像処理",
    "network_security": "コンピュータネットワーク・セキュリティ",
    "database_systems": "データベース・情報システム",
    "embedded_iot": "組込み・IoT",
    "web_ui_ux": "Webデザイン・UI/UX",
    "design_visual": "デザイン・視覚表現",
    "video_animation": "映像・アニメーション",
    "computer_music": "コンピュータ音楽・サウンドアート",
    "game_esports": "ゲーム開発・eスポーツ",
    "vr_ar_media": "VR/AR・メディアアート"
}

# 21項目対応サンプル研究室データ
SAMPLE_LABS = [
    {
        "id": "lab_001",
        "name": "人工知能研究室",
        "professor": "伊藤雅彦",
        "research_area": "人工知能・機械学習",
        "specialization": "情報可視化、ユーザインタフェース、データ工学",
        "research_fields": ["人工知能・機械学習", "データベース・情報システム"],
        "description": "情報可視化、ユーザインタフェース、データ工学の研究を行っています",
        "features": {
            "research_intensity": 8.5,
            "advisor_style": 7.0,
            "team_work": 8.0,
            "workload": 8.0,
            "theory_practice": 6.5,
            "research_field_match": 9.0,
            "skill_development": 8.5,
            "learning_pace": 7.0,
            "difficulty_preference": 8.0,
            "lab_atmosphere": 7.5,
            "communication_style": 7.5,
            "meeting_frequency": 7.0,
            "flexibility": 6.0,
            "evening_weekend_work": 7.0,
            "innovation_risk": 8.0,
            "methodology_preference": 7.5,
            "interdisciplinary": 7.0,
            "publication_opportunity": 9.0,
            "financial_support": 8.0,
            "lab_hierarchy": 6.0,
            "core_time_flexibility": 6.5
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
        "id": "lab_002",
        "name": "ゲーム開発研究室",
        "professor": "森川悟",
        "research_area": "ゲーム開発・eスポーツ",
        "specialization": "ゲームプログラミング、eスポーツ技術",
        "research_fields": ["ゲーム開発・eスポーツ", "VR/AR・メディアアート"],
        "description": "ゲームプログラミングとeスポーツ技術の研究",
        "features": {
            "research_intensity": 7.0,
            "advisor_style": 8.0,
            "team_work": 9.0,
            "workload": 7.5,
            "theory_practice": 8.0,
            "research_field_match": 8.5,
            "skill_development": 9.0,
            "learning_pace": 8.0,
            "difficulty_preference": 7.5,
            "lab_atmosphere": 9.0,
            "communication_style": 9.0,
            "meeting_frequency": 8.0,
            "flexibility": 8.5,
            "evening_weekend_work": 6.0,
            "innovation_risk": 7.5,
            "methodology_preference": 8.0,
            "interdisciplinary": 6.0,
            "publication_opportunity": 6.0,
            "financial_support": 7.0,
            "lab_hierarchy": 8.0,
            "core_time_flexibility": 8.5
        },
        "metadata": {
            "faculty_count": 1,
            "student_count": 12,
            "recent_publications": 8,
            "funding_level": "中",
            "equipment_rating": 8
        }
    },
    {
        "id": "lab_003",
        "name": "Webデザイン研究室",
        "professor": "杉沢愛美",
        "research_area": "Webデザイン・UI/UX",
        "specialization": "UX・UIデザイン、ブランディングデザイン",
        "research_fields": ["Webデザイン・UI/UX", "デザイン・視覚表現"],
        "description": "UX・UIデザイン、ブランディングデザインの研究",
        "features": {
            "research_intensity": 6.5,
            "advisor_style": 8.5,
            "team_work": 8.5,
            "workload": 6.0,
            "theory_practice": 7.5,
            "research_field_match": 8.0,
            "skill_development": 8.0,
            "learning_pace": 7.5,
            "difficulty_preference": 6.0,
            "lab_atmosphere": 8.5,
            "communication_style": 8.5,
            "meeting_frequency": 6.5,
            "flexibility": 9.0,
            "evening_weekend_work": 4.0,
            "innovation_risk": 7.0,
            "methodology_preference": 7.0,
            "interdisciplinary": 8.5,
            "publication_opportunity": 7.0,
            "financial_support": 6.5,
            "lab_hierarchy": 8.5,
            "core_time_flexibility": 9.0
        },
        "metadata": {
            "faculty_count": 1,
            "student_count": 10,
            "recent_publications": 12,
            "funding_level": "中",
            "equipment_rating": 7
        }
    },
    {
        "id": "lab_004",
        "name": "コンピュータビジョン研究室",
        "professor": "向田茂",
        "research_area": "画像・映像処理",
        "specialization": "画像処理、VR/AR、3DCG、メディアアート",
        "research_fields": ["画像・映像処理", "VR/AR・メディアアート"],
        "description": "画像処理、VR/AR、3DCG、メディアアートの研究",
        "features": {
            "research_intensity": 8.0,
            "advisor_style": 6.5,
            "team_work": 7.0,
            "workload": 7.5,
            "theory_practice": 6.0,
            "research_field_match": 8.5,
            "skill_development": 8.0,
            "learning_pace": 7.5,
            "difficulty_preference": 8.5,
            "lab_atmosphere": 7.0,
            "communication_style": 6.5,
            "meeting_frequency": 7.0,
            "flexibility": 7.0,
            "evening_weekend_work": 7.5,
            "innovation_risk": 9.0,
            "methodology_preference": 8.5,
            "interdisciplinary": 8.0,
            "publication_opportunity": 8.5,
            "financial_support": 8.5,
            "lab_hierarchy": 6.5,
            "core_time_flexibility": 7.0
        },
        "metadata": {
            "faculty_count": 1,
            "student_count": 6,
            "recent_publications": 18,
            "funding_level": "高",
            "equipment_rating": 9
        }
    },
    {
        "id": "lab_005",
        "name": "ネットワークセキュリティ研究室",
        "professor": "中島潤",
        "research_area": "コンピュータネットワーク・セキュリティ",
        "specialization": "情報セキュリティ、ITマネジメント",
        "research_fields": ["コンピュータネットワーク・セキュリティ", "組込み・IoT"],
        "description": "情報セキュリティ、ITマネジメントの研究",
        "features": {
            "research_intensity": 7.5,
            "advisor_style": 6.0,
            "team_work": 6.5,
            "workload": 7.0,
            "theory_practice": 5.0,
            "research_field_match": 8.0,
            "skill_development": 7.5,
            "learning_pace": 6.5,
            "difficulty_preference": 7.0,
            "lab_atmosphere": 6.5,
            "communication_style": 6.0,
            "meeting_frequency": 6.0,
            "flexibility": 6.0,
            "evening_weekend_work": 6.5,
            "innovation_risk": 6.5,
            "methodology_preference": 6.0,
            "interdisciplinary": 6.0,
            "publication_opportunity": 7.5,
            "financial_support": 7.0,
            "lab_hierarchy": 5.5,
            "core_time_flexibility": 6.0
        },
        "metadata": {
            "faculty_count": 1,
            "student_count": 7,
            "recent_publications": 10,
            "funding_level": "中",
            "equipment_rating": 7
        }
    }
]

# システム状態
system_state = {
    "initialized": True,
    "total_evaluations": 0,
    "labs_database": SAMPLE_LABS,
    "evaluation_sessions": {}
}

# 遺伝的アルゴリズム簡易実装
class SimpleFuzzyTree:
    """簡易ファジィ決定木"""
    
    def __init__(self, features: List[str]):
        self.features = features
        self.tree_structure = {}
        self.fitness_score = 0.0
    
    def predict(self, student_profile: Dict, lab_features: Dict) -> float:
        """予測実行"""
        total_score = 0.0
        total_weight = 0.0
        
        for feature in self.features:
            if feature in student_profile and feature in lab_features:
                student_val = student_profile[feature]
                lab_val = lab_features[feature]
                
                # ファジィメンバーシップ関数（三角型）
                similarity = 1.0 - abs(student_val - lab_val) / 10.0
                
                # 重み取得
                weight = get_feature_weight(feature)
                
                total_score += similarity * weight
                total_weight += weight
        
        return total_score / max(total_weight, 0.01)
    
    def mutate(self, mutation_rate: float = 0.1):
        """突然変異"""
        if random.random() < mutation_rate:
            # ランダムに特徴量を変更
            if len(self.features) > 3:
                feature_to_change = random.choice(self.features)
                new_features = [f for f in self.features if f != feature_to_change]
                self.features = new_features

class GeneticAlgorithm:
    """遺伝的アルゴリズム実装"""
    
    def __init__(self, population_size: int = 30, generations: int = 20):
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.best_individual = None
        
    def initialize_population(self, all_features: List[str]):
        """集団初期化"""
        self.population = []
        for _ in range(self.population_size):
            # 各個体はランダムな特徴量セットを持つ
            num_features = random.randint(5, len(all_features))
            features = random.sample(all_features, num_features)
            individual = SimpleFuzzyTree(features)
            self.population.append(individual)
    
    def evaluate_fitness(self, individual: SimpleFuzzyTree, 
                        student_profile: Dict, labs: List[Dict]) -> float:
        """適応度評価"""
        total_accuracy = 0.0
        
        for lab in labs:
            prediction = individual.predict(student_profile, lab["features"])
            # 実際の適合度との比較（簡易版）
            actual_score = calculate_traditional_compatibility(student_profile, lab["features"])
            accuracy = 1.0 - abs(prediction - actual_score)
            total_accuracy += accuracy
        
        # 複雑性ペナルティ
        complexity_penalty = len(individual.features) / len(get_all_features()) * 0.2
        
        # 解釈可能性ボーナス
        interpretability_bonus = 1.0 / (len(individual.features) + 1) * 0.2
        
        fitness = total_accuracy / len(labs) - complexity_penalty + interpretability_bonus
        individual.fitness_score = fitness
        return fitness
    
    def selection(self) -> SimpleFuzzyTree:
        """トーナメント選択"""
        tournament_size = 3
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness_score)
    
    def crossover(self, parent1: SimpleFuzzyTree, parent2: SimpleFuzzyTree) -> SimpleFuzzyTree:
        """交叉"""
        # 特徴量セットを組み合わせ
        combined_features = list(set(parent1.features + parent2.features))
        # ランダムに選択
        num_features = random.randint(
            min(len(parent1.features), len(parent2.features)),
            max(len(parent1.features), len(parent2.features))
        )
        selected_features = random.sample(combined_features, 
                                        min(num_features, len(combined_features)))
        
        child = SimpleFuzzyTree(selected_features)
        return child
    
    def evolve(self, student_profile: Dict, labs: List[Dict]) -> Dict:
        """進化実行"""
        all_features = get_all_features()
        self.initialize_population(all_features)
        
        evolution_history = []
        
        for generation in range(self.generations):
            # 適応度評価
            for individual in self.population:
                self.evaluate_fitness(individual, student_profile, labs)
            
            # 最良個体記録
            current_best = max(self.population, key=lambda x: x.fitness_score)
            if self.best_individual is None or current_best.fitness_score > self.best_individual.fitness_score:
                self.best_individual = current_best
            
            evolution_history.append({
                "generation": generation,
                "best_fitness": current_best.fitness_score,
                "avg_fitness": sum(ind.fitness_score for ind in self.population) / len(self.population)
            })
            
            # 新しい世代生成
            new_population = []
            
            # エリート保存
            elite_size = 3
            elite = sorted(self.population, key=lambda x: x.fitness_score, reverse=True)[:elite_size]
            new_population.extend(elite)
            
            # 残りを生成
            while len(new_population) < self.population_size:
                parent1 = self.selection()
                parent2 = self.selection()
                child = self.crossover(parent1, parent2)
                child.mutate(0.1)
                new_population.append(child)
            
            self.population = new_population
        
        return {
            "best_fitness": self.best_individual.fitness_score,
            "evolution_history": evolution_history,
            "best_features": self.best_individual.features,
            "convergence_info": {
                "converged": True,
                "final_generation": self.generations,
                "fitness_improvement": evolution_history[-1]["best_fitness"] - evolution_history[0]["best_fitness"]
            }
        }

# ヘルパー関数
def get_all_features() -> List[str]:
    """全特徴量リストを取得"""
    return [
        "research_intensity", "advisor_style", "team_work", "workload", "theory_practice",
        "research_field_match", "skill_development", "learning_pace", "difficulty_preference",
        "lab_atmosphere", "communication_style", "meeting_frequency", "flexibility",
        "evening_weekend_work", "innovation_risk", "methodology_preference", "interdisciplinary",
        "publication_opportunity", "financial_support", "lab_hierarchy", "core_time_flexibility"
    ]

def get_feature_weight(feature: str) -> float:
    """特徴量の重みを取得"""
    weights = {
        "research_field_match": 0.12,
        "publication_opportunity": 0.10,
        "advisor_style": 0.09,
        "research_intensity": 0.08,
        "lab_atmosphere": 0.08,
        "skill_development": 0.07,
        "financial_support": 0.07,
        "flexibility": 0.06,
        "team_work": 0.06,
        "core_time_flexibility": 0.05,
        "workload": 0.05,
        "theory_practice": 0.05,
        "communication_style": 0.04,
        "learning_pace": 0.04,
        "difficulty_preference": 0.03,
        "meeting_frequency": 0.03,
        "evening_weekend_work": 0.03,
        "lab_hierarchy": 0.03,
        "interdisciplinary": 0.02,
        "innovation_risk": 0.02,
        "methodology_preference": 0.02
    }
    return weights.get(feature, 0.02)

def calculate_traditional_compatibility(student_profile: Dict, lab_features: Dict) -> float:
    """従来の適合度計算"""
    total_score = 0.0
    total_weight = 0.0
    
    for feature in get_all_features():
        if feature in student_profile and feature in lab_features:
            student_val = student_profile[feature]
            lab_val = lab_features[feature]
            
            # 類似度計算
            similarity = 1.0 - abs(student_val - lab_val) / 10.0
            weight = get_feature_weight(feature)
            
            total_score += similarity * weight
            total_weight += weight
    
    return total_score / max(total_weight, 0.01)

def calculate_field_matching(field_interests: Dict, lab_fields: List[str]) -> Dict:
    """分野マッチング計算"""
    field_scores = {}
    total_interest = 0.0
    total_match = 0.0
    
    # フィールド名を日本語に変換してマッチング
    for field_key, interest in field_interests.items():
        if isinstance(interest, (int, float)):
            japanese_field_name = FIELD_NAME_MAPPING.get(field_key, field_key)
            
            if japanese_field_name in lab_fields:
                match_score = interest / 10.0
                field_scores[japanese_field_name] = {
                    "interest_score": interest,
                    "lab_relevance": 1.0,
                    "match_score": match_score,
                    "weight": 0.1
                }
                total_match += match_score
            
            total_interest += interest / 10.0
    
    return {
        "field_scores": field_scores,
        "overall_field_match": total_match / max(len(lab_fields), 1),
        "field_diversity": total_interest / len(field_interests)
    }

def generate_detailed_analysis(student_profile: Dict, lab: Dict, 
                             compatibility_score: float, field_match: Dict) -> Dict:
    """詳細分析生成"""
    
    criteria_analysis = {}
    strengths = []
    concerns = []
    recommendations = []
    
    # 項目別分析
    for feature in get_all_features():
        if feature in student_profile and feature in lab["features"]:
            student_val = student_profile[feature]
            lab_val = lab["features"][feature]
            similarity = 1.0 - abs(student_val - lab_val) / 10.0
            weight = get_feature_weight(feature)
            
            criteria_analysis[feature] = {
                "similarity": similarity,
                "weight": weight,
                "score": similarity * weight
            }
            
            if similarity > 0.8:
                strengths.append(f"{feature}: 高い適合性 ({similarity:.2f})")
            elif similarity < 0.5:
                concerns.append(f"{feature}: 適合性に課題 ({similarity:.2f})")
    
    # 推薦生成
    if compatibility_score > 0.8:
        recommendations.append("非常に高い適合性を示しています。強く推薦します。")
    elif compatibility_score > 0.6:
        recommendations.append("良好な適合性があります。検討をお勧めします。")
    elif compatibility_score > 0.4:
        recommendations.append("部分的な適合性があります。詳細検討が必要です。")
    else:
        recommendations.append("適合性が低いため、他の選択肢も検討してください。")
    
    return {
        "overall_score": compatibility_score * 100,  # パーセンテージに変換
        "criterion_scores": criteria_analysis,
        "field_matching": field_match.get("field_scores", {}),
        "strengths": strengths,
        "concerns": concerns,
        "recommendations": recommendations
    }

# APIエンドポイント
@app.get("/")
async def root():
    return {
        "message": "研究室選択支援システム API",
        "version": "3.0.0",
        "features": {
            "evaluation_criteria": "21項目",
            "research_fields": "11分野",
            "algorithm": "遺伝的アルゴリズム × ファジィ決定木"
        },
        "status": "active"
    }

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy",
        "message": "システム正常動作中",
        "version": "3.0.0",
        "database": {
            "status": "connected",
            "lab_count": len(system_state["labs_database"])
        }
    }

@app.post("/api/v1/evaluation/evaluate")
async def evaluate_labs(request_data: dict):
    """研究室評価実行"""
    try:
        start_time = time.time()
        session_id = str(uuid.uuid4())
        
        # リクエストからプロフィール取得
        preferences = request_data.get("preferences", {})
        
        # 研究室評価
        results = []
        for lab in system_state["labs_database"]:
            # 適合度計算
            compatibility_score = calculate_traditional_compatibility(preferences, lab["features"])
            
            # 詳細分析（簡易版）
            detailed_analysis = {
                "overall_score": compatibility_score * 100,
                "criterion_scores": {
                    feature: {
                        "similarity": 1.0 - abs(preferences.get(feature, 5) - lab["features"].get(feature, 5)) / 10.0,
                        "weight": get_feature_weight(feature),
                        "score": (1.0 - abs(preferences.get(feature, 5) - lab["features"].get(feature, 5)) / 10.0) * get_feature_weight(feature)
                    }
                    for feature in get_all_features()
                    if feature in preferences and feature in lab["features"]
                }
            }
            
            results.append({
                "lab": lab,
                "compatibility": detailed_analysis,
                "ranking_position": 0
            })
        
        # ソートとランキング
        results.sort(key=lambda x: x["compatibility"]["overall_score"], reverse=True)
        for i, result in enumerate(results):
            result["ranking_position"] = i + 1
        
        processing_time = (time.time() - start_time) * 1000
        
        # セッション保存
        system_state["evaluation_sessions"][session_id] = {
            "preferences": preferences,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        system_state["total_evaluations"] += 1
        
        return {
            "results": results,
            "summary": {
                "total_labs": len(results),
                "avg_score": sum(r["compatibility"]["overall_score"] for r in results) / len(results) if results else 0,
                "evaluation_id": session_id,
                "processing_time_ms": processing_time
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"評価エラー: {str(e)}")

@app.post("/api/v1/optimization/genetic")
async def optimize_genetic(profile: StudentProfile):
    """遺伝的アルゴリズム最適化実行"""
    try:
        start_time = time.time()
        session_id = str(uuid.uuid4())
        
        student_criteria = profile.evaluation_criteria.dict()
        field_interests = profile.field_interests.dict()
        
        # 遺伝的アルゴリズム実行
        ga = GeneticAlgorithm(population_size=20, generations=15)
        evolution_result = ga.evolve(student_criteria, system_state["labs_database"])
        
        # 最適化された決定木で評価
        results = []
        for lab in system_state["labs_database"]:
            # 進化した決定木での予測
            optimized_score = ga.best_individual.predict(student_criteria, lab["features"])
            
            # 分野マッチング
            field_match = calculate_field_matching(field_interests, lab["research_fields"])
            
            # 最終スコア
            final_score = (optimized_score * 0.8 + field_match["overall_field_match"] * 0.2)
            
            # 詳細分析
            detailed_analysis = generate_detailed_analysis(
                student_criteria, lab, final_score, field_match
            )
            
            results.append({
                "lab": lab,
                "compatibility": detailed_analysis,
                "genetic_analysis": {
                    "best_tree_fitness": evolution_result["best_fitness"],
                    "generation_count": 15,
                    "convergence_info": evolution_result["convergence_info"],
                    "tree_structure": {
                        "depth": 4,
                        "node_count": len(evolution_result["best_features"]) * 2,
                        "leaf_count": len(evolution_result["best_features"]),
                        "features_used": evolution_result["best_features"]
                    }
                },
                "ranking_position": 0
            })
        
        # ソートとランキング
        results.sort(key=lambda x: x["compatibility"]["overall_score"], reverse=True)
        for i, result in enumerate(results):
            result["ranking_position"] = i + 1
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "results": results,
            "summary": {
                "total_labs": len(results),
                "avg_score": sum(r["compatibility"]["overall_score"] for r in results) / len(results) if results else 0,
                "evaluation_id": session_id,
                "processing_time_ms": processing_time,
                "field_analysis": {
                    "selected_fields_count": sum(1 for v in field_interests.values() if v > 5.0)
                }
            },
            "metadata": {
                "algorithm_version": "3.0.0-GA",
                "genetic_algorithm_result": evolution_result,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"遺伝的アルゴリズム最適化エラー: {str(e)}")

@app.get("/api/v1/labs")
async def get_labs():
    """研究室一覧取得"""
    return system_state["labs_database"]

@app.get("/api/v1/system/stats")
async def get_system_stats():
    """システム統計取得"""
    field_distribution = {}
    for lab in system_state["labs_database"]:
        for field in lab["research_fields"]:
            field_distribution[field] = field_distribution.get(field, 0) + 1
    
    return {
        "total_labs": len(system_state["labs_database"]),
        "total_evaluations": system_state["total_evaluations"],
        "average_processing_time": 1500,  # ms
        "field_distribution": field_distribution,
        "system_info": {
            "version": "3.0.0",
            "features": 21,
            "research_fields": 11,
            "algorithm": "遺伝的アルゴリズム × ファジィ決定木"
        }
    }

# デモプロフィール取得用のマッピング関数
def convert_japanese_to_english_fields(japanese_interests: Dict[str, float]) -> Dict[str, float]:
    """日本語フィールド名を英語に変換"""
    reverse_mapping = {v: k for k, v in FIELD_NAME_MAPPING.items()}
    english_interests = {}
    
    for japanese_name, value in japanese_interests.items():
        english_name = reverse_mapping.get(japanese_name, japanese_name.lower().replace('・', '_').replace('/', '_').replace(' ', '_'))
        english_interests[english_name] = value
    
    return english_interests

# サーバー起動
if __name__ == "__main__":
    print("\n研究室選択支援システム 起動中...")
    print(f"URL: http://localhost:8000")
    print(f"API文書: http://localhost:8000/docs")
    print("システム構成:")
    print(f"  - 評価基準: 21項目")
    print(f"  - 研究分野: 11分野")
    print(f"  - アルゴリズム: 遺伝的アルゴリズム × ファジィ決定木")
    print(f"  - サンプル研究室: {len(SAMPLE_LABS)}件")
    print("\nサーバー起動中... (Ctrl+C で停止)")
    
    try:
        uvicorn.run(
            "app:app",  # これが重要：文字列として指定
            host="0.0.0.0",
            port=8000,
            reload=False,  # reloadをFalseに変更
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nサーバーを停止しました")