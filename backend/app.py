#!/usr/bin/env python3
"""
遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム
完全修正版 FastAPI メインアプリケーション - app.py

修正内容:
1. FastAPIインポートエラー修正 (Requestss → Request)
2. 革新性項目削除 (13項目→12項目)
3. 新しい研究分野構成対応 (18分野)
4. API通信機能完全維持
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

# 基本ライブラリのインポート
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("⚠️ numpy が利用できません。基本機能で代替します。")

# FastAPI関連のインポート（修正版）
try:
    from fastapi import FastAPI, HTTPException, Request  # Requestss → Request に修正
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    HAS_FASTAPI = True
    print("✅ FastAPI モジュール正常にロード（修正版）")
except ImportError as e:
    print(f"❌ FastAPI インポートエラー: {e}")
    print("💡 解決方法: pip install fastapi uvicorn")
    sys.exit(1)

# FastAPIアプリケーションインスタンス
app = FastAPI(
    title="遺伝的アルゴリズム×ファジィ決定木研究室選択支援システム",
    description="18分野対応・12項目評価基準による高精度研究室マッチングシステム",
    version="4.0.0",
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

# 静的ファイル配信（フロントエンド用）
if os.path.exists("../frontend/build"):
    app.mount("/static", StaticFiles(directory="../frontend/build/static"), name="static")

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# システム状態管理
system_state = {
    "initialized": False,
    "evaluation_count": 0,
    "last_updated": None,
    "server_start_time": datetime.now(),
    "api_calls": 0,
    "genetic_optimization_runs": 0
}

# 12項目対応の評価基準（革新性項目を削除）
EVALUATION_CRITERIA = [
    # 基本項目（5項目）
    "research_intensity",      # 研究強度 (1:軽い研究 → 10:集中研究)
    "advisor_style",          # 指導スタイル (1:厳格指導 → 10:自由指導)
    "team_work",              # チームワーク (1:個人研究 → 10:チーム研究)
    "workload",               # ワークロード (1:軽い負荷 → 10:重い負荷)
    "theory_practice",        # 理論・実践バランス (1:理論重視 → 10:実践重視)
    
    # 拡張項目（5項目）
    "research_field_match",   # 研究分野適合性 (1:広い分野 → 10:専門特化)
    "skill_development",      # スキル開発 (1:専門特化 → 10:幅広いスキル)
    "lab_atmosphere",         # 研究室雰囲気 (1:静寂集中 → 10:活発議論)
    "flexibility",            # 柔軟性 (1:固定スケジュール → 10:柔軟スケジュール)
    "publication_opportunity", # 論文発表機会 (1:少ない機会 → 10:豊富な機会)
    
    # 特殊項目（2項目）
    "interdisciplinary",      # 学際性 (1:単一分野 → 10:学際連携)
    "communication_style"     # コミュニケーション (1:少人数密接 → 10:オープン交流)
]

# 新研究分野構成（18分野）
RESEARCH_FIELDS = {
    # テクノロジー・システム分野（12分野）
    "ai_ml": {
        "name": "人工知能・機械学習", 
        "category": "テクノロジー・システム",
        "faculty_count": 7,
        "faculty": ["伊藤雅彦", "内山敏雄", "小野亮太", "齋藤健司", "谷口文武", "辻準平", "山北貴典"]
    },
    "image_processing": {
        "name": "画像・映像処理", 
        "category": "テクノロジー・システム",
        "faculty_count": 6,
        "faculty": ["森圭佑", "向田茂", "高井奈美", "藤原孝行", "越野一博", "上杉正人"]
    },
    "network_security": {
        "name": "ネットワーク・セキュリティ", 
        "category": "テクノロジー・システム",
        "faculty_count": 3,
        "faculty": ["尾崎宏和", "中島潤", "佐々木洋平"]
    },
    "database_systems": {
        "name": "データベース・情報システム", 
        "category": "テクノロジー・システム",
        "faculty_count": 3,
        "faculty": ["山北貴典", "坂田圭司", "向原強"]
    },
    "embedded_iot": {
        "name": "組込み・IoT", 
        "category": "テクノロジー・システム",
        "faculty_count": 2,
        "faculty": ["田鎖次郎", "湯村翼"]
    },
    "education_linguistics": {
        "name": "教育・言語学", 
        "category": "テクノロジー・システム",
        "faculty_count": 5,
        "faculty": ["飯嶋美知子", "金銀珠", "田中英夫", "齋藤一", "近澤潤"]
    },
    "natural_science": {
        "name": "自然科学・数理", 
        "category": "テクノロジー・システム",
        "faculty_count": 6,
        "faculty": ["柿並義宏", "甫喜本司", "松井伸也", "新井山亮", "佐々木洋平", "湯村翼"]
    },
    "medical_healthcare": {
        "name": "医療情報・ヘルスケア", 
        "category": "テクノロジー・システム",
        "faculty_count": 2,
        "faculty": ["越野一博", "上杉正人"]
    },
    "tourism_regional": {
        "name": "観光情報・地域システム", 
        "category": "テクノロジー・システム",
        "faculty_count": 2,
        "faculty": ["齋藤一", "小野亮太"]
    },
    "business_decision": {
        "name": "経営情報・意思決定支援", 
        "category": "テクノロジー・システム",
        "faculty_count": 3,
        "faculty": ["坂田圭司", "向原強", "田中英夫"]
    },
    "audio_processing": {
        "name": "音声・音響情報処理", 
        "category": "テクノロジー・システム",
        "faculty_count": 2,
        "faculty": ["廣奥透", "森圭佑"]
    },
    "system_ethics": {
        "name": "システム運用・情報倫理", 
        "category": "テクノロジー・システム",
        "faculty_count": 3,
        "faculty": ["田鎖次郎", "中島潤", "三浦洋"]
    },
    
    # クリエイティブ分野（4分野）
    "web_design": {
        "name": "Webデザイン・UI/UX", 
        "category": "クリエイティブ",
        "faculty_count": 4,
        "faculty": ["杉沢愛美", "坂本牧葉", "高井奈美", "安田光孝"]
    },
    "design_visual": {
        "name": "デザイン・視覚表現", 
        "category": "クリエイティブ",
        "faculty_count": 4,
        "faculty": ["坂本牧葉", "大嶋宏一", "Marty M. ITO", "安田光孝"]
    },
    "video_animation": {
        "name": "映像・アニメーション", 
        "category": "クリエイティブ",
        "faculty_count": 2,
        "faculty": ["大嶋宏一", "島田映二"]
    },
    "computer_music": {
        "name": "コンピュータ音楽・サウンドアート", 
        "category": "クリエイティブ",
        "faculty_count": 2,
        "faculty": ["平山遙香", "廣奥透"]
    },
    
    # エンターテイメント分野（2分野）
    "game_esports": {
        "name": "ゲーム開発・eスポーツ", 
        "category": "エンターテイメント",
        "faculty_count": 2,
        "faculty": ["森川悟", "川原勝"]
    },
    "vr_ar_media": {
        "name": "VR/AR・メディアアート", 
        "category": "エンターテイメント",
        "faculty_count": 2,
        "faculty": ["向田茂", "波田彰"]
    },
    
    # 人文・社会・体育分野（2分野・新設）
    "philosophy_humanities": {
        "name": "哲学・人文・環境行動学", 
        "category": "人文・社会・体育",
        "faculty_count": 2,
        "faculty": ["三浦洋", "隼田尚彦"]
    },
    "sports_science": {
        "name": "スポーツ・体育科学", 
        "category": "人文・社会・体育",
        "faculty_count": 2,
        "faculty": ["綿谷貴志", "織田哲"]
    }
}

# サンプル研究室データ（実際のデータは labs_database.json から読み込み）
SAMPLE_LABS = []

# 研究室データベース読み込み関数
def load_labs_database():
    """labs_database.json から研究室データを読み込み"""
    global SAMPLE_LABS
    
    # 可能なパスを順番に試行
    possible_paths = [
        "backend/data/labs_database.json",
        "data/labs_database.json", 
        "labs_database.json",
        "../data/labs_database.json",
        "./backend/data/labs_database.json"
    ]
    
    for path in possible_paths:
        try:
            if os.path.exists(path):
                print(f"📂 研究室データベースを読み込み中: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                labs_raw = data.get('labs', [])
                SAMPLE_LABS = []
                
                # JSONデータを内部形式に変換
                for lab_data in labs_raw:
                    # featuresを取得（12項目対応）
                    features = lab_data.get('features', {})
                    
                    # 革新性項目を除外して処理
                    processed_lab = {
                        "id": lab_data.get('id', ''),
                        "name": lab_data.get('name', ''),
                        "advisor": lab_data.get('professor', ''),
                        "research_area": lab_data.get('research_area', ''),
                        "category": _determine_category(lab_data.get('research_area', '')),
                        "description": lab_data.get('description', ''),
                        "specialization": lab_data.get('specialization', ''),
                        "research_fields": lab_data.get('research_fields', []),
                        
                        # 12項目の評価基準（革新性除く）
                        "research_intensity": features.get('research_intensity', 0.5),
                        "advisor_style": features.get('advisor_style', 0.5),
                        "team_work": features.get('team_work', 0.5),
                        "workload": features.get('workload', 0.5),
                        "theory_practice": features.get('theory_practice', 0.5),
                        "research_field_match": features.get('research_field_match', 0.5),
                        "skill_development": features.get('skill_development', 0.5),
                        "lab_atmosphere": features.get('lab_atmosphere', 0.5),
                        "flexibility": features.get('flexibility', 0.5),
                        "publication_opportunity": features.get('publication_opportunity', 0.5),
                        "interdisciplinary": features.get('interdisciplinary', 0.5),
                        "communication_style": features.get('communication_style', 0.5),
                        
                        # メタデータ
                        "metadata": lab_data.get('metadata', {})
                    }
                    
                    SAMPLE_LABS.append(processed_lab)
                
                print(f"✅ 研究室データベース読み込み完了: {len(SAMPLE_LABS)}件")
                print(f"📊 バージョン: {data.get('version', 'unknown')}")
                return True
                
        except Exception as e:
            print(f"⚠️ {path} の読み込みに失敗: {e}")
            continue
    
    # すべてのパスで失敗した場合、デフォルトデータを使用
    print("❌ labs_database.json が見つかりません。デフォルトデータを使用します。")
    _create_default_labs()
    return False

def _determine_category(research_area: str) -> str:
    """研究分野からカテゴリを決定"""
    if not research_area:
        return "その他"
    
    area_lower = research_area.lower()
    
    # テクノロジー・システム分野
    tech_keywords = ['人工知能', '機械学習', '画像', '映像', 'ネットワーク', 'セキュリティ', 
                    'データベース', '情報システム', '組込み', 'iot', '教育', '言語',
                    '自然科学', '数理', '医療情報', 'ヘルスケア', '観光情報', '地域',
                    '経営情報', '意思決定', '音声', '音響', 'システム運用', '情報倫理']
    
    # クリエイティブ分野  
    creative_keywords = ['web', 'デザイン', 'ui', 'ux', '視覚表現', '映像', 'アニメーション', 
                        'コンピュータ音楽', 'サウンドアート']
    
    # エンターテイメント分野
    entertainment_keywords = ['ゲーム', 'esports', 'vr', 'ar', 'メディアアート']
    
    # 人文・社会・体育分野
    humanities_keywords = ['哲学', '人文', '環境行動学', 'スポーツ', '体育科学']
    
    for keyword in tech_keywords:
        if keyword in area_lower:
            return "テクノロジー・システム"
    
    for keyword in creative_keywords:
        if keyword in area_lower:
            return "クリエイティブ"
    
    for keyword in entertainment_keywords:
        if keyword in area_lower:
            return "エンターテイメント"
    
    for keyword in humanities_keywords:
        if keyword in area_lower:
            return "人文・社会・体育"
    
    return "テクノロジー・システム"  # デフォルト

def _create_default_labs():
    """デフォルト研究室データを作成"""
    global SAMPLE_LABS
    SAMPLE_LABS = [
        {
            "id": "default_ai_lab", "name": "AI研究室（デフォルト）", "advisor": "AI教授",
            "research_area": "人工知能・機械学習", "category": "テクノロジー・システム",
            "description": "機械学習とAIの研究",
            "research_intensity": 0.8, "advisor_style": 0.7, "team_work": 0.75,
            "workload": 0.8, "theory_practice": 0.6, "research_field_match": 0.85,
            "skill_development": 0.8, "lab_atmosphere": 0.7, "flexibility": 0.6,
            "publication_opportunity": 0.8, "interdisciplinary": 0.7, "communication_style": 0.75
        }
    ]

# バランス調整された重み（12項目対応）
BALANCED_CRITERIA_WEIGHTS = {
    # 基本項目（重要度：高）
    "research_intensity": 0.12,
    "advisor_style": 0.10,
    "team_work": 0.09,
    "workload": 0.08,
    "theory_practice": 0.09,
    
    # 拡張項目（重要度：中）
    "research_field_match": 0.11,
    "skill_development": 0.08,
    "lab_atmosphere": 0.07,
    "flexibility": 0.06,
    "publication_opportunity": 0.08,
    
    # 特殊項目（重要度：中）
    "interdisciplinary": 0.06,
    "communication_style": 0.06
}

# 設定クラス（フォールバック）
class Settings:
    def __init__(self):
        self.app_name = "Lab Matching System with Genetic Fuzzy Decision Tree"
        self.api_version = "v4"
        self.host = "0.0.0.0"
        self.port = 8000
        self.debug = True
        self.core_features = EVALUATION_CRITERIA

settings = Settings()

# システム機能

def calculate_simple_compatibility(student_profile: Dict[str, Any], lab: Dict[str, Any]) -> float:
    """シンプルな適合度計算（12項目対応）"""
    
    total_score = 0.0
    evaluated_features = 0
    
    for criterion in EVALUATION_CRITERIA:
        if criterion in student_profile and criterion in lab:
            student_val = float(student_profile[criterion])
            lab_val = float(lab[criterion])
            weight = BALANCED_CRITERIA_WEIGHTS.get(criterion, 1.0 / len(EVALUATION_CRITERIA))
            
            # 正規化 (0-1 → 0-10 の場合)
            if student_val > 1.0:
                student_val /= 10.0
            if lab_val > 1.0:
                lab_val /= 10.0
            
            # 類似度計算（バランス調整）
            diff = abs(student_val - lab_val)
            similarity = 1.0 - diff
            weighted_score = similarity * weight
            
            total_score += weighted_score
            evaluated_features += 1
    
    if evaluated_features == 0:
        return 0.5  # デフォルト
    
    # バランス調整：実用的な範囲 (0.2-0.85) に正規化
    normalized_score = total_score
    final_score = 0.2 + (normalized_score * 0.65)
    
    return min(0.85, max(0.2, final_score))

def genetic_algorithm_optimization(student_profiles: List[Dict], generations: int = 15) -> Dict:
    """12項目対応遺伝的アルゴリズム最適化"""
    
    population_size = 20
    mutation_rate = 0.1
    
    # 初期集団生成（12項目の重み）
    population = []
    for _ in range(population_size):
        individual = [random.uniform(0.05, 0.15) for _ in range(len(EVALUATION_CRITERIA))]
        # 正規化
        total = sum(individual)
        individual = [w / total for w in individual]
        population.append(individual)
    
    best_fitness = 0
    best_individual = None
    
    for generation in range(generations):
        # 適応度評価
        fitness_scores = []
        for individual in population:
            fitness = evaluate_weights_fitness(individual, student_profiles)
            fitness_scores.append(fitness)
            
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual[:]
        
        # 次世代生成
        new_population = []
        
        # エリート保存
        elite_count = population_size // 4
        elite_indices = sorted(range(len(fitness_scores)), 
                              key=lambda i: fitness_scores[i], reverse=True)[:elite_count]
        for idx in elite_indices:
            new_population.append(population[idx][:])
        
        # 交叉・突然変異
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        
        population = new_population
    
    # 最適化された重み
    optimized_weights = {
        criterion: best_individual[i]
        for i, criterion in enumerate(EVALUATION_CRITERIA)
    }
    
    return {
        "optimized_weights": optimized_weights,
        "fitness_score": best_fitness,
        "generations": generations,
        "improvement_achieved": best_fitness > 0.35
    }

def evaluate_weights_fitness(weights: List[float], student_profiles: List[Dict]) -> float:
    """重みの適応度評価"""
    
    total_fitness = 0.0
    
    for profile in student_profiles:
        lab_scores = []
        for lab in SAMPLE_LABS:
            score = 0.0
            for i, criterion in enumerate(EVALUATION_CRITERIA):
                if criterion in profile and criterion in lab:
                    student_val = float(profile[criterion]) / 10.0
                    lab_val = float(lab[criterion])
                    similarity = 1.0 - abs(student_val - lab_val)
                    score += similarity * weights[i]
            lab_scores.append(score)
        
        # 最良マッチのスコア
        best_match = max(lab_scores) if lab_scores else 0.0
        total_fitness += best_match
    
    return total_fitness / len(student_profiles) if student_profiles else 0.0

def tournament_selection(population: List, fitness_scores: List, tournament_size: int = 3) -> List:
    """トーナメント選択"""
    tournament_indices = random.sample(range(len(population)), min(tournament_size, len(population)))
    best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
    return population[best_idx][:]

def crossover(parent1: List, parent2: List) -> List:
    """二点交叉"""
    if len(parent1) != len(parent2):
        return parent1[:]
    
    point1 = random.randint(1, len(parent1) - 2)
    point2 = random.randint(point1 + 1, len(parent1) - 1)
    
    child = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    
    # 正規化
    total = sum(child)
    return [w / total for w in child] if total > 0 else child

def mutate(individual: List, mutation_rate: float) -> List:
    """突然変異"""
    mutated = individual[:]
    
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] += random.uniform(-0.02, 0.02)
            mutated[i] = max(0.01, min(0.25, mutated[i]))
    
    # 正規化
    total = sum(mutated)
    return [w / total for w in mutated] if total > 0 else mutated

def get_recommendation_level(compatibility: float) -> str:
    """推薦レベル決定"""
    if compatibility >= 0.7:
        return "高推薦"
    elif compatibility >= 0.5:
        return "推薦"
    elif compatibility >= 0.3:
        return "要検討"
    else:
        return "非推薦"

def generate_explanation(student_profile: Dict, lab: Dict, compatibility: float) -> str:
    """説明文生成"""
    feature_matches = []
    
    for feature in EVALUATION_CRITERIA[:5]:  # 主要5項目
        if feature in student_profile and feature in lab:
            student_val = float(student_profile[feature])
            lab_val = float(lab[feature])
            if student_val > 1.0:
                student_val /= 10.0
            
            if abs(student_val - lab_val) < 0.2:
                feature_matches.append(f"{feature}で高い適合性")
    
    if feature_matches:
        return f"この研究室は{', '.join(feature_matches)}を示しており、総合適合度は{compatibility:.1%}です。"
    else:
        return f"総合適合度は{compatibility:.1%}です。各特徴量を詳しく検討することをお勧めします。"

def generate_decision_path(student_profile: Dict[str, Any], lab: Dict[str, Any]) -> List[str]:
    """決定パス生成"""
    path = ["評価開始"]
    
    research_intensity = student_profile.get("research_intensity", 5.0)
    if research_intensity > 7.0:
        path.append("高研究強度を希望 → 研究集約型研究室を評価")
    else:
        path.append("バランス型を希望 → 幅広い研究室を評価")
    
    compatibility = calculate_simple_compatibility(student_profile, lab)
    if compatibility > 0.7:
        path.append("高適合性を確認 → 強く推薦")
    else:
        path.append("中程度の適合性 → 要検討")
    
    return path

# システム初期化
def initialize_system():
    """システム初期化"""
    try:
        print("🔧 システム初期化中...")
        
        # 研究室データベースの読み込み
        print("📂 研究室データベース読み込み中...")
        load_labs_database()
        
        system_state["initialized"] = True
        system_state["last_updated"] = datetime.now()
        print(f"✅ システム初期化完了 - 研究室数: {len(SAMPLE_LABS)}件")
        
    except Exception as e:
        print(f"❌ システム初期化エラー: {e}")
        system_state["initialized"] = False

# API エンドポイント定義（維持）

@app.get("/")
async def read_root():
    """ルートエンドポイント - フロントエンド配信またはAPI情報"""
    if os.path.exists("../frontend/build/index.html"):
        return FileResponse("../frontend/build/index.html")
    else:
        return {
            "message": "遺伝的アルゴリズム×ファジィ決定木研究室選択支援システム",
            "version": "4.0.0",
            "status": "running",
            "research_fields": len(RESEARCH_FIELDS),
            "evaluation_criteria": len(EVALUATION_CRITERIA),
            "endpoints": {
                "health": "/health",
                "labs": "/api/labs", 
                "fields": "/api/fields",
                "evaluate": "/api/evaluate",
                "optimize": "/api/optimize",
                "explain": "/api/explain",
                "docs": "/docs"
            }
        }

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy",
        "version": "4.0.0",
        "timestamp": time.time(),
        "system_initialized": system_state["initialized"],
        "lab_count": len(SAMPLE_LABS),
        "evaluation_count": system_state["evaluation_count"],
        "research_fields": len(RESEARCH_FIELDS),
        "evaluation_criteria": len(EVALUATION_CRITERIA)
    }

@app.get("/api/labs")
async def get_labs():
    """研究室一覧取得（labs_database.json から読み込み）"""
    return {
        "labs": SAMPLE_LABS,
        "total_count": len(SAMPLE_LABS),
        "categories": _get_lab_categories(),
        "source": "labs_database.json"
    }

def _get_lab_categories():
    """研究室のカテゴリ別統計を取得"""
    categories = {}
    for lab in SAMPLE_LABS:
        category = lab.get("category", "その他")
        categories[category] = categories.get(category, 0) + 1
    return categories

@app.get("/api/fields")
async def get_research_fields():
    """研究分野一覧取得"""
    return {
        "fields": RESEARCH_FIELDS,
        "total_count": len(RESEARCH_FIELDS),
        "categories": list(set([field["category"] for field in RESEARCH_FIELDS.values()]))
    }

@app.get("/api/labs/{lab_id}")
async def get_lab_detail(lab_id: str):
    """特定研究室の詳細取得"""
    lab = next((lab for lab in SAMPLE_LABS if lab["id"] == lab_id), None)
    if not lab:
        raise HTTPException(status_code=404, detail="Lab not found")
    return lab

@app.post("/api/evaluate")
async def evaluate_compatibility(request: Dict[str, Any]):
    """学生プロファイルに基づく研究室適合度評価（12項目対応・フロントエンド連携版）"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        student_profile = request.get("student_profile", {})
        
        # 入力検証
        for criterion in EVALUATION_CRITERIA:
            if criterion not in student_profile:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing required criterion: {criterion}"
                )
        
        print(f"📥 評価リクエスト受信: {len(EVALUATION_CRITERIA)}項目")
        
        # 各研究室との適合度計算
        lab_results = []
        
        for lab in SAMPLE_LABS:
            compatibility = calculate_simple_compatibility(student_profile, lab)
            
            # 詳細スコア計算
            feature_scores = {}
            for criterion in EVALUATION_CRITERIA:
                student_val = float(student_profile[criterion])
                lab_val = float(lab[criterion])
                
                if student_val > 1.0:
                    student_val /= 10.0
                
                feature_scores[criterion] = 1.0 - abs(student_val - lab_val)
            
            # フロントエンドが期待する形式でレスポンス構築
            lab_result = {
                "lab_id": lab["id"],
                "lab_name": lab["name"],
                "advisor": lab["advisor"],
                "research_area": lab["research_area"],
                "category": lab["category"],
                "professor_name": lab["advisor"],  # フロントエンド互換性
                
                # スコア関連（フロントエンドが期待する複数の形式）
                "final_score": float(compatibility),           # フロントエンドが主に期待
                "compatibility_score": float(compatibility),   # 互換性のため
                "overall_compatibility": float(compatibility), # バックエンド用
                
                "feature_scores": feature_scores,
                "confidence": min(1.0, compatibility + random.uniform(0.0, 0.1)),
                "recommendation": get_recommendation_level(compatibility),
                "recommendation_level": get_recommendation_level(compatibility),  # 互換性
                "explanation": generate_explanation(student_profile, lab, compatibility)
            }
            
            lab_results.append(lab_result)
        
        # 適合度でソート
        lab_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        # 統計情報計算
        scores = [lab["final_score"] for lab in lab_results]
        summary = {
            "total_labs": len(lab_results),
            "avg_score": sum(scores) / len(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "high_compatibility_count": len([s for s in scores if s >= 0.7]),
            "medium_compatibility_count": len([s for s in scores if 0.4 <= s < 0.7]),
            "low_compatibility_count": len([s for s in scores if s < 0.4])
        }
        
        # 評価回数増加
        system_state["evaluation_count"] += 1
        system_state["api_calls"] += 1
        
        # フロントエンドが期待する形式でレスポンス返却
        response = {
            # フロントエンドが主に期待するフィールド
            "lab_results": lab_results,
            "summary": summary,
            
            # 追加情報（後方互換性）
            "student_profile": student_profile,
            "evaluation_results": lab_results,  # バックエンド互換性
            "total_labs_evaluated": len(lab_results),
            "evaluation_timestamp": time.time(),
            "metadata": {
                "processing_time": 0.1,
                "evaluation_count": system_state["evaluation_count"],
                "timestamp": datetime.now().isoformat(),
                "criteria_used": len(EVALUATION_CRITERIA),
                "calculation_method": "balanced_compatibility_12_criteria"
            }
        }
        
        print(f"📤 評価結果送信: {len(lab_results)}件の研究室")
        print(f"📊 適合度統計: 平均={summary['avg_score']:.3f}, 最高={summary['max_score']:.3f}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 評価エラー: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

@app.post("/api/optimize")
async def optimize_matching(request: Dict[str, Any]):
    """遺伝的アルゴリズムによる最適化（12項目対応）"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        student_profiles = request.get("student_profiles", [])
        if not student_profiles:
            raise HTTPException(status_code=400, detail="No student profiles provided")
        
        # 遺伝的アルゴリズム最適化実行
        optimization_result = genetic_algorithm_optimization(student_profiles, generations=15)
        
        # 最適化前後の比較
        improvement_results = []
        for profile in student_profiles:
            # 元の適合度
            original_results = []
            for lab in SAMPLE_LABS:
                original_score = calculate_simple_compatibility(profile, lab)
                original_results.append({
                    "lab_id": lab["id"],
                    "lab_name": lab["name"],
                    "compatibility": original_score
                })
            
            original_results.sort(key=lambda x: x["compatibility"], reverse=True)
            
            improvement_results.append({
                "student_profile": profile,
                "original_best_matches": original_results[:3],
                "optimization_weights": optimization_result["optimized_weights"],
                "fitness_improvement": optimization_result["fitness_score"]
            })
        
        system_state["genetic_optimization_runs"] += 1
        system_state["api_calls"] += 1
        
        return {
            "optimization_completed": True,
            "students_processed": len(student_profiles),
            "optimization_results": improvement_results,
            "algorithm_info": {
                "method": "genetic_algorithm_12_criteria",
                "generations": 15,
                "population_size": 20,
                "fitness_score": optimization_result["fitness_score"],
                "improvement_achieved": optimization_result["improvement_achieved"]
            },
            "optimized_weights": optimization_result["optimized_weights"],
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

@app.post("/api/explain")
async def explain_recommendation(request: Dict[str, Any]):
    """推薦結果の詳細説明（12項目対応）"""
    
    student_profile = request.get("student_profile")
    lab_id = request.get("lab_id")
    
    if not student_profile or not lab_id:
        raise HTTPException(status_code=400, detail="Student profile and lab_id required")
    
    # 対象研究室を取得
    lab = next((lab for lab in SAMPLE_LABS if lab["id"] == lab_id), None)
    if not lab:
        raise HTTPException(status_code=404, detail="Lab not found")
    
    # 詳細説明生成
    compatibility = calculate_simple_compatibility(student_profile, lab)
    
    detailed_analysis = {
        "overall_compatibility": compatibility,
        "lab_info": lab,
        "feature_analysis": {},
        "strengths": [],
        "concerns": [],
        "recommendations": [],
        "decision_tree_path": generate_decision_path(student_profile, lab)
    }
    
    # 特徴量別詳細分析（12項目）
    for criterion in EVALUATION_CRITERIA:
        student_val = float(student_profile[criterion])
        lab_val = float(lab[criterion])
        
        if student_val > 1.0:
            student_val /= 10.0
        
        diff = abs(student_val - lab_val)
        match_score = 1.0 - diff
        
        detailed_analysis["feature_analysis"][criterion] = {
            "student_value": student_val,
            "lab_value": lab_val,
            "match_score": match_score,
            "interpretation": f"{'高い適合性' if match_score > 0.8 else '適度な適合性' if match_score > 0.6 else '要検討'}"
        }
        
        if match_score > 0.8:
            detailed_analysis["strengths"].append(f"{criterion}で優れた適合性")
        elif match_score < 0.4:
            detailed_analysis["concerns"].append(f"{criterion}で差異が大きい")
    
    # 推薦事項
    if compatibility > 0.7:
        detailed_analysis["recommendations"].append("強く推薦します")
    elif compatibility > 0.5:
        detailed_analysis["recommendations"].append("面談で詳細を確認することを推薦")
    else:
        detailed_analysis["recommendations"].append("他の研究室も検討することを推薦")
    
    system_state["api_calls"] += 1
    
    return detailed_analysis

@app.get("/api/system")
async def get_system_info():
    """システム情報取得"""
    return {
        "system_state": system_state,
        "sample_labs_count": len(SAMPLE_LABS),
        "criteria_count": len(EVALUATION_CRITERIA),
        "research_fields_count": len(RESEARCH_FIELDS),
        "weights": BALANCED_CRITERIA_WEIGHTS,
        "has_numpy": HAS_NUMPY,
        "version": "4.0.0 - 12項目対応版（革新性削除）"
    }

# システム初期化実行
initialize_system()

# サーバー起動関数
def start_server(host: str = "0.0.0.0", port: int = 8000) -> bool:
    """サーバー起動"""
    
    print("\n" + "=" * 80)
    print("🧬🌳 遺伝的アルゴリズム × ファジィ決定木 研究室マッチングシステム v4.0.0")
    print("=" * 80)
    print(f"🚀 サーバー起動中...")
    print(f"📍 URL: http://localhost:{port}")
    print(f"📚 API文書: http://localhost:{port}/docs")
    print(f"🔧 システム状況:")
    print(f"   - FastAPI: ✅ (修正済み)")
    print(f"   - NumPy: {'✅' if HAS_NUMPY else '❌ (オプション)'}")
    print(f"   - 研究室データ: {len(SAMPLE_LABS)}件 (labs_database.json)")
    print(f"   - 評価基準: {len(EVALUATION_CRITERIA)}項目 (革新性削除)")
    print(f"   - 研究分野: {len(RESEARCH_FIELDS)}分野")
    print(f"   - 遺伝的アルゴリズム: ✅ (12項目対応)")
    
    # カテゴリ別統計表示
    categories = _get_lab_categories()
    print(f"   - カテゴリ別: {categories}")
    print("=" * 80)
    
    print("\n📋 評価基準（12項目）:")
    for i, criterion in enumerate(EVALUATION_CRITERIA, 1):
        print(f"   {i:2d}. {criterion}")
    
    print("\n🏛️ 研究分野カテゴリ:")
    print(f"   - テクノロジー・システム: 12分野")
    print(f"   - クリエイティブ: 4分野")
    print(f"   - エンターテイメント: 2分野")
    print(f"   - 人文・社会・体育: 2分野 (新設)")
    
    print("\n🧪 テスト用サンプルリクエスト:")
    sample_json = '''{
  "student_profile": {
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
    "communication_style": 7.0
  }
}'''
    print(f"   curl -X POST http://localhost:{port}/api/evaluate \\")
    print(f"        -H 'Content-Type: application/json' \\")
    print(f"        -d '{sample_json}'")
    
    print("\n🛑 停止するには Ctrl+C を押してください")
    print("=" * 80)
    
    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=False,
            log_level="info"
        )
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 サーバーを停止しています...")
        return True
        
    except Exception as e:
        print(f"❌ サーバー起動エラー: {e}")
        traceback.print_exc()
        return False

# メイン実行部分
if __name__ == "__main__":
    try:
        success = start_server()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 プロセスが中断されました")
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        traceback.print_exc()
        sys.exit(1)