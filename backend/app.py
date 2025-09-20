#!/usr/bin/env python3
"""
遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム
完全版 FastAPI メインアプリケーション - app.py (バランス調整済み)
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

# FastAPI関連のインポート
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import uvicorn
    HAS_FASTAPI = True
    print("✅ FastAPI モジュール正常にロード")
except ImportError as e:
    print(f"❌ FastAPI インポートエラー: {e}")
    print("💡 解決方法: pip install fastapi uvicorn")
    sys.exit(1)

# FastAPIアプリケーションインスタンスを最初に作成
app = FastAPI(
    title="研究室選択支援システム",
    description="遺伝的アルゴリズムを用いたファジィ決定木による研究室マッチングシステム",
    version="3.1.0",
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

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# システム状態管理
system_state = {
    "initialized": False,
    "evaluation_count": 0,
    "last_updated": None,
    "server_start_time": datetime.now(),
    "api_calls": 0
}

# 13項目完全対応の評価基準
COMPLETE_EVALUATION_CRITERIA = [
    "research_intensity",      # 研究強度
    "advisor_style",          # 指導スタイル 
    "team_work",              # チームワーク
    "workload",               # ワークロード
    "theory_practice",        # 理論・実践バランス
    "research_field_match",   # 研究分野適合性
    "skill_development",      # スキル開発
    "lab_atmosphere",         # 研究室雰囲気
    "flexibility",            # 柔軟性
    "publication_opportunity", # 論文発表機会
    "interdisciplinary",      # 学際性
    "communication_style",    # コミュニケーション
    "innovation_risk"         # 革新性・リスク許容度
]

# サンプル研究室データ（42件の多様な研究室）
SAMPLE_LABS = [
    # AI・機械学習分野（7件）
    {"name": "伊藤雅彦研究室", "research_area": "人工知能・機械学習", "professor": "伊藤雅彦"},
    {"name": "内山敏雄研究室", "research_area": "人工知能・機械学習", "professor": "内山敏雄"},
    {"name": "小野亮太研究室", "research_area": "人工知能・機械学習", "professor": "小野亮太"},
    {"name": "齋藤健司研究室", "research_area": "人工知能・機械学習", "professor": "齋藤健司"},
    {"name": "谷口文武研究室", "research_area": "人工知能・機械学習", "professor": "谷口文武"},
    {"name": "辻準平研究室", "research_area": "人工知能・機械学習", "professor": "辻準平"},
    {"name": "山北貴典研究室", "research_area": "データベース・情報システム", "professor": "山北貴典"},
    
    # 画像・映像処理分野（6件）
    {"name": "森圭佑研究室", "research_area": "画像・映像処理", "professor": "森圭佑"},
    {"name": "向田茂研究室", "research_area": "画像・映像処理", "professor": "向田茂"},
    {"name": "高井奈美研究室", "research_area": "画像・映像処理", "professor": "高井奈美"},
    {"name": "藤原孝行研究室", "research_area": "画像・映像処理", "professor": "藤原孝行"},
    {"name": "越野一博研究室", "research_area": "医療情報・ヘルスケア", "professor": "越野一博"},
    {"name": "上杉正人研究室", "research_area": "医療情報・ヘルスケア", "professor": "上杉正人"},
    
    # ネットワーク・セキュリティ分野（3件）
    {"name": "尾崎宏和研究室", "research_area": "ネットワーク・セキュリティ", "professor": "尾崎宏和"},
    {"name": "中島潤研究室", "research_area": "ネットワーク・セキュリティ", "professor": "中島潤"},
    {"name": "佐々木洋平研究室", "research_area": "自然科学・数理", "professor": "佐々木洋平"},
    
    # データベース・情報システム分野（3件）
    {"name": "坂田圭司研究室", "research_area": "経営情報・意思決定支援", "professor": "坂田圭司"},
    {"name": "向原強研究室", "research_area": "経営情報・意思決定支援", "professor": "向原強"},
    
    # 組込み・IoT分野（2件）
    {"name": "田鎖次郎研究室", "research_area": "システム運用・情報倫理", "professor": "田鎖次郎"},
    {"name": "湯村翼研究室", "research_area": "組込み・IoT", "professor": "湯村翼"},
    
    # 教育・言語学分野（5件）
    {"name": "飯嶋美知子研究室", "research_area": "教育・言語学", "professor": "飯嶋美知子"},
    {"name": "金銀珠研究室", "research_area": "教育・言語学", "professor": "金銀珠"},
    {"name": "田中英夫研究室", "research_area": "教育・言語学", "professor": "田中英夫"},
    {"name": "齋藤一研究室", "research_area": "教育・言語学", "professor": "齋藤一"},
    {"name": "近澤潤研究室", "research_area": "教育・言語学", "professor": "近澤潤"},
    
    # 自然科学・数理分野（6件）
    {"name": "柿並義宏研究室", "research_area": "自然科学・数理", "professor": "柿並義宏"},
    {"name": "甫喜本司研究室", "research_area": "自然科学・数理", "professor": "甫喜本司"},
    {"name": "松井伸也研究室", "research_area": "自然科学・数理", "professor": "松井伸也"},
    {"name": "新井山亮研究室", "research_area": "自然科学・数理", "professor": "新井山亮"},
    
    # クリエイティブ分野（8件）
    {"name": "杉沢愛美研究室", "research_area": "Webデザイン・UI/UX", "professor": "杉沢愛美"},
    {"name": "坂本牧葉研究室", "research_area": "デザイン・視覚表現", "professor": "坂本牧葉"},
    {"name": "大嶋宏一研究室", "research_area": "映像・アニメーション", "professor": "大嶋宏一"},
    {"name": "島田映二研究室", "research_area": "映像・アニメーション", "professor": "島田映二"},
    {"name": "平山遙香研究室", "research_area": "コンピュータ音楽・サウンドアート", "professor": "平山遙香"},
    {"name": "廣奥透研究室", "research_area": "音声・音響情報処理", "professor": "廣奥透"},
    
    # エンターテイメント分野（4件）
    {"name": "森川悟研究室", "research_area": "ゲーム開発・eスポーツ", "professor": "森川悟"},
    {"name": "川原勝研究室", "research_area": "ゲーム開発・eスポーツ", "professor": "川原勝"},
    {"name": "波田彰研究室", "research_area": "VR/AR・メディアアート", "professor": "波田彰"},
    
    # 人文・社会・体育分野（4件）
    {"name": "三浦洋研究室", "research_area": "システム運用・情報倫理", "professor": "三浦洋"},
    {"name": "隼田尚彦研究室", "research_area": "哲学・人文・環境行動学", "professor": "隼田尚彦"},
    {"name": "綿谷貴志研究室", "research_area": "スポーツ・体育科学", "professor": "綿谷貴志"},
    {"name": "織田哲研究室", "research_area": "スポーツ・体育科学", "professor": "織田哲"}
]

# バランスの取れた重み設定（実用的な差別化）
BALANCED_CRITERIA_WEIGHTS = {
    "research_field_match": 2.0,          # 分野一致は重要（緩和）
    "research_intensity": 1.8,            # 研究強度
    "advisor_style": 1.5,                 # 指導スタイル
    "theory_practice": 1.4,               # 理論実践バランス
    "workload": 1.2,                      # 作業負荷
    "team_work": 1.0,
    "skill_development": 1.0,
    "lab_atmosphere": 1.0,
    "publication_opportunity": 1.0,
    "flexibility": 0.9,
    "interdisciplinary": 0.8,
    "communication_style": 0.8,
    "innovation_risk": 0.7
}

def calculate_balanced_similarity(student_val: float, lab_val: float) -> float:
    """バランスの取れた類似度計算"""
    diff = abs(student_val - lab_val)
    
    # より寛容な類似度関数
    if diff <= 0.5:
        return 1.0          # 非常に近い
    elif diff <= 1.0:
        return 0.9          # 近い
    elif diff <= 1.5:
        return 0.8          # やや近い  
    elif diff <= 2.0:
        return 0.65         # 普通
    elif diff <= 2.5:
        return 0.5          # やや違う
    elif diff <= 3.0:
        return 0.35         # 違う
    elif diff <= 4.0:
        return 0.2          # かなり違う
    else:
        return 0.1          # 全く違う

def get_balanced_lab_value(lab_data: Dict[str, Any], criterion: str, lab_name: str) -> float:
    """バランスの取れた研究室特性値を生成"""
    
    if criterion in lab_data:
        return float(lab_data[criterion])
    
    # 研究室名から一意のハッシュ値を生成
    name_hash = hash(lab_name + criterion) % 10000
    
    # より現実的な研究分野別プロファイル
    field_profiles = {
        "AI研究室": {
            "research_intensity": 8.0,
            "theory_practice": 6.0,
            "innovation_risk": 7.5,
            "workload": 7.5,
            "advisor_style": 6.0,
            "team_work": 7.0,
            "publication_opportunity": 8.0
        },
        "デザイン研究室": {
            "research_intensity": 6.5,
            "theory_practice": 8.0,
            "innovation_risk": 7.0,
            "workload": 6.0,
            "advisor_style": 7.5,
            "flexibility": 8.0,
            "lab_atmosphere": 8.5
        },
        "理論研究室": {
            "research_intensity": 7.5,
            "theory_practice": 4.0,
            "innovation_risk": 5.5,
            "workload": 7.0,
            "advisor_style": 5.0,
            "publication_opportunity": 8.0,
            "interdisciplinary": 6.0
        },
        "システム研究室": {
            "research_intensity": 7.0,
            "theory_practice": 7.5,
            "innovation_risk": 6.5,
            "workload": 7.5,
            "team_work": 8.0,
            "skill_development": 8.0,
            "communication_style": 7.5
        },
        "実験研究室": {
            "research_intensity": 8.5,
            "theory_practice": 8.5,
            "workload": 8.0,
            "team_work": 7.5,
            "skill_development": 8.5,
            "publication_opportunity": 7.5,
            "innovation_risk": 7.0
        }
    }
    
    # 研究室タイプを判定
    lab_type = "一般研究室"
    if any(keyword in lab_name for keyword in ["人工知能", "機械学習", "AI", "データ"]):
        lab_type = "AI研究室"
    elif any(keyword in lab_name for keyword in ["デザイン", "UI", "UX", "視覚", "映像"]):
        lab_type = "デザイン研究室"
    elif any(keyword in lab_name for keyword in ["数理", "理論", "哲学"]):
        lab_type = "理論研究室"
    elif any(keyword in lab_name for keyword in ["システム", "組込み", "ネットワーク"]):
        lab_type = "システム研究室"
    elif any(keyword in lab_name for keyword in ["医療", "スポーツ", "実験"]):
        lab_type = "実験研究室"
    
    # プロファイル値を取得
    if lab_type in field_profiles and criterion in field_profiles[lab_type]:
        base_value = field_profiles[lab_type][criterion]
        # 適度なランダム性を加える
        variation = (name_hash % 30 - 15) / 30.0  # -0.5 to 0.5
        return max(1.0, min(10.0, base_value + variation))
    
    # 一般的なデフォルト値（適度な分散）
    base_ranges = {
        "research_intensity": (5.5, 8.5),
        "advisor_style": (4.5, 8.5),
        "team_work": (4.5, 8.5),
        "workload": (5.0, 8.0),
        "theory_practice": (4.0, 8.0),
        "research_field_match": (6.0, 8.5),
        "skill_development": (5.0, 8.0),
        "lab_atmosphere": (5.0, 8.5),
        "flexibility": (4.5, 8.5),
        "publication_opportunity": (4.5, 8.5),
        "interdisciplinary": (4.0, 8.0),
        "communication_style": (5.0, 8.5),
        "innovation_risk": (4.5, 8.0)
    }
    
    min_val, max_val = base_ranges.get(criterion, (5.0, 8.0))
    range_val = max_val - min_val
    normalized_hash = (name_hash % 1000) / 1000.0
    return min_val + (normalized_hash * range_val)

def calculate_balanced_compatibility(
    student_profile: Dict[str, float], 
    lab_data: Dict[str, Any]
) -> Dict[str, Any]:
    """バランスの取れた適合度計算"""
    
    lab_name = lab_data.get('name', 'Unknown')
    print(f"🧮 バランス適合度計算: 研究室={lab_name}")
    
    criteria_scores = {}
    total_weighted_score = 0.0
    total_weights = 0.0
    major_mismatches = 0
    
    for criterion in COMPLETE_EVALUATION_CRITERIA:
        if criterion in student_profile:
            student_value = float(student_profile[criterion])
            lab_value = get_balanced_lab_value(lab_data, criterion, lab_name)
            
            diff = abs(student_value - lab_value)
            similarity = calculate_balanced_similarity(student_value, lab_value)
            
            # 重要基準での大きな不一致に適度なペナルティ
            weight = BALANCED_CRITERIA_WEIGHTS.get(criterion, 1.0)
            if weight >= 1.5 and diff > 3.0:  # 重要基準で非常に大きな差
                major_mismatches += 1
                similarity *= 0.7  # 30%減点（緩和）
                print(f"  ⚠️ 重要基準大差: {criterion}, 差={diff:.1f}")
            
            weighted_score = similarity * weight
            
            criteria_scores[criterion] = {
                "student_value": student_value,
                "lab_value": lab_value,
                "difference": diff,
                "similarity": similarity,
                "weight": weight,
                "weighted_score": weighted_score
            }
            
            total_weighted_score += weighted_score
            total_weights += weight
            
            print(f"  📊 {criterion}: 差={diff:.1f}, 類似度={similarity:.3f}")
    
    if total_weights == 0:
        return {"lab_name": lab_name, "overall_score": 0.0}
    
    base_score = total_weighted_score / total_weights
    
    # 適度なペナルティシステム
    penalty_factor = 1.0
    
    # 重大不一致ペナルティ（緩和）
    if major_mismatches > 0:
        penalty_factor *= (0.8 ** major_mismatches)  # より緩やかな減衰
        print(f"  📉 重要不一致ペナルティ: {major_mismatches}件, 係数={penalty_factor:.3f}")
    
    # 分野不一致の適度なペナルティ
    field_match = check_reasonable_field_compatibility(student_profile, lab_data)
    if not field_match:
        penalty_factor *= 0.75  # 25%減点（緩和）
        print(f"  📉 分野不一致ペナルティ: -25%")
    
    # 最終スコア計算（実用的な範囲：0.20-0.85）
    penalized_score = base_score * penalty_factor
    
    # スコア分布の正規化（平均を0.50前後に）
    normalized_score = 0.25 + (penalized_score * 0.60)
    final_score = max(0.20, min(0.85, normalized_score))
    
    print(f"📊 計算完了: 基本={base_score:.3f}, ペナルティ後={penalized_score:.3f}, 最終={final_score:.3f}")
    
    return {
        "lab_name": lab_name,
        "overall_score": final_score,
        "base_score": base_score,
        "penalty_factor": penalty_factor,
        "major_mismatches": major_mismatches,
        "field_match": field_match,
        "criteria_scores": criteria_scores,
        "recommendation_level": get_practical_recommendation_level(final_score)
    }

def check_reasonable_field_compatibility(student_profile: Dict[str, float], lab_data: Dict[str, Any]) -> bool:
    """現実的な分野適合性チェック"""
    
    # 分野興味度の確認（緩和）
    field_interests = student_profile.get("field_interests", {})
    lab_research_area = lab_data.get("research_area", "")
    
    if field_interests:
        # 中程度以上の興味(7.0以上)の分野と研究室分野の一致をチェック
        interested_fields = [field for field, interest in field_interests.items() if interest >= 7.0]
        
        if interested_fields:
            for field_id in interested_fields:
                if reasonable_field_match(field_id, lab_research_area):
                    return True
            # 完全不一致でも部分的類似を考慮
            return check_partial_field_match(interested_fields, lab_research_area)
    
    # research_field_match スコアによる判定（緩和）
    field_match_score = student_profile.get("research_field_match", 0)
    return field_match_score >= 7.0  # 閾値を下げる

def reasonable_field_match(field_id: str, research_area: str) -> bool:
    """現実的なフィールドマッチング"""
    
    # 完全一致
    exact_mappings = {
        "ai_ml": ["人工知能・機械学習"],
        "image_processing": ["画像・映像処理"],
        "web_design": ["Webデザイン・UI/UX"],
        "game_esports": ["ゲーム開発・eスポーツ"],
        "vr_ar_media": ["VR/AR・メディアアート"],
        "database_systems": ["データベース・情報システム"],
        "network_security": ["ネットワーク・セキュリティ"]
    }
    
    # 部分一致も考慮
    partial_mappings = {
        "ai_ml": ["データベース", "システム", "情報"],
        "image_processing": ["メディアアート", "デザイン"],
        "web_design": ["デザイン", "視覚表現"],
        "system_ethics": ["システム", "情報"],
        "natural_science_math": ["数理", "科学"]
    }
    
    # 完全一致チェック
    exact_areas = exact_mappings.get(field_id, [])
    if any(area in research_area for area in exact_areas):
        return True
    
    # 部分一致チェック
    partial_keywords = partial_mappings.get(field_id, [])
    return any(keyword in research_area for keyword in partial_keywords)

def check_partial_field_match(interested_fields: List[str], research_area: str) -> bool:
    """部分的フィールドマッチング"""
    
    # 関連分野の緩やかなマッチング
    broad_categories = {
        "technology": ["ai_ml", "image_processing", "database_systems", "network_security"],
        "creative": ["web_design", "design_visual", "video_animation"],
        "interdisciplinary": ["education_linguistics", "medical_healthcare", "business_decision"]
    }
    
    research_category = None
    if any(keyword in research_area for keyword in ["AI", "データ", "システム", "ネットワーク"]):
        research_category = "technology"
    elif any(keyword in research_area for keyword in ["デザイン", "映像", "Web"]):
        research_category = "creative"
    elif any(keyword in research_area for keyword in ["教育", "医療", "経営"]):
        research_category = "interdisciplinary"
    
    if research_category:
        category_fields = broad_categories.get(research_category, [])
        return any(field in category_fields for field in interested_fields)
    
    return False

def get_practical_recommendation_level(score: float) -> str:
    """実用的な推薦レベル"""
    
    if score >= 0.70:
        return "強く推薦"
    elif score >= 0.55:
        return "推薦"
    elif score >= 0.40:
        return "検討可能"
    elif score >= 0.25:
        return "要検討"
    else:
        return "推薦しない"

def generate_explanation(student_profile: Dict[str, float], lab_data: Dict[str, Any], compatibility_score: float) -> str:
    """説明文生成"""
    score_percent = compatibility_score * 100
    recommendation = get_practical_recommendation_level(compatibility_score)
    criteria_count = len(COMPLETE_EVALUATION_CRITERIA)
    
    explanation_parts = [
        f"総合適合度: {score_percent:.1f}% ({recommendation})",
        f"評価基準数: {criteria_count}/13項目"
    ]
    
    # 特に適合している基準を特定
    high_score_criteria = []
    for criterion in ["research_intensity", "advisor_style", "theory_practice"]:
        if criterion in student_profile:
            student_val = student_profile[criterion]
            lab_val = get_balanced_lab_value(lab_data, criterion, lab_data.get('name', ''))
            if abs(student_val - lab_val) <= 1.0:
                high_score_criteria.append(criterion)
    
    if high_score_criteria:
        explanation_parts.append(f"特に適合: {', '.join(high_score_criteria[:2])}")
    
    # 分野マッチ
    if check_reasonable_field_compatibility(student_profile, lab_data):
        explanation_parts.append("研究分野が一致")
    
    return "。".join(explanation_parts) + "。"

# バランス調整された遺伝的アルゴリズム
def balanced_genetic_algorithm(
    student_profile: Dict[str, float], 
    lab_data: List[Dict], 
    population_size: int = 12, 
    generations: int = 8
) -> Dict[str, Any]:
    """バランス調整された遺伝的アルゴリズム"""
    
    print(f"🧬 バランス遺伝的アルゴリズム開始: 集団={population_size}, 世代={generations}")
    
    def balanced_fitness_function(weights_vector: List[float]) -> float:
        """バランスの取れた適応度関数"""
        temp_weights = {
            criterion: max(0.3, min(2.5, weights_vector[i]))
            for i, criterion in enumerate(COMPLETE_EVALUATION_CRITERIA)
        }
        
        fitness_scores = []
        sample_labs = lab_data[:min(10, len(lab_data))]
        
        for lab in sample_labs:
            try:
                # 一時的に重みを変更して適合度計算
                global BALANCED_CRITERIA_WEIGHTS
                original_weights = BALANCED_CRITERIA_WEIGHTS.copy()
                BALANCED_CRITERIA_WEIGHTS.update(temp_weights)
                
                compatibility = calculate_balanced_compatibility(student_profile, lab)
                score = compatibility["overall_score"]
                fitness_scores.append(score)
                
                # 重みを元に戻す
                BALANCED_CRITERIA_WEIGHTS = original_weights
                
            except Exception as e:
                fitness_scores.append(0.2)
        
        if not fitness_scores:
            return 0.2
        
        mean_score = sum(fitness_scores) / len(fitness_scores)
        
        # 目標は適度な分散（平均0.4-0.6）
        target_mean = 0.5
        if abs(mean_score - target_mean) > 0.2:
            penalty = abs(mean_score - target_mean) * 0.5
            mean_score *= (1.0 - penalty)
        
        return max(0.1, min(0.8, mean_score))
    
    # 初期集団生成
    population = []
    for _ in range(population_size):
        individual = [random.uniform(0.7, 2.0) for _ in range(len(COMPLETE_EVALUATION_CRITERIA))]
        population.append(individual)
    
    best_individual = None
    best_fitness = 0.0
    
    for generation in range(generations):
        # 適応度評価
        fitness_scores = [(individual, balanced_fitness_function(individual)) for individual in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        current_best_fitness = fitness_scores[0][1]
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = fitness_scores[0][0].copy()
            print(f"📈 世代{generation}: 改善 適応度={best_fitness:.3f}")
        
        # 選択・交叉・突然変異
        elite_count = max(2, int(population_size * 0.25))
        new_population = [individual for individual, _ in fitness_scores[:elite_count]]
        
        while len(new_population) < population_size:
            # 適応度比例選択
            total_fitness = sum(fitness for _, fitness in fitness_scores)
            if total_fitness > 0:
                selection_point = random.uniform(0, total_fitness)
                
                cumulative = 0
                parent1 = fitness_scores[0][0]  # デフォルト
                for individual, fitness in fitness_scores:
                    cumulative += fitness
                    if cumulative >= selection_point:
                        parent1 = individual
                        break
            else:
                parent1 = random.choice(fitness_scores)[0]
            
            parent2 = random.choice(fitness_scores[:population_size//2])[0]
            
            # 交叉
            child = [p1 if random.random() < 0.5 else p2 for p1, p2 in zip(parent1, parent2)]
            
            # 突然変異
            if random.random() < 0.15:
                for i in range(len(child)):
                    if random.random() < 0.08:
                        child[i] = max(0.3, min(2.5, child[i] + random.gauss(0, 0.2)))
            
            new_population.append(child)
        
        population = new_population
    
    # 最適化された重み
    optimized_weights = {
        criterion: best_individual[i]
        for i, criterion in enumerate(COMPLETE_EVALUATION_CRITERIA)
    }
    
    print(f"✅ バランス遺伝的最適化完了: 最終適応度={best_fitness:.3f}")
    
    return {
        "optimized_weights": optimized_weights,
        "fitness_score": best_fitness,
        "generations": generations,
        "improvement_achieved": best_fitness > 0.35
    }

# システム初期化
def initialize_system():
    """システム初期化"""
    try:
        print("🔧 システム初期化中...")
        system_state["initialized"] = True
        system_state["last_updated"] = datetime.now()
        print("✅ システム初期化完了")
    except Exception as e:
        print(f"❌ システム初期化エラー: {e}")
        system_state["initialized"] = False

# APIエンドポイント定義

@app.get("/")
async def read_root():
    """ルートエンドポイント"""
    return {
        "message": "遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム",
        "version": "3.1.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "evaluate": "/api/evaluate",
            "optimize": "/api/optimize",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy",
        "version": "3.1.0",
        "timestamp": time.time(),
        "system_initialized": system_state["initialized"],
        "lab_count": len(SAMPLE_LABS),
        "evaluation_count": system_state["evaluation_count"]
    }

@app.get("/api/labs")
async def get_labs():
    """研究室一覧取得"""
    return {
        "labs": SAMPLE_LABS,
        "total_count": len(SAMPLE_LABS),
        "categories": {
            "AI・機械学習": 7,
            "画像・映像処理": 6,
            "ネットワーク・セキュリティ": 3,
            "教育・言語学": 5,
            "自然科学・数理": 6,
            "クリエイティブ": 8,
            "エンターテイメント": 4,
            "人文・社会・体育": 4
        }
    }

@app.get("/api/fields")
async def get_research_fields():
    """研究分野一覧取得"""
    fields = [
        {"id": "ai_ml", "name": "人工知能・機械学習", "category": "テクノロジー・システム"},
        {"id": "image_processing", "name": "画像・映像処理", "category": "テクノロジー・システム"},
        {"id": "network_security", "name": "ネットワーク・セキュリティ", "category": "テクノロジー・システム"},
        {"id": "database_systems", "name": "データベース・情報システム", "category": "テクノロジー・システム"},
        {"id": "web_design", "name": "Webデザイン・UI/UX", "category": "クリエイティブ"},
        {"id": "game_esports", "name": "ゲーム開発・eスポーツ", "category": "エンターテイメント"},
        {"id": "vr_ar_media", "name": "VR/AR・メディアアート", "category": "エンターテイメント"}
    ]
    
    return {
        "fields": fields,
        "categories": ["テクノロジー・システム", "クリエイティブ", "エンターテイメント", "人文・社会・体育"]
    }

@app.post("/api/evaluate")
async def evaluate_compatibility(request: Request):
    """研究室適合性評価エンドポイント"""
    try:
        print(f"📥 POST /api/evaluate - 開始")
        
        # リクエストデータ解析
        request_data = await request.json()
        student_profile = request_data.get("student_profile", {})
        
        print(f"📊 プロフィールデータ解析: {len(student_profile)}項目")
        
        # プロフィール検証
        if not student_profile:
            raise HTTPException(status_code=400, detail="student_profile が必要です")
        
        # 各研究室との適合度計算
        results = []
        calculation_errors = []
        
        for i, lab in enumerate(SAMPLE_LABS):
            try:
                print(f"\n--- 研究室 {i+1}: {lab.get('name', 'Unknown')}研究室 ---")
                
                # バランス調整された適合度計算
                compatibility_result = calculate_balanced_compatibility(student_profile, lab)
                overall_score = compatibility_result["overall_score"]
                
                # 統一されたレスポンス形式
                lab_result = {
                    "lab_name": lab.get("name", "Unknown"),
                    "compatibility_score": float(overall_score),
                    "final_score": float(overall_score),  # 後方互換性
                    "research_area": lab.get("research_area", "Unknown"),
                    "professor_name": lab.get("professor", "Unknown"),
                    "recommendation_level": compatibility_result.get("recommendation_level", "不明"),
                    "explanation": generate_explanation(student_profile, lab, overall_score),
                    "detailed_analysis": {
                        "strengths": [],
                        "concerns": [],
                        "recommendations": [],
                        "criteria_scores": compatibility_result.get("criteria_scores", {})
                    }
                }
                
                results.append(lab_result)
                print(f"✅ 計算成功: {lab.get('name')}研究室 = {overall_score:.3f}")
                
            except Exception as e:
                error_msg = f"研究室 {lab.get('name', f'ID:{i}')} の計算エラー: {str(e)}"
                calculation_errors.append(error_msg)
                print(f"❌ {error_msg}")
                continue
        
        if not results:
            raise HTTPException(
                status_code=500,
                detail="すべての研究室の計算に失敗しました。"
            )
        
        # スコア順でソート
        results.sort(key=lambda x: x["compatibility_score"], reverse=True)
        
        # 統計計算
        scores = [r["compatibility_score"] for r in results]
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        
        # レスポンス形式
        response = {
            "lab_results": results,
            "summary": {
                "total_labs": len(results),
                "avg_score": avg_score,
                "max_score": max_score,
                "min_score": min_score,
                "high_compatibility_count": len([r for r in results if r["compatibility_score"] >= 0.70]),
                "medium_compatibility_count": len([r for r in results if 0.40 <= r["compatibility_score"] < 0.70]),
                "low_compatibility_count": len([r for r in results if r["compatibility_score"] < 0.40])
            },
            "metadata": {
                "processing_time": 0.0,
                "evaluation_count": system_state.get("evaluation_count", 0) + 1,
                "timestamp": datetime.now().isoformat(),
                "endpoint": "/api/evaluate",
                "calculation_method": "balanced_compatibility_v5",
                "criteria_used": len(COMPLETE_EVALUATION_CRITERIA)
            }
        }
        
        # 警告情報追加
        if calculation_errors:
            response["warnings"] = {
                "calculation_errors": calculation_errors,
                "message": f"{len(calculation_errors)}件の研究室で計算エラーが発生しました。"
            }
        
        # 評価回数更新
        system_state["evaluation_count"] = system_state.get("evaluation_count", 0) + 1
        
        print(f"📤 /api/evaluate レスポンス送信: {len(results)}件")
        print(f"📊 適合度統計: 平均={avg_score:.3f}, 最高={max_score:.3f}, 最低={min_score:.3f}")
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ /api/evaluate 予期しないエラー: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"評価処理でエラーが発生しました: {str(e)}"
        )

@app.post("/api/optimize")
async def optimize_matching(request: Request):
    """遺伝的アルゴリズム最適化エンドポイント"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        request_data = await request.json()
        student_profile = request_data.get("student_profile", {})
        
        if not student_profile:
            raise HTTPException(status_code=400, detail="student_profile required")
        
        print("🔧 バランス遺伝的アルゴリズム最適化処理開始...")
        
        # バランス調整された遺伝的アルゴリズム実行
        optimization_result = balanced_genetic_algorithm(
            student_profile, 
            SAMPLE_LABS,
            population_size=12,
            generations=8
        )
        
        # 最適化された重みで再評価
        optimized_weights = optimization_result["optimized_weights"]
        
        # グローバル重みを一時的に更新
        global BALANCED_CRITERIA_WEIGHTS
        original_weights = BALANCED_CRITERIA_WEIGHTS.copy()
        BALANCED_CRITERIA_WEIGHTS.update(optimized_weights)
        
        # 全研究室を最適化重みで評価
        optimized_results = []
        for lab in SAMPLE_LABS:
            compatibility = calculate_balanced_compatibility(student_profile, lab)
            
            optimized_results.append({
                "lab_name": lab["name"],
                "optimized_score": compatibility["overall_score"],
                "research_area": lab.get("research_area", "Unknown"),
                "recommendation_level": compatibility.get("recommendation_level", "不明")
            })
        
        # 重みを元に戻す
        BALANCED_CRITERIA_WEIGHTS = original_weights
        
        # スコア順でソート
        optimized_results.sort(key=lambda x: x["optimized_score"], reverse=True)
        
        return {
            "optimization_completed": True,
            "optimization_method": "balanced_genetic_algorithm_v5",
            "optimization_summary": {
                "fitness_achieved": optimization_result["fitness_score"],
                "generations_completed": optimization_result["generations"],
                "improvement_achieved": optimization_result["improvement_achieved"]
            },
            "optimized_weights": optimized_weights,
            "lab_results": optimized_results,
            "statistics": {
                "total_labs": len(optimized_results),
                "high_compatibility": len([r for r in optimized_results if r["optimized_score"] >= 0.70]),
                "medium_compatibility": len([r for r in optimized_results if 0.40 <= r["optimized_score"] < 0.70]),
                "low_compatibility": len([r for r in optimized_results if r["optimized_score"] < 0.40]),
                "average_score": sum(r["optimized_score"] for r in optimized_results) / len(optimized_results),
                "max_score": max(r["optimized_score"] for r in optimized_results),
                "min_score": min(r["optimized_score"] for r in optimized_results)
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        print(f"❌ 最適化エラー: {e}")
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")

# デバッグエンドポイント
@app.get("/api/debug/status")
async def debug_status():
    """デバッグ用システム状態"""
    return {
        "system_state": system_state,
        "sample_labs_count": len(SAMPLE_LABS),
        "criteria_count": len(COMPLETE_EVALUATION_CRITERIA),
        "weights": BALANCED_CRITERIA_WEIGHTS,
        "has_numpy": HAS_NUMPY,
        "version": "3.1.0 - バランス調整版"
    }

# システム初期化実行
initialize_system()

# サーバー起動関数
def start_server(host: str = "0.0.0.0", port: int = 8000) -> bool:
    """サーバー起動"""
    
    print("\n" + "=" * 80)
    print("🧬🌳 遺伝的アルゴリズム × ファジィ決定木 研究室マッチングシステム v3.1.0")
    print("=" * 80)
    print(f"🚀 サーバー起動中...")
    print(f"📍 URL: http://localhost:{port}")
    print(f"📚 API文書: http://localhost:{port}/docs")
    print(f"🔧 システム状況:")
    print(f"   - FastAPI: ✅")
    print(f"   - NumPy: {'✅' if HAS_NUMPY else '❌ (オプション)'}")
    print(f"   - 研究室データ: {len(SAMPLE_LABS)}件")
    print(f"   - 評価基準: {len(COMPLETE_EVALUATION_CRITERIA)}項目")
    print(f"   - 遺伝的アルゴリズム: ✅ (バランス調整版)")
    print(f"   - 適合度計算: バランス調整版 (実用的範囲: 20-85%)")
    print("=" * 80)
    
    print("\n📋 期待される結果:")
    print("   - 高適合 (70%以上): 10-20%の研究室")
    print("   - 中適合 (40-69%): 40-60%の研究室")
    print("   - 低適合 (40%未満): 20-40%の研究室")
    print("   - 平均適合度: 45-55%程度")
    
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
    "communication_style": 7.0,
    "innovation_risk": 8.0,
    "field_interests": {
      "ai_ml": 9.0,
      "image_processing": 7.0
    }
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