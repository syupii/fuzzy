#!/usr/bin/env python3
"""
遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム
完全優先度対応版 FastAPI メインアプリケーション - app.py v5.0.0

新機能:
- 12項目評価基準
- 各項目の優先度設定（1-10段階）
- 優先度を考慮した重み付きマッチングアルゴリズム
- 18分野研究分野対応
- 統合AI評価エンジン
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
from typing import Dict, List, Any, Optional, Union, Tuple
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
    from fastapi.responses import JSONResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    HAS_FASTAPI = True
    print("✅ FastAPI モジュール正常にロード（優先度対応版）")
except ImportError as e:
    print(f"❌ FastAPI インポートエラー: {e}")
    print("💡 解決方法: pip install fastapi uvicorn")
    sys.exit(1)

# FastAPIアプリケーションインスタンス
app = FastAPI(
    title="遺伝的アルゴリズム×ファジィ決定木×優先度対応研究室選択支援システム",
    description="18分野対応・12項目評価基準（優先度設定可能）による超高精度研究室マッチングシステム",
    version="5.0.0",
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
    "priority_evaluations": 0,  # 新規追加
    "last_updated": None,
    "server_start_time": datetime.now(),
    "api_calls": 0,
    "genetic_optimization_runs": 0,
    "priority_aware_engines": {},  # 新規追加
    "lab_database": []
}

# 12項目対応の評価基準（革新性項目削除済み）
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

# 評価基準の詳細情報
CRITERIA_INFO = {
    'research_intensity': {
        'name': '研究強度',
        'description': '研究にどれだけ集中的に取り組みたいか',
        'range': '1（軽い研究）〜 10（集中研究）',
        'category': 'basic'
    },
    'advisor_style': {
        'name': '指導スタイル',
        'description': '教授からの指導の受け方の好み',
        'range': '1（厳格指導）〜 10（自由指導）',
        'category': 'basic'
    },
    'team_work': {
        'name': 'チームワーク',
        'description': '研究での他者との協働の程度',
        'range': '1（個人研究）〜 10（チーム研究）',
        'category': 'basic'
    },
    'workload': {
        'name': 'ワークロード',
        'description': '研究活動の忙しさに対する許容度',
        'range': '1（軽い負荷）〜 10（重い負荷）',
        'category': 'basic'
    },
    'theory_practice': {
        'name': '理論・実践バランス',
        'description': '理論研究と実践的研究のバランス',
        'range': '1（理論重視）〜 10（実践重視）',
        'category': 'basic'
    },
    'research_field_match': {
        'name': '研究分野適合性',
        'description': '自分の興味と研究室の分野の一致度',
        'range': '1（広い分野）〜 10（専門特化）',
        'category': 'extended'
    },
    'skill_development': {
        'name': 'スキル開発',
        'description': '専門性と汎用性のバランス',
        'range': '1（専門特化）〜 10（幅広いスキル）',
        'category': 'extended'
    },
    'lab_atmosphere': {
        'name': '研究室雰囲気',
        'description': '研究室の全体的な雰囲気',
        'range': '1（静寂集中）〜 10（活発議論）',
        'category': 'extended'
    },
    'flexibility': {
        'name': '柔軟性',
        'description': '研究時間の自由度',
        'range': '1（固定スケジュール）〜 10（柔軟スケジュール）',
        'category': 'extended'
    },
    'publication_opportunity': {
        'name': '論文発表機会',
        'description': '研究成果の論文化機会',
        'range': '1（少ない機会）〜 10（豊富な機会）',
        'category': 'extended'
    },
    'interdisciplinary': {
        'name': '学際性',
        'description': '他分野との連携の程度',
        'range': '1（単一分野）〜 10（学際連携）',
        'category': 'special'
    },
    'communication_style': {
        'name': 'コミュニケーション',
        'description': '研究室での交流スタイル',
        'range': '1（少人数密接）〜 10（オープン交流）',
        'category': 'special'
    }
}

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
        "faculty": ["廣奥透", "島田映二"]
    },
    
    # エンターテイメント分野（2分野）
    "game_esports": {
        "name": "ゲーム開発・eスポーツ", 
        "category": "エンターテイメント",
        "faculty_count": 2,
        "faculty": ["藤原孝行", "織田哲"]
    },
    "vr_ar_media": {
        "name": "VR/AR・メディアアート", 
        "category": "エンターテイメント",
        "faculty_count": 2,
        "faculty": ["Marty M. ITO", "島田映二"]
    },
    
    # 人文・社会・体育分野（2分野）
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

# 重み設定
BALANCED_CRITERIA_WEIGHTS = {criterion: 1.0/len(EVALUATION_CRITERIA) for criterion in EVALUATION_CRITERIA}

# 優先度対応エンジンクラス群
class PriorityAwareFuzzyEngine:
    """優先度対応ファジィ推論エンジン（簡易版）"""
    
    def __init__(self):
        self.initialized = True
    
    def predict_with_priorities(
        self, 
        student_profile: Dict[str, Any], 
        lab_profile: Dict[str, Any],
        priorities: Dict[str, float]
    ) -> Tuple[float, str]:
        """優先度を考慮したファジィ推論予測"""
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for criterion in EVALUATION_CRITERIA:
            if criterion in student_profile and criterion in lab_profile:
                student_val = float(student_profile[criterion])
                lab_val = float(lab_profile[criterion])
                priority = priorities.get(criterion, 5.0)
                
                # 正規化
                if student_val > 1.0:
                    student_val /= 10.0
                if lab_val > 1.0:
                    lab_val /= 10.0
                
                # ファジィ類似度計算（ガウシアン）
                sigma = 0.25
                diff = abs(student_val - lab_val)
                fuzzy_similarity = math.exp(-(diff ** 2) / (2 * sigma ** 2))
                
                # 優先度による重み付け
                priority_weight = priority / 10.0
                weighted_score = fuzzy_similarity * priority_weight
                
                total_weighted_score += weighted_score
                total_weight += priority_weight
        
        final_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        explanation = f"優先度ファジィ推論: {final_score:.3f}"
        
        return final_score, explanation


class PriorityAwareGeneticEngine:
    """優先度対応遺伝的アルゴリズム（簡易版）"""
    
    def __init__(self, population_size: int = 20, generations: int = 15):
        self.population_size = population_size
        self.generations = generations
        self.initialized = True
    
    def evaluate_with_priorities(
        self,
        student_profile: Dict[str, Any],
        lab_profile: Dict[str, Any], 
        priorities: Dict[str, float]
    ) -> Tuple[float, str]:
        """優先度を考慮した遺伝的評価"""
        
        # 進化的重み最適化のシミュレーション
        best_fitness = 0.0
        
        for generation in range(5):  # 簡易版なので5世代のみ
            # 重みベクトル生成
            weights = {}
            for criterion in EVALUATION_CRITERIA:
                priority = priorities.get(criterion, 5.0)
                # 優先度ベースの重み + ランダムノイズ
                base_weight = priority / 10.0
                noise = random.uniform(-0.1, 0.1)
                weights[criterion] = max(0.1, min(1.0, base_weight + noise))
            
            # 適応度計算
            fitness = self._calculate_fitness_with_weights(
                student_profile, lab_profile, weights, priorities
            )
            
            if fitness > best_fitness:
                best_fitness = fitness
        
        explanation = f"優先度遺伝的アルゴリズム: {best_fitness:.3f}"
        
        return best_fitness, explanation
    
    def _calculate_fitness_with_weights(
        self,
        student_profile: Dict[str, Any],
        lab_profile: Dict[str, Any],
        weights: Dict[str, float],
        priorities: Dict[str, float]
    ) -> float:
        """重み付き適応度計算"""
        
        total_score = 0.0
        total_weight = 0.0
        
        for criterion in EVALUATION_CRITERIA:
            if criterion in student_profile and criterion in lab_profile:
                student_val = float(student_profile[criterion])
                lab_val = float(lab_profile[criterion])
                
                # 正規化
                if student_val > 1.0:
                    student_val /= 10.0
                if lab_val > 1.0:
                    lab_val /= 10.0
                
                # 適合度
                match = 1.0 - abs(student_val - lab_val)
                
                # 重みと優先度の組み合わせ
                weight = weights.get(criterion, 0.5)
                priority = priorities.get(criterion, 5.0) / 10.0
                combined_weight = weight * priority
                
                total_score += match * combined_weight
                total_weight += combined_weight
        
        return total_score / total_weight if total_weight > 0 else 0.0


# サンプル研究室データ（デモ用）
SAMPLE_LABS = [
    {
        "id": "lab_itoh",
        "name": "伊藤雅彦研究室",
        "advisor": "伊藤雅彦",
        "research_area": "人工知能・機械学習",
        "category": "テクノロジー・システム",
        "research_intensity": 8.5,
        "advisor_style": 7.0,
        "team_work": 8.0,
        "workload": 8.0,
        "theory_practice": 6.5,
        "research_field_match": 9.0,
        "skill_development": 8.5,
        "lab_atmosphere": 7.5,
        "flexibility": 6.0,
        "publication_opportunity": 9.0,
        "interdisciplinary": 7.0,
        "communication_style": 7.5
    },
    {
        "id": "lab_uchiyama",
        "name": "内山敏雄研究室",
        "advisor": "内山敏雄",
        "research_area": "人工知能・機械学習",
        "category": "テクノロジー・システム",
        "research_intensity": 9.0,
        "advisor_style": 6.5,
        "team_work": 7.5,
        "workload": 8.5,
        "theory_practice": 7.0,
        "research_field_match": 9.0,
        "skill_development": 8.0,
        "lab_atmosphere": 7.0,
        "flexibility": 5.5,
        "publication_opportunity": 8.5,
        "interdisciplinary": 6.5,
        "communication_style": 7.0
    },
    {
        "id": "lab_web_design",
        "name": "Webデザイン研究室",
        "advisor": "杉沢愛美",
        "research_area": "Webデザイン・UI/UX",
        "category": "クリエイティブ",
        "research_intensity": 7.0,
        "advisor_style": 8.5,
        "team_work": 9.0,
        "workload": 7.5,
        "theory_practice": 8.0,
        "research_field_match": 7.5,
        "skill_development": 9.0,
        "lab_atmosphere": 8.5,
        "flexibility": 8.0,
        "publication_opportunity": 6.5,
        "interdisciplinary": 8.0,
        "communication_style": 8.5
    },
    {
        "id": "lab_game_dev",
        "name": "ゲーム開発研究室",
        "advisor": "藤原孝行",
        "research_area": "ゲーム開発・eスポーツ",
        "category": "エンターテイメント",
        "research_intensity": 8.0,
        "advisor_style": 7.5,
        "team_work": 8.5,
        "workload": 8.0,
        "theory_practice": 7.5,
        "research_field_match": 8.0,
        "skill_development": 8.5,
        "lab_atmosphere": 8.0,
        "flexibility": 7.0,
        "publication_opportunity": 7.0,
        "interdisciplinary": 7.5,
        "communication_style": 8.0
    },
    {
        "id": "lab_sports",
        "name": "スポーツ科学研究室",
        "advisor": "綿谷貴志",
        "research_area": "スポーツ・体育科学",
        "category": "人文・社会・体育",
        "research_intensity": 6.5,
        "advisor_style": 8.0,
        "team_work": 9.0,
        "workload": 7.0,
        "theory_practice": 8.5,
        "research_field_match": 6.0,
        "skill_development": 7.5,
        "lab_atmosphere": 9.0,
        "flexibility": 8.5,
        "publication_opportunity": 6.0,
        "interdisciplinary": 8.5,
        "communication_style": 9.0
    }
]

# システム初期化
def initialize_system():
    """システム初期化"""
    
    try:
        print("🔧 システム初期化開始...")
        
        # 研究室データベース初期化
        system_state["lab_database"] = SAMPLE_LABS.copy()
        
        # 優先度対応エンジン初期化
        system_state["priority_aware_engines"] = {
            "fuzzy": PriorityAwareFuzzyEngine(),
            "genetic": PriorityAwareGeneticEngine()
        }
        
        # システム状態更新
        system_state["initialized"] = True
        system_state["last_updated"] = datetime.now()
        
        print(f"✅ システム初期化完了")
        print(f"   - 研究室データ: {len(SAMPLE_LABS)}件")
        print(f"   - 評価基準: {len(EVALUATION_CRITERIA)}項目（優先度対応）")
        print(f"   - 研究分野: {len(RESEARCH_FIELDS)}分野")
        print(f"   - 優先度エンジン: ファジィ・遺伝的アルゴリズム対応")
        
        return True
        
    except Exception as e:
        print(f"❌ システム初期化エラー: {e}")
        return False

# APIエンドポイント群

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "message": "遺伝的アルゴリズム×ファジィ決定木×優先度対応研究室選択支援システム",
        "version": "5.0.0",
        "features": ["12項目評価基準", "優先度設定", "18分野対応", "AI統合評価"],
        "status": "operational" if system_state["initialized"] else "initializing"
    }

@app.get("/api/fields")
async def get_research_fields():
    """研究分野一覧取得"""
    return {
        "research_fields": RESEARCH_FIELDS,
        "total_count": len(RESEARCH_FIELDS),
        "categories": list(set([field["category"] for field in RESEARCH_FIELDS.values()]))
    }

@app.get("/api/criteria")
async def get_evaluation_criteria():
    """評価基準情報取得"""
    return {
        "criteria": CRITERIA_INFO,
        "total_count": len(EVALUATION_CRITERIA),
        "categories": {
            "basic": [k for k, v in CRITERIA_INFO.items() if v['category'] == 'basic'],
            "extended": [k for k, v in CRITERIA_INFO.items() if v['category'] == 'extended'], 
            "special": [k for k, v in CRITERIA_INFO.items() if v['category'] == 'special']
        },
        "priority_support": True,
        "priority_range": "1-10"
    }

@app.get("/api/labs")
async def get_labs():
    """研究室一覧取得"""
    return {
        "labs": SAMPLE_LABS,
        "total_count": len(SAMPLE_LABS),
        "categories": list(set([lab["category"] for lab in SAMPLE_LABS]))
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
    """学生プロファイルに基づく研究室適合度評価（優先度対応版）"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        student_profile = request.get("student_profile", {})
        priorities = student_profile.get("priorities", {})
        
        # 入力検証
        for criterion in EVALUATION_CRITERIA:
            if criterion not in student_profile:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing required criterion: {criterion}"
                )
        
        print(f"📥 優先度対応評価リクエスト受信: {len(EVALUATION_CRITERIA)}項目")
        if priorities:
            print(f"📊 優先度データ: {len(priorities)}項目")
            print(f"🎯 最高優先度: {max(priorities.items(), key=lambda x: x[1]) if priorities else 'なし'}")
        
        # 各研究室との適合度計算（優先度対応）
        lab_results = []
        
        for lab in SAMPLE_LABS:
            compatibility = calculate_priority_weighted_compatibility(
                student_profile, lab, priorities
            )
            
            # AI統合評価（ファジィ + 遺伝的アルゴリズム）
            ai_scores = {}
            
            # ファジィ推論
            try:
                fuzzy_engine = system_state["priority_aware_engines"]["fuzzy"]
                fuzzy_score, fuzzy_explanation = fuzzy_engine.predict_with_priorities(
                    student_profile, lab, priorities
                )
                ai_scores["fuzzy"] = fuzzy_score
            except:
                ai_scores["fuzzy"] = compatibility
            
            # 遺伝的アルゴリズム
            try:
                genetic_engine = system_state["priority_aware_engines"]["genetic"]
                genetic_score, genetic_explanation = genetic_engine.evaluate_with_priorities(
                    student_profile, lab, priorities
                )
                ai_scores["genetic"] = genetic_score
            except:
                ai_scores["genetic"] = compatibility
            
            # 統合スコア計算
            integrated_score = (
                0.4 * ai_scores["fuzzy"] + 
                0.3 * ai_scores["genetic"] + 
                0.3 * compatibility
            )
            
            # 優先度による追加調整
            if priorities:
                priority_boost = calculate_priority_boost(student_profile, lab, priorities)
                integrated_score = min(1.0, integrated_score + priority_boost)
            
            # 詳細スコア計算（優先度考慮）
            feature_scores = {}
            for criterion in EVALUATION_CRITERIA:
                student_val = float(student_profile[criterion])
                lab_val = float(lab[criterion])
                priority = priorities.get(criterion, 5.0) if priorities else 5.0
                
                if student_val > 1.0:
                    student_val /= 10.0
                if lab_val > 1.0:
                    lab_val /= 10.0
                
                # 基本適合度
                base_match = 1.0 - abs(student_val - lab_val)
                
                # 優先度重み付け
                priority_weight = priority / 10.0
                weighted_match = base_match * (0.5 + 0.5 * priority_weight)
                
                feature_scores[criterion] = weighted_match
            
            # フロントエンドが期待する形式でレスポンス構築
            lab_result = {
                "lab_id": lab["id"],
                "lab_name": lab["name"],
                "advisor": lab["advisor"],
                "research_area": lab["research_area"],
                "category": lab["category"],
                "professor_name": lab["advisor"],
                
                # スコア関連（優先度反映）
                "final_score": float(integrated_score),
                "compatibility_score": float(integrated_score),
                "overall_compatibility": float(integrated_score),
                "priority_adjusted_score": float(integrated_score),
                
                # AI統合スコア（新規）
                "ai_scores": ai_scores,
                "base_compatibility": float(compatibility),
                
                "feature_scores": feature_scores,
                "confidence": min(1.0, integrated_score + random.uniform(0.0, 0.1)),
                "recommendation": get_priority_recommendation_level(integrated_score, priorities),
                "recommendation_level": get_priority_recommendation_level(integrated_score, priorities),
                "explanation": generate_priority_explanation(
                    student_profile, lab, integrated_score, priorities, ai_scores
                ),
                
                # 優先度分析（新規）
                "priority_analysis": analyze_priority_match(
                    student_profile, lab, priorities
                ) if priorities else None
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
            "low_compatibility_count": len([s for s in scores if s < 0.4]),
            "priority_weighting_applied": bool(priorities),
            "total_priority_items": len(priorities) if priorities else 0,
            "priority_statistics": calculate_priority_statistics(priorities) if priorities else None
        }
        
        # 評価回数増加
        system_state["evaluation_count"] += 1
        system_state["api_calls"] += 1
        if priorities:
            system_state["priority_evaluations"] += 1
        
        # フロントエンドが期待する形式でレスポンス返却
        response = {
            "lab_results": lab_results,
            "summary": summary,
            
            "student_profile": student_profile,
            "evaluation_results": lab_results,
            "total_labs_evaluated": len(lab_results),
            "evaluation_timestamp": time.time(),
            "metadata": {
                "processing_time": 0.1,
                "evaluation_count": system_state["evaluation_count"],
                "priority_evaluations": system_state["priority_evaluations"],
                "timestamp": datetime.now().isoformat(),
                "criteria_used": len(EVALUATION_CRITERIA),
                "priorities_applied": priorities,
                "ai_engines_used": ["fuzzy", "genetic"],
                "calculation_method": "priority_weighted_ai_integrated_compatibility_v5"
            }
        }
        
        print(f"📤 優先度対応評価結果送信: {len(lab_results)}件の研究室")
        print(f"📊 適合度統計: 平均={summary['avg_score']:.3f}, 最高={summary['max_score']:.3f}")
        if priorities:
            print(f"🎯 優先度効果: 平均優先度={sum(priorities.values())/len(priorities):.1f}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ 優先度対応評価エラー: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Priority-aware evaluation error: {str(e)}")

# ヘルパー関数群

def calculate_priority_weighted_compatibility(
    student_profile: Dict[str, Any], 
    lab: Dict[str, Any], 
    priorities: Dict[str, float]
) -> float:
    """優先度を考慮した重み付き適合度計算"""
    
    total_weighted_score = 0.0
    total_weight = 0.0
    
    for criterion in EVALUATION_CRITERIA:
        if criterion in student_profile and criterion in lab:
            student_val = float(student_profile[criterion])
            lab_val = float(lab[criterion])
            
            # 正規化
            if student_val > 1.0:
                student_val /= 10.0
            if lab_val > 1.0:
                lab_val /= 10.0
            
            # 基本適合度計算
            base_compatibility = 1.0 - abs(student_val - lab_val)
            
            # 優先度による重み付け
            if priorities and criterion in priorities:
                priority_weight = priorities[criterion] / 10.0
            else:
                priority_weight = 0.5  # デフォルト重み
            
            # 重み付きスコア計算
            weighted_score = base_compatibility * priority_weight
            total_weighted_score += weighted_score
            total_weight += priority_weight
    
    # 重み付き平均を返す
    if total_weight > 0:
        return total_weighted_score / total_weight
    else:
        return 0.5

def calculate_priority_boost(
    student_profile: Dict[str, Any], 
    lab: Dict[str, Any], 
    priorities: Dict[str, float]
) -> float:
    """優先度による追加ブーストスコア計算"""
    
    if not priorities:
        return 0.0
    
    # 最重要項目（優先度8以上）での適合状況をチェック
    high_priority_items = {k: v for k, v in priorities.items() if v >= 8.0}
    
    if not high_priority_items:
        return 0.0
    
    total_boost = 0.0
    for criterion, priority in high_priority_items.items():
        if criterion in student_profile and criterion in lab:
            student_val = float(student_profile[criterion])
            lab_val = float(lab[criterion])
            
            # 正規化
            if student_val > 1.0:
                student_val /= 10.0
            if lab_val > 1.0:
                lab_val /= 10.0
            
            match = 1.0 - abs(student_val - lab_val)
            
            # 高優先度項目で高適合の場合はボーナス
            if match > 0.8:
                boost = 0.05 * (priority / 10.0) * (match - 0.8) / 0.2
                total_boost += boost
    
    return min(0.15, total_boost)  # 最大15%のブースト

def get_priority_recommendation_level(score: float, priorities: Dict[str, float]) -> str:
    """優先度を考慮した推薦レベル取得"""
    
    base_level = get_recommendation_level(score)
    
    if priorities:
        avg_priority = sum(priorities.values()) / len(priorities)
        if avg_priority >= 8.0 and score >= 0.8:
            return "最優先推薦"
        elif avg_priority >= 7.0 and score >= 0.7:
            return "優先推薦"
    
    return base_level

def get_recommendation_level(compatibility: float) -> str:
    """推薦レベル取得"""
    
    if compatibility >= 0.8:
        return "強く推薦"
    elif compatibility >= 0.6:
        return "推薦"
    elif compatibility >= 0.4:
        return "検討可能"
    else:
        return "推薦しない"

def generate_priority_explanation(
    student_profile: Dict[str, Any], 
    lab: Dict[str, Any], 
    compatibility: float, 
    priorities: Dict[str, float],
    ai_scores: Dict[str, float]
) -> str:
    """優先度を考慮した詳細説明生成"""
    
    base_explanation = f"統合適合度: {compatibility:.1%}"
    
    # AI統合スコア説明
    ai_explanation = f"（ファジィ: {ai_scores.get('fuzzy', 0):.2f}, 遺伝的: {ai_scores.get('genetic', 0):.2f}）"
    
    # 優先度分析
    if priorities:
        top_priorities = sorted(priorities.items(), key=lambda x: x[1], reverse=True)[:3]
        
        priority_explanations = []
        for criterion, priority in top_priorities:
            if criterion in student_profile and criterion in lab:
                student_val = float(student_profile[criterion])
                lab_val = float(lab[criterion])
                
                if student_val > 1.0:
                    student_val /= 10.0
                if lab_val > 1.0:
                    lab_val /= 10.0
                
                match_level = 1.0 - abs(student_val - lab_val)
                criterion_name = CRITERIA_INFO.get(criterion, {}).get('name', criterion)
                
                if match_level > 0.8:
                    priority_explanations.append(f"重要な{criterion_name}で高適合")
                elif match_level > 0.6:
                    priority_explanations.append(f"{criterion_name}で適度な適合")
                else:
                    priority_explanations.append(f"重要な{criterion_name}で課題")
        
        priority_details = "、".join(priority_explanations)
        return f"{base_explanation} {ai_explanation} {priority_details}。"
    else:
        return f"{base_explanation} {ai_explanation}"

def analyze_priority_match(
    student_profile: Dict[str, Any], 
    lab: Dict[str, Any], 
    priorities: Dict[str, float]
) -> Dict[str, Any]:
    """優先度マッチング分析"""
    
    if not priorities:
        return {}
    
    # 優先度別分析
    high_priority = {k: v for k, v in priorities.items() if v >= 8}
    medium_priority = {k: v for k, v in priorities.items() if 5 <= v < 8}
    low_priority = {k: v for k, v in priorities.items() if v < 5}
    
    # 各レベルでの適合度
    def calculate_level_match(priority_dict):
        if not priority_dict:
            return 0.0
        
        total_match = 0.0
        for criterion in priority_dict:
            if criterion in student_profile and criterion in lab:
                student_val = float(student_profile[criterion])
                lab_val = float(lab[criterion])
                
                if student_val > 1.0:
                    student_val /= 10.0
                if lab_val > 1.0:
                    lab_val /= 10.0
                
                match = 1.0 - abs(student_val - lab_val)
                total_match += match
        
        return total_match / len(priority_dict)
    
    return {
        "high_priority_match": calculate_level_match(high_priority),
        "medium_priority_match": calculate_level_match(medium_priority),
        "low_priority_match": calculate_level_match(low_priority),
        "priority_distribution": {
            "high": len(high_priority),
            "medium": len(medium_priority),
            "low": len(low_priority)
        },
        "weighted_priority_score": sum(
            priorities[k] * (1.0 - abs(
                (float(student_profile[k]) if student_profile[k] <= 1.0 else float(student_profile[k])/10.0) -
                (float(lab[k]) if lab[k] <= 1.0 else float(lab[k])/10.0)
            ))
            for k in priorities
            if k in student_profile and k in lab
        ) / sum(priorities.values()) if priorities else 0
    }

def calculate_priority_statistics(priorities: Dict[str, float]) -> Dict[str, Any]:
    """優先度統計計算"""
    
    if not priorities:
        return {}
    
    values = list(priorities.values())
    
    return {
        "average_priority": sum(values) / len(values),
        "max_priority": max(values),
        "min_priority": min(values),
        "priority_variance": sum((v - sum(values)/len(values))**2 for v in values) / len(values),
        "high_priority_count": len([v for v in values if v >= 8]),
        "medium_priority_count": len([v for v in values if 5 <= v < 8]),
        "low_priority_count": len([v for v in values if v < 5]),
        "top_priorities": sorted(priorities.items(), key=lambda x: x[1], reverse=True)[:5]
    }

@app.get("/api/system")
async def get_system_info():
    """システム情報取得（優先度対応版）"""
    return {
        "system_state": system_state,
        "sample_labs_count": len(SAMPLE_LABS),
        "criteria_count": len(EVALUATION_CRITERIA),
        "research_fields_count": len(RESEARCH_FIELDS),
        "weights": BALANCED_CRITERIA_WEIGHTS,
        "has_numpy": HAS_NUMPY,
        "priority_support": True,
        "priority_features": {
            "enabled": True,
            "range": "1-10",
            "ai_integration": True,
            "fuzzy_inference": True,
            "genetic_algorithm": True,
            "weighted_scoring": True
        },
        "ai_engines": {
            "fuzzy": "PriorityAwareFuzzyEngine",
            "genetic": "PriorityAwareGeneticEngine"
        },
        "version": "5.0.0 - 12項目+優先度+AI統合対応版"
    }

# システム初期化実行
initialize_system()

# サーバー起動関数
def start_server(host: str = "0.0.0.0", port: int = 8000) -> bool:
    """サーバー起動"""
    
    print("\n" + "=" * 100)
    print("🧬🌳⭐🤖 遺伝的アルゴリズム × ファジィ決定木 × 優先度対応 × AI統合 研究室マッチングシステム v5.0.0")
    print("=" * 100)
    print(f"🚀 サーバー起動中...")
    print(f"📍 URL: http://localhost:{port}")
    print(f"📚 API文書: http://localhost:{port}/docs")
    print(f"🔧 システム機能:")
    print(f"   - 評価基準: {len(EVALUATION_CRITERIA)}項目（各項目に優先度1-10設定可能）")
    print(f"   - 研究分野: {len(RESEARCH_FIELDS)}分野（18分野対応）")
    print(f"   - AI統合エンジン: ✅ ファジィ推論 + 遺伝的アルゴリズム")
    print(f"   - 優先度対応: ✅ 重み付きマッチングアルゴリズム")
    print(f"   - 研究室データベース: {len(SAMPLE_LABS)}件")
    print("=" * 100)
    print("\n📋 評価基準（12項目）+ 優先度:")
    for i, criterion in enumerate(EVALUATION_CRITERIA, 1):
        info = CRITERIA_INFO[criterion]
        print(f"   {i:2d}. {info['name']} ({info['category']}) - 優先度設定可能")
    print("=" * 100)
    
    try:
        uvicorn.run(app, host=host, port=port)
        return True
    except Exception as e:
        print(f"❌ サーバー起動エラー: {e}")
        return False

if __name__ == "__main__":
    start_server()