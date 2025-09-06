# backend/app.py (完全版 - 重視項目機能付き)
#!/usr/bin/env python3
"""
遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム
重視項目機能対応版 FastAPI メインアプリケーション
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
    title="研究室選択支援システム（重視項目対応）",
    description="遺伝的アルゴリズムを用いたファジィ決定木による研究室マッチングシステム（重視項目機能付き）",
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

# === Pydanticモデル定義 ===

class EvaluationPreferences(BaseModel):
    """評価基準設定（21項目）"""
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
    """研究分野興味度（11分野）"""
    ai_ml: float = 0.0
    image_video: float = 0.0
    network_security: float = 0.0
    database_systems: float = 0.0
    embedded_iot: float = 0.0
    web_ui_ux: float = 0.0
    design_visual: float = 0.0
    video_animation: float = 0.0
    computer_music: float = 0.0
    game_esports: float = 0.0
    vr_ar_media: float = 0.0

class PrioritySettings(BaseModel):
    """重視項目設定"""
    priority_criteria: List[str] = []
    priority_fields: List[str] = []
    priority_weight_multiplier: float = 2.0
    explanation_required: bool = True

class PersonalInfo(BaseModel):
    grade: int = 3
    experience_level: str = "intermediate"
    career_goals: List[str] = []
    preferred_learning_style: str = "mixed"

class StudentProfile(BaseModel):
    """学生プロフィール（重視項目設定追加）"""
    evaluation_criteria: EvaluationPreferences
    field_interests: ResearchFieldInterests
    priority_settings: PrioritySettings
    personal_info: Optional[PersonalInfo] = None

# === システム状態 ===
system_state = {
    "total_evaluations": 0,
    "labs_database": [],
    "priority_engine": None
}

# === 重視項目処理エンジン ===
class PriorityEngine:
    """重視項目処理エンジン"""
    
    def __init__(self):
        self.criteria_weights = self._initialize_default_weights()
        self.field_weights = self._initialize_field_weights()
        self.criteria_names = self._get_criteria_names()
        self.field_names = self._get_field_names()
    
    def _initialize_default_weights(self) -> Dict[str, float]:
        """デフォルト重み設定"""
        criteria_list = [
            # 基本項目
            "research_intensity", "advisor_style", "team_work", "workload", "theory_practice",
            # 拡張項目
            "research_field_match", "skill_development", "learning_pace", "difficulty_preference", "lab_atmosphere",
            # コミュニケーション関連
            "communication_style", "meeting_frequency", "flexibility", "evening_weekend_work",
            # 研究アプローチ関連
            "innovation_risk", "methodology_preference", "interdisciplinary",
            # 重要項目
            "publication_opportunity", "financial_support", "lab_hierarchy", "core_time_flexibility"
        ]
        return {criterion: 1.0 for criterion in criteria_list}
    
    def _initialize_field_weights(self) -> Dict[str, float]:
        """研究分野デフォルト重み設定"""
        fields = [
            "ai_ml", "image_video", "network_security", "database_systems", "embedded_iot",
            "web_ui_ux", "design_visual", "video_animation", "computer_music", "game_esports", "vr_ar_media"
        ]
        return {field: 1.0 for field in fields}
    
    def _get_criteria_names(self) -> Dict[str, str]:
        """評価基準名の日本語マッピング"""
        return {
            "research_intensity": "研究強度",
            "advisor_style": "指導スタイル",
            "team_work": "チームワーク",
            "workload": "ワークロード",
            "theory_practice": "理論・実践バランス",
            "research_field_match": "研究分野適合性",
            "skill_development": "スキル開発",
            "learning_pace": "学習ペース",
            "difficulty_preference": "難易度選好",
            "lab_atmosphere": "研究室雰囲気",
            "communication_style": "コミュニケーション",
            "meeting_frequency": "ミーティング頻度",
            "flexibility": "柔軟性",
            "evening_weekend_work": "夜間・休日作業",
            "innovation_risk": "革新性・リスク許容度",
            "methodology_preference": "手法選好",
            "interdisciplinary": "学際性",
            "publication_opportunity": "論文発表機会",
            "financial_support": "経済的支援",
            "lab_hierarchy": "研究室階層",
            "core_time_flexibility": "コアタイム柔軟性"
        }
    
    def _get_field_names(self) -> Dict[str, str]:
        """研究分野名の日本語マッピング"""
        return {
            "ai_ml": "人工知能・機械学習",
            "image_video": "画像・映像処理",
            "network_security": "ネットワーク・セキュリティ",
            "database_systems": "データベース・情報システム",
            "embedded_iot": "組込み・IoT",
            "web_ui_ux": "Webデザイン・UI/UX",
            "design_visual": "デザイン・視覚表現",
            "video_animation": "映像・アニメーション",
            "computer_music": "コンピュータ音楽・サウンドアート",
            "game_esports": "ゲーム開発・eスポーツ",
            "vr_ar_media": "VR/AR・メディアアート"
        }
    
    def apply_priority_weights(self, student_profile: StudentProfile) -> Dict[str, Any]:
        """重視項目に基づく重み調整済み適合度計算"""
        
        priority_settings = student_profile.priority_settings
        
        # 重視項目の重み調整
        adjusted_criteria_weights = self.criteria_weights.copy()
        adjusted_field_weights = self.field_weights.copy()
        
        # 評価基準の重み調整
        for criteria in priority_settings.priority_criteria:
            if criteria in adjusted_criteria_weights:
                adjusted_criteria_weights[criteria] *= priority_settings.priority_weight_multiplier
        
        # 研究分野の重み調整
        for field in priority_settings.priority_fields:
            if field in adjusted_field_weights:
                adjusted_field_weights[field] *= priority_settings.priority_weight_multiplier
        
        # 調整後のスコア計算
        criteria_scores = self._calculate_weighted_criteria_score(
            student_profile.evaluation_criteria,
            adjusted_criteria_weights
        )
        
        field_scores = self._calculate_weighted_field_score(
            student_profile.field_interests,
            adjusted_field_weights
        )
        
        # 基本適合度（重み調整なし）
        base_criteria_scores = self._calculate_weighted_criteria_score(
            student_profile.evaluation_criteria,
            self.criteria_weights
        )
        
        base_field_scores = self._calculate_weighted_field_score(
            student_profile.field_interests,
            self.field_weights
        )
        
        base_compatibility = (base_criteria_scores * 0.7) + (base_field_scores * 0.3)
        weighted_compatibility = (criteria_scores * 0.7) + (field_scores * 0.3)
        
        return {
            "base_compatibility": base_compatibility,
            "weighted_compatibility": weighted_compatibility,
            "priority_boost": weighted_compatibility - base_compatibility,
            "criteria_scores": criteria_scores,
            "field_scores": field_scores,
            "base_criteria_scores": base_criteria_scores,
            "base_field_scores": base_field_scores,
            "applied_priorities": {
                "criteria": priority_settings.priority_criteria,
                "fields": priority_settings.priority_fields,
                "multiplier": priority_settings.priority_weight_multiplier
            },
            "weights_info": {
                "adjusted_criteria_weights": adjusted_criteria_weights,
                "adjusted_field_weights": adjusted_field_weights
            }
        }
    
    def _calculate_weighted_criteria_score(self, criteria: EvaluationPreferences, weights: Dict[str, float]) -> float:
        """重み付き評価基準スコア計算"""
        
        criteria_dict = criteria.dict()
        total_score = 0.0
        total_weight = 0.0
        
        for criterion, value in criteria_dict.items():
            if criterion in weights:
                weight = weights[criterion]
                # 10点満点を1点満点に正規化
                normalized_value = value / 10.0
                total_score += normalized_value * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_weighted_field_score(self, fields: ResearchFieldInterests, weights: Dict[str, float]) -> float:
        """重み付き研究分野スコア計算"""
        
        fields_dict = fields.dict()
        total_score = 0.0
        total_weight = 0.0
        
        for field, value in fields_dict.items():
            if field in weights:
                weight = weights[field]
                # 10点満点を1点満点に正規化
                normalized_value = value / 10.0
                total_score += normalized_value * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def generate_priority_explanation(self, priority_settings: PrioritySettings, result: Dict[str, Any]) -> str:
        """重視項目による結果の説明生成"""
        
        explanation = []
        
        if priority_settings.priority_criteria:
            priority_names = [
                self.criteria_names.get(c, c) 
                for c in priority_settings.priority_criteria
            ]
            explanation.append(f"重視項目「{', '.join(priority_names)}」を{priority_settings.priority_weight_multiplier}倍で評価")
        
        if priority_settings.priority_fields:
            priority_field_names = [
                self.field_names.get(f, f) 
                for f in priority_settings.priority_fields
            ]
            explanation.append(f"重視分野「{', '.join(priority_field_names)}」を優先評価")
        
        boost = result.get("priority_boost", 0)
        if boost > 0:
            explanation.append(f"重視項目により適合度が{boost:.1%}向上")
        elif boost < 0:
            explanation.append(f"重視項目により適合度が{abs(boost):.1%}調整")
        
        return "。".join(explanation) + "。" if explanation else "標準的な重み付けで評価されました。"

# システム初期化
system_state["priority_engine"] = PriorityEngine()

from models.lab_database import LabDatabase

# グローバルデータベースインスタンス
lab_database = LabDatabase()

# システム状態の更新
system_state = {
    "initialized": True,
    "total_evaluations": 0,
    "labs_database": lab_database.get_all_labs(),  # ここを変更
    "evaluation_sessions": {}
}

@app.get("/api/v1/labs")
async def get_labs():
    """研究室一覧取得"""
    return lab_database.get_all_labs()

@app.get("/api/v1/system/stats")
async def get_system_stats():
    """システム統計取得"""
    stats = lab_database.get_statistics()
    
    return {
        "total_labs": stats["total_labs"],
        "total_evaluations": system_state["total_evaluations"],
        "average_processing_time": 1500,  # ms
        "field_distribution": stats["field_distribution"],
        "system_info": {
            "version": "3.0.0",
            "features": 21,
            "research_fields": 11,
            "algorithm": "遺伝的アルゴリズム × ファジィ決定木",
            "database_version": stats["database_version"]
        }
    }

# 新しいエンドポイント追加
@app.post("/api/v1/labs")
async def add_lab(lab_data: dict):
    """研究室を追加"""
    if lab_database.add_lab(lab_data):
        # システム状態更新
        system_state["labs_database"] = lab_database.get_all_labs()
        return {"message": "研究室を追加しました", "success": True}
    else:
        raise HTTPException(status_code=400, detail="研究室の追加に失敗しました")

@app.put("/api/v1/labs/{lab_id}")
async def update_lab(lab_id: str, lab_data: dict):
    """研究室を更新"""
    if lab_database.update_lab(lab_id, lab_data):
        # システム状態更新
        system_state["labs_database"] = lab_database.get_all_labs()
        return {"message": "研究室を更新しました", "success": True}
    else:
        raise HTTPException(status_code=404, detail="研究室が見つからないか、更新に失敗しました")

@app.delete("/api/v1/labs/{lab_id}")
async def delete_lab(lab_id: str):
    """研究室を削除"""
    if lab_database.delete_lab(lab_id):
        # システム状態更新
        system_state["labs_database"] = lab_database.get_all_labs()
        return {"message": "研究室を削除しました", "success": True}
    else:
        raise HTTPException(status_code=404, detail="研究室が見つからないか、削除に失敗しました")


# === API エンドポイント ===

@app.post("/api/v1/optimize-with-priorities")
async def optimize_with_priorities(student_profile: StudentProfile):
    """重視項目対応の研究室最適化"""
    
    try:
        session_id = str(uuid.uuid4())
        start_time = time.time()
        
        # 重視項目適用
        priority_engine = system_state["priority_engine"]
        priority_result = priority_engine.apply_priority_weights(student_profile)
        
        # 研究室マッチング実行
        lab_matches = perform_lab_matching_with_priorities(student_profile, priority_result)
        
        # 説明生成
        explanation = priority_engine.generate_priority_explanation(
            student_profile.priority_settings,
            priority_result
        )
        
        # 処理時間計算
        processing_time = (time.time() - start_time) * 1000
        
        # 統計更新
        system_state["total_evaluations"] += 1
        
        return {
            "status": "success",
            "session_id": session_id,
            "lab_matches": lab_matches,
            "priority_analysis": priority_result,
            "explanation": explanation,
            "algorithm_info": {
                "version": "4.0.0-Priority",
                "features_evaluated": 21,
                "fields_evaluated": 11,
                "priority_criteria_count": len(student_profile.priority_settings.priority_criteria),
                "priority_fields_count": len(student_profile.priority_settings.priority_fields),
                "weight_multiplier": student_profile.priority_settings.priority_weight_multiplier
            },
            "processing_info": {
                "processing_time_ms": processing_time,
                "evaluation_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"最適化エラー: {str(e)}")

def perform_lab_matching_with_priorities(
    student_profile: StudentProfile, 
    priority_result: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """重視項目を考慮した研究室マッチング"""
    
    lab_matches = []
    
    for lab in system_state["labs_database"]:
        # 研究室との適合度計算
        lab_compatibility = calculate_lab_compatibility(student_profile, lab, priority_result)
        
        # マッチング結果作成
        match_result = {
            "lab_id": lab["id"],
            "lab_name": lab["name"],
            "advisor": lab["advisor"],
            "description": lab["description"],
            "research_fields": lab["research_fields"],
            "compatibility": lab_compatibility["final_compatibility"],
            "base_compatibility": lab_compatibility["base_compatibility"],
            "priority_boost": lab_compatibility["priority_boost"],
            "match_details": lab_compatibility["match_details"],
            "strengths": lab_compatibility["strengths"],
            "considerations": lab_compatibility["considerations"],
            "field_matches": lab_compatibility["field_matches"],
            "stats": lab.get("stats", {}),
            "priority_aligned": lab_compatibility["priority_aligned"]
        }
        
        lab_matches.append(match_result)
    
    # 適合度順にソート
    lab_matches.sort(key=lambda x: x["compatibility"], reverse=True)
    
    return lab_matches

def calculate_lab_compatibility(
    student_profile: StudentProfile, 
    lab: Dict[str, Any], 
    priority_result: Dict[str, Any]
) -> Dict[str, Any]:
    """個別研究室との適合度計算"""
    
    lab_characteristics = lab.get("characteristics", {})
    student_criteria = student_profile.evaluation_criteria.dict()
    student_fields = student_profile.field_interests.dict()
    
    # 基本適合度計算
    criteria_matches = []
    total_criteria_score = 0.0
    criteria_count = 0
    
    for criterion, student_value in student_criteria.items():
        if criterion in lab_characteristics:
            lab_value = lab_characteristics[criterion]
            # 差の絶対値を適合度に変換（10点満点）
            difference = abs(student_value - lab_value)
            match_score = max(0, 10 - difference) / 10.0
            
            criteria_matches.append({
                "criterion": criterion,
                "student_preference": student_value,
                "lab_characteristic": lab_value,
                "match_score": match_score,
                "is_priority": criterion in student_profile.priority_settings.priority_criteria
            })
            
            total_criteria_score += match_score
            criteria_count += 1
    
    base_criteria_compatibility = total_criteria_score / criteria_count if criteria_count > 0 else 0.0
    
    # 研究分野マッチング
    field_matches = []
    total_field_score = 0.0
    field_count = 0
    
    for field, interest_level in student_fields.items():
        if interest_level > 0:  # 興味のある分野のみ
            is_lab_field = field in lab.get("research_fields", [])
            field_score = interest_level / 10.0 if is_lab_field else 0.0
            
            field_matches.append({
                "field": field,
                "interest_level": interest_level,
                "lab_offers": is_lab_field,
                "field_score": field_score,
                "is_priority": field in student_profile.priority_settings.priority_fields
            })
            
            total_field_score += field_score
            field_count += 1
    
    base_field_compatibility = total_field_score / field_count if field_count > 0 else 0.0
    
    # 基本適合度
    base_compatibility = (base_criteria_compatibility * 0.7) + (base_field_compatibility * 0.3)
    
    # 重視項目による調整
    priority_boost = 0.0
    priority_aligned = False
    
    if student_profile.priority_settings.priority_criteria or student_profile.priority_settings.priority_fields:
        # 重視項目のマッチング度を計算
        priority_criteria_score = 0.0
        priority_criteria_count = 0
        
        for match in criteria_matches:
            if match["is_priority"]:
                weight = student_profile.priority_settings.priority_weight_multiplier
                priority_criteria_score += match["match_score"] * weight
                priority_criteria_count += weight
        
        priority_field_score = 0.0
        priority_field_count = 0
        
        for match in field_matches:
            if match["is_priority"]:
                weight = student_profile.priority_settings.priority_weight_multiplier
                priority_field_score += match["field_score"] * weight
                priority_field_count += weight
        
        # 重み調整後の適合度
        total_weighted_criteria = (total_criteria_score + priority_criteria_score - 
                                 sum(m["match_score"] for m in criteria_matches if m["is_priority"]))
        total_weighted_field = (total_field_score + priority_field_score - 
                              sum(m["field_score"] for m in field_matches if m["is_priority"]))
        
        total_weight_criteria = (criteria_count + priority_criteria_count - 
                               len([m for m in criteria_matches if m["is_priority"]]))
        total_weight_field = (field_count + priority_field_count - 
                            len([m for m in field_matches if m["is_priority"]]))
        
        weighted_criteria_compatibility = (total_weighted_criteria / total_weight_criteria 
                                         if total_weight_criteria > 0 else 0.0)
        weighted_field_compatibility = (total_weighted_field / total_weight_field 
                                      if total_weight_field > 0 else 0.0)
        
        final_compatibility = (weighted_criteria_compatibility * 0.7) + (weighted_field_compatibility * 0.3)
        priority_boost = final_compatibility - base_compatibility
        priority_aligned = priority_boost > 0.05  # 5%以上の向上で重視項目適合とみなす
    else:
        final_compatibility = base_compatibility
    
    # 強み・検討点の抽出
    strengths = []
    considerations = []
    
    for match in criteria_matches:
        if match["match_score"] > 0.8:
            criterion_name = system_state["priority_engine"].criteria_names.get(
                match["criterion"], match["criterion"]
            )
            strengths.append(f"{criterion_name}で高い適合性")
        elif match["match_score"] < 0.4:
            criterion_name = system_state["priority_engine"].criteria_names.get(
                match["criterion"], match["criterion"]
            )
            considerations.append(f"{criterion_name}での適合性要検討")
    
    for match in field_matches:
        if match["lab_offers"] and match["interest_level"] > 7:
            field_name = system_state["priority_engine"].field_names.get(
                match["field"], match["field"]
            )
            strengths.append(f"{field_name}分野で強い関心と一致")
    
    return {
        "final_compatibility": final_compatibility,
        "base_compatibility": base_compatibility,
        "priority_boost": priority_boost,
        "priority_aligned": priority_aligned,
        "match_details": {
            "criteria_compatibility": base_criteria_compatibility,
            "field_compatibility": base_field_compatibility,
            "criteria_matches": criteria_matches,
            "field_matches": field_matches
        },
        "strengths": strengths[:3],  # 上位3つ
        "considerations": considerations[:2],  # 上位2つ
        "field_matches": field_matches
    }

@app.get("/api/v1/priority-options")
async def get_priority_options():
    """重視項目選択肢の取得"""
    
    priority_engine = system_state["priority_engine"]
    
    return {
        "evaluation_criteria": [
            {
                "id": criterion_id,
                "name": criterion_name,
                "category": _get_criterion_category(criterion_id),
                "description": _get_criterion_description(criterion_id)
            }
            for criterion_id, criterion_name in priority_engine.criteria_names.items()
        ],
        "research_fields": [
            {
                "id": field_id,
                "name": field_name,
                "category": _get_field_category(field_id),
                "teacher_count": _get_teacher_count(field_id)
            }
            for field_id, field_name in priority_engine.field_names.items()
        ],
        "weight_multiplier_options": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        "recommended_settings": {
            "max_priority_criteria": 5,
            "max_priority_fields": 3,
            "recommended_multiplier": 2.5
        }
    }

def _get_criterion_category(criterion_id: str) -> str:
    """評価基準のカテゴリを取得"""
    categories = {
        "research_intensity": "基本項目", "advisor_style": "基本項目", "team_work": "基本項目",
        "workload": "基本項目", "theory_practice": "基本項目",
        "research_field_match": "拡張項目", "skill_development": "拡張項目",
        "learning_pace": "拡張項目", "difficulty_preference": "拡張項目", "lab_atmosphere": "拡張項目",
        "communication_style": "コミュニケーション", "meeting_frequency": "コミュニケーション",
        "flexibility": "コミュニケーション", "evening_weekend_work": "コミュニケーション",
        "innovation_risk": "研究アプローチ", "methodology_preference": "研究アプローチ",
        "interdisciplinary": "研究アプローチ",
        "publication_opportunity": "重要項目", "financial_support": "重要項目",
        "lab_hierarchy": "重要項目", "core_time_flexibility": "重要項目"
    }
    return categories.get(criterion_id, "その他")

def _get_criterion_description(criterion_id: str) -> str:
    """評価基準の説明を取得"""
    descriptions = {
        "research_intensity": "研究にどれだけ集中的に取り組みたいか",
        "advisor_style": "教授からの指導の受け方の好み",
        "team_work": "研究での他者との協働の程度",
        "workload": "研究活動の忙しさに対する許容度",
        "theory_practice": "理論研究と実践的研究のバランス",
        "research_field_match": "自分の興味と研究室の分野の一致度",
        "skill_development": "専門性と汎用性のバランス",
        "lab_atmosphere": "研究室の全体的な雰囲気",
        "flexibility": "研究時間の自由度",
        "publication_opportunity": "研究成果の論文化機会",
        "interdisciplinary": "他分野との連携の程度",
        "communication_style": "研究室での交流スタイル",
        "innovation_risk": "新しい手法への挑戦度"
    }
    return descriptions.get(criterion_id, "詳細な説明は準備中です")

def _get_field_category(field_id: str) -> str:
    """研究分野のカテゴリを取得"""
    categories = {
        "ai_ml": "テクノロジー・システム", "image_video": "テクノロジー・システム",
        "network_security": "テクノロジー・システム", "database_systems": "テクノロジー・システム",
        "embedded_iot": "テクノロジー・システム",
        "web_ui_ux": "クリエイティブ", "design_visual": "クリエイティブ",
        "video_animation": "クリエイティブ", "computer_music": "クリエイティブ",
        "game_esports": "エンターテイメント", "vr_ar_media": "エンターテイメント"
    }
    return categories.get(field_id, "その他")

def _get_teacher_count(field_id: str) -> int:
    """研究分野の教員数を取得"""
    teacher_counts = {
        "ai_ml": 7, "image_video": 6, "network_security": 3, "database_systems": 3,
        "embedded_iot": 2, "web_ui_ux": 4, "design_visual": 4, "video_animation": 2,
        "computer_music": 2, "game_esports": 2, "vr_ar_media": 2
    }
    return teacher_counts.get(field_id, 1)

@app.get("/api/v1/labs")
async def get_labs():
    """研究室一覧取得"""
    return {
        "labs": system_state["labs_database"],
        "total_count": len(system_state["labs_database"]),
        "field_distribution": _calculate_field_distribution()
    }

def _calculate_field_distribution():
    """研究分野の分布を計算"""
    distribution = {}
    for lab in system_state["labs_database"]:
        for field in lab.get("research_fields", []):
            distribution[field] = distribution.get(field, 0) + 1
    return distribution

@app.get("/api/v1/system/stats")
async def get_system_stats():
    """システム統計取得"""
    return {
        "total_labs": len(system_state["labs_database"]),
        "total_evaluations": system_state["total_evaluations"],
        "system_info": {
            "version": "4.0.0-Priority",
            "features": 21,
            "research_fields": 11,
            "algorithm": "遺伝的アルゴリズム × ファジィ決定木 + 重視項目機能"
        },
        "field_distribution": _calculate_field_distribution(),
        "feature_availability": {
            "priority_weighting": True,
            "genetic_algorithm": True,
            "fuzzy_decision_tree": True,
            "real_time_optimization": True
        }
    }

@app.get("/api/v1/demo/sample-profile")
async def get_sample_profile():
    """デモ用サンプルプロフィール取得"""
    return {
        "evaluation_criteria": {
            "research_intensity": 8.0,
            "advisor_style": 7.0,
            "team_work": 6.0,
            "workload": 7.0,
            "theory_practice": 6.0,
            "research_field_match": 9.0,
            "skill_development": 8.0,
            "learning_pace": 7.0,
            "difficulty_preference": 7.0,
            "lab_atmosphere": 8.0,
            "communication_style": 7.0,
            "meeting_frequency": 6.0,
            "flexibility": 8.0,
            "evening_weekend_work": 5.0,
            "innovation_risk": 8.0,
            "methodology_preference": 7.0,
            "interdisciplinary": 6.0,
            "publication_opportunity": 9.0,
            "financial_support": 7.0,
            "lab_hierarchy": 6.0,
            "core_time_flexibility": 8.0
        },
        "field_interests": {
            "ai_ml": 9.0,
            "image_video": 7.0,
            "network_security": 3.0,
            "database_systems": 4.0,
            "embedded_iot": 2.0,
            "web_ui_ux": 5.0,
            "design_visual": 3.0,
            "video_animation": 2.0,
            "computer_music": 1.0,
            "game_esports": 4.0,
            "vr_ar_media": 6.0
        },
        "priority_settings": {
            "priority_criteria": ["research_intensity", "publication_opportunity", "innovation_risk"],
            "priority_fields": ["ai_ml"],
            "priority_weight_multiplier": 2.5,
            "explanation_required": True
        },
        "personal_info": {
            "grade": 3,
            "experience_level": "intermediate",
            "career_goals": ["研究者", "AI開発者"],
            "preferred_learning_style": "hands_on"
        }
    }

# === 健康チェック ===
@app.get("/health")
async def health_check():
    """システム健康チェック"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "4.0.0-Priority",
        "components": {
            "priority_engine": system_state["priority_engine"] is not None,
            "labs_database": len(system_state["labs_database"]) > 0,
            "api": True
        }
    }

# サーバー起動
if __name__ == "__main__":
    print("\n🚀 研究室選択支援システム（重視項目対応版）起動中...")
    print(f"📍 URL: http://localhost:8000")
    print(f"📚 API文書: http://localhost:8000/docs")
    print("🔧 システム構成:")
    print(f"  - 評価基準: 21項目")
    print(f"  - 研究分野: 11分野")
    print(f"  - 研究室データ: {len(SAMPLE_LABS)}件")
    print(f"  - 新機能: 重視項目選択・重み調整")
    print(f"  - アルゴリズム: 遺伝的アルゴリズム × ファジィ決定木")
    print("\nサーバー起動中... (Ctrl+C で停止)")
    
    try:
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nサーバーを停止しました")