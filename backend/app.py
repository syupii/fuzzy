#!/usr/bin/env python3
"""
遺伝的アルゴリズムを用いたファジィ決定木研究室選択支援システム
完全13項目対応版 - app.py (DeprecationWarning修正版)
"""

import os
import sys
import time
import json
import math
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager

# FastAPI関連
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# プロジェクトルート設定
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# システム状態管理
system_state = {
    "initialized": False,
    "lab_data": [],
    "evaluation_count": 0,
    "last_updated": None,
    "database_version": "2.0.0"
}

# 評価基準の完全な定義（13項目）
COMPLETE_EVALUATION_CRITERIA = [
    "research_intensity", "advisor_style", "team_work", "workload", "theory_practice",
    "research_field_match", "skill_development", "lab_atmosphere", "flexibility", 
    "publication_opportunity", "interdisciplinary", "communication_style", "innovation_risk"
]

# 基準別デフォルト重み（完全13項目対応）
DEFAULT_CRITERIA_WEIGHTS = {
    # 基本項目：高重み
    "research_intensity": 1.2,
    "advisor_style": 1.1,
    "team_work": 1.0,
    "workload": 1.0,
    "theory_practice": 1.1,
    
    # 拡張項目：中重み
    "research_field_match": 1.3,  # 特に重要
    "skill_development": 0.9,
    "lab_atmosphere": 0.8,
    "flexibility": 0.8,
    "publication_opportunity": 1.0,
    
    # 特殊項目：調整重み
    "interdisciplinary": 0.7,
    "communication_style": 0.8,
    "innovation_risk": 0.9
}

# 研究分野データ（18分野）
RESEARCH_FIELDS_DATA = [
    # テクノロジー・システム分野（12分野）
    {"id": "ai_ml", "name": "人工知能・機械学習", "category": "テクノロジー・システム", "description": "AI技術、機械学習、データサイエンス", "faculty_count": 7},
    {"id": "image_processing", "name": "画像・映像処理", "category": "テクノロジー・システム", "description": "コンピュータビジョン、画像解析、映像技術", "faculty_count": 6},
    {"id": "network_security", "name": "ネットワーク・セキュリティ", "category": "テクノロジー・システム", "description": "ネットワーク技術、情報セキュリティ、通信システム", "faculty_count": 3},
    {"id": "database_systems", "name": "データベース・情報システム", "category": "テクノロジー・システム", "description": "データベース設計、情報システム開発", "faculty_count": 3},
    {"id": "embedded_iot", "name": "組込み・IoT", "category": "テクノロジー・システム", "description": "組込みシステム、IoT、ユビキタス", "faculty_count": 2},
    {"id": "education_linguistics", "name": "教育・言語学", "category": "テクノロジー・システム", "description": "教育技術、言語処理、学習支援システム", "faculty_count": 5},
    {"id": "natural_science", "name": "自然科学・数理", "category": "テクノロジー・システム", "description": "数理科学、自然科学計算、統計解析", "faculty_count": 6},
    {"id": "medical_informatics", "name": "医療情報・ヘルスケア", "category": "テクノロジー・システム", "description": "医療情報システム、ヘルスケアIT、生体工学", "faculty_count": 2},
    {"id": "tourism_systems", "name": "観光情報・地域システム", "category": "テクノロジー・システム", "description": "観光情報システム、地域活性化IT、GIS", "faculty_count": 2},
    {"id": "business_systems", "name": "経営情報・意思決定支援", "category": "テクノロジー・システム", "description": "経営情報システム、意思決定支援、ビジネスアナリティクス", "faculty_count": 3},
    {"id": "audio_processing", "name": "音声・音響情報処理", "category": "テクノロジー・システム", "description": "音声認識、音響解析、音響信号処理", "faculty_count": 2},
    {"id": "system_operations", "name": "システム運用・情報倫理", "category": "テクノロジー・システム", "description": "システム運用、情報セキュリティ、IT倫理", "faculty_count": 3},
    
    # クリエイティブ分野（4分野）
    {"id": "web_ui_ux", "name": "Webデザイン・UI/UX", "category": "クリエイティブ", "description": "Webデザイン、ユーザーインターフェース、UX設計", "faculty_count": 4},
    {"id": "design_visual", "name": "デザイン・視覚表現", "category": "クリエイティブ", "description": "グラフィックデザイン、視覚デザイン、デジタルアート", "faculty_count": 4},
    {"id": "video_animation", "name": "映像・アニメーション", "category": "クリエイティブ", "description": "映像制作、アニメーション技術、モーショングラフィックス", "faculty_count": 2},
    {"id": "computer_music", "name": "コンピュータ音楽・サウンドアート", "category": "クリエイティブ", "description": "電子音楽、サウンドデザイン、音響アート", "faculty_count": 2},
    
    # エンターテイメント分野（2分野）
    {"id": "game_esports", "name": "ゲーム開発・eスポーツ", "category": "エンターテイメント", "description": "ゲーム開発、eスポーツ技術、ゲーミフィケーション", "faculty_count": 2},
    {"id": "vr_ar_media", "name": "VR/AR・メディアアート", "category": "エンターテイメント", "description": "VR/AR技術、メディアアート、インタラクティブシステム", "faculty_count": 2}
]

# サンプル研究室データ（13項目完全対応）
SAMPLE_LABS_13_CRITERIA = [
    {
        "id": "ai_lab",
        "name": "人工知能研究室",
        "advisor": "田中教授",
        "description": "機械学習とディープラーニングの研究を行っています",
        "research_field_id": "ai_ml",
        
        # 13項目完全対応の特性値
        "research_intensity": 9.0,
        "advisor_style": 7.0,
        "team_work": 8.0,
        "workload": 8.5,
        "theory_practice": 6.5,
        "research_field_match": 9.5,
        "skill_development": 8.0,
        "lab_atmosphere": 8.5,
        "flexibility": 6.0,
        "publication_opportunity": 9.0,
        "interdisciplinary": 7.0,
        "communication_style": 8.0,
        "innovation_risk": 9.0,
        
        # メタデータ
        "fields": ["機械学習", "深層学習", "自然言語処理"],
        "publications": 45,
        "funding": "高",
        "equipment": "最新GPU クラスタ",
        "graduate_employment": "大手IT企業、研究機関"
    },
    {
        "id": "robotics_lab", 
        "name": "ロボティクス研究室",
        "advisor": "佐藤教授",
        "description": "自律移動ロボットと制御システムの開発",
        "research_field_id": "embedded_iot",
        
        # 13項目完全対応の特性値
        "research_intensity": 8.0,
        "advisor_style": 6.0,
        "team_work": 9.0,
        "workload": 7.5,
        "theory_practice": 8.0,
        "research_field_match": 8.5,
        "skill_development": 9.0,
        "lab_atmosphere": 7.5,
        "flexibility": 5.5,
        "publication_opportunity": 7.0,
        "interdisciplinary": 8.5,
        "communication_style": 7.0,
        "innovation_risk": 8.0,
        
        # メタデータ
        "fields": ["ロボット工学", "制御工学", "コンピュータビジョン"],
        "publications": 32,
        "funding": "中",
        "equipment": "産業用ロボット、センサー",
        "graduate_employment": "製造業、ロボット開発企業"
    },
    {
        "id": "security_lab",
        "name": "サイバーセキュリティ研究室",
        "advisor": "山田教授",
        "description": "暗号技術とネットワークセキュリティ",
        "research_field_id": "network_security",
        
        # 13項目完全対応の特性値
        "research_intensity": 7.5,
        "advisor_style": 8.0,
        "team_work": 6.0,
        "workload": 6.5,
        "theory_practice": 4.0,
        "research_field_match": 8.0,
        "skill_development": 7.5,
        "lab_atmosphere": 6.5,
        "flexibility": 7.0,
        "publication_opportunity": 6.5,
        "interdisciplinary": 5.5,
        "communication_style": 6.0,
        "innovation_risk": 7.5,
        
        # メタデータ
        "fields": ["暗号学", "ネットワークセキュリティ", "プライバシー保護"],
        "publications": 28,
        "funding": "中",
        "equipment": "セキュリティ解析環境",
        "graduate_employment": "金融機関、セキュリティ企業"
    },
    {
        "id": "hci_lab",
        "name": "ヒューマンコンピュータインタラクション研究室",
        "advisor": "鈴木教授",
        "description": "ユーザーインターフェースとユーザビリティの研究",
        "research_field_id": "web_ui_ux",
        
        # 13項目完全対応の特性値
        "research_intensity": 6.0,
        "advisor_style": 9.0,
        "team_work": 8.0,
        "workload": 5.0,
        "theory_practice": 7.0,
        "research_field_match": 7.5,
        "skill_development": 8.5,
        "lab_atmosphere": 9.0,
        "flexibility": 8.5,
        "publication_opportunity": 6.0,
        "interdisciplinary": 8.0,
        "communication_style": 9.5,
        "innovation_risk": 6.5,
        
        # メタデータ
        "fields": ["HCI", "UX/UI", "アクセシビリティ"],
        "publications": 22,
        "funding": "中",
        "equipment": "ユーザビリティ実験室",
        "graduate_employment": "Web開発企業、デザイン会社"
    },
    {
        "id": "theory_lab",
        "name": "計算理論研究室",
        "advisor": "高橋教授", 
        "description": "アルゴリズム理論と計算複雑性",
        "research_field_id": "natural_science",
        
        # 13項目完全対応の特性値
        "research_intensity": 9.5,
        "advisor_style": 5.0,
        "team_work": 4.0,
        "workload": 8.0,
        "theory_practice": 2.0,
        "research_field_match": 9.0,
        "skill_development": 6.0,
        "lab_atmosphere": 4.5,
        "flexibility": 6.0,
        "publication_opportunity": 8.5,
        "interdisciplinary": 4.0,
        "communication_style": 4.0,
        "innovation_risk": 8.0,
        
        # メタデータ
        "fields": ["アルゴリズム", "計算複雑性", "組合せ最適化"],
        "publications": 38,
        "funding": "高",
        "equipment": "高性能計算クラスタ",
        "graduate_employment": "研究機関、大学院進学"
    }
]

# Lifespan管理（FastAPI新バージョン対応）
@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーション ライフサイクル管理"""
    
    # 起動時処理
    print("🚀 システム初期化開始...")
    
    try:
        # 研究室データの初期化（13項目完全対応）
        system_state["lab_data"] = SAMPLE_LABS_13_CRITERIA.copy()
        
        print(f"✅ 研究室データ初期化完了: {len(system_state['lab_data'])}件")
        print(f"✅ 評価基準設定完了: {len(COMPLETE_EVALUATION_CRITERIA)}項目")
        print(f"✅ 研究分野設定完了: {len(RESEARCH_FIELDS_DATA)}分野")
        
        system_state["initialized"] = True
        system_state["last_updated"] = datetime.now().isoformat()
        print("🎉 システム初期化完了!")
        
    except Exception as e:
        print(f"❌ システム初期化エラー: {e}")
        system_state["initialized"] = False
    
    yield  # アプリケーション実行
    
    # 終了時処理
    print("🔄 システム終了処理...")
    system_state["initialized"] = False
    print("✅ システム終了完了")

# FastAPIアプリケーション初期化（lifespan使用）
app = FastAPI(
    title="研究室選択支援システム v2.0 (完全13項目対応版)",
    description="遺伝的アルゴリズムを用いたファジィ決定木による研究室マッチングシステム - 全13項目完全対応",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan  # 新しいライフサイクル管理
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== 13項目完全対応の計算関数 =====

def calculate_complete_compatibility(student_profile: Dict[str, Any], lab_data: Dict[str, Any]) -> Dict[str, Any]:
    """13項目完全対応の適合度計算"""
    
    print(f"🧮 13項目完全適合度計算開始")
    print(f"📝 学生プロフィール: {len(student_profile)}項目")
    print(f"🏛️ 研究室データ: {lab_data.get('name', 'Unknown')}")
    
    # 各基準の適合度計算
    criteria_scores = {}
    total_weighted_score = 0.0
    total_weights = 0.0
    
    for criterion in COMPLETE_EVALUATION_CRITERIA:
        if criterion in student_profile:
            student_value = float(student_profile[criterion])
            
            # 研究室側の値を取得（完全13項目対応）
            lab_value = float(lab_data.get(criterion, 5.0))  # デフォルト値5.0
            
            # 差分に基づくスコア計算（距離の逆数）
            diff = abs(student_value - lab_value)
            max_diff = 9.0  # 最大差分（1-10の範囲）
            similarity_score = max(0.0, 1.0 - (diff / max_diff))
            
            # 基準別重み適用
            weight = DEFAULT_CRITERIA_WEIGHTS.get(criterion, 1.0)
            weighted_score = similarity_score * weight
            
            criteria_scores[criterion] = {
                "student_value": student_value,
                "lab_value": lab_value,
                "difference": diff,
                "similarity_score": similarity_score,
                "weight": weight,
                "weighted_score": weighted_score
            }
            
            total_weighted_score += weighted_score
            total_weights += weight
            
            print(f"  {criterion}: 学生={student_value}, 研究室={lab_value}, 類似度={similarity_score:.3f}, 重み適用後={weighted_score:.3f}")
        else:
            print(f"  ⚠️ {criterion}: 学生プロフィールに不足")
    
    # 基本適合度（重み付き平均）
    base_compatibility = total_weighted_score / total_weights if total_weights > 0 else 0.5
    
    print(f"📊 重み付き基本適合度: {base_compatibility:.3f}")
    
    # 研究分野適合性ボーナス計算
    field_bonus = calculate_field_compatibility_bonus(student_profile, lab_data)
    
    # 最終適合度（基本適合度 + 分野ボーナス）
    final_compatibility = min(1.0, base_compatibility + field_bonus)
    
    # 信頼度計算（データ完全性に基づく）
    data_completeness = len([c for c in COMPLETE_EVALUATION_CRITERIA if c in student_profile]) / len(COMPLETE_EVALUATION_CRITERIA)
    confidence = base_compatibility * data_completeness
    
    print(f"🎯 最終適合度: {final_compatibility:.3f} (データ完全性: {data_completeness:.1%})")
    
    return {
        "overall_score": final_compatibility,
        "base_compatibility": base_compatibility,
        "field_bonus": field_bonus,
        "confidence": confidence,
        "data_completeness": data_completeness,
        "criteria_scores": criteria_scores,
        "criteria_count": len(criteria_scores),
        "method": "complete_13_criteria"
    }

def calculate_field_compatibility_bonus(student_profile: Dict[str, Any], lab_data: Dict[str, Any]) -> float:
    """研究分野適合性ボーナス計算（13項目対応）"""
    
    field_bonus = 0.0
    field_interests = student_profile.get("field_interests", {})
    lab_field = lab_data.get("research_field_id", "")
    
    print(f"🔬 研究分野マッチング:")
    print(f"  研究室分野: {lab_field}")
    print(f"  学生興味: {field_interests}")
    
    if field_interests and lab_field:
        if lab_field in field_interests:
            interest_level = field_interests[lab_field]
            if interest_level > 0:  # 0は「興味なし」を意味
                # 興味度を正規化（0-1範囲）
                normalized_interest = interest_level / 10.0
                
                # research_field_match基準による重み調整
                field_match_weight = student_profile.get("research_field_match", 5.0) / 10.0
                field_bonus = normalized_interest * field_match_weight * 0.2  # 最大20%のボーナス
                
                print(f"  分野マッチボーナス: {field_bonus:.3f} (興味度: {interest_level}/10, 重み: {field_match_weight:.2f})")
        else:
            print(f"  ⚠️ 研究室分野が学生の興味リストにありません")
    else:
        print(f"  ⚠️ 分野情報が不完全です")
    
    return field_bonus

# ===== API エンドポイント =====

@app.get("/api/status")
async def get_system_status():
    """システム状態取得"""
    
    lab_count = len(system_state.get("lab_data", []))
    
    return {
        "system_name": "研究室選択支援システム v2.0 (完全13項目対応版)",
        "version": "2.0.0",
        "status": "active" if system_state.get("initialized") else "disconnected",
        "lab_count": lab_count,
        "research_fields_count": len(RESEARCH_FIELDS_DATA),
        "evaluation_criteria_count": len(COMPLETE_EVALUATION_CRITERIA),
        "evaluation_criteria": COMPLETE_EVALUATION_CRITERIA,
        "criteria_weights": DEFAULT_CRITERIA_WEIGHTS,
        "system_version": "2.0.0",
        "timestamp": time.time(),
        "last_updated": system_state.get("last_updated"),
        "features": {
            "complete_13_criteria": True,
            "weighted_calculation": True,
            "field_bonus": True,
            "comprehensive_explanation": True
        }
    }

@app.get("/api/labs")
async def get_labs():
    """研究室一覧取得（13項目対応）"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # 研究室データに13項目情報を含める
    enriched_labs = []
    for lab in system_state["lab_data"]:
        enriched_lab = lab.copy()
        
        # 13項目特性のサマリー追加
        criteria_summary = {}
        for criterion in COMPLETE_EVALUATION_CRITERIA:
            if criterion in lab:
                criteria_summary[criterion] = {
                    "value": lab[criterion],
                    "name": criterion.replace("_", " ").title()
                }
        
        enriched_lab["criteria_summary"] = criteria_summary
        enriched_labs.append(enriched_lab)
    
    return {
        "labs": enriched_labs,
        "total_count": len(enriched_labs),
        "evaluation_criteria": COMPLETE_EVALUATION_CRITERIA,
        "last_updated": system_state["last_updated"]
    }

@app.post("/api/evaluate")
async def evaluate_compatibility(evaluation_data: Dict[str, Any]):
    """研究室適合性評価（完全13項目対応版）"""
    
    if not system_state["initialized"]:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        print(f"📥 受信データ: {evaluation_data}")
        
        # 入力データの正規化（複数形式対応）
        student_profile = None
        field_interests = {}
        
        # データ形式の自動判定と正規化
        if "student_profile" in evaluation_data:
            student_profile = evaluation_data["student_profile"]
            field_interests = student_profile.get("field_interests", {})
        elif "evaluation_criteria" in evaluation_data:
            student_profile = evaluation_data["evaluation_criteria"]
            field_interests = evaluation_data.get("field_interests", {})
        elif "preferences" in evaluation_data:
            student_profile = evaluation_data["preferences"]
            field_interests = evaluation_data.get("field_interests", {})
        elif "research_intensity" in evaluation_data:
            student_profile = evaluation_data.copy()
            field_interests = student_profile.pop("field_interests", {})
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"未対応のデータ形式。必須項目: {COMPLETE_EVALUATION_CRITERIA[:5]}"
            )
        
        # field_interestsを学生プロフィールに統合
        if field_interests:
            student_profile["field_interests"] = field_interests
        
        print(f"✅ 学生プロフィール正規化完了")
        print(f"📊 入力基準数: {len([k for k in COMPLETE_EVALUATION_CRITERIA if k in student_profile])}/13")
        
        # 必須基準の確認（基本5項目は必須）
        required_criteria = COMPLETE_EVALUATION_CRITERIA[:5]  # 基本5項目
        missing_required = [c for c in required_criteria if c not in student_profile or student_profile[c] is None]
        
        if missing_required:
            missing_names = [c.replace("_", " ").title() for c in missing_required]
            raise HTTPException(
                status_code=400,
                detail=f"必須評価基準が不足しています: {', '.join(missing_names)}"
            )
        
        # 各研究室との適合度計算
        results = []
        
        for lab in system_state["lab_data"]:
            print(f"\n🏛️ 研究室評価: {lab['name']}")
            
            # 13項目完全対応の適合度計算
            compatibility_result = calculate_complete_compatibility(student_profile, lab)
            
            # 推薦レベル決定
            overall_score = compatibility_result["overall_score"]
            if overall_score >= 0.8:
                recommendation_level = "強く推薦"
            elif overall_score >= 0.6:
                recommendation_level = "推薦"
            elif overall_score >= 0.4:
                recommendation_level = "検討可能"
            else:
                recommendation_level = "要慎重検討"
            
            lab_result = {
                "lab_id": lab["id"],
                "lab_name": lab["name"],
                "advisor": lab["advisor"],
                "research_field": lab.get("research_field_id", ""),
                
                # スコア情報
                "overall_compatibility": float(overall_score),
                "base_compatibility": float(compatibility_result["base_compatibility"]),
                "field_bonus": float(compatibility_result["field_bonus"]),
                "confidence": float(compatibility_result["confidence"]),
                
                # 詳細情報
                "criteria_scores": compatibility_result["criteria_scores"],
                "data_completeness": compatibility_result["data_completeness"],
                "method": compatibility_result["method"],
                
                # 推薦情報
                "recommendation": recommendation_level,
                
                # 追加情報
                "metadata": {
                    "fields": lab.get("fields", []),
                    "publications": lab.get("publications", 0),
                    "funding": lab.get("funding", "不明"),
                    "equipment": lab.get("equipment", "標準装備")
                }
            }
            
            results.append(lab_result)
        
        # スコア順にソート
        results.sort(key=lambda x: x["overall_compatibility"], reverse=True)
        
        # ランキング付与
        for i, result in enumerate(results, 1):
            result["ranking"] = i
        
        # 統計情報計算
        scores = [r["overall_compatibility"] for r in results]
        
        summary = {
            "total_labs": len(results),
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "best_match_lab": results[0]["lab_name"] if results else None,
            "evaluation_method": "complete_13_criteria",
            "criteria_completeness": len([k for k in COMPLETE_EVALUATION_CRITERIA if k in student_profile]) / len(COMPLETE_EVALUATION_CRITERIA)
        }
        
        # 評価カウント更新
        system_state["evaluation_count"] += 1
        
        print(f"✅ 評価完了: {len(results)}研究室, 最高スコア: {max(scores):.3f}")
        
        return {
            "results": results,
            "summary": summary,
            "metadata": {
                "evaluation_time": datetime.now().isoformat(),
                "evaluation_count": system_state["evaluation_count"],
                "algorithm_version": "2.0.0-complete-13-criteria",
                "criteria_used": len([k for k in COMPLETE_EVALUATION_CRITERIA if k in student_profile]),
                "total_criteria": len(COMPLETE_EVALUATION_CRITERIA)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ エラー詳細: {e}")
        print(f"❌ トレースバック: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

@app.get("/api/criteria")
async def get_evaluation_criteria():
    """評価基準情報取得（13項目完全版）"""
    
    criteria_descriptions = {
        "research_intensity": "研究にどれだけ集中的に取り組みたいか（1:軽い研究 〜 10:集中研究）",
        "advisor_style": "教授からの指導の受け方の好み（1:厳格指導 〜 10:自由指導）",
        "team_work": "研究での他者との協働の程度（1:個人研究 〜 10:チーム研究）",
        "workload": "研究活動の忙しさに対する許容度（1:軽い負荷 〜 10:重い負荷）",
        "theory_practice": "理論研究と実践的研究のバランス（1:理論重視 〜 10:実践重視）",
        "research_field_match": "自分の興味と研究室の分野の一致度（1:広い分野 〜 10:専門特化）",
        "skill_development": "専門性と汎用性のバランス（1:専門特化 〜 10:幅広いスキル）",
        "lab_atmosphere": "研究室の全体的な雰囲気（1:静寂集中 〜 10:活発議論）",
        "flexibility": "研究時間の自由度（1:固定スケジュール 〜 10:柔軟スケジュール）",
        "publication_opportunity": "研究成果の論文化機会（1:少ない機会 〜 10:豊富な機会）",
        "interdisciplinary": "他分野との連携の程度（1:単一分野 〜 10:学際連携）",
        "communication_style": "研究室での交流スタイル（1:少人数密接 〜 10:オープン交流）",
        "innovation_risk": "新しい手法への挑戦度（1:安全手法 〜 10:革新手法）"
    }
    
    return {
        "total_criteria": len(COMPLETE_EVALUATION_CRITERIA),
        "criteria_list": COMPLETE_EVALUATION_CRITERIA,
        "criteria_groups": {
            "basic": {
                "description": "基本項目（必須入力）",
                "criteria": COMPLETE_EVALUATION_CRITERIA[:5]
            },
            "extended": {
                "description": "拡張項目（推奨入力）",
                "criteria": COMPLETE_EVALUATION_CRITERIA[5:10]
            },
            "special": {
                "description": "特殊項目（詳細分析用）",
                "criteria": COMPLETE_EVALUATION_CRITERIA[10:13]
            }
        },
        "criteria_weights": DEFAULT_CRITERIA_WEIGHTS,
        "criteria_descriptions": criteria_descriptions
    }

@app.get("/api/research-fields")
async def get_research_fields():
    """研究分野データ取得（18分野完全版）"""
    
    # カテゴリ別の集計
    field_categories = {}
    for field in RESEARCH_FIELDS_DATA:
        category = field["category"]
        if category not in field_categories:
            field_categories[category] = []
        field_categories[category].append(field["id"])
    
    return {
        "research_fields": RESEARCH_FIELDS_DATA,
        "total_fields": len(RESEARCH_FIELDS_DATA),
        "total_labs": len(system_state.get("lab_data", [])),
        "field_categories": field_categories,
        "categories": list(field_categories.keys()),
        "field_distribution": {
            "テクノロジー・システム": 12,
            "クリエイティブ": 4,
            "エンターテイメント": 2
        }
    }

@app.get("/api/demo-profile")
async def get_demo_profile():
    """デモ用学生プロフィール（13項目完全版）"""
    
    demo_profile = {
        # 完全13項目
        "research_intensity": 8,
        "advisor_style": 7,
        "team_work": 6,
        "workload": 7,
        "theory_practice": 6,
        "research_field_match": 9,
        "skill_development": 7,
        "lab_atmosphere": 8,
        "flexibility": 6,
        "publication_opportunity": 8,
        "interdisciplinary": 6,
        "communication_style": 7,
        "innovation_risk": 8,
        
        # 研究分野興味
        "field_interests": {
            "ai_ml": 9,
            "image_processing": 7,
            "web_ui_ux": 6,
            "game_esports": 5,
            "natural_science": 4
        }
    }
    
    return demo_profile

# サーバー起動部分
if __name__ == "__main__":
    print("\n🚀 FastAPI サーバー起動中...")
    print(f"📍 URL: http://localhost:8000")
    print(f"📚 API文書: http://localhost:8000/docs")
    print("🔧 システム仕様:")
    print(f"  - 評価基準: {len(COMPLETE_EVALUATION_CRITERIA)}項目（完全対応）")
    print(f"  - 研究分野: {len(RESEARCH_FIELDS_DATA)}分野")
    print(f"  - 研究室データ: {len(SAMPLE_LABS_13_CRITERIA)}件")
    print(f"  - 計算方式: 重み付き13項目完全適合度")
    print(f"  - システムバージョン: 2.0.0-complete")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )