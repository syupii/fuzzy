#!/usr/bin/env python3
"""
backend/app.py - 堅牢版
500エラー対策、詳細なログ出力、エラーハンドリング強化
"""

import os
import sys
import json
import time
import uvicorn
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
        app_name = "Lab Matching System"
        host = "0.0.0.0"
        port = 8000
        debug = True
    settings = FallbackSettings()

# FastAPIアプリケーション
app = FastAPI(
    title="研究室選択支援システム",
    version="3.1.0",
    description="priorities対応・エラーハンドリング強化版"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# グローバル例外ハンドラー
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全ての例外をキャッチしてログに出力"""
    print(f"\n{'='*70}")
    print(f"❌ グローバル例外ハンドラー")
    print(f"{'='*70}")
    print(f"パス: {request.url.path}")
    print(f"メソッド: {request.method}")
    print(f"エラー: {str(exc)}")
    print(f"トレースバック:")
    traceback.print_exc()
    print(f"{'='*70}\n")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "type": type(exc).__name__
        }
    )

# 完全な評価基準リスト（12項目）
COMPLETE_CRITERIA = [
    "research_intensity",
    "advisor_style",
    "team_work",
    "workload",
    "theory_practice",
    "research_field_match",
    "skill_development",
    "lab_atmosphere",
    "flexibility",
    "publication_opportunity",
    "interdisciplinary",
    "communication_style"
]

# 基準名マッピング
CRITERIA_NAMES = {
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

# サンプル研究室データ
SAMPLE_LABS = [
    {
        "id": "ai_lab",
        "name": "人工知能研究室",
        "advisor": "田中教授",
        "field_id": "ai_ml",
        "research_area": "人工知能・機械学習",
        "category": "テクノロジー・システム",
        "research_intensity": 0.9,
        "advisor_style": 0.7,
        "team_work": 0.8,
        "workload": 0.8,
        "theory_practice": 0.6,
        "research_field_match": 0.9,
        "skill_development": 0.8,
        "lab_atmosphere": 0.7,
        "flexibility": 0.6,
        "publication_opportunity": 0.9,
        "interdisciplinary": 0.7,
        "communication_style": 0.8
    },
    {
        "id": "image_lab",
        "name": "画像処理研究室",
        "advisor": "佐藤教授",
        "field_id": "image_processing",
        "research_area": "画像・映像処理",
        "category": "テクノロジー・システム",
        "research_intensity": 0.8,
        "advisor_style": 0.6,
        "team_work": 0.7,
        "workload": 0.7,
        "theory_practice": 0.7,
        "research_field_match": 0.8,
        "skill_development": 0.7,
        "lab_atmosphere": 0.6,
        "flexibility": 0.7,
        "publication_opportunity": 0.8,
        "interdisciplinary": 0.6,
        "communication_style": 0.7
    },
    {
        "id": "network_lab",
        "name": "ネットワークセキュリティ研究室",
        "advisor": "鈴木教授",
        "field_id": "network_security",
        "research_area": "ネットワーク・セキュリティ",
        "category": "テクノロジー・システム",
        "research_intensity": 0.85,
        "advisor_style": 0.5,
        "team_work": 0.9,
        "workload": 0.9,
        "theory_practice": 0.5,
        "research_field_match": 0.85,
        "skill_development": 0.9,
        "lab_atmosphere": 0.8,
        "flexibility": 0.5,
        "publication_opportunity": 0.7,
        "interdisciplinary": 0.5,
        "communication_style": 0.9
    },
    {
        "id": "web_design_lab",
        "name": "Webデザイン研究室",
        "advisor": "高橋教授",
        "field_id": "web_design",
        "research_area": "Webデザイン・UI/UX",
        "category": "クリエイティブ",
        "research_intensity": 0.6,
        "advisor_style": 0.8,
        "team_work": 0.8,
        "workload": 0.6,
        "theory_practice": 0.9,
        "research_field_match": 0.7,
        "skill_development": 0.9,
        "lab_atmosphere": 0.9,
        "flexibility": 0.9,
        "publication_opportunity": 0.5,
        "interdisciplinary": 0.8,
        "communication_style": 0.9
    },
    {
        "id": "game_lab",
        "name": "ゲーム開発研究室",
        "advisor": "山田教授",
        "field_id": "game_esports",
        "research_area": "ゲーム開発・eスポーツ",
        "category": "エンターテイメント",
        "research_intensity": 0.7,
        "advisor_style": 0.8,
        "team_work": 0.9,
        "workload": 0.8,
        "theory_practice": 0.8,
        "research_field_match": 0.7,
        "skill_development": 0.8,
        "lab_atmosphere": 0.9,
        "flexibility": 0.8,
        "publication_opportunity": 0.6,
        "interdisciplinary": 0.7,
        "communication_style": 0.9
    }
]

# システム状態
system_state = {
    "initialized": False,
    "lab_database": [],
    "evaluation_count": 0
}


def initialize_system():
    """システム初期化"""
    try:
        print("\n🔧 システム初期化中...")
        system_state["initialized"] = True
        system_state["lab_database"] = SAMPLE_LABS.copy()
        print(f"✅ 研究室データ: {len(SAMPLE_LABS)}件")
        print(f"✅ 評価基準: {len(COMPLETE_CRITERIA)}項目")
        print("✅ システム初期化完了\n")
    except Exception as e:
        print(f"❌ システム初期化エラー: {e}")
        traceback.print_exc()
        system_state["initialized"] = False


# 起動時に初期化
initialize_system()


# ==================== API エンドポイント ====================

@app.get("/")
async def read_root():
    """ルートエンドポイント"""
    return {
        "message": "研究室選択支援システム API",
        "version": "3.1.0",
        "status": "running" if system_state["initialized"] else "error",
        "features": ["priorities対応", "12項目評価", "エラーハンドリング強化"],
        "endpoints": {
            "health": "/health",
            "evaluate": "/api/evaluate",
            "labs": "/api/labs",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {
        "status": "healthy" if system_state["initialized"] else "unhealthy",
        "version": "3.1.0",
        "timestamp": datetime.now().isoformat(),
        "lab_count": len(system_state["lab_database"]),
        "evaluation_count": system_state["evaluation_count"],
        "criteria_count": len(COMPLETE_CRITERIA)
    }


@app.get("/api/labs")
async def get_labs():
    """研究室一覧取得"""
    try:
        if not system_state["initialized"]:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        return {
            "labs": system_state["lab_database"],
            "total_count": len(system_state["lab_database"])
        }
    except Exception as e:
        print(f"❌ 研究室一覧取得エラー: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evaluate")
async def evaluate_compatibility(request_data: Dict[str, Any]):
    """
    学生プロファイルに基づく研究室適合度評価
    priorities オブジェクト完全対応版
    """
    
    try:
        print("\n" + "="*70)
        print("📥 受信したリクエスト:")
        print(json.dumps(request_data, indent=2, ensure_ascii=False))
        print("="*70)
        
        # システム初期化チェック
        if not system_state["initialized"]:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        # リクエストデータの取得
        student_profile_raw = request_data.get("student_profile", {})
        
        if not student_profile_raw:
            raise HTTPException(
                status_code=400,
                detail="student_profile が必要です"
            )
        
        # データのコピーを作成（元のデータを変更しない）
        student_data = student_profile_raw.copy()
        
        # prioritiesとfield_interestsを抽出
        priorities = student_data.pop("priorities", {})
        field_interests = student_data.pop("field_interests", {})
        
        print(f"\n📊 データ抽出:")
        print(f"  評価基準: {len(student_data)}項目")
        print(f"  優先度: {len(priorities)}項目")
        print(f"  研究分野興味: {len(field_interests)}分野")
        
        # 評価基準の正規化（1-10 → 0.0-1.0）
        student_profile = {}
        for criterion in COMPLETE_CRITERIA:
            value = student_data.get(criterion)
            if value is not None:
                try:
                    normalized_value = float(value) / 10.0
                    student_profile[criterion] = max(0.0, min(1.0, normalized_value))
                except (ValueError, TypeError) as e:
                    print(f"⚠️ 値の変換エラー ({criterion}): {value} -> デフォルト0.5")
                    student_profile[criterion] = 0.5
            else:
                student_profile[criterion] = 0.5
        
        print(f"\n✅ 正規化完了: {len(student_profile)}項目")
        
        # 各研究室との適合度計算
        results = []
        
        for lab in system_state["lab_database"]:
            try:
                # 項目別スコア計算
                feature_scores = {}
                for criterion in COMPLETE_CRITERIA:
                    student_val = student_profile.get(criterion, 0.5)
                    lab_val = lab.get(criterion, 0.5)
                    
                    # 類似度計算
                    similarity = 1.0 - abs(student_val - lab_val)
                    feature_scores[criterion] = float(similarity)
                
                # 総合スコア計算（優先度考慮）
                if priorities:
                    # 優先度による重み付き平均
                    total_weighted = 0.0
                    total_weight = 0.0
                    
                    for criterion in COMPLETE_CRITERIA:
                        score = feature_scores.get(criterion, 0.5)
                        priority_value = priorities.get(criterion, 5)
                        
                        try:
                            priority = float(priority_value) / 10.0
                        except (ValueError, TypeError):
                            priority = 0.5
                        
                        total_weighted += score * priority
                        total_weight += priority
                    
                    final_score = total_weighted / max(total_weight, 0.001)
                else:
                    # 優先度なしの場合は単純平均
                    scores_list = list(feature_scores.values())
                    final_score = sum(scores_list) / max(len(scores_list), 1)
                
                # 研究分野ボーナス
                field_match_score = 0.0
                if field_interests:
                    lab_field = lab.get("field_id", "")
                    if lab_field in field_interests:
                        try:
                            field_interest_value = field_interests[lab_field]
                            field_match_score = float(field_interest_value) / 10.0
                            # 分野ボーナスを適用
                            final_score = final_score * 0.7 + field_match_score * 0.3
                        except (ValueError, TypeError):
                            pass
                
                # 範囲制限
                final_score = max(0.0, min(1.0, final_score))
                
                # 信頼度
                confidence = min(1.0, final_score + 0.1)
                
                # 推薦レベル
                if final_score >= 0.8:
                    recommendation = "最優先推薦"
                elif final_score >= 0.7:
                    recommendation = "強く推薦"
                elif final_score >= 0.6:
                    recommendation = "推薦"
                elif final_score >= 0.5:
                    recommendation = "検討可能"
                else:
                    recommendation = "要検討"
                
                # 優先度分析
                priority_analysis = None
                if priorities:
                    try:
                        high_priority_criteria = [c for c, p in priorities.items() if float(p) >= 8]
                        medium_priority_criteria = [c for c, p in priorities.items() if 5 <= float(p) < 8]
                        low_priority_criteria = [c for c, p in priorities.items() if float(p) < 5]
                        
                        high_priority_match = sum(
                            feature_scores.get(c, 0.5) for c in high_priority_criteria
                        ) / max(len(high_priority_criteria), 1)
                        
                        medium_priority_match = sum(
                            feature_scores.get(c, 0.5) for c in medium_priority_criteria
                        ) / max(len(medium_priority_criteria), 1)
                        
                        low_priority_match = sum(
                            feature_scores.get(c, 0.5) for c in low_priority_criteria
                        ) / max(len(low_priority_criteria), 1)
                        
                        priority_analysis = {
                            "high_priority_match": float(high_priority_match),
                            "medium_priority_match": float(medium_priority_match),
                            "low_priority_match": float(low_priority_match),
                            "priority_distribution": {
                                "high": len(high_priority_criteria),
                                "medium": len(medium_priority_criteria),
                                "low": len(low_priority_criteria)
                            },
                            "weighted_priority_score": float(final_score)
                        }
                    except Exception as e:
                        print(f"⚠️ 優先度分析エラー: {e}")
                        priority_analysis = None
                
                # 説明文生成
                explanation = generate_explanation(
                    student_profile, lab, final_score, priorities, feature_scores
                )
                
                # 結果オブジェクト
                lab_result = {
                    "lab_id": str(lab.get("id", "")),
                    "lab_name": str(lab.get("name", "")),
                    "advisor": str(lab.get("advisor", "")),
                    "research_area": str(lab.get("research_area", "")),
                    "category": str(lab.get("category", "")),
                    "final_score": float(final_score),
                    "feature_scores": {k: float(v) for k, v in feature_scores.items()},
                    "confidence": float(confidence),
                    "recommendation": str(recommendation),
                    "explanation": str(explanation)
                }
                
                if priority_analysis:
                    lab_result["priority_analysis"] = priority_analysis
                
                results.append(lab_result)
                
            except Exception as e:
                print(f"⚠️ 研究室 {lab.get('name', '不明')} の評価エラー: {e}")
                traceback.print_exc()
                continue
        
        # 結果をソート
        results.sort(key=lambda x: x["final_score"], reverse=True)
        
        # サマリー計算
        if results:
            scores = [r["final_score"] for r in results]
            high_compatibility_count = len([s for s in scores if s >= 0.7])
            
            summary = {
                "total_labs": len(results),
                "high_compatibility_count": high_compatibility_count,
                "avg_compatibility": float(sum(scores) / len(scores)),
                "max_score": float(max(scores)),
                "min_score": float(min(scores)),
                "priority_weighting_applied": bool(priorities)
            }
        else:
            summary = {
                "total_labs": 0,
                "high_compatibility_count": 0,
                "avg_compatibility": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
                "priority_weighting_applied": False
            }
        
        # 評価回数増加
        system_state["evaluation_count"] += 1
        
        # レスポンス構築
        response = {
            "lab_results": results,
            "summary": summary,
            "metadata": {
                "processing_time": 0.0,
                "evaluation_count": system_state["evaluation_count"],
                "criteria_used": len(student_profile),
                "priorities_used": len(priorities),
                "field_interests_used": len(field_interests),
                "timestamp": datetime.now().isoformat(),
                "ai_engines_used": ["基本アルゴリズム", "優先度重み付け"],
                "calculation_method": "優先度対応重み付き評価"
            }
        }
        
        print(f"\n✅ 評価完了:")
        print(f"  研究室数: {len(results)}件")
        if results:
            print(f"  トップスコア: {max(r['final_score'] for r in results):.3f}")
            print(f"  平均スコア: {summary['avg_compatibility']:.3f}")
        if priorities:
            print(f"  優先度重み付け: 適用済み ({len(priorities)}項目)")
        print("="*70 + "\n")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"❌ 評価処理エラー:")
        print(f"{'='*70}")
        print(f"エラー: {str(e)}")
        print(f"タイプ: {type(e).__name__}")
        print("\nトレースバック:")
        traceback.print_exc()
        print(f"{'='*70}\n")
        
        raise HTTPException(
            status_code=500,
            detail={
                "error": "評価処理エラー",
                "message": str(e),
                "type": type(e).__name__
            }
        )


def generate_explanation(
    student_profile: Dict[str, float],
    lab: Dict[str, Any],
    final_score: float,
    priorities: Dict[str, Any],
    feature_scores: Dict[str, float]
) -> str:
    """説明文生成"""
    
    try:
        # 高優先度項目での適合性
        high_priority_matches = []
        
        if priorities:
            for criterion, priority in priorities.items():
                try:
                    if float(priority) >= 8:
                        score = feature_scores.get(criterion, 0.5)
                        if score >= 0.8:
                            criterion_name = CRITERIA_NAMES.get(criterion, criterion)
                            high_priority_matches.append(criterion_name)
                except (ValueError, TypeError):
                    continue
        
        # 基本説明
        if final_score >= 0.8:
            base_text = "非常に高い適合性を示しています。"
        elif final_score >= 0.7:
            base_text = "高い適合性を示しています。"
        elif final_score >= 0.6:
            base_text = "適度な適合性があります。"
        else:
            base_text = "検討の余地があります。"
        
        # 優先度による補足
        if high_priority_matches:
            priority_text = f"特に重視する{', '.join(high_priority_matches[:2])}において良好です。"
            return f"{base_text} {priority_text}"
        
        return base_text
        
    except Exception as e:
        print(f"⚠️ 説明文生成エラー: {e}")
        return "適合度を評価しました。"


# ==================== サーバー起動 ====================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 研究室選択支援システム - 堅牢版")
    print("="*70)
    print(f"📍 URL: http://localhost:{settings.port}")
    print(f"📚 API文書: http://localhost:{settings.port}/docs")
    print(f"✅ 評価基準: {len(COMPLETE_CRITERIA)}項目")
    print(f"✅ 研究室データ: {len(SAMPLE_LABS)}件")
    print(f"✅ エラーハンドリング: 強化済み")
    print("="*70 + "\n")
    
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )