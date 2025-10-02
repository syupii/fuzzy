#!/usr/bin/env python3
"""
統合システムのテストスクリプト
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.matching.field_matcher import FieldMatcher, FieldInterest
from core.matching.integrated_matcher import IntegratedMatcher
import numpy as np


def test_field_matcher():
    """分野マッチャーのテスト"""
    print("\n" + "="*70)
    print("🧪 テスト1: 分野マッチャー")
    print("="*70)
    
    matcher = FieldMatcher()
    
    # テストケース1: 完全一致
    interests = {
        "ai_ml": FieldInterest("ai_ml", interest_level=10, experience_level=5, importance_level=10)
    }
    
    score = matcher.calculate_field_matching_score(interests, "ai_ml")
    print(f"\n✅ 完全一致: {score:.2f} (期待値: 0.95)")
    assert score >= 0.9, "完全一致のスコアが低すぎます"
    
    # テストケース2: 部分一致
    score = matcher.calculate_field_matching_score(interests, "image_processing")  # 同じテクノロジーカテゴリ
    print(f"✅ 部分一致（同カテゴリ）: {score:.2f} (期待値: ~0.48)")
    assert 0.4 <= score <= 0.6, "部分一致のスコアが範囲外です"
    
    # テストケース3: 無関係
    score = matcher.calculate_field_matching_score(interests, "web_design")  # 異なるカテゴリ
    print(f"✅ 無関係: {score:.2f} (期待値: 0.1)")
    assert score == 0.1, "無関係のスコアが正しくありません"
    
    print("\n✅ 分野マッチャーテスト完了")


def test_integrated_matcher():
    """統合マッチャーのテスト"""
    print("\n" + "="*70)
    print("🧪 テスト2: 統合マッチャー")
    print("="*70)
    
    # デフォルト重みで初期化
    matcher = IntegratedMatcher(
        fuzzy_engine=None,
        decision_tree=None,
        field_matcher=FieldMatcher(),
        optimized_weights=np.ones(14) / 14
    )
    
    # 学生プロファイル（分野重視）
    student = {
        "research_intensity": 8.5,
        "advisor_style": 6.0,
        "team_work": 7.5,
        "workload": 8.0,
        "theory_practice": 5.5,
        "research_field_match": 9,  # 分野を重視
        "skill_development": 7.0,
        "lab_atmosphere": 8.0,
        "flexibility": 6.5,
        "publication_opportunity": 9.0,
        "interdisciplinary": 5.0,
        "communication_style": 7.0,
        "field_interests": {
            "ai_ml": {
                "interest_level": 10,
                "experience_level": 5,
                "importance_level": 10
            }
        }
    }
    
    # AI研究室（興味あり）
    ai_lab = {
        "id": "ai_lab",
        "name": "AI研究室",
        "field_id": "ai_ml",
        "research_intensity": 9.0,
        "advisor_style": 7.0,
        "team_work": 8.0,
        "workload": 8.5,
        "theory_practice": 6.0,
        "skill_development": 8.0,
        "lab_atmosphere": 7.0,
        "flexibility": 6.0,
        "publication_opportunity": 9.5,
        "interdisciplinary": 7.0,
        "communication_style": 8.0
    }
    
    result = matcher.calculate_compatibility(student, ai_lab)
    
    print(f"\n【AI研究室との適合度】")
    print(f"  総合適合度: {result.total_compatibility:.1%}")
    print(f"  分野スコア: {result.field_score:.1%} (重み: {result.field_weight:.1%})")
    print(f"  基本スコア: {result.basic_score:.1%} (重み: {result.basic_weight:.1%})")
    print(f"  説明: {result.explanation}")
    
    assert result.total_compatibility > 0.7, "適合度が低すぎます"
    assert result.field_score > 0.9, "分野スコアが低すぎます"
    
    # Web研究室（興味なし）
    web_lab = {
        "id": "web_lab",
        "name": "Web研究室",
        "field_id": "web_design",
        "research_intensity": 6.0,
        "advisor_style": 8.0,
        "team_work": 8.0,
        "workload": 6.0,
        "theory_practice": 9.0,
        "skill_development": 9.0,
        "lab_atmosphere": 9.0,
        "flexibility": 9.0,
        "publication_opportunity": 5.0,
        "interdisciplinary": 8.0,
        "communication_style": 9.0
    }
    
    result2 = matcher.calculate_compatibility(student, web_lab)
    
    print(f"\n【Web研究室との適合度】")
    print(f"  総合適合度: {result2.total_compatibility:.1%}")
    print(f"  分野スコア: {result2.field_score:.1%} (重み: {result2.field_weight:.1%})")
    print(f"  基本スコア: {result2.basic_score:.1%} (重み: {result2.basic_weight:.1%})")
    print(f"  説明: {result2.explanation}")
    
    assert result.total_compatibility > result2.total_compatibility, \
        "分野重視学生の場合、AI研究室の方が高スコアになるべき"
    
    print("\n✅ 統合マッチャーテスト完了")


def test_api_endpoint():
    """APIエンドポイントのテスト"""
    print("\n" + "="*70)
    print("🧪 テスト3: APIエンドポイント")
    print("="*70)
    
    try:
        import requests
        
        # ヘルスチェック
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ ヘルスチェック成功")
            print(f"  ステータス: {data['status']}")
            print(f"  統合マッチング: {data['features']['integrated_matching']}")
        else:
            print(f"\n⚠️ サーバーが起動していません")
            return
        
        # 評価API
        student_profile = {
            "research_intensity": 8.5,
            "advisor_style": 6.0,
            "team_work": 7.5,
            "workload": 8.0,
            "theory_practice": 5.5,
            "research_field_match": 7,
            "skill_development": 7.0,
            "lab_atmosphere": 8.0,
            "flexibility": 6.5,
            "publication_opportunity": 9.0,
            "interdisciplinary": 5.0,
            "communication_style": 7.0,
            "field_interests": {
                "ai_ml": {
                    "interest_level": 10,
                    "experience_level": 5,
                    "importance_level": 10
                }
            }
        }
        
        response = requests.post(
            "http://localhost:8000/api/evaluate",
            json=student_profile
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ 評価API成功")
            print(f"  評価件数: {data['total_labs_evaluated']}")
            print(f"\n  トップ3:")
            for i, result in enumerate(data['evaluation_results'][:3], 1):
                print(f"    {i}. {result['lab_name']}: {result['overall_compatibility']:.1%}")
        else:
            print(f"\n❌ 評価API失敗: {response.status_code}")
        
        print("\n✅ APIエンドポイントテスト完了")
        
    except ImportError:
        print("\n⚠️ requestsライブラリがインストールされていません")
        print("   pip install requests でインストールしてください")
    except Exception as e:
        print(f"\n⚠️ サーバーに接続できません: {e}")
        print("   先に 'python app.py' でサーバーを起動してください")


def main():
    """全テストを実行"""
    print("\n" + "="*70)
    print("🚀 統合システムテスト開始")
    print("="*70)
    
    test_field_matcher()
    test_integrated_matcher()
    test_api_endpoint()
    
    print("\n" + "="*70)
    print("✅ 全テスト完了")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()