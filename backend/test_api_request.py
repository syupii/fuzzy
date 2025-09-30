#!/usr/bin/env python3
"""
API動作確認スクリプト
詳細なエラーメッセージを表示
"""

import requests
import json

BASE_URL = "http://localhost:8000"

print("=" * 70)
print("API動作確認")
print("=" * 70)

# ===== 1. ヘルスチェック =====
print("\n1️⃣ ヘルスチェック:")
print("-" * 70)

try:
    response = requests.get(f"{BASE_URL}/health")
    print(f"ステータスコード: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ システム状態: {data.get('status')}")
        
        system_info = data.get('system_info', {})
        print(f"\nシステム情報:")
        print(f"  研究室数: {system_info.get('labs_count')}")
        print(f"  評価基準: {system_info.get('evaluation_criteria')}項目")
        
        modules = data.get('modules', {})
        print(f"\nモジュール状態:")
        for module, available in modules.items():
            status = "✅" if available else "❌"
            print(f"  {status} {module}")
    else:
        print(f"❌ エラー: {response.text}")
        
except Exception as e:
    print(f"❌ 接続エラー: {e}")

# ===== 2. 評価基準取得 =====
print("\n" + "=" * 70)
print("2️⃣ 評価基準取得:")
print("-" * 70)

try:
    response = requests.get(f"{BASE_URL}/api/criteria")
    print(f"ステータスコード: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ 評価基準数: {data.get('total_count')}項目")
        
        criteria_list = data.get('criteria', [])
        print(f"\n評価基準（最初の5項目）:")
        for i, criterion in enumerate(criteria_list[:5], 1):
            print(f"  {i}. {criterion['name']}")
    else:
        print(f"❌ エラー: {response.text}")
        
except Exception as e:
    print(f"❌ エラー: {e}")

# ===== 3. 研究室データ取得 =====
print("\n" + "=" * 70)
print("3️⃣ 研究室データ取得:")
print("-" * 70)

try:
    response = requests.get(f"{BASE_URL}/api/labs")
    print(f"ステータスコード: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        labs = data.get('labs', [])
        print(f"✅ 研究室数: {len(labs)}件")
        
        if labs:
            print(f"\n研究室サンプル（最初の3件）:")
            for i, lab in enumerate(labs[:3], 1):
                print(f"  {i}. {lab.get('name')} ({lab.get('professor')})")
                
                # featuresの確認
                features = lab.get('features', {})
                print(f"     評価項目数: {len(features)}")
                
                # 13項目が揃っているか確認
                required_criteria = [
                    "research_intensity", "advisor_style", "team_work",
                    "workload", "theory_practice", "research_field_match",
                    "skill_development", "lab_atmosphere", "flexibility",
                    "publication_opportunity", "interdisciplinary",
                    "communication_style", "innovation_risk"
                ]
                
                missing = [c for c in required_criteria if c not in features]
                if missing:
                    print(f"     ⚠️ 不足項目: {', '.join(missing)}")
                else:
                    print(f"     ✅ 13項目完備")
    else:
        print(f"❌ エラー: {response.text}")
        
except Exception as e:
    print(f"❌ エラー: {e}")

# ===== 4. 適合度評価（シンプル版） =====
print("\n" + "=" * 70)
print("4️⃣ 適合度評価（シンプル版）:")
print("-" * 70)

# すべて5.0の中間値でテスト
simple_profile = {
    "student_profile": {
        "research_intensity": 5.0,
        "advisor_style": 5.0,
        "team_work": 5.0,
        "workload": 5.0,
        "theory_practice": 5.0,
        "research_field_match": 5.0,
        "skill_development": 5.0,
        "lab_atmosphere": 5.0,
        "flexibility": 5.0,
        "publication_opportunity": 5.0,
        "interdisciplinary": 5.0,
        "communication_style": 5.0,
        "innovation_risk": 5.0
    }
}

print("\nリクエストデータ:")
print(json.dumps(simple_profile, indent=2, ensure_ascii=False))

try:
    response = requests.post(
        f"{BASE_URL}/api/evaluate",
        json=simple_profile
    )
    
    print(f"\nステータスコード: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ 評価成功")
        
        results = data.get('evaluation_results', [])
        print(f"\n評価結果: {len(results)}件")
        
        top_matches = data.get('top_matches', [])
        if top_matches:
            print(f"\n上位3件:")
            for i, match in enumerate(top_matches[:3], 1):
                compatibility = match.get('overall_compatibility', 0)
                print(f"  {i}. {match.get('lab_name')}")
                print(f"     適合度: {compatibility:.2%}")
                print(f"     推薦: {match.get('recommendation_level')}")
    else:
        print(f"❌ エラー発生")
        print(f"\nレスポンス:")
        try:
            error_data = response.json()
            print(json.dumps(error_data, indent=2, ensure_ascii=False))
        except:
            print(response.text)
        
except Exception as e:
    print(f"❌ エラー: {e}")
    import traceback
    traceback.print_exc()

# ===== 5. 適合度評価（詳細版） =====
print("\n" + "=" * 70)
print("5️⃣ 適合度評価（分野興味度付き）:")
print("-" * 70)

detailed_profile = {
    "student_profile": {
        "research_intensity": 8.0,
        "advisor_style": 7.0,
        "team_work": 7.5,
        "workload": 8.0,
        "theory_practice": 6.0,
        "research_field_match": 8.5,
        "skill_development": 8.0,
        "lab_atmosphere": 7.0,
        "flexibility": 6.5,
        "publication_opportunity": 8.5,
        "interdisciplinary": 7.0,
        "communication_style": 7.0,
        "innovation_risk": 7.5
    },
    "field_interests": {
        "人工知能・機械学習": 9.0,
        "画像・映像処理": 7.5
    }
}

print("\nリクエストデータ:")
print(json.dumps(detailed_profile, indent=2, ensure_ascii=False))

try:
    response = requests.post(
        f"{BASE_URL}/api/evaluate",
        json=detailed_profile
    )
    
    print(f"\nステータスコード: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ 評価成功")
        
        top_matches = data.get('top_matches', [])
        if top_matches:
            print(f"\n上位3件:")
            for i, match in enumerate(top_matches[:3], 1):
                compatibility = match.get('overall_compatibility', 0)
                print(f"  {i}. {match.get('lab_name')}")
                print(f"     適合度: {compatibility:.2%}")
                print(f"     推薦: {match.get('recommendation_level')}")
                print(f"     分野: {', '.join(match.get('research_fields', [])[:2])}")
    else:
        print(f"❌ エラー発生")
        print(f"\nレスポンス:")
        try:
            error_data = response.json()
            print(json.dumps(error_data, indent=2, ensure_ascii=False))
        except:
            print(response.text)
        
except Exception as e:
    print(f"❌ エラー: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("テスト完了")
print("=" * 70)