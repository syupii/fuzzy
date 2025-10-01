#!/usr/bin/env python3
"""
labs_database.jsonからinnovation_riskを削除
12項目評価基準に統一
"""

import json
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent

print("=" * 70)
print("innovation_risk 削除スクリプト")
print("=" * 70)

# データベースファイルを探す
possible_paths = [
    project_root / "data" / "labs_database.json",
    project_root / "labs_database.json",
]

db_path = None
for path in possible_paths:
    if path.exists():
        db_path = path
        break

if not db_path:
    print("❌ labs_database.json が見つかりません")
    exit(1)

print(f"\n📁 データベースパス: {db_path}")

# データベース読み込み
try:
    with open(db_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except Exception as e:
    print(f"❌ 読み込みエラー: {e}")
    exit(1)

labs = data.get('labs', [])
print(f"📊 研究室数: {len(labs)}件")

# innovation_riskを削除
removed_count = 0
labs_with_innovation_risk = []

for i, lab in enumerate(labs):
    features = lab.get('features', {})
    
    if 'innovation_risk' in features:
        del features['innovation_risk']
        removed_count += 1
        labs_with_innovation_risk.append(lab.get('name', f'Lab #{i+1}'))

print(f"\n🔍 innovation_risk を持っていた研究室: {len(labs_with_innovation_risk)}件")

if labs_with_innovation_risk:
    print(f"\n削除対象の研究室（最初の10件）:")
    for i, lab_name in enumerate(labs_with_innovation_risk[:10], 1):
        print(f"  {i}. {lab_name}")
    
    if len(labs_with_innovation_risk) > 10:
        print(f"  ... 他 {len(labs_with_innovation_risk) - 10}件")

# 12項目が揃っているか確認
required_criteria = [
    "research_intensity", "advisor_style", "team_work",
    "workload", "theory_practice", "research_field_match",
    "skill_development", "lab_atmosphere", "flexibility",
    "publication_opportunity", "interdisciplinary",
    "communication_style"
]

print(f"\n✅ 12項目完備チェック:")
all_complete = True

for i, lab in enumerate(labs):
    features = lab.get('features', {})
    missing = [c for c in required_criteria if c not in features]
    
    if missing:
        all_complete = False
        print(f"  ❌ {lab.get('name', f'Lab #{i+1}')}: 不足 {', '.join(missing)}")

if all_complete:
    print(f"  ✅ すべての研究室が12項目完備")

# バックアップ作成
if removed_count > 0:
    backup_path = db_path.parent / f"labs_database_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n📦 バックアップ作成: {backup_path.name}")
    except Exception as e:
        print(f"\n❌ バックアップ作成失敗: {e}")

# 更新データを保存
data['labs'] = labs
data['last_updated'] = datetime.now().isoformat()
data['version'] = '3.0.0'

try:
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 更新完了:")
    print(f"  - innovation_risk削除: {removed_count}件")
    print(f"  - 保存先: {db_path}")
    
except Exception as e:
    print(f"\n❌ 保存エラー: {e}")
    exit(1)

print("\n" + "=" * 70)
print("✅ 完了！")
print("=" * 70)
print("\n次のステップ:")
print("  1. サーバーを再起動: python app.py")
print("  2. 動作確認: python test_api_request.py")
print("=" * 70)