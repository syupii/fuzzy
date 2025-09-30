#!/usr/bin/env python3
"""
詳細なエラー診断
"""

import sys
from pathlib import Path
import traceback

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("詳細エラー診断")
print("=" * 70)

# ===== 1. ファジィ推論 =====
print("\n1️⃣ ファジィ推論モジュール:")
print("-" * 70)
try:
    from core.fuzzy.inference import SimpleFuzzyInferenceEngine
    print("✅ SimpleFuzzyInferenceEngine インポート成功")
except Exception as e:
    print(f"❌ エラー発生:")
    print(f"   タイプ: {type(e).__name__}")
    print(f"   メッセージ: {e}")
    print("\n詳細なトレースバック:")
    traceback.print_exc()

# ===== 2. 決定木 =====
print("\n" + "=" * 70)
print("2️⃣ ファジィ決定木モジュール:")
print("-" * 70)
try:
    from core.decision_tree.tree import FuzzyDecisionTree
    print("✅ FuzzyDecisionTree インポート成功")
except Exception as e:
    print(f"❌ エラー発生:")
    print(f"   タイプ: {type(e).__name__}")
    print(f"   メッセージ: {e}")
    print("\n詳細なトレースバック:")
    traceback.print_exc()

# ===== 3. __init__.pyの内容確認 =====
print("\n" + "=" * 70)
print("3️⃣ __init__.py ファイルの内容確認:")
print("-" * 70)

init_files = [
    "core/fuzzy/__init__.py",
    "core/decision_tree/__init__.py"
]

for init_file in init_files:
    init_path = project_root / init_file
    print(f"\n📄 {init_file}:")
    
    if init_path.exists():
        try:
            content = init_path.read_text(encoding='utf-8')
            lines = content.split('\n')
            print(f"   行数: {len(lines)}")
            print(f"   最初の10行:")
            for i, line in enumerate(lines[:10], 1):
                print(f"   {i:2d}: {line[:70]}")
        except Exception as e:
            print(f"   ❌ 読み込みエラー: {e}")
    else:
        print(f"   ❌ ファイルが存在しません")

print("\n" + "=" * 70)