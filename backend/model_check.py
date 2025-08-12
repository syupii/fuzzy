#!/usr/bin/env python3
"""
🔍 FDTLSS 遺伝的モデル構造診断ツール
現在の遺伝的モデルファイルを詳しく調べて構造の問題を診断します
"""

import pickle
import os
import sys
from pathlib import Path


def inspect_model_file(file_path: str):
    """モデルファイルの構造を詳しく調査"""

    if not os.path.exists(file_path):
        print(f"❌ ファイルが見つかりません: {file_path}")
        return False

    try:
        print(f"🔍 調査中: {file_path}")
        print(f"📁 ファイルサイズ: {os.path.getsize(file_path)} bytes")
        print("-" * 50)

        # モデル読み込み
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)

        # 構造分析
        print(f"📊 ルートオブジェクトタイプ: {type(model_data)}")

        if isinstance(model_data, dict):
            print("📋 辞書キー:")
            for key in model_data.keys():
                value = model_data[key]
                print(f"  - {key}: {type(value)}")

                # 期待される構造をチェック
                if key == 'best_individual' and hasattr(value, 'tree'):
                    print(f"    ✅ best_individualにtree属性を発見")
                    if hasattr(value.tree, 'predict_with_explanation'):
                        print(f"    ✅ treeにpredict_with_explanationメソッドがあります")
                    else:
                        print(f"    ⚠️  treeにpredict_with_explanationメソッドがありません")

                if hasattr(value, '__dict__'):
                    attrs = list(value.__dict__.keys())
                    print(
                        f"    📝 属性: {attrs[:5]}{'...' if len(attrs) > 5 else ''}")

        elif hasattr(model_data, 'tree'):
            print("🌳 直接的なtreeオブジェクトを検出")
            print(f"   Treeタイプ: {type(model_data.tree)}")
            if hasattr(model_data, 'individual_id'):
                print(f"   個体ID: {model_data.individual_id}")
            if hasattr(model_data, 'generation'):
                print(f"   世代: {model_data.generation}")
            if hasattr(model_data, 'fitness_components'):
                print(f"   適応度: {model_data.fitness_components}")

        else:
            print("🤔 不明なモデル構造")
            if hasattr(model_data, '__dict__'):
                print(f"📝 利用可能な属性: {list(model_data.__dict__.keys())}")
            elif hasattr(model_data, '__len__'):
                print(f"📏 オブジェクト長: {len(model_data)}")

        return True

    except Exception as e:
        print(f"❌ モデル読み込みエラー: {e}")
        return False


def main():
    """メイン診断関数"""

    print("🧬 FDTLSS 遺伝的モデル構造診断ツール")
    print("=" * 60)

    # 一般的なモデルの場所をチェック
    potential_paths = [
        "models/genetic_optimization_results.pkl",
        "models/best_genetic_tree.pkl",
        "genetic_optimization_results.pkl",
        "best_genetic_tree.pkl"
    ]

    # modelsディレクトリ内のファイルを追加
    models_dir = Path("models")
    if models_dir.exists():
        for file_path in models_dir.glob("*.pkl"):
            potential_paths.append(str(file_path))

    found_files = []
    for path in potential_paths:
        if os.path.exists(path):
            found_files.append(path)

    if not found_files:
        print("❌ 期待される場所にモデルファイルが見つかりません:")
        for path in potential_paths[:4]:  # 最初の4つの期待されるパスを表示
            print(f"   - {path}")
        print("\n💡 提案: 最適化を実行してモデルを生成してください:")
        print("   python run_genetic_optimization.py --mode optimize")
        return 1

    print(f"📁 {len(found_files)}個のモデルファイルを発見:")

    # 見つかったファイルを各々調査
    success_count = 0
    for file_path in found_files:
        print(f"\n{'='*60}")
        if inspect_model_file(file_path):
            success_count += 1
        print("="*60)

    # 要約
    print(f"\n📊 要約: {success_count}/{len(found_files)} ファイルが正常に調査されました")

    if success_count == 0:
        print("\n🔧 推奨対応:")
        print("1. モデル再生成: python run_genetic_optimization.py --mode optimize")
        print("2. ファイルアクセス権限と整合性の確認")
        print("3. Python環境の互換性確認")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())