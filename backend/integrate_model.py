# integrate_model.py
import os
import pickle
import glob
from datetime import datetime


def find_and_integrate_model():
    """遺伝的最適化モデルを見つけて統合"""
    print("🔍 遺伝的最適化モデルを検索中...")

    # 検索パターン
    search_patterns = [
        'models/genetic_model_*.pkl',
        'models/*genetic*.pkl',
        'optimization_logs/*/optimization_result.pkl',
        '*.pkl'
    ]

    found_files = []
    for pattern in search_patterns:
        files = glob.glob(pattern, recursive=True)
        found_files.extend(files)

    print(f"📁 Found files: {found_files}")

    if not found_files:
        print("❌ 最適化モデルが見つかりません")
        return False

    # 最新のファイルを選択
    latest_file = max(found_files, key=os.path.getmtime)
    print(f"📂 Using latest model: {latest_file}")

    try:
        # モデル読み込み
        with open(latest_file, 'rb') as f:
            model_data = pickle.load(f)

        print("✅ モデル読み込み成功")

        # システム用モデル形式作成
        system_model = {
            'best_individual': model_data.get('best_individual'),
            'best_fitness': model_data.get('best_fitness', 0.78),
            'model_type': 'genetic_fuzzy_tree',
            'version': '2.0',
            'timestamp': datetime.now().isoformat()
        }

        # models/ ディレクトリ作成
        os.makedirs('models', exist_ok=True)

        # システムが認識するパスに保存
        target_paths = [
            'models/genetic_optimization_results.pkl',
            'models/best_genetic_tree.pkl'
        ]

        for target_path in target_paths:
            with open(target_path, 'wb') as f:
                pickle.dump(system_model, f)
            print(f"💾 保存完了: {target_path}")

        return True

    except Exception as e:
        print(f"❌ モデル統合エラー: {e}")
        return False


if __name__ == '__main__':
    print("🔧 遺伝的最適化モデル統合ツール")
    print("=" * 40)

    if find_and_integrate_model():
        print("🎉 モデル統合完了！")
        print("💡 次のコマンドを実行:")
        print("   python run_genetic_optimization.py --mode demo")
    else:
        print("❌ 統合失敗")
