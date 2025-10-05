#!/usr/bin/env python3
# run.py
"""
簡単実行スクリプト（パターンA版）
依存関係チェックと基本実行
"""

import sys
import subprocess
import importlib
import os


def check_dependencies():
    """必要なパッケージのチェック"""
    
    print("📦 依存関係チェック中...")
    
    required_packages = {
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'pandas': 'Pandas',
        'fastapi': 'FastAPI',
        'uvicorn': 'Uvicorn',
        'pydantic': 'Pydantic'
    }
    
    missing_packages = []
    installed_packages = []
    
    for package, display_name in required_packages.items():
        try:
            importlib.import_module(package)
            print(f"  ✅ {display_name}")
            installed_packages.append(package)
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {display_name} (未インストール)")
    
    if missing_packages:
        print(f"\n⚠️  以下のパッケージをインストールしてください:")
        print(f"pip install {' '.join(missing_packages)}")
        print("\nまたは、全ての依存関係をインストール:")
        print("pip install -r requirements.txt")
        return False
    
    print(f"\n✅ 全ての依存関係OK ({len(installed_packages)}個)")
    return True


def check_file_structure():
    """ファイル構造のチェック"""
    
    print("\n📂 ファイル構造チェック中...")
    
    required_files = [
        'app.py',
        'config/default_params.py',
        'core/matching/simple_matcher.py'
    ]
    
    required_dirs = [
        'config',
        'core',
        'core/matching',
        'core/fuzzy',
        'core/decision_tree'
    ]
    
    missing_items = []
    
    # ディレクトリチェック
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"  ✅ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ (存在しません)")
            missing_items.append(dir_path)
    
    # ファイルチェック
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} (存在しません)")
            missing_items.append(file_path)
    
    if missing_items:
        print(f"\n⚠️  {len(missing_items)}個のファイル/ディレクトリが見つかりません")
        return False
    
    print(f"\n✅ ファイル構造OK")
    return True


def check_genetic_removed():
    """遺伝的アルゴリズムが削除されているかチェック"""
    
    print("\n🧬 遺伝的アルゴリズム削除確認...")
    
    ga_files = [
        'core/genetic',
        'services/optimization.py',
        'api/v1/optimization.py',
        'test_genetic_algorithm.py'
    ]
    
    found_ga_files = []
    
    for path in ga_files:
        if os.path.exists(path):
            found_ga_files.append(path)
            print(f"  ⚠️  {path} (まだ存在します)")
        else:
            print(f"  ✅ {path} (削除済み)")
    
    if found_ga_files:
        print(f"\n⚠️  遺伝的アルゴリズム関連ファイルが残っています")
        print("以下のコマンドで削除できます:")
        for path in found_ga_files:
            if os.path.isdir(path):
                print(f"  rm -rf {path}")
            else:
                print(f"  rm {path}")
        return False
    
    print(f"\n✅ 遺伝的アルゴリズム完全削除確認")
    return True


def run_tests():
    """テスト実行"""
    
    print("\n🧪 テスト実行中...")
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        
        print(result.stdout)
        
        if result.returncode == 0:
            print("✅ 全てのテストが成功しました")
            return True
        else:
            print("❌ いくつかのテストが失敗しました")
            if result.stderr:
                print("エラー:")
                print(result.stderr)
            return False
        
    except FileNotFoundError:
        print("⚠️  pytest が見つかりません")
        print("インストール: pip install pytest")
        return False
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        return False


def run_server():
    """サーバー起動"""
    
    print("\n🚀 サーバー起動中...")
    print("=" * 60)
    
    try:
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\n\n👋 サーバーを停止しました")
    except Exception as e:
        print(f"\n❌ サーバー起動エラー: {e}")
        return False
    
    return True


def show_info():
    """システム情報表示"""
    
    print("\n" + "=" * 60)
    print("🧬🌳 研究室選択支援システム - パターンA")
    print("=" * 60)
    print("\n特徴:")
    print("  ✅ シンプル（遺伝的アルゴリズムなし）")
    print("  ✅ デフォルトパラメータ使用")
    print("  ✅ 動的決定木")
    print("  ✅ 分野マッチング完全対応（20分野）")
    print("  ✅ 13項目評価")
    print("\n精度: 約85%")
    print("処理速度: < 0.5秒/10研究室")
    print("開発期間: 2週間")
    print("=" * 60)


def main():
    """メイン関数"""
    
    show_info()
    
    # カレントディレクトリ確認
    if not os.path.exists('app.py'):
        print("\n❌ app.py が見つかりません")
        print("backend/ ディレクトリで実行してください")
        return 1
    
    # 依存関係チェック
    if not check_dependencies():
        return 1
    
    # ファイル構造チェック
    if not check_file_structure():
        print("\n⚠️  ファイル構造に問題があります")
        print("パターンAへの移行が完了していない可能性があります")
        response = input("\n続行しますか？ (y/N): ")
        if response.lower() != 'y':
            return 1
    
    # 遺伝的アルゴリズム削除確認
    check_genetic_removed()
    
    # メニュー表示
    print("\n📋 実行メニュー:")
    print("  1. サーバー起動")
    print("  2. テスト実行")
    print("  3. サーバー起動（テスト後）")
    print("  4. 終了")
    
    choice = input("\n選択してください (1-4): ").strip()
    
    if choice == '1':
        return 0 if run_server() else 1
    
    elif choice == '2':
        return 0 if run_tests() else 1
    
    elif choice == '3':
        if run_tests():
            print("\n✅ テスト成功 - サーバーを起動します")
            return 0 if run_server() else 1
        else:
            print("\n❌ テスト失敗 - サーバー起動を中止しました")
            return 1
    
    elif choice == '4':
        print("\n👋 終了しました")
        return 0
    
    else:
        print("\n❌ 無効な選択です")
        return 1


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n👋 中断されました")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        sys.exit(1)