#!/usr/bin/env python3
"""
簡単実行スクリプト - run.py
依存関係チェックと基本実行
"""

import sys
import subprocess
import importlib

def check_dependencies():
    """必要なパッケージのチェック"""
    
    required_packages = [
        'numpy',
        'pandas', 
        'scipy',
        'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} (未インストール)")
    
    if missing_packages:
        print(f"\n⚠️ 以下のパッケージをインストールしてください:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def run_quick_demo():
    """クイックデモ実行"""
    
    print("🚀 クイックデモ実行中...")
    
    try:
        # main.pyを実行
        result = subprocess.run([
            sys.executable, 'main.py', '--mode', 'demo'
        ], capture_output=True, text=True, encoding='utf-8')
        
        print("標準出力:")
        print(result.stdout)
        
        if result.stderr:
            print("エラー出力:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        return False

def main():
    """メイン関数"""
    
    print("🧬🌳 遺伝的ファジィ決定木システム - クイック起動")
    print("=" * 50)
    
    # 依存関係チェック
    print("📦 依存関係チェック中...")
    
    if not check_dependencies():
        print("\n❌ 依存関係に問題があります")
        return 1
    
    print("\n✅ 依存関係OK")
    
    # デモ実行
    print("\n🎯 デモ実行...")
    
    success = run_quick_demo()
    
    if success:
        print("\n✅ デモ実行完了!")
        print("\n📚 その他の実行モード:")
        print("  python main.py --mode train    # 訓練モード")
        print("  python main.py --mode predict  # 予測モード") 
        print("  python main.py --mode full     # フルモード")
        return 0
    else:
        print("\n❌ デモ実行失敗")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)