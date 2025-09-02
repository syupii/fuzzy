#!/usr/bin/env python3
# project_cleanup.py
# -*- coding: utf-8 -*-
"""
プロジェクトファイル整理スクリプト
不要なファイルを削除し、重複を解消します
"""

import os
import sys
import shutil
import time
from pathlib import Path
from typing import List, Dict

# Windows文字エンコーディング設定
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

class ProjectCleaner:
    """プロジェクト整理クラス"""
    
    def __init__(self):
        self.deleted_files: List[str] = []
        self.moved_files: List[str] = []
        self.kept_files: List[str] = []
        
    def analyze_project(self):
        """プロジェクト分析"""
        print("=" * 60)
        print("[CLEANUP] プロジェクトファイル分析開始")
        print("=" * 60)
        
        # 不要ファイルの特定
        self.redundant_files = self._identify_redundant_files()
        self.backup_files = self._identify_backup_files()
        self.temporary_files = self._identify_temporary_files()
        self.duplicate_files = self._identify_duplicate_files()
        
        print(f"\n[ANALYSIS] 分析結果:")
        print(f"   重複ファイル: {len(self.redundant_files)}個")
        print(f"   バックアップファイル: {len(self.backup_files)}個")
        print(f"   一時ファイル: {len(self.temporary_files)}個")
        print(f"   複製ファイル: {len(self.duplicate_files)}個")
        
        return {
            'redundant': self.redundant_files,
            'backup': self.backup_files,
            'temporary': self.temporary_files,
            'duplicate': self.duplicate_files
        }
    
    def _identify_redundant_files(self) -> List[str]:
        """重複・不要ファイルの特定"""
        redundant = []
        
        # 重複する遺伝的アルゴリズム実装
        genetic_files = [
            'genetic_algorithm_complete_fix.py',  # パート分割版（不要）
            'run_genetic_optimization.py',        # 古い版（不要）
            'genetic_fuzzy_tree.py'              # 部分実装（不要）
        ]
        
        # 重複するエンジンファイル
        engine_files = [
            'fuzzy_engine_genetic_fix.py',       # 古い版（不要）
            'fuzzy_engine_windows_fix.py'        # Windows版が最新（保持）
        ]
        
        # 重複する実行スクリプト
        script_files = [
            'run_genetic_system.py',             # 古い版（不要）
            'run_genetic_windows.py'             # Windows版が最新（保持）
        ]
        
        # テスト・デバッグファイル（統合後不要）
        test_debug_files = [
            'test_genetic_integration.py',
            'model_debug_inspector.py',
            'model_compatibility_fix.py',
            'integrate_model_fixed.py',
            'advanced_genetic_fuzzy_tree.py',
            'train_genetic_model.py'
        ]
        
        # データベース関連（統合されたので古い版は不要）
        db_files = [
            'migrate_database.py',               # 統合済み
            'fresh_db_setup.py'                  # models.pyに統合済み
        ]
        
        all_candidates = genetic_files + engine_files + script_files + test_debug_files + db_files
        
        for file in all_candidates:
            if os.path.exists(file):
                redundant.append(file)
        
        return redundant
    
    def _identify_backup_files(self) -> List[str]:
        """バックアップファイルの特定"""
        backup = []
        
        patterns = [
            '*_backup.*',
            '*_old.*',
            '*_original.*',
            '*.bak',
            'temp_*.py'
        ]
        
        for pattern in patterns:
            for file in Path('.').glob(pattern):
                if file.is_file():
                    backup.append(str(file))
        
        # modelsディレクトリ内のバックアップ
        models_dir = Path('models')
        if models_dir.exists():
            for file in models_dir.glob('*backup*'):
                backup.append(str(file))
            for file in models_dir.glob('*old*'):
                backup.append(str(file))
        
        return backup
    
    def _identify_temporary_files(self) -> List[str]:
        """一時ファイルの特定"""
        temporary = []
        
        patterns = [
            '*.tmp',
            '*.log',
            '*_temp.py',
            'temp_*.py',
            '__pycache__/*',
            '*.pyc',
            '.pytest_cache/*'
        ]
        
        for pattern in patterns:
            for file in Path('.').rglob(pattern):
                if file.is_file():
                    temporary.append(str(file))
        
        return temporary
    
    def _identify_duplicate_files(self) -> List[Dict[str, List[str]]]:
        """重複ファイルの特定（内容ベース）"""
        duplicates = []
        
        # 重複の可能性がある拡張子
        target_extensions = ['.py', '.pkl', '.json']
        
        file_hashes = {}
        
        for ext in target_extensions:
            for file in Path('.').rglob(f'*{ext}'):
                if file.is_file() and file.stat().st_size > 0:
                    try:
                        with open(file, 'rb') as f:
                            import hashlib
                            file_hash = hashlib.md5(f.read()).hexdigest()
                            
                        if file_hash in file_hashes:
                            file_hashes[file_hash].append(str(file))
                        else:
                            file_hashes[file_hash] = [str(file)]
                    except:
                        continue
        
        # 重複グループの特定
        for file_hash, files in file_hashes.items():
            if len(files) > 1:
                duplicates.append({
                    'hash': file_hash,
                    'files': files,
                    'size': Path(files[0]).stat().st_size if Path(files[0]).exists() else 0
                })
        
        return duplicates
    
    def cleanup_files(self, analysis_result: Dict, interactive: bool = True):
        """ファイル整理実行"""
        
        print(f"\n[CLEANUP] ファイル整理開始")
        
        # 1. 一時ファイル削除（自動）
        if analysis_result['temporary']:
            print(f"\n[AUTO] 一時ファイル削除中...")
            for file in analysis_result['temporary']:
                if self._safe_delete(file):
                    self.deleted_files.append(file)
                    print(f"   削除: {file}")
        
        # 2. 重複ファイル処理
        if analysis_result['redundant']:
            print(f"\n[REDUNDANT] 重複ファイルの処理")
            
            if interactive:
                for file in analysis_result['redundant']:
                    if os.path.exists(file):
                        print(f"\n   ファイル: {file}")
                        print(f"   サイズ: {os.path.getsize(file)} bytes")
                        
                        choice = input("   削除しますか？ (y/N/s=スキップ): ").lower()
                        
                        if choice == 'y':
                            if self._safe_delete(file):
                                self.deleted_files.append(file)
                                print(f"   → 削除しました")
                        elif choice == 's':
                            continue
                        else:
                            self.kept_files.append(file)
                            print(f"   → 保持します")
            else:
                # 非対話モード：安全な削除のみ
                safe_to_delete = [
                    'temp_*.py',
                    '*_backup.*',
                    '*.tmp',
                    '*.log'
                ]
                
                for file in analysis_result['redundant']:
                    file_name = os.path.basename(file)
                    if any(self._matches_pattern(file_name, pattern) for pattern in safe_to_delete):
                        if self._safe_delete(file):
                            self.deleted_files.append(file)
                            print(f"   削除: {file}")
        
        # 3. バックアップファイル処理
        if analysis_result['backup']:
            print(f"\n[BACKUP] バックアップファイルの処理")
            
            # バックアップディレクトリ作成
            backup_dir = Path('archive/backups')
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            for file in analysis_result['backup']:
                if os.path.exists(file):
                    if interactive:
                        choice = input(f"   {file} を archive/ に移動しますか？ (y/N): ").lower()
                        if choice == 'y':
                            if self._move_to_archive(file, backup_dir):
                                self.moved_files.append(file)
                    else:
                        # 自動でバックアップファイルを移動
                        if self._move_to_archive(file, backup_dir):
                            self.moved_files.append(file)
        
        # 4. 重複ファイルグループ処理
        if analysis_result['duplicate']:
            print(f"\n[DUPLICATE] 重複ファイルグループの処理")
            
            for group in analysis_result['duplicate']:
                print(f"\n   重複グループ (サイズ: {group['size']} bytes):")
                for i, file in enumerate(group['files']):
                    print(f"     {i+1}. {file}")
                
                if interactive and len(group['files']) > 1:
                    keep_choice = input(f"   保持するファイル番号 (1-{len(group['files'])}, s=スキップ): ")
                    
                    if keep_choice.isdigit():
                        keep_index = int(keep_choice) - 1
                        if 0 <= keep_index < len(group['files']):
                            keep_file = group['files'][keep_index]
                            
                            for i, file in enumerate(group['files']):
                                if i != keep_index and os.path.exists(file):
                                    if self._safe_delete(file):
                                        self.deleted_files.append(file)
                                        print(f"     削除: {file}")
                            
                            self.kept_files.append(keep_file)
                            print(f"     保持: {keep_file}")
    
    def _safe_delete(self, filepath: str) -> bool:
        """安全なファイル削除"""
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                return True
        except Exception as e:
            print(f"     削除失敗 {filepath}: {e}")
        return False
    
    def _move_to_archive(self, filepath: str, archive_dir: Path) -> bool:
        """アーカイブディレクトリに移動"""
        try:
            if os.path.exists(filepath):
                filename = os.path.basename(filepath)
                archive_path = archive_dir / filename
                
                # 同名ファイルがある場合はタイムスタンプ追加
                if archive_path.exists():
                    name, ext = os.path.splitext(filename)
                    timestamp = int(time.time())
                    filename = f"{name}_{timestamp}{ext}"
                    archive_path = archive_dir / filename
                
                shutil.move(filepath, archive_path)
                print(f"     移動: {filepath} → {archive_path}")
                return True
        except Exception as e:
            print(f"     移動失敗 {filepath}: {e}")
        return False
    
    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """パターンマッチング"""
        import fnmatch
        return fnmatch.fnmatch(filename, pattern)
    
    def create_cleanup_summary(self):
        """整理サマリー作成"""
        
        print(f"\n" + "=" * 60)
        print(f"[SUMMARY] 整理完了サマリー")
        print(f"=" * 60)
        
        print(f"削除されたファイル: {len(self.deleted_files)}個")
        for file in self.deleted_files:
            print(f"  - {file}")
        
        print(f"\n移動されたファイル: {len(self.moved_files)}個")
        for file in self.moved_files:
            print(f"  - {file}")
        
        print(f"\n保持されたファイル: {len(self.kept_files)}個")
        
        # 推奨される次のステップ
        print(f"\n[RECOMMENDED] 推奨される最終ファイル構成:")
        print(f"  メインファイル:")
        print(f"    - genetic_algorithm_windows_fix.py    # 遺伝的アルゴリズム")
        print(f"    - fuzzy_engine_windows_fix.py         # ファジィエンジン")
        print(f"    - run_genetic_windows.py              # 実行スクリプト")
        print(f"    - fuzzy_decision_tree_learning.py     # 決定木学習")
        print(f"    - app.py                               # APIサーバー")
        print(f"    - models.py                            # データベースモデル")
        print(f"  ")
        print(f"  サポートファイル:")
        print(f"    - model_fix_script.py                 # モデル修復")
        print(f"    - test_suite.py                       # テストスイート")
        
        # 使用方法
        print(f"\n[USAGE] 使用方法:")
        print(f"  1. 遺伝的アルゴリズム実行: python run_genetic_windows.py")
        print(f"  2. ファジィ決定木学習: python fuzzy_decision_tree_learning.py --mode train")
        print(f"  3. APIサーバー起動: python app.py")

def main():
    """メイン実行"""
    
    print("プロジェクトファイル整理ツール")
    print("不要なファイルを削除し、プロジェクトを整理します")
    
    cleaner = ProjectCleaner()
    
    # プロジェクト分析
    analysis = cleaner.analyze_project()
    
    total_files = (len(analysis['redundant']) + 
                   len(analysis['backup']) + 
                   len(analysis['temporary']) + 
                   len(analysis['duplicate']))
    
    if total_files == 0:
        print(f"\n[CLEAN] プロジェクトは既に整理されています！")
        return 0
    
    print(f"\n[QUESTION] {total_files}個のファイルの整理が可能です。")
    
    # 実行モード選択
    print(f"\n整理モードを選択してください:")
    print(f"  1. 対話モード（推奨）- 各ファイルを確認しながら整理")
    print(f"  2. 自動モード - 安全なファイルのみ自動削除")
    print(f"  3. キャンセル")
    
    choice = input("\n選択 (1-3): ").strip()
    
    if choice == '1':
        cleaner.cleanup_files(analysis, interactive=True)
    elif choice == '2':
        cleaner.cleanup_files(analysis, interactive=False)
    elif choice == '3':
        print("[CANCELLED] 整理をキャンセルしました")
        return 0
    else:
        print("[ERROR] 無効な選択です")
        return 1
    
    # サマリー表示
    cleaner.create_cleanup_summary()
    
    print(f"\n[COMPLETE] プロジェクト整理完了！")
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    
    if exit_code == 0:
        print(f"\n整理が完了しました。プロジェクトがより整然としました！")
    else:
        print(f"\n整理中に問題が発生しました。")
    
    input("\nEnterキーを押して終了...")
    sys.exit(exit_code)