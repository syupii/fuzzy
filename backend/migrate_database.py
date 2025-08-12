# backend/migrate_database.py
"""
🗄️ Database Migration System
データベースマイグレーションシステム

既存のシンプルシステムから拡張システムへの安全な移行を提供
"""

import os
import sys
import sqlite3
import json
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional

# プロジェクトパス追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class DatabaseMigrator:
    """データベースマイグレーター"""

    def __init__(self, db_path: str = "fdtlss.db"):
        self.db_path = db_path
        self.backup_path = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.migration_log = []

    def check_current_schema(self) -> Dict[str, Any]:
        """現在のスキーマチェック"""

        schema_info = {
            'existing_tables': [],
            'table_structures': {},
            'data_counts': {},
            'version': 'unknown'
        }

        if not os.path.exists(self.db_path):
            schema_info['version'] = 'none'
            return schema_info

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # テーブル一覧取得
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            schema_info['existing_tables'] = tables

            # 各テーブル構造取得
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                schema_info['table_structures'][table] = [
                    {'name': col[1], 'type': col[2],
                        'nullable': not col[3], 'primary_key': bool(col[5])}
                    for col in columns
                ]

                # データ数取得
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                schema_info['data_counts'][table] = count

            # バージョン判定
            if 'genetic_individuals' in tables:
                schema_info['version'] = 'extended'
            elif 'evaluations' in tables and 'labs' in tables:
                schema_info['version'] = 'simple'
            else:
                schema_info['version'] = 'unknown'

            conn.close()

        except Exception as e:
            print(f"⚠️ Schema check error: {e}")
            schema_info['error'] = str(e)

        return schema_info

    def create_backup(self) -> bool:
        """データベースバックアップ作成"""

        if not os.path.exists(self.db_path):
            print("📝 No existing database to backup")
            return True

        try:
            print(f"💾 Creating database backup: {self.backup_path}")
            shutil.copy2(self.db_path, self.backup_path)

            # バックアップ検証
            if os.path.exists(self.backup_path):
                backup_size = os.path.getsize(self.backup_path)
                original_size = os.path.getsize(self.db_path)

                if backup_size == original_size:
                    print(
                        f"✅ Backup created successfully ({backup_size} bytes)")
                    self.migration_log.append(
                        f"Backup created: {self.backup_path}")
                    return True
                else:
                    print(
                        f"❌ Backup size mismatch: {backup_size} vs {original_size}")
                    return False
            else:
                print("❌ Backup file not created")
                return False

        except Exception as e:
            print(f"❌ Backup creation failed: {e}")
            return False

    def migrate_simple_to_extended(self) -> bool:
        """シンプル版から拡張版への移行"""

        print("🔄 Migrating from simple to extended schema...")

        try:
            # 既存データ保存
            existing_data = self._extract_existing_data()

            # 新スキーマ作成
            from models import init_extended_database
            success = init_extended_database()

            if not success:
                print("❌ Failed to create extended schema")
                return False

            # データ復元
            restored_count = self._restore_existing_data(existing_data)

            print(f"✅ Migration completed: {restored_count} records migrated")
            self.migration_log.append(
                f"Migrated {restored_count} records to extended schema")

            return True

        except Exception as e:
            print(f"❌ Migration failed: {e}")
            self.migration_log.append(f"Migration error: {e}")
            return False

    def _extract_existing_data(self) -> Dict[str, List[Dict]]:
        """既存データ抽出"""

        data = {
            'labs': [],
            'evaluations': []
        }

        if not os.path.exists(self.db_path):
            return data

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row  # 辞書アクセス可能
            cursor = conn.cursor()

            # 研究室データ
            cursor.execute("SELECT * FROM labs")
            for row in cursor.fetchall():
                data['labs'].append(dict(row))

            # 評価データ
            cursor.execute("SELECT * FROM evaluations")
            for row in cursor.fetchall():
                data['evaluations'].append(dict(row))

            conn.close()

            print(
                f"📊 Extracted {len(data['labs'])} labs, {len(data['evaluations'])} evaluations")

        except Exception as e:
            print(f"⚠️ Data extraction error: {e}")

        return data

    def _restore_existing_data(self, data: Dict[str, List[Dict]]) -> int:
        """既存データ復元"""

        restored_count = 0

        try:
            from models import create_app, db, Lab, Evaluation

            app = create_app()
            with app.app_context():
                # 研究室データ復元
                for lab_data in data['labs']:
                    # 既存チェック
                    existing_lab = Lab.query.filter_by(
                        name=lab_data['name']).first()
                    if not existing_lab:
                        lab = Lab(**lab_data)
                        db.session.add(lab)
                        restored_count += 1

                # 評価データ復元
                for eval_data in data['evaluations']:
                    # IDを除いて新規作成
                    eval_data_clean = {k: v for k,
                                       v in eval_data.items() if k != 'id'}
                    evaluation = Evaluation(**eval_data_clean)
                    db.session.add(evaluation)
                    restored_count += 1

                db.session.commit()

        except Exception as e:
            print(f"⚠️ Data restoration error: {e}")

        return restored_count

    def add_missing_columns(self) -> bool:
        """不足カラムの追加"""

        print("🔧 Adding missing columns...")

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # 必要なカラム定義
            column_additions = [
                # labs テーブル拡張
                {
                    'table': 'labs',
                    'columns': [
                        ('created_at', 'DATETIME', datetime.utcnow().isoformat()),
                        ('updated_at', 'DATETIME', datetime.utcnow().isoformat()),
                        ('is_active', 'BOOLEAN', 1)
                    ]
                },
                # evaluations テーブル拡張
                {
                    'table': 'evaluations',
                    'columns': [
                        ('created_at', 'DATETIME', datetime.utcnow().isoformat())
                    ]
                }
            ]

            added_columns = 0

            for addition in column_additions:
                table = addition['table']

                # テーブル存在チェック
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
                if not cursor.fetchone():
                    continue

                # 既存カラム取得
                cursor.execute(f"PRAGMA table_info({table})")
                existing_columns = {col[1] for col in cursor.fetchall()}

                # カラム追加
                for col_name, col_type, default_value in addition['columns']:
                    if col_name not in existing_columns:
                        try:
                            if default_value is not None:
                                cursor.execute(
                                    f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type} DEFAULT ?", (default_value,))
                            else:
                                cursor.execute(
                                    f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")

                            added_columns += 1
                            print(f"   ✅ Added {table}.{col_name}")

                        except sqlite3.Error as e:
                            print(
                                f"   ⚠️ Failed to add {table}.{col_name}: {e}")

            conn.commit()
            conn.close()

            if added_columns > 0:
                print(f"✅ Added {added_columns} missing columns")
                self.migration_log.append(f"Added {added_columns} columns")
            else:
                print("📝 No missing columns to add")

            return True

        except Exception as e:
            print(f"❌ Column addition failed: {e}")
            return False

    def create_new_tables(self) -> bool:
        """新テーブル作成"""

        print("🆕 Creating new tables...")

        try:
            from models import create_app, db

            app = create_app()
            with app.app_context():
                # 新テーブル作成（既存テーブルは保持）
                db.create_all()

                print("✅ New tables created successfully")
                self.migration_log.append("Created new tables")
                return True

        except Exception as e:
            print(f"❌ New table creation failed: {e}")
            return False

    def migrate_to_latest(self) -> bool:
        """最新版への移行"""

        print("🚀 Starting migration to latest version...")
        print("=" * 50)

        # 現在のスキーマチェック
        schema_info = self.check_current_schema()
        current_version = schema_info['version']

        print(f"📊 Current database version: {current_version}")
        print(f"📊 Existing tables: {schema_info['existing_tables']}")

        if current_version == 'extended':
            print("✅ Database is already at latest version")
            return True

        # バックアップ作成
        if not self.create_backup():
            print("❌ Cannot proceed without backup")
            return False

        try:
            # バージョン別移行
            if current_version == 'none':
                # 新規インストール
                print("🆕 New installation - creating fresh database")
                from models import init_extended_database
                success = init_extended_database()

            elif current_version == 'simple':
                # シンプル→拡張移行
                success = self.migrate_simple_to_extended()

            elif current_version == 'unknown':
                # 不明→拡張移行（慎重に）
                print("⚠️ Unknown schema detected - performing cautious migration")
                success = self.add_missing_columns() and self.create_new_tables()

            else:
                print(f"❌ Unsupported version: {current_version}")
                return False

            if success:
                # 移行後検証
                final_schema = self.check_current_schema()

                if final_schema['version'] == 'extended':
                    print("✅ Migration completed successfully")
                    self._log_migration_summary(schema_info, final_schema)
                    return True
                else:
                    print("❌ Migration verification failed")
                    return False
            else:
                print("❌ Migration failed")
                return False

        except Exception as e:
            print(f"❌ Migration error: {e}")
            print("🔄 Attempting to restore from backup...")
            self._restore_from_backup()
            return False

    def _restore_from_backup(self) -> bool:
        """バックアップからの復元"""

        try:
            if os.path.exists(self.backup_path):
                shutil.copy2(self.backup_path, self.db_path)
                print(f"✅ Database restored from backup: {self.backup_path}")
                return True
            else:
                print("❌ Backup file not found")
                return False

        except Exception as e:
            print(f"❌ Restore failed: {e}")
            return False

    def _log_migration_summary(self, before: Dict, after: Dict):
        """移行サマリーログ"""

        print("\n📋 Migration Summary:")
        print(f"   Version: {before['version']} → {after['version']}")
        print(f"   Tables before: {len(before['existing_tables'])}")
        print(f"   Tables after: {len(after['existing_tables'])}")

        # 新テーブル
        new_tables = set(after['existing_tables']) - \
            set(before['existing_tables'])
        if new_tables:
            print(f"   New tables: {', '.join(new_tables)}")

        # データ保持確認
        for table in before['existing_tables']:
            if table in after['data_counts']:
                before_count = before['data_counts'].get(table, 0)
                after_count = after['data_counts'][table]
                print(f"   {table}: {before_count} → {after_count} records")

        # ログファイル保存
        log_file = f"migration_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'before_schema': before,
            'after_schema': after,
            'migration_log': self.migration_log,
            'backup_path': self.backup_path
        }

        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2,
                          ensure_ascii=False, default=str)
            print(f"📄 Migration log saved: {log_file}")
        except Exception as e:
            print(f"⚠️ Failed to save migration log: {e}")


def main():
    """メイン実行関数"""

    import argparse

    parser = argparse.ArgumentParser(
        description="FDTLSS Database Migration Tool")
    parser.add_argument('--db-path', default='fdtlss.db',
                        help='Database file path')
    parser.add_argument('--check-only', action='store_true',
                        help='Check schema only')
    parser.add_argument('--force', action='store_true',
                        help='Force migration without confirmation')

    args = parser.parse_args()

    print("🗄️ FDTLSS Database Migration Tool")
    print("=" * 40)

    migrator = DatabaseMigrator(args.db_path)

    # スキーマチェック
    schema_info = migrator.check_current_schema()

    print(f"📊 Database file: {args.db_path}")
    print(f"📊 Current version: {schema_info['version']}")
    print(f"📊 Tables: {schema_info['existing_tables']}")

    if schema_info['data_counts']:
        print("📊 Data counts:")
        for table, count in schema_info['data_counts'].items():
            print(f"   {table}: {count} records")

    if args.check_only:
        print("✅ Schema check completed")
        return 0

    # 移行が必要かチェック
    if schema_info['version'] == 'extended':
        print("✅ Database is already at latest version")
        return 0

    # 確認
    if not args.force:
        print(
            f"\n⚠️ This will migrate your database from '{schema_info['version']}' to 'extended' version")
        print("   A backup will be created automatically")
        response = input("   Continue? (y/N): ")

        if response.lower() != 'y':
            print("Migration cancelled")
            return 0

    # 移行実行
    success = migrator.migrate_to_latest()

    if success:
        print("\n🎉 Migration completed successfully!")
        print(f"💾 Backup saved as: {migrator.backup_path}")
        return 0
    else:
        print("\n❌ Migration failed!")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
