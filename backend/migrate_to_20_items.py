# backend/migrate_to_20_items.py
# 既存のデータベースを20項目対応に拡張するマイグレーションスクリプト

import sqlite3
import os
import shutil
from datetime import datetime

def backup_database():
    """データベースのバックアップ作成"""
    db_file = 'fdtlss.db'
    if os.path.exists(db_file):
        backup_file = f'fdtlss_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db'
        shutil.copy2(db_file, backup_file)
        print(f"📦 バックアップ作成: {backup_file}")
        return True
    else:
        print("📝 新規データベースのためバックアップをスキップ")
        return False

def check_current_schema():
    """現在のデータベーススキーマを確認"""
    try:
        conn = sqlite3.connect('fdtlss.db')
        cursor = conn.cursor()
        
        # テーブル一覧取得
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema_info = {'tables': tables, 'version': 'unknown'}
        
        # labsテーブルのカラム確認
        if 'labs' in tables:
            cursor.execute("PRAGMA table_info(labs)")
            columns = cursor.fetchall()
            lab_columns = [col[1] for col in columns]
            
            schema_info['lab_columns'] = lab_columns
            schema_info['lab_column_count'] = len(lab_columns)
            
            # バージョン判定
            if 'core_time_flexibility' in lab_columns:
                schema_info['version'] = 'extended'
            elif 'research_field_match' in lab_columns:
                schema_info['version'] = 'partial'
            elif len(lab_columns) <= 10:
                schema_info['version'] = 'basic'
        
        conn.close()
        return schema_info
        
    except Exception as e:
        print(f"⚠️ スキーマ確認エラー: {e}")
        return None

def add_new_columns_to_labs():
    """Labsテーブルに新しい15項目を追加"""
    
    # 追加する新しいカラム（15項目）
    new_columns = [
        # 分野適合性（元からの重要項目）
        ('research_field_match', 'REAL DEFAULT 5.0'),
        
        # 学習・成長関連（3項目）
        ('skill_development', 'REAL DEFAULT 5.0'),
        ('learning_pace', 'REAL DEFAULT 5.0'),
        ('difficulty_preference', 'REAL DEFAULT 5.0'),
        
        # コミュニケーション・環境関連（3項目）
        ('communication_style', 'REAL DEFAULT 5.0'),
        ('meeting_frequency', 'REAL DEFAULT 5.0'),
        ('lab_atmosphere', 'REAL DEFAULT 5.0'),
        
        # 研究アプローチ関連（3項目）
        ('innovation_risk', 'REAL DEFAULT 5.0'),
        ('methodology_preference', 'REAL DEFAULT 5.0'),
        ('interdisciplinary', 'REAL DEFAULT 5.0'),
        
        # 時間・ライフスタイル関連（2項目）
        ('flexibility', 'REAL DEFAULT 5.0'),
        ('evening_weekend_work', 'REAL DEFAULT 5.0'),
        
        # 調査結果に基づく追加項目（3項目）
        ('publication_opportunity', 'REAL DEFAULT 5.0'),
        ('financial_support', 'REAL DEFAULT 5.0'),
        ('lab_hierarchy', 'REAL DEFAULT 5.0'),
        ('core_time_flexibility', 'REAL DEFAULT 5.0'),
    ]
    
    try:
        conn = sqlite3.connect('fdtlss.db')
        cursor = conn.cursor()
        
        # 既存のカラムチェック
        cursor.execute("PRAGMA table_info(labs)")
        existing_columns = {col[1] for col in cursor.fetchall()}
        
        print(f"📊 現在のLabsカラム数: {len(existing_columns)}")
        
        added_count = 0
        
        for col_name, col_definition in new_columns:
            if col_name not in existing_columns:
                try:
                    cursor.execute(f"ALTER TABLE labs ADD COLUMN {col_name} {col_definition}")
                    print(f"  ✅ 追加: {col_name}")
                    added_count += 1
                except sqlite3.Error as e:
                    print(f"  ❌ エラー {col_name}: {e}")
            else:
                print(f"  📝 既存: {col_name}")
        
        conn.commit()
        
        # 最終確認
        cursor.execute("PRAGMA table_info(labs)")
        final_columns = cursor.fetchall()
        print(f"\n📊 最終カラム数: {len(final_columns)}")
        
        conn.close()
        
        print(f"🎉 {added_count}個の新しいカラムを追加しました")
        return True
        
    except Exception as e:
        print(f"❌ カラム追加エラー: {e}")
        return False

def add_new_columns_to_evaluations():
    """Evaluationテーブルに20項目のカラム追加"""
    
    eval_columns = [
        # 新しい15項目
        ('research_field_match', 'REAL'),
        ('skill_development', 'REAL'),
        ('learning_pace', 'REAL'),
        ('difficulty_preference', 'REAL'),
        ('communication_style', 'REAL'),
        ('meeting_frequency', 'REAL'),
        ('lab_atmosphere', 'REAL'),
        ('innovation_risk', 'REAL'),
        ('methodology_preference', 'REAL'),
        ('interdisciplinary', 'REAL'),
        ('flexibility', 'REAL'),
        ('evening_weekend_work', 'REAL'),
        ('publication_opportunity', 'REAL'),
        ('financial_support', 'REAL'),
        ('lab_hierarchy', 'REAL'),
        ('core_time_flexibility', 'REAL'),
        
        # メタデータ追加
        ('user_preferences', 'TEXT'),
        ('evaluation_count', 'INTEGER'),
        ('avg_score', 'REAL'),
        ('best_lab_id', 'INTEGER'),
        ('engine_used', 'TEXT'),
    ]
    
    try:
        conn = sqlite3.connect('fdtlss.db')
        cursor = conn.cursor()
        
        # 既存のカラムチェック
        cursor.execute("PRAGMA table_info(evaluations)")
        existing_columns = {col[1] for col in cursor.fetchall()}
        
        added_count = 0
        
        for col_name, col_definition in eval_columns:
            if col_name not in existing_columns:
                try:
                    cursor.execute(f"ALTER TABLE evaluations ADD COLUMN {col_name} {col_definition}")
                    print(f"  ✅ 評価テーブルに追加: {col_name}")
                    added_count += 1
                except sqlite3.Error as e:
                    print(f"  ❌ エラー {col_name}: {e}")
        
        conn.commit()
        conn.close()
        
        print(f"🎉 評価テーブルに{added_count}個のカラムを追加しました")
        return True
        
    except Exception as e:
        print(f"❌ 評価テーブルカラム追加エラー: {e}")
        return False

def migrate_existing_lab_data():
    """既存の研究室データを20項目対応に移行"""
    try:
        conn = sqlite3.connect('fdtlss.db')
        cursor = conn.cursor()
        
        # 既存の研究室データ取得
        cursor.execute("SELECT id, name, research_intensity, advisor_style, team_work, workload, theory_practice FROM labs")
        existing_labs = cursor.fetchall()
        
        print(f"📊 既存研究室データ: {len(existing_labs)}件")
        
        if len(existing_labs) == 0:
            print("📝 既存データがないため、移行をスキップします")
            conn.close()
            return True
        
        # 各研究室に新しい15項目のランダム値を設定
        new_columns = [
            'research_field_match', 'skill_development', 'learning_pace', 'difficulty_preference',
            'communication_style', 'meeting_frequency', 'lab_atmosphere', 'innovation_risk',
            'methodology_preference', 'interdisciplinary', 'flexibility', 'evening_weekend_work',
            'publication_opportunity', 'financial_support', 'lab_hierarchy', 'core_time_flexibility'
        ]
        
        for lab_data in existing_labs:
            lab_id, name, intensity, advisor, team, workload, theory = lab_data
            
            # 既存の5項目の特性に基づいて新しい項目の値を推定
            base_score = (intensity + advisor + team + workload + theory) / 5.0
            
            updates = {}
            for column in new_columns:
                if column in ['publication_opportunity', 'financial_support']:
                    # 重要項目: 基本スコア + ランダム調整
                    updates[column] = round(min(10.0, max(1.0, base_score + random.uniform(-1, 2))), 1)
                elif column == 'evening_weekend_work':
                    # 時間外作業: 研究強度に関連
                    updates[column] = round(min(10.0, max(1.0, intensity * 0.7 + random.uniform(-1, 1))), 1)
                elif column in ['communication_style', 'lab_atmosphere']:
                    # コミュニケーション: チームワークに関連
                    updates[column] = round(min(10.0, max(1.0, team + random.uniform(-1.5, 1.5))), 1)
                else:
                    # その他: 基本スコア周辺
                    updates[column] = round(min(10.0, max(1.0, base_score + random.uniform(-2, 2))), 1)
            
            # 更新実行
            for column, value in updates.items():
                cursor.execute(f"UPDATE labs SET {column} = ? WHERE id = ?", (value, lab_id))
            
            print(f"  ✅ {name} を20項目対応に更新")
        
        conn.commit()
        conn.close()
        
        print(f"🎉 {len(existing_labs)}件の研究室データを20項目対応に移行完了")
        return True
        
    except Exception as e:
        print(f"❌ データ移行エラー: {e}")
        return False

def verify_migration():
    """マイグレーション結果の検証"""
    try:
        conn = sqlite3.connect('fdtlss.db')
        cursor = conn.cursor()
        
        # Labsテーブル検証
        cursor.execute("PRAGMA table_info(labs)")
        lab_columns = cursor.fetchall()
        
        print(f"📊 Labsテーブル最終カラム数: {len(lab_columns)}")
        
        # 20項目がすべて存在するかチェック
        required_columns = [
            'research_intensity', 'advisor_style', 'team_work', 'workload', 'theory_practice',
            'research_field_match', 'skill_development', 'learning_pace', 'difficulty_preference',
            'communication_style', 'meeting_frequency', 'lab_atmosphere', 'innovation_risk',
            'methodology_preference', 'interdisciplinary', 'flexibility', 'evening_weekend_work',
            'publication_opportunity', 'financial_support', 'lab_hierarchy', 'core_time_flexibility'
        ]
        
        existing_column_names = [col[1] for col in lab_columns]
        missing_columns = [col for col in required_columns if col not in existing_column_names]
        
        if missing_columns:
            print(f"⚠️ 不足カラム: {missing_columns}")
            return False
        else:
            print("✅ 20項目すべてのカラムが存在します")
        
        # サンプルデータ確認
        cursor.execute("SELECT COUNT(*) FROM labs")
        lab_count = cursor.fetchone()[0]
        print(f"📊 研究室データ数: {lab_count}")
        
        if lab_count > 0:
            # サンプル研究室の20項目データ確認
            cursor.execute(f"SELECT name, {', '.join(required_columns[:5])} FROM labs LIMIT 1")
            sample = cursor.fetchone()
            if sample:
                print(f"📋 サンプル: {sample[0]} - 基本5項目: {sample[1:]}")
        
        conn.close()
        
        print("✅ マイグレーション検証完了")
        return True
        
    except Exception as e:
        print(f"❌ 検証エラー: {e}")
        return False

def create_new_tables():
    """新しいテーブル（genetic_individuals等）の作成"""
    
    sql_commands = [
        # genetic_individuals テーブル
        """
        CREATE TABLE IF NOT EXISTS genetic_individuals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            individual_id TEXT NOT NULL UNIQUE,
            generation INTEGER NOT NULL,
            genome_data TEXT,
            accuracy REAL,
            simplicity REAL,
            interpretability REAL,
            generalization REAL,
            validity REAL,
            overall_fitness REAL,
            parent1_id TEXT,
            parent2_id TEXT,
            model_complexity INTEGER,
            tree_depth INTEGER,
            evaluation_time REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """,
        
        # decision_paths テーブル
        """
        CREATE TABLE IF NOT EXISTS decision_paths (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluation_id INTEGER,
            step_order INTEGER,
            criterion TEXT,
            threshold REAL,
            user_value REAL,
            lab_value REAL,
            decision_result TEXT,
            criterion_weight REAL,
            confidence REAL,
            rule_explanation TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (evaluation_id) REFERENCES evaluations (id)
        )
        """,
        
        # optimization_runs テーブル
        """
        CREATE TABLE IF NOT EXISTS optimization_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL UNIQUE,
            population_size INTEGER,
            generations INTEGER,
            mutation_rate REAL,
            crossover_rate REAL,
            max_depth INTEGER,
            tournament_size INTEGER,
            training_samples INTEGER,
            test_samples INTEGER,
            feature_names TEXT,
            target_column TEXT,
            best_fitness REAL,
            best_individual_id TEXT,
            convergence_generation INTEGER,
            final_diversity REAL,
            fitness_history TEXT,
            diversity_history TEXT,
            execution_time REAL,
            status TEXT DEFAULT 'running',
            description TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            completed_at DATETIME
        )
        """,
        
        # model_registry テーブル
        """
        CREATE TABLE IF NOT EXISTS model_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL UNIQUE,
            model_name TEXT,
            model_type TEXT,
            version TEXT,
            model_filepath TEXT,
            result_filepath TEXT,
            file_size_bytes INTEGER,
            checksum TEXT,
            best_fitness REAL,
            model_complexity INTEGER,
            validation_accuracy REAL,
            test_accuracy REAL,
            usage_count INTEGER DEFAULT 0,
            last_used_at DATETIME,
            is_active BOOLEAN DEFAULT 1,
            is_production_ready BOOLEAN DEFAULT 0,
            description TEXT,
            tags TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """,
        
        # system_config テーブル
        """
        CREATE TABLE IF NOT EXISTS system_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_key TEXT NOT NULL UNIQUE,
            config_value TEXT,
            config_type TEXT DEFAULT 'string',
            description TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    ]
    
    try:
        conn = sqlite3.connect('fdtlss.db')
        cursor = conn.cursor()
        
        for sql in sql_commands:
            cursor.execute(sql)
            print(f"  ✅ テーブル作成実行")
        
        conn.commit()
        conn.close()
        
        print("✅ 新テーブル作成完了")
        return True
        
    except Exception as e:
        print(f"❌ テーブル作成エラー: {e}")
        return False

def update_existing_lab_data():
    """既存の研究室データに20項目のランダム値を設定"""
    
    try:
        conn = sqlite3.connect('fdtlss.db')
        cursor = conn.cursor()
        
        # 既存の研究室取得
        cursor.execute("SELECT id, name, research_intensity, advisor_style, team_work, workload, theory_practice FROM labs")
        existing_labs = cursor.fetchall()
        
        if len(existing_labs) == 0:
            print("📝 既存研究室データがありません")
            conn.close()
            return True
        
        print(f"📊 既存研究室データ: {len(existing_labs)}件")
        
        # 新しい15項目のカラム
        new_columns = [
            'research_field_match', 'skill_development', 'learning_pace', 'difficulty_preference',
            'communication_style', 'meeting_frequency', 'lab_atmosphere', 'innovation_risk',
            'methodology_preference', 'interdisciplinary', 'flexibility', 'evening_weekend_work',
            'publication_opportunity', 'financial_support', 'lab_hierarchy', 'core_time_flexibility'
        ]
        
        for lab_data in existing_labs:
            lab_id, name, intensity, advisor, team, workload, theory = lab_data
            
            # 既存の5項目の平均を基準値とする
            base_score = (intensity + advisor + team + workload + theory) / 5.0
            
            # 各新項目に特性に応じた値を設定
            updates = {}
            
            for column in new_columns:
                if column in ['publication_opportunity', 'financial_support']:
                    # 重要項目: やや高めに設定
                    updates[column] = round(min(10.0, max(1.0, base_score + random.uniform(0, 2))), 1)
                elif column == 'evening_weekend_work':
                    # 時間外作業: 研究強度と相関
                    updates[column] = round(min(10.0, max(1.0, intensity * 0.8 + random.uniform(-1, 1))), 1)
                elif column in ['communication_style', 'lab_atmosphere']:
                    # コミュニケーション: チームワークと相関
                    updates[column] = round(min(10.0, max(1.0, team + random.uniform(-1, 1))), 1)
                elif column == 'flexibility':
                    # 柔軟性: 指導スタイルと相関
                    updates[column] = round(min(10.0, max(1.0, advisor + random.uniform(-1, 1))), 1)
                else:
                    # その他: 基準値周辺
                    updates[column] = round(min(10.0, max(1.0, base_score + random.uniform(-1.5, 1.5))), 1)
            
            # 更新実行
            for column, value in updates.items():
                cursor.execute(f"UPDATE labs SET {column} = ? WHERE id = ?", (value, lab_id))
            
            print(f"  ✅ {name} を20項目対応に更新")
        
        conn.commit()
        conn.close()
        
        print(f"🎉 {len(existing_labs)}件の研究室データを20項目対応に移行完了")
        return True
        
    except Exception as e:
        print(f"❌ データ移行エラー: {e}")
        return False

def main():
    """メインマイグレーション実行"""
    print("🚀 20項目対応データベースマイグレーション開始")
    print("=" * 60)
    
    # 1. 現在の状態確認
    print("1️⃣ 現在のデータベース状態確認...")
    schema_info = check_current_schema()
    if schema_info:
        print(f"📊 データベースバージョン: {schema_info['version']}")
        print(f"📊 既存テーブル: {schema_info['tables']}")
        if 'lab_column_count' in schema_info:
            print(f"📊 Labテーブルカラム数: {schema_info['lab_column_count']}")
    
    # バージョンチェック
    if schema_info and schema_info['version'] == 'extended':
        print("✅ データベースは既に20項目対応です")
        return
    
    # 2. バックアップ作成
    print("\n2️⃣ バックアップ作成...")
    backup_database()
    
    # 3. 新テーブル作成
    print("\n3️⃣ 新テーブル作成...")
    if not create_new_tables():
        print("❌ 新テーブル作成に失敗しました")
        return
    
    # 4. Labテーブルにカラム追加
    print("\n4️⃣ Labテーブルを20項目対応に拡張...")
    if not add_new_columns_to_labs():
        print("❌ Labテーブル拡張に失敗しました")
        return
    
    # 5. Evaluationテーブルにカラム追加
    print("\n5️⃣ Evaluationテーブルを20項目対応に拡張...")
    if not add_new_columns_to_evaluations():
        print("❌ Evaluationテーブル拡張に失敗しました")
        return
    
    # 6. 既存データ移行
    print("\n6️⃣ 既存データを20項目対応に移行...")
    if not update_existing_lab_data():
        print("❌ データ移行に失敗しました")
        return
    
    # 7. 最終検証
    print("\n7️⃣ マイグレーション結果検証...")
    if verify_migration():
        print("✅ マイグレーション検証成功")
    else:
        print("⚠️ マイグレーション検証で問題が発見されました")
    
    print("\n🎉 20項目対応マイグレーション完了！")
    print("\n次のステップ:")
    print("  1. python create_sample_labs.py (新しい研究室データを作成する場合)")
    print("  2. python app.py (サーバー起動)")
    print("  3. ブラウザでhttp://localhost:3000 (フロントエンドテスト)")


if __name__ == '__main__':
    main()