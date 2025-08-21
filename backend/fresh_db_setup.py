# backend/fresh_db_setup.py
# 20項目対応の完全新規データベースを作成

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from models import db, Lab, SystemConfig
import random

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fdtlss.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    return app

def create_20_item_sample_labs():
    """20項目対応のサンプル研究室データ作成"""
    
    # 20項目すべてを含む研究室データ
    labs_data = [
        {
            'name': 'AI・機械学習研究室',
            'professor': '田中教授',
            'research_area': '人工知能、機械学習、深層学習、自然言語処理',
            'description': '最先端のAI技術を使った応用研究を行っています。学生の自主性を重視し、国際学会での発表を積極的に支援しています。GPU クラスターを完備。',
            'features': {
                # 基本的な研究環境（6項目）
                'research_intensity': 8.5, 'advisor_style': 6.0, 'team_work': 7.5, 
                'workload': 7.5, 'theory_practice': 7.0, 'research_field_match': 9.0,
                # 学習・成長関連（3項目）
                'skill_development': 8.0, 'learning_pace': 7.5, 'difficulty_preference': 8.5,
                # コミュニケーション・環境関連（3項目）
                'communication_style': 7.0, 'meeting_frequency': 7.0, 'lab_atmosphere': 8.0,
                # 研究アプローチ関連（3項目）
                'innovation_risk': 8.0, 'methodology_preference': 8.5, 'interdisciplinary': 7.5,
                # 時間・ライフスタイル関連（2項目）
                'flexibility': 6.5, 'evening_weekend_work': 7.0,
                # 調査結果に基づく追加項目（3項目）
                'publication_opportunity': 8.5, 'financial_support': 8.0, 'lab_hierarchy': 6.0, 'core_time_flexibility': 6.5
            }
        },
        {
            'name': 'データサイエンス研究室',
            'professor': '佐藤教授',
            'research_area': 'ビッグデータ解析、統計的機械学習、ビジネスアナリティクス',
            'description': 'データ駆動型の問題解決に取り組んでいます。企業との共同研究が多く、実践的なスキルを身につけられます。',
            'features': {
                'research_intensity': 7.5, 'advisor_style': 7.0, 'team_work': 8.0,
                'workload': 6.5, 'theory_practice': 6.5, 'research_field_match': 8.5,
                'skill_development': 7.5, 'learning_pace': 6.5, 'difficulty_preference': 7.0,
                'communication_style': 8.0, 'meeting_frequency': 6.5, 'lab_atmosphere': 7.5,
                'innovation_risk': 6.5, 'methodology_preference': 7.0, 'interdisciplinary': 8.5,
                'flexibility': 7.5, 'evening_weekend_work': 5.5,
                'publication_opportunity': 7.5, 'financial_support': 7.0, 'lab_hierarchy': 7.0, 'core_time_flexibility': 7.5
            }
        },
        {
            'name': 'ソフトウェア工学研究室',
            'professor': '山田教授',
            'research_area': 'ソフトウェア設計、開発方法論、品質管理',
            'description': '高品質なソフトウェアを効率的に開発する手法を研究します。学生同士の協力を重視した環境です。',
            'features': {
                'research_intensity': 7.0, 'advisor_style': 6.5, 'team_work': 8.5,
                'workload': 6.0, 'theory_practice': 8.0, 'research_field_match': 7.5,
                'skill_development': 8.5, 'learning_pace': 7.0, 'difficulty_preference': 6.5,
                'communication_style': 8.5, 'meeting_frequency': 7.5, 'lab_atmosphere': 8.0,
                'innovation_risk': 6.0, 'methodology_preference': 7.5, 'interdisciplinary': 7.0,
                'flexibility': 8.0, 'evening_weekend_work': 5.0,
                'publication_opportunity': 6.5, 'financial_support': 6.0, 'lab_hierarchy': 7.5, 'core_time_flexibility': 8.0
            }
        },
        {
            'name': 'ロボティクス研究室',
            'professor': '鈴木教授',
            'research_area': 'ロボット工学、制御システム、自律移動ロボット',
            'description': '自律移動ロボットや産業用ロボットの制御技術を研究します。実機での実験が多いのが特徴です。',
            'features': {
                'research_intensity': 8.0, 'advisor_style': 5.5, 'team_work': 7.0,
                'workload': 8.0, 'theory_practice': 8.5, 'research_field_match': 8.0,
                'skill_development': 8.0, 'learning_pace': 6.5, 'difficulty_preference': 8.5,
                'communication_style': 6.5, 'meeting_frequency': 6.0, 'lab_atmosphere': 7.0,
                'innovation_risk': 7.5, 'methodology_preference': 6.5, 'interdisciplinary': 7.5,
                'flexibility': 6.0, 'evening_weekend_work': 7.5,
                'publication_opportunity': 7.0, 'financial_support': 7.5, 'lab_hierarchy': 6.5, 'core_time_flexibility': 6.0
            }
        },
        {
            'name': '情報セキュリティ研究室',
            'professor': '高橋教授',
            'research_area': 'サイバーセキュリティ、暗号理論、ネットワークセキュリティ',
            'description': 'セキュリティの最前線で研究を行っています。厳格な環境で高いレベルの研究を目指します。',
            'features': {
                'research_intensity': 8.5, 'advisor_style': 4.0, 'team_work': 5.0,
                'workload': 7.5, 'theory_practice': 6.0, 'research_field_match': 8.5,
                'skill_development': 7.0, 'learning_pace': 7.0, 'difficulty_preference': 8.0,
                'communication_style': 6.0, 'meeting_frequency': 5.5, 'lab_atmosphere': 6.5,
                'innovation_risk': 7.0, 'methodology_preference': 6.0, 'interdisciplinary': 6.5,
                'flexibility': 6.5, 'evening_weekend_work': 6.5,
                'publication_opportunity': 8.0, 'financial_support': 7.5, 'lab_hierarchy': 5.5, 'core_time_flexibility': 6.5
            }
        },
        {
            'name': 'HCI・UI/UX研究室',
            'professor': '伊藤教授',
            'research_area': 'ヒューマンコンピュータインタラクション、UI/UXデザイン',
            'description': 'ユーザー中心のシステム設計を研究しています。デザイン思考やユーザビリティテストなど、人間の視点を重視。',
            'features': {
                'research_intensity': 6.5, 'advisor_style': 7.5, 'team_work': 8.0,
                'workload': 5.5, 'theory_practice': 7.5, 'research_field_match': 7.0,
                'skill_development': 7.0, 'learning_pace': 6.0, 'difficulty_preference': 6.0,
                'communication_style': 8.5, 'meeting_frequency': 8.0, 'lab_atmosphere': 8.5,
                'innovation_risk': 7.0, 'methodology_preference': 7.5, 'interdisciplinary': 8.5,
                'flexibility': 8.5, 'evening_weekend_work': 4.0,
                'publication_opportunity': 6.0, 'financial_support': 6.5, 'lab_hierarchy': 8.0, 'core_time_flexibility': 8.5
            }
        }
    ]
    
    return labs_data

def main():
    """メイン実行関数"""
    print("🚀 20項目対応データベース完全新規作成")
    print("=" * 50)
    
    # 既存ファイル削除確認
    if os.path.exists('fdtlss.db'):
        print("⚠️ 既存のfdtlss.dbが見つかりました")
        response = input("削除して新しく作成しますか？ (y/N): ")
        if response.lower() != 'y':
            print("❌ 処理を中止しました")
            return
        
        # バックアップ作成
        backup_file = f'fdtlss_old_{datetime.now().strftime("%Y%m%d_%H%M%S")}.db'
        shutil.copy2('fdtlss.db', backup_file)
        print(f"📦 バックアップ作成: {backup_file}")
        
        # 削除
        os.remove('fdtlss.db')
        print("🗑️ 既存データベースを削除しました")
    
    print("\n1️⃣ 新しいFlaskアプリケーションでデータベース作成...")
    app = create_app()
    
    with app.app_context():
        # 全テーブル作成
        db.create_all()
        print("✅ 20項目対応テーブルを作成しました")
        
        # サンプル研究室データ作成
        print("\n2️⃣ 20項目対応研究室データを作成中...")
        sample_labs = create_20_item_sample_labs()
        
        for lab_data in sample_labs:
            lab = Lab(
                name=lab_data['name'],
                professor=lab_data['professor'],
                research_area=lab_data['research_area'],
                description=lab_data['description'],
                **lab_data['features']  # 20項目すべてを展開
            )
            
            db.session.add(lab)
            print(f"  ✅ {lab_data['name']} を追加")
        
        # システム設定追加
        print("\n3️⃣ システム設定を作成中...")
        default_configs = [
            ('default_fuzzy_engine', 'hybrid', 'string', 'デフォルトのファジィエンジン'),
            ('system_version', '2.0.0', 'string', 'システムバージョン'),
            ('max_evaluation_results', '20', 'int', '最大評価結果表示数'),
        ]
        
        for key, value, config_type, description in default_configs:
            config = SystemConfig(
                config_key=key,
                config_type=config_type,
                description=description
            )
            config.set_value(value)
            db.session.add(config)
            print(f"  ✅ {key} = {value}")
        
        # 保存
        db.session.commit()
        
        # 確認
        lab_count = Lab.query.count()
        config_count = SystemConfig.query.count()
        
        print(f"\n📊 作成完了:")
        print(f"  - 研究室: {lab_count}件")
        print(f"  - システム設定: {config_count}件")
        
        # サンプル確認
        first_lab = Lab.query.first()
        if first_lab:
            print(f"\n📋 サンプル研究室:")
            print(f"  名前: {first_lab.name}")
            print(f"  研究強度: {first_lab.research_intensity}")
            print(f"  論文機会: {first_lab.publication_opportunity}")
            print(f"  経済支援: {first_lab.financial_support}")
    
    print("\n🎉 20項目対応データベース作成完了！")
    print("\n次のステップ:")
    print("  python app.py でサーバー起動")
    print("  ブラウザでhttp://localhost:3000 にアクセス")

if __name__ == '__main__':
    main()