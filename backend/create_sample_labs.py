# backend/create_sample_labs.py
# 20項目対応の研究室サンプルデータを作成するスクリプト

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from models import db, Lab
import json

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fdtlss.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    return app

def create_sample_labs():
    """20項目対応の研究室サンプルデータを作成"""
    
    sample_labs = [
        {
            'name': 'AI・機械学習研究室',
            'professor': '田中教授',
            'research_area': '人工知能、機械学習、深層学習、自然言語処理',
            'description': '最先端のAI技術を使った応用研究を行っています。学生の自主性を重視し、国際学会での発表を積極的に支援しています。GPU クラスターを完備し、大規模な実験が可能です。',
            'features': {
                # 基本的な研究環境（6項目）
                'research_intensity': 8.5,
                'advisor_style': 6.0,
                'team_work': 7.5,
                'workload': 7.5,
                'theory_practice': 7.0,
                'research_field_match': 9.0,
                
                # 学習・成長関連（3項目）
                'skill_development': 8.0,
                'learning_pace': 7.5,
                'difficulty_preference': 8.5,
                
                # コミュニケーション・環境関連（3項目）
                'communication_style': 7.0,
                'meeting_frequency': 7.0,
                'lab_atmosphere': 8.0,
                
                # 研究アプローチ関連（3項目）
                'innovation_risk': 8.0,
                'methodology_preference': 8.5,
                'interdisciplinary': 7.5,
                
                # 時間・ライフスタイル関連（2項目）
                'flexibility': 6.5,
                'evening_weekend_work': 7.0,
                
                # 調査結果に基づく追加項目（3項目）
                'publication_opportunity': 8.5,
                'financial_support': 8.0,
                'lab_hierarchy': 6.0,
                'core_time_flexibility': 6.5
            }
        },
        
        {
            'name': 'データサイエンス研究室',
            'professor': '佐藤教授',
            'research_area': 'ビッグデータ解析、統計的機械学習、ビジネスアナリティクス',
            'description': 'データ駆動型の問題解決に取り組んでいます。企業との共同研究が多く、実践的なスキルを身につけられます。Python、R、SQLを使った分析手法を習得できます。',
            'features': {
                'research_intensity': 7.5,
                'advisor_style': 7.0,
                'team_work': 8.0,
                'workload': 6.5,
                'theory_practice': 6.5,
                'research_field_match': 8.5,
                'skill_development': 7.5,
                'learning_pace': 6.5,
                'difficulty_preference': 7.0,
                'communication_style': 8.0,
                'meeting_frequency': 6.5,
                'lab_atmosphere': 7.5,
                'innovation_risk': 6.5,
                'methodology_preference': 7.0,
                'interdisciplinary': 8.5,
                'flexibility': 7.5,
                'evening_weekend_work': 5.5,
                'publication_opportunity': 7.5,
                'financial_support': 7.0,
                'lab_hierarchy': 7.0,
                'core_time_flexibility': 7.5
            }
        },
        
        {
            'name': 'ソフトウェア工学研究室',
            'professor': '山田教授',
            'research_area': 'ソフトウェア設計、開発方法論、品質管理',
            'description': '高品質なソフトウェアを効率的に開発する手法を研究します。アジャイル開発やDevOpsなど実践的な内容も学べます。学生同士の協力を重視した環境です。',
            'features': {
                'research_intensity': 7.0,
                'advisor_style': 6.5,
                'team_work': 8.5,
                'workload': 6.0,
                'theory_practice': 8.0,
                'research_field_match': 7.5,
                'skill_development': 8.5,
                'learning_pace': 7.0,
                'difficulty_preference': 6.5,
                'communication_style': 8.5,
                'meeting_frequency': 7.5,
                'lab_atmosphere': 8.0,
                'innovation_risk': 6.0,
                'methodology_preference': 7.5,
                'interdisciplinary': 7.0,
                'flexibility': 8.0,
                'evening_weekend_work': 5.0,
                'publication_opportunity': 6.5,
                'financial_support': 6.0,
                'lab_hierarchy': 7.5,
                'core_time_flexibility': 8.0
            }
        },
        
        {
            'name': 'ロボティクス研究室',
            'professor': '鈴木教授',
            'research_area': 'ロボット工学、制御システム、自律移動ロボット',
            'description': '自律移動ロボットや産業用ロボットの制御技術を研究します。ハードウェアとソフトウェアの両方を扱う総合的な研究室です。実機での実験が多いのが特徴です。',
            'features': {
                'research_intensity': 8.0,
                'advisor_style': 5.5,
                'team_work': 7.0,
                'workload': 8.0,
                'theory_practice': 8.5,
                'research_field_match': 8.0,
                'skill_development': 8.0,
                'learning_pace': 6.5,
                'difficulty_preference': 8.5,
                'communication_style': 6.5,
                'meeting_frequency': 6.0,
                'lab_atmosphere': 7.0,
                'innovation_risk': 7.5,
                'methodology_preference': 6.5,
                'interdisciplinary': 7.5,
                'flexibility': 6.0,
                'evening_weekend_work': 7.5,
                'publication_opportunity': 7.0,
                'financial_support': 7.5,
                'lab_hierarchy': 6.5,
                'core_time_flexibility': 6.0
            }
        },
        
        {
            'name': '情報セキュリティ研究室',
            'professor': '高橋教授',
            'research_area': 'サイバーセキュリティ、暗号理論、ネットワークセキュリティ',
            'description': 'セキュリティの最前線で研究を行っています。倫理的ハッキングやペネトレーションテストなど実践的な内容も学べます。厳格な環境で高いレベルの研究を目指します。',
            'features': {
                'research_intensity': 8.5,
                'advisor_style': 4.0,
                'team_work': 5.0,
                'workload': 7.5,
                'theory_practice': 6.0,
                'research_field_match': 8.5,
                'skill_development': 7.0,
                'learning_pace': 7.0,
                'difficulty_preference': 8.0,
                'communication_style': 6.0,
                'meeting_frequency': 5.5,
                'lab_atmosphere': 6.5,
                'innovation_risk': 7.0,
                'methodology_preference': 6.0,
                'interdisciplinary': 6.5,
                'flexibility': 6.5,
                'evening_weekend_work': 6.5,
                'publication_opportunity': 8.0,
                'financial_support': 7.5,
                'lab_hierarchy': 5.5,
                'core_time_flexibility': 6.5
            }
        },
        
        {
            'name': 'HCI・UI/UX研究室',
            'professor': '伊藤教授',
            'research_area': 'ヒューマンコンピュータインタラクション、UI/UXデザイン',
            'description': 'ユーザー中心のシステム設計を研究しています。デザイン思考やユーザビリティテストなど、人間の視点を重視した研究アプローチが特徴です。',
            'features': {
                'research_intensity': 6.5,
                'advisor_style': 7.5,
                'team_work': 8.0,
                'workload': 5.5,
                'theory_practice': 7.5,
                'research_field_match': 7.0,
                'skill_development': 7.0,
                'learning_pace': 6.0,
                'difficulty_preference': 6.0,
                'communication_style': 8.5,
                'meeting_frequency': 8.0,
                'lab_atmosphere': 8.5,
                'innovation_risk': 7.0,
                'methodology_preference': 7.5,
                'interdisciplinary': 8.5,
                'flexibility': 8.5,
                'evening_weekend_work': 4.0,
                'publication_opportunity': 6.0,
                'financial_support': 6.5,
                'lab_hierarchy': 8.0,
                'core_time_flexibility': 8.5
            }
        },
        
        {
            'name': 'コンピュータビジョン研究室',
            'professor': '渡辺教授',
            'research_area': '画像認識、映像解析、コンピュータビジョン',
            'description': '画像・映像処理技術の最先端研究を行っています。医療画像解析や自動運転技術など社会実装を意識した研究テーマが豊富です。',
            'features': {
                'research_intensity': 8.0,
                'advisor_style': 5.0,
                'team_work': 6.5,
                'workload': 7.0,
                'theory_practice': 7.5,
                'research_field_match': 8.5,
                'skill_development': 7.5,
                'learning_pace': 7.0,
                'difficulty_preference': 8.0,
                'communication_style': 6.0,
                'meeting_frequency': 6.0,
                'lab_atmosphere': 7.0,
                'innovation_risk': 8.5,
                'methodology_preference': 8.0,
                'interdisciplinary': 7.0,
                'flexibility': 6.0,
                'evening_weekend_work': 6.5,
                'publication_opportunity': 8.0,
                'financial_support': 7.5,
                'lab_hierarchy': 5.0,
                'core_time_flexibility': 6.0
            }
        },
        
        {
            'name': '分散システム研究室',
            'professor': '中村教授',
            'research_area': '分散システム、クラウドコンピューティング、ブロックチェーン',
            'description': '大規模分散システムの設計と運用技術を研究しています。クラウドネイティブ技術やマイクロサービスアーキテクチャなど最新技術を扱います。',
            'features': {
                'research_intensity': 7.5,
                'advisor_style': 6.0,
                'team_work': 7.0,
                'workload': 7.0,
                'theory_practice': 8.0,
                'research_field_match': 8.0,
                'skill_development': 8.0,
                'learning_pace': 7.5,
                'difficulty_preference': 7.5,
                'communication_style': 7.0,
                'meeting_frequency': 6.5,
                'lab_atmosphere': 7.5,
                'innovation_risk': 7.5,
                'methodology_preference': 8.0,
                'interdisciplinary': 7.0,
                'flexibility': 7.0,
                'evening_weekend_work': 6.0,
                'publication_opportunity': 7.0,
                'financial_support': 7.0,
                'lab_hierarchy': 6.5,
                'core_time_flexibility': 7.0
            }
        }
    ]
    
    return sample_labs

def main():
    """メイン実行関数"""
    app = create_app()
    
    with app.app_context():
        # 既存の研究室データを削除
        print("🗑️ 既存の研究室データを削除中...")
        Lab.query.delete()
        
        # サンプル研究室データを作成
        sample_labs = create_sample_labs()
        print(f"📚 {len(sample_labs)}個の研究室データを作成中...")
        
        for lab_data in sample_labs:
            lab = Lab(
                name=lab_data['name'],
                professor=lab_data['professor'],
                research_area=lab_data['research_area'],
                description=lab_data['description'],
                
                # 20項目の特性データを個別に設定
                research_intensity=lab_data['features']['research_intensity'],
                advisor_style=lab_data['features']['advisor_style'],
                team_work=lab_data['features']['team_work'],
                workload=lab_data['features']['workload'],
                theory_practice=lab_data['features']['theory_practice'],
                research_field_match=lab_data['features']['research_field_match'],
                skill_development=lab_data['features']['skill_development'],
                learning_pace=lab_data['features']['learning_pace'],
                difficulty_preference=lab_data['features']['difficulty_preference'],
                communication_style=lab_data['features']['communication_style'],
                meeting_frequency=lab_data['features']['meeting_frequency'],
                lab_atmosphere=lab_data['features']['lab_atmosphere'],
                innovation_risk=lab_data['features']['innovation_risk'],
                methodology_preference=lab_data['features']['methodology_preference'],
                interdisciplinary=lab_data['features']['interdisciplinary'],
                flexibility=lab_data['features']['flexibility'],
                evening_weekend_work=lab_data['features']['evening_weekend_work'],
                publication_opportunity=lab_data['features']['publication_opportunity'],
                financial_support=lab_data['features']['financial_support'],
                lab_hierarchy=lab_data['features']['lab_hierarchy'],
                core_time_flexibility=lab_data['features']['core_time_flexibility'],
                
                is_active=True
            )
            
            db.session.add(lab)
            print(f"  ✅ {lab_data['name']} を追加")
        
        # データベースに保存
        db.session.commit()
        print(f"\n🎉 {len(sample_labs)}個の研究室データを作成完了！")
        
        # 検証
        lab_count = Lab.query.count()
        print(f"📊 データベース内の研究室数: {lab_count}")
        
        # サンプル表示
        first_lab = Lab.query.first()
        if first_lab:
            print(f"\n📋 サンプル研究室情報:")
            print(f"   名前: {first_lab.name}")
            print(f"   教授: {first_lab.professor}")
            print(f"   研究分野: {first_lab.research_area}")
            print(f"   研究強度: {first_lab.research_intensity}")
            print(f"   論文機会: {first_lab.publication_opportunity}")
            print(f"   経済支援: {first_lab.financial_support}")

if __name__ == '__main__':
    main()