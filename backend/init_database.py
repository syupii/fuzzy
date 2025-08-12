from models import db, Lab
from flask import Flask
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fdtlss.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.init_app(app)
    return app


def init_database():
    """データベース初期化とサンプルデータ投入"""
    app = create_app()

    with app.app_context():
        print("データベースを初期化しています...")

        # テーブル作成
        db.create_all()
        print("テーブルを作成しました")

        # サンプル研究室データ
        sample_labs = [
            {
                'name': 'AI・機械学習研究室',
                'professor': '田中教授',
                'research_area': '機械学習、深層学習、自然言語処理、画像認識',
                'description': '最新のAI技術を使った応用研究を行っています。学生の自主性を重視し、国際学会での発表を積極的に支援しています。GPU クラスターを完備し、大規模な実験が可能です。',
                'research_intensity': 8.5,
                'advisor_style': 6.0,
                'team_work': 7.5,
                'workload': 7.0,
                'theory_practice': 7.0
            },
            {
                'name': 'データサイエンス研究室',
                'professor': '佐藤教授',
                'research_area': 'ビッグデータ解析、統計学、データマイニング、ビジネスアナリティクス',
                'description': 'データ駆動型の問題解決に取り組んでいます。企業との共同研究が多く、実践的なスキルを身につけられます。Python、R、SQLを使った分析手法を習得できます。',
                'research_intensity': 7.0,
                'advisor_style': 5.5,
                'team_work': 6.0,
                'workload': 6.5,
                'theory_practice': 8.5
            },
            {
                'name': 'ロボティクス研究室',
                'professor': '山田教授',
                'research_area': 'ロボット工学、制御工学、メカトロニクス、自動運転技術',
                'description': '実用的なロボットシステムの開発を行っています。ハードウェアとソフトウェアの両方を扱う総合的な研究室です。ROS、組み込みシステム、センサー技術を学べます。',
                'research_intensity': 9.0,
                'advisor_style': 8.0,
                'team_work': 8.5,
                'workload': 8.5,
                'theory_practice': 9.0
            },
            {
                'name': 'セキュリティ研究室',
                'professor': '鈴木教授',
                'research_area': 'サイバーセキュリティ、暗号学、ネットワークセキュリティ、ブロックチェーン',
                'description': 'セキュリティの最前線で研究を行っています。倫理的ハッキングやペネトレーションテストなど実践的な内容も学べます。情報セキュリティスペシャリストの資格取得支援もあります。',
                'research_intensity': 7.5,
                'advisor_style': 4.0,
                'team_work': 5.0,
                'workload': 7.5,
                'theory_practice': 6.0
            },
            {
                'name': 'HCI・UI/UX研究室',
                'professor': '高橋教授',
                'research_area': 'ヒューマンコンピュータインタラクション、UI/UXデザイン、認知科学',
                'description': 'ユーザーエクスペリエンスの向上を目指した研究を行っています。デザイン思考と技術の融合が特徴です。Figma、Adobe Creative Suite、プロトタイピングツールを活用します。',
                'research_intensity': 6.5,
                'advisor_style': 7.5,
                'team_work': 8.0,
                'workload': 5.5,
                'theory_practice': 8.0
            },
            {
                'name': 'ソフトウェア工学研究室',
                'professor': '伊藤教授',
                'research_area': 'ソフトウェア設計、アジャイル開発、DevOps、クラウドコンピューティング',
                'description': '大規模ソフトウェア開発の方法論を研究しています。実際の企業開発プロセスを学べ、Docker、Kubernetes、CI/CDパイプラインなどモダンな技術スタックを習得できます。',
                'research_intensity': 6.0,
                'advisor_style': 6.5,
                'team_work': 9.0,
                'workload': 6.0,
                'theory_practice': 8.5
            }
        ]

        # データベースに追加
        added_count = 0
        for lab_data in sample_labs:
            if not Lab.query.filter_by(name=lab_data['name']).first():
                lab = Lab(**lab_data)
                db.session.add(lab)
                added_count += 1

        db.session.commit()
        print(f"✅ {added_count}個の研究室データを追加しました")

        # データベース確認
        total_labs = Lab.query.count()
        print(f"データベース内の研究室数: {total_labs}")

        print("データベース初期化完了！")
        return True


if __name__ == '__main__':
    init_database()
