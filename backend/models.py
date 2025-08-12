# backend/models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

# 既存のクラス（完全に保持）


class Lab(db.Model):
    """研究室モデル"""
    __tablename__ = 'labs'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    professor = db.Column(db.String(100), nullable=False)
    research_area = db.Column(db.String(500), nullable=False)
    description = db.Column(db.Text)

    # ファジィ特徴量（1-10スケール）
    research_intensity = db.Column(db.Float, default=5.0)    # 研究強度
    # 指導スタイル (1:厳格 ↔ 10:自由)
    advisor_style = db.Column(db.Float, default=5.0)
    # チームワーク (1:個人 ↔ 10:チーム)
    team_work = db.Column(db.Float, default=5.0)
    # ワークロード (1:軽い ↔ 10:重い)
    workload = db.Column(db.Float, default=5.0)
    theory_practice = db.Column(
        db.Float, default=5.0)       # 理論/実践 (1:理論 ↔ 10:実践)

    # メタデータ
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'professor': self.professor,
            'research_area': self.research_area,
            'description': self.description,
            'features': {
                'research_intensity': self.research_intensity,
                'advisor_style': self.advisor_style,
                'team_work': self.team_work,
                'workload': self.workload,
                'theory_practice': self.theory_practice
            },
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Evaluation(db.Model):
    """評価履歴モデル"""
    __tablename__ = 'evaluations'

    id = db.Column(db.Integer, primary_key=True)

    # セッション識別（匿名ユーザー用）
    session_id = db.Column(db.String(100), nullable=True)

    # ユーザー入力データ（1-10スケール）
    research_intensity = db.Column(db.Float, nullable=False)
    advisor_style = db.Column(db.Float, nullable=False)
    team_work = db.Column(db.Float, nullable=False)
    workload = db.Column(db.Float, nullable=False)
    theory_practice = db.Column(db.Float, nullable=False)

    # 結果データ（JSON形式）
    results_json = db.Column(db.Text)

    # メタデータ
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_results(self, results):
        """結果をJSON形式で保存"""
        self.results_json = json.dumps(results, ensure_ascii=False)

    def get_results(self):
        """結果をオブジェクト形式で取得"""
        if self.results_json:
            return json.loads(self.results_json)
        return None

    def to_dict(self):
        return {
            'id': self.id,
            'session_id': self.session_id,
            'preferences': {
                'research_intensity': self.research_intensity,
                'advisor_style': self.advisor_style,
                'team_work': self.team_work,
                'workload': self.workload,
                'theory_practice': self.theory_practice
            },
            'results': self.get_results(),
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# 🆕 新しいテーブル（遺伝的アルゴリズム用）


class GeneticIndividual(db.Model):
    """遺伝的アルゴリズムの個体記録"""
    __tablename__ = 'genetic_individuals'

    id = db.Column(db.Integer, primary_key=True)
    individual_id = db.Column(db.String(100), nullable=False, unique=True)
    generation = db.Column(db.Integer, nullable=False)
    genome_data = db.Column(db.Text)  # JSON形式で遺伝子保存

    # 適応度スコア
    accuracy = db.Column(db.Float)
    simplicity = db.Column(db.Float)
    interpretability = db.Column(db.Float)
    generalization = db.Column(db.Float)
    validity = db.Column(db.Float)
    overall_fitness = db.Column(db.Float)

    # 系譜情報
    parent1_id = db.Column(db.String(100))
    parent2_id = db.Column(db.String(100))

    # モデル情報
    model_complexity = db.Column(db.Integer)
    tree_depth = db.Column(db.Integer)
    evaluation_time = db.Column(db.Float)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_genome_data(self, genome):
        """ゲノムデータをJSON形式で保存"""
        self.genome_data = json.dumps(genome, ensure_ascii=False, default=str)

    def get_genome_data(self):
        """ゲノムデータをオブジェクト形式で取得"""
        if self.genome_data:
            return json.loads(self.genome_data)
        return None

    def to_dict(self):
        return {
            'id': self.id,
            'individual_id': self.individual_id,
            'generation': self.generation,
            'genome': self.get_genome_data(),
            'fitness': {
                'accuracy': self.accuracy,
                'simplicity': self.simplicity,
                'interpretability': self.interpretability,
                'generalization': self.generalization,
                'validity': self.validity,
                'overall': self.overall_fitness
            },
            'parents': [self.parent1_id, self.parent2_id],
            'model_info': {
                'complexity': self.model_complexity,
                'depth': self.tree_depth,
                'evaluation_time': self.evaluation_time
            },
            'created_at': self.created_at.isoformat()
        }


class DecisionPath(db.Model):
    """決定パス記録テーブル"""
    __tablename__ = 'decision_paths'

    id = db.Column(db.Integer, primary_key=True)
    path_id = db.Column(db.String(100), nullable=False, unique=True)
    evaluation_id = db.Column(db.Integer, db.ForeignKey('evaluations.id'))

    # 使用されたモデル情報
    model_type = db.Column(db.String(50))  # 'simple' or 'genetic'
    model_version = db.Column(db.String(50))
    model_id = db.Column(db.String(100))  # genetic modelの場合のindividual_id

    # 決定プロセス詳細（JSON）
    decision_nodes = db.Column(db.Text)  # 通った決定ノード
    feature_contributions = db.Column(db.Text)  # 特徴量の貢献度
    confidence_scores = db.Column(db.Text)  # 各段階の信頼度

    # 結果
    final_prediction = db.Column(db.Float)
    explanation_text = db.Column(db.Text)

    # パフォーマンス情報
    prediction_time = db.Column(db.Float)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_decision_data(self, nodes, contributions, confidences):
        """決定データをJSON形式で保存"""
        self.decision_nodes = json.dumps(
            nodes, ensure_ascii=False, default=str)
        self.feature_contributions = json.dumps(
            contributions, ensure_ascii=False, default=str)
        self.confidence_scores = json.dumps(
            confidences, ensure_ascii=False, default=str)

    def get_decision_data(self):
        """決定データをオブジェクト形式で取得"""
        return {
            'nodes': json.loads(self.decision_nodes) if self.decision_nodes else [],
            'contributions': json.loads(self.feature_contributions) if self.feature_contributions else {},
            'confidences': json.loads(self.confidence_scores) if self.confidence_scores else []
        }

    def to_dict(self):
        return {
            'id': self.id,
            'path_id': self.path_id,
            'evaluation_id': self.evaluation_id,
            'model_info': {
                'type': self.model_type,
                'version': self.model_version,
                'model_id': self.model_id
            },
            'decision_data': self.get_decision_data(),
            'prediction': self.final_prediction,
            'explanation': self.explanation_text,
            'performance': {
                'prediction_time': self.prediction_time
            },
            'created_at': self.created_at.isoformat()
        }


class OptimizationRun(db.Model):
    """最適化実行記録"""
    __tablename__ = 'optimization_runs'

    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.String(100), nullable=False, unique=True)

    # 最適化設定
    population_size = db.Column(db.Integer)
    generations = db.Column(db.Integer)
    mutation_rate = db.Column(db.Float)
    crossover_rate = db.Column(db.Float)
    max_depth = db.Column(db.Integer)
    tournament_size = db.Column(db.Integer)

    # データ情報
    training_samples = db.Column(db.Integer)
    test_samples = db.Column(db.Integer)
    feature_names = db.Column(db.Text)  # JSON配列
    target_column = db.Column(db.String(100))

    # 結果
    best_fitness = db.Column(db.Float)
    best_individual_id = db.Column(db.String(100))
    convergence_generation = db.Column(db.Integer)
    final_diversity = db.Column(db.Float)

    # 統計（JSON）
    fitness_history = db.Column(db.Text)  # 適応度履歴
    diversity_history = db.Column(db.Text)  # 多様性履歴

    # メタデータ
    execution_time = db.Column(db.Float)  # 秒
    # running, completed, failed
    status = db.Column(db.String(20), default='running')
    description = db.Column(db.Text)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)

    def set_feature_names(self, feature_names):
        """特徴名をJSON形式で保存"""
        self.feature_names = json.dumps(feature_names, ensure_ascii=False)

    def get_feature_names(self):
        """特徴名をリスト形式で取得"""
        if self.feature_names:
            return json.loads(self.feature_names)
        return []

    def set_fitness_history(self, history):
        """適応度履歴をJSON形式で保存"""
        self.fitness_history = json.dumps(history, ensure_ascii=False)

    def get_fitness_history(self):
        """適応度履歴をリスト形式で取得"""
        if self.fitness_history:
            return json.loads(self.fitness_history)
        return []

    def set_diversity_history(self, history):
        """多様性履歴をJSON形式で保存"""
        self.diversity_history = json.dumps(history, ensure_ascii=False)

    def get_diversity_history(self):
        """多様性履歴をリスト形式で取得"""
        if self.diversity_history:
            return json.loads(self.diversity_history)
        return []

    def to_dict(self):
        return {
            'id': self.id,
            'run_id': self.run_id,
            'configuration': {
                'population_size': self.population_size,
                'generations': self.generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'max_depth': self.max_depth,
                'tournament_size': self.tournament_size
            },
            'data_info': {
                'training_samples': self.training_samples,
                'test_samples': self.test_samples,
                'feature_names': self.get_feature_names(),
                'target_column': self.target_column
            },
            'results': {
                'best_fitness': self.best_fitness,
                'best_individual_id': self.best_individual_id,
                'convergence_generation': self.convergence_generation,
                'final_diversity': self.final_diversity,
                'fitness_history': self.get_fitness_history(),
                'diversity_history': self.get_diversity_history()
            },
            'execution': {
                'execution_time': self.execution_time,
                'status': self.status,
                'description': self.description
            },
            'timestamps': {
                'created_at': self.created_at.isoformat(),
                'completed_at': self.completed_at.isoformat() if self.completed_at else None
            }
        }


class ModelRegistry(db.Model):
    """モデル登録簿"""
    __tablename__ = 'model_registry'

    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.String(100), nullable=False, unique=True)
    model_name = db.Column(db.String(200))
    model_type = db.Column(db.String(50))  # 'simple', 'genetic_fuzzy_tree'
    version = db.Column(db.String(50))

    # ファイル情報
    model_filepath = db.Column(db.String(500))
    result_filepath = db.Column(db.String(500))
    file_size_bytes = db.Column(db.BigInteger)
    checksum = db.Column(db.String(100))

    # 性能情報
    best_fitness = db.Column(db.Float)
    model_complexity = db.Column(db.Integer)
    tree_depth = db.Column(db.Integer)

    # 訓練情報
    training_samples = db.Column(db.Integer)
    test_samples = db.Column(db.Integer)

    # メタデータ
    description = db.Column(db.Text)
    tags = db.Column(db.Text)  # JSON配列
    is_active = db.Column(db.Boolean, default=True)
    is_production = db.Column(db.Boolean, default=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_used_at = db.Column(db.DateTime)
    usage_count = db.Column(db.Integer, default=0)

    def set_tags(self, tags):
        """タグをJSON形式で保存"""
        self.tags = json.dumps(tags, ensure_ascii=False)

    def get_tags(self):
        """タグをリスト形式で取得"""
        if self.tags:
            return json.loads(self.tags)
        return []

    def increment_usage(self):
        """使用回数をインクリメント"""
        self.usage_count = (self.usage_count or 0) + 1
        self.last_used_at = datetime.utcnow()

    def to_dict(self):
        return {
            'id': self.id,
            'model_id': self.model_id,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'version': self.version,
            'file_info': {
                'model_filepath': self.model_filepath,
                'result_filepath': self.result_filepath,
                'file_size_bytes': self.file_size_bytes,
                'checksum': self.checksum
            },
            'performance': {
                'best_fitness': self.best_fitness,
                'model_complexity': self.model_complexity,
                'tree_depth': self.tree_depth
            },
            'training_info': {
                'training_samples': self.training_samples,
                'test_samples': self.test_samples
            },
            'metadata': {
                'description': self.description,
                'tags': self.get_tags(),
                'is_active': self.is_active,
                'is_production': self.is_production
            },
            'usage': {
                'usage_count': self.usage_count,
                'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None
            },
            'timestamps': {
                'created_at': self.created_at.isoformat(),
            }
        }


class SystemConfig(db.Model):
    """システム設定"""
    __tablename__ = 'system_config'

    id = db.Column(db.Integer, primary_key=True)
    config_key = db.Column(db.String(100), nullable=False, unique=True)
    config_value = db.Column(db.Text)
    # string, int, float, bool, json
    config_type = db.Column(db.String(20), default='string')
    description = db.Column(db.Text)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def get_value(self):
        """型に応じて値を変換して取得"""
        if self.config_value is None:
            return None

        if self.config_type == 'int':
            return int(self.config_value)
        elif self.config_type == 'float':
            return float(self.config_value)
        elif self.config_type == 'bool':
            return self.config_value.lower() in ('true', '1', 'yes', 'on')
        elif self.config_type == 'json':
            return json.loads(self.config_value)
        else:
            return self.config_value

    def set_value(self, value):
        """型に応じて値を変換して保存"""
        if self.config_type == 'json':
            self.config_value = json.dumps(value, ensure_ascii=False)
        else:
            self.config_value = str(value)

    def to_dict(self):
        return {
            'id': self.id,
            'config_key': self.config_key,
            'config_value': self.get_value(),
            'config_type': self.config_type,
            'description': self.description,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }

# ユーティリティ関数


def create_app():
    """Flaskアプリケーション作成（既存の関数と同じ）"""
    from flask import Flask

    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'development-secret-key'
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fdtlss.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)

    return app


def init_extended_database():
    """拡張データベース初期化"""
    print("🗄️ 拡張データベースを初期化しています...")

    app = create_app()

    with app.app_context():
        try:
            # 全テーブル作成
            db.create_all()
            print("✅ 全テーブルを作成しました")

            # 基本設定データ投入
            init_system_config()

            print("🎉 拡張データベース初期化完了！")
            return True

        except Exception as e:
            print(f"❌ データベース初期化失敗: {e}")
            return False


def init_system_config():
    """システム設定初期化"""

    default_configs = [
        {
            'config_key': 'default_fuzzy_engine',
            'config_value': 'hybrid',
            'config_type': 'string',
            'description': 'デフォルトで使用するファジィエンジンタイプ (simple/hybrid)'
        },
        {
            'config_key': 'genetic_model_auto_reload',
            'config_value': 'true',
            'config_type': 'bool',
            'description': '遺伝的モデルの自動再読み込み'
        },
        {
            'config_key': 'max_prediction_cache',
            'config_value': '1000',
            'config_type': 'int',
            'description': '予測結果の最大キャッシュ数'
        },
        {
            'config_key': 'model_cleanup_threshold',
            'config_value': '50',
            'config_type': 'int',
            'description': 'モデル自動清理のしきい値'
        },
        {
            'config_key': 'feature_weights',
            'config_value': '{"research_intensity": 0.25, "advisor_style": 0.20, "team_work": 0.20, "workload": 0.15, "theory_practice": 0.20}',
            'config_type': 'json',
            'description': '基準の重み設定'
        }
    ]

    for config_data in default_configs:
        existing = SystemConfig.query.filter_by(
            config_key=config_data['config_key']).first()
        if not existing:
            config = SystemConfig(**config_data)
            db.session.add(config)

    db.session.commit()
    print(f"✅ システム設定を初期化しました")


def get_system_config(key: str, default=None):
    """システム設定取得"""
    try:
        config = SystemConfig.query.filter_by(config_key=key).first()
        if config:
            return config.get_value()
        return default
    except:
        return default


def set_system_config(key: str, value, config_type: str = 'string', description: str = ''):
    """システム設定保存"""
    try:
        config = SystemConfig.query.filter_by(config_key=key).first()
        if config:
            config.set_value(value)
            config.config_type = config_type
            if description:
                config.description = description
        else:
            config = SystemConfig(
                config_key=key,
                config_type=config_type,
                description=description
            )
            config.set_value(value)
            db.session.add(config)

        db.session.commit()
        return True
    except Exception as e:
        print(f"⚠️ システム設定保存失敗: {e}")
        db.session.rollback()
        return False

# データベース管理ユーティリティ


class DatabaseManager:
    """データベース管理クラス"""

    @staticmethod
    def get_table_counts():
        """各テーブルのレコード数取得"""
        try:
            return {
                'labs': Lab.query.count(),
                'evaluations': Evaluation.query.count(),
                'genetic_individuals': GeneticIndividual.query.count(),
                'decision_paths': DecisionPath.query.count(),
                'optimization_runs': OptimizationRun.query.count(),
                'model_registry': ModelRegistry.query.count(),
                'system_config': SystemConfig.query.count()
            }
        except:
            return {}

    @staticmethod
    def cleanup_old_records(days_threshold: int = 30):
        """古いレコードの清理"""
        from datetime import timedelta

        cutoff_date = datetime.utcnow() - timedelta(days=days_threshold)

        cleanup_count = 0

        try:
            # 古い評価データ
            old_evaluations = Evaluation.query.filter(
                Evaluation.created_at < cutoff_date).all()
            for eval_record in old_evaluations:
                # 関連する決定パスも削除
                DecisionPath.query.filter_by(
                    evaluation_id=eval_record.id).delete()
                db.session.delete(eval_record)
                cleanup_count += 1

            # 古い遺伝的個体記録
            old_individuals = GeneticIndividual.query.filter(
                GeneticIndividual.created_at < cutoff_date).all()
            for individual in old_individuals:
                db.session.delete(individual)
                cleanup_count += 1

            db.session.commit()

            print(f"🧹 {cleanup_count}件の古いレコードを清理しました")
            return cleanup_count

        except Exception as e:
            db.session.rollback()
            print(f"❌ レコード清理失敗: {e}")
            return 0

    @staticmethod
    def get_database_size():
        """データベースサイズ取得"""
        try:
            import os
            db_path = 'fdtlss.db'
            if os.path.exists(db_path):
                size_bytes = os.path.getsize(db_path)
                size_mb = size_bytes / (1024 * 1024)
                return {
                    'size_bytes': size_bytes,
                    'size_mb': round(size_mb, 2)
                }
        except:
            pass

        return {'size_bytes': 0, 'size_mb': 0}


if __name__ == '__main__':
    init_extended_database()
