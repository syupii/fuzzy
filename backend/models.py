# backend/models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

# 🔥 修正版: 20項目対応のLabモデル
class Lab(db.Model):
    """研究室モデル（20項目対応版）"""
    __tablename__ = 'labs'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    professor = db.Column(db.String(100), nullable=False)
    research_area = db.Column(db.String(500), nullable=False)
    description = db.Column(db.Text)

    # 🔥 20項目のファジィ特徴量（1-10スケール）
    
    # 基本的な研究環境（6項目）
    research_intensity = db.Column(db.Float, default=5.0)    # 研究強度
    advisor_style = db.Column(db.Float, default=5.0)         # 指導スタイル (1:厳格 ↔ 10:自由)
    team_work = db.Column(db.Float, default=5.0)             # チームワーク (1:個人 ↔ 10:チーム)
    workload = db.Column(db.Float, default=5.0)              # ワークロード (1:軽い ↔ 10:重い)
    theory_practice = db.Column(db.Float, default=5.0)       # 理論/実践 (1:理論 ↔ 10:実践)
    research_field_match = db.Column(db.Float, default=5.0)  # 研究分野適合性
    
    # 学習・成長関連（3項目）
    skill_development = db.Column(db.Float, default=5.0)     # スキル開発
    learning_pace = db.Column(db.Float, default=5.0)         # 学習ペース
    difficulty_preference = db.Column(db.Float, default=5.0) # 難易度志向
    
    # コミュニケーション・環境関連（3項目）
    communication_style = db.Column(db.Float, default=5.0)   # コミュニケーション
    meeting_frequency = db.Column(db.Float, default=5.0)     # ミーティング頻度
    lab_atmosphere = db.Column(db.Float, default=5.0)        # 研究室雰囲気
    
    # 研究アプローチ関連（3項目）
    innovation_risk = db.Column(db.Float, default=5.0)       # 革新性リスク
    methodology_preference = db.Column(db.Float, default=5.0) # 手法志向
    interdisciplinary = db.Column(db.Float, default=5.0)     # 学際性
    
    # 時間・ライフスタイル関連（2項目）
    flexibility = db.Column(db.Float, default=5.0)           # 時間の柔軟性
    evening_weekend_work = db.Column(db.Float, default=5.0)  # 時間外研究
    
    # 調査結果に基づく追加項目（3項目）
    publication_opportunity = db.Column(db.Float, default=5.0) # 論文執筆機会
    financial_support = db.Column(db.Float, default=5.0)       # 経済的支援
    lab_hierarchy = db.Column(db.Float, default=5.0)           # 研究室上下関係
    core_time_flexibility = db.Column(db.Float, default=5.0)   # コアタイム柔軟性

    # メタデータ
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

    def to_dict(self):
        """20項目すべてを含むdict形式で返す"""
        return {
            'id': self.id,
            'name': self.name,
            'professor': self.professor,
            'research_area': self.research_area,
            'description': self.description,
            'features': {
                # 基本的な研究環境
                'research_intensity': self.research_intensity,
                'advisor_style': self.advisor_style,
                'team_work': self.team_work,
                'workload': self.workload,
                'theory_practice': self.theory_practice,
                'research_field_match': self.research_field_match,
                
                # 学習・成長関連
                'skill_development': self.skill_development,
                'learning_pace': self.learning_pace,
                'difficulty_preference': self.difficulty_preference,
                
                # コミュニケーション・環境関連
                'communication_style': self.communication_style,
                'meeting_frequency': self.meeting_frequency,
                'lab_atmosphere': self.lab_atmosphere,
                
                # 研究アプローチ関連
                'innovation_risk': self.innovation_risk,
                'methodology_preference': self.methodology_preference,
                'interdisciplinary': self.interdisciplinary,
                
                # 時間・ライフスタイル関連
                'flexibility': self.flexibility,
                'evening_weekend_work': self.evening_weekend_work,
                
                # 調査結果に基づく追加項目
                'publication_opportunity': self.publication_opportunity,
                'financial_support': self.financial_support,
                'lab_hierarchy': self.lab_hierarchy,
                'core_time_flexibility': self.core_time_flexibility,
            },
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Evaluation(db.Model):
    """評価履歴モデル（20項目対応版）"""
    __tablename__ = 'evaluations'

    id = db.Column(db.Integer, primary_key=True)

    # セッション識別（匿名ユーザー用）
    session_id = db.Column(db.String(100), nullable=True)

    # ユーザー入力データ（20項目すべて）
    # 基本的な研究環境
    research_intensity = db.Column(db.Float, nullable=False)
    advisor_style = db.Column(db.Float, nullable=False)
    team_work = db.Column(db.Float, nullable=False)
    workload = db.Column(db.Float, nullable=False)
    theory_practice = db.Column(db.Float, nullable=False)
    research_field_match = db.Column(db.Float, nullable=False)
    
    # 学習・成長関連
    skill_development = db.Column(db.Float, nullable=False)
    learning_pace = db.Column(db.Float, nullable=False)
    difficulty_preference = db.Column(db.Float, nullable=False)
    
    # コミュニケーション・環境関連
    communication_style = db.Column(db.Float, nullable=False)
    meeting_frequency = db.Column(db.Float, nullable=False)
    lab_atmosphere = db.Column(db.Float, nullable=False)
    
    # 研究アプローチ関連
    innovation_risk = db.Column(db.Float, nullable=False)
    methodology_preference = db.Column(db.Float, nullable=False)
    interdisciplinary = db.Column(db.Float, nullable=False)
    
    # 時間・ライフスタイル関連
    flexibility = db.Column(db.Float, nullable=False)
    evening_weekend_work = db.Column(db.Float, nullable=False)
    
    # 調査結果に基づく追加項目
    publication_opportunity = db.Column(db.Float, nullable=False)
    financial_support = db.Column(db.Float, nullable=False)
    lab_hierarchy = db.Column(db.Float, nullable=False)
    core_time_flexibility = db.Column(db.Float, nullable=False)

    # 結果データ（JSON形式で保存）
    user_preferences = db.Column(db.Text)  # 完全なユーザー設定
    evaluation_count = db.Column(db.Integer)  # 評価した研究室数
    avg_score = db.Column(db.Float)        # 平均適合度スコア
    best_lab_id = db.Column(db.Integer)    # 最高適合度の研究室ID
    engine_used = db.Column(db.String(100)) # 使用したエンジン

    # メタデータ
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_results(self, results):
        """結果をJSON形式で保存"""
        self.results_json = json.dumps(results, ensure_ascii=False)

    def get_results(self):
        """結果をオブジェクト形式で取得"""
        if hasattr(self, 'results_json') and self.results_json:
            return json.loads(self.results_json)
        return None

    def to_dict(self):
        """20項目すべてを含むdict形式で返す"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'preferences': {
                # 基本的な研究環境
                'research_intensity': self.research_intensity,
                'advisor_style': self.advisor_style,
                'team_work': self.team_work,
                'workload': self.workload,
                'theory_practice': self.theory_practice,
                'research_field_match': self.research_field_match,
                
                # 学習・成長関連
                'skill_development': self.skill_development,
                'learning_pace': self.learning_pace,
                'difficulty_preference': self.difficulty_preference,
                
                # コミュニケーション・環境関連
                'communication_style': self.communication_style,
                'meeting_frequency': self.meeting_frequency,
                'lab_atmosphere': self.lab_atmosphere,
                
                # 研究アプローチ関連
                'innovation_risk': self.innovation_risk,
                'methodology_preference': self.methodology_preference,
                'interdisciplinary': self.interdisciplinary,
                
                # 時間・ライフスタイル関連
                'flexibility': self.flexibility,
                'evening_weekend_work': self.evening_weekend_work,
                
                # 調査結果に基づく追加項目
                'publication_opportunity': self.publication_opportunity,
                'financial_support': self.financial_support,
                'lab_hierarchy': self.lab_hierarchy,
                'core_time_flexibility': self.core_time_flexibility,
            },
            'evaluation_summary': {
                'evaluation_count': self.evaluation_count,
                'avg_score': self.avg_score,
                'best_lab_id': self.best_lab_id,
                'engine_used': self.engine_used
            },
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
            'genealogy': {
                'parent1_id': self.parent1_id,
                'parent2_id': self.parent2_id
            },
            'model_info': {
                'complexity': self.model_complexity,
                'depth': self.tree_depth,
                'evaluation_time': self.evaluation_time
            },
            'created_at': self.created_at.isoformat()
        }


class DecisionPath(db.Model):
    """決定パス記録"""
    __tablename__ = 'decision_paths'

    id = db.Column(db.Integer, primary_key=True)
    evaluation_id = db.Column(db.Integer, db.ForeignKey('evaluations.id'))
    step_order = db.Column(db.Integer)
    
    # 決定情報
    criterion = db.Column(db.String(100))
    threshold = db.Column(db.Float)
    user_value = db.Column(db.Float)
    lab_value = db.Column(db.Float)
    decision_result = db.Column(db.String(50))  # 'match', 'partial', 'mismatch'
    
    # 重み情報
    criterion_weight = db.Column(db.Float)
    confidence = db.Column(db.Float)
    
    # 説明情報
    rule_explanation = db.Column(db.Text)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'evaluation_id': self.evaluation_id,
            'step_order': self.step_order,
            'decision': {
                'criterion': self.criterion,
                'threshold': self.threshold,
                'user_value': self.user_value,
                'lab_value': self.lab_value,
                'result': self.decision_result
            },
            'weights': {
                'criterion_weight': self.criterion_weight,
                'confidence': self.confidence
            },
            'explanation': self.rule_explanation,
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
    status = db.Column(db.String(20), default='running')  # running, completed, failed
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
    validation_accuracy = db.Column(db.Float)
    test_accuracy = db.Column(db.Float)

    # 使用統計
    usage_count = db.Column(db.Integer, default=0)
    last_used_at = db.Column(db.DateTime)

    # 状態
    is_active = db.Column(db.Boolean, default=True)
    is_production_ready = db.Column(db.Boolean, default=False)

    # メタデータ
    description = db.Column(db.Text)
    tags = db.Column(db.Text)  # JSON配列

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def increment_usage(self):
        """使用回数を増加"""
        self.usage_count += 1
        self.last_used_at = datetime.utcnow()

    def set_tags(self, tags):
        """タグをJSON形式で保存"""
        self.tags = json.dumps(tags, ensure_ascii=False)

    def get_tags(self):
        """タグをリスト形式で取得"""
        if self.tags:
            return json.loads(self.tags)
        return []

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
                'validation_accuracy': self.validation_accuracy,
                'test_accuracy': self.test_accuracy
            },
            'usage': {
                'usage_count': self.usage_count,
                'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None
            },
            'status': {
                'is_active': self.is_active,
                'is_production_ready': self.is_production_ready
            },
            'metadata': {
                'description': self.description,
                'tags': self.get_tags()
            },
            'timestamps': {
                'created_at': self.created_at.isoformat(),
                'updated_at': self.updated_at.isoformat()
            }
        }


class SystemConfig(db.Model):
    """システム設定"""
    __tablename__ = 'system_config'

    id = db.Column(db.Integer, primary_key=True)
    config_key = db.Column(db.String(100), nullable=False, unique=True)
    config_value = db.Column(db.Text)
    config_type = db.Column(db.String(20), default='string')  # string, int, float, bool, json
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
def get_system_config(key: str, default=None):
    """システム設定取得"""
    config = SystemConfig.query.filter_by(config_key=key).first()
    if config:
        return config.get_value()
    return default


def set_system_config(key: str, value, config_type: str = 'string', description: str = ''):
    """システム設定更新"""
    try:
        config = SystemConfig.query.filter_by(config_key=key).first()
        if config:
            config.set_value(value)
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
        db.session.rollback()
        print(f"❌ 設定保存エラー: {e}")
        return False


class DatabaseManager:
    """データベース管理ユーティリティ"""
    
    @staticmethod
    def get_table_counts():
        """各テーブルのレコード数を取得"""
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
        except Exception as e:
            print(f"⚠️ テーブル統計取得エラー: {e}")
            return {}

    @staticmethod
    def get_database_size():
        """データベースサイズ情報を取得"""
        try:
            import os
            
            # SQLiteファイルサイズ取得
            db_file = 'fdtlss.db'
            if os.path.exists(db_file):
                size_bytes = os.path.getsize(db_file)
                size_mb = size_bytes / (1024 * 1024)
                return {
                    'size_bytes': size_bytes,
                    'size_mb': round(size_mb, 2),
                    'file_path': os.path.abspath(db_file)
                }
        except Exception as e:
            print(f"⚠️ データベースサイズ取得エラー: {e}")
        
        return {'size_bytes': 0, 'size_mb': 0, 'file_path': 'unknown'}


def create_app():
    """Flaskアプリケーション作成"""
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
            default_configs = [
                ('default_fuzzy_engine', 'hybrid', 'string', 'デフォルトのファジィエンジン'),
                ('max_evaluation_results', '20', 'int', '最大評価結果表示数'),
                ('genetic_population_size', '50', 'int', '遺伝的アルゴリズムの個体数'),
                ('genetic_generations', '100', 'int', '遺伝的アルゴリズムの世代数'),
                ('mutation_rate', '0.1', 'float', '突然変異率'),
                ('crossover_rate', '0.8', 'float', '交叉率'),
                ('tournament_size', '5', 'int', 'トーナメント選択のサイズ'),
                ('max_tree_depth', '10', 'int', '決定木の最大深度'),
                ('enable_tracking', 'true', 'bool', '詳細追跡の有効化'),
                ('model_auto_save', 'true', 'bool', 'モデル自動保存'),
                ('evaluation_cache_size', '1000', 'int', '評価キャッシュサイズ'),
                ('system_version', '2.0.0', 'string', 'システムバージョン'),
            ]

            for key, value, config_type, description in default_configs:
                existing = SystemConfig.query.filter_by(config_key=key).first()
                if not existing:
                    config = SystemConfig(
                        config_key=key,
                        config_type=config_type,
                        description=description
                    )
                    config.set_value(value)
                    db.session.add(config)

            db.session.commit()
            print("✅ システム設定を初期化しました")

            print("🎉 拡張データベース初期化完了！")
            return True

        except Exception as e:
            print(f"❌ データベース初期化失敗: {e}")
            db.session.rollback()
            return False


def check_database_schema():
    """データベーススキーマの確認"""
    try:
        import sqlite3
        
        conn = sqlite3.connect('fdtlss.db')
        cursor = conn.cursor()
        
        # 全テーブル一覧取得
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        schema_info = {}
        
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            schema_info[table] = {
                'columns': [col[1] for col in columns],
                'column_count': len(columns)
            }
        
        conn.close()
        
        return {
            'tables': tables,
            'table_count': len(tables),
            'schema_details': schema_info
        }
        
    except Exception as e:
        print(f"⚠️ スキーマ確認エラー: {e}")
        return None


def migrate_lab_data_to_20_items():
    """既存の研究室データを20項目対応に移行"""
    try:
        import sqlite3
        
        conn = sqlite3.connect('fdtlss.db')
        cursor = conn.cursor()
        
        # 既存の研究室データ取得
        cursor.execute("SELECT id, name, professor, research_area, description FROM labs")
        existing_labs = cursor.fetchall()
        
        print(f"📊 既存研究室データ: {len(existing_labs)}件")
        
        # 各研究室に20項目のデフォルト値を設定
        for lab_id, name, professor, research_area, description in existing_labs:
            # ランダムな値を生成（研究室の特性に応じて調整可能）
            import random
            
            updates = {
                'research_field_match': round(random.uniform(6, 9), 1),
                'skill_development': round(random.uniform(5, 8), 1),
                'learning_pace': round(random.uniform(5, 8), 1),
                'difficulty_preference': round(random.uniform(6, 9), 1),
                'communication_style': round(random.uniform(5, 8), 1),
                'meeting_frequency': round(random.uniform(4, 7), 1),
                'lab_atmosphere': round(random.uniform(6, 9), 1),
                'innovation_risk': round(random.uniform(5, 8), 1),
                'methodology_preference': round(random.uniform(5, 8), 1),
                'interdisciplinary': round(random.uniform(5, 8), 1),
                'flexibility': round(random.uniform(6, 9), 1),
                'evening_weekend_work': round(random.uniform(3, 7), 1),
                'publication_opportunity': round(random.uniform(6, 9), 1),
                'financial_support': round(random.uniform(5, 8), 1),
                'lab_hierarchy': round(random.uniform(5, 8), 1),
                'core_time_flexibility': round(random.uniform(6, 9), 1),
            }
            
            # 各カラムを更新
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
    print("🚀 20項目対応マイグレーション開始")
    print("=" * 50)
    
    # 1. バックアップ作成
    print("1️⃣ データベースバックアップ作成...")
    backup_database()
    
    # 2. 現在のスキーマ確認
    print("\n2️⃣ 現在のデータベーススキーマ確認...")
    schema_info = check_database_schema()
    if schema_info:
        print(f"📊 既存テーブル: {schema_info['tables']}")
        if 'labs' in schema_info['schema_details']:
            lab_columns = schema_info['schema_details']['labs']['columns']
            print(f"📊 Labテーブルの現在のカラム数: {len(lab_columns)}")
    
    # 3. 新しいカラム追加
    print("\n3️⃣ 新しいカラムを追加中...")
    
    # 追加する新しいカラム（15項目）
    new_columns = [
        ('research_field_match', 'REAL DEFAULT 5.0'),
        ('skill_development', 'REAL DEFAULT 5.0'),
        ('learning_pace', 'REAL DEFAULT 5.0'),
        ('difficulty_preference', 'REAL DEFAULT 5.0'),
        ('communication_style', 'REAL DEFAULT 5.0'),
        ('meeting_frequency', 'REAL DEFAULT 5.0'),
        ('lab_atmosphere', 'REAL DEFAULT 5.0'),
        ('innovation_risk', 'REAL DEFAULT 5.0'),
        ('methodology_preference', 'REAL DEFAULT 5.0'),
        ('interdisciplinary', 'REAL DEFAULT 5.0'),
        ('flexibility', 'REAL DEFAULT 5.0'),
        ('evening_weekend_work', 'REAL DEFAULT 5.0'),
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
        
        added_count = 0
        
        for col_name, col_definition in new_columns:
            if col_name not in existing_columns:
                try:
                    cursor.execute(f"ALTER TABLE labs ADD COLUMN {col_name} {col_definition}")
                    print(f"  ✅ 追加: {col_name}")
                    added_count += 1
                except sqlite3.Error as e:
                    print(f"  ❌ エラー {col_name}: {e}")
        
        conn.commit()
        conn.close()
        
        print(f"\n🎉 {added_count}個の新しいカラムを追加しました")
        
    except Exception as e:
        print(f"❌ カラム追加エラー: {e}")
        return False
    
    # 4. 既存データの移行
    print("\n4️⃣ 既存データを20項目対応に移行中...")
    migrate_lab_data_to_20_items()
    
    # 5. 完了確認
    print("\n5️⃣ マイグレーション完了確認...")
    final_schema = check_database_schema()
    if final_schema and 'labs' in final_schema['schema_details']:
        final_column_count = final_schema['schema_details']['labs']['column_count']
        print(f"📊 マイグレーション後のLabテーブルカラム数: {final_column_count}")
        
        if final_column_count >= 25:  # 基本情報5 + 20項目 + メタデータ
            print("✅ 20項目マイグレーション成功！")
        else:
            print("⚠️ 一部のカラムが不足している可能性があります")
    
    print("\n🎉 マイグレーション完了！")
    print("次のステップ:")
    print("  1. python create_sample_labs.py を実行")
    print("  2. python app.py でサーバー起動")


if __name__ == '__main__':
    main()