# backend/models.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

# 🔥 20項目完全対応のLabモデル
class Lab(db.Model):
    """研究室モデル（20項目完全対応版）"""
    __tablename__ = 'labs'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    professor = db.Column(db.String(100), nullable=False)
    research_area = db.Column(db.String(500), nullable=False)
    description = db.Column(db.Text)

    # === 20項目のファジィ特徴量（1-10スケール） ===
    
    # 基本的な研究環境（6項目）
    research_intensity = db.Column(db.Float, default=5.0)    # 研究強度
    advisor_style = db.Column(db.Float, default=5.0)         # 指導スタイル
    team_work = db.Column(db.Float, default=5.0)             # チームワーク
    workload = db.Column(db.Float, default=5.0)              # ワークロード
    theory_practice = db.Column(db.Float, default=5.0)       # 理論/実践
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
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
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
                # 基本的な研究環境（6項目）
                'research_intensity': self.research_intensity,
                'advisor_style': self.advisor_style,
                'team_work': self.team_work,
                'workload': self.workload,
                'theory_practice': self.theory_practice,
                'research_field_match': self.research_field_match,
                
                # 学習・成長関連（3項目）
                'skill_development': self.skill_development,
                'learning_pace': self.learning_pace,
                'difficulty_preference': self.difficulty_preference,
                
                # コミュニケーション・環境関連（3項目）
                'communication_style': self.communication_style,
                'meeting_frequency': self.meeting_frequency,
                'lab_atmosphere': self.lab_atmosphere,
                
                # 研究アプローチ関連（3項目）
                'innovation_risk': self.innovation_risk,
                'methodology_preference': self.methodology_preference,
                'interdisciplinary': self.interdisciplinary,
                
                # 時間・ライフスタイル関連（2項目）
                'flexibility': self.flexibility,
                'evening_weekend_work': self.evening_weekend_work,
                
                # 調査結果に基づく追加項目（3項目）
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
    session_id = db.Column(db.String(100), nullable=True)

    # === 20項目のユーザー入力データ ===
    # 基本的な研究環境（6項目）
    research_intensity = db.Column(db.Float, nullable=False)
    advisor_style = db.Column(db.Float, nullable=False)
    team_work = db.Column(db.Float, nullable=False)
    workload = db.Column(db.Float, nullable=False)
    theory_practice = db.Column(db.Float, nullable=False)
    research_field_match = db.Column(db.Float, nullable=False)
    
    # 学習・成長関連（3項目）
    skill_development = db.Column(db.Float, nullable=False)
    learning_pace = db.Column(db.Float, nullable=False)
    difficulty_preference = db.Column(db.Float, nullable=False)
    
    # コミュニケーション・環境関連（3項目）
    communication_style = db.Column(db.Float, nullable=False)
    meeting_frequency = db.Column(db.Float, nullable=False)
    lab_atmosphere = db.Column(db.Float, nullable=False)
    
    # 研究アプローチ関連（3項目）
    innovation_risk = db.Column(db.Float, nullable=False)
    methodology_preference = db.Column(db.Float, nullable=False)
    interdisciplinary = db.Column(db.Float, nullable=False)
    
    # 時間・ライフスタイル関連（2項目）
    flexibility = db.Column(db.Float, nullable=False)
    evening_weekend_work = db.Column(db.Float, nullable=False)
    
    # 調査結果に基づく追加項目（3項目）
    publication_opportunity = db.Column(db.Float, nullable=False)
    financial_support = db.Column(db.Float, nullable=False)
    lab_hierarchy = db.Column(db.Float, nullable=False)
    core_time_flexibility = db.Column(db.Float, nullable=False)

    # 結果データ
    user_preferences = db.Column(db.Text)  # 完全なユーザー設定JSON
    evaluation_count = db.Column(db.Integer)  # 評価した研究室数
    avg_score = db.Column(db.Float)        # 平均適合度スコア
    best_lab_id = db.Column(db.Integer)    # 最高適合度の研究室ID
    engine_used = db.Column(db.String(100)) # 使用したエンジン

    # メタデータ
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

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


# === 既存の他のテーブル（遺伝的アルゴリズム用等）===

class GeneticIndividual(db.Model):
    """遺伝的アルゴリズムの個体記録"""
    __tablename__ = 'genetic_individuals'

    id = db.Column(db.Integer, primary_key=True)
    individual_id = db.Column(db.String(100), nullable=False, unique=True)
    generation = db.Column(db.Integer, nullable=False)
    genome_data = db.Column(db.Text)

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
        self.genome_data = json.dumps(genome, ensure_ascii=False, default=str)

    def get_genome_data(self):
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
            'created_at': self.created_at.isoformat()
        }


class DecisionPath(db.Model):
    """決定パス記録"""
    __tablename__ = 'decision_paths'

    id = db.Column(db.Integer, primary_key=True)
    evaluation_id = db.Column(db.Integer, db.ForeignKey('evaluations.id'))
    step_order = db.Column(db.Integer)
    criterion = db.Column(db.String(100))
    threshold = db.Column(db.Float)
    user_value = db.Column(db.Float)
    lab_value = db.Column(db.Float)
    decision_result = db.Column(db.String(50))
    criterion_weight = db.Column(db.Float)
    confidence = db.Column(db.Float)
    rule_explanation = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'evaluation_id': self.evaluation_id,
            'step_order': self.step_order,
            'criterion': self.criterion,
            'threshold': self.threshold,
            'user_value': self.user_value,
            'lab_value': self.lab_value,
            'decision_result': self.decision_result,
            'criterion_weight': self.criterion_weight,
            'confidence': self.confidence,
            'explanation': self.rule_explanation,
            'created_at': self.created_at.isoformat()
        }


class OptimizationRun(db.Model):
    """最適化実行記録"""
    __tablename__ = 'optimization_runs'

    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.String(100), nullable=False, unique=True)
    population_size = db.Column(db.Integer)
    generations = db.Column(db.Integer)
    mutation_rate = db.Column(db.Float)
    crossover_rate = db.Column(db.Float)
    max_depth = db.Column(db.Integer)
    tournament_size = db.Column(db.Integer)
    training_samples = db.Column(db.Integer)
    test_samples = db.Column(db.Integer)
    feature_names = db.Column(db.Text)
    target_column = db.Column(db.String(100))
    best_fitness = db.Column(db.Float)
    best_individual_id = db.Column(db.String(100))
    convergence_generation = db.Column(db.Integer)
    final_diversity = db.Column(db.Float)
    fitness_history = db.Column(db.Text)
    diversity_history = db.Column(db.Text)
    execution_time = db.Column(db.Float)
    status = db.Column(db.String(20), default='running')
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)

    def to_dict(self):
        return {
            'id': self.id,
            'run_id': self.run_id,
            'status': self.status,
            'best_fitness': self.best_fitness,
            'execution_time': self.execution_time,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


class ModelRegistry(db.Model):
    """モデル登録簿"""
    __tablename__ = 'model_registry'

    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.String(100), nullable=False, unique=True)
    model_name = db.Column(db.String(200))
    model_type = db.Column(db.String(50))
    version = db.Column(db.String(50))
    model_filepath = db.Column(db.String(500))
    result_filepath = db.Column(db.String(500))
    file_size_bytes = db.Column(db.BigInteger)
    checksum = db.Column(db.String(100))
    best_fitness = db.Column(db.Float)
    model_complexity = db.Column(db.Integer)
    validation_accuracy = db.Column(db.Float)
    test_accuracy = db.Column(db.Float)
    usage_count = db.Column(db.Integer, default=0)
    last_used_at = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    is_production_ready = db.Column(db.Boolean, default=False)
    description = db.Column(db.Text)
    tags = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'model_id': self.model_id,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat()
        }


class SystemConfig(db.Model):
    """システム設定"""
    __tablename__ = 'system_config'

    id = db.Column(db.Integer, primary_key=True)
    config_key = db.Column(db.String(100), nullable=False, unique=True)
    config_value = db.Column(db.Text)
    config_type = db.Column(db.String(20), default='string')
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

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