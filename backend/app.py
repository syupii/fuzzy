# backend/app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
from models import (
    db, Lab, Evaluation, GeneticIndividual, DecisionPath,
    OptimizationRun, ModelRegistry, SystemConfig,
    get_system_config, set_system_config, DatabaseManager
)
from fuzzy_engine import HybridFuzzyEngine, FuzzyLogicEngine, create_fuzzy_engine
import uuid
import os
import time
from datetime import datetime, timedelta
import json


def create_app():
    app = Flask(__name__)

    # 設定
    app.config['SECRET_KEY'] = os.environ.get(
        'SECRET_KEY', 'development-secret-key')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get(
        'DATABASE_URL', 'sqlite:///fdtlss.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # 拡張機能初期化
    db.init_app(app)
    CORS(app, origins=["http://localhost:3000"])

    return app


app = create_app()

# ファジィエンジン初期化
try:
    # システム設定から エンジンタイプを取得
    with app.app_context():
        engine_type = get_system_config('default_fuzzy_engine', 'hybrid')

    fuzzy_engine = create_fuzzy_engine(engine_type)
    print(f"✅ {type(fuzzy_engine).__name__} を初期化しました")
except Exception as e:
    print(f"⚠️ ハイブリッドエンジン初期化失敗、シンプルエンジンを使用: {e}")
    fuzzy_engine = FuzzyLogicEngine()

# === 既存のAPIエンドポイント（完全保持） ===


@app.route('/api/health', methods=['GET'])
def health_check():
    """ヘルスチェック（拡張版）"""
    try:
        # データベース接続確認
        lab_count = Lab.query.count()
        evaluation_count = Evaluation.query.count()

        # エンジン情報
        engine_info = {}
        if hasattr(fuzzy_engine, 'get_engine_info'):
            engine_info = fuzzy_engine.get_engine_info()

        # データベース統計
        table_counts = DatabaseManager.get_table_counts()
        db_size = DatabaseManager.get_database_size()

        return jsonify({
            'status': 'healthy',
            'message': 'FDTLSS Backend API is running',
            'version': '2.0.0',
            'database': {
                'status': 'connected',
                'lab_count': lab_count,
                'evaluation_count': evaluation_count,
                'table_counts': table_counts,
                'size_info': db_size
            },
            'engine_info': engine_info,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': 'System health check failed',
            'error': str(e)
        }), 500


@app.route('/api/labs', methods=['GET'])
def get_labs():
    """研究室一覧取得（既存）"""
    try:
        labs = Lab.query.filter_by(is_active=True).all()
        return jsonify({
            'labs': [lab.to_dict() for lab in labs],
            'count': len(labs)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluate', methods=['POST'])
def evaluate_compatibility():
    """研究室適合度評価（拡張版）"""
    evaluation_start_time = time.time()

    try:
        user_prefs = request.get_json()

        # バリデーション（既存）
        required_fields = ['research_intensity', 'advisor_style',
                           'team_work', 'workload', 'theory_practice']
        for field in required_fields:
            if field not in user_prefs:
                return jsonify({'error': f'Missing required field: {field}'}), 400
            if not isinstance(user_prefs[field], (int, float)) or not (1 <= user_prefs[field] <= 10):
                return jsonify({'error': f'Invalid value for {field}: must be between 1 and 10'}), 400

        # セッションID生成
        session_id = str(uuid.uuid4())

        # 評価データ保存
        evaluation = Evaluation(
            session_id=session_id,
            **{field: user_prefs[field] for field in required_fields}
        )

        # 全研究室との適合度計算
        labs = Lab.query.filter_by(is_active=True).all()
        results = []

        for lab in labs:
            lab_features = {
                'research_intensity': lab.research_intensity,
                'advisor_style': lab.advisor_style,
                'team_work': lab.team_work,
                'workload': lab.workload,
                'theory_practice': lab.theory_practice
            }

            # 🆕 ハイブリッドエンジンで予測
            if hasattr(fuzzy_engine, 'predict_compatibility'):
                compatibility, explanation = fuzzy_engine.predict_compatibility(
                    user_prefs, lab_features)

                # 🆕 決定パス記録（遺伝的モデル使用時）
                if compatibility.get('prediction_method') == 'genetic_optimization':
                    decision_path = DecisionPath(
                        path_id=f"{session_id}_{lab.id}",
                        evaluation_id=evaluation.id,
                        model_type='genetic',
                        model_version='2.0',
                        model_id=compatibility.get(
                            'genetic_info', {}).get('individual_id', ''),
                        final_prediction=compatibility['overall_score'] / 100,
                        explanation_text=explanation,
                        prediction_time=time.time() - evaluation_start_time
                    )

                    # 決定詳細記録
                    if 'decision_path' in compatibility:
                        decision_path.set_decision_data(
                            compatibility['decision_path'],
                            compatibility.get('criterion_scores', {}),
                            [compatibility.get('confidence', 80) / 100]
                        )

                    db.session.add(decision_path)

            else:
                # 既存エンジンフォールバック
                compatibility = fuzzy_engine.fuzzy_inference(
                    user_prefs, lab_features)
                explanation = fuzzy_engine.generate_explanation(
                    compatibility, user_prefs, lab_features)

            results.append({
                'lab': lab.to_dict(),
                'compatibility': {
                    **compatibility,
                    'explanation': explanation
                }
            })

        # 適合度順でソート
        results.sort(key=lambda x: x['compatibility']
                     ['overall_score'], reverse=True)

        # 結果保存
        evaluation.set_results(results)
        db.session.add(evaluation)
        db.session.commit()

        # サマリー情報（拡張）
        summary = {
            'total_labs': len(results),
            'best_match': results[0]['lab']['name'] if results else None,
            'avg_score': round(sum(r['compatibility']['overall_score'] for r in results) / len(results), 2),
            'evaluation_id': evaluation.id,
            'session_id': session_id,
            'engine_used': results[0]['compatibility'].get('prediction_method', 'simple_fuzzy') if results else 'simple_fuzzy',
            'evaluation_time': time.time() - evaluation_start_time
        }

        # アルゴリズム情報（拡張）
        algorithm_info = fuzzy_engine.get_engine_info() if hasattr(fuzzy_engine, 'get_engine_info') else {
            'engine': 'Simple Fuzzy Logic',
            'version': '1.0'
        }

        return jsonify({
            'results': results,
            'summary': summary,
            'algorithm_info': algorithm_info
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/demo-data', methods=['GET'])
def get_demo_data():
    """デモ用データ生成（既存）"""
    import random

    demo_preferences = {
        'research_intensity': round(random.uniform(6, 9), 1),
        'advisor_style': round(random.uniform(4, 8), 1),
        'team_work': round(random.uniform(5, 9), 1),
        'workload': round(random.uniform(5, 8), 1),
        'theory_practice': round(random.uniform(6, 9), 1)
    }

    return jsonify({
        'demo_preferences': demo_preferences,
        'message': 'デモ用設定を生成しました'
    })

# === 🆕 新しいAPIエンドポイント ===


@app.route('/api/engine/info', methods=['GET'])
def get_engine_info():
    """エンジン情報取得"""
    try:
        if hasattr(fuzzy_engine, 'get_engine_info'):
            engine_info = fuzzy_engine.get_engine_info()
        else:
            engine_info = {
                'current_mode': 'simple',
                'genetic_model_loaded': False,
                'engine_type': type(fuzzy_engine).__name__
            }

        # 統計情報追加
        if hasattr(fuzzy_engine, 'get_model_statistics'):
            engine_info['statistics'] = fuzzy_engine.get_model_statistics()

        return jsonify(engine_info)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/engine/switch', methods=['POST'])
def switch_engine():
    """エンジン切り替え"""
    try:
        data = request.get_json()
        mode = data.get('mode', 'simple')  # 'simple' or 'genetic'

        if hasattr(fuzzy_engine, 'switch_mode'):
            success = fuzzy_engine.switch_mode(mode)
            if success:
                return jsonify({
                    'success': True,
                    'current_mode': mode,
                    'message': f'エンジンを{mode}モードに切り替えました'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': f'{mode}モードは利用できません'
                }), 400
        else:
            return jsonify({
                'success': False,
                'message': 'エンジン切り替え機能は利用できません'
            }), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/engine/reload', methods=['POST'])
def reload_genetic_model():
    """遺伝的モデル再読み込み"""
    try:
        if hasattr(fuzzy_engine, 'reload_genetic_model'):
            success = fuzzy_engine.reload_genetic_model()
            return jsonify({
                'success': success,
                'message': '遺伝的モデルの再読み込みが完了しました' if success else '遺伝的モデルの読み込みに失敗しました'
            })
        else:
            return jsonify({
                'success': False,
                'message': '遺伝的モデル再読み込み機能は利用できません'
            }), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/optimization/status', methods=['GET'])
def get_optimization_status():
    """最適化状況取得"""
    try:
        latest_run = OptimizationRun.query.order_by(
            OptimizationRun.created_at.desc()).first()

        if latest_run:
            return jsonify({
                'has_optimization': True,
                'run_info': latest_run.to_dict()
            })
        else:
            return jsonify({
                'has_optimization': False,
                'message': '最適化実行履歴がありません'
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/optimization/runs', methods=['GET'])
def get_optimization_runs():
    """最適化実行履歴一覧"""
    try:
        limit = request.args.get('limit', 10, type=int)
        offset = request.args.get('offset', 0, type=int)

        runs = OptimizationRun.query.order_by(
            OptimizationRun.created_at.desc()
        ).offset(offset).limit(limit).all()

        total_count = OptimizationRun.query.count()

        return jsonify({
            'runs': [run.to_dict() for run in runs],
            'pagination': {
                'total': total_count,
                'limit': limit,
                'offset': offset,
                'has_more': offset + limit < total_count
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/decision-paths/<int:evaluation_id>', methods=['GET'])
def get_decision_paths(evaluation_id):
    """決定パス詳細取得"""
    try:
        paths = DecisionPath.query.filter_by(evaluation_id=evaluation_id).all()

        results = []
        for path in paths:
            results.append(path.to_dict())

        return jsonify({
            'evaluation_id': evaluation_id,
            'decision_paths': results,
            'count': len(results)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/genetic/individuals', methods=['GET'])
def get_genetic_individuals():
    """遺伝的個体履歴取得"""
    try:
        # クエリパラメータ
        generation = request.args.get('generation', type=int)
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)

        query = GeneticIndividual.query

        if generation is not None:
            query = query.filter_by(generation=generation)

        individuals = query.order_by(
            GeneticIndividual.overall_fitness.desc()
        ).offset(offset).limit(limit).all()

        total_count = query.count()

        return jsonify({
            'individuals': [ind.to_dict() for ind in individuals],
            'pagination': {
                'total': total_count,
                'limit': limit,
                'offset': offset,
                'filter': {'generation': generation}
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """モデル一覧取得"""
    try:
        model_type = request.args.get('type')
        is_active = request.args.get('active', 'true').lower() == 'true'
        limit = request.args.get('limit', 20, type=int)

        query = ModelRegistry.query

        if model_type:
            query = query.filter_by(model_type=model_type)

        if is_active:
            query = query.filter_by(is_active=True)

        models = query.order_by(
            ModelRegistry.created_at.desc()
        ).limit(limit).all()

        return jsonify({
            'models': [model.to_dict() for model in models],
            'count': len(models),
            'filter': {
                'type': model_type,
                'active': is_active
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/<model_id>', methods=['GET'])
def get_model_detail(model_id):
    """モデル詳細取得"""
    try:
        model = ModelRegistry.query.filter_by(model_id=model_id).first()

        if not model:
            return jsonify({'error': 'Model not found'}), 404

        # 使用回数更新
        model.increment_usage()
        db.session.commit()

        return jsonify(model.to_dict())

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models/<model_id>/activate', methods=['POST'])
def activate_model(model_id):
    """モデルをアクティブ化"""
    try:
        model = ModelRegistry.query.filter_by(model_id=model_id).first()

        if not model:
            return jsonify({'error': 'Model not found'}), 404

        model.is_active = True
        db.session.commit()

        # ハイブリッドエンジンのモデル再読み込み
        global fuzzy_engine
        if hasattr(fuzzy_engine, 'reload_genetic_model'):
            fuzzy_engine.reload_genetic_model()

        return jsonify({
            'success': True,
            'message': f'Model {model_id} activated successfully'
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluations', methods=['GET'])
def get_evaluations():
    """評価履歴取得"""
    try:
        limit = request.args.get('limit', 20, type=int)
        offset = request.args.get('offset', 0, type=int)
        session_id = request.args.get('session_id')

        query = Evaluation.query

        if session_id:
            query = query.filter_by(session_id=session_id)

        evaluations = query.order_by(
            Evaluation.created_at.desc()
        ).offset(offset).limit(limit).all()

        total_count = query.count()

        return jsonify({
            'evaluations': [eval.to_dict() for eval in evaluations],
            'pagination': {
                'total': total_count,
                'limit': limit,
                'offset': offset
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/system/config', methods=['GET'])
def get_system_config_api():
    """システム設定取得"""
    try:
        configs = SystemConfig.query.all()

        return jsonify({
            'configs': [config.to_dict() for config in configs],
            'count': len(configs)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/system/config', methods=['POST'])
def update_system_config_api():
    """システム設定更新"""
    try:
        data = request.get_json()

        config_key = data.get('config_key')
        config_value = data.get('config_value')
        config_type = data.get('config_type', 'string')
        description = data.get('description', '')

        if not config_key:
            return jsonify({'error': 'config_key is required'}), 400

        success = set_system_config(
            config_key, config_value, config_type, description)

        if success:
            return jsonify({
                'success': True,
                'message': f'Configuration {config_key} updated successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to update configuration'
            }), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/system/statistics', methods=['GET'])
def get_system_statistics():
    """システム統計取得"""
    try:
        # データベース統計
        table_counts = DatabaseManager.get_table_counts()
        db_size = DatabaseManager.get_database_size()

        # 評価統計
        recent_evaluations = Evaluation.query.filter(
            Evaluation.created_at >= datetime.utcnow() - timedelta(days=7)
        ).count()

        # モデル統計
        active_models = ModelRegistry.query.filter_by(is_active=True).count()
        total_models = ModelRegistry.query.count()

        # 最新の最適化実行
        latest_optimization = OptimizationRun.query.order_by(
            OptimizationRun.created_at.desc()
        ).first()

        stats = {
            'database': {
                'table_counts': table_counts,
                'size_info': db_size
            },
            'activity': {
                'recent_evaluations': recent_evaluations,
                'active_models': active_models,
                'total_models': total_models
            },
            'latest_optimization': latest_optimization.to_dict() if latest_optimization else None,
            'system_uptime': 'N/A',  # 実際の実装では起動時間から計算
            'timestamp': datetime.utcnow().isoformat()
        }

        return jsonify(stats)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/system/cleanup', methods=['POST'])
def cleanup_system():
    """システム清理"""
    try:
        data = request.get_json() or {}
        days_threshold = data.get('days_threshold', 30)

        # データベース清理
        cleanup_count = DatabaseManager.cleanup_old_records(days_threshold)

        return jsonify({
            'success': True,
            'cleanup_count': cleanup_count,
            'message': f'{cleanup_count}件のレコードを清理しました'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/explain', methods=['POST'])
def predict_with_explanation():
    """説明付き予測"""
    try:
        data = request.get_json()
        user_preferences = data.get('user_preferences', {})
        lab_features = data.get('lab_features', {})
        model_id = data.get('model_id')  # 特定モデルを指定可能

        # バリデーション
        required_fields = ['research_intensity', 'advisor_style',
                           'team_work', 'workload', 'theory_practice']
        for field in required_fields:
            if field not in user_preferences:
                return jsonify({'error': f'Missing user preference: {field}'}), 400
            if field not in lab_features:
                return jsonify({'error': f'Missing lab feature: {field}'}), 400

        # 予測実行
        if hasattr(fuzzy_engine, 'predict_compatibility'):
            compatibility, explanation = fuzzy_engine.predict_compatibility(
                user_preferences, lab_features)

            # 詳細説明生成（遺伝的モデルの場合）
            detailed_explanation = None
            if compatibility.get('prediction_method') == 'genetic_optimization':
                try:
                    from explanation_engine import FuzzyExplanationEngine, NaturalLanguageGenerator
                    explanation_engine = FuzzyExplanationEngine()

                    lab_info = {'name': 'Target Lab'}
                    detailed_explanation = explanation_engine.generate_comprehensive_explanation(
                        compatibility, lab_info, user_preferences
                    )

                    formatted_explanation = NaturalLanguageGenerator.format_explanation_for_ui(
                        detailed_explanation, 'markdown'
                    )

                    return jsonify({
                        'prediction': compatibility,
                        'explanation': explanation,
                        'detailed_explanation': {
                            'structured': detailed_explanation.__dict__,
                            'formatted': formatted_explanation
                        }
                    })

                except ImportError:
                    pass  # 説明エンジンが利用できない場合は基本説明のみ
        else:
            # フォールバック
            compatibility = fuzzy_engine.fuzzy_inference(
                user_preferences, lab_features)
            explanation = fuzzy_engine.generate_explanation(
                compatibility, user_preferences, lab_features)

        return jsonify({
            'prediction': compatibility,
            'explanation': explanation,
            'detailed_explanation': None
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# エラーハンドラー


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # データベーステーブル作成
    with app.app_context():
        db.create_all()
        print("✅ データベーステーブルを確認・作成しました")

    print("🚀 FDTLSS Backend (Extended) starting...")
    print(f"🔧 Engine: {type(fuzzy_engine).__name__}")
    print("🌐 API Server: http://localhost:5000")
    print("\n📋 Available endpoints:")

    # 既存エンドポイント
    print("   === 既存エンドポイント ===")
    print("   GET  /api/health          - ヘルスチェック")
    print("   GET  /api/labs            - 研究室一覧")
    print("   POST /api/evaluate        - 適合度評価")
    print("   GET  /api/demo-data       - デモデータ")

    # 新エンドポイント
    print("   === 新エンドポイント ===")
    print("   GET  /api/engine/info     - エンジン情報")
    print("   POST /api/engine/switch   - エンジン切り替え")
    print("   POST /api/engine/reload   - モデル再読み込み")
    print("   GET  /api/optimization/status - 最適化状況")
    print("   GET  /api/optimization/runs   - 最適化履歴")
    print("   GET  /api/models          - モデル一覧")
    print("   GET  /api/genetic/individuals - 遺伝的個体履歴")
    print("   GET  /api/evaluations     - 評価履歴")
    print("   GET  /api/system/statistics - システム統計")
    print("   POST /api/predict/explain - 説明付き予測")

    app.run(debug=True, port=5000, host='0.0.0.0')
