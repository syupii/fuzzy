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
import random


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
        return jsonify([lab.to_dict() for lab in labs])
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/evaluate', methods=['POST'])
def evaluate_compatibility():
    """適合度評価（20項目対応版）"""
    try:
        # リクエストデータ取得
        user_preferences = request.get_json()
        
        if not user_preferences:
            return jsonify({'error': 'No preferences provided'}), 400

        # 20項目の必須チェック
        required_fields = [
            # 既存項目（6項目）
            'research_intensity', 'advisor_style', 'team_work', 
            'workload', 'theory_practice', 'research_field_match',
            
            # 学習・成長関連（3項目）
            'skill_development', 'learning_pace', 'difficulty_preference',
            
            # コミュニケーション・環境関連（3項目）
            'communication_style', 'meeting_frequency', 'lab_atmosphere',
            
            # 研究アプローチ関連（3項目）
            'innovation_risk', 'methodology_preference', 'interdisciplinary',
            
            # 時間・ライフスタイル関連（2項目）
            'flexibility', 'evening_weekend_work',
            
            # 調査結果に基づく追加項目（3項目）
            'publication_opportunity', 'financial_support', 'lab_hierarchy', 'core_time_flexibility'
        ]

        # 不足項目チェック
        missing_fields = [field for field in required_fields if field not in user_preferences]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'missing_count': len(missing_fields),
                'total_required': len(required_fields)
            }), 400

        print(f"🎯 20項目評価開始: {list(user_preferences.keys())}")

        # 研究室取得
        labs = Lab.query.filter_by(is_active=True).all()
        if not labs:
            return jsonify({'error': 'No active labs found'}), 404

        print(f"📊 対象研究室数: {len(labs)}")

        # 各研究室との適合度計算
        results = []
        session_id = str(uuid.uuid4())
        
        for lab in labs:
            try:
                # Lab.featuresが20項目対応していることを確認
                lab_features = lab.to_dict().get('features', {})
                
                # 欠損項目の補完（デフォルト値）
                for field in required_fields:
                    if field not in lab_features:
                        lab_features[field] = 5.0  # 中央値で補完

                # ファジィエンジンでの適合度計算
                if hasattr(fuzzy_engine, 'evaluate_extended_compatibility'):
                    # 20項目対応の拡張評価
                    compatibility = fuzzy_engine.evaluate_extended_compatibility(
                        user_preferences, lab_features
                    )
                else:
                    # 従来の評価方法
                    compatibility = fuzzy_engine.fuzzy_inference(
                        user_preferences, lab_features
                    )

                results.append({
                    'lab': lab.to_dict(),
                    'compatibility': compatibility
                })

            except Exception as lab_error:
                print(f"⚠️ 研究室 {lab.name} の評価中にエラー: {lab_error}")
                # エラーが発生した研究室はスキップ
                continue

        # 結果をソート（適合度の高い順）
        results.sort(key=lambda x: x['compatibility'].get('overall_score', 0), reverse=True)

        # 評価履歴をデータベースに保存
        try:
            evaluation = Evaluation(
                user_preferences=json.dumps(user_preferences, ensure_ascii=False),
                session_id=session_id,
                evaluation_count=len(results),
                avg_score=sum(r['compatibility'].get('overall_score', 0) for r in results) / len(results) if results else 0,
                best_lab_id=results[0]['lab']['id'] if results else None,
                engine_used=type(fuzzy_engine).__name__
            )
            db.session.add(evaluation)
            db.session.commit()
            evaluation_id = evaluation.id
        except Exception as db_error:
            print(f"⚠️ 評価履歴保存エラー: {db_error}")
            evaluation_id = None

        # レスポンス作成
        summary = {
            'total_labs': len(results),
            'best_match': results[0]['lab']['name'] if results else 'なし',
            'avg_score': sum(r['compatibility'].get('overall_score', 0) for r in results) / len(results) if results else 0,
            'evaluation_id': evaluation_id,
            'session_id': session_id,
            'engine_used': type(fuzzy_engine).__name__
        }

        algorithm_info = {
            'engine': type(fuzzy_engine).__name__,
            'current_mode': getattr(fuzzy_engine, 'current_mode', 'unknown'),
            'genetic_model_loaded': hasattr(fuzzy_engine, 'genetic_model') and 
                                  getattr(fuzzy_engine, 'genetic_model', None) is not None
        }

        print(f"✅ 評価完了: {len(results)}件, 平均スコア: {summary['avg_score']:.2f}")

        return jsonify({
            'results': results,
            'summary': summary,
            'algorithm_info': algorithm_info
        })

    except Exception as e:
        print(f"❌ 評価エラー: {str(e)}")
        return jsonify({
            'error': 'Evaluation failed',
            'details': str(e)
        }), 500


@app.route('/api/demo-data', methods=['GET'])
def get_demo_data():
    """デモ用データ生成（20項目対応版）"""
    try:
        # 20項目すべてのデモデータ生成
        demo_preferences = {
            # 既存項目（6項目）
            'research_intensity': round(random.uniform(6, 9), 1),
            'advisor_style': round(random.uniform(4, 8), 1),
            'team_work': round(random.uniform(5, 9), 1),
            'workload': round(random.uniform(5, 8), 1),
            'theory_practice': round(random.uniform(6, 9), 1),
            'research_field_match': round(random.uniform(7, 9.5), 1),
            
            # 学習・成長関連（3項目）
            'skill_development': round(random.uniform(5, 8.5), 1),
            'learning_pace': round(random.uniform(5.5, 8), 1),
            'difficulty_preference': round(random.uniform(6, 9), 1),
            
            # コミュニケーション・環境関連（3項目）
            'communication_style': round(random.uniform(5, 8.5), 1),
            'meeting_frequency': round(random.uniform(4, 7.5), 1),
            'lab_atmosphere': round(random.uniform(6, 9), 1),
            
            # 研究アプローチ関連（3項目）
            'innovation_risk': round(random.uniform(5.5, 8.5), 1),
            'methodology_preference': round(random.uniform(5, 8), 1),
            'interdisciplinary': round(random.uniform(5.5, 8.5), 1),
            
            # 時間・ライフスタイル関連（2項目）
            'flexibility': round(random.uniform(6, 9), 1),
            'evening_weekend_work': round(random.uniform(3, 7), 1),
            
            # 調査結果に基づく追加項目（3項目）
            'publication_opportunity': round(random.uniform(7, 9.5), 1),
            'financial_support': round(random.uniform(6.5, 9), 1),
            'lab_hierarchy': round(random.uniform(5, 8.5), 1),
            'core_time_flexibility': round(random.uniform(6.5, 9), 1),
        }

        # 推奨研究分野も生成（オプション）
        suggested_fields = [
            'ai', 'data_science', 'software_engineering', 'robotics'
        ]
        random.shuffle(suggested_fields)
        suggested_fields = suggested_fields[:3]  # 3つランダム選択

        print(f"🎲 20項目デモデータ生成: {len(demo_preferences)}項目")

        return jsonify({
            'demo_preferences': demo_preferences,
            'suggested_fields': suggested_fields,
            'message': '20項目対応のデモ用設定を生成しました'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
            'evaluations': {
                'total': Evaluation.query.count(),
                'recent_week': recent_evaluations
            },
            'models': {
                'total': total_models,
                'active': active_models
            },
            'optimization': {
                'has_runs': latest_optimization is not None,
                'latest_run': latest_optimization.to_dict() if latest_optimization else None
            },
            'engine': {
                'type': type(fuzzy_engine).__name__,
                'mode': getattr(fuzzy_engine, 'current_mode', 'unknown')
            }
        }

        return jsonify(stats)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/explain', methods=['POST'])
def predict_with_explanation():
    """説明付き予測"""
    try:
        data = request.get_json()
        user_preferences = data.get('user_preferences', {})
        lab_features = data.get('lab_features', {})
        model_id = data.get('model_id')

        if not user_preferences or not lab_features:
            return jsonify({'error': 'user_preferences and lab_features are required'}), 400

        # 遺伝的モデルを使った予測＋説明生成を試行
        if hasattr(fuzzy_engine, 'predict_with_explanation'):
            try:
                result = fuzzy_engine.predict_with_explanation(
                    user_preferences, lab_features, model_id)

                # 自然言語での説明生成を試行
                try:
                    from explainable_ai import ExplainableAIEngine, NaturalLanguageGenerator
                    
                    explain_engine = ExplainableAIEngine(fuzzy_engine)
                    detailed_explanation = explain_engine.generate_detailed_explanation(
                        result, user_preferences, lab_features)
                    
                    # UIフレンドリーな説明フォーマット
                    formatted_explanation = NaturalLanguageGenerator.format_explanation_for_ui(
                        detailed_explanation, 'markdown'
                    )

                    return jsonify({
                        'prediction': result,
                        'detailed_explanation': {
                            'structured': detailed_explanation.__dict__,
                            'formatted': formatted_explanation
                        }
                    })

                except ImportError:
                    pass  # 説明エンジンが利用できない場合は基本説明のみ
                
                return jsonify({
                    'prediction': result,
                    'explanation': 'Basic prediction completed',
                    'detailed_explanation': None
                })
            except Exception as prediction_error:
                print(f"⚠️ 遺伝的予測エラー: {prediction_error}")
                # フォールバックに進む
        
        # フォールバック: 基本的なファジィ推論
        compatibility = fuzzy_engine.fuzzy_inference(
            user_preferences, lab_features)
        
        if hasattr(fuzzy_engine, 'generate_explanation'):
            explanation = fuzzy_engine.generate_explanation(
                compatibility, user_preferences, lab_features)
        else:
            explanation = 'Basic compatibility calculated'

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
    print("   POST /api/evaluate        - 適合度評価（20項目対応）")
    print("   GET  /api/demo-data       - デモデータ（20項目対応）")

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