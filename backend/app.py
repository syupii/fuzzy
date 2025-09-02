# app_fixed.py
# -*- coding: utf-8 -*-
"""
APIサーバー（インポートエラー修正版）
HybridFuzzyEngineFixedに対応
"""

import os
import sys
import time
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS

# Windows文字エンコーディング設定
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# プロジェクトパス追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# モデルとエンジンのインポート
from models import db, Lab, Evaluation, create_app

# ファジィエンジンの動的インポート
try:
    # Windows互換版を優先
    from fuzzy_engine import HybridFuzzyEngineFixed as HybridFuzzyEngine
    print("[IMPORT] HybridFuzzyEngineFixed を使用")
except ImportError:
    try:
        # 代替インポート
        from fuzzy_engine import fuzzy_engine as HybridFuzzyEngine
        print("[IMPORT] fuzzy_engine インスタンスを使用")
    except ImportError:
        # 最終フォールバック
        print("[WARNING] HybridFuzzyEngineが見つかりません。基本エンジンを作成します")
        
        class BasicFuzzyEngine:
            """基本ファジィエンジン（フォールバック）"""
            
            def __init__(self):
                self.current_mode = 'basic'
                self.genetic_model_loaded = False
                
            def predict_compatibility(self, user_prefs, lab_features):
                """基本適合度予測"""
                
                criteria = ['research_intensity', 'advisor_style', 'team_work', 'workload', 'theory_practice']
                weights = [0.25, 0.20, 0.20, 0.15, 0.20]
                
                similarities = []
                for criterion in criteria:
                    user_val = user_prefs.get(criterion, 5.0)
                    lab_val = lab_features.get(criterion, 5.0)
                    similarity = 1.0 - abs(user_val - lab_val) / 10.0
                    similarities.append(max(0.0, similarity))
                
                overall_score = sum(w * s for w, s in zip(weights, similarities))
                
                result = {
                    'overall_score': overall_score * 100,
                    'confidence': 75.0,
                    'prediction_method': 'basic_fuzzy',
                    'criterion_scores': {}
                }
                
                # 基準別スコア
                for i, criterion in enumerate(criteria):
                    user_val = user_prefs.get(criterion, 5.0)
                    lab_val = lab_features.get(criterion, 5.0)
                    similarity = similarities[i]
                    weight = weights[i]
                    
                    result['criterion_scores'][criterion] = {
                        'similarity': similarity,
                        'weighted_score': similarity * weight * 100,
                        'user_preference': user_val,
                        'lab_feature': lab_val,
                        'weight': weight
                    }
                
                explanation = f"基本ファジィ論理による予測: {overall_score:.1%}"
                
                return result, explanation
        
        HybridFuzzyEngine = BasicFuzzyEngine

# Flaskアプリ作成
app = create_app()

# CORS設定
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# ファジィエンジン初期化
try:
    if isinstance(HybridFuzzyEngine, type):
        # クラスの場合はインスタンス化
        fuzzy_engine = HybridFuzzyEngine()
    else:
        # 既にインスタンスの場合はそのまま使用
        fuzzy_engine = HybridFuzzyEngine
        
    print(f"[ENGINE] ファジィエンジン初期化完了: {type(fuzzy_engine).__name__}")
    
    # エンジン情報表示
    if hasattr(fuzzy_engine, 'get_engine_info'):
        info = fuzzy_engine.get_engine_info()
        print(f"[ENGINE] モード: {info.get('current_mode', 'unknown')}")
        print(f"[ENGINE] 遺伝的モデル: {'OK' if info.get('genetic_model_loaded', False) else 'NG'}")
    
except Exception as e:
    print(f"[ERROR] エンジン初期化失敗: {e}")
    # 最終フォールバック
    fuzzy_engine = None

# APIエンドポイント
@app.route('/api/health', methods=['GET'])
def health_check():
    """ヘルスチェック"""
    try:
        # データベース接続確認
        lab_count = Lab.query.count()
        
        # エンジン状態確認
        engine_status = "active" if fuzzy_engine else "unavailable"
        engine_type = type(fuzzy_engine).__name__ if fuzzy_engine else "None"
        
        return jsonify({
            'status': 'healthy',
            'timestamp': time.time(),
            'database': {
                'connected': True,
                'lab_count': lab_count
            },
            'engine': {
                'status': engine_status,
                'type': engine_type,
                'mode': getattr(fuzzy_engine, 'current_mode', 'unknown') if fuzzy_engine else None,
                'genetic_loaded': getattr(fuzzy_engine, 'genetic_model_loaded', False) if fuzzy_engine else False
            },
            'version': '2.0'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/labs', methods=['GET'])
def get_labs():
    """研究室一覧取得"""
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
    """研究室適合度評価"""
    evaluation_start_time = time.time()
    
    try:
        user_prefs = request.get_json()
        
        # バリデーション
        required_fields = ['research_intensity', 'advisor_style', 'team_work', 'workload', 'theory_practice']
        for field in required_fields:
            if field not in user_prefs:
                return jsonify({'error': f'Missing required field: {field}'}), 400
            if not isinstance(user_prefs[field], (int, float)) or not (1 <= user_prefs[field] <= 10):
                return jsonify({'error': f'Invalid value for {field}: must be between 1 and 10'}), 400
        
        # セッションID生成
        session_id = str(uuid.uuid4())
        
        # 評価データ保存
        try:
            evaluation = Evaluation(
                session_id=session_id,
                **{field: user_prefs[field] for field in required_fields}
            )
            db.session.add(evaluation)
            db.session.commit()
        except Exception as e:
            print(f"[WARNING] 評価データ保存失敗: {e}")
            db.session.rollback()
        
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
            
            # ファジィエンジンで予測
            if fuzzy_engine:
                try:
                    compatibility, explanation = fuzzy_engine.predict_compatibility(
                        user_prefs, lab_features)
                except Exception as e:
                    print(f"[ERROR] 予測エラー: {e}")
                    # フォールバック予測
                    compatibility = {
                        'overall_score': 50.0,
                        'confidence': 60.0,
                        'prediction_method': 'fallback_error',
                        'criterion_scores': {}
                    }
                    explanation = "エラーが発生したため、フォールバック予測を使用"
            else:
                # エンジンなしフォールバック
                compatibility = {
                    'overall_score': 50.0,
                    'confidence': 50.0,
                    'prediction_method': 'no_engine',
                    'criterion_scores': {}
                }
                explanation = "ファジィエンジンが利用できません"
            
            # 結果追加
            results.append({
                'lab': lab.to_dict(),
                'compatibility': compatibility
            })
        
        # 結果をスコア順でソート
        results.sort(key=lambda x: x['compatibility']['overall_score'], reverse=True)
        
        # サマリー作成
        if results:
            best_match = results[0]['lab']['name']
            avg_score = sum(r['compatibility']['overall_score'] for r in results) / len(results)
        else:
            best_match = "該当なし"
            avg_score = 0.0
        
        summary = {
            'total_labs': len(results),
            'best_match': best_match,
            'avg_score': avg_score,
            'evaluation_id': getattr(evaluation, 'id', None) if 'evaluation' in locals() else None,
            'session_id': session_id
        }
        
        # アルゴリズム情報
        algorithm_info = {
            'engine': type(fuzzy_engine).__name__ if fuzzy_engine else "None",
            'mode': getattr(fuzzy_engine, 'current_mode', 'unknown') if fuzzy_engine else 'none',
            'genetic_loaded': getattr(fuzzy_engine, 'genetic_model_loaded', False) if fuzzy_engine else False,
            'processing_time': time.time() - evaluation_start_time
        }
        
        return jsonify({
            'results': results,
            'summary': summary,
            'algorithm_info': algorithm_info
        })
        
    except Exception as e:
        print(f"[ERROR] 評価処理エラー: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/demo-data', methods=['GET'])
def get_demo_data():
    """デモデータ取得"""
    demo_preferences = {
        'research_intensity': 7.5,
        'advisor_style': 6.0,
        'team_work': 7.0,
        'workload': 6.5,
        'theory_practice': 8.0
    }
    
    return jsonify({
        'demo_preferences': demo_preferences,
        'message': 'これはデモ用のサンプル設定です。自由に変更してください。'
    })

@app.route('/api/engine/info', methods=['GET'])
def get_engine_info():
    """エンジン情報取得"""
    if not fuzzy_engine:
        return jsonify({'error': 'エンジンが利用できません'}), 503
    
    try:
        if hasattr(fuzzy_engine, 'get_engine_info'):
            info = fuzzy_engine.get_engine_info()
        else:
            info = {
                'current_mode': getattr(fuzzy_engine, 'current_mode', 'unknown'),
                'genetic_model_loaded': getattr(fuzzy_engine, 'genetic_model_loaded', False),
                'engine_type': type(fuzzy_engine).__name__
            }
        
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# エラーハンドラー
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    if 'db' in globals():
        db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

def main():
    """メイン実行"""
    print("=" * 60)
    print("[SERVER] FDTLSS APIサーバー (修正版) 起動中...")
    print("=" * 60)
    
    # データベース初期化
    with app.app_context():
        try:
            db.create_all()
            print("[DB] データベーステーブル確認・作成完了")
        except Exception as e:
            print(f"[ERROR] データベース初期化失敗: {e}")
    
    # エンジン状態表示
    if fuzzy_engine:
        print(f"[ENGINE] ファジィエンジン: {type(fuzzy_engine).__name__}")
        if hasattr(fuzzy_engine, 'current_mode'):
            print(f"[ENGINE] モード: {fuzzy_engine.current_mode}")
        if hasattr(fuzzy_engine, 'genetic_model_loaded'):
            print(f"[ENGINE] 遺伝的モデル: {'✅' if fuzzy_engine.genetic_model_loaded else '❌'}")
    else:
        print("[ENGINE] ファジィエンジン: ❌ 利用不可")
    
    # APIエンドポイント一覧
    print(f"\n[API] 利用可能なエンドポイント:")
    print(f"   GET  /api/health          - ヘルスチェック")
    print(f"   GET  /api/labs            - 研究室一覧")
    print(f"   POST /api/evaluate        - 適合度評価")
    print(f"   GET  /api/demo-data       - デモデータ")
    print(f"   GET  /api/engine/info     - エンジン情報")
    
    print(f"\n[SERVER] サーバー起動: http://localhost:5000")
    print(f"[INFO] フロントエンドと連携するにはCORSが有効です")
    print(f"[INFO] Ctrl+C で停止")
    
    # サーバー起動
    try:
        app.run(debug=False, port=5000, host='0.0.0.0', use_reloader=False)
    except KeyboardInterrupt:
        print(f"\n[SHUTDOWN] サーバーを停止しました")
    except Exception as e:
        print(f"[ERROR] サーバー起動エラー: {e}")

if __name__ == '__main__':
    main()