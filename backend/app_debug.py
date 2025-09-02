# app_debug.py
# -*- coding: utf-8 -*-
"""
デバッグ版APIサーバー（問題特定用）
インポート処理を段階的に実行して問題箇所を特定
"""

import os
import sys
import time
import uuid
from datetime import datetime

print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] スクリプト開始")

# Windows エンコーディング設定（最小限）
if sys.platform.startswith('win'):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] Windows環境検出")
    try:
        import io
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] UTF-8エンコーディング設定完了")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [WARN] エンコーディング設定失敗: {e}")

# プロジェクトパス追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] パス設定完了")

# 段階1: Flask基本インポート
print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] === 段階1: Flask基本インポート ===")
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [SUCCESS] Flask基本インポート完了")
except Exception as e:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [ERROR] Flask基本インポート失敗: {e}")
    sys.exit(1)

# 段階2: データベースモデルインポート
print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] === 段階2: データベースモデルインポート ===")
try:
    from models import db, Lab, Evaluation, create_app
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [SUCCESS] ✅ データベースモデル読み込み完了")
except Exception as e:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [ERROR] データベースモデル読み込み失敗: {e}")
    import traceback
    print(traceback.format_exc())
    sys.exit(1)

# 段階3: ファジィエンジンインポート（詳細デバッグ）
print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] === 段階3: ファジィエンジンインポート ===")
fuzzy_engine = None
engine_type = "none"

# インポート試行順序（詳細ログ付き）
import_attempts = [
    ("fuzzy_engine", "HybridFuzzyEngineFixed"),
    ("fuzzy_engine", "fuzzy_engine"),
    ("fuzzy_engine_windows_fix", "HybridFuzzyEngineFixed"),
]

for module_name, target_name in import_attempts:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] 試行: {module_name}.{target_name}")
    
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] __import__開始: {module_name}")
        module = __import__(module_name)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] __import__成功: {module_name}")
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] 属性確認: {target_name}")
        if hasattr(module, target_name):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] 属性発見: {target_name}")
            
            target = getattr(module, target_name)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] 属性取得成功: {type(target)}")
            
            # インスタンス化試行
            if isinstance(target, type):
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] クラスインスタンス化開始")
                fuzzy_engine = target()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] クラスインスタンス化完了")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] インスタンス使用")
                fuzzy_engine = target
                
            engine_type = f"{module_name}.{target_name}"
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [SUCCESS] ✅ ファジィエンジン読み込み完了: {engine_type}")
            break
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] 属性なし: {target_name}")
            
    except ImportError as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] インポートエラー {module_name}: {e}")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [ERROR] 予期しないエラー {module_name}: {e}")
        import traceback
        print(traceback.format_exc())

# フォールバックエンジン作成
if fuzzy_engine is None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] === フォールバックエンジン作成 ===")
    
    class DebugFallbackEngine:
        def __init__(self):
            self.current_mode = 'debug_fallback'
            self.genetic_model_loaded = False
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] デバッグフォールバックエンジン初期化")
            
        def predict_compatibility(self, user_prefs, lab_features):
            """デバッグ用基本予測"""
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] 予測実行開始")
            
            criteria = ['research_intensity', 'advisor_style', 'team_work', 'workload', 'theory_practice']
            total_diff = 0.0
            
            for criterion in criteria:
                user_val = user_prefs.get(criterion, 5.0)
                lab_val = lab_features.get(criterion, 5.0)
                total_diff += abs(user_val - lab_val)
            
            # 簡単な適合度計算
            max_diff = len(criteria) * 10.0
            similarity = 1.0 - (total_diff / max_diff)
            overall_score = max(0.0, min(100.0, similarity * 100))
            
            result = {
                'overall_score': overall_score,
                'confidence': 60.0,
                'prediction_method': 'debug_fallback',
                'criterion_scores': {}
            }
            
            explanation = f"デバッグフォールバック予測: {overall_score:.1f}%"
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] 予測実行完了: {overall_score:.1f}%")
            return result, explanation
        
        def get_engine_info(self):
            return {
                'current_mode': self.current_mode,
                'engine_type': 'DebugFallbackEngine',
                'genetic_model_loaded': self.genetic_model_loaded
            }
    
    fuzzy_engine = DebugFallbackEngine()
    engine_type = "DebugFallbackEngine"
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [SUCCESS] ✅ デバッグフォールバックエンジン作成完了")

# 段階4: Flaskアプリ作成
print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] === 段階4: Flaskアプリ作成 ===")
try:
    app = create_app()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [SUCCESS] ✅ Flaskアプリ作成完了")
except Exception as e:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [ERROR] Flaskアプリ作成失敗: {e}")
    import traceback
    print(traceback.format_exc())
    sys.exit(1)

# 段階5: CORS設定
print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] === 段階5: CORS設定 ===")
try:
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
            "methods": ["GET", "POST", "PUT", "DELETE"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [SUCCESS] ✅ CORS設定完了")
except Exception as e:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [ERROR] CORS設定失敗: {e}")

# 段階6: APIエンドポイント定義
print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] === 段階6: APIエンドポイント定義 ===")

@app.route('/api/health', methods=['GET'])
def health_check():
    """デバッグ用ヘルスチェック"""
    try:
        lab_count = Lab.query.count()
        
        response = {
            'status': 'healthy',
            'message': f'デバッグAPIサーバー正常動作中',
            'timestamp': time.time(),
            'version': 'debug-1.0',
            'database': {
                'connected': True,
                'lab_count': lab_count
            },
            'engine': {
                'type': engine_type,
                'loaded': fuzzy_engine is not None
            }
        }
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] ヘルスチェック実行完了")
        return jsonify(response)
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [ERROR] ヘルスチェックエラー: {e}")
        return jsonify({
            'status': 'error',
            'message': f'ヘルスチェック失敗: {str(e)}'
        }), 500

@app.route('/api/labs', methods=['GET'])
def get_labs():
    """研究室一覧取得（デバッグ版）"""
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] 研究室一覧取得開始")
        
        labs = Lab.query.filter_by(is_active=True).all()
        labs_data = [lab.to_dict() for lab in labs]
        
        response = {
            'labs': labs_data,
            'count': len(labs_data)
        }
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] 研究室一覧取得完了: {len(labs_data)}件")
        return jsonify(response)
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [ERROR] 研究室一覧取得エラー: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/evaluate', methods=['POST'])
def evaluate_compatibility():
    """適合度評価（デバッグ版）"""
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] 適合度評価開始")
        
        user_prefs = request.get_json()
        if not user_prefs:
            return jsonify({'error': '評価データが必要です'}), 400
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] ユーザー設定受信: {len(user_prefs)}項目")
        
        # 基本項目チェック
        required_fields = ['research_intensity', 'advisor_style', 'team_work', 'workload', 'theory_practice']
        for field in required_fields:
            if field not in user_prefs:
                user_prefs[field] = 5.0
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] デフォルト値設定: {field} = 5.0")
        
        # 研究室取得
        labs = Lab.query.filter_by(is_active=True).all()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] 対象研究室数: {len(labs)}")
        
        results = []
        
        for i, lab in enumerate(labs):
            try:
                lab_features = {
                    'research_intensity': lab.research_intensity,
                    'advisor_style': lab.advisor_style,
                    'team_work': lab.team_work,
                    'workload': lab.workload,
                    'theory_practice': lab.theory_practice
                }
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] 評価中: {lab.name} ({i+1}/{len(labs)})")
                
                compatibility, explanation = fuzzy_engine.predict_compatibility(user_prefs, lab_features)
                
                results.append({
                    'lab': lab.to_dict(),
                    'compatibility': compatibility
                })
                
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] [ERROR] 研究室評価エラー {lab.name}: {e}")
                
                # エラー時のフォールバック
                results.append({
                    'lab': lab.to_dict(),
                    'compatibility': {
                        'overall_score': 50.0,
                        'confidence': 30.0,
                        'prediction_method': 'error_fallback',
                        'error': str(e)
                    }
                })
        
        # 結果ソート
        results.sort(key=lambda x: x['compatibility']['overall_score'], reverse=True)
        
        summary = {
            'total_labs': len(results),
            'best_match': results[0]['lab']['name'] if results else '該当なし',
            'avg_score': sum(r['compatibility']['overall_score'] for r in results) / len(results) if results else 0.0,
            'session_id': str(uuid.uuid4())
        }
        
        algorithm_info = {
            'engine': engine_type,
            'mode': 'debug',
            'processing_time': 0.1
        }
        
        response = {
            'results': results,
            'summary': summary,
            'algorithm_info': algorithm_info
        }
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [SUCCESS] 適合度評価完了: {len(results)}件処理")
        return jsonify(response)
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [ERROR] 適合度評価エラー: {e}")
        import traceback
        print(traceback.format_exc())
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
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] デモデータ提供")
    
    return jsonify({
        'demo_preferences': demo_preferences,
        'message': 'デバッグ版デモデータです'
    })

# エラーハンドラー
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [ERROR] 内部エラー: {error}")
    if 'db' in globals():
        db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

print(f"[{datetime.now().strftime('%H:%M:%S')}] [SUCCESS] ✅ APIエンドポイント定義完了")

# 段階7: データベース初期化
print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] === 段階7: データベース初期化 ===")
with app.app_context():
    try:
        db.create_all()
        lab_count = Lab.query.count()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [SUCCESS] ✅ データベース初期化完了: {lab_count}件の研究室")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [ERROR] データベース初期化失敗: {e}")

# 段階8: 最終テスト
print(f"[{datetime.now().strftime('%H:%M:%S')}] [DEBUG] === 段階8: 最終テスト ===")
if fuzzy_engine:
    try:
        test_prefs = {'research_intensity': 5.0, 'advisor_style': 6.0, 'team_work': 7.0, 'workload': 5.5, 'theory_practice': 6.5}
        test_lab = {'research_intensity': 6.0, 'advisor_style': 6.5, 'team_work': 7.2, 'workload': 6.0, 'theory_practice': 7.0}
        
        result, explanation = fuzzy_engine.predict_compatibility(test_prefs, test_lab)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [SUCCESS] ✅ エンジンテスト成功: {result.get('overall_score', 0):.1f}%")
        
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [ERROR] エンジンテストエラー: {e}")

# 起動情報表示
print(f"[{datetime.now().strftime('%H:%M:%S')}] [INFO] " + "="*50)
print(f"[{datetime.now().strftime('%H:%M:%S')}] [INFO] 🚀 デバッグAPIサーバー起動準備完了")
print(f"[{datetime.now().strftime('%H:%M:%S')}] [INFO] ファジィエンジン: {engine_type}")
print(f"[{datetime.now().strftime('%H:%M:%S')}] [INFO] 利用可能エンドポイント:")
print(f"[{datetime.now().strftime('%H:%M:%S')}] [INFO]   GET  /api/health")
print(f"[{datetime.now().strftime('%H:%M:%S')}] [INFO]   GET  /api/labs")
print(f"[{datetime.now().strftime('%H:%M:%S')}] [INFO]   POST /api/evaluate")
print(f"[{datetime.now().strftime('%H:%M:%S')}] [INFO]   GET  /api/demo-data")
print(f"[{datetime.now().strftime('%H:%M:%S')}] [INFO] サーバーURL: http://localhost:5000")
print(f"[{datetime.now().strftime('%H:%M:%S')}] [INFO] " + "="*50)

def main():
    """メイン実行"""
    try:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [INFO] サーバー起動中...")
        
        app.run(
            debug=True,  # デバッグモード有効
            port=5000,
            host='127.0.0.1',
            use_reloader=False,  # リローダー無効（デバッグ時の問題を避ける）
            threaded=True
        )
        
    except KeyboardInterrupt:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] [INFO] ユーザーによる停止")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [ERROR] サーバー起動エラー: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == '__main__':
    main()