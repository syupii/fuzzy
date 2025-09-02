# app_fixed.py
# -*- coding: utf-8 -*-
"""
修正版APIサーバー（I/O操作エラー解決版）
安全版ファジィエンジンを使用
"""

import os
import sys
import time
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS

# プロジェクトパス追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 安全な出力関数
def safe_print(message):
    try:
        print(f"[{time.strftime('%H:%M:%S')}] {message}")
    except:
        import builtins
        builtins.print(message)

# === データベースインポート ===
try:
    from models import db, Lab, Evaluation, create_app
    safe_print("✅ データベースモデル読み込み完了")
except Exception as e:
    safe_print(f"❌ データベースモデル読み込み失敗: {e}")
    sys.exit(1)

# === ファジィエンジンインポート（安全版） ===
fuzzy_engine = None
engine_info = {"type": "none", "loaded": False}

# インポート試行順序（安全版優先）
import_attempts = [
    ("fuzzy_engine_safe", "HybridFuzzyEngineSafe", "安全版"),
    ("fuzzy_engine_fixed", "HybridFuzzyEngineSafe", "修正版"),
    ("fuzzy_engine", "fuzzy_engine", "標準版インスタンス")
]

for module_name, target_name, description in import_attempts:
    try:
        safe_print(f"🔧 {description}エンジン読み込み試行中...")
        module = __import__(module_name)
        
        if hasattr(module, target_name):
            target = getattr(module, target_name)
            
            if isinstance(target, type):
                fuzzy_engine = target()
                safe_print(f"✅ {description}エンジン (クラス) 初期化完了")
            else:
                fuzzy_engine = target
                safe_print(f"✅ {description}エンジン (インスタンス) 読み込み完了")
            
            engine_info = {
                "type": target_name,
                "module": module_name, 
                "description": description,
                "loaded": True
            }
            break
            
    except Exception as e:
        safe_print(f"⚠️ {description}エンジン読み込み失敗: {str(e)[:50]}...")

# フォールバックエンジン
if fuzzy_engine is None:
    safe_print("🔧 フォールバック基本エンジンを作成...")
    
    class BasicFallbackEngine:
        def __init__(self):
            self.current_mode = 'basic_fallback'
            self.genetic_model_loaded = False
            
        def predict_compatibility(self, user_prefs, lab_features):
            criteria = ['research_intensity', 'advisor_style', 'team_work', 'workload', 'theory_practice']
            weights = [0.25, 0.20, 0.20, 0.15, 0.20]
            
            similarities = []
            for criterion in criteria:
                user_val = user_prefs.get(criterion, 5.0)
                lab_val = lab_features.get(criterion, 5.0)
                similarity = 1.0 - abs(user_val - lab_val) / 10.0
                similarities.append(max(0.0, similarity))
            
            overall_score = sum(w * s for w, s in zip(weights, similarities)) * 100
            
            result = {
                'overall_score': overall_score,
                'confidence': 75.0,
                'prediction_method': 'basic_fallback',
                'criterion_scores': {}
            }
            
            for i, criterion in enumerate(criteria):
                result['criterion_scores'][criterion] = {
                    'similarity': similarities[i],
                    'weighted_score': similarities[i] * weights[i] * 100,
                    'user_preference': user_prefs.get(criterion, 5.0),
                    'lab_feature': lab_features.get(criterion, 5.0),
                    'weight': weights[i]
                }
            
            explanation = f"基本評価システム: {overall_score:.1f}%"
            return result, explanation
        
        def get_engine_info(self):
            return {
                'current_mode': self.current_mode,
                'genetic_model_loaded': self.genetic_model_loaded,
                'engine_type': 'BasicFallbackEngine'
            }
    
    fuzzy_engine = BasicFallbackEngine()
    engine_info = {"type": "BasicFallbackEngine", "loaded": True, "description": "基本フォールバック"}

# Flaskアプリ作成
try:
    app = create_app()
    safe_print("✅ Flaskアプリ作成完了")
except Exception as e:
    safe_print(f"❌ Flaskアプリ作成失敗: {e}")
    sys.exit(1)

# CORS設定
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# === APIエンドポイント ===

@app.route('/api/health', methods=['GET'])
def health_check():
    """ヘルスチェック"""
    try:
        lab_count = Lab.query.count()
        
        response = {
            'status': 'healthy',
            'message': f'FDTLSS APIサーバー正常動作中',
            'timestamp': time.time(),
            'version': '2.1',
            'database': {
                'connected': True,
                'lab_count': lab_count
            },
            'engine': {
                'type': engine_info["type"],
                'description': engine_info["description"],
                'loaded': engine_info["loaded"]
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'ヘルスチェック失敗: {str(e)}'
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
    """適合度評価"""
    try:
        user_prefs = request.get_json()
        
        if not user_prefs:
            return jsonify({'error': '評価データが必要です'}), 400
        
        # 基本項目チェック
        required_fields = ['research_intensity', 'advisor_style', 'team_work', 'workload', 'theory_practice']
        for field in required_fields:
            if field not in user_prefs:
                user_prefs[field] = 5.0
        
        # セッションID
        session_id = str(uuid.uuid4())
        
        # 研究室評価
        labs = Lab.query.filter_by(is_active=True).all()
        results = []
        
        safe_print(f"🔍 {len(labs)}件の研究室を評価中...")
        
        for lab in labs:
            try:
                lab_features = {
                    'research_intensity': lab.research_intensity,
                    'advisor_style': lab.advisor_style,
                    'team_work': lab.team_work,
                    'workload': lab.workload,
                    'theory_practice': lab.theory_practice
                }
                
                compatibility, explanation = fuzzy_engine.predict_compatibility(
                    user_prefs, lab_features)
                
                results.append({
                    'lab': lab.to_dict(),
                    'compatibility': compatibility
                })
                
            except Exception as e:
                safe_print(f"⚠️ 研究室評価エラー {lab.name}: {e}")
                # エラー時フォールバック
                results.append({
                    'lab': lab.to_dict(),
                    'compatibility': {
                        'overall_score': 50.0,
                        'confidence': 30.0,
                        'prediction_method': 'error_fallback'
                    }
                })
        
        # 結果ソート
        results.sort(key=lambda x: x['compatibility']['overall_score'], reverse=True)
        
        # サマリー作成
        summary = {
            'total_labs': len(results),
            'best_match': results[0]['lab']['name'] if results else '該当なし',
            'avg_score': sum(r['compatibility']['overall_score'] for r in results) / len(results) if results else 0.0,
            'session_id': session_id
        }
        
        # アルゴリズム情報
        algorithm_info = {
            'engine': engine_info["type"],
            'description': engine_info["description"],
            'processing_time': 0.1
        }
        
        response = {
            'results': results,
            'summary': summary,
            'algorithm_info': algorithm_info
        }
        
        safe_print(f"✅ 評価完了: {len(results)}件処理")
        return jsonify(response)
        
    except Exception as e:
        safe_print(f"❌ 評価処理エラー: {e}")
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
        'message': '修正版デモデータです'
    })

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
    safe_print("=" * 50)
    safe_print("🚀 FDTLSS APIサーバー (修正版) 起動中...")
    safe_print("=" * 50)
    
    # データベース初期化
    with app.app_context():
        try:
            db.create_all()
            lab_count = Lab.query.count()
            safe_print(f"✅ データベース初期化完了: {lab_count}件の研究室")
        except Exception as e:
            safe_print(f"❌ データベース初期化失敗: {e}")
    
    # エンジンテスト
    if fuzzy_engine:
        try:
            test_prefs = {'research_intensity': 7.0, 'advisor_style': 6.0, 'team_work': 7.0, 'workload': 6.0, 'theory_practice': 7.0}
            test_lab = {'research_intensity': 6.5, 'advisor_style': 6.5, 'team_work': 7.5, 'workload': 6.5, 'theory_practice': 7.5}
            
            result, explanation = fuzzy_engine.predict_compatibility(test_prefs, test_lab)
            safe_print(f"✅ エンジンテスト成功: {result.get('overall_score', 0):.1f}%")
            
        except Exception as e:
            safe_print(f"⚠️ エンジンテスト警告: {e}")
    
    safe_print(f"\n🌐 APIエンドポイント:")
    safe_print(f"   GET  /api/health")
    safe_print(f"   GET  /api/labs")
    safe_print(f"   POST /api/evaluate")
    safe_print(f"   GET  /api/demo-data")
    
    safe_print(f"\n🌍 サーバーURL: http://localhost:5000")
    safe_print(f"🎯 システム準備完了！")
    safe_print("=" * 50)
    
    # サーバー起動
    try:
        app.run(debug=False, port=5000, host='0.0.0.0', use_reloader=False)
    except KeyboardInterrupt:
        safe_print("\n🛑 ユーザーによる停止")
    except Exception as e:
        safe_print(f"❌ サーバー起動エラー: {e}")

if __name__ == '__main__':
    main()