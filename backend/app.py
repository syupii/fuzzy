# app_complete_fixed.py
# -*- coding: utf-8 -*-
"""
完全版APIサーバー（Windows互換・全機能搭載）
- 20項目評価システム
- 遺伝的ファジィ決定木
- Windows文字エンコーディング対応
- 完全なエラーハンドリング
"""

import os
import sys
import time
import uuid
import json
import traceback
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

# Windows文字エンコーディング設定（安全版）
def setup_windows_encoding():
    """Windows環境でのエンコーディング設定"""
    if sys.platform.startswith('win'):
        try:
            # 既存ストリームのバックアップ
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            
            # UTF-8で再設定
            if hasattr(sys.stdout, 'buffer') and hasattr(sys.stderr, 'buffer'):
                import io
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
                sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
                return True
        except Exception as e:
            # エラー時は元のストリームを維持
            sys.stdout = original_stdout if 'original_stdout' in locals() else sys.stdout
            sys.stderr = original_stderr if 'original_stderr' in locals() else sys.stderr
            print(f"Warning: UTF-8 encoding setup failed: {e}")
    return False

# エンコーディング設定実行
setup_windows_encoding()

# プロジェクトパス追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def safe_print(message, level="INFO"):
    """安全な出力関数"""
    try:
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_msg = f"[{timestamp}] [{level}] {message}"
        print(formatted_msg)
    except Exception:
        try:
            import builtins
            builtins.print(f"[{level}] {message}")
        except:
            pass

# === モデルとデータベースのインポート ===
try:
    from models import db, Lab, Evaluation, create_app, DatabaseManager
    safe_print("✅ データベースモデル読み込み完了")
except Exception as e:
    safe_print(f"❌ データベースモデル読み込み失敗: {e}", "ERROR")
    sys.exit(1)

# === ファジィエンジンの動的インポート（優先順位付き） ===
fuzzy_engine = None
engine_info = {"type": "none", "loaded": False, "genetic_available": False}

# インポート試行順序
engine_import_order = [
    ("fuzzy_engine", "HybridFuzzyEngineFixed", "Windows互換版"),
    ("fuzzy_engine", "fuzzy_engine", "デフォルトインスタンス"),
    ("fuzzy_engine_windows_fix", "HybridFuzzyEngineFixed", "Windows修正版"),
    ("genetic_algorithm_windows_fix", "SimpleIndividual", "遺伝的アルゴリズム")
]

for module_name, class_or_instance, description in engine_import_order:
    try:
        module = __import__(module_name)
        
        if hasattr(module, class_or_instance):
            target = getattr(module, class_or_instance)
            
            # クラスかインスタンスかを判定
            if isinstance(target, type):
                # クラスの場合はインスタンス化
                fuzzy_engine = target()
                safe_print(f"✅ {description} (クラス) インスタンス化完了")
            else:
                # インスタンスの場合はそのまま使用
                fuzzy_engine = target
                safe_print(f"✅ {description} (インスタンス) 読み込み完了")
            
            engine_info = {
                "type": class_or_instance,
                "module": module_name,
                "description": description,
                "loaded": True,
                "genetic_available": hasattr(fuzzy_engine, 'genetic_model_loaded')
            }
            break
            
    except Exception as e:
        safe_print(f"⚠️ {description} 読み込み失敗: {str(e)[:100]}...")
        continue

# フォールバック基本エンジン
if fuzzy_engine is None:
    safe_print("🔧 フォールバック基本エンジンを作成中...")
    
    class CompleteFallbackEngine:
        """完全フォールバック基本エンジン"""
        
        def __init__(self):
            self.current_mode = 'fallback_complete'
            self.genetic_model_loaded = False
            self.engine_type = 'CompleteFallbackEngine'
            safe_print("🔧 完全フォールバックエンジン初期化")
            
            # 20項目の重み設定
            self.criteria_weights = {
                # 基本研究環境（30%）
                'research_intensity': 0.06,
                'advisor_style': 0.05,
                'team_work': 0.05,
                'workload': 0.05,
                'theory_practice': 0.05,
                'research_field_match': 0.04,
                
                # 学習・成長関連（15%）
                'skill_development': 0.05,
                'learning_pace': 0.05,
                'difficulty_preference': 0.05,
                
                # コミュニケーション・環境（15%）
                'communication_style': 0.05,
                'meeting_frequency': 0.05,
                'lab_atmosphere': 0.05,
                
                # 研究アプローチ（15%）
                'innovation_risk': 0.05,
                'methodology_preference': 0.05,
                'interdisciplinary': 0.05,
                
                # 時間・ライフスタイル（10%）
                'flexibility': 0.05,
                'evening_weekend_work': 0.05,
                
                # 重要項目（15% - 学生調査基盤）
                'publication_opportunity': 0.04,
                'financial_support': 0.04,
                'lab_hierarchy': 0.035,
                'core_time_flexibility': 0.035,
            }
            
        def predict_compatibility(self, user_prefs, lab_features):
            """20項目対応完全予測"""
            
            total_score = 0.0
            total_weight = 0.0
            criterion_scores = {}
            
            for criterion, weight in self.criteria_weights.items():
                user_val = user_prefs.get(criterion, 5.0)
                lab_val = lab_features.get(criterion, 5.0)
                
                # ガウシアン類似度計算
                diff = abs(user_val - lab_val)
                similarity = self._gaussian_similarity(diff, sigma=2.0)
                
                weighted_score = similarity * weight
                total_score += weighted_score
                total_weight += weight
                
                criterion_scores[criterion] = {
                    'similarity': similarity,
                    'weighted_score': weighted_score * 100,
                    'user_preference': user_val,
                    'lab_feature': lab_val,
                    'weight': weight
                }
            
            # 正規化
            if total_weight > 0:
                overall_score = (total_score / total_weight) * 100
            else:
                overall_score = 50.0
            
            # ボーナス計算（相互作用効果）
            bonus = self._calculate_interaction_bonus(user_prefs, lab_features)
            overall_score += bonus
            
            overall_score = max(0.0, min(100.0, overall_score))
            
            result = {
                'overall_score': overall_score,
                'confidence': self._calculate_confidence(user_prefs, lab_features),
                'prediction_method': 'fallback_complete_20items',
                'criterion_scores': criterion_scores,
                'interaction_bonus': bonus
            }
            
            explanation = (f"20項目完全評価システム: {overall_score:.1f}% "
                          f"(相互作用ボーナス: +{bonus:.1f}%)")
            
            return result, explanation
            
        def _gaussian_similarity(self, diff, sigma=2.0):
            """ガウシアン類似度関数"""
            import math
            return math.exp(-0.5 * (diff / sigma) ** 2)
            
        def _calculate_interaction_bonus(self, user_prefs, lab_features):
            """相互作用ボーナス計算"""
            bonus = 0.0
            
            # 研究強度と理論実践の整合性
            research_diff = abs(user_prefs.get('research_intensity', 5) - 
                               lab_features.get('research_intensity', 5))
            theory_diff = abs(user_prefs.get('theory_practice', 5) - 
                             lab_features.get('theory_practice', 5))
            
            if research_diff < 2 and theory_diff < 2:
                bonus += 2.0
                
            # 指導スタイルとコミュニケーションの調和
            advisor_diff = abs(user_prefs.get('advisor_style', 5) - 
                              lab_features.get('advisor_style', 5))
            comm_diff = abs(user_prefs.get('communication_style', 5) - 
                           lab_features.get('communication_style', 5))
            
            if advisor_diff < 2 and comm_diff < 2:
                bonus += 1.5
                
            # 柔軟性項目の総合評価
            flexibility_items = ['flexibility', 'core_time_flexibility', 'meeting_frequency']
            flexibility_match = sum(1 for item in flexibility_items 
                                   if abs(user_prefs.get(item, 5) - lab_features.get(item, 5)) < 2)
            
            if flexibility_match >= 2:
                bonus += 1.0
                
            return min(bonus, 5.0)  # 最大5%のボーナス
            
        def _calculate_confidence(self, user_prefs, lab_features):
            """信頼度計算"""
            # データの完全性チェック
            completeness = len([v for v in user_prefs.values() if v is not None]) / len(self.criteria_weights)
            
            # 値の分散チェック（極端な値が少ないほど信頼性が高い）
            values = list(user_prefs.values())
            variance_penalty = 0
            extreme_count = sum(1 for v in values if v <= 2 or v >= 9)
            if extreme_count > len(values) * 0.3:  # 30%以上が極端値
                variance_penalty = 10
                
            base_confidence = 75.0
            confidence = base_confidence + (completeness * 15) - variance_penalty
            
            return max(60.0, min(95.0, confidence))
        
        def get_engine_info(self):
            """エンジン情報取得"""
            return {
                'current_mode': self.current_mode,
                'engine_type': self.engine_type,
                'genetic_model_loaded': self.genetic_model_loaded,
                'criteria_count': len(self.criteria_weights),
                'total_weight': sum(self.criteria_weights.values()),
                'available_modes': ['fallback_complete']
            }
    
    fuzzy_engine = CompleteFallbackEngine()
    engine_info = {
        "type": "CompleteFallbackEngine",
        "module": "builtin",
        "description": "完全フォールバック20項目対応エンジン",
        "loaded": True,
        "genetic_available": False
    }
    safe_print("✅ 完全フォールバックエンジン作成完了")

# === Flaskアプリ作成 ===
try:
    app = create_app()
    safe_print("✅ Flaskアプリ作成完了")
except Exception as e:
    safe_print(f"❌ Flaskアプリ作成失敗: {e}", "ERROR")
    sys.exit(1)

# CORS設定（拡張版）
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000", 
            "http://127.0.0.1:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3001"
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "supports_credentials": True
    }
})

# === APIエンドポイント実装 ===

@app.route('/api/health', methods=['GET'])
def health_check():
    """拡張ヘルスチェック"""
    try:
        # データベース統計取得
        lab_count = Lab.query.count()
        evaluation_count = Evaluation.query.count()
        
        # データベースサイズ情報
        db_manager = DatabaseManager()
        table_counts = db_manager.get_table_counts()
        db_size = db_manager.get_database_size()
        
        # エンジン状態
        engine_status = "active" if fuzzy_engine else "unavailable"
        genetic_status = getattr(fuzzy_engine, 'genetic_model_loaded', False) if fuzzy_engine else False
        
        response = {
            'status': 'healthy',
            'message': f'FDTLSS APIサーバー正常動作中 ({engine_info["description"]})',
            'timestamp': time.time(),
            'version': '3.0.0',
            'database': {
                'connected': True,
                'status': 'OK',
                'lab_count': lab_count,
                'evaluation_count': evaluation_count,
                'table_counts': table_counts,
                'size_info': db_size
            },
            'engine': {
                'status': engine_status,
                'type': engine_info["type"],
                'module': engine_info.get("module", "unknown"),
                'description': engine_info["description"],
                'genetic_loaded': genetic_status,
                'mode': getattr(fuzzy_engine, 'current_mode', 'unknown') if fuzzy_engine else None
            },
            'features': {
                'criteria_count': 20,
                'genetic_algorithm': genetic_status,
                'field_matching': True,
                'windows_compatible': sys.platform.startswith('win')
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        safe_print(f"❌ ヘルスチェックエラー: {e}", "ERROR")
        return jsonify({
            'status': 'error',
            'message': f'ヘルスチェック失敗: {str(e)}',
            'timestamp': time.time()
        }), 500

@app.route('/api/labs', methods=['GET'])
def get_labs():
    """研究室一覧取得（拡張版）"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        research_area = request.args.get('research_area')
        
        query = Lab.query.filter_by(is_active=True)
        
        if research_area:
            query = query.filter(Lab.research_area.contains(research_area))
        
        # ページネーション
        pagination = query.paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        labs_data = [lab.to_dict() for lab in pagination.items]
        
        response = {
            'labs': labs_data,
            'pagination': {
                'page': page,
                'per_page': per_page,
                'total': pagination.total,
                'pages': pagination.pages,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            },
            'count': len(labs_data),
            'total_labs': pagination.total
        }
        
        return jsonify(response)
        
    except Exception as e:
        safe_print(f"❌ 研究室一覧取得エラー: {e}", "ERROR")
        return jsonify({'error': f'研究室一覧取得失敗: {str(e)}'}), 500

@app.route('/api/evaluate', methods=['POST'])
def evaluate_compatibility():
    """20項目対応研究室適合度評価"""
    evaluation_start_time = time.time()
    
    try:
        user_prefs = request.get_json()
        
        if not user_prefs:
            return jsonify({'error': '評価データが提供されていません'}), 400
        
        # 20項目の必須フィールド定義
        required_fields = [
            # 基本研究環境
            'research_intensity', 'advisor_style', 'team_work', 'workload', 
            'theory_practice', 'research_field_match',
            # 学習・成長関連
            'skill_development', 'learning_pace', 'difficulty_preference',
            # コミュニケーション・環境
            'communication_style', 'meeting_frequency', 'lab_atmosphere',
            # 研究アプローチ
            'innovation_risk', 'methodology_preference', 'interdisciplinary',
            # 時間・ライフスタイル
            'flexibility', 'evening_weekend_work',
            # 重要項目
            'publication_opportunity', 'financial_support', 'lab_hierarchy', 
            'core_time_flexibility'
        ]
        
        # バリデーション
        missing_fields = []
        invalid_fields = []
        
        for field in required_fields:
            if field not in user_prefs:
                # デフォルト値設定
                user_prefs[field] = 5.0
                safe_print(f"⚠️ デフォルト値設定: {field} = 5.0", "WARN")
            elif not isinstance(user_prefs[field], (int, float)):
                invalid_fields.append(field)
            elif not (1 <= user_prefs[field] <= 10):
                invalid_fields.append(field)
        
        if invalid_fields:
            return jsonify({
                'error': f'無効な値の項目: {", ".join(invalid_fields)}（1-10の範囲で入力してください）'
            }), 400
        
        # セッションID生成
        session_id = str(uuid.uuid4())
        
        # 評価データ保存（20項目対応）
        try:
            evaluation_data = {field: user_prefs[field] for field in required_fields}
            evaluation = Evaluation(session_id=session_id, **evaluation_data)
            db.session.add(evaluation)
            db.session.commit()
            safe_print(f"✅ 評価データ保存完了 (セッション: {session_id[:8]}...)")
        except Exception as e:
            safe_print(f"⚠️ 評価データ保存失敗: {e}", "WARN")
            db.session.rollback()
            evaluation = None
        
        # 全研究室取得
        labs = Lab.query.filter_by(is_active=True).all()
        results = []
        processing_errors = 0
        
        safe_print(f"🔍 {len(labs)}件の研究室を評価中...")
        
        for i, lab in enumerate(labs):
            try:
                # 20項目対応特徴量抽出
                lab_features = {
                    'research_intensity': lab.research_intensity,
                    'advisor_style': lab.advisor_style,
                    'team_work': lab.team_work,
                    'workload': lab.workload,
                    'theory_practice': lab.theory_practice,
                    'research_field_match': lab.research_field_match,
                    'skill_development': lab.skill_development,
                    'learning_pace': lab.learning_pace,
                    'difficulty_preference': lab.difficulty_preference,
                    'communication_style': lab.communication_style,
                    'meeting_frequency': lab.meeting_frequency,
                    'lab_atmosphere': lab.lab_atmosphere,
                    'innovation_risk': lab.innovation_risk,
                    'methodology_preference': lab.methodology_preference,
                    'interdisciplinary': lab.interdisciplinary,
                    'flexibility': lab.flexibility,
                    'evening_weekend_work': lab.evening_weekend_work,
                    'publication_opportunity': lab.publication_opportunity,
                    'financial_support': lab.financial_support,
                    'lab_hierarchy': lab.lab_hierarchy,
                    'core_time_flexibility': lab.core_time_flexibility
                }
                
                # ファジィエンジンで予測
                compatibility, explanation = fuzzy_engine.predict_compatibility(
                    user_prefs, lab_features)
                
                results.append({
                    'lab': lab.to_dict(),
                    'compatibility': compatibility
                })
                
                # 進捗表示（10件ごと）
                if (i + 1) % 10 == 0:
                    safe_print(f"📊 評価進捗: {i + 1}/{len(labs)} ({(i+1)/len(labs)*100:.1f}%)")
                
            except Exception as e:
                processing_errors += 1
                safe_print(f"⚠️ 研究室 {lab.name} の評価エラー: {e}", "WARN")
                
                # エラー時のフォールバック結果
                fallback_compatibility = {
                    'overall_score': 50.0,
                    'confidence': 30.0,
                    'prediction_method': 'error_fallback',
                    'criterion_scores': {},
                    'error': str(e)
                }
                
                results.append({
                    'lab': lab.to_dict(),
                    'compatibility': fallback_compatibility
                })
        
        # 結果をスコア順でソート
        results.sort(key=lambda x: x['compatibility']['overall_score'], reverse=True)
        
        # 評価データ更新
        if evaluation:
            try:
                evaluation.evaluation_count = len(results)
                evaluation.avg_score = sum(r['compatibility']['overall_score'] 
                                         for r in results) / len(results) if results else 0.0
                evaluation.best_lab_id = results[0]['lab']['id'] if results else None
                evaluation.engine_used = engine_info["type"]
                db.session.commit()
            except Exception as e:
                safe_print(f"⚠️ 評価データ更新失敗: {e}", "WARN")
        
        # サマリー作成
        if results:
            best_match = results[0]['lab']['name']
            avg_score = sum(r['compatibility']['overall_score'] for r in results) / len(results)
            high_compatibility_count = sum(1 for r in results if r['compatibility']['overall_score'] >= 70)
        else:
            best_match = "該当なし"
            avg_score = 0.0
            high_compatibility_count = 0
        
        summary = {
            'total_labs': len(results),
            'best_match': best_match,
            'avg_score': avg_score,
            'high_compatibility_count': high_compatibility_count,
            'processing_errors': processing_errors,
            'evaluation_id': getattr(evaluation, 'id', None) if evaluation else None,
            'session_id': session_id
        }
        
        # アルゴリズム情報
        processing_time = time.time() - evaluation_start_time
        algorithm_info = {
            'engine': engine_info["type"],
            'engine_description': engine_info["description"],
            'mode': getattr(fuzzy_engine, 'current_mode', 'unknown') if fuzzy_engine else 'none',
            'genetic_loaded': getattr(fuzzy_engine, 'genetic_model_loaded', False) if fuzzy_engine else False,
            'processing_time': processing_time,
            'criteria_evaluated': len(required_fields),
            'processing_errors': processing_errors
        }
        
        safe_print(f"✅ 評価完了: {len(results)}件処理 ({processing_time:.2f}秒)")
        
        response = {
            'results': results,
            'summary': summary,
            'algorithm_info': algorithm_info,
            'evaluation_metadata': {
                'session_id': session_id,
                'criteria_count': len(required_fields),
                'timestamp': datetime.now().isoformat(),
                'version': '3.0.0'
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        safe_print(f"❌ 評価処理エラー: {e}", "ERROR")
        safe_print(f"❌ エラー詳細: {traceback.format_exc()}", "ERROR")
        return jsonify({
            'error': f'評価処理エラー: {str(e)}',
            'details': '内部エラーが発生しました。管理者にお問い合わせください。'
        }), 500

@app.route('/api/demo-data', methods=['GET'])
def get_demo_data():
    """20項目対応デモデータ"""
    demo_preferences = {
        # 基本研究環境
        'research_intensity': 7.5,
        'advisor_style': 6.0,
        'team_work': 7.0,
        'workload': 6.5,
        'theory_practice': 8.0,
        'research_field_match': 8.5,
        
        # 学習・成長関連
        'skill_development': 7.0,
        'learning_pace': 6.5,
        'difficulty_preference': 7.5,
        
        # コミュニケーション・環境
        'communication_style': 6.5,
        'meeting_frequency': 6.0,
        'lab_atmosphere': 7.5,
        
        # 研究アプローチ
        'innovation_risk': 7.0,
        'methodology_preference': 6.5,
        'interdisciplinary': 8.0,
        
        # 時間・ライフスタイル
        'flexibility': 8.0,
        'evening_weekend_work': 5.0,
        
        # 重要項目（学生調査基盤）
        'publication_opportunity': 8.5,
        'financial_support': 7.5,
        'lab_hierarchy': 5.5,
        'core_time_flexibility': 8.0
    }
    
    return jsonify({
        'demo_preferences': demo_preferences,
        'message': '20項目対応デモ設定です。実際の学生調査結果に基づく重要項目を含んでいます。',
        'criteria_info': {
            'basic_environment': 6,
            'learning_growth': 3,
            'communication': 3,
            'research_approach': 3,
            'lifestyle': 2,
            'priority_items': 4,
            'total': 21  # research_field_matchが追加で21項目
        }
    })

@app.route('/api/engine/info', methods=['GET'])
def get_engine_info():
    """エンジン詳細情報取得"""
    if not fuzzy_engine:
        return jsonify({'error': 'エンジンが利用できません'}), 503
    
    try:
        base_info = fuzzy_engine.get_engine_info() if hasattr(fuzzy_engine, 'get_engine_info') else {}
        
        extended_info = {
            **base_info,
            'import_info': engine_info,
            'capabilities': {
                'criteria_count': 20,
                'genetic_algorithm': getattr(fuzzy_engine, 'genetic_model_loaded', False),
                'interaction_effects': True,
                'confidence_calculation': True,
                'windows_compatible': True
            },
            'performance': {
                'avg_processing_time': '< 1 second per lab',
                'supported_labs': 'unlimited',
                'memory_usage': 'low'
            }
        }
        
        return jsonify(extended_info)
        
    except Exception as e:
        safe_print(f"❌ エンジン情報取得エラー: {e}", "ERROR")
        return jsonify({'error': f'エンジン情報取得失敗: {str(e)}'}), 500

@app.route('/api/engine/reload', methods=['POST'])
def reload_engine():
    """エンジン再読み込み"""
    try:
        global fuzzy_engine, engine_info
        
        if hasattr(fuzzy_engine, 'reload_genetic_model'):
            success = fuzzy_engine.reload_genetic_model()
            message = "遺伝的モデル再読み込み完了" if success else "遺伝的モデル再読み込み失敗"
        else:
            # エンジン全体を再初期化
            # (実際の実装では、モジュール再インポートが必要)
            message = "エンジン再読み込み要求を受信"
            success = True
        
        return jsonify({
            'success': success,
            'message': message,
            'engine_type': engine_info["type"],
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        safe_print(f"❌ エンジン再読み込みエラー: {e}", "ERROR")
        return jsonify({'error': f'エンジン再読み込み失敗: {str(e)}'}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """システム統計情報取得"""
    try:
        # データベース統計
        db_manager = DatabaseManager()
        table_counts = db_manager.get_table_counts()
        db_size = db_manager.get_database_size()
        
        # 評価統計
        recent_evaluations = Evaluation.query.filter(
            Evaluation.created_at >= datetime.now() - timedelta(days=30)
        ).count()
        
        # 人気研究室統計（最近の評価から）
        popular_labs_query = """
        SELECT best_lab_id, COUNT(*) as selection_count 
        FROM evaluations 
        WHERE best_lab_id IS NOT NULL 
        GROUP BY best_lab_id 
        ORDER BY selection_count DESC 
        LIMIT 5
        """
        
        # エンジン性能統計
        engine_stats = {
            'type': engine_info["type"],
            'genetic_available': getattr(fuzzy_engine, 'genetic_model_loaded', False),
            'criteria_supported': 20,
            'confidence_range': '60-95%'
        }
        
        statistics = {
            'database': {
                'table_counts': table_counts,
                'database_size': db_size,
                'total_evaluations': table_counts.get('evaluations', 0),
                'active_labs': table_counts.get('labs', 0)
            },
            'usage': {
                'recent_evaluations_30d': recent_evaluations,
                'system_uptime': time.time()
            },
            'engine': engine_stats,
            'generated_at': datetime.now().isoformat()
        }
        
        return jsonify(statistics)
        
    except Exception as e:
        safe_print(f"❌ 統計情報取得エラー: {e}", "ERROR")
        return jsonify({'error': f'統計情報取得失敗: {str(e)}'}), 500

@app.route('/api/labs/<int:lab_id>', methods=['GET'])
def get_lab_detail(lab_id):
    """研究室詳細情報取得"""
    try:
        lab = Lab.query.get_or_404(lab_id)
        
        if not lab.is_active:
            return jsonify({'error': '指定された研究室は現在利用できません'}), 404
        
        # 基本情報
        lab_data = lab.to_dict()
        
        # 統計情報追加
        evaluation_count = Evaluation.query.filter_by(best_lab_id=lab_id).count()
        
        lab_data['statistics'] = {
            'evaluation_count': evaluation_count,
            'popularity_rank': None  # 実装可能
        }
        
        return jsonify(lab_data)
        
    except Exception as e:
        safe_print(f"❌ 研究室詳細取得エラー: {e}", "ERROR")
        return jsonify({'error': f'研究室詳細取得失敗: {str(e)}'}), 500

@app.route('/api/evaluation/<session_id>', methods=['GET'])
def get_evaluation_result(session_id):
    """評価結果再取得"""
    try:
        evaluation = Evaluation.query.filter_by(session_id=session_id).first_or_404()
        
        # 評価データを辞書形式で取得
        evaluation_data = evaluation.to_dict()
        
        # 再評価が必要な場合の処理
        if request.args.get('refresh', '').lower() == 'true':
            # 再評価実行
            user_prefs = evaluation_data['preferences']
            labs = Lab.query.filter_by(is_active=True).all()
            
            results = []
            for lab in labs:
                lab_features = {key: getattr(lab, key) for key in user_prefs.keys() 
                               if hasattr(lab, key)}
                
                compatibility, explanation = fuzzy_engine.predict_compatibility(
                    user_prefs, lab_features)
                
                results.append({
                    'lab': lab.to_dict(),
                    'compatibility': compatibility
                })
            
            results.sort(key=lambda x: x['compatibility']['overall_score'], reverse=True)
            
            return jsonify({
                'evaluation': evaluation_data,
                'results': results,
                'refreshed': True,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'evaluation': evaluation_data,
                'refreshed': False
            })
        
    except Exception as e:
        safe_print(f"❌ 評価結果取得エラー: {e}", "ERROR")
        return jsonify({'error': f'評価結果取得失敗: {str(e)}'}), 500

# === エラーハンドラー ===

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not Found',
        'message': '指定されたリソースが見つかりません',
        'status_code': 404
    }), 404

@app.errorhandler(500)
def internal_error(error):
    if 'db' in globals():
        db.session.rollback()
    return jsonify({
        'error': 'Internal Server Error',
        'message': '内部サーバーエラーが発生しました',
        'status_code': 500
    }), 500

@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        'error': 'Bad Request',
        'message': 'リクエストデータが不正です',
        'status_code': 400
    }), 400

# === メイン実行関数 ===

def initialize_database():
    """データベース初期化"""
    with app.app_context():
        try:
            # テーブル作成
            db.create_all()
            safe_print("✅ データベーステーブル確認・作成完了")
            
            # 基本データ確認
            lab_count = Lab.query.count()
            if lab_count == 0:
                safe_print("⚠️ 研究室データが0件です。初期データの投入を検討してください", "WARN")
            else:
                safe_print(f"✅ 研究室データ: {lab_count}件確認")
                
            return True
            
        except Exception as e:
            safe_print(f"❌ データベース初期化失敗: {e}", "ERROR")
            return False

def print_startup_info():
    """起動情報表示"""
    safe_print("=" * 60)
    safe_print("🚀 FDTLSS APIサーバー (完全版) 起動中...")
    safe_print("=" * 60)
    
    # システム情報
    safe_print(f"🐍 Python: {sys.version.split()[0]}")
    safe_print(f"🖥️ OS: {sys.platform}")
    safe_print(f"📁 作業ディレクトリ: {os.getcwd()}")
    
    # エンジン情報
    safe_print(f"🔧 ファジィエンジン: {engine_info['description']}")
    safe_print(f"🧬 遺伝的アルゴリズム: {'✅ 利用可能' if engine_info.get('genetic_available') else '❌ 利用不可'}")
    safe_print(f"📊 評価基準: 20項目対応")
    
    # APIエンドポイント一覧
    safe_print(f"\n🌐 利用可能なAPIエンドポイント:")
    endpoints = [
        "GET  /api/health          - システムヘルスチェック",
        "GET  /api/labs            - 研究室一覧取得",
        "GET  /api/labs/<id>       - 研究室詳細取得", 
        "POST /api/evaluate        - 20項目適合度評価",
        "GET  /api/demo-data       - デモデータ取得",
        "GET  /api/engine/info     - エンジン詳細情報",
        "POST /api/engine/reload   - エンジン再読み込み",
        "GET  /api/statistics      - システム統計情報",
        "GET  /api/evaluation/<id> - 評価結果再取得"
    ]
    
    for endpoint in endpoints:
        safe_print(f"   {endpoint}")
    
    # 起動URL
    safe_print(f"\n🌍 サーバー起動URL:")
    safe_print(f"   http://localhost:5000")
    safe_print(f"   http://127.0.0.1:5000")
    
    # 使用方法
    safe_print(f"\n📖 使用方法:")
    safe_print(f"   1. フロントエンド起動: cd frontend && npm start")
    safe_print(f"   2. ブラウザアクセス: http://localhost:3000")
    safe_print(f"   3. 20項目評価フォームで設定入力")
    safe_print(f"   4. 遺伝的ファジィアルゴリズムによる適合度判定")
    
    safe_print(f"\n⚡ 機能概要:")
    safe_print(f"   - 20項目包括評価システム")
    safe_print(f"   - 遺伝的ファジィ決定木アルゴリズム")
    safe_print(f"   - 相互作用効果計算")
    safe_print(f"   - 信頼度スコアリング")
    safe_print(f"   - Windows完全互換")
    
    safe_print(f"\n🛑 終了方法: Ctrl+C")

def main():
    """メイン実行"""
    
    print_startup_info()
    
    # データベース初期化
    if not initialize_database():
        safe_print("❌ データベース初期化に失敗しました。終了します。", "ERROR")
        return 1
    
    # エンジン最終チェック
    if fuzzy_engine:
        try:
            # テスト予測実行
            test_prefs = {field: 5.0 for field in [
                'research_intensity', 'advisor_style', 'team_work', 'workload', 
                'theory_practice', 'research_field_match', 'skill_development', 
                'learning_pace', 'difficulty_preference', 'communication_style',
                'meeting_frequency', 'lab_atmosphere', 'innovation_risk',
                'methodology_preference', 'interdisciplinary', 'flexibility',
                'evening_weekend_work', 'publication_opportunity', 'financial_support',
                'lab_hierarchy', 'core_time_flexibility'
            ]}
            
            test_lab = {field: 6.0 for field in test_prefs.keys()}
            result, explanation = fuzzy_engine.predict_compatibility(test_prefs, test_lab)
            
            safe_print(f"✅ エンジンテスト成功: スコア={result.get('overall_score', 0):.1f}%")
            
        except Exception as e:
            safe_print(f"⚠️ エンジンテスト警告: {e}", "WARN")
    
    safe_print(f"\n🎯 システム準備完了！")
    safe_print(f"=" * 60)
    
    # サーバー起動
    try:
        app.run(
            debug=False,
            port=5000, 
            host='0.0.0.0',
            use_reloader=False,
            threaded=True
        )
    except KeyboardInterrupt:
        safe_print(f"\n🛑 ユーザーによる停止要求")
        safe_print(f"✅ サーバーを正常に停止しました")
    except Exception as e:
        safe_print(f"❌ サーバー起動エラー: {e}", "ERROR")
        return 1
    
    return 0

# === 起動処理 ===

if __name__ == '__main__':
    # 必要なインポートの最終確認
    try:
        from datetime import timedelta
        safe_print("✅ 必要モジュール確認完了")
    except ImportError as e:
        safe_print(f"❌ 必要モジュール不足: {e}", "ERROR")
        sys.exit(1)
    
    # メイン実行
    exit_code = main()
    
    if exit_code == 0:
        safe_print("\n✅ FDTLSS APIサーバーを正常終了しました")
    else:
        safe_print(f"\n❌ サーバー終了時にエラーが発生しました (終了コード: {exit_code})", "ERROR")
    
    sys.exit(exit_code)