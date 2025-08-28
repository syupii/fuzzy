# fuzzy_engine_genetic_fix.py
"""
🔧 HybridFuzzyEngine遺伝的アルゴリズム統合修正
遺伝的アルゴリズムを確実に動作させるためのHybridFuzzyEngine修正版
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# プロジェクトパス追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class GeneticFuzzyEngine:
    """遺伝的ファジィエンジン（独立版）"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.genetic_model = None
        self.best_individual = None
        self.is_genetic_loaded = False
        
        # モデル読み込み試行
        self.load_genetic_model()
    
    def load_genetic_model(self) -> bool:
        """遺伝的モデル読み込み"""
        
        # 複数の場所を試行
        possible_paths = [
            os.path.join(self.models_dir, "genetic_optimization_results.pkl"),
            os.path.join(self.models_dir, "best_genetic_tree.pkl"),
            os.path.join(self.models_dir, "genetic_model_latest.pkl"),
            "genetic_optimization_results.pkl",  # カレントディレクトリ
        ]
        
        for model_path in possible_paths:
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        self.genetic_model = pickle.load(f)
                    
                    self.best_individual = self.genetic_model.get('best_individual')
                    
                    if self.best_individual and hasattr(self.best_individual, 'tree'):
                        self.is_genetic_loaded = True
                        print(f"✅ 遺伝的モデル読み込み成功: {model_path}")
                        print(f"   個体ID: {self.best_individual.individual_id}")
                        print(f"   適応度: {self.best_individual.fitness_value:.4f}")
                        return True
                        
                except Exception as e:
                    print(f"⚠️ モデル読み込み失敗 {model_path}: {e}")
                    continue
        
        print(f"❌ 遺伝的モデルが見つかりません。新規作成が必要です。")
        self.is_genetic_loaded = False
        return False
    
    def predict_compatibility(self, user_prefs: Dict[str, float], 
                            lab_features: Dict[str, float]) -> Tuple[Dict[str, Any], str]:
        """適合度予測"""
        
        if not self.is_genetic_loaded:
            return self._fallback_prediction(user_prefs, lab_features)
        
        try:
            return self._genetic_prediction(user_prefs, lab_features)
        
        except Exception as e:
            print(f"⚠️ 遺伝的予測エラー: {e}")
            return self._fallback_prediction(user_prefs, lab_features)
    
    def _genetic_prediction(self, user_prefs: Dict[str, float], 
                          lab_features: Dict[str, float]) -> Tuple[Dict[str, Any], str]:
        """遺伝的決定木による予測"""
        
        # 特徴量準備（類似度ベース）
        features = {}
        criteria = ['research_intensity', 'advisor_style', 'team_work', 'workload', 'theory_practice']
        
        for criterion in criteria:
            user_val = user_prefs.get(criterion, 5.0)
            lab_val = lab_features.get(criterion, 5.0)
            
            # 類似度計算
            similarity = 1.0 - abs(user_val - lab_val) / 10.0
            features[criterion] = max(0.0, min(1.0, similarity)) * 10.0
        
        # 遺伝的決定木で予測
        prediction = self.best_individual.tree.predict(features)
        
        # 説明付き予測
        try:
            detailed_prediction, explanation = self.best_individual.tree.predict_with_explanation(
                features, criteria
            )
        except Exception as e:
            print(f"⚠️ 説明生成エラー: {e}")
            explanation = {
                'confidence': 0.8,
                'rationale': f'遺伝的最適化による予測: {prediction:.3f}',
                'decision_steps': ['遺伝的決定木による判定']
            }
        
        # 結果フォーマット
        result = {
            'overall_score': prediction * 100,
            'confidence': explanation.get('confidence', 0.8) * 100,
            'prediction_method': 'genetic_optimization',
            'genetic_info': {
                'individual_id': self.best_individual.individual_id,
                'generation': getattr(self.best_individual, 'generation', 0),
                'fitness': self.best_individual.fitness_value
            },
            'criterion_scores': self._create_criterion_scores(user_prefs, lab_features, features),
            'decision_path': explanation.get('decision_steps', [])
        }
        
        explanation_text = (f"遺伝的最適化による予測: {prediction:.1%} "
                          f"(適応度: {self.best_individual.fitness_value:.3f})")
        
        return result, explanation_text
    
    def _create_criterion_scores(self, user_prefs: Dict[str, float], 
                               lab_features: Dict[str, float], 
                               processed_features: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """基準別スコア作成"""
        
        scores = {}
        weights = {
            'research_intensity': 0.25,
            'advisor_style': 0.20,
            'team_work': 0.20,
            'workload': 0.15,
            'theory_practice': 0.20
        }
        
        for criterion in processed_features:
            user_val = user_prefs.get(criterion, 5.0)
            lab_val = lab_features.get(criterion, 5.0)
            similarity = processed_features[criterion] / 10.0
            weight = weights.get(criterion, 0.2)
            
            scores[criterion] = {
                'similarity': similarity,
                'weighted_score': similarity * weight * 100,
                'user_preference': user_val,
                'lab_feature': lab_val,
                'weight': weight
            }
        
        return scores
    
    def _fallback_prediction(self, user_prefs: Dict[str, float], 
                           lab_features: Dict[str, float]) -> Tuple[Dict[str, Any], str]:
        """フォールバック予測（遺伝的モデル無効時）"""
        
        criteria = ['research_intensity', 'advisor_style', 'team_work', 'workload', 'theory_practice']
        weights = [0.25, 0.20, 0.20, 0.15, 0.20]
        
        similarities = []
        
        for criterion in criteria:
            user_val = user_prefs.get(criterion, 5.0)
            lab_val = lab_features.get(criterion, 5.0)
            similarity = 1.0 - abs(user_val - lab_val) / 10.0
            similarities.append(max(0.0, similarity))
        
        # 重み付き平均
        overall_score = sum(w * s for w, s in zip(weights, similarities))
        
        result = {
            'overall_score': overall_score * 100,
            'confidence': 75.0,
            'prediction_method': 'fallback_weighted',
            'genetic_info': {
                'individual_id': 'fallback',
                'generation': 0,
                'fitness': overall_score
            },
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
        
        explanation_text = f"重み付き類似度による予測: {overall_score:.1%}"
        
        return result, explanation_text


class HybridFuzzyEngineFixed:
    """修正版HybridFuzzyEngine（遺伝的アルゴリズム対応）"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.current_mode = 'genetic'  # デフォルトで遺伝的モード
        
        # エンジン初期化
        self.genetic_engine = GeneticFuzzyEngine(models_dir)
        self.genetic_model_loaded = self.genetic_engine.is_genetic_loaded
        
        print(f"🧬 HybridFuzzyEngine (Fixed) 初期化完了")
        print(f"   現在のモード: {self.current_mode}")
        print(f"   遺伝的モデル: {'✅' if self.genetic_model_loaded else '❌'}")
        
        if not self.genetic_model_loaded:
            print(f"⚠️ 遺伝的モデルが利用できません。フォールバックモードで動作します。")
    
    def predict_compatibility(self, user_prefs: Dict[str, float], 
                            lab_features: Dict[str, float]) -> Tuple[Dict[str, Any], str]:
        """適合度予測（統一インターフェース）"""
        
        try:
            if self.current_mode == 'genetic' and self.genetic_model_loaded:
                return self.genetic_engine.predict_compatibility(user_prefs, lab_features)
            else:
                return self._simple_fuzzy_prediction(user_prefs, lab_features)
                
        except Exception as e:
            print(f"⚠️ 予測エラー: {e}")
            return self._emergency_fallback(user_prefs, lab_features)
    
    def _simple_fuzzy_prediction(self, user_prefs: Dict[str, float], 
                                lab_features: Dict[str, float]) -> Tuple[Dict[str, Any], str]:
        """シンプルファジィ予測"""
        
        criteria = ['research_intensity', 'advisor_style', 'team_work', 'workload', 'theory_practice']
        
        # ファジィ類似度計算
        similarities = []
        
        for criterion in criteria:
            user_val = user_prefs.get(criterion, 5.0)
            lab_val = lab_features.get(criterion, 5.0)
            
            # ガウシアン類似度
            diff = abs(user_val - lab_val)
            similarity = np.exp(-0.5 * (diff / 2.0) ** 2)
            similarities.append(similarity)
        
        # 重み付き平均
        weights = [0.25, 0.20, 0.20, 0.15, 0.20]
        overall_score = sum(w * s for w, s in zip(weights, similarities))
        
        result = {
            'overall_score': overall_score * 100,
            'confidence': 70.0,
            'prediction_method': 'simple_fuzzy',
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
        
        explanation_text = f"シンプルファジィ論理による予測: {overall_score:.1%}"
        
        return result, explanation_text
    
    def _emergency_fallback(self, user_prefs: Dict[str, float], 
                          lab_features: Dict[str, float]) -> Tuple[Dict[str, Any], str]:
        """緊急時フォールバック"""
        
        # 最もシンプルな線形予測
        criteria = ['research_intensity', 'advisor_style', 'team_work', 'workload', 'theory_practice']
        
        total_diff = 0.0
        for criterion in criteria:
            user_val = user_prefs.get(criterion, 5.0)
            lab_val = lab_features.get(criterion, 5.0)
            total_diff += abs(user_val - lab_val)
        
        # 0-100スケールに正規化
        max_diff = len(criteria) * 10.0
        similarity = 1.0 - (total_diff / max_diff)
        overall_score = max(0.0, min(1.0, similarity))
        
        result = {
            'overall_score': overall_score * 100,
            'confidence': 60.0,
            'prediction_method': 'emergency_linear'
        }
        
        explanation_text = f"緊急フォールバック予測: {overall_score:.1%}"
        
        return result, explanation_text
    
    def switch_mode(self, mode: str) -> bool:
        """モード切り替え"""
        if mode == 'genetic' and self.genetic_model_loaded:
            self.current_mode = 'genetic'
            print(f"✅ 遺伝的モードに切り替えました")
            return True
        elif mode == 'simple':
            self.current_mode = 'simple'
            print(f"✅ シンプルモードに切り替えました")
            return True
        else:
            print(f"❌ モード切り替え失敗: {mode}")
            return False
    
    def get_engine_info(self) -> Dict[str, Any]:
        """エンジン情報取得"""
        return {
            'current_mode': self.current_mode,
            'genetic_model_loaded': self.genetic_model_loaded,
            'genetic_model_info': {
                'individual_id': self.genetic_engine.best_individual.individual_id if self.genetic_engine.best_individual else None,
                'fitness': self.genetic_engine.best_individual.fitness_value if self.genetic_engine.best_individual else 0.0,
                'complexity': self.genetic_engine.best_individual.complexity_score if self.genetic_engine.best_individual else 0
            } if self.genetic_model_loaded else {},
            'available_modes': ['genetic', 'simple']
        }
    
    def reload_genetic_model(self) -> bool:
        """遺伝的モデル再読み込み"""
        print("🔄 遺伝的モデル再読み込み中...")
        
        success = self.genetic_engine.load_genetic_model()
        self.genetic_model_loaded = success
        
        if success:
            print("✅ 再読み込み成功")
        else:
            print("❌ 再読み込み失敗")
        
        return success

# app.pyとの互換性のため、デフォルトインスタンスを作成
fuzzy_engine = HybridFuzzyEngineFixed()

def main():
    """テスト実行"""
    
    print("🧪 HybridFuzzyEngine修正版テスト")
    print("=" * 50)
    
    # エンジン初期化
    engine = HybridFuzzyEngineFixed()
    
    # テスト用データ
    test_user_prefs = {
        'research_intensity': 8.0,
        'advisor_style': 6.5,
        'team_work': 7.0,
        'workload': 6.0,
        'theory_practice': 8.5
    }
    
    test_lab_features = {
        'research_intensity': 7.5,
        'advisor_style': 7.0,
        'team_work': 7.2,
        'workload': 6.8,
        'theory_practice': 8.2
    }
    
    # 予測テスト
    print("\n🔍 予測テスト実行中...")
    
    try:
        result, explanation = engine.predict_compatibility(test_user_prefs, test_lab_features)
        
        print(f"✅ 予測成功!")
        print(f"   スコア: {result.get('overall_score', 0):.1f}")
        print(f"   信頼度: {result.get('confidence', 0):.1f}%")
        print(f"   手法: {result.get('prediction_method', 'unknown')}")
        print(f"   説明: {explanation}")
        
        if 'genetic' in result.get('prediction_method', ''):
            print("\n🎉 遺伝的アルゴリズムが正常に動作しています!")
            
            # 遺伝的情報表示
            genetic_info = result.get('genetic_info', {})
            print(f"   個体ID: {genetic_info.get('individual_id', 'N/A')}")
            print(f"   世代: {genetic_info.get('generation', 'N/A')}")
            print(f"   適応度: {genetic_info.get('fitness', 'N/A'):.4f}")
        else:
            print("\n⚠️ フォールバックモードで動作しています")
        
        # エンジン情報表示
        print(f"\n📊 エンジン情報:")
        info = engine.get_engine_info()
        for key, value in info.items():
            if key != 'genetic_model_info':
                print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ テスト失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = main()
    print(f"\n{'✅ テスト成功' if success else '❌ テスト失敗'}")