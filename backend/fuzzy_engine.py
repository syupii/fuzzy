# backend/fuzzy_engine.py - 完全修正版
import math
import os
import pickle
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 既存のFuzzyLogicEngineクラス（後方互換性のため保持）


class FuzzyLogicEngine:
    """ファジィ論理エンジン（既存バージョン）"""

    def __init__(self):
        self.criteria = ['research_intensity', 'advisor_style',
                         'team_work', 'workload', 'theory_practice']
        self.weights = [0.25, 0.20, 0.20, 0.15, 0.20]  # 重み（合計1.0）

        self.criteria_labels = {
            'research_intensity': '研究強度',
            'advisor_style': '指導スタイル',
            'team_work': 'チームワーク',
            'workload': 'ワークロード',
            'theory_practice': '理論・実践バランス'
        }

        # 各基準の許容範囲（ファジィパラメータ）
        self.tolerances = {
            'research_intensity': 2.0,
            'advisor_style': 2.5,
            'team_work': 2.5,
            'workload': 2.0,
            'theory_practice': 2.0
        }

    def gaussian_membership(self, x: float, mean: float, sigma: float) -> float:
        """ガウシアンメンバーシップ関数"""
        return math.exp(-0.5 * ((x - mean) / sigma) ** 2)

    def calculate_similarity(self, user_pref: float, lab_feature: float, criterion: str) -> float:
        """特徴量間の類似度計算"""
        diff = abs(user_pref - lab_feature)
        tolerance = self.tolerances.get(criterion, 2.0)

        # ガウシアンメンバーシップで類似度計算
        similarity = self.gaussian_membership(diff, 0, tolerance/2)
        return similarity

    def fuzzy_inference(self, user_preferences: Dict, lab_features: Dict) -> Dict:
        """ファジィ推論による適合度計算"""

        criterion_scores = {}

        # 各基準での適合度計算
        for i, criterion in enumerate(self.criteria):
            user_val = user_preferences.get(criterion, 5.0)
            lab_val = lab_features.get(criterion, 5.0)

            similarity = self.calculate_similarity(
                user_val, lab_val, criterion)
            weighted_score = similarity * self.weights[i]

            criterion_scores[criterion] = {
                'similarity': similarity,
                'weighted_score': weighted_score,
                'user_preference': user_val,
                'lab_feature': lab_val,
                'weight': self.weights[i]
            }

        # 総合適合度計算
        total_score = sum(score['weighted_score']
                          for score in criterion_scores.values())
        overall_compatibility = total_score * 100  # 0-100スケール

        # 信頼度計算
        confidence = self._calculate_confidence(criterion_scores)

        return {
            'overall_score': round(overall_compatibility, 2),
            'criterion_scores': criterion_scores,
            'confidence': confidence,
            'weights_used': self.weights
        }

    def _calculate_confidence(self, criterion_scores: Dict) -> float:
        """推論結果の信頼度計算"""
        similarities = [score['similarity']
                        for score in criterion_scores.values()]

        # 平均類似度
        avg_similarity = sum(similarities) / len(similarities)

        # 分散計算（類似度のばらつき）
        variance = sum((s - avg_similarity) **
                       2 for s in similarities) / len(similarities)

        # 信頼度：分散が小さいほど信頼度が高い
        confidence = max(0, 1 - variance) * 100
        return round(confidence, 2)

    def generate_explanation(self, compatibility_result: Dict, user_prefs: Dict, lab_features: Dict) -> str:
        """判断根拠の説明文生成"""
        explanations = []

        # 高適合度・低適合度の基準を特定
        high_match_criteria = []
        low_match_criteria = []

        for criterion, score_data in compatibility_result['criterion_scores'].items():
            similarity = score_data['similarity']
            label = self.criteria_labels[criterion]

            if similarity >= 0.8:
                high_match_criteria.append(label)
            elif similarity < 0.5:
                low_match_criteria.append(label)

        # 総合評価コメント
        overall_score = compatibility_result['overall_score']
        if overall_score >= 85:
            explanations.append("🎯 非常に高い適合度を示しており、あなたの希望に極めてよく合致しています。")
        elif overall_score >= 70:
            explanations.append("✅ 高い適合度を示しており、あなたの希望によく合致しています。")
        elif overall_score >= 55:
            explanations.append("👍 適合度は良好です。いくつかの点で調整が必要かもしれません。")
        elif overall_score >= 40:
            explanations.append("⚠️ 適合度は中程度です。慎重に検討することをお勧めします。")
        else:
            explanations.append("❌ 適合度は低めです。他の研究室も検討することをお勧めします。")

        # 強み
        if high_match_criteria:
            explanations.append(
                f"特に【{', '.join(high_match_criteria)}】で高い適合度を示しています。")

        # 注意点
        if low_match_criteria:
            explanations.append(
                f"【{', '.join(low_match_criteria)}】については適合度が低いため、事前に詳しく確認することをお勧めします。")

        # 信頼度コメント
        confidence = compatibility_result['confidence']
        if confidence >= 80:
            explanations.append("この評価結果は高い信頼度を持っています。")
        elif confidence < 60:
            explanations.append("評価基準間でばらつきがあるため、総合的に判断してください。")

        return " ".join(explanations)


# 新しいHybridFuzzyEngineクラス
class HybridFuzzyEngine:
    """ハイブリッド型ファジィエンジン（既存 + 遺伝的最適化）- 完全修正版"""

    def __init__(self, models_dir: str = "models"):
        # 既存エンジン（フォールバック用）
        self.simple_engine = FuzzyLogicEngine()

        # 遺伝的最適化エンジン
        self.genetic_engine = None
        self.genetic_model_loaded = False
        self.models_dir = models_dir

        # 現在の動作モード
        self.current_mode = 'simple'  # 'simple' or 'genetic'

        # 説明エンジン
        try:
            from explanation_engine import FuzzyExplanationEngine
            self.explanation_engine = FuzzyExplanationEngine()
        except ImportError:
            self.explanation_engine = None

        # 遺伝的モデル読み込み試行
        self._load_genetic_model()

        print(f"🔧 HybridFuzzyEngine initialized - Mode: {self.current_mode}")

    def _load_genetic_model(self):
        """遺伝的最適化モデルの読み込み"""

        # 最適化結果ファイルを探索
        potential_paths = [
            os.path.join(self.models_dir, 'genetic_optimization_results.pkl'),
            os.path.join(self.models_dir, 'best_genetic_tree.pkl'),
            'genetic_optimization_results.pkl',
            'best_genetic_tree.pkl'
        ]

        # 最新のモデルファイルを探索
        if os.path.exists(self.models_dir):
            for filename in os.listdir(self.models_dir):
                if filename.endswith('_model.pkl') or filename.endswith('_model.pkl.gz'):
                    potential_paths.append(
                        os.path.join(self.models_dir, filename))

        for path in potential_paths:
            if os.path.exists(path):
                try:
                    # ファイル拡張子に応じて読み込み方法を変更
                    if path.endswith('.gz'):
                        import gzip
                        with gzip.open(path, 'rb') as f:
                            model_data = pickle.load(f)
                    else:
                        with open(path, 'rb') as f:
                            model_data = pickle.load(f)

                    # モデルタイプに応じて処理
                    if hasattr(model_data, 'tree'):
                        # Individual オブジェクトの場合
                        self.genetic_engine = model_data
                    elif isinstance(model_data, dict) and 'best_individual' in model_data:
                        # 最適化結果辞書の場合
                        self.genetic_engine = model_data['best_individual']
                    else:
                        # その他の形式
                        self.genetic_engine = model_data

                    self.genetic_model_loaded = True
                    self.current_mode = 'genetic'

                    print(f"✅ 遺伝的最適化モデル読み込み成功: {path}")

                    # モデル情報表示
                    if hasattr(self.genetic_engine, 'fitness_components'):
                        fitness = self.genetic_engine.fitness_components
                        if fitness:
                            print(f"🎯 Model fitness: {fitness.overall:.4f}")

                    break

                except Exception as e:
                    print(f"⚠️ モデル読み込み失敗 {path}: {e}")
                    continue

        if not self.genetic_model_loaded:
            print("📝 遺伝的最適化モデルが見つかりません。シンプルモードで動作します。")

    def predict_compatibility(self, user_preferences: Dict, lab_features: Dict) -> Tuple[Dict, str]:
        """統合予測メソッド"""

        if self.current_mode == 'genetic' and self.genetic_model_loaded:
            try:
                # 遺伝的最適化による予測
                result, explanation = self._genetic_predict(
                    user_preferences, lab_features)
                result['prediction_method'] = 'genetic_optimization'
                result['engine_version'] = '2.0'
                return result, explanation

            except Exception as e:
                print(f"⚠️ 遺伝的予測失敗、シンプルモードに切り替え: {e}")
                self.current_mode = 'simple'

        # シンプルエンジンによる予測（フォールバック）
        result = self.simple_engine.fuzzy_inference(
            user_preferences, lab_features)
        explanation = self.simple_engine.generate_explanation(
            result, user_preferences, lab_features)

        result['prediction_method'] = 'simple_fuzzy'
        result['engine_version'] = '1.0'

        return result, explanation

    def _genetic_predict(self, user_prefs: Dict, lab_features: Dict) -> Tuple[Dict, str]:
        """遺伝的最適化による予測（完全修正版）"""

        # 🔧 修正1: 特徴量ベクトル作成（user_prefs と lab_features から特徴量辞書を作成）
        features = {}
        criteria = ['research_intensity', 'advisor_style',
                    'team_work', 'workload', 'theory_practice']

        for criterion in criteria:
            user_val = user_prefs.get(criterion, 5.0)
            lab_val = lab_features.get(criterion, 5.0)

            # 類似度計算：ユーザー希望と研究室特徴の類似度を算出
            similarity = 1.0 - abs(user_val - lab_val) / 10.0
            features[criterion] = max(
                0.0, min(1.0, similarity)) * 10.0  # 0-10スケール

        # 遺伝的決定木で予測
        if hasattr(self.genetic_engine, 'tree') and self.genetic_engine.tree:
            try:
                # 🔧 修正2: 基本予測に特徴量辞書を使用
                prediction = self.genetic_engine.tree.predict(features)

                # 🔧 修正3: 説明付き予測に正しい引数を渡す
                try:
                    detailed_prediction, detailed_explanation = self.genetic_engine.tree.predict_with_explanation(
                        features,  # 特徴量辞書
                        criteria   # 特徴名リスト
                    )

                    # 結果フォーマット
                    result = {
                        'overall_score': prediction * 100,  # 0-100スケールに変換
                        'confidence': detailed_explanation.get('confidence', 0.8) * 100,
                        'criterion_scores': self._extract_criterion_scores_fixed(user_prefs, lab_features, detailed_explanation),
                        'decision_path': detailed_explanation.get('decision_steps', []),
                        'genetic_info': {
                            'individual_id': getattr(self.genetic_engine, 'individual_id', 'unknown'),
                            'generation': getattr(self.genetic_engine, 'generation', 0),
                            'fitness': getattr(self.genetic_engine, 'fitness_value', prediction) if hasattr(self.genetic_engine, 'fitness_value') else prediction
                        }
                    }

                    # 説明文生成
                    explanation = detailed_explanation.get(
                        'rationale', f'遺伝的最適化による予測: {prediction:.1%}')

                    return result, explanation

                except Exception as exp_error:
                    print(f"⚠️ 説明付き予測失敗、基本予測を使用: {exp_error}")

                    # フォールバック: 基本予測結果をフォーマット
                    result = {
                        'overall_score': prediction * 100,
                        'confidence': 75.0,  # デフォルト信頼度
                        'criterion_scores': self._create_basic_criterion_scores(user_prefs, lab_features),
                        'decision_path': [],
                        'genetic_info': {
                            'individual_id': getattr(self.genetic_engine, 'individual_id', 'unknown'),
                            'generation': getattr(self.genetic_engine, 'generation', 0),
                            'fitness': prediction
                        }
                    }

                    explanation = f"遺伝的最適化モデルによる予測結果: {prediction*100:.1f}%の適合度"
                    return result, explanation

            except Exception as pred_error:
                print(f"❌ 遺伝的予測失敗: {pred_error}")
                raise Exception(f"Genetic prediction failed: {pred_error}")

        else:
            # genetic_engineが適切でない場合
            raise Exception(
                "Genetic model structure invalid: no tree or tree is None")

    def _extract_criterion_scores_fixed(self, user_prefs: Dict, lab_features: Dict, detailed_explanation: Dict) -> Dict:
        """修正版基準別スコア抽出"""
        criterion_scores = {}

        criteria = ['research_intensity', 'advisor_style',
                    'team_work', 'workload', 'theory_practice']

        for criterion in criteria:
            user_val = user_prefs.get(criterion, 5.0)
            lab_val = lab_features.get(criterion, 5.0)

            # 基本的な類似度計算
            similarity = max(0.0, 1.0 - abs(user_val - lab_val) / 10.0)

            criterion_scores[criterion] = {
                'similarity': similarity,
                'weighted_score': similarity * 0.2,  # 等重み
                'user_preference': user_val,
                'lab_feature': lab_val,
                'weight': 0.2
            }

            # decision_stepsから詳細情報を抽出（可能な場合）
            decision_steps = detailed_explanation.get('decision_steps', [])
            for step in decision_steps:
                if hasattr(step, 'feature_name') and step.feature_name == criterion:
                    memberships = getattr(step, 'membership_evaluations', {})
                    if memberships:
                        max_membership = max(
                            details.get('membership_value', 0)
                            for details in memberships.values()
                        )
                        criterion_scores[criterion]['similarity'] = max_membership
                        criterion_scores[criterion]['weighted_score'] = max_membership * 0.2
                    break

        return criterion_scores

    def _create_basic_criterion_scores(self, user_prefs: Dict, lab_features: Dict) -> Dict:
        """基本的な基準別スコア作成"""
        criterion_scores = {}

        criteria = ['research_intensity', 'advisor_style',
                    'team_work', 'workload', 'theory_practice']
        for criterion in criteria:
            user_val = user_prefs.get(criterion, 5.0)
            lab_val = lab_features.get(criterion, 5.0)

            # 基本的なガウシアン類似度
            diff = abs(user_val - lab_val)
            similarity = math.exp(-0.5 * (diff / 2.0) ** 2)

            criterion_scores[criterion] = {
                'similarity': similarity,
                'weighted_score': similarity * 0.2,
                'user_preference': user_val,
                'lab_feature': lab_val,
                'weight': 0.2
            }

        return criterion_scores

    def switch_mode(self, mode: str) -> bool:
        """エンジンモード切り替え"""
        if mode == 'simple':
            self.current_mode = 'simple'
            print(f"🔄 エンジンモードを{mode}に切り替えました")
            return True
        elif mode == 'genetic' and self.genetic_model_loaded:
            self.current_mode = 'genetic'
            print(f"🔄 エンジンモードを{mode}に切り替えました")
            return True
        else:
            print(
                f"⚠️ モード'{mode}'に切り替えできません（genetic_model_loaded: {self.genetic_model_loaded}）")
            return False

    def get_engine_info(self) -> Dict:
        """エンジン情報取得"""
        info = {
            'current_mode': self.current_mode,
            'genetic_model_loaded': self.genetic_model_loaded,
            'genetic_available': self.genetic_model_loaded,
            'simple_available': True,
            'model_info': {
                'genetic': {
                    'individual_id': getattr(self.genetic_engine, 'individual_id', None) if self.genetic_engine else None,
                    'generation': getattr(self.genetic_engine, 'generation', None) if self.genetic_engine else None,
                    'fitness': self.genetic_engine.fitness_components.overall if (self.genetic_engine and hasattr(self.genetic_engine, 'fitness_components') and self.genetic_engine.fitness_components) else None,
                    'complexity': getattr(self.genetic_engine, 'complexity_score', None) if self.genetic_engine else None
                }
            }
        }

        return info

    def reload_genetic_model(self) -> bool:
        """遺伝的モデル再読み込み"""
        print("🔄 遺伝的モデルを再読み込み中...")

        # 現在のモデルをクリア
        self.genetic_engine = None
        self.genetic_model_loaded = False
        self.current_mode = 'simple'

        # 再読み込み試行
        self._load_genetic_model()

        return self.genetic_model_loaded

    def get_model_statistics(self) -> Dict:
        """モデル統計情報取得"""
        stats = {
            'simple_engine': {
                'criteria_count': len(self.simple_engine.criteria),
                'weights': dict(zip(self.simple_engine.criteria, self.simple_engine.weights)),
                'tolerances': self.simple_engine.tolerances
            },
            'genetic_engine': None
        }

        if self.genetic_model_loaded and self.genetic_engine:
            try:
                if hasattr(self.genetic_engine, 'tree') and self.genetic_engine.tree:
                    tree = self.genetic_engine.tree
                    stats['genetic_engine'] = {
                        'model_complexity': tree.calculate_complexity(),
                        'tree_depth': tree.calculate_depth(),
                        'individual_id': getattr(self.genetic_engine, 'individual_id', 'unknown'),
                        'generation': getattr(self.genetic_engine, 'generation', 0),
                        'fitness_components': self.genetic_engine.fitness_components.__dict__ if hasattr(self.genetic_engine, 'fitness_components') and self.genetic_engine.fitness_components else {}
                    }
            except Exception as e:
                stats['genetic_engine'] = {'error': str(e)}

        return stats


# エンジンファクトリー関数
def create_fuzzy_engine(engine_type: str = 'hybrid', **kwargs) -> Union[FuzzyLogicEngine, HybridFuzzyEngine]:
    """ファジィエンジン作成ファクトリー"""

    if engine_type == 'simple':
        return FuzzyLogicEngine()
    elif engine_type == 'hybrid':
        return HybridFuzzyEngine(**kwargs)
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")


# モジュール初期化時のデフォルトエンジン作成
try:
    default_engine = HybridFuzzyEngine()
    print("🎯 Default HybridFuzzyEngine created successfully")
except Exception as e:
    print(
        f"⚠️ HybridFuzzyEngine creation failed, falling back to simple engine: {e}")
    default_engine = FuzzyLogicEngine()


def get_default_engine() -> Union[FuzzyLogicEngine, HybridFuzzyEngine]:
    """デフォルトエンジン取得"""
    return default_engine
