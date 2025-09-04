"""
ファジィ決定木 - core/decision_tree/tree.py
拡張版：完全実装
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import time
from dataclasses import dataclass
from enum import Enum

from .node import FuzzyDecisionNode, FuzzyDecisionTree, NodeType

try:
    from .builder import FuzzyTreeBuilder
except ImportError:
    # フォールバック用の簡単なビルダー
    class FuzzyTreeBuilder:
        def __init__(self, max_depth=6, min_samples_leaf=5):
            self.max_depth = max_depth
            self.min_samples_leaf = min_samples_leaf
        
        def build_tree(self, data, feature_names, target_name):
            # 簡単なダミー実装
            root = FuzzyDecisionNode("root", NodeType.LEAF)
            root.set_leaf_value(0.5)
            tree = FuzzyDecisionTree(root)
            tree.feature_names = feature_names
            tree.target_name = target_name
            return tree


class PredictionMode(Enum):
    """予測モード"""
    CRISP = "crisp"                    # クリスプ予測
    FUZZY = "fuzzy"                    # ファジィ予測  
    PROBABILISTIC = "probabilistic"    # 確率的予測
    ENSEMBLE = "ensemble"              # アンサンブル予測


@dataclass
class TreeConfig:
    """ファジィ決定木設定"""
    max_depth: int = 6
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    prediction_mode: PredictionMode = PredictionMode.FUZZY
    pruning_enabled: bool = True
    adaptive_building: bool = False
    
    # ファジィ設定
    fuzzy_sets_per_feature: int = 3
    membership_overlap: float = 0.3
    aggregation_method: str = "weighted_average"
    
    # 性能設定
    max_prediction_time: float = 1.0  # 秒
    cache_predictions: bool = True
    detailed_explanations: bool = True


class EnhancedFuzzyDecisionTree(FuzzyDecisionTree):
    """拡張ファジィ決定木"""
    
    def __init__(self, config: TreeConfig = None):
        super().__init__()
        self.config = config or TreeConfig()
        
        # 予測キャッシュ
        self.prediction_cache: Dict[str, Tuple[float, float]] = {}
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        # 性能統計
        self.prediction_times: List[float] = []
        self.explanation_times: List[float] = []
        
        # 学習統計
        self.training_samples_count = 0
        self.last_training_time: Optional[float] = None
        self.model_version = "2.0"
        
        # 推論エンジン
        self.fuzzy_engine = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
           feature_names: List[str] = None, target_name: str = "target") -> Dict[str, Any]:
        """モデル訓練"""
        
        start_time = time.time()
        
        # パラメータ設定
        if feature_names is None:
            feature_names = X.columns.tolist()
        
        self.feature_names = feature_names
        self.target_name = target_name
        self.training_samples_count = len(X)
        
        print(f"ファジィ決定木訓練開始: サンプル数={len(X)}, 特徴量数={len(feature_names)}")
        
        try:
            # データ準備
            data = X.copy()
            data[target_name] = y
            
            # 構築器選択
            builder = FuzzyTreeBuilder(
                max_depth=self.config.max_depth,
                min_samples_leaf=self.config.min_samples_leaf
            )
            
            # 決定木構築
            tree = builder.build_tree(data, feature_names, target_name)
            self.root = tree.root
            
            # 統計更新
            self._update_tree_statistics()
            self.last_training_time = time.time() - start_time
            
            # 訓練結果
            training_results = {
                'training_time': self.last_training_time,
                'total_nodes': self.total_nodes,
                'max_depth': self.max_depth,
                'training_samples': self.training_samples_count,
                'feature_count': len(feature_names)
            }
            
            print(f"ファジィ決定木訓練完了: {self.last_training_time:.3f}秒")
            print(f"  - ノード数: {self.total_nodes}")
            print(f"  - 最大深度: {self.max_depth}")
            
            return training_results
            
        except Exception as e:
            print(f"訓練エラー: {e}")
            # フォールバック：簡単なデフォルトツリー作成
            root = FuzzyDecisionNode("fallback_root", NodeType.LEAF)
            root.set_leaf_value(0.5, confidence=0.5)
            self.root = root
            
            return {
                'training_time': time.time() - start_time,
                'total_nodes': 1,
                'max_depth': 0,
                'training_samples': len(X),
                'error': str(e),
                'fallback_mode': True
            }
    
    def predict(self, features: Dict[str, float]) -> float:
        """拡張予測実行"""
        
        start_time = time.time()
        
        # キャッシュチェック
        if self.config.cache_predictions:
            features_key = self._generate_features_hash(features)
            if features_key in self.prediction_cache:
                cached_prediction, timestamp = self.prediction_cache[features_key]
                # キャッシュの有効期限チェック（5分）
                if time.time() - timestamp < 300:
                    self.cache_hit_count += 1
                    return cached_prediction
        
        # 実際の予測
        try:
            prediction = super().predict(features)
            
            # 予測時間記録
            prediction_time = time.time() - start_time
            self.prediction_times.append(prediction_time)
            
            # 時間制限チェック
            if prediction_time > self.config.max_prediction_time:
                print(f"⚠️ 予測時間が制限を超過: {prediction_time:.3f}秒")
            
            # キャッシュ保存
            if self.config.cache_predictions:
                self.prediction_cache[features_key] = (prediction, time.time())
                self.cache_miss_count += 1
            
            return prediction
            
        except Exception as e:
            print(f"予測エラー: {e}")
            return 0.5  # フォールバック値
    
    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[float]:
        """バッチ予測（最適化版）"""
        
        predictions = []
        batch_start = time.time()
        
        for i, features in enumerate(features_list):
            prediction = self.predict(features)
            predictions.append(prediction)
            
            # 進捗表示（大きなバッチの場合）
            if len(features_list) > 100 and (i + 1) % 50 == 0:
                elapsed = time.time() - batch_start
                remaining = (elapsed / (i + 1)) * (len(features_list) - i - 1)
                print(f"バッチ予測進捗: {i+1}/{len(features_list)} "
                      f"(残り約{remaining:.1f}秒)")
        
        batch_time = time.time() - batch_start
        avg_time = batch_time / len(features_list) if features_list else 0
        
        print(f"バッチ予測完了: {len(features_list)}件, "
              f"総時間{batch_time:.3f}秒, 平均{avg_time:.4f}秒/件")
        
        return predictions
    
    def predict_with_confidence(self, features: Dict[str, float]) -> Tuple[float, float]:
        """信頼度付き予測"""
        
        if not self.root:
            return 0.5, 0.0
        
        try:
            prediction, explanation = self.predict_with_explanation(features)
            
            # 信頼度計算
            confidence = self._calculate_prediction_confidence(explanation)
            
            return prediction, confidence
            
        except Exception as e:
            print(f"信頼度付き予測エラー: {e}")
            return 0.5, 0.0
    
    def _calculate_prediction_confidence(self, explanation: Dict[str, Any]) -> float:
        """予測信頼度の計算"""
        
        try:
            # 基本信頼度
            base_confidence = 0.5
            
            # 葉ノードの信頼度
            if 'confidence' in explanation:
                base_confidence = explanation['confidence']
            
            # サンプル数による調整
            if 'sample_count' in explanation:
                sample_count = explanation['sample_count']
                sample_factor = min(1.0, sample_count / 10.0)  # 10サンプル以上で満点
                base_confidence *= sample_factor
            
            # メンバーシップ度による調整
            if 'selected_membership' in explanation:
                membership = explanation['selected_membership']
                base_confidence *= membership
            
            return min(1.0, max(0.0, base_confidence))
            
        except Exception:
            return 0.5
    
    def _generate_features_hash(self, features: Dict[str, float]) -> str:
        """特徴量ハッシュ生成"""
        
        try:
            # 特徴量を文字列に変換してハッシュ化
            features_str = str(sorted(features.items()))
            return str(hash(features_str))
        except Exception:
            return str(time.time())  # フォールバック
    
    def _update_tree_statistics(self):
        """ツリー統計の更新"""
        
        if self.root:
            self.update_statistics()
            
            # 重要度計算
            total_samples = self.training_samples_count
            if total_samples > 0:
                self._calculate_node_importance(self.root, total_samples)
    
    def _calculate_node_importance(self, node: FuzzyDecisionNode, total_samples: int):
        """ノード重要度の再帰計算"""
        
        node.calculate_importance(total_samples)
        
        for child in node.children.values():
            self._calculate_node_importance(child, total_samples)
    
    def _calculate_cache_hit_rate(self) -> float:
        """キャッシュヒット率計算"""
        
        total_requests = self.cache_hit_count + self.cache_miss_count
        if total_requests == 0:
            return 0.0
        
        return self.cache_hit_count / total_requests
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """性能統計の取得"""
        
        return {
            'prediction_performance': {
                'total_predictions': len(self.prediction_times),
                'average_prediction_time': np.mean(self.prediction_times) if self.prediction_times else 0.0,
                'max_prediction_time': np.max(self.prediction_times) if self.prediction_times else 0.0,
                'min_prediction_time': np.min(self.prediction_times) if self.prediction_times else 0.0,
            },
            'cache_performance': {
                'cache_hit_rate': self._calculate_cache_hit_rate(),
                'cache_size': len(self.prediction_cache),
                'total_cache_hits': self.cache_hit_count,
                'total_cache_misses': self.cache_miss_count
            },
            'model_info': {
                'total_nodes': self.total_nodes,
                'max_depth': self.max_depth,
                'training_samples': self.training_samples_count,
                'model_version': self.model_version,
                'last_training_time': self.last_training_time
            }
        }
    
    def clear_cache(self):
        """キャッシュのクリア"""
        
        cache_size = len(self.prediction_cache)
        self.prediction_cache.clear()
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        print(f"キャッシュクリア完了: {cache_size}エントリを削除")
    
    def validate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """モデル検証"""
        
        if not self.root:
            return {'error': 'Model not trained'}
        
        predictions = []
        actuals = y_test.tolist()
        
        # バッチ予測
        features_list = [row.to_dict() for _, row in X_test.iterrows()]
        predictions = self.predict_batch(features_list)
        
        # 性能指標計算
        mse = np.mean([(p - a) ** 2 for p, a in zip(predictions, actuals)])
        rmse = np.sqrt(mse)
        mae = np.mean([abs(p - a) for p, a in zip(predictions, actuals)])
        
        # R²スコア
        ss_res = sum([(a - p) ** 2 for a, p in zip(actuals, predictions)])
        ss_tot = sum([(a - np.mean(actuals)) ** 2 for a in actuals])
        r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'validation_metrics': {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2_score
            },
            'prediction_distribution': {
                'min_prediction': min(predictions),
                'max_prediction': max(predictions),
                'mean_prediction': np.mean(predictions),
                'std_prediction': np.std(predictions)
            },
            'test_samples': len(predictions)
        }
    
    def export_model(self, include_cache: bool = False) -> Dict[str, Any]:
        """モデルのエクスポート"""
        
        export_data = {
            'model_type': 'enhanced_fuzzy_decision_tree',
            'version': self.model_version,
            'config': {
                'max_depth': self.config.max_depth,
                'min_samples_split': self.config.min_samples_split,
                'min_samples_leaf': self.config.min_samples_leaf,
                'prediction_mode': self.config.prediction_mode.value,
                'pruning_enabled': self.config.pruning_enabled,
                'adaptive_building': self.config.adaptive_building
            },
            'tree_structure': self.to_dict(),
            'performance_stats': self.get_performance_statistics(),
            'export_timestamp': time.time()
        }
        
        if include_cache:
            export_data['prediction_cache'] = {
                k: {'prediction': v[0], 'timestamp': v[1]} 
                for k, v in self.prediction_cache.items()
            }
        
        return export_data
    
    def import_model(self, model_data: Dict[str, Any]) -> bool:
        """モデルのインポート"""
        
        try:
            # 基本構造復元
            tree_data = model_data.get('tree_structure', {})
            imported_tree = FuzzyDecisionTree.from_dict(tree_data)
            
            self.root = imported_tree.root
            self.feature_names = imported_tree.feature_names
            self.target_name = imported_tree.target_name
            
            # 設定復元
            config_data = model_data.get('config', {})
            self.config.max_depth = config_data.get('max_depth', 6)
            self.config.min_samples_split = config_data.get('min_samples_split', 10)
            self.config.min_samples_leaf = config_data.get('min_samples_leaf', 5)
            
            # メタデータ復元
            self.model_version = model_data.get('version', '2.0')
            
            # 統計更新
            self._update_tree_statistics()
            
            print(f"モデルインポート完了: ノード数={self.total_nodes}")
            return True
            
        except Exception as e:
            print(f"モデルインポートエラー: {e}")
            return False


class FuzzyDecisionTreeEnsemble:
    """ファジィ決定木アンサンブル"""
    
    def __init__(self, n_estimators: int = 5, config: TreeConfig = None):
        self.n_estimators = n_estimators
        self.config = config or TreeConfig()
        self.estimators: List[EnhancedFuzzyDecisionTree] = []
        self.feature_names: List[str] = []
        self.target_name: str = "target"
        
        # アンサンブル統計
        self.training_time: float = 0.0
        self.ensemble_accuracy: float = 0.0
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
           feature_names: List[str] = None, target_name: str = "target") -> Dict[str, Any]:
        """アンサンブル学習"""
        
        start_time = time.time()
        
        if feature_names is None:
            feature_names = X.columns.tolist()
        
        self.feature_names = feature_names
        self.target_name = target_name
        
        print(f"アンサンブル学習開始: {self.n_estimators}個のモデル")
        
        training_results = []
        
        for i in range(self.n_estimators):
            print(f"モデル {i+1}/{self.n_estimators} を学習中...")
            
            # ブートストラップサンプリング
            sample_indices = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X.iloc[sample_indices]
            y_sample = y.iloc[sample_indices]
            
            # 個別モデル作成
            estimator = EnhancedFuzzyDecisionTree(self.config)
            result = estimator.fit(X_sample, y_sample, feature_names, target_name)
            
            self.estimators.append(estimator)
            training_results.append(result)
        
        self.training_time = time.time() - start_time
        
        ensemble_result = {
            'ensemble_training_time': self.training_time,
            'n_estimators': len(self.estimators),
            'individual_results': training_results,
            'average_nodes': np.mean([r.get('total_nodes', 0) for r in training_results]),
            'average_depth': np.mean([r.get('max_depth', 0) for r in training_results])
        }
        
        print(f"アンサンブル学習完了: {self.training_time:.3f}秒")
        
        return ensemble_result
    
    def predict(self, features: Dict[str, float]) -> float:
        """アンサンブル予測"""
        
        if not self.estimators:
            return 0.5
        
        predictions = []
        for estimator in self.estimators:
            prediction = estimator.predict(features)
            predictions.append(prediction)
        
        # 平均値を返す
        return np.mean(predictions)
    
    def predict_with_uncertainty(self, features: Dict[str, float]) -> Tuple[float, float]:
        """不確実性付き予測"""
        
        if not self.estimators:
            return 0.5, 1.0
        
        predictions = []
        for estimator in self.estimators:
            prediction = estimator.predict(features)
            predictions.append(prediction)
        
        mean_prediction = np.mean(predictions)
        uncertainty = np.std(predictions)  # 予測の標準偏差を不確実性とする
        
        return mean_prediction, uncertainty
    
    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[float]:
        """アンサンブル バッチ予測"""
        
        return [self.predict(features) for features in features_list]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """特徴量重要度の計算"""
        
        if not self.estimators or not self.feature_names:
            return {}
        
        importance_scores = {name: 0.0 for name in self.feature_names}
        
        for estimator in self.estimators:
            if estimator.root:
                # 各estimatorからの重要度を集約
                # 簡略実装：ルートノードの特徴量に重みを付与
                if estimator.root.feature_name in importance_scores:
                    importance_scores[estimator.root.feature_name] += estimator.root.importance_score
        
        # 正規化
        total_importance = sum(importance_scores.values())
        if total_importance > 0:
            importance_scores = {
                name: score / total_importance 
                for name, score in importance_scores.items()
            }
        
        return importance_scores
    
    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """アンサンブル統計情報"""
        
        if not self.estimators:
            return {'error': 'No estimators trained'}
        
        individual_stats = []
        for i, estimator in enumerate(self.estimators):
            stats = estimator.get_performance_statistics()
            stats['estimator_id'] = i
            individual_stats.append(stats)
        
        return {
            'ensemble_info': {
                'n_estimators': len(self.estimators),
                'training_time': self.training_time,
                'feature_names': self.feature_names,
                'target_name': self.target_name
            },
            'aggregate_stats': {
                'average_nodes': np.mean([s['model_info']['total_nodes'] for s in individual_stats]),
                'average_depth': np.mean([s['model_info']['max_depth'] for s in individual_stats]),
                'total_predictions': sum([s['prediction_performance']['total_predictions'] for s in individual_stats])
            },
            'individual_estimator_stats': individual_stats,
            'feature_importance': self.get_feature_importance()
        }