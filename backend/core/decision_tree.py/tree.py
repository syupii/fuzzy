"""
ファジィ決定木 - core/decision_tree/tree.py  
ファジィ決定木の統合クラスと高レベルインターフェース
"""

from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np
import pandas as pd
import time
import json
from dataclasses import dataclass
from enum import Enum

from .node import FuzzyDecisionNode, FuzzyDecisionTree, NodeType
from .builder import FuzzyTreeBuilder, AdaptiveFuzzyTreeBuilder, TreePruner
from ..fuzzy.inference import SimpleFuzzyInferenceEngine


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
        self.prediction_cache: Dict[str, Tuple[float, float]] = {}  # features_hash -> (prediction, timestamp)
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        # 性能統計
        self.prediction_times: List[float] = []
        self.explanation_times: List[float] = []
        
        # 学習統計
        self.training_samples_count = 0
        self.last_training_time: Optional[float] = None
        self.model_version = "1.0"
        
        # 予測モード別の推論エンジン
        self.fuzzy_engine: Optional[SimpleFuzzyInferenceEngine] = None
        
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
            if self.config.adaptive_building:
                builder = AdaptiveFuzzyTreeBuilder(
                    max_depth=self.config.max_depth,
                    min_samples_leaf=self.config.min_samples_leaf
                )
            else:
                builder = FuzzyTreeBuilder(
                    max_depth=self.config.max_depth,
                    min_samples_leaf=self.config.min_samples_leaf
                )
            
            # 決定木構築
            tree = builder.build_tree(data, feature_names, target_name)
            self.root = tree.root
            
            # 剪定
            if self.config.pruning_enabled:
                pruner = TreePruner(
                    min_samples_leaf=self.config.min_samples_leaf,
                    max_depth=self.config.max_depth
                )
                pruned_count = pruner.prune_tree(self.root)
                print(f"剪定完了: {pruned_count}ノードを剪定")
            
            # ファジィ推論エンジン初期化
            if self.config.prediction_mode == PredictionMode.FUZZY:
                self.fuzzy_engine = SimpleFuzzyInferenceEngine(feature_names, target_name)
            
            # 統計更新
            self._update_tree_statistics()
            self.last_training_time = time.time() - start_time
            
            training_result = {
                'success': True,
                'training_time': self.last_training_time,
                'tree_info': self.get_tree_info(),
                'builder_stats': builder.get_builder_statistics()
            }
            
            print(f"訓練完了: {self.last_training_time:.2f}秒, ノード数={self.total_nodes}")
            
            return training_result
            
        except Exception as e:
            print(f"訓練エラー: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'training_time': time.time() - start_time
            }
    
    def predict(self, features: Dict[str, float]) -> float:
        """予測実行（キャッシュ対応）"""
        
        start_time = time.time()
        
        try:
            # キャッシュチェック
            if self.config.cache_predictions:
                prediction = self._get_cached_prediction(features)
                if prediction is not None:
                    self.cache_hit_count += 1
                    return prediction
                
                self.cache_miss_count += 1
            
            # 予測実行
            if self.config.prediction_mode == PredictionMode.FUZZY and self.fuzzy_engine:
                prediction = self.fuzzy_engine.infer(features)
            elif self.root:
                prediction = self.root.predict(features)
            else:
                prediction = 0.5
            
            # キャッシュ更新
            if self.config.cache_predictions:
                self._update_prediction_cache(features, prediction)
            
            # 統計更新
            prediction_time = time.time() - start_time
            self.prediction_times.append(prediction_time)
            
            return prediction
            
        except Exception as e:
            print(f"予測エラー: {e}")
            return 0.5
    
    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[float]:
        """バッチ予測"""
        
        predictions = []
        
        for features in features_list:
            prediction = self.predict(features)
            predictions.append(prediction)
        
        return predictions
    
    def predict_with_explanation(self, features: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
        """詳細説明付き予測"""
        
        start_time = time.time()
        
        if not self.root:
            return 0.5, {'error': 'Model not trained'}
        
        try:
            # 基本予測と説明
            prediction, node_explanation = self.root.predict_with_explanation(features)
            
            # 拡張説明情報
            enhanced_explanation = {
                'prediction': prediction,
                'model_info': {
                    'model_type': 'enhanced_fuzzy_decision_tree',
                    'prediction_mode': self.config.prediction_mode.value,
                    'tree_nodes': self.total_nodes,
                    'tree_depth': self.max_depth,
                    'training_samples': self.training_samples_count,
                    'model_version': self.model_version
                },
                'prediction_details': node_explanation,
                'confidence_analysis': self._analyze_prediction_confidence(features, prediction),
                'feature_sensitivity': self._analyze_feature_sensitivity(features) if self.config.detailed_explanations else {},
                'cache_info': {
                    'cache_hit_rate': self._calculate_cache_hit_rate(),
                    'prediction_from_cache': False  # この予測はキャッシュから来ていない
                }
            }
            
            # 説明時間記録
            explanation_time = time.time() - start_time
            self.explanation_times.append(explanation_time)
            
            return prediction, enhanced_explanation
            
        except Exception as e:
            print(f"説明付き予測エラー: {e}")
            return 0.5, {'error': str(e)}
    
    def _get_cached_prediction(self, features: Dict[str, float]) -> Optional[float]:
        """キャッシュから予測を取得"""
        
        features_hash = self._hash_features(features)
        
        if features_hash in self.prediction_cache:
            prediction, timestamp = self.prediction_cache[features_hash]
            
            # キャッシュの有効性チェック（5分間有効）
            if time.time() - timestamp < 300:
                return prediction
            else:
                # 期限切れキャッシュを削除
                del self.prediction_cache[features_hash]
        
        return None
    
    def _update_prediction_cache(self, features: Dict[str, float], prediction: float):
        """予測キャッシュの更新"""
        
        features_hash = self._hash_features(features)
        self.prediction_cache[features_hash] = (prediction, time.time())
        
        # キャッシュサイズ制限
        if len(self.prediction_cache) > 1000:
            # 最も古いエントリを削除
            oldest_hash = min(self.prediction_cache.keys(), 
                            key=lambda k: self.prediction_cache[k][1])
            del self.prediction_cache[oldest_hash]
    
    def _hash_features(self, features: Dict[str, float]) -> str:
        """特徴量のハッシュ値計算"""
        
        # 特徴量を精度2桁で丸めてハッシュ化
        rounded_features = {k: round(v, 2) for k, v in features.items()}
        feature_str = str(sorted(rounded_features.items()))
        return str(hash(feature_str))
    
    def _analyze_prediction_confidence(self, features: Dict[str, float], 
                                     prediction: float) -> Dict[str, Any]:
        """予測信頼度の分析"""
        
        if not self.root:
            return {'confidence': 0.0, 'factors': []}
        
        # 基本信頼度
        base_confidence = self.root._calculate_confidence(features)
        
        # 信頼度要因分析
        confidence_factors = []
        
        # 1. 訓練データとの距離
        if self.root.training_samples:
            min_distance = float('inf')
            for sample_features, _ in self.root.training_samples:
                distance = sum((features.get(k, 0) - sample_features.get(k, 0))**2 
                             for k in set(features.keys()) | set(sample_features.keys()))
                min_distance = min(min_distance, distance)
            
            distance_confidence = 1.0 / (1.0 + min_distance * 0.1)
            confidence_factors.append({
                'factor': 'training_data_proximity',
                'score': distance_confidence,
                'description': f'最近傍訓練サンプルとの距離: {min_distance:.2f}'
            })
        
        # 2. 特徴量の範囲内かチェック
        range_confidence = 1.0
        out_of_range_features = []
        
        for feature, value in features.items():
            if feature in self.feature_names:
                # 簡易範囲チェック（0-10の範囲を想定）
                if value < 0 or value > 10:
                    range_confidence *= 0.8
                    out_of_range_features.append(feature)
        
        if out_of_range_features:
            confidence_factors.append({
                'factor': 'feature_range_validity',
                'score': range_confidence,
                'description': f'範囲外特徴量: {out_of_range_features}'
            })
        
        # 3. 予測の極端さ
        extremeness_penalty = 0.0
        if prediction < 0.1 or prediction > 0.9:
            extremeness_penalty = abs(prediction - 0.5) * 0.2
        
        extremeness_confidence = 1.0 - extremeness_penalty
        confidence_factors.append({
            'factor': 'prediction_extremeness',
            'score': extremeness_confidence,
            'description': f'予測値の極端度: {extremeness_penalty:.2f}'
        })
        
        # 総合信頼度
        factor_scores = [f['score'] for f in confidence_factors]
        overall_confidence = base_confidence * np.mean(factor_scores) if factor_scores else base_confidence
        
        return {
            'confidence': overall_confidence,
            'base_confidence': base_confidence,
            'factors': confidence_factors,
            'confidence_level': self._categorize_confidence(overall_confidence)
        }
    
    def _categorize_confidence(self, confidence: float) -> str:
        """信頼度のカテゴリ化"""
        
        if confidence >= 0.8:
            return "高"
        elif confidence >= 0.6:
            return "中"
        elif confidence >= 0.4:
            return "低"
        else:
            return "非常に低"
    
    def _analyze_feature_sensitivity(self, features: Dict[str, float]) -> Dict[str, float]:
        """特徴量感度分析"""
        
        if not self.root:
            return {}
        
        baseline_prediction = self.root.predict(features)
        sensitivity = {}
        
        # 各特徴量を少し変動させて感度を測定
        perturbation = 0.1
        
        for feature in features:
            if feature in self.feature_names:
                # 正方向の変動
                modified_features = features.copy()
                modified_features[feature] += perturbation
                pos_prediction = self.root.predict(modified_features)
                
                # 負方向の変動
                modified_features[feature] = features[feature] - perturbation
                neg_prediction = self.root.predict(modified_features)
                
                # 感度計算（変動に対する予測値の変化率）
                sensitivity_score = abs(pos_prediction - neg_prediction) / (2 * perturbation)
                sensitivity[feature] = sensitivity_score
        
        return sensitivity
    
    def _calculate_cache_hit_rate(self) -> float:
        """キャッシュヒット率の計算"""
        
        total_requests = self.cache_hit_count + self.cache_miss_count
        if total_requests == 0:
            return 0.0
        
        return self.cache_hit_count / total_requests
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """性能統計の取得"""
        
        stats = {
            'prediction_performance': {
                'total_predictions': len(self.prediction_times),
                'average_prediction_time': np.mean(self.prediction_times) if self.prediction_times else 0.0,
                'max_prediction_time': max(self.prediction_times) if self.prediction_times else 0.0,
                'prediction_time_std': np.std(self.prediction_times) if self.prediction_times else 0.0
            },
            'cache_performance': {
                'cache_hit_rate': self._calculate_cache_hit_rate(),
                'cache_size': len(self.prediction_cache),
                'total_cache_hits': self.cache_hit_count,
                'total_cache_misses': self.cache_miss_count
            },
            'explanation_performance': {
                'total_explanations': len(self.explanation_times),
                'average_explanation_time': np.mean(self.explanation_times) if self.explanation_times else 0.0
            },
            'model_info': {
                'training_samples': self.training_samples_count,
                'last_training_time': self.last_training_time,
                'tree_nodes': self.total_nodes,
                'tree_depth': self.max_depth,
                'model_version': self.model_version
            }
        }
        
        return stats
    
    def validate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """モデル検証"""
        
        if not self.root:
            return {'error': 'Model not trained'}
        
        predictions = []
        actuals = y_test.tolist()
        
        # バッチ予測
        for idx, row in X_test.iterrows():
            features = row.to_dict()
            prediction = self.predict(features)
            predictions.append(prediction)
        
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
            self.model_version = model_data.get('version', '1.0')
            
            # 統計更新
            self._update_tree_statistics()
            
            print(f"モデルインポート完了: ノード数={self.total_nodes}")
            return True
            
        except Exception as e:
            print(f"モデルインポートエラー: {e}")
            return False
    
    def clear_cache(self):
        """キャッシュのクリア"""
        
        cache_size = len(self.prediction_cache)
        self.prediction_cache.clear()
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        
        print(f"キャッシュクリア完了: {cache_size}エントリを削除")
    
    def optimize_model(self) -> Dict[str, Any]:
        """モデル最適化"""
        
        optimization_results = {}
        
        # 1. キャッシュ最適化
        old_cache_size = len(self.prediction_cache)
        current_time = time.time()
        
        # 古いキャッシュエントリを削除（10分以上古い）
        expired_keys = []
        for key, (_, timestamp) in self.prediction_cache.items():
            if current_time - timestamp > 600:  # 10分
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.prediction_cache[key]
        
        optimization_results['cache_optimization'] = {
            'expired_entries_removed': len(expired_keys),
            'cache_size_before': old_cache_size,
            'cache_size_after': len(self.prediction_cache)
        }
        
        # 2. 統計履歴の圧縮
        if len(self.prediction_times) > 1000:
            # 最新の1000件のみ保持
            self.prediction_times = self.prediction_times[-1000:]
            optimization_results['prediction_times_compressed'] = True
        
        if len(self.explanation_times) > 1000:
            self.explanation_times = self.explanation_times[-1000:]
            optimization_results['explanation_times_compressed'] = True
        
        # 3. ツリー統計更新
        self._update_tree_statistics()
        optimization_results['tree_statistics_updated'] = True
        
        return optimization_results


class FuzzyDecisionTreeEnsemble:
    """ファジィ決定木アンサンブル"""
    
    def __init__(self, n_trees: int = 5, config: TreeConfig = None):
        self.n_trees = n_trees
        self.config = config or TreeConfig()
        self.trees: List[EnhancedFuzzyDecisionTree] = []
        self.tree_weights: List[float] = []
        self.is_trained = False
    
    def fit(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str] = None) -> Dict[str, Any]:
        """アンサンブル訓練"""
        
        start_time = time.time()
        
        print(f"ファジィ決定木アンサンブル訓練開始: {self.n_trees}本の木")
        
        self.trees = []
        self.tree_weights = []
        training_results = []
        
        for i in range(self.n_trees):
            print(f"木 {i+1}/{self.n_trees} 訓練中...")
            
            # 個別設定（多様性のため）
            tree_config = TreeConfig(
                max_depth=self.config.max_depth + random.randint(-1, 1),
                min_samples_split=max(5, self.config.min_samples_split + random.randint(-2, 2)),
                min_samples_leaf=max(3, self.config.min_samples_leaf + random.randint(-1, 1)),
                prediction_mode=self.config.prediction_mode
            )
            
            tree = EnhancedFuzzyDecisionTree(tree_config)
            
            # ブートストラップサンプリング
            sample_indices = np.random.choice(len(X), size=len(X), replace=True)
            X_bootstrap = X.iloc[sample_indices]
            y_bootstrap = y.iloc[sample_indices]
            
            # 訓練
            result = tree.fit(X_bootstrap, y_bootstrap, feature_names)
            
            if result['success']:
                self.trees.append(tree)
                
                # 重み計算（訓練誤差の逆数）
                train_predictions = []
                for _, row in X_bootstrap.iterrows():
                    pred = tree.predict(row.to_dict())
                    train_predictions.append(pred)
                
                mse = np.mean([(p - a) ** 2 for p, a in zip(train_predictions, y_bootstrap)])
                weight = 1.0 / (1.0 + mse)
                self.tree_weights.append(weight)
                
                training_results.append(result)
            else:
                print(f"木 {i+1} の訓練に失敗")
        
        # 重みの正規化
        if self.tree_weights:
            total_weight = sum(self.tree_weights)
            self.tree_weights = [w / total_weight for w in self.tree_weights]
        
        self.is_trained = len(self.trees) > 0
        training_time = time.time() - start_time
        
        result = {
            'success': self.is_trained,
            'trained_trees': len(self.trees),
            'training_time': training_time,
            'individual_results': training_results
        }
        
        print(f"アンサンブル訓練完了: {len(self.trees)}本中{len(self.trees)}本成功")
        
        return result
    
    def predict(self, features: Dict[str, float]) -> float:
        """アンサンブル予測"""
        
        if not self.is_trained:
            return 0.5
        
        # 各木の予測を重み付き平均
        weighted_predictions = []
        
        for tree, weight in zip(self.trees, self.tree_weights):
            prediction = tree.predict(features)
            weighted_predictions.append(prediction * weight)
        
        return sum(weighted_predictions)
    
    def predict_with_uncertainty(self, features: Dict[str, float]) -> Tuple[float, float]:
        """不確実性付き予測"""
        
        if not self.is_trained:
            return 0.5, 0.0
        
        predictions = []
        
        for tree in self.trees:
            prediction = tree.predict(features)
            predictions.append(prediction)
        
        mean_prediction = np.mean(predictions)
        uncertainty = np.std(predictions)
        
        return mean_prediction, uncertainty
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """アンサンブル情報"""
        
        return {
            'n_trees': len(self.trees),
            'tree_weights': self.tree_weights,
            'is_trained': self.is_trained,
            'individual_tree_info': [
                {
                    'tree_index': i,
                    'weight': weight,
                    'nodes': tree.total_nodes,
                    'depth': tree.max_depth
                }
                for i, (tree, weight) in enumerate(zip(self.trees, self.tree_weights))
            ]
        }