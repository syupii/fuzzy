import os
import sys
import numpy as np
import pandas as pd
import pickle
import gzip
import json
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Windows文字エンコーディング設定
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print(f"[SYSTEM] 完全版遺伝的ファジィ決定木システム Part 1 読み込み中...")

# =============================================================================
# メンバーシップ関数とファジィ論理の実装
# =============================================================================

class MembershipType(Enum):
    """メンバーシップ関数タイプ"""
    TRIANGULAR = "triangular"
    GAUSSIAN = "gaussian"
    TRAPEZOIDAL = "trapezoidal"

class MembershipFunction:
    """高度なメンバーシップ関数クラス"""
    
    def __init__(self, name: str, mf_type: MembershipType, parameters: List[float]):
        self.name = name
        self.mf_type = mf_type
        self.parameters = parameters
        self.activation_count = 0
        self.total_membership = 0.0
    
    def membership(self, value: float) -> float:
        """メンバーシップ度計算"""
        try:
            if self.mf_type == MembershipType.TRIANGULAR:
                result = self._triangular_membership(value)
            elif self.mf_type == MembershipType.GAUSSIAN:
                result = self._gaussian_membership(value)
            elif self.mf_type == MembershipType.TRAPEZOIDAL:
                result = self._trapezoidal_membership(value)
            else:
                result = 0.0
            
            # 統計更新
            if result > 0.1:
                self.activation_count += 1
                self.total_membership += result
            
            return max(0.0, min(1.0, result))
        except:
            return 0.0
    
    def _triangular_membership(self, value: float) -> float:
        """三角メンバーシップ関数"""
        if len(self.parameters) < 3:
            return 0.0
        
        a, b, c = self.parameters[:3]
        
        if value <= a or value >= c:
            return 0.0
        elif value == b:
            return 1.0
        elif value < b:
            return (value - a) / (b - a) if b != a else 0.0
        else:
            return (c - value) / (c - b) if c != b else 0.0
    
    def _gaussian_membership(self, value: float) -> float:
        """ガウシアンメンバーシップ関数"""
        if len(self.parameters) < 2:
            return 0.0
        
        center, sigma = self.parameters[:2]
        if sigma <= 0:
            sigma = 0.1
        return np.exp(-0.5 * ((value - center) / sigma) ** 2)
    
    def _trapezoidal_membership(self, value: float) -> float:
        """台形メンバーシップ関数"""
        if len(self.parameters) < 4:
            return 0.0
        
        a, b, c, d = self.parameters[:4]
        
        if value <= a or value >= d:
            return 0.0
        elif b <= value <= c:
            return 1.0
        elif a < value < b:
            return (value - a) / (b - a) if b != a else 0.0
        else:  # c < value < d
            return (d - value) / (d - c) if d != c else 0.0
    
    def get_statistics(self) -> Dict[str, float]:
        """統計情報取得"""
        return {
            'activation_count': self.activation_count,
            'average_membership': self.total_membership / max(1, self.activation_count),
            'utilization': self.activation_count / max(1, self.activation_count + 100)
        }

# =============================================================================
# 高度なファジィ決定ノード
# =============================================================================

class AdvancedFuzzyDecisionNode:
    """高度なファジィ決定ノード"""
    
    def __init__(self, node_id: str = None, is_leaf: bool = False, 
                 feature_name: str = None, leaf_value: float = None):
        self.node_id = node_id or f"node_{int(time.time())}_{random.randint(1000, 9999)}"
        self.is_leaf = is_leaf
        self.feature_name = feature_name
        self.leaf_value = leaf_value
        
        # メンバーシップ関数と子ノード
        self.membership_functions: Dict[str, MembershipFunction] = {}
        self.children: Dict[str, 'AdvancedFuzzyDecisionNode'] = {}
        
        # ノード統計
        self.depth = 0
        self.sample_count = 0
        self.weighted_target_sum = 0.0
        self.prediction_count = 0
        self.total_prediction_value = 0.0
        
        # 学習用データ
        self.training_samples = []
        self.training_targets = []
        
        # 重要度スコア
        self.importance_score = 0.0
        self.split_quality = 0.0
    
    def add_membership_function(self, label: str, mf: MembershipFunction):
        """メンバーシップ関数追加"""
        self.membership_functions[label] = mf
    
    def add_child(self, label: str, child: 'AdvancedFuzzyDecisionNode'):
        """子ノード追加"""
        self.children[label] = child
        if child:
            child.depth = self.depth + 1
    
    def predict(self, features: Dict[str, float]) -> float:
        """高度なファジィ推論予測"""
        self.prediction_count += 1
        
        if self.is_leaf:
            prediction = self.leaf_value if self.leaf_value is not None else 0.5
            self.total_prediction_value += prediction
            return prediction
        
        if not self.feature_name or self.feature_name not in features:
            default_prediction = 0.5
            self.total_prediction_value += default_prediction
            return default_prediction
        
        feature_value = features[self.feature_name]
        
        # 高度なファジィ推論
        weighted_predictions = []
        total_weight = 0.0
        
        for label, mf in self.membership_functions.items():
            membership_degree = mf.membership(feature_value)
            
            if membership_degree > 0.01 and label in self.children:
                child_prediction = self.children[label].predict(features)
                
                # 動的重み調整
                adjusted_weight = membership_degree * (1 + self.children[label].importance_score)
                
                weighted_predictions.append((child_prediction, adjusted_weight))
                total_weight += adjusted_weight
        
        # 予測値計算
        if total_weight > 0:
            final_prediction = sum(pred * weight for pred, weight in weighted_predictions) / total_weight
        else:
            final_prediction = 0.5
        
        # 不確実性を考慮した調整
        uncertainty_factor = 1.0 / (1.0 + total_weight)
        final_prediction = final_prediction * (1 - uncertainty_factor) + 0.5 * uncertainty_factor
        
        self.total_prediction_value += final_prediction
        return max(0.0, min(1.0, final_prediction))
    
    def predict_with_explanation(self, features: Dict[str, float], 
                               feature_names: List[str]) -> Tuple[float, Dict[str, Any]]:
        """詳細説明付き予測"""
        prediction = self.predict(features)
        decision_path = []
        
        # 決定パスの構築
        current_node = self
        visited_nodes = []
        
        while not current_node.is_leaf and len(visited_nodes) < 20:
            visited_nodes.append(current_node)
            
            if current_node.feature_name and current_node.feature_name in features:
                feature_value = features[current_node.feature_name]
                
                # 最も高いメンバーシップ度の分岐を選択
                best_label = None
                best_membership = 0.0
                
                for label, mf in current_node.membership_functions.items():
                    membership = mf.membership(feature_value)
                    if membership > best_membership and label in current_node.children:
                        best_membership = membership
                        best_label = label
                
                if best_label:
                    decision_path.append(f"{current_node.feature_name}={feature_value:.2f} → {best_label} (確信度: {best_membership:.3f})")
                    current_node = current_node.children[best_label]
                else:
                    break
            else:
                break
        
        # 特徴量重要度計算
        feature_importance = self._calculate_feature_importance(features, feature_names)
        
        # 信頼度計算
        confidence = self._calculate_prediction_confidence(features)
        
        # 説明生成
        explanation = {
            'confidence': confidence,
            'rationale': f'高度ファジィ推論による予測: {prediction:.4f}',
            'decision_steps': decision_path,
            'feature_importance': feature_importance,
            'node_statistics': {
                'prediction_count': self.prediction_count,
                'average_prediction': self.total_prediction_value / max(1, self.prediction_count),
                'tree_depth': self.calculate_depth(),
                'tree_complexity': self.calculate_complexity()
            },
            'membership_activations': self._get_membership_statistics()
        }
        
        return prediction, explanation
    
    def _calculate_prediction_confidence(self, features: Dict[str, float]) -> float:
        """予測信頼度計算"""
        if self.is_leaf:
            return 0.9
        
        if not self.feature_name or self.feature_name not in features:
            return 0.3
        
        feature_value = features[self.feature_name]
        max_membership = 0.0
        
        for mf in self.membership_functions.values():
            membership = mf.membership(feature_value)
            max_membership = max(max_membership, membership)
        
        # 複数の要素を考慮した信頼度
        base_confidence = max_membership
        depth_factor = 0.9 ** self.depth  # 深いほど信頼度減少
        sample_factor = min(1.0, self.prediction_count / 100.0)  # 予測回数による調整
        
        return base_confidence * depth_factor * (0.5 + 0.5 * sample_factor)
    
    def _calculate_feature_importance(self, features: Dict[str, float], 
                                   feature_names: List[str]) -> Dict[str, float]:
        """特徴量重要度計算"""
        importance = {name: 0.0 for name in feature_names}
        
        if self.is_leaf:
            return importance
        
        if self.feature_name and self.feature_name in features:
            # 現在のノードの重要度
            importance[self.feature_name] = self.importance_score + 1.0
            
            # 子ノードの重要度を再帰的に計算
            for child in self.children.values():
                child_importance = child._calculate_feature_importance(features, feature_names)
                for feature, imp in child_importance.items():
                    importance[feature] += imp * 0.6  # 減衰係数
        
        return importance
    
    def _get_membership_statistics(self) -> Dict[str, Dict[str, float]]:
        """メンバーシップ関数統計"""
        stats = {}
        for label, mf in self.membership_functions.items():
            stats[label] = mf.get_statistics()
        return stats
    
    def calculate_complexity(self) -> int:
        """複雑度計算"""
        if self.is_leaf:
            return 1
        
        complexity = 1 + len(self.membership_functions)
        for child in self.children.values():
            complexity += child.calculate_complexity()
        
        return complexity
    
    def calculate_depth(self) -> int:
        """深度計算"""
        if self.is_leaf:
            return 1
        
        max_child_depth = 0
        for child in self.children.values():
            child_depth = child.calculate_depth()
            max_child_depth = max(max_child_depth, child_depth)
        
        return 1 + max_child_depth
    
    def update_importance_score(self, score: float):
        """重要度スコア更新"""
        self.importance_score = max(self.importance_score, score)

# =============================================================================
# 遺伝的アルゴリズムパラメータ
# =============================================================================

@dataclass
class GeneticParameters:
    """遺伝的アルゴリズムパラメータ"""
    population_size: int = 50
    generations: int = 30
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    elite_size: int = 5
    tournament_size: int = 3
    
    # ファジィ決定木パラメータ
    max_depth: int = 6
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    max_features_per_node: int = 5
    
    # メンバーシップ関数パラメータ
    min_membership_functions: int = 2
    max_membership_functions: int = 4
    membership_types: List[MembershipType] = None
    
    # 最適化パラメータ
    fitness_weights: Dict[str, float] = None
    convergence_threshold: float = 0.001
    max_stagnant_generations: int = 10
    
    def __post_init__(self):
        if self.membership_types is None:
            self.membership_types = [MembershipType.TRIANGULAR, MembershipType.GAUSSIAN]
        
        if self.fitness_weights is None:
            self.fitness_weights = {
                'accuracy': 0.4,
                'simplicity': 0.2,
                'interpretability': 0.2,
                'generalization': 0.2
            }

# =============================================================================
# 個体クラス
# =============================================================================

class GeneticIndividual:
    """遺伝的アルゴリズム個体"""
    
    def __init__(self, individual_id: str = None):
        self.individual_id = individual_id or f"individual_{int(time.time())}_{random.randint(1000, 9999)}"
        self.generation = 0
        
        # ファジィ決定木
        self.tree: Optional[AdvancedFuzzyDecisionNode] = None
        
        # 適応度関連
        self.fitness_value = 0.0
        self.fitness_components = {
            'accuracy': 0.0,
            'simplicity': 0.0,
            'interpretability': 0.0,
            'generalization': 0.0
        }
        
        # ゲノム（木の構造を表現）
        self.genome = {
            'max_depth': random.randint(3, 7),
            'split_features': [],
            'membership_configs': {},
            'node_types': {}
        }
        
        # 統計情報
        self.prediction_history = []
        self.training_time = 0.0
        self.evaluation_count = 0
    
    def evaluate_fitness(self, training_data: pd.DataFrame, test_data: pd.DataFrame, 
                        target_column: str, parameters: GeneticParameters) -> float:
        """適応度評価"""
        start_time = time.time()
        
        try:
            # 訓練データとテストデータ分離
            X_train = training_data.drop(columns=[target_column])
            y_train = training_data[target_column]
            X_test = test_data.drop(columns=[target_column])
            y_test = test_data[target_column]
            
            # 決定木が存在しない場合は構築
            if self.tree is None:
                self.tree = self._build_tree_from_genome(X_train, y_train, parameters)
            
            # 各適応度成分を計算
            accuracy = self._calculate_accuracy(X_train, y_train, X_test, y_test)
            simplicity = self._calculate_simplicity()
            interpretability = self._calculate_interpretability()
            generalization = self._calculate_generalization(X_train, y_train, X_test, y_test)
            
            # 適応度成分を保存
            self.fitness_components = {
                'accuracy': accuracy,
                'simplicity': simplicity,
                'interpretability': interpretability,
                'generalization': generalization
            }
            
            # 重み付き適応度計算
            weights = parameters.fitness_weights
            self.fitness_value = (
                weights['accuracy'] * accuracy +
                weights['simplicity'] * simplicity +
                weights['interpretability'] * interpretability +
                weights['generalization'] * generalization
            )
            
            self.training_time = time.time() - start_time
            self.evaluation_count += 1
            
            return self.fitness_value
            
        except Exception as e:
            print(f"適応度評価エラー ({self.individual_id}): {e}")
            self.fitness_value = 0.0
            return 0.0
    
    def _build_tree_from_genome(self, X: pd.DataFrame, y: pd.Series, 
                              parameters: GeneticParameters) -> AdvancedFuzzyDecisionNode:
        """ゲノムからファジィ決定木を構築"""
        feature_names = list(X.columns)
        
        # ルートノード作成
        root = self._build_node_recursive(X, y, feature_names, depth=0, 
                                        max_depth=self.genome['max_depth'],
                                        parameters=parameters)
        return root
    
    def _build_node_recursive(self, X: pd.DataFrame, y: pd.Series, 
                            feature_names: List[str], depth: int, max_depth: int,
                            parameters: GeneticParameters) -> AdvancedFuzzyDecisionNode:
        """再帰的ノード構築"""
        
        # 停止条件
        if (depth >= max_depth or 
            len(X) < parameters.min_samples_split or
            self._is_pure(y)):
            
            leaf_value = y.mean() if len(y) > 0 else 0.5
            leaf = AdvancedFuzzyDecisionNode(is_leaf=True, leaf_value=leaf_value)
            leaf.depth = depth
            leaf.sample_count = len(X)
            return leaf
        
        # 最良特徴量選択
        best_feature = self._select_best_feature(X, y, feature_names)
        
        if best_feature is None:
            leaf_value = y.mean() if len(y) > 0 else 0.5
            leaf = AdvancedFuzzyDecisionNode(is_leaf=True, leaf_value=leaf_value)
            leaf.depth = depth
            return leaf
        
        # 内部ノード作成
        node = AdvancedFuzzyDecisionNode(feature_name=best_feature)
        node.depth = depth
        node.sample_count = len(X)
        
        # メンバーシップ関数生成
        membership_functions = self._generate_membership_functions(
            best_feature, X[best_feature], parameters)
        
        for label, mf in membership_functions.items():
            node.add_membership_function(label, mf)
        
        # 子ノード生成
        for label, mf in membership_functions.items():
            # ファジィ分割
            child_indices = self._fuzzy_split_indices(X[best_feature], mf)
            
            if len(child_indices) >= parameters.min_samples_leaf:
                child_X = X.iloc[child_indices]
                child_y = y.iloc[child_indices]
                
                child = self._build_node_recursive(
                    child_X, child_y, feature_names, depth + 1, max_depth, parameters)
                node.add_child(label, child)
        
        # 重要度スコア更新
        importance = self._calculate_node_importance(X, y, best_feature)
        node.update_importance_score(importance)
        
        return node
    
    def _select_best_feature(self, X: pd.DataFrame, y: pd.Series, 
                           feature_names: List[str]) -> Optional[str]:
        """最良特徴量選択（情報ゲインベース）"""
        best_feature = None
        best_score = -np.inf
        
        candidate_features = random.sample(feature_names, 
                                         min(len(feature_names), 
                                             self.genome.get('max_features', 5)))
        
        for feature in candidate_features:
            score = self._calculate_information_gain(X, y, feature)
            
            if score > best_score:
                best_score = score
                best_feature = feature
        
        return best_feature if best_score > 0.01 else None
    
    def _calculate_information_gain(self, X: pd.DataFrame, y: pd.Series, 
                                  feature: str) -> float:
        """情報ゲイン計算"""
        try:
            # 親ノードのエントロピー
            parent_entropy = self._calculate_entropy(y)
            
            # 特徴量の値域からメンバーシップ関数を仮生成
            values = X[feature]
            min_val, max_val = values.min(), values.max()
            
            if max_val - min_val < 0.001:
                return 0.0
            
            # 簡単な3分割でのエントロピー計算
            split_points = [min_val + (max_val - min_val) * i / 3 for i in range(1, 3)]
            
            weighted_entropy = 0.0
            total_weight = 0.0
            
            for i, split_point in enumerate(split_points + [max_val]):
                if i == 0:
                    mask = values <= split_points[0]
                elif i == len(split_points):
                    mask = values > split_points[-1]
                else:
                    mask = (values > split_points[i-1]) & (values <= split_point)
                
                if mask.sum() > 0:
                    weight = mask.sum() / len(y)
                    entropy = self._calculate_entropy(y[mask])
                    weighted_entropy += weight * entropy
                    total_weight += weight
            
            if total_weight > 0:
                return parent_entropy - weighted_entropy
            else:
                return 0.0
                
        except Exception as e:
            return 0.0
    
    def _calculate_entropy(self, y: pd.Series) -> float:
        """エントロピー計算"""
        if len(y) <= 1:
            return 0.0
        
        # 回帰の場合は分散ベース
        variance = y.var()
        return np.log(1 + variance)
    
    def _generate_membership_functions(self, feature: str, values: pd.Series,
                                     parameters: GeneticParameters) -> Dict[str, MembershipFunction]:
        """メンバーシップ関数生成"""
        min_val, max_val = values.min(), values.max()
        range_size = max_val - min_val
        
        if range_size < 0.001:
            range_size = 1.0
        
        num_mf = random.randint(parameters.min_membership_functions,
                               parameters.max_membership_functions)
        
        membership_functions = {}
        labels = ['very_low', 'low', 'medium', 'high', 'very_high'][:num_mf]
        
        for i, label in enumerate(labels):
            mf_type = random.choice(parameters.membership_types)
            
            if mf_type == MembershipType.TRIANGULAR:
                center = min_val + (i + 0.5) * range_size / num_mf
                spread = range_size / num_mf * random.uniform(0.6, 1.2)
                
                a = max(min_val, center - spread)
                b = center
                c = min(max_val, center + spread)
                
                mf = MembershipFunction(label, mf_type, [a, b, c])
                
            elif mf_type == MembershipType.GAUSSIAN:
                center = min_val + (i + 0.5) * range_size / num_mf
                sigma = range_size / num_mf * random.uniform(0.3, 0.8)
                
                mf = MembershipFunction(label, mf_type, [center, sigma])
            
            membership_functions[label] = mf
        
        return membership_functions
    
    def _fuzzy_split_indices(self, values: pd.Series, mf: MembershipFunction) -> List[int]:
        """ファジィ分割インデックス"""
        indices = []
        threshold = 0.1
        
        for idx, value in values.items():
            if mf.membership(value) >= threshold:
                indices.append(idx)
        
        return indices
    
    def _calculate_node_importance(self, X: pd.DataFrame, y: pd.Series, 
                                 feature: str) -> float:
        """ノード重要度計算"""
        try:
            correlation = abs(X[feature].corr(y))
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _is_pure(self, y: pd.Series, threshold: float = 0.01) -> bool:
        """純度チェック"""
        if len(y) <= 1:
            return True
        variance = y.var()
        return variance < threshold
    
    def _calculate_accuracy(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series) -> float:
        """精度計算"""
        if self.tree is None:
            return 0.0
        
        try:
            # テストデータで予測
            predictions = []
            for idx, row in X_test.iterrows():
                features = row.to_dict()
                prediction = self.tree.predict(features)
                predictions.append(prediction)
            
            predictions = np.array(predictions)
            
            # MSEベースの精度
            mse = np.mean((predictions - y_test) ** 2)
            accuracy = 1.0 / (1.0 + mse)
            
            return min(1.0, max(0.0, accuracy))
            
        except Exception as e:
            return 0.0
    
    def _calculate_simplicity(self) -> float:
        """簡潔性計算"""
        if self.tree is None:
            return 0.0
        
        complexity = self.tree.calculate_complexity()
        max_complexity = 100  # 正規化用
        simplicity = 1.0 - min(complexity / max_complexity, 1.0)
        
        return max(0.0, simplicity)
    
    def _calculate_interpretability(self) -> float:
        """解釈可能性計算"""
        if self.tree is None:
            return 0.0
        
        depth = self.tree.calculate_depth()
        max_depth = 10  # 正規化用
        
        # 深度が浅いほど解釈しやすい
        depth_score = 1.0 - min(depth / max_depth, 1.0)
        
        # メンバーシップ関数の単純さも考



# =============================================================================
# 完全版遺伝的ファジィ決定木最適化器
# =============================================================================

class CompleteGeneticFuzzyTreeOptimizer:
    """完全版遺伝的ファジィ決定木最適化器"""
    
    def __init__(self, parameters: GeneticParameters = None, random_seed: int = None):
        self.parameters = parameters or GeneticParameters()
        self.random_seed = random_seed
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # 最適化状態
        self.population: List[GeneticIndividual] = []
        self.best_individual: Optional[GeneticIndividual] = None
        self.tracker = OptimizationTracker()
        
        # 実行情報
        self.is_optimized = False
        self.optimization_time = 0.0
        
        print(f"[OPTIMIZER] 完全版遺伝的ファジィ決定木最適化器初期化")
        print(f"   集団サイズ: {self.parameters.population_size}")
        print(f"   世代数: {self.parameters.generations}")
        print(f"   突然変異率: {self.parameters.mutation_rate}")
        print(f"   交叉率: {self.parameters.crossover_rate}")
    
    def optimize(self, training_data: pd.DataFrame, test_data: pd.DataFrame,
                target_column: str, run_id: str = None) -> Dict[str, Any]:
        """完全最適化実行"""
        
        print(f"\n{'='*60}")
        print(f"[OPTIMIZATION] 完全版遺伝的最適化開始")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # 初期集団生成
            print(f"[STEP 1] 初期集団生成 ({self.parameters.population_size}個体)")
            self.population = self._initialize_population()
            
            # 初期評価
            print(f"[STEP 2] 初期適応度評価")
            self._evaluate_population(training_data, test_data, target_column)
            
            # 進化ループ
            print(f"[STEP 3] 進化ループ開始 ({self.parameters.generations}世代)")
            
            stagnant_count = 0
            best_fitness = 0.0
            
            for generation in range(self.parameters.generations):
                gen_start_time = time.time()
                
                # 選択、交叉、突然変異
                new_population = self._evolve_population()
                
                # 新集団評価
                self.population = new_population
                self._evaluate_population(training_data, test_data, target_column)
                
                # 最良個体更新
                current_best = max(self.population, key=lambda x: x.fitness_value)
                
                if current_best.fitness_value > best_fitness:
                    self.best_individual = current_best
                    best_fitness = current_best.fitness_value
                    stagnant_count = 0
                else:
                    stagnant_count += 1
                
                # 世代記録
                self.tracker.record_generation(generation, self.population)
                
                # 進捗表示
                if generation % 5 == 0 or generation == self.parameters.generations - 1:
                    gen_time = time.time() - gen_start_time
                    avg_fitness = np.mean([ind.fitness_value for ind in self.population])
                    diversity = self.tracker.diversity_history[-1] if self.tracker.diversity_history else 0.0
                    
                    print(f"   世代 {generation:3d}: 最良={best_fitness:.4f}, 平均={avg_fitness:.4f}, 多様性={diversity:.4f} ({gen_time:.2f}s)")
                
                # 早期収束チェック
                if stagnant_count >= self.parameters.max_stagnant_generations:
                    print(f"   早期収束 (世代 {generation})")
                    break
                
                if self.tracker.is_converged():
                    print(f"   収束検出 (世代 {generation})")
                    break
            
            self.optimization_time = time.time() - start_time
            self.is_optimized = True
            
            # 結果生成
            result = self._generate_optimization_result(run_id)
            
            print(f"\n[COMPLETE] 最適化完了")
            print(f"   最良適応度: {result['best_fitness']:.4f}")
            print(f"   実行時間: {self.optimization_time:.2f}秒")
            print(f"   収束世代: {result.get('convergence_generation', 'N/A')}")
            
            return result
            
        except Exception as e:
            print(f"[ERROR] 最適化エラー: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'best_fitness': 0.0
            }
    
    def _initialize_population(self) -> List[GeneticIndividual]:
        """初期集団生成"""
        population = []
        
        for i in range(self.parameters.population_size):
            individual = GeneticIndividual(f"gen0_ind{i}")
            individual.generation = 0
            
            # ランダムゲノム生成
            individual.genome = {
                'max_depth': random.randint(3, self.parameters.max_depth),
                'max_features': random.randint(2, self.parameters.max_features_per_node),
                'split_strategy': random.choice(['information_gain', 'gini', 'random']),
                'membership_preference': random.choice(['triangular', 'gaussian', 'mixed'])
            }
            
            population.append(individual)
        
        return population
    
    def _evaluate_population(self, training_data: pd.DataFrame, 
                           test_data: pd.DataFrame, target_column: str):
        """集団適応度評価"""
        for individual in self.population:
            if individual.fitness_value == 0.0:  # 未評価の個体のみ
                individual.evaluate_fitness(training_data, test_data, 
                                          target_column, self.parameters)
    
    def _evolve_population(self) -> List[GeneticIndividual]:
        """集団進化"""
        new_population = []
        
        # エリート保存
        sorted_population = sorted(self.population, key=lambda x: x.fitness_value, reverse=True)
        elite_count = min(self.parameters.elite_size, len(sorted_population))
        
        for i in range(elite_count):
            elite = sorted_population[i]
            elite.generation += 1
            new_population.append(elite)
        
        # 交叉と突然変異で残りを生成
        while len(new_population) < self.parameters.population_size:
            if random.random() < self.parameters.crossover_rate:
                # 交叉
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                child = parent1.crossover(parent2, self.parameters)
                child.generation = parent1.generation + 1
            else:
                # 複製
                parent = self._tournament_selection()
                child = GeneticIndividual()
                child.genome = parent.genome.copy()
                child.generation = parent.generation + 1
            
            # 突然変異
            child.mutate(self.parameters)
            new_population.append(child)
        
        return new_population[:self.parameters.population_size]
    
    def _tournament_selection(self) -> GeneticIndividual:
        """トーナメント選択"""
        tournament_size = min(self.parameters.tournament_size, len(self.population))
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness_value)
    
    def _generate_optimization_result(self, run_id: str = None) -> Dict[str, Any]:
        """最適化結果生成"""
        if not self.best_individual:
            return {'success': False, 'error': 'No best individual found'}
        
        stats = self.tracker.get_statistics()
        
        result = {
            'success': True,
            'run_id': run_id or f"run_{int(time.time())}",
            'best_individual': self.best_individual,
            'best_fitness': self.best_individual.fitness_value,
            'fitness_components': self.best_individual.fitness_components,
            
            # 最適化統計
            'optimization_time': self.optimization_time,
            'generations_run': len(self.tracker.generation_history),
            'convergence_generation': stats.get('convergence_generation'),
            
            # 進化統計
            'best_fitness_history': self.tracker.best_fitness_history,
            'average_fitness_history': self.tracker.average_fitness_history,
            'diversity_history': self.tracker.diversity_history,
            'final_diversity': stats.get('final_diversity', 0.0),
            
            # モデル統計
            'tree_depth': self.best_individual.tree.calculate_depth() if self.best_individual.tree else 0,
            'tree_complexity': self.best_individual.tree.calculate_complexity() if self.best_individual.tree else 0,
            'training_time': self.best_individual.training_time,
            
            # システム情報
            'parameters': {
                'population_size': self.parameters.population_size,
                'generations': self.parameters.generations,
                'mutation_rate': self.parameters.mutation_rate,
                'crossover_rate': self.parameters.crossover_rate,
                'max_depth': self.parameters.max_depth
            },
            
            'created_at': datetime.now().isoformat(),
            'model_type': 'complete_genetic_fuzzy_tree'
        }
        
        return result

# =============================================================================
# 完全版モデル管理システム
# =============================================================================

class CompleteModelManager:
    """完全版モデル管理システム"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.ensure_directory_exists()
        
    def ensure_directory_exists(self):
        """ディレクトリ確保"""
        os.makedirs(self.models_dir, exist_ok=True)
        
    def save_complete_model(self, optimizer: CompleteGeneticFuzzyTreeOptimizer, 
                          result: Dict[str, Any], 
                          model_id: str = None) -> str:
        """完全モデル保存"""
        
        if model_id is None:
            model_id = f"complete_genetic_fuzzy_{int(time.time())}"
        
        print(f"[SAVE] 完全モデル保存開始: {model_id}")
        
        try:
            # メインモデルファイル
            model_path = os.path.join(self.models_dir, f"{model_id}_model.pkl.gz")
            
            model_data = {
                'optimizer': optimizer,
                'best_individual': optimizer.best_individual,
                'optimization_result': result,
                'model_metadata': {
                    'model_id': model_id,
                    'creation_time': datetime.now().isoformat(),
                    'model_type': 'complete_genetic_fuzzy_tree',
                    'version': '2.0_complete'
                }
            }
            
            # 圧縮保存
            with gzip.open(model_path, 'wb') as f:
                pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # 結果サマリー（JSON）
            summary_path = os.path.join(self.models_dir, f"{model_id}_summary.json")
            
            summary = {
                'model_id': model_id,
                'creation_time': datetime.now().isoformat(),
                'best_fitness': result.get('best_fitness', 0.0),
                'fitness_components': result.get('fitness_components', {}),
                'optimization_time': result.get('optimization_time', 0.0),
                'generations_run': result.get('generations_run', 0),
                'tree_depth': result.get('tree_depth', 0),
                'tree_complexity': result.get('tree_complexity', 0),
                'model_type': 'complete_genetic_fuzzy_tree',
                'file_size_bytes': os.path.getsize(model_path)
            }
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            print(f"[SAVE] モデル保存完了")
            print(f"   モデルファイル: {model_path}")
            print(f"   サマリー: {summary_path}")
            print(f"   ファイルサイズ: {os.path.getsize(model_path)} bytes")
            
            return model_id
            
        except Exception as e:
            print(f"[ERROR] モデル保存エラー: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_complete_model(self, model_id: str = None) -> Optional[Dict[str, Any]]:
        """完全モデル読み込み"""
        
        if model_id is None:
            model_id = self._find_latest_model()
        
        if model_id is None:
            print(f"[ERROR] 利用可能なモデルが見つかりません")
            return None
        
        model_path = os.path.join(self.models_dir, f"{model_id}_model.pkl.gz")
        
        if not os.path.exists(model_path):
            print(f"[ERROR] モデルファイルが見つかりません: {model_path}")
            return None
        
        try:
            print(f"[LOAD] 完全モデル読み込み: {model_id}")
            
            with gzip.open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            print(f"[LOAD] モデル読み込み完了")
            print(f"   適応度: {model_data['optimization_result'].get('best_fitness', 'N/A'):.4f}")
            print(f"   木の深度: {model_data['optimization_result'].get('tree_depth', 'N/A')}")
            
            return model_data
            
        except Exception as e:
            print(f"[ERROR] モデル読み込みエラー: {e}")
            return None
    
    def _find_latest_model(self) -> Optional[str]:
        """最新モデル検索"""
        try:
            model_files = [f for f in os.listdir(self.models_dir) 
                          if f.endswith('_model.pkl.gz') and 'complete_genetic_fuzzy' in f]
            
            if not model_files:
                return None
            
            # タイムスタンプでソート
            model_files.sort(reverse=True)
            latest_file = model_files[0]
            
            # model_idを抽出
            model_id = latest_file.replace('_model.pkl.gz', '')
            return model_id
            
        except Exception as e:
            return None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """モデル一覧取得"""
        models = []
        
        try:
            for filename in os.listdir(self.models_dir):
                if filename.endswith('_summary.json'):
                    summary_path = os.path.join(self.models_dir, filename)
                    
                    with open(summary_path, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                    
                    models.append(summary)
            
            # 作成時間でソート
            models.sort(key=lambda x: x.get('creation_time', ''), reverse=True)
            
        except Exception as e:
            print(f"[WARNING] モデル一覧取得エラー: {e}")
        
        return models

# =============================================================================
# 統合エンジンクラス
# =============================================================================

class CompleteGeneticFuzzyEngine:
    """完全版遺伝的ファジィエンジン"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.model_manager = CompleteModelManager(models_dir)
        
        # 現在のモデル
        self.current_model = None
        self.best_individual = None
        self.is_model_loaded = False
        
        # エンジン情報
        self.engine_info = {
            'type': 'CompleteGeneticFuzzyEngine',
            'version': '2.0_complete',
            'capabilities': [
                'complete_genetic_optimization',
                'advanced_fuzzy_decision_trees',
                'detailed_explanations',
                'model_persistence',
                'performance_tracking'
            ]
        }
        
        print(f"[ENGINE] 完全版遺伝的ファジィエンジン初期化")
        self._try_load_model()
    
    def _try_load_model(self):
        """モデル読み込み試行"""
        try:
            model_data = self.model_manager.load_complete_model()
            if model_data:
                self.current_model = model_data
                self.best_individual = model_data['best_individual']
                self.is_model_loaded = True
                print(f"[ENGINE] 既存モデル読み込み成功")
            else:
                print(f"[ENGINE] 既存モデルなし、新規作成が必要")
        except Exception as e:
            print(f"[WARNING] モデル読み込み失敗: {e}")
            self.is_model_loaded = False
    
    def create_and_optimize_model(self, training_samples: int = 1000, 
                                test_samples: int = 200) -> bool:
        """モデル作成と最適化"""
        
        print(f"\n{'='*60}")
        print(f"[ENGINE] 完全版モデル作成・最適化開始")
        print(f"{'='*60}")
        
        try:
            # 合成データ生成
            print(f"[STEP 1] 合成データ生成")
            training_data = self._create_advanced_synthetic_data(training_samples)
            test_data = self._create_advanced_synthetic_data(test_samples)
            
            # 最適化器作成
            print(f"[STEP 2] 最適化器初期化")
            parameters = GeneticParameters(
                population_size=50,
                generations=30,
                mutation_rate=0.15,
                crossover_rate=0.8,
                max_depth=6
            )
            
            optimizer = CompleteGeneticFuzzyTreeOptimizer(parameters, random_seed=42)
            
            # 最適化実行
            print(f"[STEP 3] 遺伝的最適化実行")
            result = optimizer.optimize(
                training_data=training_data,
                test_data=test_data,
                target_column='compatibility',
                run_id=f'complete_optimization_{int(time.time())}'
            )
            
            if result.get('success', False):
                # モデル保存
                print(f"[STEP 4] モデル保存")
                model_id = self.model_manager.save_complete_model(optimizer, result)
                
                if model_id:
                    self.current_model = {
                        'optimizer': optimizer,
                        'best_individual': optimizer.best_individual,
                        'optimization_result': result
                    }
                    self.best_individual = optimizer.best_individual
                    self.is_model_loaded = True
                    
                    print(f"[SUCCESS] 完全版モデル作成完了!")
                    print(f"   モデルID: {model_id}")
                    print(f"   最良適応度: {result['best_fitness']:.4f}")
                    return True
            
            print(f"[ERROR] 最適化失敗")
            return False
            
        except Exception as e:
            print(f"[ERROR] モデル作成エラー: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_compatibility(self, user_prefs: Dict[str, float], 
                            lab_features: Dict[str, float]) -> Tuple[Dict[str, Any], str]:
        """高度な適合度予測"""
        
        if not self.is_model_loaded or self.best_individual is None:
            return self._fallback_prediction(user_prefs, lab_features)
        
        try:
            return self._complete_genetic_prediction(user_prefs, lab_features)
        except Exception as e:
            print(f"[WARNING] 完全予測エラー: {e}")
            return self._fallback_prediction(user_prefs, lab_features)
    
    def _complete_genetic_prediction(self, user_prefs: Dict[str, float], 
                                   lab_features: Dict[str, float]) -> Tuple[Dict[str, Any], str]:
        """完全版遺伝的予測"""
        
        # 高度な特徴量準備
        features = self._prepare_advanced_features(user_prefs, lab_features)
        
        # 完全版ファジィ決定木で予測
        if self.best_individual.tree is None:
            return self._fallback_prediction(user_prefs, lab_features)
        
        prediction = self.best_individual.tree.predict(features)
        
        # 詳細説明付き予測
        detailed_prediction, explanation = self.best_individual.tree.predict_with_explanation(
            features, list(features.keys())
        )
        
        # 信頼度計算
        confidence = explanation.get('confidence', 0.8)
        
        # 高度な結果構築
        result = {
            'overall_score': prediction * 100,
            'confidence': confidence * 100,
            'prediction_method': 'complete_genetic_fuzzy_optimization',
            
            # 詳細情報
            'detailed_scores': self._calculate_detailed_scores(features, prediction),
            'feature_contributions': explanation.get('feature_importance', {}),
            'decision_path': explanation.get('decision_steps', []),
            'model_statistics': explanation.get('node_statistics', {}),
            'membership_analysis': explanation.get('membership_activations', {}),
            
            # 遺伝的アルゴリズム情報
            'genetic_info': {
                'individual_id': self.best_individual.individual_id,
                'generation': self.best_individual.generation,
                'fitness_value': self.best_individual.fitness_value,
                'fitness_components': self.best_individual.fitness_components,
                'genome': self.best_individual.genome
            },
            
            # システム情報
            'model_info': {
                'model_type': 'complete_genetic_fuzzy_tree',
                'tree_depth': self.best_individual.tree.calculate_depth(),
                'tree_complexity': self.best_individual.tree.calculate_complexity(),
                'optimization_time': self.current_model['optimization_result'].get('optimization_time', 0.0),
                'prediction_count': self.best_individual.tree.prediction_count
            }
        }
        
        # 詳細説明文生成
        explanation_text = self._generate_detailed_explanation(result, explanation)
        
        return result, explanation_text
    
    def _prepare_advanced_features(self, user_prefs: Dict[str, float], 
                                 lab_features: Dict[str, float]) -> Dict[str, float]:
        """高度な特徴量準備"""
        features = {}
        
        # 基本的な類似度特徴量
        basic_criteria = ['research_intensity', 'advisor_style', 'team_work', 'workload', 'theory_practice']
        
        for criterion in basic_criteria:
            user_val = user_prefs.get(criterion, 5.0)
            lab_val = lab_features.get(criterion, 5.0)
            
            # 複数の類似度指標
            similarity = 1.0 - abs(user_val - lab_val) / 10.0
            compatibility = np.exp(-((user_val - lab_val) / 5.0) ** 2)  # ガウシアン類似度
            
            features[criterion] = max(0.0, min(10.0, (similarity + compatibility) * 5.0))
        
        # 高次特徴量
        features['overall_alignment'] = np.mean(list(features.values()))
        features['preference_variance'] = np.std(list(user_prefs.values())[:5])
        features['lab_consistency'] = 1.0 - np.std(list(lab_features.values())[:5]) / 10.0
        
        return features
    
    def _calculate_detailed_scores(self, features: Dict[str, float], 
                                 overall_prediction: float) -> Dict[str, float]:
        """詳細スコア計算"""
        detailed_scores = {}
        
        # カテゴリ別スコア
        research_features = ['research_intensity', 'theory_practice']
        social_features = ['advisor_style', 'team_work']
        workload_features = ['workload']
        
        feature_groups = {
            'research_match': research_features,
            'social_fit': social_features,
            'workload_balance': workload_features
        }
        
        for category, feature_list in feature_groups.items():
            category_values = [features.get(f, 5.0) for f in feature_list if f in features]
            if category_values:
                detailed_scores[category] = np.mean(category_values) * 10.0
        
        # 全体調整
        detailed_scores['overall'] = overall_prediction * 100
        
        return detailed_scores
    
    def _generate_detailed_explanation(self, result: Dict[str, Any], 
                                     explanation: Dict[str, Any]) -> str:
        """詳細説明文生成"""
        
        score = result['overall_score']
        confidence = result['confidence']
        
        # 基本評価
        if score >= 85:
            base_assessment = "非常に高い適合性"
        elif score >= 70:
            base_assessment = "高い適合性"
        elif score >= 55:
            base_assessment = "中程度の適合性"
        elif score >= 40:
            base_assessment = "やや低い適合性"
        else:
            base_assessment = "適合性に課題"
        
        # 詳細分析
        genetic_info = result.get('genetic_info', {})
        model_info = result.get('model_info', {})
        
        explanation_parts = [
            f"完全版遺伝的ファジィ決定木による分析結果: {base_assessment} (スコア: {score:.1f}%)",
            f"予測信頼度: {confidence:.1f}%",
            "",
            "🧬 詳細分析:",
            f"・決定木深度: {model_info.get('tree_depth', 'N/A')}",
            f"・モデル複雑度: {model_info.get('tree_complexity', 'N/A')}",
            f"・遺伝的最適化世代: {genetic_info.get('generation', 'N/A')}",
            f"・個体適応度: {genetic_info.get('fitness_value', 0):.4f}",
        ]
        
        # 特徴量重要度
        feature_importance = result.get('feature_contributions', {})
        if feature_importance:
            explanation_parts.append("")
            explanation_parts.append("📊 主要な判定要因:")
            
            sorted_features = sorted(feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)[:3]
            
            for feature, importance in sorted_features:
                explanation_parts.append(f"・{feature}: {importance:.3f}")
        
        return "\n".join(explanation_parts)
    
    def _fallback_prediction(self, user_prefs: Dict[str, float], 
                           lab_features: Dict[str, float]) -> Tuple[Dict[str, Any], str]:
        """フォールバック予測"""
        
        # 基本的な類似度計算
        criteria = ['research_intensity', 'advisor_style', 'team_work', 'workload', 'theory_practice']
        similarities = []
        
        for criterion in criteria:
            user_val = user_prefs.get(criterion, 5.0)
            lab_val = lab_features.get(criterion, 5.0)
            similarity = 1.0 - abs(user_val - lab_val) / 10.0
            similarities.append(max(0.0, similarity))
        
        overall_score = np.mean(similarities) * 100
        
        result = {
            'overall_score': overall_score,
            'confidence': 70.0,
            'prediction_method': 'fallback_basic',
            'detailed_scores': {
                'research_match': similarities[0] * 100,
                'social_fit': np.mean(similarities[1:3]) * 100,
                'workload_balance': similarities[3] * 100,
                'theory_practice_fit': similarities[4] * 100,
                'overall': overall_score
            }
        }
        
        explanation = f"基本アルゴリズムによる分析: スコア {overall_score:.1f}% (信頼度: 70%)\n注意: 完全版モデルが利用できないため、基本的な計算を使用しています。"
        
        return result, explanation
    
    def _create_advanced_synthetic_data(self, n_samples: int) -> pd.DataFrame:
        """高度な合成データ生成"""
        
        np.random.seed(42)
        data = []
        
        for i in range(n_samples):
            # より複雑で現実的な特徴量生成
            sample = {}
            
            # 基本特徴量（相関を考慮）
            base_tendency = np.random.normal(6.0, 1.5)  # 個人の基本傾向
            
            sample['research_intensity'] = max(1, min(10, 
                base_tendency + np.random.normal(0, 1.2)))
            sample['advisor_style'] = max(1, min(10, 
                base_tendency * 0.7 + np.random.normal(0, 1.5)))
            sample['team_work'] = max(1, min(10, 
                10 - base_tendency * 0.3 + np.random.normal(0, 1.3)))
            sample['workload'] = max(1, min(10, 
                base_tendency * 0.8 + np.random.normal(0, 1.4)))
            sample['theory_practice'] = max(1, min(10, 
                base_tendency + np.random.normal(0, 1.6)))
            
            # 高度な適合度計算
            weights = {
                'research_intensity': 0.25,
                'advisor_style': 0.20,
                'team_work': 0.20,
                'workload': 0.15,
                'theory_practice': 0.20
            }
            
            base_compatibility = sum(weights[key] * value 
                                   for key, value in sample.items()) / 10.0
            
            # 非線形効果とインタラクション
            research_theory_synergy = (sample['research_intensity'] * 
                                     sample['theory_practice']) / 100.0 * 0.1
            
            social_balance = 1.0 - abs(sample['advisor_style'] - 
                                     sample['team_work']) / 10.0 * 0.05
            
            workload_optimality = np.exp(-((sample['workload'] - 6.0) / 3.0) ** 2) * 0.1
            
            # 最終適合度
            compatibility = (base_compatibility + research_theory_synergy + 
                           social_balance + workload_optimality)
            
            # ノイズ追加
            compatibility += np.random.normal(0, 0.05)
            compatibility = max(0.0, min(1.0, compatibility))
            
            sample['compatibility'] = compatibility
            data.append(sample)
        
        return pd.DataFrame(data)
    
    def get_engine_info(self) -> Dict[str, Any]:
        """エンジン情報取得"""
        info = self.engine_info.copy()
        info.update({
            'model_loaded': self.is_model_loaded,
            'models_available': len(self.model_manager.list_models()),
            'current_model_info': {}
        })
        
        if self.current_model and self.best_individual:
            info['current_model_info'] = {
                'individual_id': self.best_individual.individual_id,
                'fitness': self.best_individual.fitness_value,
                'tree_depth': self.best_individual.tree.calculate_depth() if self.best_individual.tree else 0,
                'prediction_count': self.best_individual.tree.prediction_count if self.best_individual.tree else 0
            }
        
        return info
    
    def test_complete_system(self) -> bool:
        """完全システムテスト"""
        
        print(f"\n{'='*60}")
        print(f"[TEST] 完全システム統合テスト開始")
        print(f"{'='*60}")
        
        try:
            # テスト1: モデル作成・最適化
            print(f"[TEST 1] モデル作成・最適化テスト")
            if not self.is_model_loaded:
                success = self.create_and_optimize_model(training_samples=200, test_samples=50)
                if not success:
                    print(f"[FAIL] モデル作成失敗")
                    return False
            
            print(f"[PASS] モデル作成・最適化成功")
            
            # テスト2: 予測機能
            print(f"[TEST 2] 予測機能テスト")
            test_user_prefs = {
                'research_intensity': 8.5,
                'advisor_style': 6.2,
                'team_work': 7.8,
                'workload': 5.5,
                'theory_practice': 8.1
            }
            
            test_lab_features = {
                'research_intensity': 8.0,
                'advisor_style': 6.8,
                'team_work': 7.5,
                'workload': 6.0,
                'theory_practice': 7.8
            }
            
            result, explanation = self.predict_compatibility(test_user_prefs, test_lab_features)
            
            # 結果検証
            if not isinstance(result, dict) or 'overall_score' not in result:
                print(f"[FAIL] 予測結果が不正")
                return False
            
            score = result['overall_score']
            if not (0 <= score <= 100):
                print(f"[FAIL] スコア範囲が不正: {score}")
                return False
            
            print(f"[PASS] 予測機能テスト成功")
            print(f"   スコア: {score:.1f}%")
            print(f"   信頼度: {result.get('confidence', 0):.1f}%")
            print(f"   手法: {result.get('prediction_method', 'unknown')}")
            
            # テスト3: 高度な機能
            print(f"[TEST 3] 高度機能テスト")
            
            # 詳細情報の確認
            required_fields = ['detailed_scores', 'feature_contributions', 'genetic_info', 'model_info']
            for field in required_fields:
                if field not in result:
                    print(f"[FAIL] 必須フィールド不足: {field}")
                    return False
            
            # 遺伝的アルゴリズム情報の確認
            genetic_info = result['genetic_info']
            if not all(key in genetic_info for key in ['individual_id', 'fitness_value', 'fitness_components']):
                print(f"[FAIL] 遺伝的アルゴリズム情報不足")
                return False
            
            print(f"[PASS] 高度機能テスト成功")
            print(f"   個体ID: {genetic_info['individual_id']}")
            print(f"   適応度: {genetic_info['fitness_value']:.4f}")
            print(f"   木の深度: {result['model_info'].get('tree_depth', 'N/A')}")
            
            # テスト4: エンジン情報
            print(f"[TEST 4] エンジン情報テスト")
            info = self.get_engine_info()
            
            if not info.get('model_loaded', False):
                print(f"[FAIL] モデル読み込み状態が不正")
                return False
            
            print(f"[PASS] エンジン情報テスト成功")
            print(f"   エンジンタイプ: {info['type']}")
            print(f"   バージョン: {info['version']}")
            print(f"   利用可能モデル数: {info['models_available']}")
            
            print(f"\n[SUCCESS] 完全システム統合テスト完了!")
            print(f"すべての機能が正常に動作しています。")
            return True
            
        except Exception as e:
            print(f"[ERROR] システムテストエラー: {e}")
            import traceback
            traceback.print_exc()
            return False

# =============================================================================
# メイン実行関数
# =============================================================================

def create_complete_genetic_fuzzy_system():
    """完全版遺伝的ファジィシステム作成"""
    
    print(f"{'='*80}")
    print(f"完全版遺伝的ファジィ決定木システム")
    print(f"Complete Genetic Fuzzy Decision Tree System v2.0")
    print(f"{'='*80}")
    
    try:
        # エンジン初期化
        print(f"\n[INIT] システム初期化中...")
        engine = CompleteGeneticFuzzyEngine()
        
        # システムテスト実行
        print(f"\n[TEST] システムテスト実行中...")
        test_success = engine.test_complete_system()
        
        if test_success:
            print(f"\n[SUCCESS] 完全版システム作成完了!")
            print(f"以下の機能が利用可能です:")
            print(f"  ✓ 完全な遺伝的アルゴリズム最適化")
            print(f"  ✓ 高度なファジィ決定木")
            print(f"  ✓ 詳細な説明機能")
            print(f"  ✓ モデル永続化")
            print(f"  ✓ 性能追跡")
            
            return engine
        else:
            print(f"\n[ERROR] システムテスト失敗")
            return None
            
    except Exception as e:
        print(f"\n[ERROR] システム作成エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_demonstration():
    """デモンストレーション実行"""
    
    print(f"\n{'='*60}")
    print(f"[DEMO] 完全版システムデモンストレーション")
    print(f"{'='*60}")
    
    # システム作成
    engine = create_complete_genetic_fuzzy_system()
    
    if engine is None:
        print(f"[ERROR] システム作成失敗")
        return False
    
    # 実際的なテストケース
    test_cases = [
        {
            'name': '理論重視学生 vs 理論系研究室',
            'user_prefs': {
                'research_intensity': 9.0,
                'advisor_style': 7.0,
                'team_work': 6.0,
                'workload': 7.0,
                'theory_practice': 9.5
            },
            'lab_features': {
                'research_intensity': 8.5,
                'advisor_style': 7.5,
                'team_work': 6.5,
                'workload': 7.5,
                'theory_practice': 9.0
            }
        },
        {
            'name': '実践重視学生 vs 応用系研究室',
            'user_prefs': {
                'research_intensity': 7.0,
                'advisor_style': 8.0,
                'team_work': 9.0,
                'workload': 6.0,
                'theory_practice': 4.0
            },
            'lab_features': {
                'research_intensity': 7.5,
                'advisor_style': 8.5,
                'team_work': 8.0,
                'workload': 6.5,
                'theory_practice': 4.5
            }
        },
        {
            'name': 'バランス型学生 vs ミスマッチ研究室',
            'user_prefs': {
                'research_intensity': 6.0,
                'advisor_style': 6.0,
                'team_work': 6.0,
                'workload': 6.0,
                'theory_practice': 6.0
            },
            'lab_features': {
                'research_intensity': 9.0,
                'advisor_style': 3.0,
                'team_work': 9.0,
                'workload': 9.0,
                'theory_practice': 2.0
            }
        }
    ]
    
    print(f"\n[DEMO] テストケース実行")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- テストケース {i}: {test_case['name']} ---")
        
        result, explanation = engine.predict_compatibility(
            test_case['user_prefs'], test_case['lab_features']
        )
        
        print(f"適合度スコア: {result['overall_score']:.1f}%")
        print(f"信頼度: {result['confidence']:.1f}%")
        print(f"予測手法: {result['prediction_method']}")
        
        if 'detailed_scores' in result:
            detailed = result['detailed_scores']
            print(f"詳細スコア:")
            for category, score in detailed.items():
                print(f"  {category}: {score:.1f}%")
        
        if 'genetic_info' in result:
            genetic = result['genetic_info']
            print(f"遺伝的最適化情報:")
            print(f"  個体適応度: {genetic.get('fitness_value', 0):.4f}")
            print(f"  世代: {genetic.get('generation', 'N/A')}")
        
        print(f"\n説明:")
        print(explanation[:200] + "..." if len(explanation) > 200 else explanation)
    
    print(f"\n[COMPLETE] デモンストレーション完了!")
    return True

def main():
    """メイン実行"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='完全版遺伝的ファジィ決定木システム')
    parser.add_argument('--mode', choices=['create', 'demo', 'test'], 
                       default='demo', help='実行モード')
    parser.add_argument('--samples', type=int, default=1000, 
                       help='訓練サンプル数')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'create':
            print("[MODE] システム作成")
            engine = create_complete_genetic_fuzzy_system()
            success = engine is not None
            
        elif args.mode == 'demo':
            print("[MODE] デモンストレーション")
            success = run_demonstration()
            
        elif args.mode == 'test':
            print("[MODE] システムテスト")
            engine = CompleteGeneticFuzzyEngine()
            success = engine.test_complete_system()
        
        if success:
            print(f"\n[SUCCESS] 実行完了!")
            print(f"完全な遺伝的アルゴリズム最適化と高度なファジィ決定木機能が利用可能です。")
            return 0
        else:
            print(f"\n[FAILED] 実行失敗")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] ユーザーによって中断されました")
        return 1
    except Exception as e:
        print(f"\n[ERROR] 実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1

print(f"[SYSTEM] Part 4 (最終) 完了 - 統合エンジン、実行関数、エントリーポイント実装済み")

# =============================================================================
# エントリーポイント
# =============================================================================

if __name__ == '__main__':
    print(f"[SYSTEM] 完全版遺伝的ファジィ決定木システム起動")
    
    exit_code = main()
    
    if exit_code == 0:
        print(f"\n🎉 完全版遺伝的ファジィ決定木システムが正常に動作しました!")
        print(f"📊 高度なファジィ論理と遺伝的アルゴリズムによる最適化が実装されています")
        print(f"🔧 APIサーバー統合用にこのファイルを使用してください")
    else:
        print(f"\n❌ システム実行中にエラーが発生しました")
    
    # Windows環境での一時停止
    if sys.platform.startswith('win'):
        try:
            input("\nEnterキーを押して終了...")
        except (EOFError, KeyboardInterrupt):
            pass
    
    sys.exit(exit_code)

# 完全版システム完了
print(f"[SYSTEM] 🎯 完全版遺伝的ファジィ決定木システム全4パート実装完了!")
print(f"[SYSTEM] ✅ 完全な遺伝的アルゴリズム最適化機能搭載")
print(f"[SYSTEM] ✅ 高度なファジィ決定木機能搭載")

