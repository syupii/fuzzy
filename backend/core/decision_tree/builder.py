"""
ファジィ決定木構築 - core/decision_tree/builder.py
修正版：構文エラーを解消し、完全実装
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import random
from dataclasses import dataclass
from enum import Enum

from .node import FuzzyDecisionNode, FuzzyDecisionTree, NodeType

try:
    from ..fuzzy.membership import (
        MembershipFunction, TriangularMF, GaussianMF, 
        MembershipFunctionFactory, MembershipType
    )
except ImportError:
    # フォールバック用の簡単な実装
    class MembershipFunction:
        def __init__(self, name="default"):
            self.name = name
        
        def membership_degree(self, value):
            return 0.5
    
    class TriangularMF(MembershipFunction):
        def __init__(self, name, a, b, c):
            super().__init__(name)
            self.a, self.b, self.c = a, b, c
        
        def membership_degree(self, value):
            if value <= self.a or value >= self.c:
                return 0.0
            elif value == self.b:
                return 1.0
            elif self.a < value < self.b:
                return (value - self.a) / (self.b - self.a)
            else:  # self.b < value < self.c
                return (self.c - value) / (self.c - self.b)
    
    class GaussianMF(MembershipFunction):
        def __init__(self, name, center, sigma):
            super().__init__(name)
            self.center = center
            self.sigma = sigma
        
        def membership_degree(self, value):
            return np.exp(-0.5 * ((value - self.center) / self.sigma) ** 2)
    
    class MembershipType(Enum):
        TRIANGULAR = "triangular"
        GAUSSIAN = "gaussian"
    
    class MembershipFunctionFactory:
        @staticmethod
        def create_triangular(name, a, b, c):
            return TriangularMF(name, a, b, c)
        
        @staticmethod
        def create_gaussian(name, center, sigma):
            return GaussianMF(name, center, sigma)


class SplitCriterion(Enum):
    """分割基準"""
    INFORMATION_GAIN = "information_gain"
    GINI_IMPURITY = "gini_impurity"
    VARIANCE_REDUCTION = "variance_reduction"
    FUZZY_ENTROPY = "fuzzy_entropy"


@dataclass
class BuilderConfig:
    """構築設定"""
    max_depth: int = 6
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    max_features: Optional[int] = None
    split_criterion: SplitCriterion = SplitCriterion.VARIANCE_REDUCTION
    fuzzy_sets_per_feature: int = 3
    membership_type: MembershipType = MembershipType.TRIANGULAR
    min_information_gain: float = 0.01


class FuzzyTreeBuilder:
    """ファジィ決定木構築器"""
    
    def __init__(self, max_depth: int = 6, min_samples_leaf: int = 5,
                 membership_params: Dict[str, Dict[str, Any]] = None,
                 feature_selection_probs: np.ndarray = None):
        
        self.config = BuilderConfig(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf
        )
        
        self.membership_params = membership_params or {}
        self.feature_selection_probs = feature_selection_probs
        
        # 構築統計
        self.nodes_created = 0
        self.splits_attempted = 0
        self.successful_splits = 0
        
    def build_tree(self, data: pd.DataFrame, feature_names: List[str], 
                   target_name: str) -> FuzzyDecisionTree:
        """標準的な決定木構築"""
        
        print(f"決定木構築開始: サンプル数={len(data)}, 特徴量数={len(feature_names)}")
        
        # データ準備
        X = data[feature_names]
        y = data[target_name]
        
        # ルートノード作成
        root = self._build_node_recursive(X, y, feature_names, depth=0)
        
        # ツリー作成
        tree = FuzzyDecisionTree(root)
        tree.feature_names = feature_names
        tree.target_name = target_name
        
        print(f"決定木構築完了: ノード数={self.nodes_created}")
        
        return tree
    
    def _build_node_recursive(self, X: pd.DataFrame, y: pd.Series, 
                             feature_names: List[str], depth: int) -> FuzzyDecisionNode:
        """再帰的ノード構築"""
        
        self.nodes_created += 1
        
        # ベースケース：葉ノード作成条件
        if (depth >= self.config.max_depth or 
            len(X) < self.config.min_samples_split or 
            len(X) < self.config.min_samples_leaf * 2):
            
            return self._create_leaf_node(y, depth)
        
        # 最適分割特徴量とメンバーシップ関数を見つける
        best_feature, best_membership_functions, best_score = self._find_best_split(
            X, y, feature_names
        )
        
        if best_feature is None or best_score < self.config.min_information_gain:
            return self._create_leaf_node(y, depth)
        
        # 内部ノード作成
        node = FuzzyDecisionNode(f"node_{self.nodes_created}", NodeType.INTERNAL)
        node.feature_name = best_feature
        node.depth = depth
        node.statistics.sample_count = len(X)
        node.statistics.information_gain = best_score
        
        # メンバーシップ関数追加
        for label, mf in best_membership_functions.items():
            node.add_membership_function(label, mf)
        
        # 子ノード作成
        feature_values = X[best_feature].values
        children_data = self._split_data_fuzzy(X, y, best_feature, best_membership_functions)
        
        for label, (child_X, child_y) in children_data.items():
            if len(child_X) >= self.config.min_samples_leaf:
                child_node = self._build_node_recursive(
                    child_X, child_y, feature_names, depth + 1
                )
                node.add_child(label, child_node)
            else:
                # 最小サンプル数未満の場合は葉ノードを作成
                leaf_node = self._create_leaf_node(child_y, depth + 1)
                node.add_child(label, leaf_node)
        
        # 子ノードがない場合は葉ノードに変換
        if not node.children:
            return self._create_leaf_node(y, depth)
        
        self.successful_splits += 1
        return node
    
    def _create_leaf_node(self, y: pd.Series, depth: int) -> FuzzyDecisionNode:
        """葉ノード作成"""
        
        leaf_node = FuzzyDecisionNode(f"leaf_{self.nodes_created}", NodeType.LEAF)
        leaf_node.depth = depth
        leaf_node.statistics.sample_count = len(y)
        
        # 予測値計算
        if len(y) > 0:
            leaf_value = float(y.mean())
            confidence = 1.0 - (y.std() / (y.mean() + 1e-10)) if y.mean() != 0 else 0.5
            confidence = max(0.0, min(1.0, confidence))
        else:
            leaf_value = 0.5
            confidence = 0.0
        
        leaf_node.set_leaf_value(leaf_value, confidence)
        
        # クラス分布計算（回帰問題でも離散化して分布を作成）
        if len(y) > 0:
            # 値を3つのカテゴリに分類
            y_min, y_max = y.min(), y.max()
            if y_max > y_min:
                y_normalized = (y - y_min) / (y_max - y_min)
                low_count = np.sum(y_normalized < 0.33)
                mid_count = np.sum((y_normalized >= 0.33) & (y_normalized < 0.67))
                high_count = np.sum(y_normalized >= 0.67)
                total = low_count + mid_count + high_count
                
                if total > 0:
                    leaf_node.class_distribution = {
                        'low': low_count / total,
                        'medium': mid_count / total,
                        'high': high_count / total
                    }
        
        return leaf_node
    
    def _find_best_split(self, X: pd.DataFrame, y: pd.Series, 
                        feature_names: List[str]) -> Tuple[str, Dict[str, MembershipFunction], float]:
        """最適分割の探索"""
        
        best_feature = None
        best_membership_functions = {}
        best_score = -float('inf')
        
        # 特徴量選択
        if self.feature_selection_probs is not None:
            # 確率に基づく特徴量選択
            selected_features = np.random.choice(
                feature_names, 
                size=min(len(feature_names), 3),
                replace=False,
                p=self.feature_selection_probs
            )
        else:
            selected_features = feature_names
        
        for feature in selected_features:
            self.splits_attempted += 1
            
            # この特徴量に対するメンバーシップ関数生成
            membership_functions = self._generate_membership_functions(
                X[feature], feature
            )
            
            # 分割品質評価
            score = self._evaluate_split_quality(X, y, feature, membership_functions)
            
            if score > best_score:
                best_score = score
                best_feature = feature
                best_membership_functions = membership_functions
        
        return best_feature, best_membership_functions, best_score
    
    def _generate_membership_functions(self, feature_values: pd.Series, 
                                     feature_name: str) -> Dict[str, MembershipFunction]:
        """メンバーシップ関数の生成"""
        
        # 既存のパラメータがあるか確認
        if feature_name in self.membership_params:
            return self._create_membership_functions_from_params(
                self.membership_params[feature_name]
            )
        
        # 自動生成
        min_val, max_val = feature_values.min(), feature_values.max()
        
        if max_val <= min_val:
            # 定数値の場合
            mf = MembershipFunction("constant")
            return {"constant": mf}
        
        # 3つのメンバーシップ関数を生成（Low, Medium, High）
        membership_functions = {}
        
        if self.config.membership_type == MembershipType.TRIANGULAR:
            # 三角形メンバーシップ関数
            range_size = max_val - min_val
            overlap = range_size * 0.1  # 10%のオーバーラップ
            
            # Low
            low_mf = TriangularMF(
                "low", 
                min_val - overlap, 
                min_val + range_size * 0.1, 
                min_val + range_size * 0.4
            )
            membership_functions["low"] = low_mf
            
            # Medium
            mid_mf = TriangularMF(
                "medium",
                min_val + range_size * 0.2,
                min_val + range_size * 0.5,
                min_val + range_size * 0.8
            )
            membership_functions["medium"] = mid_mf
            
            # High
            high_mf = TriangularMF(
                "high",
                min_val + range_size * 0.6,
                min_val + range_size * 0.9,
                max_val + overlap
            )
            membership_functions["high"] = high_mf
            
        else:  # Gaussian
            # ガウシアンメンバーシップ関数
            sigma = (max_val - min_val) / 6  # 3σで全範囲をカバー
            
            membership_functions["low"] = GaussianMF("low", min_val + (max_val - min_val) * 0.2, sigma)
            membership_functions["medium"] = GaussianMF("medium", min_val + (max_val - min_val) * 0.5, sigma)
            membership_functions["high"] = GaussianMF("high", min_val + (max_val - min_val) * 0.8, sigma)
        
        return membership_functions
    
    def _create_membership_functions_from_params(self, params: Dict[str, Any]) -> Dict[str, MembershipFunction]:
        """パラメータからメンバーシップ関数を作成"""
        
        membership_functions = {}
        
        for name, mf_params in params.items():
            mf_type = mf_params.get('type', 'triangular')
            
            if mf_type == 'triangular':
                mf = TriangularMF(
                    name,
                    mf_params['a'],
                    mf_params['b'],
                    mf_params['c']
                )
            elif mf_type == 'gaussian':
                mf = GaussianMF(
                    name,
                    mf_params['center'],
                    mf_params['sigma']
                )
            else:
                mf = MembershipFunction(name)
            
            membership_functions[name] = mf
        
        return membership_functions
    
    def _evaluate_split_quality(self, X: pd.DataFrame, y: pd.Series, 
                               feature_name: str, 
                               membership_functions: Dict[str, MembershipFunction]) -> float:
        """分割品質の評価"""
        
        try:
            # データを分割
            children_data = self._split_data_fuzzy(X, y, feature_name, membership_functions)
            
            # 各分割の品質計算
            total_samples = len(y)
            parent_variance = y.var() if len(y) > 1 else 0
            
            weighted_child_variance = 0.0
            total_child_samples = 0
            
            for label, (child_X, child_y) in children_data.items():
                if len(child_y) > 0:
                    child_variance = child_y.var() if len(child_y) > 1 else 0
                    weight = len(child_y) / total_samples
                    weighted_child_variance += weight * child_variance
                    total_child_samples += len(child_y)
            
            # 分散減少量を品質指標とする
            if total_child_samples == 0:
                return 0.0
            
            variance_reduction = parent_variance - weighted_child_variance
            
            # 正規化（0-1の範囲）
            normalized_score = min(1.0, max(0.0, variance_reduction / (parent_variance + 1e-10)))
            
            return normalized_score
            
        except Exception as e:
            print(f"分割品質評価エラー: {e}")
            return 0.0
    
    def _split_data_fuzzy(self, X: pd.DataFrame, y: pd.Series, feature_name: str,
                         membership_functions: Dict[str, MembershipFunction]) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """ファジィ分割によるデータ分割"""
        
        children_data = {}
        
        for label, mf in membership_functions.items():
            # このメンバーシップ関数に最も適合するサンプルを選択
            feature_values = X[feature_name]
            memberships = [mf.membership_degree(val) for val in feature_values]
            
            # 閾値以上のメンバーシップ度を持つサンプルを選択
            threshold = 0.5
            selected_indices = [i for i, m in enumerate(memberships) if m >= threshold]
            
            if selected_indices:
                child_X = X.iloc[selected_indices]
                child_y = y.iloc[selected_indices]
                children_data[label] = (child_X, child_y)
            else:
                # 閾値を満たすサンプルがない場合、最大メンバーシップ度のサンプルを選択
                max_membership_idx = np.argmax(memberships)
                child_X = X.iloc[[max_membership_idx]]
                child_y = y.iloc[[max_membership_idx]]
                children_data[label] = (child_X, child_y)
        
        return children_data
    
    def build_from_genes(self, training_data: np.ndarray, feature_names: List[str],
                        target_name: str, genome: np.ndarray) -> FuzzyDecisionNode:
        """遺伝子から決定木を構築"""
        
        try:
            # データフレーム作成
            columns = feature_names + [target_name]
            data = pd.DataFrame(training_data, columns=columns)
            
            # 遺伝子から構築パラメータを抽出
            if len(genome) > 0:
                # 深度調整
                depth_factor = genome[0]
                adjusted_depth = int(self.config.max_depth * (0.5 + depth_factor * 0.5))
                self.config.max_depth = max(2, adjusted_depth)
                
                # 最小サンプル数調整
                if len(genome) > 1:
                    sample_factor = genome[1]
                    adjusted_samples = int(self.config.min_samples_leaf * (0.5 + sample_factor * 1.5))
                    self.config.min_samples_leaf = max(2, adjusted_samples)
            
            # 決定木構築
            tree = self.build_tree(data, feature_names, target_name)
            return tree.root
            
        except Exception as e:
            print(f"遺伝子からの構築エラー: {e}")
            # フォールバック：簡単な葉ノード
            fallback_node = FuzzyDecisionNode("fallback_leaf", NodeType.LEAF)
            fallback_node.set_leaf_value(0.5, 0.5)
            return fallback_node
    
    def get_builder_statistics(self) -> Dict[str, Any]:
        """構築器統計の取得"""
        
        success_rate = (self.successful_splits / max(1, self.splits_attempted)) * 100
        
        return {
            'nodes_created': self.nodes_created,
            'splits_attempted': self.splits_attempted,
            'successful_splits': self.successful_splits,
            'success_rate': success_rate,
            'config': {
                'max_depth': self.config.max_depth,
                'min_samples_leaf': self.config.min_samples_leaf,
                'split_criterion': self.config.split_criterion.value,
                'membership_type': self.config.membership_type.value
            }
        }


class AdaptiveFuzzyTreeBuilder(FuzzyTreeBuilder):
    """適応的ファジィ決定木構築器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptation_count = 0
        self.performance_history = []
    
    def build_tree(self, data: pd.DataFrame, feature_names: List[str], 
                   target_name: str) -> FuzzyDecisionTree:
        """適応的決定木構築"""
        
        print(f"適応的決定木構築開始: サンプル数={len(data)}")
        
        # 初期構築
        tree = super().build_tree(data, feature_names, target_name)
        
        # 性能評価
        initial_performance = self._evaluate_tree_performance(tree, data, feature_names, target_name)
        self.performance_history.append(initial_performance)
        
        # 適応的改善
        improved_tree = self._adaptive_improvement(tree, data, feature_names, target_name)
        
        print(f"適応的構築完了: 改善回数={self.adaptation_count}")
        
        return improved_tree
    
    def _evaluate_tree_performance(self, tree: FuzzyDecisionTree, 
                                 data: pd.DataFrame, feature_names: List[str], 
                                 target_name: str) -> Dict[str, float]:
        """ツリー性能評価"""
        
        X = data[feature_names]
        y = data[target_name]
        
        # 予測実行
        predictions = []
        for _, row in X.iterrows():
            features = row.to_dict()
            prediction = tree.predict(features)
            predictions.append(prediction)
        
        # 性能指標計算
        actuals = y.tolist()
        mse = np.mean([(p - a) ** 2 for p, a in zip(predictions, actuals)])
        mae = np.mean([abs(p - a) for p, a in zip(predictions, actuals)])
        
        return {
            'mse': mse,
            'mae': mae,
            'complexity': tree.total_nodes,
            'depth': tree.max_depth
        }
    
    def _adaptive_improvement(self, tree: FuzzyDecisionTree, 
                            data: pd.DataFrame, feature_names: List[str], 
                            target_name: str) -> FuzzyDecisionTree:
        """適応的改善"""
        
        current_tree = tree
        best_performance = self.performance_history[-1]
        
        max_iterations = 3
        
        for iteration in range(max_iterations):
            # パラメータ調整
            self._adapt_parameters(best_performance)
            
            # 再構築
            candidate_tree = super().build_tree(data, feature_names, target_name)
            
            # 評価
            candidate_performance = self._evaluate_tree_performance(
                candidate_tree, data, feature_names, target_name
            )
            
            # 改善判定（MSEが小さく、複雑さも考慮）
            if self._is_better_tree(candidate_performance, best_performance):
                current_tree = candidate_tree
                best_performance = candidate_performance
                self.adaptation_count += 1
                print(f"  改善 {iteration+1}: MSE={candidate_performance['mse']:.4f}")
            
            self.performance_history.append(candidate_performance)
        
        return current_tree
    
    def _adapt_parameters(self, performance: Dict[str, float]):
        """パラメータ適応"""
        
        # MSEが高い場合は深度を増加
        if performance['mse'] > 0.1:
            self.config.max_depth = min(10, self.config.max_depth + 1)
            self.config.min_samples_leaf = max(2, self.config.min_samples_leaf - 1)
        
        # 複雑すぎる場合は簡素化
        elif performance['complexity'] > 20:
            self.config.max_depth = max(3, self.config.max_depth - 1)
            self.config.min_samples_leaf += 1
    
    def _is_better_tree(self, candidate: Dict[str, float], current_best: Dict[str, float]) -> bool:
        """ツリー改善判定"""
        
        # MSEの改善度
        mse_improvement = (current_best['mse'] - candidate['mse']) / current_best['mse']
        
        # 複雑度のペナルティ
        complexity_penalty = (candidate['complexity'] - current_best['complexity']) / max(1, current_best['complexity'])
        
        # 総合スコア（MSE改善 - 複雑度ペナルティ）
        improvement_score = mse_improvement - 0.1 * complexity_penalty
        
        return improvement_score > 0.01  # 1%以上の改善で採用


class TreePruner:
    """決定木剪定器"""
    
    def __init__(self, min_samples_leaf: int = 5, max_depth: int = 10):
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.pruned_count = 0
    
    def prune_tree(self, root: FuzzyDecisionNode) -> int:
        """ツリーの剪定"""
        
        if not root:
            return 0
        
        self.pruned_count = 0
        self._prune_recursive(root)
        
        return self.pruned_count
    
    def _prune_recursive(self, node: FuzzyDecisionNode) -> bool:
        """再帰的剪定"""
        
        if node.is_leaf:
            return False
        
        # 子ノードを先に剪定
        children_to_remove = []
        for label, child in node.children.items():
            if self._prune_recursive(child):
                children_to_remove.append(label)
        
        # 不要な子ノードを削除
        for label in children_to_remove:
            node.remove_child(label)
            self.pruned_count += 1
        
        # このノード自体の剪定判定
        should_prune = self._should_prune_node(node)
        
        if should_prune:
            # 内部ノードを葉ノードに変換
            self._convert_to_leaf(node)
            self.pruned_count += 1
            return True
        
        return False
    
    def _should_prune_node(self, node: FuzzyDecisionNode) -> bool:
        """ノード剪定判定"""
        
        # 深度制限
        if node.depth > self.max_depth:
            return True
        
        # サンプル数制限
        if node.statistics.sample_count < self.min_samples_leaf:
            return True
        
        # 子ノードがすべて葉で、性能向上が期待できない場合
        if all(child.is_leaf for child in node.children.values()):
            if len(node.children) <= 1:
                return True
            
            # 子ノードの予測値がほぼ同じ場合
            child_values = [child.leaf_value for child in node.children.values() 
                          if child.leaf_value is not None]
            if len(child_values) > 1:
                value_range = max(child_values) - min(child_values)
                if value_range < 0.05:  # 5%未満の差
                    return True
        
        return False
    
    def _convert_to_leaf(self, node: FuzzyDecisionNode):
        """内部ノードを葉ノードに変換"""
        
        # 子ノードの予測値から平均を計算
        if node.children:
            child_values = []
            child_confidences = []
            
            for child in node.children.values():
                if child.leaf_value is not None:
                    child_values.append(child.leaf_value)
                    child_confidences.append(child.statistics.confidence_score)
            
            if child_values:
                avg_value = np.mean(child_values)
                avg_confidence = np.mean(child_confidences) * 0.8  # 剪定により信頼度を下げる
            else:
                avg_value = 0.5
                avg_confidence = 0.3
        else:
            avg_value = 0.5
            avg_confidence = 0.3
        
        # 葉ノードに変換
        node.children.clear()
        node.membership_functions.clear()
        node.feature_name = None
        node.set_leaf_value(avg_value, avg_confidence)
    
    def get_pruning_statistics(self) -> Dict[str, Any]:
        """剪定統計の取得"""
        
        return {
            'pruned_nodes': self.pruned_count,
            'pruning_config': {
                'min_samples_leaf': self.min_samples_leaf,
                'max_depth': self.max_depth
            }
        }