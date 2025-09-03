"""
ファジィ決定木構築 - core/decision_tree/builder.py
遺伝的アルゴリズムの個体から決定木を構築
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import random
from dataclasses import dataclass
from enum import Enum

from .node import FuzzyDecisionNode, FuzzyDecisionTree, NodeType
from ..fuzzy.membership import (
    MembershipFunction, TriangularMF, GaussianMF, 
    MembershipFunctionFactory, MembershipType
)


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
    
    def build_from_genes(self, training_data: np.ndarray, feature_names: List[str],
                        target_name: str, genome: np.ndarray) -> FuzzyDecisionNode:
        """遺伝子から決定木を構築"""
        
        # データフレーム作成
        data = pd.DataFrame(training_data, columns=feature_names + [target_name])
        
        # 遺伝子解釈
        self._interpret_genome(genome, feature_names)
        
        # ルートノード構築
        X = data[feature_names]
        y = data[target_name]
        
        root = self._build_node_from_genes(X, y, feature_names, depth=0, gene_index=0)
        
        return root
    
    def _interpret_genome(self, genome: np.ndarray, feature_names: List[str]):
        """遺伝子の解釈"""
        
        gene_idx = 0
        
        # 木構造パラメータ
        if gene_idx < len(genome):
            # 最大深度（2-8の範囲）
            self.config.max_depth = int(2 + genome[gene_idx] * 6)
            gene_idx += 1
        
        # 特徴量選択確率
        if self.feature_selection_probs is None:
            num_features = len(feature_names)
            if gene_idx + num_features <= len(genome):
                self.feature_selection_probs = genome[gene_idx:gene_idx + num_features]
                self.feature_selection_probs /= np.sum(self.feature_selection_probs)
                gene_idx += num_features
        
        # メンバーシップ関数パラメータの生成
        self._generate_membership_functions_from_genome(genome, feature_names, gene_idx)
    
    def _generate_membership_functions_from_genome(self, genome: np.ndarray, 
                                                 feature_names: List[str], start_idx: int):
        """遺伝子からメンバーシップ関数パラメータを生成"""
        
        if not self.membership_params:
            self.membership_params = {}
        
        gene_idx = start_idx
        
        for feature in feature_names:
            if feature not in self.membership_params:
                self.membership_params[feature] = {}
                
                # Low, Medium, High の3つのファジィ集合
                for i, fuzzy_name in enumerate(['Low', 'Medium', 'High']):
                    if gene_idx + 2 < len(genome):
                        # 基準位置から調整
                        base_center = i * 3.33  # 0, 3.33, 6.66
                        center_offset = (genome[gene_idx] - 0.5) * 2.0
                        width_factor = 0.8 + genome[gene_idx + 1] * 0.8
                        
                        center = max(0, min(10, base_center + center_offset))
                        width = width_factor * 1.5
                        
                        left = max(0, center - width)
                        right = min(10, center + width)
                        
                        self.membership_params[feature][fuzzy_name] = {
                            'type': 'triangular',
                            'a': left,
                            'b': center,
                            'c': right
                        }
                        
                        gene_idx += 2
                    else:
                        # デフォルト値
                        self.membership_params[feature][fuzzy_name] = {
                            'type': 'triangular',
                            'a': i * 2.5,
                            'b': i * 3.33 + 1.67,
                            'c': (i + 1) * 3.33
                        }
    
    def _build_node_recursive(self, X: pd.DataFrame, y: pd.Series, 
                             feature_names: List[str], depth: int) -> FuzzyDecisionNode:
        """再帰的ノード構築（標準版）"""
        
        self.nodes_created += 1
        
        # 停止条件チェック
        if self._should_stop_splitting(X, y, depth):
            return self._create_leaf_node(X, y, depth)
        
        # 最良分割の探索
        best_feature, best_gain = self._find_best_split(X, y, feature_names)
        
        if best_feature is None or best_gain < self.config.min_information_gain:
            return self._create_leaf_node(X, y, depth)
        
        # 内部ノード作成
        node = FuzzyDecisionNode(f"node_{self.nodes_created}", NodeType.INTERNAL)
        node.feature_name = best_feature
        node.depth = depth
        
        if depth == 0:
            node.node_type = NodeType.ROOT
        
        # メンバーシップ関数生成
        fuzzy_sets = self._generate_membership_functions(best_feature, X[best_feature])
        
        for label, mf in fuzzy_sets.items():
            node.add_membership_function(label, mf)
        
        # 訓練データ保存
        training_samples = [(X.iloc[i].to_dict(), y.iloc[i]) for i in range(len(X))]
        node.update_training_statistics(training_samples)
        
        # 子ノード生成
        for label, mf in fuzzy_sets.items():
            child_X, child_y = self._fuzzy_split_data(X, y, feature, mf)
            
            if len(child_y) > 0:
                child_variance = np.var(child_y)
                weight = len(child_y) / len(y)
                weighted_variance += weight * child_variance
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        variance_reduction = total_variance - weighted_variance
        return max(0.0, variance_reduction)
    
    def _calculate_information_gain(self, X: pd.DataFrame, y: pd.Series, feature: str) -> float:
        """情報ゲイン計算（回帰用）"""
        # 回帰問題では分散削減量を情報ゲインとして使用
        return self._calculate_variance_reduction(X, y, feature)
    
    def _generate_membership_functions(self, feature_name: str, 
                                     feature_values: pd.Series) -> Dict[str, MembershipFunction]:
        """メンバーシップ関数の生成"""
        
        # パラメータが指定されている場合
        if (feature_name in self.membership_params and 
            self.membership_params[feature_name]):
            return self._create_membership_functions_from_params(feature_name)
        
        # デフォルト生成
        min_val, max_val = feature_values.min(), feature_values.max()
        if min_val == max_val:
            min_val, max_val = 0, 10
        
        return MembershipFunctionFactory.create_fuzzy_sets(
            feature_name, 
            (min_val, max_val),
            num_sets=self.config.fuzzy_sets_per_feature,
            mf_type=self.config.membership_type
        )
    
    def _create_membership_functions_from_params(self, feature_name: str) -> Dict[str, MembershipFunction]:
        """パラメータからメンバーシップ関数作成"""
        
        fuzzy_sets = {}
        
        if feature_name not in self.membership_params:
            # デフォルトパラメータ
            return MembershipFunctionFactory.create_fuzzy_sets(
                feature_name, (0, 10), num_sets=3
            )
        
        for fuzzy_set_name, params in self.membership_params[feature_name].items():
            if params['type'] == 'triangular':
                mf = TriangularMF(
                    f"{feature_name}_{fuzzy_set_name}",
                    params['a'], params['b'], params['c']
                )
            elif params['type'] == 'gaussian':
                mf = GaussianMF(
                    f"{feature_name}_{fuzzy_set_name}",
                    params.get('center', 5.0),
                    params.get('sigma', 1.0)
                )
            else:
                # デフォルトは三角形
                mf = TriangularMF(
                    f"{feature_name}_{fuzzy_set_name}",
                    params.get('a', 0), params.get('b', 5), params.get('c', 10)
                )
            
            fuzzy_sets[fuzzy_set_name] = mf
        
        return fuzzy_sets
    
    def _fuzzy_split_data(self, X: pd.DataFrame, y: pd.Series, 
                         feature: str, mf: MembershipFunction) -> Tuple[pd.DataFrame, pd.Series]:
        """ファジィ分割によるデータ分割"""
        
        feature_values = X[feature]
        
        # メンバーシップ度が閾値以上のサンプルを選択
        threshold = 0.3
        selected_indices = []
        
        for idx, value in feature_values.items():
            membership = mf.membership(value)
            if membership > threshold:
                selected_indices.append(idx)
        
        if not selected_indices:
            # 閾値以上のサンプルがない場合、最大メンバーシップ度のサンプルを選択
            max_membership = -1
            best_idx = None
            
            for idx, value in feature_values.items():
                membership = mf.membership(value)
                if membership > max_membership:
                    max_membership = membership
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices = [best_idx]
        
        # 選択されたサンプルを返す
        child_X = X.loc[selected_indices]
        child_y = y.loc[selected_indices]
        
        return child_X, child_y
    
    def _create_leaf_node(self, X: pd.DataFrame, y: pd.Series, depth: int) -> FuzzyDecisionNode:
        """葉ノード作成"""
        
        node = FuzzyDecisionNode(f"leaf_{self.nodes_created}", NodeType.LEAF)
        node.is_leaf = True
        node.depth = depth
        
        # 葉ノードの値設定
        if len(y) > 0:
            node.leaf_value = float(y.mean())
        else:
            node.leaf_value = 0.5
        
        # 訓練統計更新
        training_samples = [(X.iloc[i].to_dict(), y.iloc[i]) for i in range(len(X))]
        node.update_training_statistics(training_samples)
        
        return node
    
    def get_builder_statistics(self) -> Dict[str, Any]:
        """構築統計の取得"""
        
        return {
            'nodes_created': self.nodes_created,
            'splits_attempted': self.splits_attempted,
            'successful_splits': self.successful_splits,
            'split_success_rate': self.successful_splits / max(1, self.splits_attempted),
            'config': {
                'max_depth': self.config.max_depth,
                'min_samples_split': self.config.min_samples_split,
                'min_samples_leaf': self.config.min_samples_leaf,
                'split_criterion': self.config.split_criterion.value,
                'fuzzy_sets_per_feature': self.config.fuzzy_sets_per_feature
            },
            'membership_params_count': len(self.membership_params)
        }
    
    def reset_statistics(self):
        """統計のリセット"""
        self.nodes_created = 0
        self.splits_attempted = 0
        self.successful_splits = 0


class AdaptiveFuzzyTreeBuilder(FuzzyTreeBuilder):
    """適応的ファジィ決定木構築器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_importance_history: Dict[str, List[float]] = {}
        self.split_quality_history: List[float] = []
    
    def _find_best_split(self, X: pd.DataFrame, y: pd.Series, 
                        feature_names: List[str]) -> Tuple[Optional[str], float]:
        """適応的最良分割探索"""
        
        # 特徴量重要度履歴に基づく候補選択
        candidate_features = self._select_candidate_features(feature_names)
        
        best_feature = None
        best_gain = -np.inf
        
        for feature in candidate_features:
            gain = self._calculate_split_criterion(X, y, feature)
            
            # 特徴量重要度履歴更新
            if feature not in self.feature_importance_history:
                self.feature_importance_history[feature] = []
            self.feature_importance_history[feature].append(gain)
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        
        # 分割品質履歴更新
        self.split_quality_history.append(best_gain)
        
        return best_feature, best_gain
    
    def _select_candidate_features(self, feature_names: List[str]) -> List[str]:
        """候補特徴量の適応的選択"""
        
        if not self.feature_importance_history:
            # 履歴がない場合はランダム選択
            return random.sample(feature_names, min(3, len(feature_names)))
        
        # 重要度履歴に基づく重み付け選択
        feature_weights = {}
        for feature in feature_names:
            if feature in self.feature_importance_history:
                # 最近の重要度の平均
                recent_importance = self.feature_importance_history[feature][-5:]
                feature_weights[feature] = np.mean(recent_importance)
            else:
                feature_weights[feature] = 0.1  # 新しい特徴量には低い重み
        
        # 重み付きランダム選択
        total_weight = sum(feature_weights.values())
        if total_weight > 0:
            probabilities = [feature_weights[f] / total_weight for f in feature_names]
            num_candidates = min(4, len(feature_names))
            
            try:
                candidates = np.random.choice(
                    feature_names, size=num_candidates, replace=False, p=probabilities
                ).tolist()
            except:
                # エラーの場合はランダム選択
                candidates = random.sample(feature_names, num_candidates)
        else:
            candidates = random.sample(feature_names, min(3, len(feature_names)))
        
        return candidates
    
    def adapt_parameters(self, generation: int):
        """パラメータの適応的調整"""
        
        if len(self.split_quality_history) < 10:
            return
        
        # 最近の分割品質の傾向を分析
        recent_quality = self.split_quality_history[-10:]
        quality_trend = np.mean(np.diff(recent_quality))
        
        # 品質が低下傾向の場合、探索を強化
        if quality_trend < 0:
            self.config.max_features = min(
                len(self.feature_importance_history),
                self.config.max_features + 1 if self.config.max_features else 4
            )
            self.config.min_information_gain *= 0.9
        else:
            # 品質が向上傾向の場合、効率化
            if self.config.max_features and self.config.max_features > 2:
                self.config.max_features -= 1
            self.config.min_information_gain *= 1.05
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """適応統計の取得"""
        
        stats = self.get_builder_statistics()
        
        # 適応的統計を追加
        stats['adaptation_info'] = {
            'feature_importance_history': self.feature_importance_history,
            'split_quality_trend': np.mean(np.diff(self.split_quality_history[-10:])) if len(self.split_quality_history) > 10 else 0.0,
            'average_split_quality': np.mean(self.split_quality_history) if self.split_quality_history else 0.0,
            'quality_variance': np.var(self.split_quality_history) if self.split_quality_history else 0.0
        }
        
        return stats


class TreePruner:
    """決定木剪定器"""
    
    def __init__(self, min_samples_leaf: int = 5, max_depth: int = 10):
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.pruned_nodes = 0
    
    def prune_tree(self, root: FuzzyDecisionNode, validation_data: pd.DataFrame = None) -> int:
        """決定木の剪定"""
        
        self.pruned_nodes = 0
        
        # 後剪定
        self._post_prune_recursive(root, validation_data)
        
        return self.pruned_nodes
    
    def _post_prune_recursive(self, node: FuzzyDecisionNode, validation_data: pd.DataFrame = None):
        """再帰的後剪定"""
        
        if node.is_leaf:
            return
        
        # 子ノードを先に剪定
        for child in list(node.children.values()):
            self._post_prune_recursive(child, validation_data)
        
        # 現在のノードを剪定すべきか判定
        should_prune = False
        
        # サンプル数による剪定
        if node.statistics.sample_count < self.min_samples_leaf:
            should_prune = True
        
        # 深度による剪定
        if node.depth > self.max_depth:
            should_prune = True
        
        # 検証データがある場合の性能ベース剪定
        if validation_data is not None and not should_prune:
            should_prune = self._should_prune_based_on_performance(node, validation_data)
        
        # 剪定実行
        if should_prune:
            node.prune_subtree()
            self.pruned_nodes += 1
    
    def _should_prune_based_on_performance(self, node: FuzzyDecisionNode, 
                                         validation_data: pd.DataFrame) -> bool:
        """性能ベースの剪定判定"""
        
        if len(validation_data) == 0:
            return False
        
        # 剪定前後の性能を比較
        # 実装簡略化のため、基本的な判定のみ
        
        # 子ノードがすべて同じ値を出力する場合は剪定
        if not node.is_leaf:
            child_values = []
            for child in node.children.values():
                if child.is_leaf and child.leaf_value is not None:
                    child_values.append(child.leaf_value)
            
            if len(set(child_values)) == 1:  # すべて同じ値
                return True
        
        return False_data(X, y, best_feature, mf)
            
            if len(child_X) >= self.config.min_samples_leaf:
                child_node = self._build_node_recursive(child_X, child_y, feature_names, depth + 1)
                node.add_child(label, child_node)
        
        # 情報ゲイン計算
        feature_values = X[best_feature].tolist()
        target_values = y.tolist()
        node.calculate_information_gain(feature_values, target_values)
        
        return node
    
    def _build_node_from_genes(self, X: pd.DataFrame, y: pd.Series,
                              feature_names: List[str], depth: int, gene_index: int) -> FuzzyDecisionNode:
        """遺伝子からノード構築"""
        
        self.nodes_created += 1
        
        # 停止条件
        if self._should_stop_splitting(X, y, depth):
            return self._create_leaf_node(X, y, depth)
        
        # 遺伝子による特徴量選択
        if self.feature_selection_probs is not None:
            # 確率に基づく選択
            selected_feature = np.random.choice(feature_names, p=self.feature_selection_probs)
        else:
            # ランダム選択
            selected_feature = random.choice(feature_names)
        
        # 内部ノード作成
        node = FuzzyDecisionNode(f"node_{self.nodes_created}", NodeType.INTERNAL)
        node.feature_name = selected_feature
        node.depth = depth
        
        if depth == 0:
            node.node_type = NodeType.ROOT
        
        # 遺伝子からメンバーシップ関数生成
        fuzzy_sets = self._create_membership_functions_from_params(selected_feature)
        
        for label, mf in fuzzy_sets.items():
            node.add_membership_function(label, mf)
        
        # 訓練データ保存
        training_samples = [(X.iloc[i].to_dict(), y.iloc[i]) for i in range(len(X))]
        node.update_training_statistics(training_samples)
        
        # 子ノード生成
        for label, mf in fuzzy_sets.items():
            child_X, child_y = self._fuzzy_split_data(X, y, selected_feature, mf)
            
            if len(child_X) >= self.config.min_samples_leaf:
                child_node = self._build_node_from_genes(child_X, child_y, feature_names, depth + 1, gene_index)
                node.add_child(label, child_node)
        
        return node
    
    def _should_stop_splitting(self, X: pd.DataFrame, y: pd.Series, depth: int) -> bool:
        """分割停止条件の判定"""
        
        # 深度制限
        if depth >= self.config.max_depth:
            return True
        
        # サンプル数制限
        if len(X) < self.config.min_samples_split:
            return True
        
        # 純度チェック（分散が小さい場合）
        if len(y.unique()) == 1:
            return True
        
        target_variance = np.var(y)
        if target_variance < 0.01:  # 分散が非常に小さい
            return True
        
        return False
    
    def _find_best_split(self, X: pd.DataFrame, y: pd.Series, 
                        feature_names: List[str]) -> Tuple[Optional[str], float]:
        """最良分割の探索"""
        
        best_feature = None
        best_gain = -np.inf
        
        # 考慮する特徴量数の制限
        max_features = self.config.max_features or len(feature_names)
        candidate_features = random.sample(feature_names, min(max_features, len(feature_names)))
        
        for feature in candidate_features:
            self.splits_attempted += 1
            
            # 情報ゲイン計算
            gain = self._calculate_split_criterion(X, y, feature)
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                self.successful_splits += 1
        
        return best_feature, best_gain
    
    def _calculate_split_criterion(self, X: pd.DataFrame, y: pd.Series, feature: str) -> float:
        """分割基準の計算"""
        
        if self.config.split_criterion == SplitCriterion.VARIANCE_REDUCTION:
            return self._calculate_variance_reduction(X, y, feature)
        elif self.config.split_criterion == SplitCriterion.INFORMATION_GAIN:
            return self._calculate_information_gain(X, y, feature)
        else:
            return self._calculate_variance_reduction(X, y, feature)
    
    def _calculate_variance_reduction(self, X: pd.DataFrame, y: pd.Series, feature: str) -> float:
        """分散削減量の計算"""
        
        # 全体の分散
        total_variance = np.var(y)
        
        # 仮のメンバーシップ関数で分割
        temp_fuzzy_sets = self._generate_membership_functions(feature, X[feature])
        
        weighted_variance = 0.0
        total_weight = 0.0
        
        for label, mf in temp_fuzzy_sets.items():
            child_X, child_y = self._fuzzy_split