# core/decision_tree/builder.py - ファジィ決定木構築

import numpy as np
import random
import math
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
import logging

from core.decision_tree.node import (
    FuzzyTreeNode, FuzzyInternalNode, FuzzyLeafNode, 
    SplitCondition, FuzzyRuleNode
)
from core.fuzzy.membership import (
    FuzzyVariable, MembershipFunctionFactory, 
    TriangularMF, GaussianMF
)

logger = logging.getLogger(__name__)

@dataclass
class BuilderConfig:
    """決定木構築設定"""
    # 基本パラメータ
    max_depth: int = 8
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: Optional[int] = None
    
    # ファジィ関連
    fuzzy_threshold: float = 0.1
    membership_overlap: float = 0.3
    linguistic_terms: int = 3  # low, medium, high
    
    # 分岐基準
    split_criterion: str = "fuzzy_gain"  # "fuzzy_gain", "gini", "entropy"
    min_impurity_decrease: float = 1e-7
    
    # 枝刈り
    pruning_enabled: bool = True
    min_confidence_threshold: float = 0.1
    
    # ルール生成
    rule_extraction: bool = True
    max_rules_per_path: int = 10
    
    # その他
    random_state: Optional[int] = None
    parallel_building: bool = False

@dataclass
class SplitEvaluation:
    """分岐評価結果"""
    feature: str
    threshold: float
    linguistic_value: str
    impurity_decrease: float
    left_samples: List[int]
    right_samples: List[int]
    left_purity: float
    right_purity: float
    confidence: float
    
    def is_valid(self) -> bool:
        """有効な分岐かチェック"""
        return (
            self.impurity_decrease > 0 and
            len(self.left_samples) > 0 and
            len(self.right_samples) > 0 and
            self.confidence > 0.1
        )

class FuzzySplitEvaluator:
    """ファジィ分岐評価器"""
    
    def __init__(self, config: BuilderConfig):
        self.config = config
        self.fuzzy_variables: Dict[str, FuzzyVariable] = {}
    
    def setup_fuzzy_variables(self, feature_names: List[str], 
                             feature_ranges: Dict[str, Tuple[float, float]]):
        """ファジィ変数のセットアップ"""
        
        for feature_name in feature_names:
            range_min, range_max = feature_ranges.get(feature_name, (0.0, 10.0))
            
            if self.config.linguistic_terms == 3:
                fuzzy_var = MembershipFunctionFactory.create_standard_sets(
                    feature_name, (range_min, range_max)
                )
            elif self.config.linguistic_terms == 5:
                fuzzy_var = MembershipFunctionFactory.create_five_level_sets(
                    feature_name, (range_min, range_max)
                )
            else:
                # カスタム作成
                fuzzy_var = self._create_custom_fuzzy_variable(
                    feature_name, (range_min, range_max)
                )
            
            self.fuzzy_variables[feature_name] = fuzzy_var
    
    def _create_custom_fuzzy_variable(self, name: str, 
                                    range_tuple: Tuple[float, float]) -> FuzzyVariable:
        """カスタムファジィ変数作成"""
        
        fuzzy_var = FuzzyVariable(name, range_tuple)
        min_val, max_val = range_tuple
        range_val = max_val - min_val
        
        # 均等分割でファジィ集合を作成
        num_terms = self.config.linguistic_terms
        overlap = self.config.membership_overlap
        
        for i in range(num_terms):
            center = min_val + (i / (num_terms - 1)) * range_val
            width = range_val / (num_terms - 1) * (1 + overlap)
            
            # ガウシアンメンバーシップ関数
            mf = GaussianMF(f"term_{i}", center, width / 4)
            
            from core.fuzzy.membership import FuzzySet
            fuzzy_set = FuzzySet(f"term_{i}", mf, range_tuple)
            fuzzy_var.add_set(fuzzy_set)
        
        return fuzzy_var
    
    def evaluate_split(self, X: List[Dict[str, Any]], y: List[str], 
                      samples: List[int], feature: str) -> List[SplitEvaluation]:
        """ファジィ分岐の評価"""
        
        if feature not in self.fuzzy_variables:
            return []
        
        fuzzy_var = self.fuzzy_variables[feature]
        evaluations = []
        
        # 各言語値での分岐を評価
        for set_name, fuzzy_set in fuzzy_var.sets.items():
            
            # ファジィメンバーシップに基づく分割
            left_samples, right_samples = self._fuzzy_split(
                X, samples, feature, fuzzy_set
            )
            
            if len(left_samples) < self.config.min_samples_leaf or \
               len(right_samples) < self.config.min_samples_leaf:
                continue
            
            # 不純度の計算
            parent_impurity = self._calculate_impurity(y, samples)
            left_impurity = self._calculate_impurity(y, left_samples)
            right_impurity = self._calculate_impurity(y, right_samples)
            
            # 重み付き不純度減少
            n_total = len(samples)
            n_left = len(left_samples)
            n_right = len(right_samples)
            
            weighted_impurity = (n_left / n_total) * left_impurity + \
                              (n_right / n_total) * right_impurity
            
            impurity_decrease = parent_impurity - weighted_impurity
            
            # 信頼度の計算
            confidence = self._calculate_split_confidence(
                X, samples, feature, fuzzy_set
            )
            
            # 分岐評価の作成
            evaluation = SplitEvaluation(
                feature=feature,
                threshold=fuzzy_set.centroid(),
                linguistic_value=set_name,
                impurity_decrease=impurity_decrease,
                left_samples=left_samples,
                right_samples=right_samples,
                left_purity=1.0 - left_impurity,
                right_purity=1.0 - right_impurity,
                confidence=confidence
            )
            
            if evaluation.is_valid():
                evaluations.append(evaluation)
        
        return sorted(evaluations, key=lambda x: x.impurity_decrease, reverse=True)
    
    def _fuzzy_split(self, X: List[Dict[str, Any]], samples: List[int], 
                    feature: str, fuzzy_set) -> Tuple[List[int], List[int]]:
        """ファジィメンバーシップに基づく分割"""
        
        left_samples = []
        right_samples = []
        
        for sample_idx in samples:
            feature_value = X[sample_idx].get(feature, 0.0)
            membership = fuzzy_set.membership(feature_value)
            
            # メンバーシップ度に基づく分割
            if membership >= self.config.fuzzy_threshold:
                left_samples.append(sample_idx)
            else:
                right_samples.append(sample_idx)
        
        return left_samples, right_samples
    
    def _calculate_impurity(self, y: List[str], samples: List[int]) -> float:
        """不純度計算"""
        
        if not samples:
            return 0.0
        
        # クラス分布の計算
        class_counts = Counter(y[i] for i in samples)
        total_samples = len(samples)
        
        if self.config.split_criterion == "gini":
            return self._gini_impurity(class_counts, total_samples)
        elif self.config.split_criterion == "entropy":
            return self._entropy_impurity(class_counts, total_samples)
        else:  # fuzzy_gain
            return self._fuzzy_impurity(class_counts, total_samples)
    
    def _gini_impurity(self, class_counts: Counter, total_samples: int) -> float:
        """ジニ不純度"""
        if total_samples == 0:
            return 0.0
        
        impurity = 1.0
        for count in class_counts.values():
            prob = count / total_samples
            impurity -= prob * prob
        
        return impurity
    
    def _entropy_impurity(self, class_counts: Counter, total_samples: int) -> float:
        """エントロピー不純度"""
        if total_samples == 0:
            return 0.0
        
        entropy = 0.0
        for count in class_counts.values():
            if count > 0:
                prob = count / total_samples
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def _fuzzy_impurity(self, class_counts: Counter, total_samples: int) -> float:
        """ファジィ不純度（重み付きエントロピー）"""
        if total_samples == 0:
            return 0.0
        
        # 基本エントロピー
        entropy = self._entropy_impurity(class_counts, total_samples)
        
        # クラス分布の均等性による重み
        num_classes = len(class_counts)
        if num_classes <= 1:
            return 0.0
        
        # 分布の偏りを考慮
        max_count = max(class_counts.values())
        uniformity = 1.0 - (max_count / total_samples)
        
        return entropy * uniformity
    
    def _calculate_split_confidence(self, X: List[Dict[str, Any]], 
                                   samples: List[int], feature: str, 
                                   fuzzy_set) -> float:
        """分岐の信頼度計算"""
        
        if not samples:
            return 0.0
        
        # メンバーシップ度の分布を分析
        memberships = []
        for sample_idx in samples:
            feature_value = X[sample_idx].get(feature, 0.0)
            membership = fuzzy_set.membership(feature_value)
            memberships.append(membership)
        
        # 信頼度は明確な分離度合いで計算
        avg_membership = np.mean(memberships)
        std_membership = np.std(memberships)
        
        # 標準偏差が大きいほど明確な分離
        confidence = min(1.0, std_membership * 2)
        
        return confidence

class FuzzyTreeBuilder:
    """ファジィ決定木構築クラス"""
    
    def __init__(self, config: BuilderConfig):
        self.config = config
        self.split_evaluator = FuzzySplitEvaluator(config)
        
        # 構築統計
        self.nodes_created = 0
        self.max_depth_reached = 0
        self.total_splits_evaluated = 0
        
        # ルール抽出
        self.extracted_rules: List[Dict[str, Any]] = []
        
        if config.random_state is not None:
            random.seed(config.random_state)
            np.random.seed(config.random_state)
    
    def build_tree(self, X: List[Dict[str, Any]], y: List[str], 
                   feature_names: Optional[List[str]] = None) -> FuzzyTreeNode:
        """ファジィ決定木を構築"""
        
        # 特徴量名の設定
        if feature_names is None and X:
            feature_names = list(X[0].keys())
        
        if not feature_names:
            raise ValueError("特徴量名が指定されていません")
        
        # 特徴量の範囲を計算
        feature_ranges = self._calculate_feature_ranges(X, feature_names)
        
        # ファジィ変数のセットアップ
        self.split_evaluator.setup_fuzzy_variables(feature_names, feature_ranges)
        
        # サンプルインデックス
        samples = list(range(len(X)))
        
        logger.info(f"ファジィ決定木構築開始: {len(X)}サンプル, {len(feature_names)}特徴量")
        
        # ルート作成
        root = self._build_node(X, y, samples, depth=0, node_id="root")
        
        # ルール抽出
        if self.config.rule_extraction:
            self._extract_rules(root, X, y)
        
        # 枝刈り
        if self.config.pruning_enabled:
            root = self._prune_tree(root, X, y)
        
        logger.info(f"ファジィ決定木構築完了: {self.nodes_created}ノード, 最大深度{self.max_depth_reached}")
        
        return root
    
    def _calculate_feature_ranges(self, X: List[Dict[str, Any]], 
                                 feature_names: List[str]) -> Dict[str, Tuple[float, float]]:
        """特徴量の範囲を計算"""
        
        ranges = {}
        
        for feature in feature_names:
            values = []
            for sample in X:
                value = sample.get(feature)
                if value is not None and isinstance(value, (int, float)):
                    values.append(float(value))
            
            if values:
                ranges[feature] = (min(values), max(values))
            else:
                ranges[feature] = (0.0, 10.0)  # デフォルト範囲
        
        return ranges
    
    def _build_node(self, X: List[Dict[str, Any]], y: List[str], 
                   samples: List[int], depth: int, node_id: str) -> FuzzyTreeNode:
        """ノードを構築"""
        
        self.nodes_created += 1
        self.max_depth_reached = max(self.max_depth_reached, depth)
        
        # 停止条件のチェック
        if self._should_stop_splitting(samples, depth, y):
            return self._create_leaf_node(y, samples, node_id, depth)
        
        # 最適分岐の探索
        best_split = self._find_best_split(X, y, samples)
        
        if best_split is None or not best_split.is_valid():
            return self._create_leaf_node(y, samples, node_id, depth)
        
        # 内部ノードの作成
        split_condition = SplitCondition(
            feature=best_split.feature,
            threshold=best_split.threshold,
            linguistic_value=best_split.linguistic_value,
            membership_threshold=self.config.fuzzy_threshold
        )
        
        internal_node = FuzzyInternalNode(node_id, split_condition, depth)
        internal_node.samples_count = len(samples)
        
        # 子ノードの再帰的構築
        left_child = self._build_node(
            X, y, best_split.left_samples, depth + 1, f"{node_id}_left"
        )
        right_child = self._build_node(
            X, y, best_split.right_samples, depth + 1, f"{node_id}_right"
        )
        
        internal_node.add_child("left", left_child)
        internal_node.add_child("right", right_child)
        
        return internal_node
    
    def _should_stop_splitting(self, samples: List[int], depth: int, y: List[str]) -> bool:
        """分岐停止条件のチェック"""
        
        # 深度制限
        if depth >= self.config.max_depth:
            return True
        
        # サンプル数制限
        if len(samples) < self.config.min_samples_split:
            return True
        
        # 純粋なノード
        sample_labels = [y[i] for i in samples]
        if len(set(sample_labels)) <= 1:
            return True
        
        return False
    
    def _find_best_split(self, X: List[Dict[str, Any]], y: List[str], 
                        samples: List[int]) -> Optional[SplitEvaluation]:
        """最適分岐の探索"""
        
        feature_names = list(self.split_evaluator.fuzzy_variables.keys())
        
        # 特徴量選択
        if self.config.max_features:
            feature_names = random.sample(
                feature_names, 
                min(self.config.max_features, len(feature_names))
            )
        
        best_split = None
        best_score = -float('inf')
        
        for feature in feature_names:
            # 各特徴量での分岐を評価
            split_evaluations = self.split_evaluator.evaluate_split(X, y, samples, feature)
            self.total_splits_evaluated += len(split_evaluations)
            
            for evaluation in split_evaluations:
                if evaluation.impurity_decrease > self.config.min_impurity_decrease:
                    # スコア計算（不純度減少 + 信頼度）
                    score = evaluation.impurity_decrease * evaluation.confidence
                    
                    if score > best_score:
                        best_score = score
                        best_split = evaluation
        
        return best_split
    
    def _create_leaf_node(self, y: List[str], samples: List[int], 
                         node_id: str, depth: int) -> FuzzyLeafNode:
        """葉ノードの作成"""
        
        # クラス分布の計算
        sample_labels = [y[i] for i in samples]
        class_counts = Counter(sample_labels)
        
        # 多数決クラス
        predicted_class = class_counts.most_common(1)[0][0]
        
        # 信頼度（多数決クラスの割合）
        confidence = class_counts[predicted_class] / len(samples) if samples else 0.0
        
        # クラス確率分布
        class_probabilities = {}
        for class_name, count in class_counts.items():
            class_probabilities[class_name] = count / len(samples)
        
        leaf_node = FuzzyLeafNode(
            node_id=node_id,
            predicted_class=predicted_class,
            class_probabilities=class_probabilities,
            confidence=confidence,
            depth=depth
        )
        
        leaf_node.samples_count = len(samples)
        
        return leaf_node
    
    def _extract_rules(self, root: FuzzyTreeNode, X: List[Dict[str, Any]], y: List[str]):
        """ルールの抽出"""
        
        self.extracted_rules = []
        
        def extract_path_rules(node: FuzzyTreeNode, path: List[str], conditions: List[str]):
            if node.is_leaf():
                # 葉ノードに到達したらルールを抽出
                rule = {
                    "conditions": conditions.copy(),
                    "conclusion": node.predicted_class,
                    "confidence": node.confidence,
                    "path": " -> ".join(path),
                    "samples": node.samples_count
                }
                self.extracted_rules.append(rule)
            else:
                # 内部ノードの場合は子ノードを探索
                for branch_name, child in node.children.items():
                    condition = f"{node.split_condition.feature} IS {node.split_condition.linguistic_value}"
                    if branch_name == "right":
                        condition = f"NOT ({condition})"
                    
                    new_path = path + [condition]
                    new_conditions = conditions + [condition]
                    
                    extract_path_rules(child, new_path, new_conditions)
        
        extract_path_rules(root, [], [])
        
        logger.info(f"ルール抽出完了: {len(self.extracted_rules)}ルール")
    
    def _prune_tree(self, root: FuzzyTreeNode, X: List[Dict[str, Any]], y: List[str]) -> FuzzyTreeNode:
        """決定木の枝刈り"""
        
        def prune_node(node: FuzzyTreeNode) -> FuzzyTreeNode:
            if node.is_leaf():
                return node
            
            # 子ノードを再帰的に枝刈り
            pruned_children = {}
            for branch_name, child in node.children.items():
                pruned_child = prune_node(child)
                pruned_children[branch_name] = pruned_child
            
            # 信頼度による枝刈り判定
            if all(child.is_leaf() for child in pruned_children.values()):
                # 全ての子が葉ノードの場合、枝刈りを検討
                if self._should_prune_node(node, pruned_children):
                    # 葉ノードに変換
                    return self._convert_to_leaf(node, pruned_children)
            
            # 子ノードを更新
            node.children = pruned_children
            return node
        
        pruned_root = prune_node(root)
        logger.info("決定木の枝刈り完了")
        
        return pruned_root
    
    def _should_prune_node(self, node: FuzzyInternalNode, children: Dict[str, FuzzyTreeNode]) -> bool:
        """ノードを枝刈りすべきかの判定"""
        
        # 子ノードの信頼度が低い場合は枝刈り
        min_confidence = min(child.confidence for child in children.values())
        
        return min_confidence < self.config.min_confidence_threshold
    
    def _convert_to_leaf(self, node: FuzzyInternalNode, children: Dict[str, FuzzyTreeNode]) -> FuzzyLeafNode:
        """内部ノードを葉ノードに変換"""
        
        # 子ノードのクラス分布を統合
        combined_probabilities = defaultdict(float)
        total_samples = 0
        
        for child in children.values():
            for class_name, prob in child.class_probabilities.items():
                combined_probabilities[class_name] += prob * child.samples_count
            total_samples += child.samples_count
        
        # 正規化
        if total_samples > 0:
            for class_name in combined_probabilities:
                combined_probabilities[class_name] /= total_samples
        
        # 多数決クラス
        predicted_class = max(combined_probabilities, key=combined_probabilities.get)
        confidence = combined_probabilities[predicted_class]
        
        return FuzzyLeafNode(
            node_id=node.node_id + "_pruned",
            predicted_class=predicted_class,
            class_probabilities=dict(combined_probabilities),
            confidence=confidence,
            depth=node.depth
        )
    
    def get_build_statistics(self) -> Dict[str, Any]:
        """構築統計の取得"""
        
        return {
            "nodes_created": self.nodes_created,
            "max_depth_reached": self.max_depth_reached,
            "total_splits_evaluated": self.total_splits_evaluated,
            "extracted_rules_count": len(self.extracted_rules),
            "config": self.config
        }
    
    def get_extracted_rules(self) -> List[Dict[str, Any]]:
        """抽出されたルールを取得"""
        return self.extracted_rules.copy()

# 使用例とテスト
def test_fuzzy_tree_builder():
    """ファジィ決定木構築のテスト"""
    
    print("🌳 ファジィ決定木構築テスト開始")
    
    # 設定の作成
    config = BuilderConfig(
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        fuzzy_threshold=0.2,
        rule_extraction=True,
        pruning_enabled=True
    )
    
    # 構築器の初期化
    builder = FuzzyTreeBuilder(config)
    
    # テストデータの作成
    X = [
        {"feature1": 8.0, "feature2": 7.0, "feature3": 6.0},
        {"feature1": 2.0, "feature2": 3.0, "feature3": 4.0},
        {"feature1": 7.0, "feature2": 8.0, "feature3": 9.0},
        {"feature1": 3.0, "feature2": 2.0, "feature3": 1.0},
        {"feature1": 6.0, "feature2": 6.0, "feature3": 7.0},
        {"feature1": 4.0, "feature2": 5.0, "feature3": 3.0}
    ]
    
    y = ["high_match", "low_match", "high_match", "low_match", "medium_match", "medium_match"]
    
    # 決定木構築
    root = builder.build_tree(X, y, ["feature1", "feature2", "feature3"])
    
    print(f"✅ 決定木構築完了")
    
    # 統計情報
    stats = builder.get_build_statistics()
    print(f"📊 構築統計:")
    print(f"  作成ノード数: {stats['nodes_created']}")
    print(f"  最大深度: {stats['max_depth_reached']}")
    print(f"  分岐評価回数: {stats['total_splits_evaluated']}")
    
    # 抽出ルール
    rules = builder.get_extracted_rules()
    print(f"\n📋 抽出ルール数: {len(rules)}")
    for i, rule in enumerate(rules[:3]):  # 最初の3つのみ表示
        print(f"  ルール{i+1}: {rule['path']}")
        print(f"    結論: {rule['conclusion']} (信頼度: {rule['confidence']:.3f})")
    
    print("✅ ファジィ決定木構築テスト完了")

if __name__ == "__main__":
    test_fuzzy_tree_builder()