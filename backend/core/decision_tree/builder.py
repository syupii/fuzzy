# core/decision_tree/builder.py - ファジィ決定木構築

from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
from collections import defaultdict, Counter

from core.decision_tree.tree import FuzzyDecisionTree
from core.decision_tree.node import FuzzyTreeNode, FuzzyInternalNode, FuzzyLeafNode, NodeFactory
from models.schemas import StudentProfile, Laboratory

@dataclass
class BuilderConfig:
    """構築設定"""
    max_depth: int = 8
    min_samples_split: int = 3
    min_samples_leaf: int = 1
    fuzzy_threshold: float = 0.1
    max_features: Optional[int] = None
    criterion: str = "entropy"  # "entropy", "gini", "fuzzy_entropy"
    splitter: str = "best"      # "best", "random"
    random_state: Optional[int] = None

class SplitCriterion(ABC):
    """分岐基準の抽象基底クラス"""
    
    @abstractmethod
    def calculate_impurity(self, y: List[str]) -> float:
        """不純度を計算"""
        pass
    
    @abstractmethod
    def calculate_gain(self, y_parent: List[str], splits: Dict[str, List[str]]) -> float:
        """情報ゲインを計算"""
        pass

class EntropyCriterion(SplitCriterion):
    """エントロピー基準"""
    
    def calculate_impurity(self, y: List[str]) -> float:
        """エントロピーを計算"""
        if not y:
            return 0.0
        
        class_counts = Counter(y)
        total = len(y)
        
        entropy = 0.0
        for count in class_counts.values():
            if count > 0:
                probability = count / total
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def calculate_gain(self, y_parent: List[str], splits: Dict[str, List[str]]) -> float:
        """情報ゲインを計算"""
        parent_entropy = self.calculate_impurity(y_parent)
        
        total_samples = len(y_parent)
        weighted_entropy = 0.0
        
        for split_y in splits.values():
            if len(split_y) > 0:
                weight = len(split_y) / total_samples
                split_entropy = self.calculate_impurity(split_y)
                weighted_entropy += weight * split_entropy
        
        return parent_entropy - weighted_entropy

class GiniCriterion(SplitCriterion):
    """ジニ不純度基準"""
    
    def calculate_impurity(self, y: List[str]) -> float:
        """ジニ不純度を計算"""
        if not y:
            return 0.0
        
        class_counts = Counter(y)
        total = len(y)
        
        gini = 1.0
        for count in class_counts.values():
            probability = count / total
            gini -= probability ** 2
        
        return gini
    
    def calculate_gain(self, y_parent: List[str], splits: Dict[str, List[str]]) -> float:
        """ジニゲインを計算"""
        parent_gini = self.calculate_impurity(y_parent)
        
        total_samples = len(y_parent)
        weighted_gini = 0.0
        
        for split_y in splits.values():
            if len(split_y) > 0:
                weight = len(split_y) / total_samples
                split_gini = self.calculate_impurity(split_y)
                weighted_gini += weight * split_gini
        
        return parent_gini - weighted_gini

class FuzzyEntropyCriterion(SplitCriterion):
    """ファジィエントロピー基準"""
    
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha  # ファジィネス重み
    
    def calculate_impurity(self, y: List[str]) -> float:
        """ファジィエントロピーを計算"""
        if not y:
            return 0.0
        
        class_counts = Counter(y)
        total = len(y)
        
        # 通常のエントロピー
        entropy = 0.0
        for count in class_counts.values():
            if count > 0:
                probability = count / total
                entropy -= probability * np.log2(probability)
        
        # ファジィネス項（クラス分布の偏り）
        max_prob = max(class_counts.values()) / total
        fuzziness = 1.0 - max_prob
        
        return entropy + self.alpha * fuzziness
    
    def calculate_gain(self, y_parent: List[str], splits: Dict[str, List[str]]) -> float:
        """ファジィ情報ゲインを計算"""
        parent_fuzzy_entropy = self.calculate_impurity(y_parent)
        
        total_samples = len(y_parent)
        weighted_fuzzy_entropy = 0.0
        
        for split_y in splits.values():
            if len(split_y) > 0:
                weight = len(split_y) / total_samples
                split_fuzzy_entropy = self.calculate_impurity(split_y)
                weighted_fuzzy_entropy += weight * split_fuzzy_entropy
        
        return parent_fuzzy_entropy - weighted_fuzzy_entropy

class FuzzyTreeBuilder:
    """ファジィ決定木構築クラス"""
    
    def __init__(self, config: BuilderConfig):
        self.config = config
        
        # 分岐基準設定
        if config.criterion == "entropy":
            self.criterion = EntropyCriterion()
        elif config.criterion == "gini":
            self.criterion = GiniCriterion()
        elif config.criterion == "fuzzy_entropy":
            self.criterion = FuzzyEntropyCriterion()
        else:
            raise ValueError(f"未知の分岐基準: {config.criterion}")
        
        # ランダムシード設定
        if config.random_state is not None:
            random.seed(config.random_state)
            np.random.seed(config.random_state)
        
        # 構築統計
        self.build_stats = {
            "nodes_created": 0,
            "splits_evaluated": 0,
            "max_depth_reached": 0
        }
    
    def build_tree(self, X: List[Dict[str, Any]], y: List[str]) -> FuzzyDecisionTree:
        """ファジィ決定木を構築"""
        
        print(f"🌳 ファジィ決定木構築開始")
        print(f"   サンプル数: {len(X)}")
        print(f"   基準: {self.config.criterion}")
        print(f"   最大深度: {self.config.max_depth}")
        
        # 決定木オブジェクト作成
        tree = FuzzyDecisionTree(
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            fuzzy_threshold=self.config.fuzzy_threshold
        )
        
        # 基本情報設定
        tree.feature_names = list(X[0].keys()) if X else []
        tree.class_names = list(set(y))
        tree.n_features = len(tree.feature_names)
        tree.n_classes = len(tree.class_names)
        tree.training_samples_count = len(X)
        
        # ルートノード構築
        tree.root = self._build_node(X, y, depth=0, node_id="root")
        
        # 統計情報
        print(f"✅ 構築完了")
        print(f"   作成ノード数: {self.build_stats['nodes_created']}")
        print(f"   評価分岐数: {self.build_stats['splits_evaluated']}")
        print(f"   実際の深度: {self.build_stats['max_depth_reached']}")
        
        return tree
    
    def _build_node(self, X: List[Dict[str, Any]], y: List[str], 
                   depth: int, node_id: str) -> FuzzyTreeNode:
        """ノードを再帰的に構築"""
        
        self.build_stats["nodes_created"] += 1
        self.build_stats["max_depth_reached"] = max(self.build_stats["max_depth_reached"], depth)
        
        # 停止条件チェック
        if self._should_stop(X, y, depth):
            return self._create_leaf(X, y, depth, node_id)
        
        # 最適分岐探索
        best_split = self._find_best_split(X, y)
        
        if best_split is None:
            return self._create_leaf(X, y, depth, node_id)
        
        # 内部ノード作成
        internal_node = NodeFactory.create_internal_node(
            node_id=node_id,
            feature=best_split["feature"],
            threshold=best_split["threshold"],
            linguistic_value=best_split["linguistic_value"],
            depth=depth
        )
        
        internal_node.samples_count = len(X)
        internal_node.purity = self._calculate_purity(y)
        
        # データ分割
        splits = self._split_data(X, y, best_split)
        
        # 子ノード構築
        for branch_name, (X_split, y_split) in splits.items():
            if len(X_split) >= self.config.min_samples_leaf:
                child_id = f"{node_id}_{branch_name}"
                child_node = self._build_node(X_split, y_split, depth + 1, child_id)
                internal_node.add_child(branch_name, child_node)
        
        # 子ノードが作成されなかった場合は葉ノードにする
        if not internal_node.children:
            return self._create_leaf(X, y, depth, node_id)
        
        return internal_node
    
    def _should_stop(self, X: List[Dict[str, Any]], y: List[str], depth: int) -> bool:
        """停止条件判定"""
        
        # 深度制限
        if depth >= self.config.max_depth:
            return True
        
        # サンプル数制限
        if len(X) < self.config.min_samples_split:
            return True
        
        # 純度チェック
        if len(set(y)) <= 1:
            return True
        
        # 不純度チェック
        impurity = self.criterion.calculate_impurity(y)
        if impurity < 1e-6:  # ほぼ純粋
            return True
        
        return False
    
    def _find_best_split(self, X: List[Dict[str, Any]], y: List[str]) -> Optional[Dict[str, Any]]:
        """最適分岐探索"""
        
        best_split = None
        best_gain = -1.0
        
        # 使用する特徴量を選択
        features_to_try = self._select_features(list(X[0].keys()) if X else [])
        
        for feature in features_to_try:
            feature_values = [sample.get(feature, 0) for sample in X]
            
            if not any(isinstance(v, (int, float)) for v in feature_values):
                continue  # 数値でない特徴量はスキップ
            
            # 閾値候補生成
            thresholds = self._generate_split_thresholds(feature_values)
            
            for threshold in thresholds:
                for linguistic_value in ["low", "medium", "high"]:
                    
                    self.build_stats["splits_evaluated"] += 1
                    
                    # 分岐評価
                    gain = self._evaluate_split(X, y, feature, threshold, linguistic_value)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_split = {
                            "feature": feature,
                            "threshold": threshold,
                            "linguistic_value": linguistic_value,
                            "gain": gain
                        }
        
        # 最小ゲイン閾値チェック
        if best_gain < 1e-6:
            return None
        
        return best_split
    
    def _select_features(self, all_features: List[str]) -> List[str]:
        """特徴量選択"""
        
        if self.config.max_features is None:
            return all_features
        
        n_features = min(self.config.max_features, len(all_features))
        
        if self.config.splitter == "random":
            return random.sample(all_features, n_features)
        else:
            return all_features[:n_features]
    
    def _generate_split_thresholds(self, values: List[float]) -> List[float]:
        """分岐閾値候補生成"""
        
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        
        if len(numeric_values) < 2:
            return [np.mean(numeric_values)] if numeric_values else [0.0]
        
        sorted_values = sorted(set(numeric_values))
        
        if len(sorted_values) < 2:
            return [sorted_values[0]]
        
        thresholds = []
        
        # 分位点ベース
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            threshold = np.percentile(sorted_values, p)
            thresholds.append(threshold)
        
        # 隣接値の中点
        if self.config.splitter == "best":
            for i in range(min(10, len(sorted_values) - 1)):  # 最大10個
                midpoint = (sorted_values[i] + sorted_values[i + 1]) / 2
                thresholds.append(midpoint)
        
        return list(set(thresholds))
    
    def _evaluate_split(self, X: List[Dict[str, Any]], y: List[str],
                       feature: str, threshold: float, linguistic_value: str) -> float:
        """分岐評価"""
        
        # データ分割
        splits = self._split_data(X, y, {
            "feature": feature,
            "threshold": threshold,
            "linguistic_value": linguistic_value
        })
        
        # 分岐されたデータのラベル部分のみ取得
        y_splits = {branch: y_split for branch, (_, y_split) in splits.items()}
        
        # 基本的な情報ゲイン
        base_gain = self.criterion.calculate_gain(y, y_splits)
        
        # ファジィ補正
        fuzzy_penalty = self._calculate_fuzzy_penalty(X, feature, threshold, linguistic_value)
        
        # バランス補正（極端に偏った分割を避ける）
        balance_bonus = self._calculate_balance_bonus(y_splits)
        
        total_gain = base_gain - fuzzy_penalty + balance_bonus
        
        return max(0.0, total_gain)
    
    def _split_data(self, X: List[Dict[str, Any]], y: List[str],
                   split_info: Dict[str, Any]) -> Dict[str, Tuple[List[Dict], List[str]]]:
        """ファジィ分割でデータを分割"""
        
        feature = split_info["feature"]
        threshold = split_info["threshold"]
        linguistic_value = split_info["linguistic_value"]
        
        splits = defaultdict(lambda: ([], []))
        
        for i, sample in enumerate(X):
            feature_value = sample.get(feature, 0)
            
            # 帰属度計算
            memberships = self._calculate_memberships(feature_value, threshold, linguistic_value)
            
            # 最大帰属度の分岐に割り当て
            best_branch = max(memberships.keys(), key=lambda k: memberships[k])
            
            if memberships[best_branch] > self.config.fuzzy_threshold:
                splits[best_branch][0].append(sample)
                splits[best_branch][1].append(y[i])
        
        return dict(splits)
    
    def _calculate_memberships(self, value: float, threshold: float, 
                             linguistic_value: str) -> Dict[str, float]:
        """帰属度計算"""
        
        if linguistic_value == "low":
            if value <= threshold:
                return {"left": 1.0, "right": 0.0}
            else:
                distance = value - threshold
                membership = max(0, 1.0 - distance / 3.0)
                return {"left": membership, "right": 1.0 - membership}
                
        elif linguistic_value == "high":
            if value >= threshold:
                return {"left": 0.0, "right": 1.0}
            else:
                distance = threshold - value
                membership = max(0, 1.0 - distance / 3.0)
                return {"left": 1.0 - membership, "right": membership}
                
        else:  # medium
            distance = abs(value - threshold)
            if distance <= 1.5:
                center_membership = 1.0 - distance / 1.5
                return {"center": center_membership, "others": 1.0 - center_membership}
            else:
                return {"center": 0.0, "others": 1.0}
    
    def _calculate_fuzzy_penalty(self, X: List[Dict[str, Any]], feature: str,
                               threshold: float, linguistic_value: str) -> float:
        """ファジィネスペナルティ計算"""
        
        feature_values = [sample.get(feature, 0) for sample in X]
        numeric_values = [v for v in feature_values if isinstance(v, (int, float))]
        
        if not numeric_values:
            return 0.0
        
        # 境界付近のサンプル割合
        boundary_samples = 0
        boundary_width = 2.0
        
        for value in numeric_values:
            if abs(value - threshold) <= boundary_width:
                boundary_samples += 1
        
        boundary_ratio = boundary_samples / len(numeric_values)
        
        return boundary_ratio * 0.05  # ペナルティ係数
    
    def _calculate_balance_bonus(self, splits: Dict[str, List[str]]) -> float:
        """分岐バランスボーナス計算"""
        
        if not splits:
            return 0.0
        
        split_sizes = [len(split) for split in splits.values()]
        total_size = sum(split_sizes)
        
        if total_size == 0:
            return 0.0
        
        # 各分岐の割合
        proportions = [size / total_size for size in split_sizes]
        
        # バランス度（エントロピーベース）
        balance = -sum(p * np.log(p + 1e-10) for p in proportions if p > 0)
        max_balance = np.log(len(split_sizes))
        
        normalized_balance = balance / max_balance if max_balance > 0 else 0
        
        return normalized_balance * 0.02  # ボーナス係数
    
    def _create_leaf(self, X: List[Dict[str, Any]], y: List[str],
                    depth: int, node_id: str) -> FuzzyLeafNode:
        """葉ノード作成"""
        
        # クラス分布計算
        class_counts = Counter(y)
        total_samples = len(y)
        
        class_distribution = {
            class_name: count / total_samples
            for class_name, count in class_counts.items()
        }
        
        # 信頼度（最多クラスの割合）
        confidence = max(class_distribution.values()) if class_distribution else 0.0
        
        # 葉ノード作成
        leaf_node = NodeFactory.create_leaf_node(
            node_id=node_id,
            class_distribution=class_distribution,
            depth=depth,
            confidence=confidence
        )
        
        leaf_node.samples_count = total_samples
        
        return leaf_node
    
    def _calculate_purity(self, y: List[str]) -> float:
        """純度計算"""
        if not y:
            return 0.0
        
        class_counts = Counter(y)
        return max(class_counts.values()) / len(y)
    
    def get_build_summary(self) -> Dict[str, Any]:
        """構築サマリー取得"""
        return {
            "config": {
                "max_depth": self.config.max_depth,
                "min_samples_split": self.config.min_samples_split,
                "min_samples_leaf": self.config.min_samples_leaf,
                "criterion": self.config.criterion,
                "splitter": self.config.splitter
            },
            "statistics": self.build_stats.copy()
        }

class RandomForestBuilder:
    """ランダムフォレスト構築（複数ファジィ決定木）"""
    
    def __init__(self, n_estimators: int = 10, base_config: BuilderConfig = None):
        self.n_estimators = n_estimators
        self.base_config = base_config or BuilderConfig()
        self.trees: List[FuzzyDecisionTree] = []
        self.feature_importances: Dict[str, float] = {}
    
    def build_forest(self, X: List[Dict[str, Any]], y: List[str]) -> List[FuzzyDecisionTree]:
        """ランダムフォレスト構築"""
        
        print(f"🌲 ランダムフォレスト構築開始 ({self.n_estimators}木)")
        
        self.trees = []
        
        for i in range(self.n_estimators):
            print(f"   木 {i+1}/{self.n_estimators} 構築中...")
            
            # ブートストラップサンプリング
            X_bootstrap, y_bootstrap = self._bootstrap_sample(X, y)
            
            # 構築設定のランダム化
            tree_config = self._randomize_config(self.base_config)
            
            # 決定木構築
            builder = FuzzyTreeBuilder(tree_config)
            tree = builder.build_tree(X_bootstrap, y_bootstrap)
            
            self.trees.append(tree)
        
        # 特徴量重要度の集約
        self._aggregate_feature_importances()
        
        print(f"✅ ランダムフォレスト構築完了")
        
        return self.trees
    
    def _bootstrap_sample(self, X: List[Dict[str, Any]], y: List[str]) -> Tuple[List[Dict], List[str]]:
        """ブートストラップサンプリング"""
        
        n_samples = len(X)
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        X_bootstrap = [X[i] for i in indices]
        y_bootstrap = [y[i] for i in indices]
        
        return X_bootstrap, y_bootstrap
    
    def _randomize_config(self, base_config: BuilderConfig) -> BuilderConfig:
        """構築設定のランダム化"""
        
        config = BuilderConfig(
            max_depth=base_config.max_depth + random.randint(-2, 2),
            min_samples_split=max(2, base_config.min_samples_split + random.randint(-1, 1)),
            min_samples_leaf=max(1, base_config.min_samples_leaf),
            fuzzy_threshold=base_config.fuzzy_threshold + random.uniform(-0.05, 0.05),
            max_features=base_config.max_features,
            criterion=random.choice(["entropy", "gini", "fuzzy_entropy"]),
            splitter="random",
            random_state=random.randint(0, 10000)
        )
        
        # 範囲制限
        config.max_depth = max(1, min(15, config.max_depth))
        config.fuzzy_threshold = max(0.01, min(0.5, config.fuzzy_threshold))
        
        return config
    
    def _aggregate_feature_importances(self) -> None:
        """特徴量重要度の集約"""
        
        importance_sum = defaultdict(float)
        
        for tree in self.trees:
            for feature, importance in tree.feature_importances.items():
                importance_sum[feature] += importance
        
        # 平均化
        total_trees = len(self.trees)
        self.feature_importances = {
            feature: importance / total_trees
            for feature, importance in importance_sum.items()
        }
    
    def predict_ensemble(self, X: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """アンサンブル予測"""
        
        if not self.trees:
            raise ValueError("フォレストが構築されていません")
        
        results = []
        
        for sample in X:
            # 各木の予測を集約
            class_votes = defaultdict(float)
            
            for tree in self.trees:
                prediction = tree.predict(sample)
                for class_name, probability in prediction.class_probabilities.items():
                    class_votes[class_name] += probability
            
            # 平均化
            total_votes = sum(class_votes.values())
            ensemble_prediction = {
                class_name: votes / total_votes
                for class_name, votes in class_votes.items()
            } if total_votes > 0 else {}
            
            results.append(ensemble_prediction)
        
        return results