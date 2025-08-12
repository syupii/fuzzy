import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json


@dataclass
class FuzzyMembershipFunction:
    """ファジィメンバーシップ関数"""
    function_type: str  # 'triangular', 'trapezoidal', 'gaussian'
    parameters: List[float]
    label: str

    def membership(self, value: float) -> float:
        """所属度計算"""
        if self.function_type == 'triangular':
            a, b, c = self.parameters
            if value <= a or value >= c:
                return 0.0
            elif a < value <= b:
                return (value - a) / (b - a) if b != a else 1.0
            else:
                return (c - value) / (c - b) if c != b else 1.0

        elif self.function_type == 'gaussian':
            mean, sigma = self.parameters
            return math.exp(-0.5 * ((value - mean) / sigma) ** 2)

        elif self.function_type == 'trapezoidal':
            a, b, c, d = self.parameters
            if value <= a or value >= d:
                return 0.0
            elif a < value <= b:
                return (value - a) / (b - a) if b != a else 1.0
            elif b < value <= c:
                return 1.0
            else:
                return (d - value) / (d - c) if d != c else 1.0

        return 0.0


class FuzzyDecisionNode:
    """ファジィ決定ノード"""

    def __init__(self, node_id: int = 0):
        self.node_id = node_id
        self.feature_index: Optional[int] = None
        self.feature_name: Optional[str] = None
        self.membership_functions: Dict[str, FuzzyMembershipFunction] = {}
        self.children: Dict[str, 'FuzzyDecisionNode'] = {}
        self.is_leaf: bool = False
        self.leaf_value: Optional[float] = None
        self.samples_count: int = 0
        self.depth: int = 0

        # 説明用情報
        self.decision_path: List[str] = []
        self.confidence_score: float = 0.0

    def add_membership_function(self, label: str, func: FuzzyMembershipFunction):
        """メンバーシップ関数を追加"""
        self.membership_functions[label] = func

    def predict(self, input_vector: np.ndarray) -> float:
        """予測実行"""
        if self.is_leaf:
            return self.leaf_value

        if self.feature_index is None:
            return 0.5  # デフォルト値

        feature_value = input_vector[self.feature_index]

        # 各言語変数の所属度計算
        memberships = {}
        for label, mf in self.membership_functions.items():
            memberships[label] = mf.membership(feature_value)

        # 重み付き予測値計算
        total_prediction = 0.0
        total_weight = 0.0

        for label, membership in memberships.items():
            if membership > 0.01 and label in self.children:
                child_prediction = self.children[label].predict(input_vector)
                total_prediction += membership * child_prediction
                total_weight += membership

        if total_weight == 0:
            return 0.5  # フォールバック

        return total_prediction / total_weight

    def get_prediction_path(self, input_vector: np.ndarray, path: List[str] = None) -> List[str]:
        """予測経路を取得（説明用）"""
        if path is None:
            path = []

        if self.is_leaf:
            path.append(f"→ 結論: {self.leaf_value:.2f}")
            return path

        feature_value = input_vector[self.feature_index]
        path.append(f"{self.feature_name}={feature_value:.1f}")

        # 最も高い所属度の子ノードを選択
        best_label = None
        best_membership = 0

        for label, mf in self.membership_functions.items():
            membership = mf.membership(feature_value)
            if membership > best_membership:
                best_membership = membership
                best_label = label

        if best_label and best_label in self.children:
            path.append(f"→ {best_label}(所属度:{best_membership:.2f})")
            return self.children[best_label].get_prediction_path(input_vector, path)

        return path

    def calculate_confidence(self, input_vector: np.ndarray) -> float:
        """予測信頼度計算"""
        if self.is_leaf:
            return 0.9  # リーフノードは高い信頼度

        feature_value = input_vector[self.feature_index]
        memberships = [mf.membership(feature_value)
                       for mf in self.membership_functions.values()]

        # 最大所属度が高いほど信頼度が高い
        max_membership = max(memberships) if memberships else 0

        # 所属度の分散が小さいほど曖昧性が高い（信頼度低下）
        if len(memberships) > 1:
            variance = np.var(memberships)
            confidence = max_membership * (1 + variance)  # 分散が大きいほど信頼度向上
        else:
            confidence = max_membership

        return min(1.0, confidence)


class FuzzyDecisionTree:
    """ファジィ決定木メインクラス"""

    def __init__(self, max_depth: int = 5, min_samples_split: int = 10, min_samples_leaf: int = 5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root: Optional[FuzzyDecisionNode] = None
        self.feature_names: List[str] = []
        self.node_counter = 0

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """ファジィ決定木の学習"""
        self.feature_names = feature_names or [
            f"feature_{i}" for i in range(X.shape[1])]
        self.node_counter = 0
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> FuzzyDecisionNode:
        """再帰的な木構築"""
        node = FuzzyDecisionNode(self.node_counter)
        self.node_counter += 1
        node.samples_count = len(y)
        node.depth = depth

        # 終了条件チェック
        if (depth >= self.max_depth or
            len(y) < self.min_samples_split or
                self._is_pure_enough(y)):

            node.is_leaf = True
            node.leaf_value = np.mean(y)
            return node

        # 最適特徴選択
        best_feature_idx, best_gain, best_splits = self._find_best_split(X, y)

        if best_feature_idx is None or best_gain <= 0:
            node.is_leaf = True
            node.leaf_value = np.mean(y)
            return node

        # ノード設定
        node.feature_index = best_feature_idx
        node.feature_name = self.feature_names[best_feature_idx]

        # メンバーシップ関数設定
        for label, mf in best_splits['membership_functions'].items():
            node.add_membership_function(label, mf)

        # 子ノード構築
        for label, (subset_X, subset_y) in best_splits['data_splits'].items():
            if len(subset_y) >= self.min_samples_leaf:
                child = self._build_tree(subset_X, subset_y, depth + 1)
                node.children[label] = child
            else:
                # サンプル数が少ない場合はリーフノード
                leaf = FuzzyDecisionNode(self.node_counter)
                self.node_counter += 1
                leaf.is_leaf = True
                leaf.leaf_value = np.mean(subset_y) if len(
                    subset_y) > 0 else np.mean(y)
                node.children[label] = leaf

        return node

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], float, Dict]:
        """最適分岐の探索"""
        best_gain = 0
        best_feature_idx = None
        best_splits = {}

        n_features = X.shape[1]

        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            gain, splits = self._evaluate_fuzzy_split(feature_values, y)

            if gain > best_gain:
                best_gain = gain
                best_feature_idx = feature_idx
                best_splits = splits

        return best_feature_idx, best_gain, best_splits

    def _evaluate_fuzzy_split(self, feature_values: np.ndarray, y: np.ndarray) -> Tuple[float, Dict]:
        """ファジィ分岐の評価"""
        min_val, max_val = feature_values.min(), feature_values.max()

        if min_val == max_val:
            return 0, {}

        # 3つの言語変数でのメンバーシップ関数を生成
        range_span = max_val - min_val

        # 三角関数のパラメータ最適化（簡易版）
        best_gain = 0
        best_config = {}

        # いくつかのしきい値候補で試行
        for split1 in np.linspace(min_val + range_span*0.2, min_val + range_span*0.4, 3):
            for split2 in np.linspace(min_val + range_span*0.6, min_val + range_span*0.8, 3):

                # メンバーシップ関数定義
                mf_low = FuzzyMembershipFunction(
                    'triangular', [min_val, min_val, split1], 'low')
                mf_med = FuzzyMembershipFunction(
                    'triangular', [min_val, split1, split2], 'medium')
                mf_high = FuzzyMembershipFunction(
                    'triangular', [split1, max_val, max_val], 'high')

                membership_functions = {
                    'low': mf_low,
                    'medium': mf_med,
                    'high': mf_high
                }

                # データ分割
                data_splits = self._split_data_fuzzy(
                    feature_values, y, membership_functions)

                # 情報ゲイン計算
                gain = self._calculate_fuzzy_information_gain(y, data_splits)

                if gain > best_gain:
                    best_gain = gain
                    best_config = {
                        'membership_functions': membership_functions,
                        'data_splits': data_splits
                    }

        return best_gain, best_config

    def _split_data_fuzzy(self, feature_values: np.ndarray, y: np.ndarray,
                          membership_functions: Dict[str, FuzzyMembershipFunction]) -> Dict[str, Tuple]:
        """ファジィ分割によるデータ分割"""
        splits = {}

        for label, mf in membership_functions.items():
            # 所属度計算
            memberships = np.array([mf.membership(val)
                                   for val in feature_values])

            # 所属度が閾値以上のサンプルを選択
            threshold = 0.1
            selected_indices = memberships > threshold

            if np.any(selected_indices):
                selected_X = feature_values[selected_indices]
                selected_y = y[selected_indices]

                # 所属度による重み付きサンプリング（簡易版）
                weights = memberships[selected_indices]

                splits[label] = (selected_X.reshape(-1, 1), selected_y)
            else:
                splits[label] = (np.array([]).reshape(0, 1), np.array([]))

        return splits

    def _calculate_fuzzy_information_gain(self, y: np.ndarray, data_splits: Dict) -> float:
        """ファジィ情報ゲイン計算"""
        # 親ノードのエントロピー
        parent_entropy = self._calculate_entropy(y)

        # 分割後の重み付きエントロピー
        total_samples = len(y)
        weighted_entropy = 0

        for label, (split_X, split_y) in data_splits.items():
            if len(split_y) > 0:
                split_entropy = self._calculate_entropy(split_y)
                weight = len(split_y) / total_samples
                weighted_entropy += weight * split_entropy

        return parent_entropy - weighted_entropy

    def _calculate_entropy(self, y: np.ndarray) -> float:
        """エントロピー計算（回帰用）"""
        if len(y) == 0:
            return 0

        # 回帰の場合は分散を使用
        variance = np.var(y)
        return variance  # より小さい方が良い

    def _is_pure_enough(self, y: np.ndarray, threshold: float = 0.1) -> bool:
        """十分に純粋か判定"""
        if len(y) <= 1:
            return True

        variance = np.var(y)
        return variance < threshold

    def predict(self, input_vector: np.ndarray) -> float:
        """予測"""
        if self.root is None:
            raise ValueError("Tree not trained")

        return self.root.predict(input_vector)

    def get_prediction_path_depth(self, input_vector: np.ndarray) -> int:
        """予測で使用した木の深さ"""
        if self.root is None:
            return 0

        return self._get_path_depth(self.root, input_vector)

    def _get_path_depth(self, node: FuzzyDecisionNode, input_vector: np.ndarray) -> int:
        """再帰的に経路深さ計算"""
        if node.is_leaf:
            return node.depth

        feature_value = input_vector[node.feature_index]

        # 最も高い所属度の子ノードを選択
        best_child = None
        best_membership = 0

        for label, mf in node.membership_functions.items():
            membership = mf.membership(feature_value)
            if membership > best_membership and label in node.children:
                best_membership = membership
                best_child = node.children[label]

        if best_child:
            return self._get_path_depth(best_child, input_vector)

        return node.depth

    def calculate_confidence(self, input_vector: np.ndarray) -> float:
        """予測信頼度計算"""
        if self.root is None:
            return 0.0

        return self.root.calculate_confidence(input_vector)
