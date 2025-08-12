# backend/advanced_nodes.py
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid


class MembershipType(Enum):
    """メンバーシップ関数の種類"""
    TRIANGULAR = "triangular"
    TRAPEZOIDAL = "trapezoidal"
    GAUSSIAN = "gaussian"
    SIGMOID = "sigmoid"
    BELL = "bell"


@dataclass
class MembershipFunction:
    """高度なメンバーシップ関数"""
    name: str
    type: MembershipType
    parameters: List[float]
    weight: float = 1.0

    def membership(self, value: float) -> float:
        """所属度計算"""
        try:
            if self.type == MembershipType.TRIANGULAR:
                return self._triangular_membership(value)
            elif self.type == MembershipType.TRAPEZOIDAL:
                return self._trapezoidal_membership(value)
            elif self.type == MembershipType.GAUSSIAN:
                return self._gaussian_membership(value)
            elif self.type == MembershipType.SIGMOID:
                return self._sigmoid_membership(value)
            elif self.type == MembershipType.BELL:
                return self._bell_membership(value)
            else:
                return 0.0
        except:
            return 0.0

    def _triangular_membership(self, x: float) -> float:
        """三角メンバーシップ関数"""
        if len(self.parameters) < 3:
            return 0.0
        a, b, c = self.parameters[0], self.parameters[1], self.parameters[2]

        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        else:
            return (c - x) / (c - b) if c != b else 1.0

    def _trapezoidal_membership(self, x: float) -> float:
        """台形メンバーシップ関数"""
        if len(self.parameters) < 4:
            return 0.0
        a, b, c, d = self.parameters[0], self.parameters[1], self.parameters[2], self.parameters[3]

        if x <= a or x >= d:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a) if b != a else 1.0
        elif b < x <= c:
            return 1.0
        else:
            return (d - x) / (d - c) if d != c else 1.0

    def _gaussian_membership(self, x: float) -> float:
        """ガウシアンメンバーシップ関数"""
        if len(self.parameters) < 2:
            return 0.0
        mean, sigma = self.parameters[0], self.parameters[1]

        if sigma <= 0:
            return 1.0 if x == mean else 0.0

        return math.exp(-0.5 * ((x - mean) / sigma) ** 2)

    def _sigmoid_membership(self, x: float) -> float:
        """シグモイドメンバーシップ関数"""
        if len(self.parameters) < 2:
            return 0.0
        a, c = self.parameters[0], self.parameters[1]

        try:
            return 1.0 / (1.0 + math.exp(-a * (x - c)))
        except OverflowError:
            return 0.0 if a * (x - c) < 0 else 1.0

    def _bell_membership(self, x: float) -> float:
        """ベル型メンバーシップ関数"""
        if len(self.parameters) < 3:
            return 0.0
        a, b, c = self.parameters[0], self.parameters[1], self.parameters[2]

        if a <= 0:
            return 0.0

        try:
            return 1.0 / (1.0 + abs((x - c) / a) ** (2 * b))
        except (OverflowError, ZeroDivisionError):
            return 0.0


@dataclass
class FitnessComponents:
    """多目的適応度評価"""
    accuracy: float = 0.0           # 予測精度
    simplicity: float = 0.0         # モデル簡潔性
    interpretability: float = 0.0   # 解釈可能性
    generalization: float = 0.0     # 汎化性能
    validity: float = 0.0           # ルール妥当性
    overall: float = 0.0            # 総合適応度

    def compute_overall(self, weights: Dict[str, float] = None) -> float:
        """総合適応度計算"""
        if weights is None:
            weights = {
                'accuracy': 0.35,
                'simplicity': 0.15,
                'interpretability': 0.20,
                'generalization': 0.20,
                'validity': 0.10
            }

        self.overall = (
            weights.get('accuracy', 0.35) * self.accuracy +
            weights.get('simplicity', 0.15) * self.simplicity +
            weights.get('interpretability', 0.20) * self.interpretability +
            weights.get('generalization', 0.20) * self.generalization +
            weights.get('validity', 0.10) * self.validity
        )

        return self.overall


@dataclass
class DecisionStep:
    """決定ステップの詳細記録"""
    node_id: str
    feature_name: str
    feature_value: float
    membership_evaluations: Dict[str, Dict] = field(default_factory=dict)
    chosen_branch: str = ""
    confidence: float = 0.0
    reasoning: str = ""


class AdvancedFuzzyDecisionNode:
    """高度なファジィ決定ノード"""

    def __init__(self,
                 node_id: str = None,
                 feature_name: str = None,
                 membership_functions: Dict[str, MembershipFunction] = None,
                 children: Dict[str, 'AdvancedFuzzyDecisionNode'] = None,
                 leaf_value: float = None,
                 confidence_threshold: float = 0.1,
                 explanation_enabled: bool = True):

        self.node_id = node_id or str(uuid.uuid4())[:8]
        self.feature_name = feature_name
        self.membership_functions = membership_functions or {}
        self.children = children or {}
        self.leaf_value = leaf_value
        self.is_leaf = leaf_value is not None
        self.confidence_threshold = confidence_threshold
        self.explanation_enabled = explanation_enabled

        # 統計情報
        self.prediction_count = 0
        self.accuracy_history = []

    def add_membership_function(self, label: str, mf: MembershipFunction):
        """メンバーシップ関数追加"""
        self.membership_functions[label] = mf

    def add_child(self, label: str, child_node: 'AdvancedFuzzyDecisionNode'):
        """子ノード追加"""
        self.children[label] = child_node

    def evaluate_memberships(self, feature_value: float) -> Dict[str, Dict]:
        """特徴値に対するメンバーシップ評価"""
        evaluations = {}

        for label, mf in self.membership_functions.items():
            membership_value = mf.membership(feature_value)
            evaluations[label] = {
                'membership_value': membership_value,
                'weight': mf.weight,
                'weighted_membership': membership_value * mf.weight,
                'function_type': mf.type.value,
                'parameters': mf.parameters.copy()
            }

        return evaluations

    def predict(self, sample: Dict[str, float]) -> float:
        """基本予測"""
        if self.is_leaf:
            return self.leaf_value

        if self.feature_name not in sample:
            return 0.5  # デフォルト値

        feature_value = sample[self.feature_name]
        memberships = self.evaluate_memberships(feature_value)

        # 重み付き予測
        total_prediction = 0.0
        total_weight = 0.0

        for label, evaluation in memberships.items():
            weighted_membership = evaluation['weighted_membership']

            if weighted_membership > self.confidence_threshold and label in self.children:
                child_prediction = self.children[label].predict(sample)
                total_prediction += weighted_membership * child_prediction
                total_weight += weighted_membership

        # 予測統計更新
        self.prediction_count += 1

        return total_prediction / total_weight if total_weight > 0 else 0.5

    def predict_with_explanation(self, sample: Dict[str, float],
                                 feature_names: List[str] = None) -> Tuple[float, Dict]:
        """説明付き予測"""

        explanation = {
            'decision_steps': [],
            'feature_importance': {},
            'confidence': 0.0,
            'rationale': "",
            'node_path': []
        }

        prediction = self._predict_recursive(
            sample, explanation, feature_names or [])

        # 最終的な信頼度計算
        explanation['confidence'] = self._calculate_overall_confidence(
            explanation['decision_steps'])

        # 根拠文生成
        explanation['rationale'] = self._generate_rationale(explanation)

        return prediction, explanation

    def _predict_recursive(self, sample: Dict[str, float],
                           explanation: Dict, feature_names: List[str]) -> float:
        """再帰的予測（説明記録付き）"""

        explanation['node_path'].append(self.node_id)

        if self.is_leaf:
            return self.leaf_value

        if self.feature_name not in sample:
            return 0.5

        feature_value = sample[self.feature_name]
        memberships = self.evaluate_memberships(feature_value)

        # 決定ステップ記録
        step = DecisionStep(
            node_id=self.node_id,
            feature_name=self.feature_name,
            feature_value=feature_value,
            membership_evaluations=memberships
        )

        # 最高メンバーシップの分岐選択
        best_label = ""
        best_weighted_membership = 0.0

        for label, evaluation in memberships.items():
            weighted_membership = evaluation['weighted_membership']
            if weighted_membership > best_weighted_membership and label in self.children:
                best_weighted_membership = weighted_membership
                best_label = label

        step.chosen_branch = best_label
        step.confidence = best_weighted_membership

        # 重み付き予測計算
        total_prediction = 0.0
        total_weight = 0.0

        for label, evaluation in memberships.items():
            weighted_membership = evaluation['weighted_membership']

            if weighted_membership > self.confidence_threshold and label in self.children:
                child_prediction = self.children[label]._predict_recursive(
                    sample, explanation, feature_names
                )
                total_prediction += weighted_membership * child_prediction
                total_weight += weighted_membership

        final_prediction = total_prediction / total_weight if total_weight > 0 else 0.5

        step.reasoning = f"特徴'{self.feature_name}'={feature_value:.2f}での分岐選択: {best_label} (信頼度: {best_weighted_membership:.3f})"

        explanation['decision_steps'].append(step.__dict__)

        # 特徴重要度更新
        if self.feature_name in feature_names:
            if self.feature_name not in explanation['feature_importance']:
                explanation['feature_importance'][self.feature_name] = 0.0
            explanation['feature_importance'][self.feature_name] += best_weighted_membership

        return final_prediction

    def _calculate_overall_confidence(self, decision_steps: List[Dict]) -> float:
        """全体信頼度計算"""
        if not decision_steps:
            return 0.0

        confidences = [step.get('confidence', 0.0) for step in decision_steps]

        # 幾何平均による信頼度計算
        if len(confidences) == 1:
            return confidences[0]

        product = 1.0
        for conf in confidences:
            product *= max(conf, 0.001)  # ゼロ回避

        return product ** (1.0 / len(confidences))

    def _generate_rationale(self, explanation: Dict) -> str:
        """根拠文生成"""
        decision_steps = explanation['decision_steps']
        confidence = explanation['confidence']

        if not decision_steps:
            return "決定プロセスが記録されていません。"

        rationale_parts = []

        # 全体評価
        if confidence >= 0.8:
            rationale_parts.append("高い信頼度で予測を実行しました。")
        elif confidence >= 0.6:
            rationale_parts.append("中程度の信頼度で予測を実行しました。")
        else:
            rationale_parts.append("低い信頼度での予測のため、慎重に解釈してください。")

        # 主要な決定要因
        feature_importance = explanation['feature_importance']
        if feature_importance:
            sorted_features = sorted(feature_importance.items(),
                                     key=lambda x: x[1], reverse=True)
            top_feature = sorted_features[0]
            rationale_parts.append(
                f"最も重要な決定要因は'{top_feature[0]}'でした（重要度: {top_feature[1]:.3f}）。"
            )

        # 決定パスの要約
        key_steps = [step for step in decision_steps if step.get(
            'confidence', 0) > 0.5]
        if key_steps:
            rationale_parts.append(f"主要な決定ステップを{len(key_steps)}段階で実行しました。")

        return " ".join(rationale_parts)

    def get_tree_structure(self) -> Dict:
        """木構造の取得"""
        structure = {
            'node_id': self.node_id,
            'feature_name': self.feature_name,
            'is_leaf': self.is_leaf,
            'leaf_value': self.leaf_value,
            'membership_functions': {},
            'children': {},
            'statistics': {
                'prediction_count': self.prediction_count,
                'avg_accuracy': np.mean(self.accuracy_history) if self.accuracy_history else 0.0
            }
        }

        # メンバーシップ関数情報
        for label, mf in self.membership_functions.items():
            structure['membership_functions'][label] = {
                'name': mf.name,
                'type': mf.type.value,
                'parameters': mf.parameters.copy(),
                'weight': mf.weight
            }

        # 子ノード再帰
        for label, child in self.children.items():
            structure['children'][label] = child.get_tree_structure()

        return structure

    def calculate_complexity(self) -> int:
        """木の複雑度計算"""
        if self.is_leaf:
            return 1

        complexity = 1 + len(self.membership_functions)
        for child in self.children.values():
            complexity += child.calculate_complexity()

        return complexity

    def calculate_depth(self) -> int:
        """木の深さ計算"""
        if self.is_leaf:
            return 1

        if not self.children:
            return 1

        return 1 + max(child.calculate_depth() for child in self.children.values())

    def prune_weak_branches(self, min_confidence: float = 0.05) -> bool:
        """弱い分岐の剪定"""
        if self.is_leaf:
            return False

        pruned = False

        # 弱いメンバーシップ関数を削除
        weak_labels = []
        for label, mf in self.membership_functions.items():
            if mf.weight < min_confidence:
                weak_labels.append(label)

        for label in weak_labels:
            del self.membership_functions[label]
            if label in self.children:
                del self.children[label]
            pruned = True

        # 子ノードの剪定
        for child in self.children.values():
            if child.prune_weak_branches(min_confidence):
                pruned = True

        return pruned

    def update_accuracy(self, actual_value: float, predicted_value: float):
        """精度履歴更新"""
        accuracy = 1.0 - abs(actual_value - predicted_value)
        self.accuracy_history.append(max(0.0, accuracy))

        # 履歴サイズ制限
        if len(self.accuracy_history) > 100:
            self.accuracy_history = self.accuracy_history[-100:]


class TreeVisualizationHelper:
    """決定木可視化ヘルパー"""

    @staticmethod
    def generate_dot_notation(root: AdvancedFuzzyDecisionNode) -> str:
        """Graphviz DOT記法生成"""
        lines = ["digraph FuzzyDecisionTree {"]
        lines.append("  rankdir=TD;")
        lines.append("  node [shape=box, style=rounded];")

        TreeVisualizationHelper._add_node_to_dot(root, lines, set())

        lines.append("}")
        return "\n".join(lines)

    @staticmethod
    def _add_node_to_dot(node: AdvancedFuzzyDecisionNode,
                         lines: List[str], visited: set):
        if node.node_id in visited:
            return
        visited.add(node.node_id)

        # ノード定義
        if node.is_leaf:
            label = f"Leaf\\nValue: {node.leaf_value:.3f}"
            color = "lightgreen"
        else:
            label = f"{node.feature_name}\\nMFs: {len(node.membership_functions)}"
            color = "lightblue"

        lines.append(
            f'  "{node.node_id}" [label="{label}", fillcolor="{color}", style=filled];')

        # エッジ定義
        for branch_label, child in node.children.items():
            lines.append(
                f'  "{node.node_id}" -> "{child.node_id}" [label="{branch_label}"];')
            TreeVisualizationHelper._add_node_to_dot(child, lines, visited)

    @staticmethod
    def generate_tree_summary(root: AdvancedFuzzyDecisionNode) -> Dict:
        """木の統計サマリー生成"""
        return {
            'total_nodes': TreeVisualizationHelper._count_nodes(root),
            'leaf_nodes': TreeVisualizationHelper._count_leaves(root),
            'max_depth': root.calculate_depth(),
            'total_complexity': root.calculate_complexity(),
            'total_membership_functions': TreeVisualizationHelper._count_membership_functions(root),
            'feature_usage': TreeVisualizationHelper._analyze_feature_usage(root)
        }

    @staticmethod
    def _count_nodes(node: AdvancedFuzzyDecisionNode) -> int:
        count = 1
        for child in node.children.values():
            count += TreeVisualizationHelper._count_nodes(child)
        return count

    @staticmethod
    def _count_leaves(node: AdvancedFuzzyDecisionNode) -> int:
        if node.is_leaf:
            return 1
        count = 0
        for child in node.children.values():
            count += TreeVisualizationHelper._count_leaves(child)
        return count

    @staticmethod
    def _count_membership_functions(node: AdvancedFuzzyDecisionNode) -> int:
        count = len(node.membership_functions)
        for child in node.children.values():
            count += TreeVisualizationHelper._count_membership_functions(child)
        return count

    @staticmethod
    def _analyze_feature_usage(node: AdvancedFuzzyDecisionNode) -> Dict:
        usage = {}
        TreeVisualizationHelper._collect_feature_usage(node, usage)
        return usage

    @staticmethod
    def _collect_feature_usage(node: AdvancedFuzzyDecisionNode, usage: Dict):
        if not node.is_leaf and node.feature_name:
            usage[node.feature_name] = usage.get(node.feature_name, 0) + 1

        for child in node.children.values():
            TreeVisualizationHelper._collect_feature_usage(child, usage)
