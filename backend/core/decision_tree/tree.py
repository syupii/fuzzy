# core/decision_tree/tree.py - ファジィ決定木メインクラス

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import json
from dataclasses import dataclass, asdict
from collections import defaultdict

from core.decision_tree.node import (
    FuzzyTreeNode, FuzzyInternalNode, FuzzyLeafNode, FuzzyRuleNode,
    NodeTraverser, SplitCondition
)

@dataclass
class TreeMetrics:
    """決定木の評価指標"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    node_count: int = 0
    depth: int = 0
    leaf_count: int = 0
    complexity: float = 0.0

@dataclass
class PredictionResult:
    """予測結果"""
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    path: List[str]  # 決定パス
    node_activations: Dict[str, float]  # ノード活性度

class FuzzyDecisionTree:
    """ファジィ決定木クラス"""
    
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, fuzzy_threshold: float = 0.1):
        
        # 木構造パラメータ
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.fuzzy_threshold = fuzzy_threshold
        
        # 木構造
        self.root: Optional[FuzzyTreeNode] = None
        self.feature_names: List[str] = []
        self.class_names: List[str] = []
        self.n_features: int = 0
        self.n_classes: int = 0
        
        # 学習データ情報
        self.training_samples_count = 0
        self.feature_importances: Dict[str, float] = {}
        
        # 評価指標
        self.metrics: Optional[TreeMetrics] = None
        
        # 予測履歴
        self.prediction_history: List[PredictionResult] = []
    
    def fit(self, X: List[Dict[str, Any]], y: List[str]) -> 'FuzzyDecisionTree':
        """
        ファジィ決定木を学習
        
        Args:
            X: 特徴量データ（辞書のリスト）
            y: ラベルデータ
            
        Returns:
            self: 学習済み決定木
        """
        
        # データ検証
        if len(X) != len(y):
            raise ValueError("特徴量データとラベルデータの長さが一致しません")
        
        if len(X) == 0:
            raise ValueError("学習データが空です")
        
        # 基本情報の設定
        self.feature_names = list(X[0].keys()) if X else []
        self.class_names = list(set(y))
        self.n_features = len(self.feature_names)
        self.n_classes = len(self.class_names)
        self.training_samples_count = len(X)
        
        print(f"🌳 ファジィ決定木学習開始")
        print(f"   サンプル数: {len(X)}")
        print(f"   特徴量数: {self.n_features}")
        print(f"   クラス数: {self.n_classes}")
        
        # 決定木構築
        self.root = self._build_tree(X, y, depth=0)
        
        # 特徴量重要度の計算
        self._calculate_feature_importance()
        
        # 評価指標の計算
        self._calculate_metrics(X, y)
        
        print(f"✅ 決定木学習完了")
        print(f"   木の深さ: {self.get_depth()}")
        print(f"   ノード数: {self.get_node_count()}")
        print(f"   葉ノード数: {self.get_leaf_count()}")
        
        return self
    
    def _build_tree(self, X: List[Dict[str, Any]], y: List[str], 
                   depth: int, node_id: str = "root") -> FuzzyTreeNode:
        """再帰的に決定木を構築"""
        
        # 停止条件チェック
        if self._should_stop_splitting(X, y, depth):
            return self._create_leaf_node(X, y, depth, node_id)
        
        # 最適な分岐を探索
        best_split = self._find_best_split(X, y)
        
        if best_split is None:
            return self._create_leaf_node(X, y, depth, node_id)
        
        # 内部ノード作成
        split_condition = SplitCondition(
            feature=best_split["feature"],
            threshold=best_split["threshold"],
            linguistic_value=best_split["linguistic_value"],
            membership_threshold=self.fuzzy_threshold
        )
        
        internal_node = FuzzyInternalNode(node_id, split_condition, depth)
        internal_node.samples_count = len(X)
        internal_node.purity = self._calculate_purity(y)
        
        # データ分割
        splits = self._split_data_fuzzy(X, y, best_split)
        
        # 子ノード再帰構築
        for branch_name, (X_subset, y_subset) in splits.items():
            if len(X_subset) > 0:
                child_id = f"{node_id}_{branch_name}"
                child_node = self._build_tree(X_subset, y_subset, depth + 1, child_id)
                internal_node.add_child(branch_name, child_node)
        
        return internal_node
    
    def _should_stop_splitting(self, X: List[Dict[str, Any]], y: List[str], depth: int) -> bool:
        """分岐を停止すべきかどうかを判定"""
        
        # 深さ制限
        if depth >= self.max_depth:
            return True
        
        # サンプル数制限
        if len(X) < self.min_samples_split:
            return True
        
        # 純度チェック（すべて同じクラス）
        if len(set(y)) <= 1:
            return True
        
        # 最小葉サンプル数チェック
        if len(X) < self.min_samples_leaf * 2:
            return True
        
        return False
    
    def _find_best_split(self, X: List[Dict[str, Any]], y: List[str]) -> Optional[Dict[str, Any]]:
        """最適な分岐を探索"""
        
        best_split = None
        best_score = -1.0
        
        # 各特徴量について分岐を探索
        for feature in self.feature_names:
            feature_values = [sample[feature] for sample in X if feature in sample]
            
            if not feature_values:
                continue
            
            # 候補閾値を生成
            thresholds = self._generate_thresholds(feature_values)
            
            for threshold in thresholds:
                for linguistic_value in ["low", "medium", "high"]:
                    
                    # 分岐評価
                    score = self._evaluate_split(X, y, feature, threshold, linguistic_value)
                    
                    if score > best_score:
                        best_score = score
                        best_split = {
                            "feature": feature,
                            "threshold": threshold,
                            "linguistic_value": linguistic_value,
                            "score": score
                        }
        
        return best_split
    
    def _generate_thresholds(self, values: List[float]) -> List[float]:
        """分岐候補の閾値を生成"""
        
        if len(values) < 2:
            return [np.mean(values)]
        
        sorted_values = sorted(set(values))
        
        # 分位点を閾値候補とする
        thresholds = []
        
        # 四分位点
        thresholds.extend(np.percentile(sorted_values, [25, 50, 75]))
        
        # 隣接値の中点
        for i in range(len(sorted_values) - 1):
            midpoint = (sorted_values[i] + sorted_values[i + 1]) / 2
            thresholds.append(midpoint)
        
        return list(set(thresholds))
    
    def _evaluate_split(self, X: List[Dict[str, Any]], y: List[str],
                       feature: str, threshold: float, linguistic_value: str) -> float:
        """分岐の品質を評価"""
        
        # ファジィ分岐でデータ分割
        splits = self._split_data_fuzzy(X, y, {
            "feature": feature,
            "threshold": threshold,
            "linguistic_value": linguistic_value
        })
        
        # 情報ゲインを計算
        original_entropy = self._calculate_entropy(y)
        weighted_entropy = 0.0
        total_samples = len(y)
        
        for branch_name, (X_subset, y_subset) in splits.items():
            if len(y_subset) > 0:
                weight = len(y_subset) / total_samples
                branch_entropy = self._calculate_entropy(y_subset)
                weighted_entropy += weight * branch_entropy
        
        information_gain = original_entropy - weighted_entropy
        
        # ファジィ補正（分岐の明確さを考慮）
        fuzziness_penalty = self._calculate_fuzziness_penalty(X, feature, threshold, linguistic_value)
        
        return information_gain - fuzziness_penalty
    
    def _split_data_fuzzy(self, X: List[Dict[str, Any]], y: List[str],
                         split_info: Dict[str, Any]) -> Dict[str, Tuple[List[Dict], List[str]]]:
        """ファジィ分岐でデータを分割"""
        
        feature = split_info["feature"]
        threshold = split_info["threshold"]
        linguistic_value = split_info["linguistic_value"]
        
        splits = {"left": ([], []), "right": ([], []), "center": ([], [])}
        
        for i, sample in enumerate(X):
            if feature not in sample:
                continue
            
            feature_value = sample[feature]
            memberships = self._calculate_split_memberships(feature_value, threshold, linguistic_value)
            
            # 最大帰属度の分岐に割り当て
            best_branch = max(memberships.keys(), key=lambda k: memberships[k])
            
            if memberships[best_branch] > self.fuzzy_threshold:
                splits[best_branch][0].append(sample)
                splits[best_branch][1].append(y[i])
        
        # 空の分岐を除去
        return {k: v for k, v in splits.items() if len(v[0]) > 0}
    
    def _calculate_split_memberships(self, value: float, threshold: float, 
                                   linguistic_value: str) -> Dict[str, float]:
        """分岐への帰属度を計算"""
        
        if linguistic_value == "low":
            if value <= threshold:
                return {"left": 1.0, "right": 0.0, "center": 0.0}
            else:
                distance = value - threshold
                membership = max(0, 1.0 - distance / 3.0)
                return {"left": membership, "right": 1.0 - membership, "center": 0.0}
                
        elif linguistic_value == "high":
            if value >= threshold:
                return {"left": 0.0, "right": 1.0, "center": 0.0}
            else:
                distance = threshold - value
                membership = max(0, 1.0 - distance / 3.0)
                return {"left": 1.0 - membership, "right": membership, "center": 0.0}
                
        else:  # medium
            distance = abs(value - threshold)
            if distance <= 1.5:
                center_membership = 1.0 - distance / 1.5
                side_membership = (1.0 - center_membership) / 2
                return {
                    "left": side_membership if value < threshold else 0,
                    "right": side_membership if value > threshold else 0,
                    "center": center_membership
                }
            else:
                return {
                    "left": 1.0 if value < threshold else 0,
                    "right": 1.0 if value > threshold else 0,
                    "center": 0.0
                }
    
    def _calculate_fuzziness_penalty(self, X: List[Dict[str, Any]], feature: str,
                                   threshold: float, linguistic_value: str) -> float:
        """ファジィネスのペナルティを計算"""
        
        feature_values = [sample[feature] for sample in X if feature in sample]
        
        if not feature_values:
            return 0.0
        
        # 境界付近のサンプル割合を計算
        boundary_count = 0
        boundary_width = 2.0
        
        for value in feature_values:
            if abs(value - threshold) <= boundary_width:
                boundary_count += 1
        
        boundary_ratio = boundary_count / len(feature_values)
        
        # ファジィネスが高いほどペナルティが大きい
        return boundary_ratio * 0.1
    
    def _create_leaf_node(self, X: List[Dict[str, Any]], y: List[str],
                         depth: int, node_id: str) -> FuzzyLeafNode:
        """葉ノードを作成"""
        
        # クラス分布を計算
        class_counts = defaultdict(int)
        for label in y:
            class_counts[label] += 1
        
        total_samples = len(y)
        class_distribution = {
            class_name: count / total_samples
            for class_name, count in class_counts.items()
        }
        
        # 信頼度計算（最多クラスの割合）
        confidence = max(class_distribution.values()) if class_distribution else 0.0
        
        leaf_node = FuzzyLeafNode(node_id, class_distribution, depth, confidence)
        leaf_node.samples_count = total_samples
        leaf_node.purity = confidence
        
        # サポートサンプルを追加
        for sample in X:
            leaf_node.add_support_sample(sample)
        
        return leaf_node
    
    def predict(self, X: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[PredictionResult, List[PredictionResult]]:
        """予測を実行"""
        
        if self.root is None:
            raise ValueError("モデルが学習されていません")
        
        # 単一サンプルの場合
        if isinstance(X, dict):
            return self._predict_single(X)
        
        # 複数サンプルの場合
        return [self._predict_single(sample) for sample in X]
    
    def _predict_single(self, sample: Dict[str, Any]) -> PredictionResult:
        """単一サンプルの予測"""
        
        prediction_path = []
        node_activations = {}
        
        # ルートから予測
        class_probabilities = self._predict_recursive(self.root, sample, prediction_path, node_activations)
        
        # 最確クラスと信頼度
        predicted_class = max(class_probabilities.keys(), key=lambda k: class_probabilities[k])
        confidence = class_probabilities[predicted_class]
        
        result = PredictionResult(
            predicted_class=predicted_class,
            confidence=confidence,
            class_probabilities=class_probabilities,
            path=prediction_path,
            node_activations=node_activations
        )
        
        self.prediction_history.append(result)
        return result
    
    def _predict_recursive(self, node: FuzzyTreeNode, sample: Dict[str, Any],
                          path: List[str], activations: Dict[str, float]) -> Dict[str, float]:
        """再帰的予測"""
        
        path.append(node.node_id)
        activations[node.node_id] = 1.0  # 簡略化
        
        return node.predict(sample)
    
    def _calculate_entropy(self, y: List[str]) -> float:
        """エントロピーを計算"""
        
        if not y:
            return 0.0
        
        class_counts = defaultdict(int)
        for label in y:
            class_counts[label] += 1
        
        total = len(y)
        entropy = 0.0
        
        for count in class_counts.values():
            if count > 0:
                probability = count / total
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _calculate_purity(self, y: List[str]) -> float:
        """純度を計算（最多クラスの割合）"""
        
        if not y:
            return 0.0
        
        class_counts = defaultdict(int)
        for label in y:
            class_counts[label] += 1
        
        return max(class_counts.values()) / len(y)
    
    def _calculate_feature_importance(self) -> None:
        """特徴量重要度を計算"""
        
        if self.root is None:
            return
        
        importance_counts = defaultdict(float)
        
        def collect_feature_usage(node):
            if isinstance(node, FuzzyInternalNode):
                feature = node.split_condition.feature
                importance_counts[feature] += node.samples_count
            return None
        
        NodeTraverser.traverse_preorder(self.root, collect_feature_usage)
        
        # 正規化
        total_importance = sum(importance_counts.values())
        if total_importance > 0:
            self.feature_importances = {
                feature: importance / total_importance
                for feature, importance in importance_counts.items()
            }
    
    def _calculate_metrics(self, X: List[Dict[str, Any]], y: List[str]) -> None:
        """評価指標を計算"""
        
        if self.root is None:
            return
        
        # 予測実行
        predictions = self.predict(X)
        predicted_classes = [pred.predicted_class for pred in predictions]
        
        # 精度計算
        correct = sum(1 for true, pred in zip(y, predicted_classes) if true == pred)
        accuracy = correct / len(y) if y else 0.0
        
        # 木の複雑さ
        node_counts = NodeTraverser.count_nodes(self.root)
        depth = NodeTraverser.calculate_tree_depth(self.root)
        complexity = node_counts["total"] / (depth + 1)  # 簡略化した複雑さ指標
        
        self.metrics = TreeMetrics(
            accuracy=accuracy,
            precision=accuracy,  # 簡略化
            recall=accuracy,     # 簡略化
            f1_score=accuracy,   # 簡略化
            node_count=node_counts["total"],
            depth=depth,
            leaf_count=node_counts["leaf"],
            complexity=complexity
        )
    
    def get_depth(self) -> int:
        """木の深さを取得"""
        if self.root is None:
            return 0
        return NodeTraverser.calculate_tree_depth(self.root)
    
    def get_node_count(self) -> int:
        """ノード数を取得"""
        if self.root is None:
            return 0
        return NodeTraverser.count_nodes(self.root)["total"]
    
    def get_leaf_count(self) -> int:
        """葉ノード数を取得"""
        if self.root is None:
            return 0
        return NodeTraverser.count_nodes(self.root)["leaf"]
    
    def print_tree(self, max_depth: Optional[int] = None) -> None:
        """決定木を表示"""
        
        if self.root is None:
            print("決定木が構築されていません")
            return
        
        print("🌳 ファジィ決定木構造")
        print("=" * 50)
        
        def print_node(node, prefix="", is_last=True):
            if max_depth is not None and node.depth > max_depth:
                return
            
            # ノード情報表示
            node_info = f"[{node.node_id}] "
            
            if isinstance(node, FuzzyInternalNode):
                node_info += f"IF {node.split_condition.feature} {node.split_condition.linguistic_value} {node.split_condition.threshold}"
            elif isinstance(node, FuzzyLeafNode):
                predicted_class = node.predicted_class
                confidence = node.confidence
                node_info += f"PREDICT: {predicted_class} ({confidence:.3f})"
            
            print(f"{prefix}{'└── ' if is_last else '├── '}{node_info}")
            
            # 子ノード表示
            if hasattr(node, 'children') and node.children:
                children = list(node.children.items())
                for i, (branch_name, child) in enumerate(children):
                    is_last_child = (i == len(children) - 1)
                    child_prefix = prefix + ("    " if is_last else "│   ")
                    print(f"{child_prefix}{'└── ' if is_last_child else '├── '}{branch_name}")
                    print_node(child, child_prefix + ("    " if is_last_child else "│   "), True)
        
        print_node(self.root)
    
    def get_tree_summary(self) -> Dict[str, Any]:
        """決定木のサマリーを取得"""
        
        summary = {
            "structure": {
                "depth": self.get_depth(),
                "node_count": self.get_node_count(),
                "leaf_count": self.get_leaf_count(),
                "feature_count": self.n_features,
                "class_count": self.n_classes
            },
            "parameters": {
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "fuzzy_threshold": self.fuzzy_threshold
            },
            "training": {
                "samples_count": self.training_samples_count,
                "feature_names": self.feature_names,
                "class_names": self.class_names
            },
            "feature_importance": self.feature_importances,
            "metrics": asdict(self.metrics) if self.metrics else None
        }
        
        return summary
    
    def export_tree(self, filepath: str) -> None:
        """決定木をJSONファイルにエクスポート"""
        
        tree_data = self.get_tree_summary()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(tree_data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 決定木をエクスポートしました: {filepath}")
    
    def clear_prediction_history(self) -> None:
        """予測履歴をクリア"""
        self.prediction_history.clear()