"""
ファジィ決定木ノード - core/decision_tree/node.py
ファジィ決定木の個別ノードクラス
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

from ..fuzzy.membership import MembershipFunction, TriangularMF, GaussianMF
from ..fuzzy.inference import SimpleFuzzyInferenceEngine


class NodeType(Enum):
    """ノードの種類"""
    INTERNAL = "internal"
    LEAF = "leaf"
    ROOT = "root"


@dataclass
class NodeStatistics:
    """ノード統計情報"""
    sample_count: int = 0
    prediction_count: int = 0
    average_prediction: float = 0.0
    confidence_score: float = 0.0
    information_gain: float = 0.0
    purity: float = 0.0


class FuzzyDecisionNode:
    """ファジィ決定木ノード"""
    
    def __init__(self, node_id: str = None, node_type: NodeType = NodeType.INTERNAL):
        # 基本属性
        self.node_id = node_id or f"node_{id(self)}"
        self.node_type = node_type
        self.is_leaf = (node_type == NodeType.LEAF)
        
        # 分割情報
        self.feature_name: Optional[str] = None
        self.membership_functions: Dict[str, MembershipFunction] = {}
        self.children: Dict[str, 'FuzzyDecisionNode'] = {}
        
        # 葉ノード用
        self.leaf_value: Optional[float] = None
        self.class_distribution: Dict[str, float] = {}
        
        # ツリー構造
        self.parent: Optional['FuzzyDecisionNode'] = None
        self.depth = 0
        self.path_from_root: List[str] = []
        
        # 統計情報
        self.statistics = NodeStatistics()
        self.training_samples: List[Tuple[Dict[str, float], float]] = []
        
        # メタ情報
        self.creation_time: float = 0.0
        self.last_update_time: float = 0.0
        self.importance_score: float = 0.0
    
    def add_membership_function(self, label: str, mf: MembershipFunction):
        """メンバーシップ関数を追加"""
        self.membership_functions[label] = mf
    
    def add_child(self, label: str, child_node: 'FuzzyDecisionNode'):
        """子ノードを追加"""
        self.children[label] = child_node
        child_node.parent = self
        child_node.depth = self.depth + 1
        child_node.path_from_root = self.path_from_root + [label]
    
    def remove_child(self, label: str) -> bool:
        """子ノードを削除"""
        if label in self.children:
            child = self.children[label]
            child.parent = None
            del self.children[label]
            return True
        return False
    
    def predict(self, features: Dict[str, float]) -> float:
        """予測実行"""
        
        self.statistics.prediction_count += 1
        
        # 葉ノードの場合
        if self.is_leaf:
            prediction = self.leaf_value if self.leaf_value is not None else 0.5
            self._update_prediction_statistics(prediction)
            return prediction
        
        # 特徴量が存在しない場合
        if not self.feature_name or self.feature_name not in features:
            # デフォルト予測（訓練データの平均）
            if self.training_samples:
                default_prediction = np.mean([sample[1] for sample in self.training_samples])
            else:
                default_prediction = 0.5
            self._update_prediction_statistics(default_prediction)
            return default_prediction
        
        feature_value = features[self.feature_name]
        
        # ファジィ推論による予測
        weighted_predictions = []
        total_weight = 0.0
        
        for label, mf in self.membership_functions.items():
            if label in self.children:
                # メンバーシップ度計算
                membership_degree = mf.membership(feature_value)
                
                if membership_degree > 0.01:  # 閾値以上の場合のみ考慮
                    child_prediction = self.children[label].predict(features)
                    weight = membership_degree
                    
                    weighted_predictions.append((child_prediction, weight))
                    total_weight += weight
        
        # 重み付き平均による最終予測
        if total_weight > 0:
            final_prediction = sum(pred * weight for pred, weight in weighted_predictions) / total_weight
        else:
            # フォールバック：訓練データの平均
            final_prediction = self.leaf_value if self.leaf_value is not None else 0.5
        
        # 予測値の範囲制限
        final_prediction = max(0.0, min(1.0, final_prediction))
        
        self._update_prediction_statistics(final_prediction)
        return final_prediction
    
    def predict_with_explanation(self, features: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
        """説明付き予測"""
        
        # 基本予測
        prediction = self.predict(features)
        
        # 決定経路の構築
        decision_path = self._build_decision_path(features)
        
        # 信頼度計算
        confidence = self._calculate_confidence(features)
        
        # 特徴量重要度
        feature_importance = self._calculate_local_feature_importance(features)
        
        explanation = {
            'prediction': prediction,
            'confidence': confidence,
            'decision_path': decision_path,
            'feature_importance': feature_importance,
            'node_info': {
                'node_id': self.node_id,
                'node_type': self.node_type.value,
                'depth': self.depth,
                'is_leaf': self.is_leaf,
                'feature_name': self.feature_name,
                'prediction_count': self.statistics.prediction_count
            },
            'membership_activations': self._get_membership_activations(features) if not self.is_leaf else {}
        }
        
        return prediction, explanation
    
    def _build_decision_path(self, features: Dict[str, float]) -> List[Dict[str, Any]]:
        """決定経路の構築"""
        
        path = []
        current_node = self
        
        while not current_node.is_leaf and len(path) < 10:  # 無限ループ防止
            if current_node.feature_name and current_node.feature_name in features:
                feature_value = features[current_node.feature_name]
                
                # 最も高いメンバーシップ度を持つ分岐を選択
                best_label = None
                best_membership = 0.0
                
                for label, mf in current_node.membership_functions.items():
                    membership = mf.membership(feature_value)
                    if membership > best_membership:
                        best_membership = membership
                        best_label = label
                
                if best_label and best_label in current_node.children:
                    path.append({
                        'node_id': current_node.node_id,
                        'feature': current_node.feature_name,
                        'value': feature_value,
                        'fuzzy_set': best_label,
                        'membership_degree': best_membership,
                        'decision': f"{current_node.feature_name} = {feature_value:.2f} → {best_label}"
                    })
                    
                    current_node = current_node.children[best_label]
                else:
                    break
            else:
                break
        
        # 最終ノード（葉ノード）
        path.append({
            'node_id': current_node.node_id,
            'node_type': 'leaf',
            'prediction': current_node.leaf_value,
            'decision': f"予測値: {current_node.leaf_value:.3f}" if current_node.leaf_value else "予測値: デフォルト"
        })
        
        return path
    
    def _calculate_confidence(self, features: Dict[str, float]) -> float:
        """予測信頼度計算"""
        
        if self.is_leaf:
            # 葉ノードの信頼度は訓練サンプル数とターゲット値の分散に基づく
            if self.statistics.sample_count == 0:
                return 0.5
            
            sample_confidence = min(1.0, self.statistics.sample_count / 20.0)  # 20サンプル以上で最大信頼度
            purity_confidence = self.statistics.purity
            
            return 0.6 * sample_confidence + 0.4 * purity_confidence
        
        if not self.feature_name or self.feature_name not in features:
            return 0.3
        
        feature_value = features[self.feature_name]
        
        # 最大メンバーシップ度を信頼度の基準とする
        max_membership = 0.0
        for mf in self.membership_functions.values():
            membership = mf.membership(feature_value)
            max_membership = max(max_membership, membership)
        
        # 深度による信頼度減衰
        depth_factor = 0.9 ** self.depth
        
        # サンプル数による調整
        sample_factor = min(1.0, self.statistics.sample_count / 10.0)
        
        return max_membership * depth_factor * (0.5 + 0.5 * sample_factor)
    
    def _calculate_local_feature_importance(self, features: Dict[str, float]) -> Dict[str, float]:
        """局所特徴量重要度計算"""
        
        importance = {}
        
        if self.is_leaf or not self.feature_name:
            return importance
        
        # 現在のノードの特徴量重要度
        importance[self.feature_name] = self.importance_score
        
        # 子ノードの重要度を再帰的に計算
        if self.feature_name in features:
            feature_value = features[self.feature_name]
            
            for label, mf in self.membership_functions.items():
                if label in self.children:
                    membership = mf.membership(feature_value)
                    if membership > 0.1:
                    fuzzy_targets.append(target_values[i])
                    total_membership += membership
            
            if fuzzy_targets and total_membership > 0:
                subset_entropy = self._calculate_entropy(fuzzy_targets)
                weight = total_membership / total_samples
                weighted_entropy += weight * subset_entropy
        
        information_gain = parent_entropy - weighted_entropy
        self.statistics.information_gain = information_gain
        
        return information_gain
    
    def _calculate_entropy(self, values: List[float]) -> float:
        """エントロピーの計算（回帰用）"""
        
        if not values:
            return 0.0
        
        # 回帰問題なので分散をエントロピーの代わりに使用
        return np.var(values)
    
    def get_subtree_size(self) -> int:
        """部分木のサイズ（ノード数）"""
        
        size = 1  # 自分自身
        
        for child in self.children.values():
            size += child.get_subtree_size()
        
        return size
    
    def get_subtree_depth(self) -> int:
        """部分木の深度"""
        
        if self.is_leaf or not self.children:
            return 1
        
        max_child_depth = 0
        for child in self.children.values():
            child_depth = child.get_subtree_depth()
            max_child_depth = max(max_child_depth, child_depth)
        
        return 1 + max_child_depth
    
    def get_leaf_nodes(self) -> List['FuzzyDecisionNode']:
        """葉ノードのリストを取得"""
        
        if self.is_leaf:
            return [self]
        
        leaf_nodes = []
        for child in self.children.values():
            leaf_nodes.extend(child.get_leaf_nodes())
        
        return leaf_nodes
    
    def get_internal_nodes(self) -> List['FuzzyDecisionNode']:
        """内部ノードのリストを取得"""
        
        nodes = []
        
        if not self.is_leaf:
            nodes.append(self)
            for child in self.children.values():
                nodes.extend(child.get_internal_nodes())
        
        return nodes
    
    def prune_subtree(self) -> bool:
        """部分木の剪定"""
        
        if self.is_leaf:
            return False
        
        # 子ノードをすべて削除して葉ノードに変換
        self.children.clear()
        self.membership_functions.clear()
        self.is_leaf = True
        self.node_type = NodeType.LEAF
        
        # 葉ノードの値を設定
        if self.training_samples:
            targets = [sample[1] for sample in self.training_samples]
            self.leaf_value = np.mean(targets)
        else:
            self.leaf_value = 0.5
        
        return True
    
    def calculate_complexity_penalty(self) -> float:
        """複雑度ペナルティの計算"""
        
        subtree_size = self.get_subtree_size()
        subtree_depth = self.get_subtree_depth()
        
        # サイズと深度に基づくペナルティ
        size_penalty = subtree_size * 0.01
        depth_penalty = subtree_depth * 0.02
        
        return size_penalty + depth_penalty
    
    def validate_structure(self) -> List[str]:
        """ノード構造の検証"""
        
        errors = []
        
        # 基本検証
        if self.is_leaf:
            if self.children:
                errors.append(f"Leaf node {self.node_id} has children")
            if self.leaf_value is None:
                errors.append(f"Leaf node {self.node_id} has no value")
        else:
            if not self.feature_name:
                errors.append(f"Internal node {self.node_id} has no feature")
            if not self.membership_functions:
                errors.append(f"Internal node {self.node_id} has no membership functions")
        
        # 子ノードの検証
        for label, child in self.children.items():
            if label not in self.membership_functions:
                errors.append(f"Child {label} has no corresponding membership function")
            
            child_errors = child.validate_structure()
            errors.extend(child_errors)
        
        # メンバーシップ関数の検証
        for label, mf in self.membership_functions.items():
            if label not in self.children:
                errors.append(f"Membership function {label} has no corresponding child")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式への変換"""
        
        result = {
            'node_id': self.node_id,
            'node_type': self.node_type.value,
            'is_leaf': self.is_leaf,
            'depth': self.depth,
            'path_from_root': self.path_from_root,
            'statistics': {
                'sample_count': self.statistics.sample_count,
                'prediction_count': self.statistics.prediction_count,
                'average_prediction': self.statistics.average_prediction,
                'confidence_score': self.statistics.confidence_score,
                'information_gain': self.statistics.information_gain,
                'purity': self.statistics.purity
            },
            'importance_score': self.importance_score
        }
        
        if self.is_leaf:
            result['leaf_value'] = self.leaf_value
            result['class_distribution'] = self.class_distribution
        else:
            result['feature_name'] = self.feature_name
            result['membership_functions'] = {
                name: mf.get_params() for name, mf in self.membership_functions.items()
            }
            result['children'] = {
                name: child.to_dict() for name, child in self.children.items()
            }
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FuzzyDecisionNode':
        """辞書からノードを復元"""
        
        # ノード作成
        node_type = NodeType(data.get('node_type', 'internal'))
        node = cls(data.get('node_id'), node_type)
        
        # 基本属性
        node.is_leaf = data.get('is_leaf', False)
        node.depth = data.get('depth', 0)
        node.path_from_root = data.get('path_from_root', [])
        node.importance_score = data.get('importance_score', 0.0)
        
        # 統計情報復元
        stats_data = data.get('statistics', {})
        node.statistics.sample_count = stats_data.get('sample_count', 0)
        node.statistics.prediction_count = stats_data.get('prediction_count', 0)
        node.statistics.average_prediction = stats_data.get('average_prediction', 0.0)
        node.statistics.confidence_score = stats_data.get('confidence_score', 0.0)
        node.statistics.information_gain = stats_data.get('information_gain', 0.0)
        node.statistics.purity = stats_data.get('purity', 0.0)
        
        if node.is_leaf:
            # 葉ノード
            node.leaf_value = data.get('leaf_value')
            node.class_distribution = data.get('class_distribution', {})
        else:
            # 内部ノード
            node.feature_name = data.get('feature_name')
            
            # メンバーシップ関数復元
            mf_data = data.get('membership_functions', {})
            for name, params in mf_data.items():
                if 'a' in params and 'b' in params and 'c' in params:
                    # 三角形メンバーシップ関数
                    mf = TriangularMF(name, params['a'], params['b'], params['c'])
                elif 'center' in params and 'sigma' in params:
                    # ガウシアンメンバーシップ関数
                    mf = GaussianMF(name, params['center'], params['sigma'])
                else:
                    # デフォルト（三角形）
                    mf = TriangularMF(name, 0, 5, 10)
                
                node.add_membership_function(name, mf)
            
            # 子ノード復元
            children_data = data.get('children', {})
            for name, child_data in children_data.items():
                child_node = cls.from_dict(child_data)
                node.add_child(name, child_node)
        
        return node
    
    def get_node_summary(self) -> Dict[str, Any]:
        """ノード概要の取得"""
        
        summary = {
            'node_id': self.node_id,
            'type': self.node_type.value,
            'is_leaf': self.is_leaf,
            'depth': self.depth,
            'sample_count': self.statistics.sample_count,
            'prediction_count': self.statistics.prediction_count,
            'subtree_size': self.get_subtree_size(),
            'subtree_depth': self.get_subtree_depth()
        }
        
        if self.is_leaf:
            summary['leaf_value'] = self.leaf_value
            summary['confidence'] = self._calculate_confidence({})
        else:
            summary['feature_name'] = self.feature_name
            summary['num_fuzzy_sets'] = len(self.membership_functions)
            summary['num_children'] = len(self.children)
            summary['information_gain'] = self.statistics.information_gain
        
        return summary
    
    def __str__(self) -> str:
        """文字列表現"""
        
        if self.is_leaf:
            return f"LeafNode(id={self.node_id}, value={self.leaf_value:.3f}, depth={self.depth})"
        else:
            return f"InternalNode(id={self.node_id}, feature={self.feature_name}, depth={self.depth}, children={len(self.children)})"
    
    def __repr__(self) -> str:
        return self.__str__()


class FuzzyDecisionTree:
    """ファジィ決定木クラス"""
    
    def __init__(self, root: Optional[FuzzyDecisionNode] = None):
        self.root = root
        self.feature_names: List[str] = []
        self.target_name: str = ""
        
        # ツリー統計
        self.total_nodes = 0
        self.total_leaves = 0
        self.max_depth = 0
        self.total_predictions = 0
        
        # メタデータ
        self.creation_time: float = 0.0
        self.last_update_time: float = 0.0
        self.version: str = "1.0"
    
    def predict(self, features: Dict[str, float]) -> float:
        """予測実行"""
        
        if self.root is None:
            return 0.5
        
        self.total_predictions += 1
        return self.root.predict(features)
    
    def predict_with_explanation(self, features: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
        """説明付き予測"""
        
        if self.root is None:
            return 0.5, {'error': 'No root node'}
        
        prediction, node_explanation = self.root.predict_with_explanation(features)
        
        # ツリー全体の説明を追加
        tree_explanation = {
            'tree_info': self.get_tree_info(),
            'node_explanation': node_explanation
        }
        
        return prediction, tree_explanation
    
    def get_tree_info(self) -> Dict[str, Any]:
        """ツリー情報の取得"""
        
        if self.root is None:
            return {'status': 'empty'}
        
        # 統計計算
        self._update_tree_statistics()
        
        return {
            'total_nodes': self.total_nodes,
            'total_leaves': self.total_leaves,
            'max_depth': self.max_depth,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'total_predictions': self.total_predictions,
            'root_node_id': self.root.node_id,
            'version': self.version
        }
    
    def _update_tree_statistics(self):
        """ツリー統計の更新"""
        
        if self.root is None:
            self.total_nodes = 0
            self.total_leaves = 0
            self.max_depth = 0
            return
        
        self.total_nodes = self.root.get_subtree_size()
        self.total_leaves = len(self.root.get_leaf_nodes())
        self.max_depth = self.root.get_subtree_depth()
    
    def validate_tree(self) -> List[str]:
        """ツリー構造の検証"""
        
        if self.root is None:
            return ['No root node']
        
        return self.root.validate_structure()
    
    def prune_tree(self, min_samples_leaf: int = 5) -> int:
        """ツリーの剪定"""
        
        if self.root is None:
            return 0
        
        pruned_count = 0
        
        # 葉ノードのサンプル数が少ない場合に剪定
        def prune_recursive(node: FuzzyDecisionNode) -> int:
            count = 0
            
            if node.is_leaf:
                return count
            
            # 子ノードを先に剪定
            for child in list(node.children.values()):
                count += prune_recursive(child)
            
            # 現在のノードを剪定すべきか判定
            if (node.statistics.sample_count < min_samples_leaf and 
                not node.node_type == NodeType.ROOT):
                node.prune_subtree()
                count += 1
            
            return count
        
        pruned_count = prune_recursive(self.root)
        
        # 統計更新
        self._update_tree_statistics()
        
        return pruned_count
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式への変換"""
        
        return {
            'root': self.root.to_dict() if self.root else None,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'tree_info': self.get_tree_info(),
            'creation_time': self.creation_time,
            'version': self.version
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FuzzyDecisionTree':
        """辞書からツリーを復元"""
        
        tree = cls()
        
        # ルートノード復元
        root_data = data.get('root')
        if root_data:
            tree.root = FuzzyDecisionNode.from_dict(root_data)
        
        # メタデータ復元
        tree.feature_names = data.get('feature_names', [])
        tree.target_name = data.get('target_name', '')
        tree.creation_time = data.get('creation_time', 0.0)
        tree.version = data.get('version', '1.0')
        
        return tree1:
                        child_importance = self.children[label]._calculate_local_feature_importance(features)
                        for feature, imp in child_importance.items():
                            if feature not in importance:
                                importance[feature] = 0.0
                            importance[feature] += imp * membership * 0.7  # 減衰係数
        
        return importance
    
    def _get_membership_activations(self, features: Dict[str, float]) -> Dict[str, float]:
        """メンバーシップ関数の活性化度"""
        
        activations = {}
        
        if self.is_leaf or not self.feature_name or self.feature_name not in features:
            return activations
        
        feature_value = features[self.feature_name]
        
        for label, mf in self.membership_functions.items():
            activations[label] = mf.membership(feature_value)
        
        return activations
    
    def _update_prediction_statistics(self, prediction: float):
        """予測統計の更新"""
        
        # 移動平均で平均予測値を更新
        if self.statistics.prediction_count == 1:
            self.statistics.average_prediction = prediction
        else:
            alpha = 0.1  # 学習率
            self.statistics.average_prediction = (
                (1 - alpha) * self.statistics.average_prediction + alpha * prediction
            )
    
    def update_training_statistics(self, training_data: List[Tuple[Dict[str, float], float]]):
        """訓練統計の更新"""
        
        self.training_samples = training_data
        self.statistics.sample_count = len(training_data)
        
        if training_data:
            targets = [sample[1] for sample in training_data]
            self.statistics.average_prediction = np.mean(targets)
            
            # 純度計算（分散の逆数として定義）
            target_var = np.var(targets)
            self.statistics.purity = 1.0 / (1.0 + target_var)
            
            # 葉ノードの値設定
            if self.is_leaf:
                self.leaf_value = self.statistics.average_prediction
    
    def calculate_information_gain(self, feature_values: List[float], 
                                 target_values: List[float]) -> float:
        """情報ゲインの計算"""
        
        if len(feature_values) != len(target_values) or len(feature_values) == 0:
            return 0.0
        
        # 全体のエントロピー
        parent_entropy = self._calculate_entropy(target_values)
        
        # 分割後の加重エントロピー
        weighted_entropy = 0.0
        total_samples = len(target_values)
        
        for label, mf in self.membership_functions.items():
            # ファジィ分割によるサンプル選択
            fuzzy_targets = []
            total_membership = 0.0
            
            for i, feature_val in enumerate(feature_values):
                membership = mf.membership(feature_val)
                if membership > 0.