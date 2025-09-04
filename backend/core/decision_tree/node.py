"""
ファジィ決定木ノード - core/decision_tree/node.py
修正版：構文エラーを解消
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

try:
    from ..fuzzy.membership import MembershipFunction
except ImportError:
    # フォールバック用の簡単な実装
    class MembershipFunction:
        def __init__(self, name="default"):
            self.name = name
        
        def membership_degree(self, value):
            return 0.5


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
        self.importance_score: float = 0.0
        self.training_samples: List[Tuple[Dict[str, float], float]] = []
        
        # メタ情報
        self.creation_time: float = 0.0
        self.last_update_time: float = 0.0
    
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
        
        if self.is_leaf:
            return self.leaf_value if self.leaf_value is not None else 0.5
        
        # ファジィ推論による子ノード選択
        best_child = None
        max_membership = 0.0
        
        for label, child in self.children.items():
            if label in self.membership_functions and self.feature_name:
                feature_value = features.get(self.feature_name, 0.0)
                membership = self.membership_functions[label].membership_degree(feature_value)
                
                if membership > max_membership:
                    max_membership = membership
                    best_child = child
        
        if best_child:
            return best_child.predict(features)
        
        # フォールバック：デフォルト予測
        return 0.5
    
    def predict_with_explanation(self, features: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
        """詳細説明付き予測"""
        self.statistics.prediction_count += 1
        
        explanation = {
            'node_id': self.node_id,
            'node_type': self.node_type.value,
            'depth': self.depth,
            'is_leaf': self.is_leaf
        }
        
        if self.is_leaf:
            prediction = self.leaf_value if self.leaf_value is not None else 0.5
            explanation.update({
                'prediction': prediction,
                'leaf_value': self.leaf_value,
                'sample_count': self.statistics.sample_count,
                'confidence': self.statistics.confidence_score
            })
            return prediction, explanation
        
        # 内部ノードの処理
        if self.feature_name:
            feature_value = features.get(self.feature_name, 0.0)
            explanation.update({
                'feature_name': self.feature_name,
                'feature_value': feature_value,
                'membership_evaluations': {}
            })
            
            best_child = None
            max_membership = 0.0
            best_label = None
            
            for label, child in self.children.items():
                if label in self.membership_functions:
                    membership = self.membership_functions[label].membership_degree(feature_value)
                    explanation['membership_evaluations'][label] = membership
                    
                    if membership > max_membership:
                        max_membership = membership
                        best_child = child
                        best_label = label
            
            explanation.update({
                'selected_branch': best_label,
                'selected_membership': max_membership
            })
            
            if best_child:
                prediction, child_explanation = best_child.predict_with_explanation(features)
                explanation['child_explanation'] = child_explanation
                return prediction, explanation
        
        # フォールバック
        return 0.5, explanation
    
    def is_pure(self, threshold: float = 0.95) -> bool:
        """ノードの純度チェック"""
        return self.statistics.purity >= threshold
    
    def get_sample_count(self) -> int:
        """サンプル数を取得"""
        return self.statistics.sample_count
    
    def set_leaf_value(self, value: float, confidence: float = 1.0):
        """葉ノードの値を設定"""
        self.leaf_value = value
        self.is_leaf = True
        self.node_type = NodeType.LEAF
        self.statistics.confidence_score = confidence
    
    def calculate_importance(self, total_samples: int) -> float:
        """ノードの重要度計算"""
        if total_samples == 0:
            return 0.0
        
        # サンプル比率による重み付け
        sample_weight = self.statistics.sample_count / total_samples
        
        # 情報利得による重み付け
        gain_weight = self.statistics.information_gain
        
        # 深度による重み付け（浅いノードほど重要）
        depth_weight = 1.0 / (self.depth + 1)
        
        self.importance_score = sample_weight * gain_weight * depth_weight
        return self.importance_score
    
    def validate_structure(self) -> List[str]:
        """構造の妥当性検証"""
        errors = []
        
        if not self.is_leaf and not self.feature_name:
            errors.append(f"Internal node {self.node_id} has no feature name")
        
        if not self.is_leaf and not self.membership_functions:
            errors.append(f"Internal node {self.node_id} has no membership functions")
        
        # 子ノードの検証
        for label, child in self.children.items():
            if label not in self.membership_functions:
                errors.append(f"Child {label} has no corresponding membership function")
            
            child_errors = child.validate_structure()
            errors.extend(child_errors)
        
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
                name: {'name': mf.name} for name, mf in self.membership_functions.items()
            }
            result['children'] = {
                name: child.to_dict() for name, child in self.children.items()
            }
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FuzzyDecisionNode':
        """辞書からノードを復元"""
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
        
        # 葉ノードまたは内部ノードの復元
        if node.is_leaf:
            node.leaf_value = data.get('leaf_value')
            node.class_distribution = data.get('class_distribution', {})
        else:
            node.feature_name = data.get('feature_name')
            
            # メンバーシップ関数の復元（簡略版）
            mf_data = data.get('membership_functions', {})
            for name, mf_info in mf_data.items():
                node.membership_functions[name] = MembershipFunction(name)
            
            # 子ノードの復元
            children_data = data.get('children', {})
            for name, child_data in children_data.items():
                child_node = cls.from_dict(child_data)
                node.add_child(name, child_node)
        
        return node


class FuzzyDecisionTree:
    """ファジィ決定木"""
    
    def __init__(self, root: FuzzyDecisionNode = None):
        self.root = root
        self.feature_names: List[str] = []
        self.target_name: str = "target"
        self.total_nodes = 0
        self.max_depth = 0
        self.creation_time: float = 0.0
        self.last_update_time: float = 0.0
    
    def predict(self, features: Dict[str, float]) -> float:
        """予測実行"""
        if self.root:
            return self.root.predict(features)
        return 0.5
    
    def predict_batch(self, features_list: List[Dict[str, float]]) -> List[float]:
        """バッチ予測"""
        return [self.predict(features) for features in features_list]
    
    def predict_with_explanation(self, features: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
        """詳細説明付き予測"""
        if self.root:
            return self.root.predict_with_explanation(features)
        return 0.5, {'error': 'No root node'}
    
    def get_depth(self) -> int:
        """木の深度を取得"""
        if not self.root:
            return 0
        
        def _get_depth(node: FuzzyDecisionNode) -> int:
            if node.is_leaf:
                return node.depth
            
            max_child_depth = node.depth
            for child in node.children.values():
                child_depth = _get_depth(child)
                max_child_depth = max(max_child_depth, child_depth)
            
            return max_child_depth
        
        return _get_depth(self.root)
    
    def count_nodes(self) -> int:
        """ノード数をカウント"""
        if not self.root:
            return 0
        
        def _count_nodes(node: FuzzyDecisionNode) -> int:
            count = 1  # 自分自身
            for child in node.children.values():
                count += _count_nodes(child)
            return count
        
        self.total_nodes = _count_nodes(self.root)
        return self.total_nodes
    
    def update_statistics(self):
        """統計情報を更新"""
        self.max_depth = self.get_depth()
        self.total_nodes = self.count_nodes()
    
    def validate(self) -> List[str]:
        """木構造の妥当性検証"""
        if not self.root:
            return ['No root node']
        
        return self.root.validate_structure()
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式への変換"""
        return {
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'total_nodes': self.total_nodes,
            'max_depth': self.max_depth,
            'root': self.root.to_dict() if self.root else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FuzzyDecisionTree':
        """辞書から木を復元"""
        tree = cls()
        tree.feature_names = data.get('feature_names', [])
        tree.target_name = data.get('target_name', 'target')
        tree.total_nodes = data.get('total_nodes', 0)
        tree.max_depth = data.get('max_depth', 0)
        
        if data.get('root'):
            tree.root = FuzzyDecisionNode.from_dict(data['root'])
        
        return tree