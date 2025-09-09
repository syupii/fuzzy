# backend/core/decision_tree/tree.py - ファジィ決定木実装
# 研究室選択支援システム用

import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json

@dataclass
class TreeConfig:
    """決定木設定"""
    max_depth: int = 8
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    fuzzy_threshold: float = 0.1
    max_features: Optional[int] = None
    criterion: str = "fuzzy_entropy"  # "fuzzy_entropy", "fuzzy_gini", "fuzzy_gain"

class FuzzySet:
    """ファジィ集合（簡易版）"""
    
    def __init__(self, name: str, low: float, medium: float, high: float):
        self.name = name
        self.low = low
        self.medium = medium
        self.high = high
    
    def membership(self, value: float) -> Dict[str, float]:
        """メンバーシップ値計算"""
        # 三角メンバーシップ関数を使用
        low_membership = max(0, min(1, (self.medium - value) / (self.medium - self.low))) if self.medium != self.low else (1 if value == self.low else 0)
        high_membership = max(0, min(1, (value - self.medium) / (self.high - self.medium))) if self.high != self.medium else (1 if value == self.high else 0)
        medium_membership = 1 - max(low_membership, high_membership)
        
        return {
            f"{self.name}_low": low_membership,
            f"{self.name}_medium": medium_membership,
            f"{self.name}_high": high_membership
        }

class FuzzyNode:
    """ファジィ決定木ノード"""
    
    def __init__(self, depth: int = 0):
        self.depth = depth
        self.feature: Optional[str] = None
        self.fuzzy_sets: Dict[str, float] = {}
        self.children: Dict[str, 'FuzzyNode'] = {}
        self.is_leaf: bool = False
        self.prediction: Dict[str, float] = {}
        self.samples: List[Dict[str, Any]] = []
        self.fuzzy_entropy: float = 0.0
        self.information_gain: float = 0.0
        self.node_id: str = ""
    
    def add_child(self, condition: str, child: 'FuzzyNode'):
        """子ノードを追加"""
        self.children[condition] = child
    
    def predict(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """予測実行"""
        if self.is_leaf:
            return self.prediction
        
        if not self.feature or self.feature not in sample:
            return {"compatibility": 0.5}  # デフォルト値
        
        # ファジィ値の計算
        feature_value = sample[self.feature]
        fuzzy_set = FuzzySet(self.feature, 1.0, 5.5, 10.0)
        memberships = fuzzy_set.membership(feature_value)
        
        # 各子ノードの予測を統合
        predictions = {}
        total_weight = 0.0
        
        for condition, child in self.children.items():
            if condition in memberships:
                weight = memberships[condition]
                if weight > 0:
                    child_prediction = child.predict(sample)
                    for key, value in child_prediction.items():
                        if key not in predictions:
                            predictions[key] = 0.0
                        predictions[key] += value * weight
                    total_weight += weight
        
        # 正規化
        if total_weight > 0:
            for key in predictions:
                predictions[key] /= total_weight
        else:
            predictions = {"compatibility": 0.5}
        
        return predictions
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = {
            "depth": self.depth,
            "feature": self.feature,
            "is_leaf": self.is_leaf,
            "prediction": self.prediction,
            "fuzzy_entropy": self.fuzzy_entropy,
            "information_gain": self.information_gain,
            "node_id": self.node_id,
            "sample_count": len(self.samples)
        }
        
        if not self.is_leaf:
            result["children"] = {condition: child.to_dict() for condition, child in self.children.items()}
        
        return result

class FuzzyDecisionTree:
    """ファジィ決定木"""
    
    def __init__(self, config: TreeConfig = None):
        self.config = config or TreeConfig()
        self.root: Optional[FuzzyNode] = None
        self.feature_importance: Dict[str, float] = {}
        self.training_history: List[Dict[str, Any]] = []
        self.is_trained: bool = False
    
    def fit(self, X: List[Dict[str, Any]], y: List[float], feature_names: List[str] = None):
        """決定木の学習"""
        
        if not X or not y or len(X) != len(y):
            raise ValueError("Invalid training data")
        
        # 特徴量名の設定
        if feature_names is None:
            if X and isinstance(X[0], dict):
                feature_names = list(X[0].keys())
            else:
                feature_names = [f"feature_{i}" for i in range(len(X[0]) if X else 0)]
        
        # 訓練データの準備
        training_samples = []
        for i, sample in enumerate(X):
            training_sample = sample.copy() if isinstance(sample, dict) else {}
            training_sample['target'] = y[i]
            training_samples.append(training_sample)
        
        # ルートノード構築
        self.root = FuzzyNode(depth=0)
        self.root.samples = training_samples
        self.root.node_id = "root"
        
        # 決定木構築
        self._build_tree(self.root, feature_names, training_samples)
        
        # 特徴量重要度計算
        self._calculate_feature_importance()
        
        self.is_trained = True
        
        # 学習履歴記録
        self.training_history.append({
            "timestamp": "current",
            "samples_count": len(training_samples),
            "features_count": len(feature_names),
            "tree_depth": self._calculate_tree_depth(),
            "leaf_nodes": self._count_leaf_nodes()
        })
    
    def _build_tree(self, node: FuzzyNode, feature_names: List[str], samples: List[Dict[str, Any]]):
        """決定木の再帰構築"""
        
        # 終了条件チェック
        if (node.depth >= self.config.max_depth or 
            len(samples) < self.config.min_samples_split):
            self._make_leaf(node, samples)
            return
        
        # 最適分割の検索
        best_feature, best_gain = self._find_best_split(samples, feature_names)
        
        if best_feature is None or best_gain < self.config.fuzzy_threshold:
            self._make_leaf(node, samples)
            return
        
        # ノード設定
        node.feature = best_feature
        node.information_gain = best_gain
        
        # ファジィ分割
        splits = self._fuzzy_split(samples, best_feature)
        
        for condition, child_samples in splits.items():
            if len(child_samples) >= self.config.min_samples_leaf:
                child_node = FuzzyNode(depth=node.depth + 1)
                child_node.samples = child_samples
                child_node.node_id = f"{node.node_id}_{condition}"
                
                node.add_child(condition, child_node)
                self._build_tree(child_node, feature_names, child_samples)
        
        # 子ノードがない場合は葉にする
        if not node.children:
            self._make_leaf(node, samples)
    
    def _find_best_split(self, samples: List[Dict[str, Any]], feature_names: List[str]) -> Tuple[Optional[str], float]:
        """最適分割の検索"""
        
        best_feature = None
        best_gain = 0.0
        
        parent_entropy = self._calculate_fuzzy_entropy(samples)
        
        for feature in feature_names:
            if feature == 'target':
                continue
            
            # 特徴量の値を取得
            feature_values = [sample.get(feature, 5.5) for sample in samples if feature in sample]
            if not feature_values:
                continue
            
            # ファジィ分割によるエントロピー計算
            splits = self._fuzzy_split(samples, feature)
            weighted_entropy = 0.0
            total_samples = len(samples)
            
            for child_samples in splits.values():
                if child_samples:
                    weight = len(child_samples) / total_samples
                    child_entropy = self._calculate_fuzzy_entropy(child_samples)
                    weighted_entropy += weight * child_entropy
            
            # 情報利得計算
            information_gain = parent_entropy - weighted_entropy
            
            if information_gain > best_gain:
                best_gain = information_gain
                best_feature = feature
        
        return best_feature, best_gain
    
    def _fuzzy_split(self, samples: List[Dict[str, Any]], feature: str) -> Dict[str, List[Dict[str, Any]]]:
        """ファジィ分割"""
        
        splits = {
            f"{feature}_low": [],
            f"{feature}_medium": [],
            f"{feature}_high": []
        }
        
        # 特徴量の範囲計算
        feature_values = [sample.get(feature, 5.5) for sample in samples if feature in sample]
        if not feature_values:
            return splits
        
        min_val = min(feature_values)
        max_val = max(feature_values)
        medium_val = (min_val + max_val) / 2
        
        fuzzy_set = FuzzySet(feature, min_val, medium_val, max_val)
        
        for sample in samples:
            if feature not in sample:
                continue
            
            value = sample[feature]
            memberships = fuzzy_set.membership(value)
            
            # 最大メンバーシップ値の条件に分類
            max_membership = max(memberships.values())
            for condition, membership in memberships.items():
                if membership == max_membership:
                    splits[condition].append(sample)
                    break
        
        return splits
    
    def _calculate_fuzzy_entropy(self, samples: List[Dict[str, Any]]) -> float:
        """ファジィエントロピー計算"""
        
        if not samples:
            return 0.0
        
        # ターゲット値の分布
        targets = [sample.get('target', 0.5) for sample in samples]
        
        # ファジィ化されたターゲット値のエントロピー
        # 連続値を3つの区間に分割
        low_count = sum(1 for t in targets if t <= 0.33)
        medium_count = sum(1 for t in targets if 0.33 < t <= 0.67)
        high_count = sum(1 for t in targets if t > 0.67)
        
        total = len(targets)
        entropy = 0.0
        
        for count in [low_count, medium_count, high_count]:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _make_leaf(self, node: FuzzyNode, samples: List[Dict[str, Any]]):
        """葉ノード作成"""
        
        node.is_leaf = True
        
        if not samples:
            node.prediction = {"compatibility": 0.5}
            return
        
        # 平均値を予測値とする
        targets = [sample.get('target', 0.5) for sample in samples]
        avg_target = sum(targets) / len(targets)
        
        node.prediction = {"compatibility": avg_target}
        node.fuzzy_entropy = self._calculate_fuzzy_entropy(samples)
    
    def predict(self, X: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """予測実行"""
        
        if not self.is_trained or not self.root:
            raise ValueError("Model is not trained")
        
        if isinstance(X, dict):
            return self.root.predict(X)
        else:
            return [self.root.predict(sample) for sample in X]
    
    def _calculate_feature_importance(self):
        """特徴量重要度計算"""
        
        if not self.root:
            return
        
        importance = {}
        self._calculate_node_importance(self.root, importance)
        
        # 正規化
        total_importance = sum(importance.values())
        if total_importance > 0:
            self.feature_importance = {
                feature: imp / total_importance 
                for feature, imp in importance.items()
            }
        else:
            self.feature_importance = {}
    
    def _calculate_node_importance(self, node: FuzzyNode, importance: Dict[str, float]):
        """ノード重要度の再帰計算"""
        
        if node.is_leaf or not node.feature:
            return
        
        if node.feature not in importance:
            importance[node.feature] = 0.0
        
        importance[node.feature] += node.information_gain * len(node.samples)
        
        for child in node.children.values():
            self._calculate_node_importance(child, importance)
    
    def _calculate_tree_depth(self) -> int:
        """決定木の深さ計算"""
        
        if not self.root:
            return 0
        
        return self._calculate_node_depth(self.root)
    
    def _calculate_node_depth(self, node: FuzzyNode) -> int:
        """ノード深さの再帰計算"""
        
        if node.is_leaf:
            return node.depth
        
        max_depth = node.depth
        for child in node.children.values():
            child_depth = self._calculate_node_depth(child)
            max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    def _count_leaf_nodes(self) -> int:
        """葉ノード数計算"""
        
        if not self.root:
            return 0
        
        return self._count_node_leaves(self.root)
    
    def _count_node_leaves(self, node: FuzzyNode) -> int:
        """葉ノードの再帰カウント"""
        
        if node.is_leaf:
            return 1
        
        count = 0
        for child in node.children.values():
            count += self._count_node_leaves(child)
        
        return count
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        
        result = {
            "is_trained": self.is_trained,
            "config": {
                "max_depth": self.config.max_depth,
                "min_samples_split": self.config.min_samples_split,
                "min_samples_leaf": self.config.min_samples_leaf,
                "fuzzy_threshold": self.config.fuzzy_threshold,
                "criterion": self.config.criterion
            },
            "feature_importance": self.feature_importance,
            "tree_depth": self._calculate_tree_depth(),
            "leaf_nodes": self._count_leaf_nodes(),
            "training_history": self.training_history
        }
        
        if self.root:
            result["tree_structure"] = self.root.to_dict()
        
        return result
    
    def save_model(self, filepath: str):
        """モデル保存"""
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
            print(f"✅ モデルを保存しました: {filepath}")
        except Exception as e:
            print(f"❌ モデル保存エラー: {e}")
    
    def get_prediction_path(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """予測パス取得（説明可能性のため）"""
        
        if not self.root:
            return []
        
        path = []
        self._trace_prediction_path(self.root, sample, path)
        return path
    
    def _trace_prediction_path(self, node: FuzzyNode, sample: Dict[str, Any], path: List[Dict[str, Any]]):
        """予測パスの追跡"""
        
        path_info = {
            "node_id": node.node_id,
            "depth": node.depth,
            "feature": node.feature,
            "is_leaf": node.is_leaf,
            "sample_count": len(node.samples)
        }
        
        if node.is_leaf:
            path_info["prediction"] = node.prediction
        
        path.append(path_info)
        
        if not node.is_leaf and node.feature and node.feature in sample:
            # 最適な子ノードを選択
            feature_value = sample[node.feature]
            fuzzy_set = FuzzySet(node.feature, 1.0, 5.5, 10.0)
            memberships = fuzzy_set.membership(feature_value)
            
            best_condition = max(memberships, key=memberships.get)
            if best_condition in node.children:
                self._trace_prediction_path(node.children[best_condition], sample, path)

# テスト用メイン関数
if __name__ == "__main__":
    print("🧪 ファジィ決定木テスト開始...")
    
    # テストデータ作成
    test_X = [
        {"research_intensity": 8.0, "advisor_style": 7.0, "team_work": 8.5},
        {"research_intensity": 6.0, "advisor_style": 8.5, "team_work": 9.0},
        {"research_intensity": 9.0, "advisor_style": 6.0, "team_work": 7.0},
        {"research_intensity": 5.0, "advisor_style": 9.0, "team_work": 8.0},
        {"research_intensity": 7.5, "advisor_style": 7.5, "team_work": 7.5}
    ]
    
    test_y = [0.85, 0.75, 0.80, 0.70, 0.75]
    
    # 決定木設定
    config = TreeConfig(
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        fuzzy_threshold=0.01
    )
    
    # 決定木作成・学習
    tree = FuzzyDecisionTree(config)
    tree.fit(test_X, test_y)
    
    # 予測テスト
    test_sample = {"research_intensity": 8.5, "advisor_style": 7.0, "team_work": 8.0}
    prediction = tree.predict(test_sample)
    
    print(f"📊 予測結果: {prediction}")
    print(f"🏗️ 決定木深さ: {tree._calculate_tree_depth()}")
    print(f"🍃 葉ノード数: {tree._count_leaf_nodes()}")
    print(f"📈 特徴量重要度: {tree.feature_importance}")
    
    # 予測パス
    path = tree.get_prediction_path(test_sample)
    print(f"🛤️ 予測パス: {len(path)}ノード")
    
    print("✅ テスト完了")