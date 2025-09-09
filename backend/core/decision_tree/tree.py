# core/decision_tree/tree.py - ファジィ決定木（完全版）

import numpy as np
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
import logging

from core.decision_tree.node import FuzzyTreeNode, FuzzyInternalNode, FuzzyLeafNode, SplitCondition
from core.fuzzy.membership import MembershipFunction

logger = logging.getLogger(__name__)

@dataclass
class TreeMetrics:
    """決定木メトリクス"""
    # 構造メトリクス
    total_nodes: int = 0
    internal_nodes: int = 0
    leaf_nodes: int = 0
    max_depth: int = 0
    average_depth: float = 0.0
    
    # 性能メトリクス
    training_accuracy: float = 0.0
    training_error: float = 0.0
    complexity_score: float = 0.0
    
    # ファジィメトリクス
    fuzzy_coverage: float = 0.0
    rule_consistency: float = 0.0
    membership_overlap: float = 0.0
    
    # 統計情報
    creation_time: float = 0.0
    training_time: float = 0.0
    prediction_count: int = 0
    average_prediction_time: float = 0.0

@dataclass
class PredictionResult:
    """予測結果"""
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    
    # 予測経路情報
    prediction_path: List[str]
    activated_rules: List[str]
    node_memberships: Dict[str, float]
    
    # メタ情報
    prediction_time: float = 0.0
    tree_depth: int = 0
    uncertainty: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "class_probabilities": self.class_probabilities,
            "prediction_path": self.prediction_path,
            "activated_rules": self.activated_rules,
            "node_memberships": self.node_memberships,
            "prediction_time": self.prediction_time,
            "tree_depth": self.tree_depth,
            "uncertainty": self.uncertainty
        }

class FuzzyDecisionTree:
    """ファジィ決定木クラス"""
    
    def __init__(self, tree_id: str = None, max_depth: int = 10):
        self.tree_id = tree_id or f"fuzzy_tree_{int(time.time())}"
        self.max_depth = max_depth
        
        # 決定木構造
        self.root: Optional[FuzzyTreeNode] = None
        self.feature_names: List[str] = []
        self.class_names: List[str] = []
        
        # 学習データ情報
        self.n_features: int = 0
        self.n_classes: int = 0
        self.n_samples: int = 0
        
        # メトリクス
        self.metrics = TreeMetrics()
        
        # 状態管理
        self.is_trained: bool = False
        self.training_history: List[Dict[str, Any]] = []
        
        # ファジィ設定
        self.fuzzy_threshold: float = 0.1
        self.membership_functions: Dict[str, MembershipFunction] = {}
        
        logger.info(f"ファジィ決定木 {self.tree_id} を初期化")
    
    def fit(self, X: Union[List[Dict[str, Any]], np.ndarray], 
            y: Union[List[str], np.ndarray],
            feature_names: Optional[List[str]] = None,
            class_names: Optional[List[str]] = None) -> 'FuzzyDecisionTree':
        """決定木の学習"""
        
        start_time = time.time()
        
        try:
            # データの前処理
            X_processed, y_processed = self._preprocess_data(X, y)
            
            # メタデータの設定
            self.n_samples = len(X_processed)
            self.n_features = len(X_processed[0]) if X_processed else 0
            
            if feature_names:
                self.feature_names = feature_names
            else:
                self.feature_names = [f"feature_{i}" for i in range(self.n_features)]
            
            if class_names:
                self.class_names = class_names
            else:
                self.class_names = list(set(y_processed))
            
            self.n_classes = len(self.class_names)
            
            # 決定木の構築
            from core.decision_tree.builder import FuzzyTreeBuilder, BuilderConfig
            
            builder_config = BuilderConfig(
                max_depth=self.max_depth,
                fuzzy_threshold=self.fuzzy_threshold
            )
            
            builder = FuzzyTreeBuilder(builder_config)
            self.root = builder.build(X_processed, y_processed, self.feature_names)
            
            # メトリクスの計算
            self._calculate_metrics()
            
            # 学習完了
            self.is_trained = True
            training_time = time.time() - start_time
            self.metrics.training_time = training_time
            
            # 学習履歴の記録
            self.training_history.append({
                "timestamp": time.time(),
                "n_samples": self.n_samples,
                "n_features": self.n_features,
                "training_time": training_time,
                "metrics": self.metrics
            })
            
            logger.info(f"決定木学習完了: {training_time:.3f}秒")
            
        except Exception as e:
            logger.error(f"決定木学習エラー: {e}")
            raise
        
        return self
    
    def predict(self, X: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Union[PredictionResult, List[PredictionResult]]:
        """予測実行"""
        
        if not self.is_trained:
            raise ValueError("決定木が学習されていません")
        
        if not self.root:
            raise ValueError("決定木のルートノードが存在しません")
        
        # 単一サンプルの場合
        if isinstance(X, dict):
            return self._predict_single(X)
        
        # 複数サンプルの場合
        return [self._predict_single(sample) for sample in X]
    
    def _predict_single(self, sample: Dict[str, Any]) -> PredictionResult:
        """単一サンプルの予測"""
        
        start_time = time.time()
        
        try:
            # ルートノードからの予測
            prediction_data = self.root.predict(sample)
            
            prediction_time = time.time() - start_time
            self.metrics.prediction_count += 1
            
            # 平均予測時間の更新
            if self.metrics.prediction_count > 0:
                self.metrics.average_prediction_time = (
                    (self.metrics.average_prediction_time * (self.metrics.prediction_count - 1) + prediction_time) /
                    self.metrics.prediction_count
                )
            
            # PredictionResultの構築
            result = PredictionResult(
                predicted_class=prediction_data.get("predicted_class", "unknown"),
                confidence=prediction_data.get("confidence", 0.0),
                class_probabilities=prediction_data.get("class_probabilities", {}),
                prediction_path=prediction_data.get("path", []),
                activated_rules=prediction_data.get("activated_rules", []),
                node_memberships=prediction_data.get("membership_values", {}),
                prediction_time=prediction_time,
                tree_depth=len(prediction_data.get("path", [])),
                uncertainty=1.0 - prediction_data.get("confidence", 0.0)
            )
            
            return result
            
        except Exception as e:
            logger.error(f"予測エラー: {e}")
            # フォールバック予測
            return PredictionResult(
                predicted_class=self.class_names[0] if self.class_names else "unknown",
                confidence=0.0,
                class_probabilities={},
                prediction_path=[],
                activated_rules=[],
                node_memberships={},
                prediction_time=time.time() - start_time,
                tree_depth=0,
                uncertainty=1.0
            )
    
    def _preprocess_data(self, X: Any, y: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
        """データの前処理"""
        
        # 辞書形式のリストに変換
        if isinstance(X, np.ndarray):
            if not self.feature_names:
                self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            X_processed = []
            for row in X:
                sample = {name: float(value) for name, value in zip(self.feature_names, row)}
                X_processed.append(sample)
        elif isinstance(X, list) and all(isinstance(item, dict) for item in X):
            X_processed = X
        else:
            raise ValueError("Xは辞書のリストまたはnumpy配列である必要があります")
        
        # ターゲットを文字列リストに変換
        if isinstance(y, np.ndarray):
            y_processed = [str(item) for item in y]
        elif isinstance(y, list):
            y_processed = [str(item) for item in y]
        else:
            raise ValueError("yはリストまたはnumpy配列である必要があります")
        
        return X_processed, y_processed
    
    def _calculate_metrics(self) -> None:
        """メトリクスの計算"""
        
        if not self.root:
            return
        
        # ノード数のカウント
        def count_nodes(node: FuzzyTreeNode) -> Tuple[int, int, int]:
            total = 1
            internal = 0
            leaf = 0
            
            if node.is_leaf():
                leaf = 1
            else:
                internal = 1
                for child in getattr(node, 'children', {}).values():
                    child_total, child_internal, child_leaf = count_nodes(child)
                    total += child_total
                    internal += child_internal
                    leaf += child_leaf
            
            return total, internal, leaf
        
        total_nodes, internal_nodes, leaf_nodes = count_nodes(self.root)
        
        # 深度の計算
        def calculate_depth(node: FuzzyTreeNode, current_depth: int = 0) -> Tuple[int, List[int]]:
            depths = [current_depth]
            max_depth = current_depth
            
            if not node.is_leaf():
                for child in getattr(node, 'children', {}).values():
                    child_max_depth, child_depths = calculate_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_max_depth)
                    depths.extend(child_depths)
            
            return max_depth, depths
        
        max_depth, all_depths = calculate_depth(self.root)
        average_depth = np.mean(all_depths) if all_depths else 0.0
        
        # メトリクスの更新
        self.metrics.total_nodes = total_nodes
        self.metrics.internal_nodes = internal_nodes
        self.metrics.leaf_nodes = leaf_nodes
        self.metrics.max_depth = max_depth
        self.metrics.average_depth = average_depth
        
        # 複雑度スコア（ノード数を正規化）
        self.metrics.complexity_score = total_nodes / max(self.n_samples, 1)
        
        logger.debug(f"メトリクス計算完了: {total_nodes}ノード, 最大深度{max_depth}")
    
    def get_rules(self) -> List[str]:
        """決定ルールを文字列で取得"""
        
        if not self.root:
            return []
        
        rules = []
        
        def extract_rules(node: FuzzyTreeNode, conditions: List[str] = None) -> None:
            if conditions is None:
                conditions = []
            
            if node.is_leaf():
                # 葉ノードに到達：ルールを生成
                if conditions:
                    rule = f"IF {' AND '.join(conditions)} THEN class = {getattr(node, 'predicted_class', 'unknown')}"
                    rules.append(rule)
            else:
                # 内部ノード：条件を追加して子ノードに進む
                split_condition = getattr(node, 'split_condition', None)
                if split_condition:
                    for branch_name, child in getattr(node, 'children', {}).items():
                        new_condition = f"{split_condition.feature} IS {split_condition.linguistic_value}"
                        extract_rules(child, conditions + [new_condition])
        
        extract_rules(self.root)
        return rules
    
    def get_feature_importance(self) -> Dict[str, float]:
        """特徴量重要度を計算"""
        
        if not self.root:
            return {}
        
        importance = defaultdict(float)
        
        def calculate_importance(node: FuzzyTreeNode, sample_ratio: float = 1.0) -> None:
            if not node.is_leaf():
                split_condition = getattr(node, 'split_condition', None)
                if split_condition:
                    # 情報利得に基づく重要度（簡易版）
                    feature_importance = sample_ratio * getattr(node, 'purity', 0.0)
                    importance[split_condition.feature] += feature_importance
                    
                    # 子ノードの重要度を再帰計算
                    for child in getattr(node, 'children', {}).values():
                        child_ratio = sample_ratio * (getattr(child, 'samples_count', 1) / max(getattr(node, 'samples_count', 1), 1))
                        calculate_importance(child, child_ratio)
        
        calculate_importance(self.root)
        
        # 正規化
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return dict(importance)
    
    def to_dict(self) -> Dict[str, Any]:
        """決定木を辞書形式で出力"""
        
        return {
            "tree_id": self.tree_id,
            "max_depth": self.max_depth,
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "n_samples": self.n_samples,
            "is_trained": self.is_trained,
            "metrics": {
                "total_nodes": self.metrics.total_nodes,
                "max_depth": self.metrics.max_depth,
                "training_accuracy": self.metrics.training_accuracy,
                "complexity_score": self.metrics.complexity_score
            },
            "rules": self.get_rules(),
            "feature_importance": self.get_feature_importance()
        }
    
    def save(self, filepath: str) -> None:
        """決定木をファイルに保存"""
        
        data = self.to_dict()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"決定木を保存: {filepath}")
    
    def load(self, filepath: str) -> None:
        """決定木をファイルから読み込み"""
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 基本情報の復元
        self.tree_id = data["tree_id"]
        self.max_depth = data["max_depth"]
        self.feature_names = data["feature_names"]
        self.class_names = data["class_names"]
        self.n_features = data["n_features"]
        self.n_classes = data["n_classes"]
        self.n_samples = data["n_samples"]
        self.is_trained = data["is_trained"]
        
        logger.info(f"決定木を読み込み: {filepath}")

# 後方互換性のためのエイリアス
EnhancedFuzzyDecisionTree = FuzzyDecisionTree

# 使用例とテスト
def test_fuzzy_decision_tree():
    """ファジィ決定木のテスト"""
    
    print("🌳 ファジィ決定木テスト開始")
    
    # サンプルデータの作成
    X = [
        {"research_intensity": 8.0, "advisor_style": 6.0, "team_work": 7.0},
        {"research_intensity": 3.0, "advisor_style": 8.0, "team_work": 5.0},
        {"research_intensity": 7.0, "advisor_style": 7.0, "team_work": 8.0},
        {"research_intensity": 4.0, "advisor_style": 5.0, "team_work": 4.0},
    ]
    
    y = ["high_match", "medium_match", "high_match", "low_match"]
    
    feature_names = ["research_intensity", "advisor_style", "team_work"]
    class_names = ["low_match", "medium_match", "high_match"]
    
    # 決定木の学習
    tree = FuzzyDecisionTree(max_depth=5)
    tree.fit(X, y, feature_names, class_names)
    
    print(f"✅ 学習完了")
    print(f"  ノード数: {tree.metrics.total_nodes}")
    print(f"  最大深度: {tree.metrics.max_depth}")
    print(f"  学習時間: {tree.metrics.training_time:.3f}秒")
    
    # 予測テスト
    test_sample = {"research_intensity": 6.0, "advisor_style": 7.0, "team_work": 6.0}
    prediction = tree.predict(test_sample)
    
    print(f"\n📊 予測結果:")
    print(f"  予測クラス: {prediction.predicted_class}")
    print(f"  信頼度: {prediction.confidence:.3f}")
    print(f"  予測経路: {' -> '.join(prediction.prediction_path)}")
    
    # ルール表示
    rules = tree.get_rules()
    print(f"\n📋 抽出ルール数: {len(rules)}")
    for i, rule in enumerate(rules[:3]):  # 最初の3ルールを表示
        print(f"  ルール{i+1}: {rule}")
    
    # 特徴量重要度
    importance = tree.get_feature_importance()
    print(f"\n🎯 特徴量重要度:")
    for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feature}: {imp:.3f}")
    
    print("✅ ファジィ決定木テスト完了")

if __name__ == "__main__":
    test_fuzzy_decision_tree()