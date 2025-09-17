# core/decision_tree/tree.py - 完全13項目対応 ファジィ決定木システム

import math
import random
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict, Counter
import json

logger = logging.getLogger(__name__)

class SplitCriterion(str, Enum):
    """分割基準"""
    FUZZY_GAIN = "fuzzy_gain"
    GINI_IMPURITY = "gini_impurity" 
    ENTROPY = "entropy"
    WEIGHTED_VARIANCE = "weighted_variance"

class NodeType(str, Enum):
    """ノードタイプ"""
    LEAF = "leaf"
    INTERNAL = "internal"
    FUZZY = "fuzzy"

@dataclass
class TreeConfig:
    """決定木設定（13項目完全対応）"""
    
    # 基本設定
    max_depth: int = 8
    min_samples_split: int = 5
    min_samples_leaf: int = 3
    
    # 13項目完全対応設定
    criteria_count: int = 13
    use_all_criteria: bool = True
    criteria_weights: Dict[str, float] = field(default_factory=dict)
    
    # ファジィ設定
    fuzzy_threshold: float = 0.2
    membership_overlap: float = 0.3
    linguistic_values: List[str] = field(default_factory=lambda: ["low", "medium", "high"])
    
    # 分割設定
    split_criterion: SplitCriterion = SplitCriterion.FUZZY_GAIN
    max_features: Optional[int] = None  # None = 全特徴量使用
    random_state: Optional[int] = None
    
    # 剪定設定
    enable_pruning: bool = True
    pruning_threshold: float = 0.01
    cross_validation_folds: int = 5

@dataclass
class FuzzySplit:
    """ファジィ分割条件"""
    
    feature: str
    linguistic_value: str
    threshold: float
    membership_function: Callable[[float], float]
    information_gain: float = 0.0
    
    def evaluate(self, value: float) -> float:
        """分割条件評価（メンバーシップ度）"""
        try:
            return self.membership_function(value)
        except Exception as e:
            logger.warning(f"ファジィ分割評価エラー ({self.feature}): {e}")
            return 0.0

@dataclass
class NodeStatistics:
    """ノード統計情報"""
    
    samples_count: int = 0
    class_distribution: Dict[str, int] = field(default_factory=dict)
    impurity: float = 0.0
    prediction_confidence: float = 0.0
    criteria_importance: Dict[str, float] = field(default_factory=dict)
    
    def add_sample(self, class_label: str, weight: float = 1.0):
        """サンプル追加"""
        self.samples_count += 1
        if class_label not in self.class_distribution:
            self.class_distribution[class_label] = 0
        self.class_distribution[class_label] += weight
    
    def get_majority_class(self) -> Tuple[str, float]:
        """多数クラス取得"""
        if not self.class_distribution:
            return "medium", 0.0
        
        majority_class = max(self.class_distribution.items(), key=lambda x: x[1])
        confidence = majority_class[1] / sum(self.class_distribution.values())
        return majority_class[0], confidence
    
    def calculate_impurity(self, criterion: SplitCriterion = SplitCriterion.ENTROPY) -> float:
        """不純度計算"""
        if not self.class_distribution or sum(self.class_distribution.values()) == 0:
            return 0.0
        
        total = sum(self.class_distribution.values())
        probabilities = [count / total for count in self.class_distribution.values()]
        
        if criterion == SplitCriterion.ENTROPY:
            return -sum(p * math.log2(p + 1e-10) for p in probabilities if p > 0)
        elif criterion == SplitCriterion.GINI_IMPURITY:
            return 1.0 - sum(p ** 2 for p in probabilities)
        else:  # WEIGHTED_VARIANCE
            return sum(p * (1 - p) for p in probabilities)

class FuzzyTreeNode(ABC):
    """ファジィ決定木ノード抽象基底クラス"""
    
    def __init__(self, node_id: str, depth: int = 0):
        self.node_id = node_id
        self.depth = depth
        self.statistics = NodeStatistics()
        self.node_type: NodeType = NodeType.INTERNAL
        
        # 13項目対応のメタデータ
        self.criteria_used: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        
    @abstractmethod
    def predict(self, instance: Dict[str, float]) -> Dict[str, Any]:
        """予測実行"""
        pass
    
    @abstractmethod
    def get_prediction_path(self, instance: Dict[str, float]) -> List[str]:
        """予測パス取得"""
        pass
    
    def update_feature_importance(self, feature: str, importance: float):
        """特徴量重要度更新"""
        if feature not in self.feature_importance:
            self.feature_importance[feature] = 0.0
        self.feature_importance[feature] += importance

class FuzzyLeafNode(FuzzyTreeNode):
    """ファジィ葉ノード（13項目対応）"""
    
    def __init__(self, node_id: str, predicted_class: str, confidence: float, depth: int = 0):
        super().__init__(node_id, depth)
        self.predicted_class = predicted_class
        self.confidence = confidence
        self.node_type = NodeType.LEAF
        
        # 13項目の寄与度
        self.criteria_contributions: Dict[str, float] = {}
        
    def predict(self, instance: Dict[str, float]) -> Dict[str, Any]:
        """葉ノードでの予測"""
        
        # 13項目各基準の寄与度計算
        total_contribution = 0.0
        criteria_analysis = {}
        
        for criterion, contribution in self.criteria_contributions.items():
            if criterion in instance:
                criteria_analysis[criterion] = {
                    "value": instance[criterion],
                    "contribution": contribution,
                    "impact": contribution * instance[criterion] / 10.0  # 正規化
                }
                total_contribution += criteria_analysis[criterion]["impact"]
        
        return {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "depth": self.depth,
            "total_contribution": total_contribution,
            "criteria_analysis": criteria_analysis,
            "samples_count": self.statistics.samples_count,
            "class_distribution": self.statistics.class_distribution
        }
    
    def get_prediction_path(self, instance: Dict[str, float]) -> List[str]:
        """予測パス取得"""
        return [f"LEAF({self.node_id}): {self.predicted_class} (confidence: {self.confidence:.3f})"]
    
    def set_criteria_contributions(self, contributions: Dict[str, float]):
        """基準寄与度設定"""
        self.criteria_contributions = contributions

class FuzzyInternalNode(FuzzyTreeNode):
    """ファジィ内部ノード（13項目対応）"""
    
    def __init__(self, node_id: str, split_feature: str, depth: int = 0):
        super().__init__(node_id, depth)
        self.split_feature = split_feature
        self.node_type = NodeType.FUZZY
        
        # ファジィ分割
        self.fuzzy_splits: Dict[str, FuzzySplit] = {}  # linguistic_value -> split
        self.children: Dict[str, FuzzyTreeNode] = {}   # linguistic_value -> child_node
        
        # 13項目対応の分割統計
        self.split_statistics = {
            "information_gain": 0.0,
            "samples_before": 0,
            "samples_after": {},
            "impurity_reduction": 0.0
        }
        
    def add_fuzzy_split(self, linguistic_value: str, fuzzy_split: FuzzySplit, 
                       child_node: FuzzyTreeNode):
        """ファジィ分割追加"""
        self.fuzzy_splits[linguistic_value] = fuzzy_split
        self.children[linguistic_value] = child_node
        
    def predict(self, instance: Dict[str, float]) -> Dict[str, Any]:
        """ファジィ内部ノードでの予測"""
        
        if self.split_feature not in instance:
            # 分割特徴量がない場合のフォールバック
            return self._fallback_prediction(instance)
        
        feature_value = instance[self.split_feature]
        
        # 各分割への所属度計算
        branch_memberships = {}
        for linguistic_value, fuzzy_split in self.fuzzy_splits.items():
            membership = fuzzy_split.evaluate(feature_value)
            if membership > 0:
                branch_memberships[linguistic_value] = membership
        
        if not branch_memberships:
            return self._fallback_prediction(instance)
        
        # ファジィ推論による統合予測
        weighted_predictions = []
        total_weight = 0.0
        
        prediction_details = {
            "node_id": self.node_id,
            "split_feature": self.split_feature,
            "feature_value": feature_value,
            "branch_memberships": branch_memberships,
            "branch_predictions": {}
        }
        
        for linguistic_value, membership in branch_memberships.items():
            if linguistic_value in self.children:
                child_prediction = self.children[linguistic_value].predict(instance)
                
                # 重み付き予測
                weight = membership
                weighted_predictions.append({
                    "prediction": child_prediction,
                    "weight": weight,
                    "branch": linguistic_value
                })
                total_weight += weight
                
                prediction_details["branch_predictions"][linguistic_value] = {
                    "membership": membership,
                    "prediction": child_prediction
                }
        
        # 統合予測計算
        if not weighted_predictions:
            return self._fallback_prediction(instance)
        
        integrated_result = self._integrate_fuzzy_predictions(
            weighted_predictions, total_weight
        )
        
        # メタデータ追加
        integrated_result.update({
            "node_type": self.node_type.value,
            "depth": self.depth,
            "split_details": prediction_details,
            "integration_method": "fuzzy_weighted_average"
        })
        
        return integrated_result
    
    def _integrate_fuzzy_predictions(self, weighted_predictions: List[Dict[str, Any]], 
                                   total_weight: float) -> Dict[str, Any]:
        """ファジィ予測統合"""
        
        # クラス別重み付き投票
        class_votes = defaultdict(float)
        confidence_sum = 0.0
        total_contribution = 0.0
        
        # 統合された基準分析
        integrated_criteria = defaultdict(lambda: {"total_impact": 0.0, "weight_sum": 0.0})
        
        for pred_data in weighted_predictions:
            prediction = pred_data["prediction"]
            weight = pred_data["weight"]
            
            # クラス投票
            predicted_class = prediction.get("predicted_class", "medium")
            pred_confidence = prediction.get("confidence", 0.5)
            
            class_votes[predicted_class] += weight * pred_confidence
            confidence_sum += weight * pred_confidence
            
            # 寄与度統合
            if "total_contribution" in prediction:
                total_contribution += weight * prediction["total_contribution"]
            
            # 基準別分析統合
            if "criteria_analysis" in prediction:
                for criterion, analysis in prediction["criteria_analysis"].items():
                    impact = analysis.get("impact", 0.0)
                    integrated_criteria[criterion]["total_impact"] += weight * impact
                    integrated_criteria[criterion]["weight_sum"] += weight
        
        # 最終予測クラス決定
        if not class_votes:
            final_class = "medium"
            final_confidence = 0.5
        else:
            final_class = max(class_votes.items(), key=lambda x: x[1])[0]
            final_confidence = class_votes[final_class] / total_weight
        
        # 統合基準分析
        final_criteria_analysis = {}
        for criterion, data in integrated_criteria.items():
            if data["weight_sum"] > 0:
                final_criteria_analysis[criterion] = {
                    "integrated_impact": data["total_impact"] / data["weight_sum"],
                    "influence_weight": data["weight_sum"] / total_weight
                }
        
        return {
            "predicted_class": final_class,
            "confidence": final_confidence,
            "total_contribution": total_contribution / total_weight,
            "criteria_analysis": final_criteria_analysis,
            "class_votes": dict(class_votes),
            "prediction_method": "fuzzy_integration"
        }
    
    def _fallback_prediction(self, instance: Dict[str, float]) -> Dict[str, Any]:
        """フォールバック予測"""
        majority_class, confidence = self.statistics.get_majority_class()
        
        return {
            "predicted_class": majority_class,
            "confidence": confidence,
            "node_id": self.node_id,
            "node_type": "fallback",
            "depth": self.depth,
            "fallback_reason": f"missing_feature_{self.split_feature}",
            "samples_count": self.statistics.samples_count
        }
    
    def get_prediction_path(self, instance: Dict[str, float]) -> List[str]:
        """予測パス取得"""
        
        path = [f"INTERNAL({self.node_id}): split on {self.split_feature}"]
        
        if self.split_feature not in instance:
            path.append("→ FALLBACK: missing feature")
            return path
        
        feature_value = instance[self.split_feature]
        
        # 最大メンバーシップ分岐を選択
        max_membership = 0.0
        selected_branch = None
        
        for linguistic_value, fuzzy_split in self.fuzzy_splits.items():
            membership = fuzzy_split.evaluate(feature_value)
            if membership > max_membership:
                max_membership = membership
                selected_branch = linguistic_value
        
        if selected_branch and selected_branch in self.children:
            path.append(f"→ {selected_branch} (membership: {max_membership:.3f})")
            path.extend(self.children[selected_branch].get_prediction_path(instance))
        else:
            path.append("→ NO VALID BRANCH")
        
        return path

class Complete13CriteriaFuzzyDecisionTree:
    """完全13項目対応ファジィ決定木"""
    
    # 完全13項目評価基準
    COMPLETE_CRITERIA = [
        "research_intensity", "advisor_style", "team_work", "workload", "theory_practice",
        "research_field_match", "skill_development", "lab_atmosphere", "flexibility",
        "publication_opportunity", "interdisciplinary", "communication_style", "innovation_risk"
    ]
    
    # 基準別重要度
    CRITERIA_IMPORTANCE = {
        # 基本項目（高重要度）
        "research_intensity": 1.3,
        "advisor_style": 1.2,
        "team_work": 1.1,
        "workload": 1.1,
        "theory_practice": 1.2,
        
        # 拡張項目（中〜高重要度）
        "research_field_match": 1.5,  # 最重要
        "skill_development": 1.0,
        "lab_atmosphere": 0.9,
        "flexibility": 0.9,
        "publication_opportunity": 1.1,
        
        # 特殊項目（調整重要度）
        "interdisciplinary": 0.8,
        "communication_style": 0.9,
        "innovation_risk": 1.0
    }
    
    def __init__(self, config: TreeConfig):
        self.config = config
        self.root: Optional[FuzzyTreeNode] = None
        self.feature_importances: Dict[str, float] = {}
        self.tree_statistics = {
            "total_nodes": 0,
            "leaf_nodes": 0,
            "internal_nodes": 0,
            "max_depth_reached": 0,
            "training_samples": 0,
            "criteria_usage": defaultdict(int)
        }
        
        # ランダムシード設定
        if config.random_state:
            random.seed(config.random_state)
            np.random.seed(config.random_state)
        
        # 基準重み設定
        if not config.criteria_weights:
            config.criteria_weights = self.CRITERIA_IMPORTANCE.copy()
        
        logger.info(f"完全13項目対応ファジィ決定木初期化: 最大深度{config.max_depth}")
    
    def fit(self, training_data: List[Dict[str, Any]], target_column: str = "compatibility_class"):
        """学習実行（13項目完全対応）"""
        
        if not training_data:
            raise ValueError("訓練データが空です")
        
        logger.info(f"ファジィ決定木学習開始: {len(training_data)}サンプル")
        
        # データ前処理
        processed_data = self._preprocess_training_data(training_data, target_column)
        
        # 根ノード構築
        self.root = self._build_tree(
            data=processed_data,
            depth=0,
            node_id="root",
            available_features=self._get_available_features(processed_data)
        )
        
        # 特徴量重要度計算
        self._calculate_feature_importances()
        
        # 後剪定（有効な場合）
        if self.config.enable_pruning:
            self._prune_tree()
        
        # 統計更新
        self.tree_statistics["training_samples"] = len(training_data)
        
        logger.info(f"学習完了: {self.tree_statistics['total_nodes']}ノード, "
                   f"深度{self.tree_statistics['max_depth_reached']}")
        
        return self
    
    def _preprocess_training_data(self, training_data: List[Dict[str, Any]], 
                                target_column: str) -> List[Dict[str, Any]]:
        """訓練データ前処理"""
        
        processed_data = []
        
        for sample in training_data:
            processed_sample = {}
            
            # 13項目基準の抽出・正規化
            for criterion in self.COMPLETE_CRITERIA:
                if criterion in sample:
                    value = float(sample[criterion])
                    # 1-10の範囲にクリップ
                    processed_sample[criterion] = max(1.0, min(10.0, value))
                else:
                    # 欠損値の処理
                    processed_sample[criterion] = 5.0  # デフォルト値
            
            # ターゲットクラスの処理
            if target_column in sample:
                target_value = sample[target_column]
                
                # 数値から言語値への変換
                if isinstance(target_value, (int, float)):
                    target_value = float(target_value)
                    if target_value >= 0.8:
                        processed_sample["class"] = "very_high"
                    elif target_value >= 0.6:
                        processed_sample["class"] = "high"
                    elif target_value >= 0.4:
                        processed_sample["class"] = "medium"
                    elif target_value >= 0.2:
                        processed_sample["class"] = "low"
                    else:
                        processed_sample["class"] = "very_low"
                else:
                    processed_sample["class"] = str(target_value)
            else:
                processed_sample["class"] = "medium"  # デフォルトクラス
            
            processed_data.append(processed_sample)
        
        logger.info(f"データ前処理完了: {len(processed_data)}サンプル")
        return processed_data
    
    def _get_available_features(self, data: List[Dict[str, Any]]) -> List[str]:
        """利用可能特徴量取得"""
        
        if self.config.use_all_criteria:
            return self.COMPLETE_CRITERIA.copy()
        
        # データに存在する特徴量のみ
        available_features = set()
        for sample in data:
            for feature in self.COMPLETE_CRITERIA:
                if feature in sample:
                    available_features.add(feature)
        
        features = list(available_features)
        
        # 特徴量数制限
        if self.config.max_features and len(features) > self.config.max_features:
            # 重要度に基づく選択
            weighted_features = [
                (f, self.config.criteria_weights.get(f, 1.0)) for f in features
            ]
            weighted_features.sort(key=lambda x: x[1], reverse=True)
            features = [f for f, _ in weighted_features[:self.config.max_features]]
        
        return features
    
    def _build_tree(self, data: List[Dict[str, Any]], depth: int, node_id: str,
                   available_features: List[str]) -> FuzzyTreeNode:
        """決定木構築（再帰的）"""
        
        self.tree_statistics["total_nodes"] += 1
        self.tree_statistics["max_depth_reached"] = max(
            self.tree_statistics["max_depth_reached"], depth
        )
        
        # 統計計算
        node_stats = NodeStatistics()
        class_counts = Counter()
        
        for sample in data:
            class_label = sample.get("class", "medium")
            node_stats.add_sample(class_label)
            class_counts[class_label] += 1
        
        # 停止条件チェック
        if self._should_stop_splitting(data, depth, available_features, class_counts):
            return self._create_leaf_node(node_id, depth, class_counts, data)
        
        # 最適分割探索
        best_split = self._find_best_split(data, available_features)
        
        if not best_split:
            return self._create_leaf_node(node_id, depth, class_counts, data)
        
        # 内部ノード作成
        internal_node = FuzzyInternalNode(node_id, best_split["feature"], depth)
        internal_node.statistics = node_stats
        
        self.tree_statistics["internal_nodes"] += 1
        self.tree_statistics["criteria_usage"][best_split["feature"]] += 1
        
        # ファジィ分割による子ノード構築
        remaining_features = [f for f in available_features if f != best_split["feature"]]
        
        for linguistic_value, split_info in best_split["splits"].items():
            # 分割データ
            branch_data = self._split_data_fuzzy(
                data, best_split["feature"], linguistic_value, split_info["membership_func"]
            )
            
            if not branch_data:
                # 空の分岐の場合はリーフノード作成
                child_node = self._create_leaf_node(
                    f"{node_id}_{linguistic_value}", depth + 1, class_counts, []
                )
            else:
                # 再帰的構築
                child_node = self._build_tree(
                    branch_data, depth + 1, f"{node_id}_{linguistic_value}", remaining_features
                )
            
            # ファジィ分割追加
            fuzzy_split = FuzzySplit(
                feature=best_split["feature"],
                linguistic_value=linguistic_value,
                threshold=split_info.get("threshold", 5.0),
                membership_function=split_info["membership_func"],
                information_gain=split_info.get("information_gain", 0.0)
            )
            
            internal_node.add_fuzzy_split(linguistic_value, fuzzy_split, child_node)
        
        return internal_node
    
    def _should_stop_splitting(self, data: List[Dict[str, Any]], depth: int,
                             available_features: List[str], class_counts: Counter) -> bool:
        """分割停止判定"""
        
        # 深度制限
        if depth >= self.config.max_depth:
            return True
        
        # サンプル数制限
        if len(data) < self.config.min_samples_split:
            return True
        
        # 純度チェック（単一クラス）
        if len(class_counts) <= 1:
            return True
        
        # 利用可能特徴量がない
        if not available_features:
            return True
        
        # 最小分割サンプル数
        max_class_count = max(class_counts.values())
        if len(data) - max_class_count < self.config.min_samples_leaf:
            return True
        
        return False
    
    def _find_best_split(self, data: List[Dict[str, Any]], 
                        available_features: List[str]) -> Optional[Dict[str, Any]]:
        """最適分割探索（13項目対応）"""
        
        best_split = None
        best_gain = -float('inf')
        
        # 現在の不純度計算
        current_impurity = self._calculate_impurity(data)
        
        for feature in available_features:
            # 特徴量の値範囲を取得
            feature_values = [sample[feature] for sample in data if feature in sample]
            if not feature_values:
                continue
            
            # ファジィメンバーシップ関数による分割
            split_info = self._create_fuzzy_splits(feature, feature_values)
            
            # 情報利得計算
            weighted_impurity = 0.0
            total_samples = len(data)
            split_details = {}
            
            for linguistic_value, membership_func in split_info.items():
                # 分割後のデータ
                branch_data = self._split_data_fuzzy(data, feature, linguistic_value, membership_func)
                
                if branch_data:
                    branch_impurity = self._calculate_impurity(branch_data)
                    branch_weight = len(branch_data) / total_samples
                    weighted_impurity += branch_weight * branch_impurity
                    
                    split_details[linguistic_value] = {
                        "membership_func": membership_func,
                        "samples": len(branch_data),
                        "impurity": branch_impurity,
                        "threshold": self._get_threshold_from_membership(membership_func)
                    }
            
            # 情報利得
            information_gain = current_impurity - weighted_impurity
            
            # 重要度重みを適用
            feature_weight = self.config.criteria_weights.get(feature, 1.0)
            weighted_gain = information_gain * feature_weight
            
            if weighted_gain > best_gain:
                best_gain = weighted_gain
                best_split = {
                    "feature": feature,
                    "information_gain": information_gain,
                    "weighted_gain": weighted_gain,
                    "splits": split_details,
                    "current_impurity": current_impurity
                }
        
        return best_split if best_gain > self.config.fuzzy_threshold else None
    
    def _create_fuzzy_splits(self, feature: str, feature_values: List[float]) -> Dict[str, Callable]:
        """ファジィ分割作成"""
        
        if not feature_values:
            return {}
        
        min_val = min(feature_values)
        max_val = max(feature_values)
        range_val = max_val - min_val
        
        if range_val == 0:
            # 値が全て同じ場合
            return {"medium": lambda x: 1.0 if x == min_val else 0.0}
        
        # 三角型メンバーシップ関数
        # Low: [min, min, (min+max)/2]
        # Medium: [min, (min+max)/2, max]  
        # High: [(min+max)/2, max, max]
        
        mid_val = (min_val + max_val) / 2
        overlap = self.config.membership_overlap * range_val
        
        def low_membership(x):
            if x <= min_val:
                return 1.0
            elif x <= mid_val - overlap:
                return max(0.0, 1.0 - (x - min_val) / (mid_val - overlap - min_val))
            else:
                return 0.0
        
        def medium_membership(x):
            if x <= min_val + overlap:
                return max(0.0, (x - min_val) / overlap)
            elif x <= max_val - overlap:
                return 1.0
            elif x <= max_val:
                return max(0.0, 1.0 - (x - (max_val - overlap)) / overlap)
            else:
                return 0.0
        
        def high_membership(x):
            if x <= mid_val + overlap:
                return 0.0
            elif x <= max_val:
                return max(0.0, (x - (mid_val + overlap)) / (max_val - mid_val - overlap))
            else:
                return 1.0
        
        return {
            "low": low_membership,
            "medium": medium_membership,
            "high": high_membership
        }
    
    def _split_data_fuzzy(self, data: List[Dict[str, Any]], feature: str, 
                         linguistic_value: str, membership_func: Callable) -> List[Dict[str, Any]]:
        """ファジィ分割によるデータ分割"""
        
        split_data = []
        
        for sample in data:
            if feature in sample:
                membership_degree = membership_func(sample[feature])
                # 閾値以上のメンバーシップ度を持つサンプルを含める
                if membership_degree >= self.config.fuzzy_threshold:
                    # サンプルにメンバーシップ度を付加
                    split_sample = sample.copy()
                    split_sample[f"membership_{linguistic_value}"] = membership_degree
                    split_data.append(split_sample)
        
        return split_data
    
    def _get_threshold_from_membership(self, membership_func: Callable) -> float:
        """メンバーシップ関数から代表閾値を計算"""
        
        # 簡易実装：1-10の範囲で最大メンバーシップ値を持つ点を探索
        max_membership = 0.0
        best_threshold = 5.0
        
        for x in np.linspace(1, 10, 100):
            membership = membership_func(x)
            if membership > max_membership:
                max_membership = membership
                best_threshold = x
        
        return best_threshold
    
    def _calculate_impurity(self, data: List[Dict[str, Any]]) -> float:
        """不純度計算"""
        
        if not data:
            return 0.0
        
        class_counts = Counter(sample.get("class", "medium") for sample in data)
        total = len(data)
        
        if self.config.split_criterion == SplitCriterion.ENTROPY:
            return -sum((count / total) * math.log2(count / total) 
                       for count in class_counts.values() if count > 0)
        
        elif self.config.split_criterion == SplitCriterion.GINI_IMPURITY:
            return 1.0 - sum((count / total) ** 2 for count in class_counts.values())
        
        else:  # WEIGHTED_VARIANCE or FUZZY_GAIN
            probabilities = [count / total for count in class_counts.values()]
            return sum(p * (1 - p) for p in probabilities)
    
    def _create_leaf_node(self, node_id: str, depth: int, class_counts: Counter,
                         data: List[Dict[str, Any]]) -> FuzzyLeafNode:
        """葉ノード作成"""
        
        self.tree_statistics["leaf_nodes"] += 1
        
        if not class_counts:
            predicted_class = "medium"
            confidence = 0.5
        else:
            predicted_class = class_counts.most_common(1)[0][0]
            confidence = class_counts[predicted_class] / sum(class_counts.values())
        
        leaf_node = FuzzyLeafNode(node_id, predicted_class, confidence, depth)
        
        # 統計設定
        leaf_node.statistics.samples_count = len(data)
        for class_label, count in class_counts.items():
            leaf_node.statistics.class_distribution[class_label] = count
        
        # 13項目基準の寄与度計算
        criteria_contributions = self._calculate_criteria_contributions(data)
        leaf_node.set_criteria_contributions(criteria_contributions)
        
        return leaf_node
    
    def _calculate_criteria_contributions(self, data: List[Dict[str, Any]]) -> Dict[str, float]:
        """基準寄与度計算（葉ノード用）"""
        
        if not data:
            return {criterion: 0.0 for criterion in self.COMPLETE_CRITERIA}
        
        contributions = {}
        
        for criterion in self.COMPLETE_CRITERIA:
            values = [sample.get(criterion, 5.0) for sample in data]
            if values:
                # 値の分散を寄与度とする（簡易実装）
                mean_val = sum(values) / len(values)
                variance = sum((v - mean_val) ** 2 for v in values) / len(values)
                
                # 正規化（0-1範囲）
                normalized_contribution = min(1.0, variance / 10.0)
                
                # 重要度重みを適用
                weight = self.config.criteria_weights.get(criterion, 1.0)
                contributions[criterion] = normalized_contribution * weight
            else:
                contributions[criterion] = 0.0
        
        return contributions
    
    def _calculate_feature_importances(self):
        """特徴量重要度計算"""
        
        if not self.root:
            return
        
        self.feature_importances = {criterion: 0.0 for criterion in self.COMPLETE_CRITERIA}
        total_samples = self.tree_statistics["training_samples"]
        
        self._traverse_importance_calculation(self.root, total_samples)
        
        # 正規化
        total_importance = sum(self.feature_importances.values())
        if total_importance > 0:
            self.feature_importances = {
                feature: importance / total_importance
                for feature, importance in self.feature_importances.items()
            }
    
    def _traverse_importance_calculation(self, node: FuzzyTreeNode, total_samples: int):
        """重要度計算の再帰的トラバース"""
        
        if isinstance(node, FuzzyInternalNode):
            # 内部ノードの場合：分割による情報利得を重要度とする
            feature = node.split_feature
            samples_ratio = node.statistics.samples_count / total_samples
            
            # 分割統計から情報利得を取得
            information_gain = node.split_statistics.get("information_gain", 0.0)
            self.feature_importances[feature] += information_gain * samples_ratio
            
            # 子ノードを再帰的に処理
            for child in node.children.values():
                self._traverse_importance_calculation(child, total_samples)
    
    def _prune_tree(self):
        """後剪定実行"""
        
        if not self.root or not self.config.enable_pruning:
            return
        
        logger.info("後剪定開始...")
        
        # 簡易剪定：小さな情報利得のノードを削除
        pruned_count = self._prune_recursive(self.root)
        
        logger.info(f"剪定完了: {pruned_count}ノードを剪定")
    
    def _prune_recursive(self, node: FuzzyTreeNode) -> int:
        """再帰的剪定"""
        
        if isinstance(node, FuzzyLeafNode):
            return 0
        
        if isinstance(node, FuzzyInternalNode):
            pruned_count = 0
            
            # 子ノードを再帰的に剪定
            for child in node.children.values():
                pruned_count += self._prune_recursive(child)
            
            # 自分自身の剪定判定
            if self._should_prune_node(node):
                # リーフノードに変換（実装簡略化のため、ここでは剪定カウントのみ）
                pruned_count += 1
            
            return pruned_count
        
        return 0
    
    def _should_prune_node(self, node: FuzzyInternalNode) -> bool:
        """ノード剪定判定"""
        
        # 情報利得が閾値未満の場合に剪定
        information_gain = node.split_statistics.get("information_gain", 0.0)
        return information_gain < self.config.pruning_threshold
    
    def predict(self, instance: Dict[str, float]) -> Dict[str, Any]:
        """予測実行（13項目対応）"""
        
        if not self.root:
            return {
                "predicted_class": "medium",
                "confidence": 0.0,
                "error": "モデルが学習されていません"
            }
        
        try:
            # 入力の前処理
            processed_instance = self._preprocess_instance(instance)
            
            # 予測実行
            prediction = self.root.predict(processed_instance)
            
            # メタデータ追加
            prediction.update({
                "model_type": "fuzzy_decision_tree_13_criteria",
                "tree_depth": self.tree_statistics["max_depth_reached"],
                "total_nodes": self.tree_statistics["total_nodes"],
                "criteria_used": len([c for c in self.COMPLETE_CRITERIA if c in processed_instance])
            })
            
            return prediction
            
        except Exception as e:
            logger.error(f"予測エラー: {e}")
            return {
                "predicted_class": "medium",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def predict_with_explanation(self, instance: Dict[str, float]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """説明付き予測"""
        
        prediction = self.predict(instance)
        
        if not self.root:
            return prediction, {"error": "モデル未学習"}
        
        # 予測パス取得
        processed_instance = self._preprocess_instance(instance)
        prediction_path = self.root.get_prediction_path(processed_instance)
        
        # 詳細説明構築
        explanation = {
            "prediction_path": prediction_path,
            "feature_importances": self.feature_importances,
            "criteria_analysis": self._analyze_instance_criteria(processed_instance),
            "tree_statistics": self.tree_statistics,
            "model_info": {
                "criteria_count": len(self.COMPLETE_CRITERIA),
                "total_nodes": self.tree_statistics["total_nodes"],
                "max_depth": self.tree_statistics["max_depth_reached"]
            }
        }
        
        return prediction, explanation
    
    def _preprocess_instance(self, instance: Dict[str, float]) -> Dict[str, float]:
        """インスタンス前処理"""
        
        processed = {}
        
        for criterion in self.COMPLETE_CRITERIA:
            if criterion in instance:
                value = float(instance[criterion])
                processed[criterion] = max(1.0, min(10.0, value))  # 1-10にクリップ
            else:
                processed[criterion] = 5.0  # デフォルト値
        
        return processed
    
    def _analyze_instance_criteria(self, instance: Dict[str, float]) -> Dict[str, Any]:
        """インスタンス基準分析"""
        
        analysis = {}
        
        for criterion in self.COMPLETE_CRITERIA:
            value = instance.get(criterion, 5.0)
            importance = self.feature_importances.get(criterion, 0.0)
            weight = self.config.criteria_weights.get(criterion, 1.0)
            
            # 値の言語的解釈
            if value <= 3:
                linguistic_value = "low"
            elif value <= 7:
                linguistic_value = "medium"
            else:
                linguistic_value = "high"
            
            analysis[criterion] = {
                "value": value,
                "linguistic_value": linguistic_value,
                "importance": importance,
                "weight": weight,
                "contribution": value * importance * weight
            }
        
        return analysis
    
    def get_tree_info(self) -> Dict[str, Any]:
        """決定木情報取得"""
        
        return {
            "configuration": {
                "max_depth": self.config.max_depth,
                "min_samples_split": self.config.min_samples_split,
                "min_samples_leaf": self.config.min_samples_leaf,
                "split_criterion": self.config.split_criterion.value,
                "criteria_count": self.config.criteria_count,
                "use_all_criteria": self.config.use_all_criteria
            },
            "statistics": self.tree_statistics,
            "feature_importances": self.feature_importances,
            "criteria_usage": dict(self.tree_statistics["criteria_usage"]),
            "model_size": {
                "total_nodes": self.tree_statistics["total_nodes"],
                "leaf_nodes": self.tree_statistics["leaf_nodes"],
                "internal_nodes": self.tree_statistics["internal_nodes"],
                "average_depth": self.tree_statistics["max_depth_reached"]
            }
        }

# バックワード互換性のためのクラス別名
EnhancedFuzzyDecisionTree = Complete13CriteriaFuzzyDecisionTree
FuzzyDecisionTree = Complete13CriteriaFuzzyDecisionTree
CompleteFuzzyDecisionTree = Complete13CriteriaFuzzyDecisionTree

# ファクトリー関数
def create_complete_fuzzy_decision_tree(max_depth: int = 8, 
                                      min_samples_split: int = 5,
                                      **kwargs) -> Complete13CriteriaFuzzyDecisionTree:
    """完全13項目対応ファジィ決定木作成"""
    
    config = TreeConfig(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criteria_count=13,
        use_all_criteria=True,
        **kwargs
    )
    
    return Complete13CriteriaFuzzyDecisionTree(config)

# エクスポート用リスト
__all__ = [
    "Complete13CriteriaFuzzyDecisionTree",
    "EnhancedFuzzyDecisionTree",
    "FuzzyDecisionTree", 
    "CompleteFuzzyDecisionTree",
    "TreeConfig",
    "FuzzyTreeNode",
    "FuzzyLeafNode",
    "FuzzyInternalNode",
    "FuzzySplit",
    "NodeStatistics",
    "SplitCriterion",
    "NodeType",
    "create_complete_fuzzy_decision_tree"
]