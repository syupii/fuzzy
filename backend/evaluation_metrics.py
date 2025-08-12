# backend/evaluation_metrics.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MetricResult:
    """メトリック結果格納"""
    value: float
    normalized_value: float
    description: str
    higher_is_better: bool = True


class MultiObjectiveEvaluator:
    """多目的評価システム"""

    def __init__(self, weights: Dict[str, float] = None):
        """
        weights: 各目的の重み
        - accuracy: 予測精度
        - simplicity: モデル簡潔性
        - interpretability: 解釈可能性
        - generalization: 汎化性能
        - validity: ルール妥当性
        """
        self.weights = weights or {
            'accuracy': 0.35,
            'simplicity': 0.15,
            'interpretability': 0.20,
            'generalization': 0.20,
            'validity': 0.10
        }

        # 正規化のための統計値
        self.normalization_stats = {
            'accuracy': {'min': 0.0, 'max': 1.0},
            'simplicity': {'min': 0.0, 'max': 1.0},
            'interpretability': {'min': 0.0, 'max': 1.0},
            'generalization': {'min': 0.0, 'max': 1.0},
            'validity': {'min': 0.0, 'max': 1.0}
        }

    def evaluate_individual(self, tree_individual,
                            training_data: pd.DataFrame,
                            test_data: pd.DataFrame,
                            target_column: str = 'compatibility',
                            feature_columns: List[str] = None) -> Dict[str, MetricResult]:
        """個体の多目的評価"""

        if feature_columns is None:
            feature_columns = ['research_intensity', 'advisor_style', 'team_work',
                               'workload', 'theory_practice']

        # 各目的の評価
        results = {}

        # 1. 予測精度評価
        results['accuracy'] = self._evaluate_accuracy(
            tree_individual, training_data, test_data, target_column, feature_columns
        )

        # 2. 簡潔性評価
        results['simplicity'] = self._evaluate_simplicity(tree_individual)

        # 3. 解釈可能性評価
        results['interpretability'] = self._evaluate_interpretability(
            tree_individual)

        # 4. 汎化性能評価
        results['generalization'] = self._evaluate_generalization(
            tree_individual, training_data, test_data, target_column, feature_columns
        )

        # 5. ルール妥当性評価
        results['validity'] = self._evaluate_validity(
            tree_individual, training_data, feature_columns
        )

        return results

    def _evaluate_accuracy(self, tree_individual, training_data: pd.DataFrame,
                           test_data: pd.DataFrame, target_column: str,
                           feature_columns: List[str]) -> MetricResult:
        """予測精度評価"""

        try:
            # テストデータでの予測
            X_test = test_data[feature_columns]
            y_test = test_data[target_column]

            predictions = []
            for _, row in X_test.iterrows():
                sample = row.to_dict()
                pred = tree_individual.tree.predict(sample) if hasattr(
                    tree_individual, 'tree') else 0.5
                predictions.append(pred)

            predictions = np.array(predictions)
            y_test_values = y_test.values

            # 複数の精度指標計算
            mse = mean_squared_error(y_test_values, predictions)
            mae = mean_absolute_error(y_test_values, predictions)

            # 正規化RMSE（0-1スケール）
            rmse = np.sqrt(mse)
            max_error = np.max(y_test_values) - np.min(y_test_values)
            normalized_rmse = rmse / max_error if max_error > 0 else 0.0

            # 精度スコア（1 - normalized_rmse）
            accuracy_score = max(0.0, 1.0 - normalized_rmse)

            return MetricResult(
                value=accuracy_score,
                normalized_value=accuracy_score,  # 既に0-1
                description=f"RMSE: {rmse:.3f}, MAE: {mae:.3f}, Score: {accuracy_score:.3f}",
                higher_is_better=True
            )

        except Exception as e:
            return MetricResult(
                value=0.0,
                normalized_value=0.0,
                description=f"Accuracy evaluation failed: {str(e)}",
                higher_is_better=True
            )

    def _evaluate_simplicity(self, tree_individual) -> MetricResult:
        """簡潔性評価（木の複雑度の逆数）"""

        try:
            if not hasattr(tree_individual, 'tree') or tree_individual.tree is None:
                return MetricResult(0.0, 0.0, "No tree available", True)

            # 木の複雑度計算
            total_nodes = tree_individual.tree.calculate_complexity()
            max_depth = tree_individual.tree.calculate_depth()

            # 複雑度正規化（0-1スケール）
            # 最大想定: 100ノード、10深度
            max_expected_nodes = 100
            max_expected_depth = 10

            node_score = max(0.0, 1.0 - (total_nodes / max_expected_nodes))
            depth_score = max(0.0, 1.0 - (max_depth / max_expected_depth))

            # 統合簡潔性スコア
            simplicity_score = 0.6 * node_score + 0.4 * depth_score

            return MetricResult(
                value=simplicity_score,
                normalized_value=simplicity_score,
                description=f"Nodes: {total_nodes}, Depth: {max_depth}, Score: {simplicity_score:.3f}",
                higher_is_better=True
            )

        except Exception as e:
            return MetricResult(
                value=0.0,
                normalized_value=0.0,
                description=f"Simplicity evaluation failed: {str(e)}",
                higher_is_better=True
            )

    def _evaluate_interpretability(self, tree_individual) -> MetricResult:
        """解釈可能性評価"""

        try:
            if not hasattr(tree_individual, 'tree') or tree_individual.tree is None:
                return MetricResult(0.0, 0.0, "No tree available", True)

            tree = tree_individual.tree

            # 解釈可能性要因
            interpretability_factors = []

            # 1. メンバーシップ関数の分かりやすさ
            mf_clarity = self._evaluate_membership_clarity(tree)
            interpretability_factors.append(mf_clarity)

            # 2. 特徴使用のバランス
            feature_balance = self._evaluate_feature_balance(tree)
            interpretability_factors.append(feature_balance)

            # 3. 決定パスの論理性
            path_logic = self._evaluate_path_logic(tree)
            interpretability_factors.append(path_logic)

            # 4. ルール抽出可能性
            rule_extractability = self._evaluate_rule_extractability(tree)
            interpretability_factors.append(rule_extractability)

            # 統合解釈可能性スコア
            interpretability_score = np.mean(interpretability_factors)

            return MetricResult(
                value=interpretability_score,
                normalized_value=interpretability_score,
                description=f"MF Clarity: {mf_clarity:.2f}, Balance: {feature_balance:.2f}, Logic: {path_logic:.2f}",
                higher_is_better=True
            )

        except Exception as e:
            return MetricResult(
                value=0.0,
                normalized_value=0.0,
                description=f"Interpretability evaluation failed: {str(e)}",
                higher_is_better=True
            )

    def _evaluate_generalization(self, tree_individual, training_data: pd.DataFrame,
                                 test_data: pd.DataFrame, target_column: str,
                                 feature_columns: List[str]) -> MetricResult:
        """汎化性能評価"""

        try:
            # 訓練データでの性能
            train_performance = self._calculate_dataset_performance(
                tree_individual, training_data, target_column, feature_columns
            )

            # テストデータでの性能
            test_performance = self._calculate_dataset_performance(
                tree_individual, test_data, target_column, feature_columns
            )

            # 汎化ギャップ計算
            generalization_gap = abs(train_performance - test_performance)

            # 汎化スコア（ギャップが小さいほど良い）
            generalization_score = max(0.0, 1.0 - generalization_gap)

            # テスト性能も考慮
            combined_score = 0.7 * generalization_score + 0.3 * test_performance

            return MetricResult(
                value=combined_score,
                normalized_value=combined_score,
                description=f"Train: {train_performance:.3f}, Test: {test_performance:.3f}, Gap: {generalization_gap:.3f}",
                higher_is_better=True
            )

        except Exception as e:
            return MetricResult(
                value=0.0,
                normalized_value=0.0,
                description=f"Generalization evaluation failed: {str(e)}",
                higher_is_better=True
            )

    def _evaluate_validity(self, tree_individual, training_data: pd.DataFrame,
                           feature_columns: List[str]) -> MetricResult:
        """ルール妥当性評価"""

        try:
            if not hasattr(tree_individual, 'tree') or tree_individual.tree is None:
                return MetricResult(0.0, 0.0, "No tree available", True)

            tree = tree_individual.tree

            # 妥当性要因
            validity_factors = []

            # 1. メンバーシップ関数の重複度
            overlap_penalty = self._evaluate_membership_overlap(tree)
            validity_factors.append(1.0 - overlap_penalty)

            # 2. 特徴値範囲の適切性
            range_appropriateness = self._evaluate_range_appropriateness(
                tree, training_data, feature_columns)
            validity_factors.append(range_appropriateness)

            # 3. 論理的一貫性
            logical_consistency = self._evaluate_logical_consistency(tree)
            validity_factors.append(logical_consistency)

            # 4. ルールカバレッジ
            rule_coverage = self._evaluate_rule_coverage(
                tree, training_data, feature_columns)
            validity_factors.append(rule_coverage)

            # 統合妥当性スコア
            validity_score = np.mean(validity_factors)

            return MetricResult(
                value=validity_score,
                normalized_value=validity_score,
                description=f"Overlap: {1-overlap_penalty:.2f}, Range: {range_appropriateness:.2f}, Coverage: {rule_coverage:.2f}",
                higher_is_better=True
            )

        except Exception as e:
            return MetricResult(
                value=0.0,
                normalized_value=0.0,
                description=f"Validity evaluation failed: {str(e)}",
                higher_is_better=True
            )

    def _calculate_dataset_performance(self, tree_individual, data: pd.DataFrame,
                                       target_column: str, feature_columns: List[str]) -> float:
        """データセットでの性能計算"""

        X = data[feature_columns]
        y = data[target_column]

        predictions = []
        for _, row in X.iterrows():
            sample = row.to_dict()
            pred = tree_individual.tree.predict(sample) if hasattr(
                tree_individual, 'tree') else 0.5
            predictions.append(pred)

        predictions = np.array(predictions)
        y_values = y.values

        # RMSE based performance
        mse = mean_squared_error(y_values, predictions)
        rmse = np.sqrt(mse)
        max_error = np.max(y_values) - np.min(y_values)

        performance = max(0.0, 1.0 - (rmse / max_error)
                          ) if max_error > 0 else 0.0
        return performance

    def _evaluate_membership_clarity(self, tree) -> float:
        """メンバーシップ関数の明確性評価"""

        clarity_scores = []

        def evaluate_node_clarity(node):
            if node.is_leaf:
                return

            for mf in node.membership_functions.values():
                # パラメータの適切性
                params = mf.parameters
                if len(params) >= 2:
                    # パラメータ範囲の適切性
                    param_range = max(params) - min(params)
                    clarity = min(1.0, param_range / 10.0)  # 0-10範囲を想定
                    clarity_scores.append(clarity)

            for child in node.children.values():
                evaluate_node_clarity(child)

        evaluate_node_clarity(tree)

        return np.mean(clarity_scores) if clarity_scores else 0.5

    def _evaluate_feature_balance(self, tree) -> float:
        """特徴使用バランス評価"""

        feature_usage = {}

        def count_feature_usage(node):
            if not node.is_leaf and node.feature_name:
                feature_usage[node.feature_name] = feature_usage.get(
                    node.feature_name, 0) + 1

            for child in node.children.values():
                count_feature_usage(child)

        count_feature_usage(tree)

        if not feature_usage:
            return 0.0

        # 使用頻度の分散（小さいほど均等）
        usage_values = list(feature_usage.values())
        variance = np.var(usage_values)
        max_variance = np.mean(usage_values) ** 2  # 最大想定分散

        balance_score = max(0.0, 1.0 - (variance / max_variance)
                            ) if max_variance > 0 else 1.0
        return balance_score

    def _evaluate_path_logic(self, tree) -> float:
        """決定パスの論理性評価"""

        # 簡易的な論理性評価
        # より複雑な実装も可能

        path_count = 0
        valid_paths = 0

        def validate_paths(node, depth=0):
            nonlocal path_count, valid_paths

            if node.is_leaf:
                path_count += 1
                # リーフノードに到達できるパスは有効
                valid_paths += 1
                return

            if depth > 10:  # 無限再帰防止
                return

            for child in node.children.values():
                validate_paths(child, depth + 1)

        validate_paths(tree)

        logic_score = valid_paths / path_count if path_count > 0 else 0.0
        return logic_score

    def _evaluate_rule_extractability(self, tree) -> float:
        """ルール抽出可能性評価"""

        # ルールとして解釈可能なパス数
        extractable_rules = 0
        total_paths = 0

        def count_extractable_rules(node, conditions=[]):
            nonlocal extractable_rules, total_paths

            if node.is_leaf:
                total_paths += 1
                # 条件が明確なルールは抽出可能
                if len(conditions) > 0 and len(conditions) <= 5:  # 5条件以下
                    extractable_rules += 1
                return

            for label, child in node.children.items():
                new_conditions = conditions + \
                    [f"{node.feature_name} is {label}"]
                count_extractable_rules(child, new_conditions)

        count_extractable_rules(tree)

        extractability = extractable_rules / total_paths if total_paths > 0 else 0.0
        return extractability

    def _evaluate_membership_overlap(self, tree) -> float:
        """メンバーシップ関数重複度評価"""

        overlap_penalties = []

        def evaluate_node_overlap(node):
            if node.is_leaf or len(node.membership_functions) < 2:
                return

            mfs = list(node.membership_functions.values())

            # 各ペアの重複計算
            for i in range(len(mfs)):
                for j in range(i+1, len(mfs)):
                    overlap = self._calculate_membership_overlap(
                        mfs[i], mfs[j])
                    overlap_penalties.append(overlap)

            for child in node.children.values():
                evaluate_node_overlap(child)

        evaluate_node_overlap(tree)

        return np.mean(overlap_penalties) if overlap_penalties else 0.0

    def _calculate_membership_overlap(self, mf1, mf2) -> float:
        """2つのメンバーシップ関数の重複計算"""

        # 簡易的な重複計算（サンプル点での評価）
        test_points = np.linspace(0, 10, 100)

        overlaps = []
        for point in test_points:
            val1 = mf1.membership(point)
            val2 = mf2.membership(point)
            overlap = min(val1, val2)
            overlaps.append(overlap)

        # 平均重複度
        avg_overlap = np.mean(overlaps)
        return avg_overlap

    def _evaluate_range_appropriateness(self, tree, training_data: pd.DataFrame,
                                        feature_columns: List[str]) -> float:
        """特徴値範囲の適切性評価"""

        appropriateness_scores = []

        # 各特徴の実際の範囲
        feature_ranges = {}
        for feature in feature_columns:
            if feature in training_data.columns:
                feature_ranges[feature] = {
                    'min': training_data[feature].min(),
                    'max': training_data[feature].max()
                }

        def evaluate_node_ranges(node):
            if node.is_leaf or not node.feature_name:
                return

            if node.feature_name in feature_ranges:
                actual_range = feature_ranges[node.feature_name]

                for mf in node.membership_functions.values():
                    if len(mf.parameters) >= 2:
                        mf_min = min(mf.parameters)
                        mf_max = max(mf.parameters)

                        # メンバーシップ関数が実際のデータ範囲をカバーしているか
                        coverage = self._calculate_range_coverage(
                            actual_range['min'], actual_range['max'],
                            mf_min, mf_max
                        )
                        appropriateness_scores.append(coverage)

            for child in node.children.values():
                evaluate_node_ranges(child)

        evaluate_node_ranges(tree)

        return np.mean(appropriateness_scores) if appropriateness_scores else 0.5

    def _calculate_range_coverage(self, data_min: float, data_max: float,
                                  mf_min: float, mf_max: float) -> float:
        """範囲カバレッジ計算"""

        # データ範囲
        data_range = data_max - data_min
        if data_range == 0:
            return 1.0

        # 重複範囲
        overlap_min = max(data_min, mf_min)
        overlap_max = min(data_max, mf_max)
        overlap_range = max(0, overlap_max - overlap_min)

        # カバレッジ率
        coverage = overlap_range / data_range
        return min(1.0, coverage)

    def _evaluate_logical_consistency(self, tree) -> float:
        """論理的一貫性評価"""

        # 簡易的な一貫性チェック
        consistency_score = 1.0

        def check_consistency(node, depth=0):
            nonlocal consistency_score

            if node.is_leaf or depth > 10:
                return

            # 子ノードが存在しないメンバーシップ関数は一貫性違反
            for label in node.membership_functions.keys():
                if label not in node.children:
                    consistency_score *= 0.9  # ペナルティ

            for child in node.children.values():
                check_consistency(child, depth + 1)

        check_consistency(tree)

        return max(0.0, consistency_score)

    def _evaluate_rule_coverage(self, tree, training_data: pd.DataFrame,
                                feature_columns: List[str]) -> float:
        """ルールカバレッジ評価"""

        total_samples = len(training_data)
        covered_samples = 0

        for _, row in training_data.iterrows():
            sample = row[feature_columns].to_dict()

            # サンプルが木で処理可能かチェック
            try:
                prediction = tree.predict(sample)
                if prediction is not None:
                    covered_samples += 1
            except:
                continue

        coverage = covered_samples / total_samples if total_samples > 0 else 0.0
        return coverage

    def calculate_weighted_fitness(self, metric_results: Dict[str, MetricResult]) -> float:
        """重み付き総合適応度計算"""

        total_fitness = 0.0
        total_weight = 0.0

        for metric_name, result in metric_results.items():
            if metric_name in self.weights:
                weight = self.weights[metric_name]
                value = result.normalized_value

                total_fitness += weight * value
                total_weight += weight

        return total_fitness / total_weight if total_weight > 0 else 0.0

    def update_normalization_stats(self, all_results: List[Dict[str, MetricResult]]):
        """正規化統計の更新"""

        for metric_name in self.weights.keys():
            values = []
            for result_dict in all_results:
                if metric_name in result_dict:
                    values.append(result_dict[metric_name].value)

            if values:
                self.normalization_stats[metric_name] = {
                    'min': min(values),
                    'max': max(values)
                }


class PerformanceProfiler:
    """性能プロファイリング"""

    @staticmethod
    def profile_prediction_performance(tree_individual, test_samples: List[Dict]) -> Dict:
        """予測性能プロファイリング"""

        import time

        times = []
        predictions = []

        for sample in test_samples:
            start_time = time.time()

            try:
                prediction = tree_individual.tree.predict(
                    sample) if hasattr(tree_individual, 'tree') else 0.5
                predictions.append(prediction)
            except Exception as e:
                predictions.append(None)

            end_time = time.time()
            times.append(end_time - start_time)

        return {
            'avg_prediction_time': np.mean(times),
            'max_prediction_time': np.max(times),
            'min_prediction_time': np.min(times),
            'total_time': np.sum(times),
            'successful_predictions': sum(1 for p in predictions if p is not None),
            'failed_predictions': sum(1 for p in predictions if p is None),
            'success_rate': sum(1 for p in predictions if p is not None) / len(predictions)
        }

    @staticmethod
    def analyze_convergence_behavior(fitness_history: List[float]) -> Dict:
        """収束挙動分析"""

        if len(fitness_history) < 2:
            return {'convergence_detected': False}

        # 収束検出
        recent_window = min(10, len(fitness_history) // 4)
        recent_values = fitness_history[-recent_window:]

        variance = np.var(recent_values)
        convergence_threshold = 0.001

        # 改善停止検出
        improvement_window = min(5, len(fitness_history) // 2)
        no_improvement_count = 0

        for i in range(len(fitness_history) - improvement_window, len(fitness_history)):
            if i > 0 and fitness_history[i] <= fitness_history[i-1] + 0.001:
                no_improvement_count += 1

        return {
            'convergence_detected': variance < convergence_threshold,
            'recent_variance': variance,
            'no_improvement_generations': no_improvement_count,
            'final_fitness': fitness_history[-1],
            'max_fitness': max(fitness_history),
            'improvement_rate': (fitness_history[-1] - fitness_history[0]) / len(fitness_history) if len(fitness_history) > 1 else 0.0
        }
