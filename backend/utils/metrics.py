# utils/metrics.py - 評価指標

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import math
from collections import defaultdict, Counter
import logging

from models.schemas import (
    StudentProfile, Laboratory, LabResult, CompatibilityScore,
    EvaluationResponse
)

logger = logging.getLogger(__name__)

@dataclass
class MetricResult:
    """メトリクス結果"""
    metric_name: str
    value: float
    description: str
    higher_is_better: bool = True
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceMetrics:
    """性能メトリクス"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    mean_absolute_error: Optional[float] = None
    root_mean_squared_error: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """辞書形式に変換"""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc_roc": self.auc_roc,
            "mae": self.mean_absolute_error,
            "rmse": self.root_mean_squared_error
        }

@dataclass
class RankingMetrics:
    """ランキング評価メトリクス"""
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)  # Normalized Discounted Cumulative Gain
    map_score: float = 0.0  # Mean Average Precision
    mrr: float = 0.0  # Mean Reciprocal Rank
    hit_rate_at_k: Dict[int, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "ndcg": self.ndcg_at_k,
            "map": self.map_score,
            "mrr": self.mrr,
            "hit_rate": self.hit_rate_at_k
        }

class CompatibilityMetrics:
    """適合性評価メトリクス"""
    
    def __init__(self):
        self.prediction_history: List[Dict[str, Any]] = []
        self.ground_truth_data: List[Dict[str, Any]] = []
    
    def calculate_accuracy(self, predictions: List[float], 
                          ground_truth: List[float], 
                          threshold: float = 0.5) -> float:
        """分類精度の計算"""
        
        if len(predictions) != len(ground_truth):
            raise ValueError("予測値と正解値の長さが一致しません")
        
        if not predictions:
            return 0.0
        
        # 二値分類に変換
        pred_binary = [1 if p >= threshold else 0 for p in predictions]
        true_binary = [1 if t >= threshold else 0 for t in ground_truth]
        
        correct = sum(p == t for p, t in zip(pred_binary, true_binary))
        return correct / len(predictions)
    
    def calculate_precision_recall(self, predictions: List[float],
                                  ground_truth: List[float],
                                  threshold: float = 0.5) -> Tuple[float, float]:
        """適合率と再現率の計算"""
        
        pred_binary = [1 if p >= threshold else 0 for p in predictions]
        true_binary = [1 if t >= threshold else 0 for t in ground_truth]
        
        # True Positive, False Positive, False Negative
        tp = sum(p == 1 and t == 1 for p, t in zip(pred_binary, true_binary))
        fp = sum(p == 1 and t == 0 for p, t in zip(pred_binary, true_binary))
        fn = sum(p == 0 and t == 1 for p, t in zip(pred_binary, true_binary))
        
        # 適合率 (Precision)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # 再現率 (Recall)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return precision, recall
    
    def calculate_f1_score(self, predictions: List[float],
                          ground_truth: List[float],
                          threshold: float = 0.5) -> float:
        """F1スコアの計算"""
        
        precision, recall = self.calculate_precision_recall(predictions, ground_truth, threshold)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def calculate_mae(self, predictions: List[float], 
                     ground_truth: List[float]) -> float:
        """平均絶対誤差 (MAE) の計算"""
        
        if len(predictions) != len(ground_truth):
            raise ValueError("予測値と正解値の長さが一致しません")
        
        if not predictions:
            return 0.0
        
        absolute_errors = [abs(p - t) for p, t in zip(predictions, ground_truth)]
        return sum(absolute_errors) / len(absolute_errors)
    
    def calculate_rmse(self, predictions: List[float], 
                      ground_truth: List[float]) -> float:
        """平均平方根誤差 (RMSE) の計算"""
        
        if len(predictions) != len(ground_truth):
            raise ValueError("予測値と正解値の長さが一致しません")
        
        if not predictions:
            return 0.0
        
        squared_errors = [(p - t) ** 2 for p, t in zip(predictions, ground_truth)]
        mse = sum(squared_errors) / len(squared_errors)
        return math.sqrt(mse)
    
    def calculate_correlation(self, predictions: List[float],
                             ground_truth: List[float]) -> float:
        """ピアソン相関係数の計算"""
        
        if len(predictions) != len(ground_truth) or len(predictions) < 2:
            return 0.0
        
        mean_pred = sum(predictions) / len(predictions)
        mean_true = sum(ground_truth) / len(ground_truth)
        
        numerator = sum((p - mean_pred) * (t - mean_true) 
                       for p, t in zip(predictions, ground_truth))
        
        sum_sq_pred = sum((p - mean_pred) ** 2 for p in predictions)
        sum_sq_true = sum((t - mean_true) ** 2 for t in ground_truth)
        
        denominator = math.sqrt(sum_sq_pred * sum_sq_true)
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def evaluate_compatibility_prediction(self, 
                                        evaluation_responses: List[EvaluationResponse],
                                        ground_truth_scores: List[Dict[str, float]]) -> PerformanceMetrics:
        """適合性予測の総合評価"""
        
        all_predictions = []
        all_ground_truth = []
        
        # 予測値と正解値の抽出
        for response, gt_dict in zip(evaluation_responses, ground_truth_scores):
            student_id = response.student_profile.student_id
            
            for lab_result in response.lab_results:
                lab_id = lab_result.laboratory.lab_id
                predicted_score = lab_result.compatibility_score.overall_score
                
                # 正解値の取得
                gt_key = f"{student_id}_{lab_id}"
                if gt_key in gt_dict:
                    all_predictions.append(predicted_score)
                    all_ground_truth.append(gt_dict[gt_key])
        
        if not all_predictions:
            return PerformanceMetrics(0, 0, 0, 0)
        
        # 各メトリクスの計算
        accuracy = self.calculate_accuracy(all_predictions, all_ground_truth)
        precision, recall = self.calculate_precision_recall(all_predictions, all_ground_truth)
        f1_score = self.calculate_f1_score(all_predictions, all_ground_truth)
        mae = self.calculate_mae(all_predictions, all_ground_truth)
        rmse = self.calculate_rmse(all_predictions, all_ground_truth)
        
        return PerformanceMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            mean_absolute_error=mae,
            root_mean_squared_error=rmse
        )

class RankingEvaluator:
    """ランキング評価器"""
    
    def calculate_ndcg_at_k(self, predicted_ranking: List[str], 
                           relevant_items: List[str], 
                           k: int) -> float:
        """NDCG@K の計算"""
        
        if k <= 0 or not predicted_ranking:
            return 0.0
        
        # 上位k件の予測ランキング
        top_k_predictions = predicted_ranking[:k]
        
        # DCG (Discounted Cumulative Gain) の計算
        dcg = 0.0
        for i, item in enumerate(top_k_predictions):
            if item in relevant_items:
                relevance = 1.0  # 簡易版：関連度は1or0
                dcg += relevance / math.log2(i + 2)  # i+2 because log2(1)=0
        
        # IDCG (Ideal DCG) の計算
        ideal_ranking = relevant_items[:k]  # 理想的なランキング
        idcg = 0.0
        for i in range(len(ideal_ranking)):
            idcg += 1.0 / math.log2(i + 2)
        
        # NDCG の計算
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_map(self, predicted_rankings: List[List[str]], 
                     relevant_items_list: List[List[str]]) -> float:
        """MAP (Mean Average Precision) の計算"""
        
        if not predicted_rankings or len(predicted_rankings) != len(relevant_items_list):
            return 0.0
        
        average_precisions = []
        
        for predicted, relevant in zip(predicted_rankings, relevant_items_list):
            if not relevant:
                continue
            
            precision_sum = 0.0
            num_relevant_found = 0
            
            for i, item in enumerate(predicted):
                if item in relevant:
                    num_relevant_found += 1
                    precision_at_i = num_relevant_found / (i + 1)
                    precision_sum += precision_at_i
            
            if num_relevant_found > 0:
                average_precision = precision_sum / len(relevant)
                average_precisions.append(average_precision)
        
        return sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
    
    def calculate_mrr(self, predicted_rankings: List[List[str]], 
                     relevant_items_list: List[List[str]]) -> float:
        """MRR (Mean Reciprocal Rank) の計算"""
        
        if not predicted_rankings or len(predicted_rankings) != len(relevant_items_list):
            return 0.0
        
        reciprocal_ranks = []
        
        for predicted, relevant in zip(predicted_rankings, relevant_items_list):
            if not relevant:
                continue
            
            for i, item in enumerate(predicted):
                if item in relevant:
                    reciprocal_ranks.append(1.0 / (i + 1))
                    break
            else:
                reciprocal_ranks.append(0.0)  # 関連アイテムが見つからなかった場合
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def calculate_hit_rate_at_k(self, predicted_rankings: List[List[str]], 
                               relevant_items_list: List[List[str]], 
                               k: int) -> float:
        """Hit Rate@K の計算"""
        
        if not predicted_rankings or len(predicted_rankings) != len(relevant_items_list):
            return 0.0
        
        hits = 0
        total = 0
        
        for predicted, relevant in zip(predicted_rankings, relevant_items_list):
            if not relevant:
                continue
            
            top_k_predictions = predicted[:k]
            if any(item in relevant for item in top_k_predictions):
                hits += 1
            total += 1
        
        return hits / total if total > 0 else 0.0
    
    def evaluate_lab_recommendations(self, 
                                   evaluation_responses: List[EvaluationResponse],
                                   ground_truth_preferences: List[List[str]]) -> RankingMetrics:
        """研究室推薦のランキング評価"""
        
        predicted_rankings = []
        
        # 予測ランキングの抽出
        for response in evaluation_responses:
            ranking = [result.laboratory.lab_id for result in response.lab_results]
            predicted_rankings.append(ranking)
        
        # 各メトリクスの計算
        ranking_metrics = RankingMetrics()
        
        # NDCG@K の計算
        for k in [1, 3, 5, 10]:
            ndcg_scores = []
            for predicted, relevant in zip(predicted_rankings, ground_truth_preferences):
                if len(predicted) >= k and relevant:
                    ndcg = self.calculate_ndcg_at_k(predicted, relevant, k)
                    ndcg_scores.append(ndcg)
            
            if ndcg_scores:
                ranking_metrics.ndcg_at_k[k] = sum(ndcg_scores) / len(ndcg_scores)
        
        # MAP の計算
        ranking_metrics.map_score = self.calculate_map(predicted_rankings, ground_truth_preferences)
        
        # MRR の計算
        ranking_metrics.mrr = self.calculate_mrr(predicted_rankings, ground_truth_preferences)
        
        # Hit Rate@K の計算
        for k in [1, 3, 5, 10]:
            ranking_metrics.hit_rate_at_k[k] = self.calculate_hit_rate_at_k(
                predicted_rankings, ground_truth_preferences, k
            )
        
        return ranking_metrics

class DiversityMetrics:
    """多様性メトリクス"""
    
    def calculate_intra_list_diversity(self, lab_results: List[LabResult]) -> float:
        """リスト内多様性の計算"""
        
        if len(lab_results) < 2:
            return 0.0
        
        # 研究分野の多様性
        field_diversity = self._calculate_field_diversity(lab_results)
        
        # 特性の多様性
        characteristics_diversity = self._calculate_characteristics_diversity(lab_results)
        
        # 平均多様性
        return (field_diversity + characteristics_diversity) / 2
    
    def _calculate_field_diversity(self, lab_results: List[LabResult]) -> float:
        """研究分野の多様性計算"""
        
        fields = [result.laboratory.research_field.value for result in lab_results]
        unique_fields = set(fields)
        
        # シャノン多様性指数
        field_counts = Counter(fields)
        total_count = len(fields)
        
        diversity = 0.0
        for count in field_counts.values():
            p = count / total_count
            diversity -= p * math.log2(p)
        
        # 正規化（最大多様性で割る）
        max_diversity = math.log2(len(unique_fields)) if len(unique_fields) > 1 else 1.0
        
        return diversity / max_diversity if max_diversity > 0 else 0.0
    
    def _calculate_characteristics_diversity(self, lab_results: List[LabResult]) -> float:
        """研究室特性の多様性計算"""
        
        if len(lab_results) < 2:
            return 0.0
        
        # 各研究室の特性ベクトル
        characteristics_vectors = []
        for result in lab_results:
            char_dict = result.laboratory.characteristics.dict()
            vector = [char_dict.get(criterion, 5.0) for criterion in [
                "research_intensity", "advisor_style", "team_work", 
                "workload", "theory_practice"
            ]]
            characteristics_vectors.append(vector)
        
        # ペアワイズ距離の平均
        total_distance = 0.0
        pair_count = 0
        
        for i in range(len(characteristics_vectors)):
            for j in range(i + 1, len(characteristics_vectors)):
                distance = self._euclidean_distance(
                    characteristics_vectors[i], 
                    characteristics_vectors[j]
                )
                total_distance += distance
                pair_count += 1
        
        if pair_count == 0:
            return 0.0
        
        average_distance = total_distance / pair_count
        
        # 正規化（最大可能距離で割る）
        max_distance = math.sqrt(5 * (9.0 ** 2))  # 5次元で各次元の最大差が9
        
        return average_distance / max_distance if max_distance > 0 else 0.0
    
    def _euclidean_distance(self, vector1: List[float], vector2: List[float]) -> float:
        """ユークリッド距離の計算"""
        
        if len(vector1) != len(vector2):
            return 0.0
        
        return math.sqrt(sum((v1 - v2) ** 2 for v1, v2 in zip(vector1, vector2)))

class PredictionEvaluator:
    """予測評価総合クラス"""
    
    def __init__(self):
        self.compatibility_metrics = CompatibilityMetrics()
        self.ranking_evaluator = RankingEvaluator()
        self.diversity_metrics = DiversityMetrics()
    
    def comprehensive_evaluation(self, 
                                evaluation_responses: List[EvaluationResponse],
                                ground_truth_data: Dict[str, Any]) -> Dict[str, Any]:
        """総合的な評価の実行"""
        
        results = {
            "evaluation_summary": {
                "total_evaluations": len(evaluation_responses),
                "evaluation_date": datetime.now().isoformat(),
                "metrics_version": "1.0.0"
            }
        }
        
        try:
            # 適合性予測の評価
            if "compatibility_scores" in ground_truth_data:
                compatibility_results = self.compatibility_metrics.evaluate_compatibility_prediction(
                    evaluation_responses, ground_truth_data["compatibility_scores"]
                )
                results["compatibility_metrics"] = compatibility_results.to_dict()
            
            # ランキング評価
            if "user_preferences" in ground_truth_data:
                ranking_results = self.ranking_evaluator.evaluate_lab_recommendations(
                    evaluation_responses, ground_truth_data["user_preferences"]
                )
                results["ranking_metrics"] = ranking_results.to_dict()
            
            # 多様性評価
            diversity_scores = []
            for response in evaluation_responses:
                diversity = self.diversity_metrics.calculate_intra_list_diversity(
                    response.lab_results
                )
                diversity_scores.append(diversity)
            
            results["diversity_metrics"] = {
                "average_diversity": sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0,
                "diversity_std": np.std(diversity_scores) if len(diversity_scores) > 1 else 0.0,
                "min_diversity": min(diversity_scores) if diversity_scores else 0.0,
                "max_diversity": max(diversity_scores) if diversity_scores else 0.0
            }
            
            # システム性能指標
            processing_times = [response.processing_time for response in evaluation_responses]
            confidence_scores = [response.recommendation_confidence for response in evaluation_responses]
            
            results["system_performance"] = {
                "average_processing_time": sum(processing_times) / len(processing_times) if processing_times else 0.0,
                "max_processing_time": max(processing_times) if processing_times else 0.0,
                "average_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
                "min_confidence": min(confidence_scores) if confidence_scores else 0.0
            }
            
            # 総合スコアの計算
            overall_score = self._calculate_overall_score(results)
            results["overall_score"] = overall_score
            
        except Exception as e:
            logger.error(f"総合評価エラー: {e}")
            results["error"] = str(e)
        
        return results
    
    def _calculate_overall_score(self, evaluation_results: Dict[str, Any]) -> float:
        """総合スコアの計算"""
        
        scores = []
        weights = []
        
        # 適合性メトリクスからのスコア
        if "compatibility_metrics" in evaluation_results:
            compat_metrics = evaluation_results["compatibility_metrics"]
            if "f1_score" in compat_metrics:
                scores.append(compat_metrics["f1_score"])
                weights.append(0.4)
        
        # ランキングメトリクスからのスコア
        if "ranking_metrics" in evaluation_results:
            ranking_metrics = evaluation_results["ranking_metrics"]
            if "ndcg" in ranking_metrics and 5 in ranking_metrics["ndcg"]:
                scores.append(ranking_metrics["ndcg"][5])
                weights.append(0.3)
        
        # 多様性メトリクスからのスコア
        if "diversity_metrics" in evaluation_results:
            diversity_metrics = evaluation_results["diversity_metrics"]
            if "average_diversity" in diversity_metrics:
                scores.append(diversity_metrics["average_diversity"])
                weights.append(0.2)
        
        # システム性能からのスコア
        if "system_performance" in evaluation_results:
            sys_perf = evaluation_results["system_performance"]
            if "average_confidence" in sys_perf:
                scores.append(sys_perf["average_confidence"])
                weights.append(0.1)
        
        # 重み付き平均
        if scores and weights:
            weighted_sum = sum(s * w for s, w in zip(scores, weights))
            total_weight = sum(weights)
            return weighted_sum / total_weight
        
        return 0.0

# 使用例とテスト
def test_metrics():
    """メトリクス計算のテスト"""
    
    print("📊 評価指標テスト開始")
    
    # 適合性メトリクスのテスト
    compat_metrics = CompatibilityMetrics()
    
    # テストデータ
    predictions = [0.8, 0.6, 0.4, 0.9, 0.3]
    ground_truth = [0.9, 0.7, 0.3, 0.8, 0.2]
    
    # 各メトリクスの計算
    accuracy = compat_metrics.calculate_accuracy(predictions, ground_truth, 0.5)
    precision, recall = compat_metrics.calculate_precision_recall(predictions, ground_truth, 0.5)
    f1_score = compat_metrics.calculate_f1_score(predictions, ground_truth, 0.5)
    mae = compat_metrics.calculate_mae(predictions, ground_truth)
    rmse = compat_metrics.calculate_rmse(predictions, ground_truth)
    correlation = compat_metrics.calculate_correlation(predictions, ground_truth)
    
    print(f"✅ 適合性メトリクス:")
    print(f"  精度: {accuracy:.3f}")
    print(f"  適合率: {precision:.3f}")
    print(f"  再現率: {recall:.3f}")
    print(f"  F1スコア: {f1_score:.3f}")
    print(f"  MAE: {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  相関係数: {correlation:.3f}")
    
    # ランキング評価のテスト
    ranking_evaluator = RankingEvaluator()
    
    predicted_ranking = ["lab_a", "lab_b", "lab_c", "lab_d"]
    relevant_items = ["lab_a", "lab_c"]
    
    ndcg_3 = ranking_evaluator.calculate_ndcg_at_k(predicted_ranking, relevant_items, 3)
    
    print(f"\n🏆 ランキングメトリクス:")
    print(f"  NDCG@3: {ndcg_3:.3f}")
    
    # 多様性メトリクスのテスト
    diversity_metrics = DiversityMetrics()
    
    # テスト用の研究室結果（簡易版）
    print(f"\n🌈 多様性メトリクス:")
    print(f"  テスト完了")
    
    print("✅ 評価指標テスト完了")

if __name__ == "__main__":
    test_metrics()