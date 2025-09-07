# utils/metrics.py - 評価指標計算ユーティリティ

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings

from models.schemas import StudentProfile, Laboratory, LabResult, CompatibilityScore

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """システム性能指標"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: Optional[float] = None
    mae: float = 0.0  # Mean Absolute Error
    rmse: float = 0.0  # Root Mean Square Error
    correlation: float = 0.0

@dataclass
class CompatibilityMetrics:
    """適合性評価指標"""
    overall_compatibility: float
    field_compatibility: float
    criteria_compatibility: float
    diversity_score: float
    consistency_score: float
    confidence: float

@dataclass
class SystemMetrics:
    """システム全体の指標"""
    prediction_accuracy: float
    average_response_time: float
    user_satisfaction: float
    algorithm_efficiency: float
    model_stability: float

class MetricsCalculator:
    """評価指標計算クラス"""
    
    def __init__(self):
        self.calculation_history = []
        self.benchmark_scores = self._initialize_benchmarks()
    
    def calculate_compatibility_metrics(self, student: StudentProfile, 
                                      lab: Laboratory, 
                                      predicted_score: float) -> CompatibilityMetrics:
        """適合性指標の計算"""
        
        # 分野適合性
        field_compatibility = self._calculate_field_compatibility(student, lab)
        
        # 評価基準適合性
        criteria_compatibility = self._calculate_criteria_compatibility(student, lab)
        
        # 多様性スコア
        diversity_score = self._calculate_diversity_score(student, lab)
        
        # 一貫性スコア
        consistency_score = self._calculate_consistency_score(student, lab)
        
        # 信頼度
        confidence = self._calculate_prediction_confidence(
            field_compatibility, criteria_compatibility, diversity_score
        )
        
        # 総合適合性
        overall_compatibility = (
            field_compatibility * 0.4 +
            criteria_compatibility * 0.3 +
            diversity_score * 0.2 +
            consistency_score * 0.1
        )
        
        return CompatibilityMetrics(
            overall_compatibility=overall_compatibility,
            field_compatibility=field_compatibility,
            criteria_compatibility=criteria_compatibility,
            diversity_score=diversity_score,
            consistency_score=consistency_score,
            confidence=confidence
        )
    
    def _calculate_field_compatibility(self, student: StudentProfile, lab: Laboratory) -> float:
        """分野適合性計算"""
        
        student_fields = {fi.field_id: fi for fi in student.field_interests}
        total_score = 0.0
        matched_count = 0
        
        for field_id in lab.research_fields:
            if field_id in student_fields:
                field_interest = student_fields[field_id]
                
                # 興味度・経験・重要度の統合
                field_score = (
                    field_interest.interest_level * 0.5 +
                    field_interest.experience_level * 0.3 +
                    field_interest.importance_level * 0.2
                ) / 10.0
                
                total_score += field_score
                matched_count += 1
        
        return total_score / matched_count if matched_count > 0 else 0.0
    
    def _calculate_criteria_compatibility(self, student: StudentProfile, lab: Laboratory) -> float:
        """評価基準適合性計算"""
        
        student_criteria = student.evaluation_criteria.dict()
        lab_features = lab.features.dict()
        
        similarities = []
        
        for criterion in student_criteria.keys():
            if criterion in lab_features:
                student_val = student_criteria[criterion]
                lab_val = lab_features[criterion]
                
                # ガウシアン類似度
                distance = abs(student_val - lab_val)
                similarity = np.exp(-(distance ** 2) / (2 * 2.0 ** 2))
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_diversity_score(self, student: StudentProfile, lab: Laboratory) -> float:
        """多様性スコア計算"""
        
        # 学際性の評価
        interdisciplinary_score = student.evaluation_criteria.interdisciplinary / 10.0
        
        # 研究室の分野数による多様性
        field_diversity = min(1.0, len(lab.research_fields) / 3.0)
        
        # 学生の分野選択多様性
        student_diversity = min(1.0, len(student.field_interests) / 5.0)
        
        return (interdisciplinary_score + field_diversity + student_diversity) / 3.0
    
    def _calculate_consistency_score(self, student: StudentProfile, lab: Laboratory) -> float:
        """一貫性スコア計算"""
        
        # 研究スタイルの一貫性
        student_criteria = student.evaluation_criteria.dict()
        lab_features = lab.features.dict()
        
        style_criteria = ["research_intensity", "theory_practice", "team_work"]
        consistency_scores = []
        
        for criterion in style_criteria:
            if criterion in student_criteria and criterion in lab_features:
                student_val = student_criteria[criterion]
                lab_val = lab_features[criterion]
                
                # 一貫性（距離の逆数）
                distance = abs(student_val - lab_val)
                consistency = 1.0 - (distance / 10.0)
                consistency_scores.append(max(0, consistency))
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _calculate_prediction_confidence(self, field_comp: float, criteria_comp: float, 
                                       diversity: float) -> float:
        """予測信頼度計算"""
        
        # 各要素の重み付き統合
        confidence = (
            field_comp * 0.5 +
            criteria_comp * 0.3 +
            diversity * 0.2
        )
        
        # 分散による信頼度調整
        components = [field_comp, criteria_comp, diversity]
        variance = np.var(components)
        
        # 分散が小さいほど信頼度が高い
        confidence_adjustment = 1.0 - min(0.3, variance)
        
        return confidence * confidence_adjustment
    
    def evaluate_ranking_performance(self, predictions: List[LabResult], 
                                   ground_truth: List[float]) -> PerformanceMetrics:
        """ランキング性能評価"""
        
        if len(predictions) != len(ground_truth):
            raise ValueError("予測結果と正解データの長さが一致しません")
        
        # 予測スコア抽出
        predicted_scores = [result.compatibility.overall_score for result in predictions]
        
        # 基本的な回帰指標
        mae = np.mean(np.abs(np.array(predicted_scores) - np.array(ground_truth)))
        rmse = np.sqrt(np.mean((np.array(predicted_scores) - np.array(ground_truth)) ** 2))
        
        # 相関係数
        try:
            correlation, _ = stats.pearsonr(predicted_scores, ground_truth)
            if np.isnan(correlation):
                correlation = 0.0
        except:
            correlation = 0.0
        
        # ランキング精度（上位k個の一致率）
        ranking_accuracy = self._calculate_ranking_accuracy(predicted_scores, ground_truth)
        
        # 分類性能（閾値ベース）
        classification_metrics = self._calculate_classification_metrics(
            predicted_scores, ground_truth
        )
        
        return PerformanceMetrics(
            accuracy=ranking_accuracy,
            precision=classification_metrics["precision"],
            recall=classification_metrics["recall"],
            f1_score=classification_metrics["f1"],
            mae=mae,
            rmse=rmse,
            correlation=correlation
        )
    
    def _calculate_ranking_accuracy(self, predicted: List[float], 
                                  ground_truth: List[float], k: int = 5) -> float:
        """ランキング精度計算（上位k個の一致率）"""
        
        # 上位k個のインデックス取得
        pred_top_k = set(np.argsort(predicted)[-k:])
        true_top_k = set(np.argsort(ground_truth)[-k:])
        
        # 一致率計算
        intersection = len(pred_top_k & true_top_k)
        return intersection / k
    
    def _calculate_classification_metrics(self, predicted: List[float], 
                                        ground_truth: List[float], 
                                        threshold: float = 7.0) -> Dict[str, float]:
        """分類性能指標計算"""
        
        # 閾値による二値分類
        pred_binary = [1 if score >= threshold else 0 for score in predicted]
        true_binary = [1 if score >= threshold else 0 for score in ground_truth]
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                precision = precision_score(true_binary, pred_binary, zero_division=0)
                recall = recall_score(true_binary, pred_binary, zero_division=0)
                f1 = f1_score(true_binary, pred_binary, zero_division=0)
                
                return {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                }
        except:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    def calculate_system_performance(self, performance_data: Dict[str, Any]) -> SystemMetrics:
        """システム性能指標計算"""
        
        # 予測精度
        prediction_accuracy = performance_data.get("accuracy", 0.0)
        
        # 応答時間
        response_times = performance_data.get("response_times", [])
        avg_response_time = np.mean(response_times) if response_times else 0.0
        
        # ユーザー満足度（仮想的な計算）
        user_satisfaction = self._calculate_user_satisfaction(performance_data)
        
        # アルゴリズム効率
        algorithm_efficiency = self._calculate_algorithm_efficiency(performance_data)
        
        # モデル安定性
        model_stability = self._calculate_model_stability(performance_data)
        
        return SystemMetrics(
            prediction_accuracy=prediction_accuracy,
            average_response_time=avg_response_time,
            user_satisfaction=user_satisfaction,
            algorithm_efficiency=algorithm_efficiency,
            model_stability=model_stability
        )
    
    def _calculate_user_satisfaction(self, data: Dict[str, Any]) -> float:
        """ユーザー満足度計算（仮想）"""
        
        # 精度とレスポンス時間からの推定
        accuracy = data.get("accuracy", 0.0)
        response_time = data.get("avg_response_time", 5.0)
        
        # 精度が高く、レスポンスが速いほど満足度が高い
        satisfaction = accuracy * 0.7 + max(0, 1.0 - response_time / 10.0) * 0.3
        
        return min(1.0, satisfaction)
    
    def _calculate_algorithm_efficiency(self, data: Dict[str, Any]) -> float:
        """アルゴリズム効率計算"""
        
        # 計算時間と精度のバランス
        accuracy = data.get("accuracy", 0.0)
        response_time = data.get("avg_response_time", 5.0)
        
        # 効率 = 精度 / 時間
        if response_time > 0:
            efficiency = accuracy / response_time
            return min(1.0, efficiency * 10)  # 正規化
        else:
            return 0.0
    
    def _calculate_model_stability(self, data: Dict[str, Any]) -> float:
        """モデル安定性計算"""
        
        accuracy_history = data.get("accuracy_history", [])
        
        if len(accuracy_history) < 2:
            return 0.5  # デフォルト値
        
        # 精度の分散（小さいほど安定）
        variance = np.var(accuracy_history)
        stability = 1.0 - min(1.0, variance * 10)  # 分散を0-1に正規化
        
        return max(0.0, stability)
    
    def _initialize_benchmarks(self) -> Dict[str, float]:
        """ベンチマークスコアの初期化"""
        
        return {
            "excellent_compatibility": 0.9,
            "good_compatibility": 0.7,
            "fair_compatibility": 0.5,
            "minimum_accuracy": 0.6,
            "target_response_time": 2.0,
            "minimum_confidence": 0.7
        }
    
    def compare_with_benchmark(self, metrics: PerformanceMetrics) -> Dict[str, str]:
        """ベンチマークとの比較"""
        
        comparisons = {}
        
        if metrics.accuracy >= self.benchmark_scores["excellent_compatibility"]:
            comparisons["accuracy"] = "優秀"
        elif metrics.accuracy >= self.benchmark_scores["good_compatibility"]:
            comparisons["accuracy"] = "良好"
        elif metrics.accuracy >= self.benchmark_scores["fair_compatibility"]:
            comparisons["accuracy"] = "普通"
        else:
            comparisons["accuracy"] = "改善必要"
        
        if metrics.correlation >= 0.8:
            comparisons["correlation"] = "強い相関"
        elif metrics.correlation >= 0.5:
            comparisons["correlation"] = "中程度の相関"
        else:
            comparisons["correlation"] = "弱い相関"
        
        return comparisons

class PredictionEvaluator:
    """予測評価専用クラス"""
    
    def __init__(self):
        self.evaluation_history = []
        self.metrics_calculator = MetricsCalculator()
    
    def evaluate_prediction_quality(self, student_profile: StudentProfile,
                                   prediction_results: List[LabResult]) -> Dict[str, Any]:
        """予測品質評価"""
        
        evaluation = {
            "overall_quality": 0.0,
            "ranking_quality": 0.0,
            "score_distribution": {},
            "coverage_analysis": {},
            "consistency_check": {},
            "recommendations": []
        }
        
        # スコア分布分析
        scores = [result.compatibility.overall_score for result in prediction_results]
        evaluation["score_distribution"] = {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "range": np.max(scores) - np.min(scores)
        }
        
        # ランキング品質
        evaluation["ranking_quality"] = self._evaluate_ranking_quality(scores)
        
        # カバレッジ分析
        evaluation["coverage_analysis"] = self._analyze_field_coverage(
            student_profile, prediction_results
        )
        
        # 一貫性チェック
        evaluation["consistency_check"] = self._check_prediction_consistency(
            prediction_results
        )
        
        # 総合品質
        evaluation["overall_quality"] = (
            evaluation["ranking_quality"] * 0.4 +
            evaluation["coverage_analysis"]["coverage_score"] * 0.3 +
            evaluation["consistency_check"]["consistency_score"] * 0.3
        )
        
        # 推奨事項
        evaluation["recommendations"] = self._generate_quality_recommendations(evaluation)
        
        return evaluation
    
    def _evaluate_ranking_quality(self, scores: List[float]) -> float:
        """ランキング品質評価"""
        
        if len(scores) < 2:
            return 0.0
        
        # スコアの分散（適度な分散が望ましい）
        score_std = np.std(scores)
        
        # 理想的な標準偏差（1.5-2.5の範囲）
        ideal_std = 2.0
        std_quality = 1.0 - abs(score_std - ideal_std) / ideal_std
        
        # 順序の妥当性（上位スコアが適切に分離されているか）
        sorted_scores = sorted(scores, reverse=True)
        separability = 0.0
        
        for i in range(min(5, len(sorted_scores) - 1)):
            diff = sorted_scores[i] - sorted_scores[i + 1]
            separability += diff
        
        separability /= min(5, len(sorted_scores) - 1) if len(sorted_scores) > 1 else 1
        separability = min(1.0, separability / 2.0)  # 正規化
        
        return (std_quality * 0.6 + separability * 0.4)
    
    def _analyze_field_coverage(self, student: StudentProfile, 
                               results: List[LabResult]) -> Dict[str, Any]:
        """分野カバレッジ分析"""
        
        student_field_ids = {fi.field_id for fi in student.field_interests}
        
        covered_fields = set()
        total_field_matches = 0
        
        for result in results:
            lab_fields = set(result.lab.research_fields)
            matches = student_field_ids & lab_fields
            
            covered_fields.update(matches)
            total_field_matches += len(matches)
        
        coverage_score = len(covered_fields) / len(student_field_ids) if student_field_ids else 0
        
        return {
            "coverage_score": coverage_score,
            "covered_fields": len(covered_fields),
            "total_student_fields": len(student_field_ids),
            "average_matches_per_lab": total_field_matches / len(results) if results else 0
        }
    
    def _check_prediction_consistency(self, results: List[LabResult]) -> Dict[str, Any]:
        """予測一貫性チェック"""
        
        # スコアの単調性チェック
        scores = [result.compatibility.overall_score for result in results]
        is_monotonic = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        
        # スコア分布の妥当性
        score_gaps = []
        for i in range(len(scores) - 1):
            gap = scores[i] - scores[i + 1]
            score_gaps.append(gap)
        
        avg_gap = np.mean(score_gaps) if score_gaps else 0
        gap_consistency = 1.0 - min(1.0, np.std(score_gaps) / max(avg_gap, 0.1))
        
        consistency_score = (
            (1.0 if is_monotonic else 0.5) * 0.5 +
            gap_consistency * 0.5
        )
        
        return {
            "consistency_score": consistency_score,
            "is_monotonic": is_monotonic,
            "average_score_gap": avg_gap,
            "gap_consistency": gap_consistency
        }
    
    def _generate_quality_recommendations(self, evaluation: Dict[str, Any]) -> List[str]:
        """品質改善推奨事項生成"""
        
        recommendations = []
        
        # 総合品質
        if evaluation["overall_quality"] < 0.6:
            recommendations.append("予測品質が低いです。モデルパラメータの調整を検討してください。")
        
        # スコア分布
        score_dist = evaluation["score_distribution"]
        if score_dist["std"] < 1.0:
            recommendations.append("スコア分布の分散が小さすぎます。判別力を向上させてください。")
        elif score_dist["std"] > 3.0:
            recommendations.append("スコア分布の分散が大きすぎます。安定性を向上させてください。")
        
        # カバレッジ
        coverage = evaluation["coverage_analysis"]["coverage_score"]
        if coverage < 0.8:
            recommendations.append("分野カバレッジが不十分です。より多様な研究室を推奨に含めてください。")
        
        # 一貫性
        consistency = evaluation["consistency_check"]["consistency_score"]
        if consistency < 0.7:
            recommendations.append("予測結果の一貫性が低いです。ランキングアルゴリズムを見直してください。")
        
        return recommendations

class MetricsReporter:
    """指標レポート生成クラス"""
    
    def __init__(self):
        self.report_templates = self._initialize_templates()
    
    def generate_performance_report(self, metrics: PerformanceMetrics, 
                                   system_metrics: SystemMetrics) -> str:
        """性能レポート生成"""
        
        report = []
        report.append("📊 システム性能レポート")
        report.append("=" * 50)
        
        # 予測性能
        report.append("\n🎯 予測性能")
        report.append(f"   精度: {metrics.accuracy:.3f}")
        report.append(f"   適合率: {metrics.precision:.3f}")
        report.append(f"   再現率: {metrics.recall:.3f}")
        report.append(f"   F1スコア: {metrics.f1_score:.3f}")
        report.append(f"   MAE: {metrics.mae:.3f}")
        report.append(f"   RMSE: {metrics.rmse:.3f}")
        report.append(f"   相関係数: {metrics.correlation:.3f}")
        
        # システム性能
        report.append("\n⚡ システム性能")
        report.append(f"   平均レスポンス時間: {system_metrics.average_response_time:.2f}秒")
        report.append(f"   ユーザー満足度: {system_metrics.user_satisfaction:.3f}")
        report.append(f"   アルゴリズム効率: {system_metrics.algorithm_efficiency:.3f}")
        report.append(f"   モデル安定性: {system_metrics.model_stability:.3f}")
        
        # 評価と推奨
        report.append("\n📈 評価と推奨事項")
        
        if metrics.accuracy >= 0.8:
            report.append("   ✅ 予測精度は優秀です")
        elif metrics.accuracy >= 0.6:
            report.append("   ⚠️ 予測精度は改善の余地があります")
        else:
            report.append("   ❌ 予測精度が低く、改善が必要です")
        
        if system_metrics.average_response_time <= 2.0:
            report.append("   ✅ レスポンス時間は良好です")
        else:
            report.append("   ⚠️ レスポンス時間の最適化を検討してください")
        
        return "\n".join(report)
    
    def generate_compatibility_report(self, compatibility: CompatibilityMetrics) -> str:
        """適合性レポート生成"""
        
        report = []
        report.append("🎯 適合性分析レポート")
        report.append("=" * 40)
        
        report.append(f"\n総合適合性: {compatibility.overall_compatibility:.3f}")
        report.append(f"分野適合性: {compatibility.field_compatibility:.3f}")
        report.append(f"基準適合性: {compatibility.criteria_compatibility:.3f}")
        report.append(f"多様性スコア: {compatibility.diversity_score:.3f}")
        report.append(f"一貫性スコア: {compatibility.consistency_score:.3f}")
        report.append(f"信頼度: {compatibility.confidence:.3f}")
        
        # 解釈
        report.append(f"\n📋 解釈:")
        
        if compatibility.overall_compatibility >= 0.8:
            report.append("   非常に高い適合性を示しています")
        elif compatibility.overall_compatibility >= 0.6:
            report.append("   良好な適合性があります")
        else:
            report.append("   適合性の向上が必要です")
        
        return "\n".join(report)
    
    def _initialize_templates(self) -> Dict[str, str]:
        """レポートテンプレートの初期化"""
        
        return {
            "performance_summary": "システム性能サマリー",
            "detailed_analysis": "詳細分析レポート",
            "comparison_report": "比較分析レポート"
        }
    
    def export_metrics_to_json(self, metrics_data: Dict[str, Any], 
                              filepath: str) -> None:
        """指標データのJSON出力"""
        
        import json
        from datetime import datetime
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "metrics": metrics_data,
            "metadata": {
                "version": "2.0.0",
                "calculation_method": "fuzzy_genetic_hybrid"
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"指標データをエクスポートしました: {filepath}")

# ユーティリティ関数

def calculate_normalized_dcg(predicted_ranking: List[float], 
                           ideal_ranking: List[float], k: int = 10) -> float:
    """正規化DCG (Normalized Discounted Cumulative Gain) 計算"""
    
    def dcg(scores, k):
        """DCG計算"""
        scores = scores[:k]
        return sum(score / np.log2(i + 2) for i, score in enumerate(scores))
    
    dcg_predicted = dcg(predicted_ranking, k)
    dcg_ideal = dcg(sorted(ideal_ranking, reverse=True), k)
    
    return dcg_predicted / dcg_ideal if dcg_ideal > 0 else 0.0

def calculate_kendall_tau(ranking1: List[int], ranking2: List[int]) -> float:
    """ケンドールの順位相関係数計算"""
    
    try:
        tau, _ = stats.kendalltau(ranking1, ranking2)
        return tau if not np.isnan(tau) else 0.0
    except:
        return 0.0

def calculate_hit_rate(predicted_top_k: List[int], 
                      relevant_items: List[int]) -> float:
    """ヒット率計算"""
    
    hits = len(set(predicted_top_k) & set(relevant_items))
    return hits / len(relevant_items) if relevant_items else 0.0