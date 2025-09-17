# core/ai_system.py - 完全13項目対応 AI統合システム

import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class EvaluationMethod(str, Enum):
    """評価手法"""
    FUZZY_ONLY = "fuzzy_only"
    GENETIC_ONLY = "genetic_only"
    DECISION_TREE_ONLY = "decision_tree_only"
    HYBRID_ALL = "hybrid_all"
    COMPLETE_13_CRITERIA = "complete_13_criteria"

@dataclass
class AISystemConfig:
    """AI統合システム設定（13項目完全対応）"""
    
    # 手法有効化設定
    fuzzy_enabled: bool = True
    genetic_enabled: bool = True
    decision_tree_enabled: bool = True
    
    # 13項目完全対応設定
    complete_criteria_mode: bool = True
    criteria_weights_enabled: bool = True
    adaptive_weights: bool = True
    
    # 統合設定
    evaluation_method: EvaluationMethod = EvaluationMethod.COMPLETE_13_CRITERIA
    score_integration_weights: Dict[str, float] = field(default_factory=lambda: {
        "fuzzy": 0.4,
        "genetic": 0.35,
        "decision_tree": 0.25
    })
    
    # パフォーマンス設定
    cache_results: bool = True
    parallel_processing: bool = False
    
    # 品質設定
    confidence_threshold: float = 0.7
    explanation_detail_level: str = "comprehensive"  # basic, detailed, comprehensive

@dataclass
class EvaluationResult:
    """評価結果（13項目完全対応）"""
    
    lab_id: str
    lab_name: str
    
    # 各手法のスコア
    fuzzy_score: float
    genetic_score: float
    decision_tree_score: float
    
    # 統合スコア
    final_score: float
    confidence: float
    
    # 13項目詳細分析
    criteria_scores: Dict[str, float]
    criteria_weights: Dict[str, float]
    field_match_score: float
    
    # 説明情報
    fuzzy_explanation: Dict[str, Any]
    genetic_explanation: Dict[str, Any]
    decision_tree_explanation: Dict[str, Any]
    comprehensive_explanation: str
    
    # メタデータ
    processing_time: float
    timestamp: str
    evaluation_method: str
    data_completeness: float

class CompleteAISystem:
    """完全13項目対応 AI統合システム"""
    
    # 完全13項目評価基準
    COMPLETE_CRITERIA = [
        # 基本項目（5項目）
        "research_intensity", "advisor_style", "team_work", "workload", "theory_practice",
        # 拡張項目（5項目）
        "research_field_match", "skill_development", "lab_atmosphere", "flexibility", "publication_opportunity",
        # 特殊項目（3項目）
        "interdisciplinary", "communication_style", "innovation_risk"
    ]
    
    # 基準別重要度重み（13項目完全対応）
    CRITERIA_IMPORTANCE_WEIGHTS = {
        # 基本項目：高重要度
        "research_intensity": 1.3,
        "advisor_style": 1.2,
        "team_work": 1.1,
        "workload": 1.1,
        "theory_practice": 1.2,
        
        # 拡張項目：中〜高重要度
        "research_field_match": 1.5,  # 最高重要度
        "skill_development": 1.0,
        "lab_atmosphere": 0.9,
        "flexibility": 0.9,
        "publication_opportunity": 1.1,
        
        # 特殊項目：調整重要度
        "interdisciplinary": 0.8,
        "communication_style": 0.9,
        "innovation_risk": 1.0
    }
    
    def __init__(self, config: AISystemConfig):
        self.config = config
        self.fuzzy_engine = None
        self.genetic_engine = None
        self.decision_tree = None
        self.optimized_weights = None
        
        # 統計情報
        self.evaluation_count = 0
        self.stats = {
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "average_processing_time": 0.0,
            "method_usage": {
                "fuzzy": 0,
                "genetic": 0,
                "decision_tree": 0,
                "hybrid": 0
            }
        }
        
        # 結果キャッシュ
        self.result_cache = {}
        
        # システム初期化
        self._initialize_complete_ai_system()
    
    def _initialize_complete_ai_system(self):
        """完全AI統合システム初期化"""
        
        logger.info("完全13項目対応AI統合システム初期化開始...")
        
        # 1. ファジィ推論エンジン初期化（13項目対応）
        if self.config.fuzzy_enabled:
            try:
                from core.fuzzy.inference import CompleteTriangularFuzzyEngine
                self.fuzzy_engine = CompleteTriangularFuzzyEngine(
                    criteria=self.COMPLETE_CRITERIA,
                    importance_weights=self.CRITERIA_IMPORTANCE_WEIGHTS
                )
                logger.info("✅ ファジィ推論エンジン初期化完了（13項目対応）")
            except ImportError:
                logger.warning("ファジィ推論エンジンが利用できません - フォールバック使用")
                self.fuzzy_engine = self._create_fallback_fuzzy_engine()
        
        # 2. 遺伝的アルゴリズム初期化（13項目対応）
        if self.config.genetic_enabled:
            try:
                from core.genetic.evolution import CompleteEvolutionEngine
                self.genetic_engine = CompleteEvolutionEngine(
                    criteria_count=len(self.COMPLETE_CRITERIA),
                    population_size=30,
                    generations=25
                )
                logger.info("✅ 遺伝的アルゴリズム初期化完了（13項目対応）")
            except ImportError:
                logger.warning("遺伝的アルゴリズムが利用できません - フォールバック使用")
                self.genetic_engine = self._create_fallback_genetic_engine()
        
        # 3. ファジィ決定木初期化（13項目対応）
        if self.config.decision_tree_enabled:
            try:
                from core.decision_tree.tree import CompleteFuzzyDecisionTree
                self.decision_tree = CompleteFuzzyDecisionTree(
                    criteria=self.COMPLETE_CRITERIA,
                    max_depth=7,
                    min_samples_leaf=3
                )
                logger.info("✅ ファジィ決定木初期化完了（13項目対応）")
            except ImportError:
                logger.warning("ファジィ決定木が利用できません - フォールバック使用")
                self.decision_tree = self._create_fallback_decision_tree()
        
        logger.info(f"🎯 完全13項目対応AI統合システム初期化完了: {len(self.COMPLETE_CRITERIA)}基準")
    
    def _create_fallback_fuzzy_engine(self):
        """フォールバック ファジィエンジン作成（13項目対応）"""
        
        class FallbackFuzzyEngine:
            def __init__(self, criteria, weights):
                self.criteria = criteria
                self.weights = weights
            
            def evaluate_compatibility(self, student_profile: Dict, lab_profile: Dict) -> Tuple[float, Dict]:
                """フォールバック適合度評価"""
                total_weighted_score = 0.0
                total_weights = 0.0
                detailed_scores = {}
                
                for criterion in self.criteria:
                    if criterion in student_profile and criterion in lab_profile:
                        student_val = float(student_profile[criterion])
                        lab_val = float(lab_profile[criterion])
                        
                        # 三角ファジィ関数による類似度計算
                        diff = abs(student_val - lab_val)
                        if diff <= 1.0:
                            similarity = 1.0  # 非常に類似
                        elif diff <= 2.0:
                            similarity = 1.0 - (diff - 1.0) / 1.0 * 0.3  # やや類似
                        elif diff <= 3.0:
                            similarity = 0.7 - (diff - 2.0) / 1.0 * 0.3  # 普通
                        else:
                            similarity = max(0.0, 0.4 - (diff - 3.0) / 6.0 * 0.4)  # 低類似
                        
                        # 重み適用
                        weight = self.weights.get(criterion, 1.0)
                        weighted_score = similarity * weight
                        
                        detailed_scores[criterion] = {
                            "similarity": similarity,
                            "weight": weight,
                            "weighted_score": weighted_score
                        }
                        
                        total_weighted_score += weighted_score
                        total_weights += weight
                
                final_score = total_weighted_score / total_weights if total_weights > 0 else 0.5
                
                explanation = {
                    "method": "triangular_fuzzy_fallback",
                    "criteria_evaluated": len(detailed_scores),
                    "total_weight": total_weights,
                    "detailed_scores": detailed_scores
                }
                
                return final_score, explanation
        
        return FallbackFuzzyEngine(self.COMPLETE_CRITERIA, self.CRITERIA_IMPORTANCE_WEIGHTS)
    
    def _create_fallback_genetic_engine(self):
        """フォールバック 遺伝的アルゴリズム作成（13項目対応）"""
        
        class FallbackGeneticEngine:
            def __init__(self, criteria_count):
                self.criteria_count = criteria_count
                # ランダム重みベクトル生成（簡易最適化シミュレーション）
                self.best_weights = np.random.uniform(0.5, 1.5, criteria_count)
                self.best_weights = self.best_weights / np.sum(self.best_weights) * criteria_count
            
            def evaluate_with_weights(self, student_profile: Dict, lab_profile: Dict, 
                                    criteria: List[str]) -> Tuple[float, Dict]:
                """重み付き評価"""
                weighted_sum = 0.0
                total_weights = 0.0
                details = {}
                
                for i, criterion in enumerate(criteria):
                    if criterion in student_profile and criterion in lab_profile:
                        student_val = float(student_profile[criterion])
                        lab_val = float(lab_profile[criterion])
                        
                        # 距離ベース類似度
                        diff = abs(student_val - lab_val)
                        similarity = max(0.0, 1.0 - diff / 9.0)
                        
                        # 遺伝的最適化重み適用
                        weight = self.best_weights[i] if i < len(self.best_weights) else 1.0
                        weighted_score = similarity * weight
                        
                        details[criterion] = {
                            "similarity": similarity,
                            "genetic_weight": weight,
                            "weighted_score": weighted_score
                        }
                        
                        weighted_sum += weighted_score
                        total_weights += weight
                
                final_score = weighted_sum / total_weights if total_weights > 0 else 0.5
                
                explanation = {
                    "method": "genetic_optimization_fallback",
                    "weights_used": len([w for w in self.best_weights if w > 0]),
                    "criteria_evaluated": len(details),
                    "optimization_info": "simulated_evolution",
                    "details": details
                }
                
                return final_score, explanation
        
        return FallbackGeneticEngine(len(self.COMPLETE_CRITERIA))
    
    def _create_fallback_decision_tree(self):
        """フォールバック 決定木作成（13項目対応）"""
        
        class FallbackDecisionTree:
            def __init__(self, criteria):
                self.criteria = criteria
            
            def predict_with_explanation(self, student_profile: Dict) -> Tuple[float, Dict]:
                """決定木予測（フォールバック）"""
                
                # 簡易決定ルール
                score = 0.5  # ベーススコア
                decision_path = ["root"]
                
                # 主要基準による分岐シミュレーション
                if "research_intensity" in student_profile:
                    research_intensity = student_profile["research_intensity"]
                    if research_intensity >= 8:
                        score += 0.2
                        decision_path.append("high_research_intensity -> +0.2")
                    elif research_intensity <= 3:
                        score -= 0.1
                        decision_path.append("low_research_intensity -> -0.1")
                
                if "research_field_match" in student_profile:
                    field_match = student_profile["research_field_match"]
                    if field_match >= 8:
                        score += 0.25
                        decision_path.append("high_field_match -> +0.25")
                    elif field_match <= 4:
                        score -= 0.15
                        decision_path.append("low_field_match -> -0.15")
                
                if "team_work" in student_profile:
                    team_work = student_profile["team_work"]
                    if team_work >= 7 and student_profile.get("communication_style", 5) >= 7:
                        score += 0.15
                        decision_path.append("collaborative_style -> +0.15")
                
                # スコア正規化
                final_score = max(0.0, min(1.0, score))
                
                explanation = {
                    "method": "rule_based_fallback",
                    "decision_path": decision_path,
                    "base_score": 0.5,
                    "adjustments": final_score - 0.5,
                    "final_score": final_score
                }
                
                return final_score, explanation
        
        return FallbackDecisionTree(self.COMPLETE_CRITERIA)
    
    def evaluate_lab_compatibility(self, student_profile: Dict[str, Any], 
                                 lab_profile: Dict[str, Any],
                                 lab_id: str, lab_name: str) -> EvaluationResult:
        """研究室適合性の統合評価（13項目完全対応）"""
        
        start_time = time.time()
        cache_key = f"{hash(str(student_profile))}_{lab_id}"
        
        # キャッシュ確認
        if self.config.cache_results and cache_key in self.result_cache:
            logger.debug(f"キャッシュヒット: {lab_id}")
            return self.result_cache[cache_key]
        
        logger.info(f"統合評価開始: {lab_name} (13項目完全対応)")
        
        # データ完全性チェック
        data_completeness = self._calculate_data_completeness(student_profile)
        
        # 各手法による評価実行
        fuzzy_score = 0.5
        genetic_score = 0.5
        decision_tree_score = 0.5
        
        fuzzy_explanation = {}
        genetic_explanation = {}
        decision_tree_explanation = {}
        
        # 1. ファジィ推論による評価（13項目対応）
        if self.config.fuzzy_enabled and self.fuzzy_engine:
            try:
                fuzzy_score, fuzzy_explanation = self.fuzzy_engine.evaluate_compatibility(
                    student_profile, lab_profile
                )
                self.stats["method_usage"]["fuzzy"] += 1
                logger.debug(f"ファジィスコア: {fuzzy_score:.3f}")
            except Exception as e:
                logger.warning(f"ファジィ推論エラー: {e}")
                fuzzy_score = 0.5
        
        # 2. 遺伝的アルゴリズムによる評価（13項目対応）
        if self.config.genetic_enabled and self.genetic_engine:
            try:
                genetic_score, genetic_explanation = self.genetic_engine.evaluate_with_weights(
                    student_profile, lab_profile, self.COMPLETE_CRITERIA
                )
                self.stats["method_usage"]["genetic"] += 1
                logger.debug(f"遺伝的スコア: {genetic_score:.3f}")
            except Exception as e:
                logger.warning(f"遺伝的アルゴリズム評価エラー: {e}")
                genetic_score = 0.5
        
        # 3. ファジィ決定木による評価（13項目対応）
        if self.config.decision_tree_enabled and self.decision_tree:
            try:
                decision_tree_score, decision_tree_explanation = self.decision_tree.predict_with_explanation(
                    student_profile
                )
                self.stats["method_usage"]["decision_tree"] += 1
                logger.debug(f"決定木スコア: {decision_tree_score:.3f}")
            except Exception as e:
                logger.warning(f"決定木評価エラー: {e}")
                decision_tree_score = 0.5
        
        # 4. 詳細基準スコア計算（13項目）
        criteria_scores, criteria_weights = self._calculate_detailed_criteria_analysis(
            student_profile, lab_profile
        )
        
        # 5. 研究分野適合性計算
        field_match_score = self._calculate_field_compatibility(
            student_profile, lab_profile
        )
        
        # 6. 統合スコア計算
        final_score, confidence = self._integrate_all_scores(
            fuzzy_score, genetic_score, decision_tree_score,
            criteria_scores, field_match_score, data_completeness
        )
        
        # 7. 包括的説明生成
        comprehensive_explanation = self._generate_comprehensive_explanation(
            student_profile, lab_profile, {
                "fuzzy_score": fuzzy_score,
                "genetic_score": genetic_score,
                "decision_tree_score": decision_tree_score,
                "final_score": final_score,
                "criteria_scores": criteria_scores,
                "field_match_score": field_match_score
            }
        )
        
        processing_time = time.time() - start_time
        
        # 結果構築
        result = EvaluationResult(
            lab_id=lab_id,
            lab_name=lab_name,
            fuzzy_score=fuzzy_score,
            genetic_score=genetic_score,
            decision_tree_score=decision_tree_score,
            final_score=final_score,
            confidence=confidence,
            criteria_scores=criteria_scores,
            criteria_weights=criteria_weights,
            field_match_score=field_match_score,
            fuzzy_explanation=fuzzy_explanation,
            genetic_explanation=genetic_explanation,
            decision_tree_explanation=decision_tree_explanation,
            comprehensive_explanation=comprehensive_explanation,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            evaluation_method=self.config.evaluation_method.value,
            data_completeness=data_completeness
        )
        
        # キャッシュに保存
        if self.config.cache_results:
            self.result_cache[cache_key] = result
        
        # 統計更新
        self.evaluation_count += 1
        self.stats["total_evaluations"] += 1
        self.stats["successful_evaluations"] += 1
        self.stats["method_usage"]["hybrid"] += 1
        self._update_processing_time_stats(processing_time)
        
        logger.info(f"統合評価完了: {lab_name}, 最終スコア: {final_score:.3f}, 処理時間: {processing_time:.3f}秒")
        
        return result
    
    def _calculate_data_completeness(self, student_profile: Dict[str, Any]) -> float:
        """データ完全性計算（13項目対応）"""
        
        completed_criteria = sum(
            1 for criterion in self.COMPLETE_CRITERIA
            if criterion in student_profile and student_profile[criterion] is not None
        )
        
        return completed_criteria / len(self.COMPLETE_CRITERIA)
    
    def _calculate_detailed_criteria_analysis(self, student_profile: Dict[str, Any],
                                            lab_profile: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """詳細基準分析（13項目完全対応）"""
        
        criteria_scores = {}
        criteria_weights = {}
        
        for criterion in self.COMPLETE_CRITERIA:
            if criterion in student_profile and criterion in lab_profile:
                student_val = float(student_profile[criterion])
                lab_val = float(lab_profile[criterion])
                
                # 基本類似度計算
                diff = abs(student_val - lab_val)
                base_similarity = max(0.0, 1.0 - diff / 9.0)
                
                # 重要度重み取得
                importance_weight = self.CRITERIA_IMPORTANCE_WEIGHTS.get(criterion, 1.0)
                
                # 適応的重み調整（学生の価値観に基づく）
                if self.config.adaptive_weights:
                    student_emphasis = student_val / 10.0  # 学生がその基準をどの程度重視するか
                    adaptive_factor = 0.8 + student_emphasis * 0.4  # 0.8〜1.2の調整
                    final_weight = importance_weight * adaptive_factor
                else:
                    final_weight = importance_weight
                
                # 重み適用後スコア
                weighted_similarity = min(1.0, base_similarity * final_weight)
                
                criteria_scores[criterion] = weighted_similarity
                criteria_weights[criterion] = final_weight
            else:
                criteria_scores[criterion] = 0.0  # データ不足
                criteria_weights[criterion] = self.CRITERIA_IMPORTANCE_WEIGHTS.get(criterion, 1.0)
        
        return criteria_scores, criteria_weights
    
    def _calculate_field_compatibility(self, student_profile: Dict[str, Any],
                                     lab_profile: Dict[str, Any]) -> float:
        """研究分野適合性計算"""
        
        field_interests = student_profile.get("field_interests", {})
        lab_field = lab_profile.get("research_field_id", "")
        
        if not field_interests or not lab_field:
            return 0.0
        
        if lab_field in field_interests:
            interest_level = field_interests[lab_field]
            normalized_interest = interest_level / 10.0
            
            # research_field_match基準による重み調整
            field_match_importance = student_profile.get("research_field_match", 5.0) / 10.0
            
            return normalized_interest * field_match_importance
        
        return 0.0
    
    def _integrate_all_scores(self, fuzzy_score: float, genetic_score: float, 
                            decision_tree_score: float, criteria_scores: Dict[str, float],
                            field_match_score: float, data_completeness: float) -> Tuple[float, float]:
        """全スコア統合計算（13項目対応）"""
        
        # 基本統合（各手法の重み付き平均）
        integration_weights = self.config.score_integration_weights
        
        base_integrated_score = (
            fuzzy_score * integration_weights["fuzzy"] +
            genetic_score * integration_weights["genetic"] +
            decision_tree_score * integration_weights["decision_tree"]
        )
        
        # 詳細基準スコアからの補正
        valid_criteria_scores = [score for score in criteria_scores.values() if score > 0]
        if valid_criteria_scores:
            criteria_average = sum(valid_criteria_scores) / len(valid_criteria_scores)
            # 基本統合スコアと基準平均の重み付き統合
            base_integrated_score = base_integrated_score * 0.7 + criteria_average * 0.3
        
        # 研究分野ボーナス適用
        field_bonus = field_match_score * 0.15  # 最大15%のボーナス
        final_score = min(1.0, base_integrated_score + field_bonus)
        
        # 信頼度計算
        method_consensus = 1.0 - abs(fuzzy_score - genetic_score) * 0.5  # 手法間一致度
        data_quality_factor = data_completeness ** 0.5  # データ品質要因
        score_stability = min(1.0, final_score * 1.2)  # スコア安定性
        
        confidence = (method_consensus * 0.4 + data_quality_factor * 0.35 + score_stability * 0.25)
        confidence = max(0.0, min(1.0, confidence))
        
        return final_score, confidence
    
    def _generate_comprehensive_explanation(self, student_profile: Dict[str, Any],
                                          lab_profile: Dict[str, Any],
                                          scores: Dict[str, Any]) -> str:
        """包括的説明生成（13項目対応）"""
        
        if self.config.explanation_detail_level == "basic":
            return self._generate_basic_explanation(scores)
        elif self.config.explanation_detail_level == "detailed":
            return self._generate_detailed_explanation(student_profile, lab_profile, scores)
        else:  # comprehensive
            return self._generate_full_comprehensive_explanation(student_profile, lab_profile, scores)
    
    def _generate_full_comprehensive_explanation(self, student_profile: Dict[str, Any],
                                               lab_profile: Dict[str, Any],
                                               scores: Dict[str, Any]) -> str:
        """包括的詳細説明生成"""
        
        explanation_parts = []
        
        # 1. 総合評価
        final_score = scores["final_score"]
        if final_score >= 0.85:
            explanation_parts.append(f"この研究室は非常に高い適合性（{final_score:.1%}）を示しており、強く推薦されます。")
        elif final_score >= 0.7:
            explanation_parts.append(f"この研究室は良好な適合性（{final_score:.1%}）があり、有力な候補として推薦されます。")
        elif final_score >= 0.5:
            explanation_parts.append(f"この研究室は中程度の適合性（{final_score:.1%}）があり、検討に値します。")
        else:
            explanation_parts.append(f"この研究室の適合性（{final_score:.1%}）は限定的で、慎重な検討が必要です。")
        
        # 2. 手法別分析
        fuzzy_score = scores["fuzzy_score"]
        genetic_score = scores["genetic_score"]
        decision_tree_score = scores["decision_tree_score"]
        
        explanation_parts.append(
            f"詳細分析では、ファジィ推論による評価が{fuzzy_score:.2f}、"
            f"遺伝的最適化による評価が{genetic_score:.2f}、"
            f"決定木による評価が{decision_tree_score:.2f}となっています。"
        )
        
        # 手法間の一致度分析
        score_variance = np.var([fuzzy_score, genetic_score, decision_tree_score])
        if score_variance < 0.02:
            explanation_parts.append("各評価手法の結果が高い一致を示しており、信頼性の高い評価です。")
        elif score_variance > 0.1:
            explanation_parts.append("評価手法間で結果にばらつきがあり、複数の観点から慎重に検討することが推奨されます。")
        
        # 3. 基準別分析（上位・下位基準の特定）
        criteria_scores = scores["criteria_scores"]
        sorted_criteria = sorted(criteria_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 高適合基準（上位3つ）
        high_match_criteria = [criterion for criterion, score in sorted_criteria[:3] if score > 0.7]
        if high_match_criteria:
            criteria_names = [self.CRITERIA_IMPORTANCE_WEIGHTS.get(c, c) for c in high_match_criteria]
            explanation_parts.append(f"特に優れた適合性を示す分野: {', '.join(criteria_names[:3])}")
        
        # 低適合基準（注意が必要な分野）
        low_match_criteria = [criterion for criterion, score in sorted_criteria[-3:] if score < 0.4]
        if low_match_criteria:
            criteria_names = [self.CRITERIA_IMPORTANCE_WEIGHTS.get(c, c) for c in low_match_criteria]
            explanation_parts.append(f"注意が必要な分野: {', '.join(criteria_names[:2])}")
        
        # 4. 研究分野適合性
        field_match_score = scores["field_match_score"]
        if field_match_score > 0.8:
            explanation_parts.append("研究分野への興味度が非常に高く、モチベーション維持が期待できます。")
        elif field_match_score < 0.3:
            explanation_parts.append("研究分野への興味度が低いため、事前の詳しい調査をお勧めします。")
        
        # 5. 推薦アクション
        if final_score >= 0.7:
            explanation_parts.append("積極的に研究室見学や教員面談を実施し、詳細な情報収集を行うことを推薦します。")
        elif final_score >= 0.5:
            explanation_parts.append("他の候補研究室と比較検討し、研究内容や環境についてより詳しく調査することをお勧めします。")
        else:
            explanation_parts.append("より適合性の高い他の研究室の検討も併せて行うことをお勧めします。")
        
        return " ".join(explanation_parts)
    
    def _generate_basic_explanation(self, scores: Dict[str, Any]) -> str:
        """基本説明生成"""
        final_score = scores["final_score"]
        return f"総合適合度: {final_score:.1%}"
    
    def _generate_detailed_explanation(self, student_profile: Dict[str, Any],
                                     lab_profile: Dict[str, Any], scores: Dict[str, Any]) -> str:
        """詳細説明生成"""
        final_score = scores["final_score"]
        field_score = scores["field_match_score"]
        
        level = "高" if final_score >= 0.7 else "中" if final_score >= 0.5 else "低"
        field_match = "良好" if field_score >= 0.7 else "普通" if field_score >= 0.4 else "要検討"
        
        return f"適合性レベル: {level}（{final_score:.1%}）、研究分野マッチ: {field_match}（{field_score:.1%}）"
    
    def _update_processing_time_stats(self, processing_time: float):
        """処理時間統計更新"""
        current_avg = self.stats["average_processing_time"]
        total_evals = self.stats["total_evaluations"]
        
        if total_evals == 1:
            self.stats["average_processing_time"] = processing_time
        else:
            self.stats["average_processing_time"] = (current_avg * (total_evals - 1) + processing_time) / total_evals
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """システム統計取得"""
        
        return {
            "evaluation_count": self.evaluation_count,
            "stats": self.stats,
            "criteria_supported": len(self.COMPLETE_CRITERIA),
            "config": {
                "fuzzy_enabled": self.config.fuzzy_enabled,
                "genetic_enabled": self.config.genetic_enabled,
                "decision_tree_enabled": self.config.decision_tree_enabled,
                "complete_criteria_mode": self.config.complete_criteria_mode,
                "evaluation_method": self.config.evaluation_method.value
            },
            "cache_size": len(self.result_cache),
            "criteria_weights": self.CRITERIA_IMPORTANCE_WEIGHTS
        }
    
    def clear_cache(self):
        """キャッシュクリア"""
        self.result_cache.clear()
        logger.info("結果キャッシュをクリアしました")
    
    def update_criteria_weights(self, new_weights: Dict[str, float]):
        """基準重み更新"""
        for criterion, weight in new_weights.items():
            if criterion in self.CRITERIA_IMPORTANCE_WEIGHTS:
                self.CRITERIA_IMPORTANCE_WEIGHTS[criterion] = weight
        
        logger.info(f"基準重み更新完了: {len(new_weights)}項目")
        
        # キャッシュクリア（重み変更により結果が変わるため）
        self.clear_cache()