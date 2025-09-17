# services/lab_matching.py - 完全13項目対応版

import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass

from models.schemas import (
    StudentProfile, Laboratory, EvaluationResponse,
    LabResult, CompatibilityScore
)

logger = logging.getLogger(__name__)

@dataclass
class MatchingConfig:
    """マッチング設定（13項目完全対応）"""
    
    # 基本設定
    max_recommendations: int = 10
    enable_genetic_optimization: bool = True
    use_fuzzy_inference: bool = True
    enable_explanation: bool = True
    
    # 13項目完全対応設定
    complete_criteria_mode: bool = True  # 13項目完全使用
    criteria_weights_auto_adjust: bool = True  # 重み自動調整
    field_bonus_enabled: bool = True  # 分野ボーナス有効
    
    # 閾値設定
    high_compatibility_threshold: float = 0.8
    medium_compatibility_threshold: float = 0.6
    confidence_threshold: float = 0.7
    
    # パフォーマンス設定
    parallel_evaluation: bool = False
    cache_results: bool = True

class LabMatchingService:
    """研究室マッチングサービス（完全13項目対応版）"""
    
    # 完全13項目評価基準
    COMPLETE_CRITERIA = [
        # 基本項目（5項目）
        "research_intensity", "advisor_style", "team_work", "workload", "theory_practice",
        # 拡張項目（5項目）  
        "research_field_match", "skill_development", "lab_atmosphere", "flexibility", "publication_opportunity",
        # 特殊項目（3項目）
        "interdisciplinary", "communication_style", "innovation_risk"
    ]
    
    # 基準別デフォルト重み（13項目完全対応）
    DEFAULT_WEIGHTS = {
        # 基本項目：高重み
        "research_intensity": 1.2,
        "advisor_style": 1.1,
        "team_work": 1.0,
        "workload": 1.0,
        "theory_practice": 1.1,
        
        # 拡張項目：中〜高重み
        "research_field_match": 1.4,  # 最重要
        "skill_development": 0.9,
        "lab_atmosphere": 0.8,
        "flexibility": 0.8,
        "publication_opportunity": 1.0,
        
        # 特殊項目：調整重み
        "interdisciplinary": 0.7,
        "communication_style": 0.8,
        "innovation_risk": 0.9
    }
    
    # 基準名の日本語マッピング
    CRITERIA_NAMES = {
        "research_intensity": "研究強度",
        "advisor_style": "指導スタイル",
        "team_work": "チームワーク",
        "workload": "ワークロード",
        "theory_practice": "理論・実践バランス",
        "research_field_match": "研究分野適合性",
        "skill_development": "スキル開発",
        "lab_atmosphere": "研究室雰囲気",
        "flexibility": "柔軟性",
        "publication_opportunity": "論文発表機会",
        "interdisciplinary": "学際性",
        "communication_style": "コミュニケーション",
        "innovation_risk": "革新性・リスク許容度"
    }
    
    def __init__(self, config: MatchingConfig):
        self.config = config
        self.fuzzy_engine = None
        self.genetic_optimizer = None
        self.optimized_weights = None
        
        # 統計情報
        self.total_evaluations = 0
        self.successful_matches = 0
        self.average_processing_time = 0.0
        
        # 結果キャッシュ
        self.result_cache = {}
        
        # 13項目完全対応の初期化
        self._initialize_complete_criteria_system()
        
        logger.info("研究室マッチングサービス初期化完了（13項目完全対応）")
    
    def _initialize_complete_criteria_system(self):
        """13項目完全対応システムの初期化"""
        
        try:
            # ファジィ推論エンジンの初期化（13項目対応）
            if self.config.use_fuzzy_inference:
                from core.fuzzy.inference import CompleteFuzzyInferenceEngine
                self.fuzzy_engine = CompleteFuzzyInferenceEngine(
                    criteria=self.COMPLETE_CRITERIA,
                    weights=self.DEFAULT_WEIGHTS
                )
            
            # 遺伝的最適化器の初期化（13項目対応）
            if self.config.enable_genetic_optimization:
                from core.genetic.evolution import CompleteGeneticOptimizer
                self.genetic_optimizer = CompleteGeneticOptimizer(
                    criteria_count=len(self.COMPLETE_CRITERIA),
                    population_size=50,
                    generations=20
                )
            
            logger.info(f"13項目完全対応システム初期化完了: {len(self.COMPLETE_CRITERIA)}基準")
            
        except ImportError as e:
            logger.warning(f"一部モジュールが利用できません: {e}")
            # フォールバックエンジンを使用
            self._initialize_fallback_engine()
    
    def _initialize_fallback_engine(self):
        """フォールバックエンジンの初期化（13項目対応）"""
        
        class FallbackEngine:
            def __init__(self, criteria, weights):
                self.criteria = criteria
                self.weights = weights
            
            def evaluate_compatibility(self, student_profile: Dict, lab_profile: Dict) -> float:
                """フォールバック適合度計算（13項目）"""
                total_weighted_score = 0.0
                total_weights = 0.0
                
                for criterion in self.criteria:
                    if criterion in student_profile and criterion in lab_profile:
                        student_val = float(student_profile[criterion])
                        lab_val = float(lab_profile[criterion])
                        
                        # 類似度計算
                        diff = abs(student_val - lab_val)
                        similarity = max(0.0, 1.0 - diff / 9.0)
                        
                        # 重み適用
                        weight = self.weights.get(criterion, 1.0)
                        total_weighted_score += similarity * weight
                        total_weights += weight
                
                return total_weighted_score / total_weights if total_weights > 0 else 0.5
        
        self.fuzzy_engine = FallbackEngine(self.COMPLETE_CRITERIA, self.DEFAULT_WEIGHTS)
        logger.info("フォールバックエンジン初期化完了（13項目対応）")
    
    def evaluate_student_lab_compatibility(self, 
                                         student_profile: StudentProfile,
                                         laboratories: List[Laboratory]) -> EvaluationResponse:
        """学生と研究室群の適合性評価（13項目完全対応）"""
        
        start_time = datetime.now()
        evaluation_id = f"eval_{int(time.time())}_{student_profile.student_id}"
        
        logger.info(f"適合性評価開始: {evaluation_id}, 対象研究室数: {len(laboratories)}")
        
        # 入力データの完全性チェック
        completeness_info = self._check_criteria_completeness(student_profile)
        logger.info(f"入力データ完全性: {completeness_info['completeness_ratio']:.1%}")
        
        lab_results = []
        
        for lab in laboratories:
            try:
                # 個別研究室との適合性評価（13項目完全）
                compatibility_score = self._calculate_complete_compatibility(
                    student_profile, lab
                )
                
                # 詳細説明生成
                explanations = self._generate_complete_explanations(
                    student_profile, lab, compatibility_score
                )
                
                # 推薦レベル決定
                recommendation_level = self._determine_recommendation_level(
                    compatibility_score.overall_score
                )
                
                lab_result = LabResult(
                    laboratory=lab,
                    compatibility_score=compatibility_score,
                    ranking=0,  # 後でソート時に設定
                    recommendation_reasons=explanations["reasons"],
                    concerns=explanations["concerns"],
                    detailed_analysis=explanations["detailed_analysis"],
                    recommendation_level=recommendation_level,
                    evaluation_timestamp=start_time
                )
                
                lab_results.append(lab_result)
                self.total_evaluations += 1
                
            except Exception as e:
                logger.warning(f"研究室{lab.lab_id}の評価でエラー: {e}")
                continue
        
        # ランキング設定
        lab_results.sort(key=lambda x: x.compatibility_score.overall_score, reverse=True)
        for i, result in enumerate(lab_results):
            result.ranking = i + 1
        
        # 上位のみ保持
        lab_results = lab_results[:self.config.max_recommendations]
        
        # 統計情報の計算
        processing_time = (datetime.now() - start_time).total_seconds()
        scores = [result.compatibility_score.overall_score for result in lab_results]
        
        score_distribution = {
            "mean": float(np.mean(scores)) if scores else 0.0,
            "std": float(np.std(scores)) if scores else 0.0,
            "min": float(np.min(scores)) if scores else 0.0,
            "max": float(np.max(scores)) if scores else 0.0
        }
        
        # 推薦信頼度の計算
        recommendation_confidence = self._calculate_recommendation_confidence(
            lab_results, completeness_info
        )
        
        # レスポンス構築
        response = EvaluationResponse(
            student_profile=student_profile,
            lab_results=lab_results,
            processing_time=processing_time,
            algorithm_version="v2.0.0-complete-13-criteria",
            total_labs_evaluated=len(laboratories),
            score_distribution=score_distribution,
            recommendation_confidence=recommendation_confidence,
            evaluation_id=evaluation_id,
            timestamp=start_time,
            metadata={
                "criteria_completeness": completeness_info,
                "evaluation_method": "complete_13_criteria_weighted",
                "weights_used": self.DEFAULT_WEIGHTS,
                "features_enabled": {
                    "complete_criteria": True,
                    "weighted_calculation": True,
                    "field_bonus": self.config.field_bonus_enabled,
                    "genetic_optimization": self.config.enable_genetic_optimization
                }
            }
        )
        
        if lab_results:
            self.successful_matches += 1
        
        logger.info(f"適合性評価完了: {len(lab_results)}件の結果, 処理時間{processing_time:.2f}秒")
        
        return response
    
    def _check_criteria_completeness(self, student_profile: StudentProfile) -> Dict[str, Any]:
        """評価基準の完全性チェック（13項目）"""
        
        criteria_dict = student_profile.evaluation_criteria.dict()
        
        # 各グループの完全性チェック
        basic_criteria = self.COMPLETE_CRITERIA[:5]
        extended_criteria = self.COMPLETE_CRITERIA[5:10]
        special_criteria = self.COMPLETE_CRITERIA[10:13]
        
        basic_complete = sum(1 for c in basic_criteria if criteria_dict.get(c) is not None)
        extended_complete = sum(1 for c in extended_criteria if criteria_dict.get(c) is not None)
        special_complete = sum(1 for c in special_criteria if criteria_dict.get(c) is not None)
        
        total_complete = basic_complete + extended_complete + special_complete
        
        return {
            "total_criteria": len(self.COMPLETE_CRITERIA),
            "completed_criteria": total_complete,
            "completeness_ratio": total_complete / len(self.COMPLETE_CRITERIA),
            "group_completeness": {
                "basic": basic_complete / len(basic_criteria),
                "extended": extended_complete / len(extended_criteria),
                "special": special_complete / len(special_criteria)
            },
            "missing_criteria": [
                c for c in self.COMPLETE_CRITERIA 
                if criteria_dict.get(c) is None
            ]
        }
    
    def _calculate_complete_compatibility(self, 
                                        student_profile: StudentProfile,
                                        laboratory: Laboratory) -> CompatibilityScore:
        """完全13項目適合性スコア計算"""
        
        # ファジィ推論による評価
        if self.fuzzy_engine:
            student_dict = student_profile.evaluation_criteria.dict()
            lab_dict = laboratory.characteristics.dict()
            
            overall_score = self.fuzzy_engine.evaluate_compatibility(student_dict, lab_dict)
            confidence = min(1.0, overall_score + 0.1)  # 信頼度調整
        else:
            overall_score = 0.5
            confidence = 0.5
        
        # 各基準の詳細スコア計算
        criteria_scores = self._calculate_detailed_criteria_scores(
            student_profile, laboratory
        )
        
        # 分野適合性スコア
        field_match_score = self._calculate_field_match_score(
            student_profile, laboratory
        )
        
        # 遺伝的最適化による調整（利用可能な場合）
        if self.genetic_optimizer and self.optimized_weights:
            overall_score = self._apply_genetic_optimization(
                criteria_scores, field_match_score, self.optimized_weights
            )
        
        return CompatibilityScore(
            overall_score=min(1.0, max(0.0, overall_score)),
            criteria_scores=criteria_scores,
            field_match_score=field_match_score,
            confidence=confidence,
            metadata={
                "evaluation_method": "complete_13_criteria",
                "criteria_used": len([c for c, score in criteria_scores.items() if score is not None]),
                "weights_applied": True,
                "field_bonus_applied": field_match_score > 0
            }
        )
    
    def _calculate_detailed_criteria_scores(self, 
                                          student_profile: StudentProfile,
                                          laboratory: Laboratory) -> Dict[str, Optional[float]]:
        """詳細基準スコア計算（13項目完全対応）"""
        
        student_criteria = student_profile.evaluation_criteria.dict()
        lab_criteria = laboratory.characteristics.dict()
        
        criteria_scores = {}
        
        for criterion in self.COMPLETE_CRITERIA:
            student_value = student_criteria.get(criterion)
            lab_value = lab_criteria.get(criterion)
            
            if student_value is not None and lab_value is not None:
                # 差分ベースの適合性計算
                diff = abs(float(student_value) - float(lab_value))
                max_diff = 9.0  # 最大差分（1-10の範囲）
                
                # 基本類似度スコア
                similarity_score = max(0.0, 1.0 - diff / max_diff)
                
                # 基準別重み適用
                weight = self.DEFAULT_WEIGHTS.get(criterion, 1.0)
                weighted_score = similarity_score * weight
                
                # 正規化（重みによる調整を考慮）
                normalized_score = min(1.0, weighted_score)
                
                criteria_scores[criterion] = normalized_score
            else:
                criteria_scores[criterion] = None  # データ不足
        
        return criteria_scores
    
    def _calculate_field_match_score(self, 
                                   student_profile: StudentProfile,
                                   laboratory: Laboratory) -> float:
        """研究分野適合性スコア計算"""
        
        if not self.config.field_bonus_enabled:
            return 0.0
        
        # 学生の分野興味と研究室分野の照合
        student_interests = {
            interest.research_field.value: interest.interest_level 
            for interest in student_profile.field_interests
        }
        
        lab_fields = [field.value for field in laboratory.research_fields]
        
        if not student_interests or not lab_fields:
            return 0.0
        
        # 最高マッチスコアを計算
        max_match_score = 0.0
        
        for lab_field in lab_fields:
            if lab_field in student_interests:
                interest_level = student_interests[lab_field]
                normalized_interest = interest_level / 10.0
                
                # research_field_match基準による重み調整
                field_weight = student_profile.evaluation_criteria.research_field_match or 5.0
                field_weight_normalized = field_weight / 10.0
                
                match_score = normalized_interest * field_weight_normalized
                max_match_score = max(max_match_score, match_score)
        
        return min(1.0, max_match_score)
    
    def _apply_genetic_optimization(self, 
                                  criteria_scores: Dict[str, Optional[float]],
                                  field_match_score: float,
                                  optimized_weights: List[float]) -> float:
        """遺伝的最適化による総合スコア計算"""
        
        if len(optimized_weights) < len(self.COMPLETE_CRITERIA):
            return 0.5  # 重み不足の場合
        
        weighted_sum = 0.0
        total_weights = 0.0
        
        for i, criterion in enumerate(self.COMPLETE_CRITERIA):
            score = criteria_scores.get(criterion)
            if score is not None:
                weight = optimized_weights[i]
                weighted_sum += score * weight
                total_weights += weight
        
        base_score = weighted_sum / total_weights if total_weights > 0 else 0.5
        
        # 分野ボーナス追加
        final_score = min(1.0, base_score + field_match_score * 0.1)
        
        return final_score
    
    def _generate_complete_explanations(self, 
                                      student_profile: StudentProfile,
                                      laboratory: Laboratory,
                                      compatibility_score: CompatibilityScore) -> Dict[str, Any]:
        """包括的説明生成（13項目対応）"""
        
        reasons = []
        concerns = []
        detailed_analysis = {}
        
        if not self.config.enable_explanation:
            return {"reasons": reasons, "concerns": concerns, "detailed_analysis": detailed_analysis}
        
        criteria_scores = compatibility_score.criteria_scores
        
        # 高スコア基準（推薦理由）
        high_score_criteria = [
            (criterion, score) for criterion, score in criteria_scores.items()
            if score is not None and score > 0.7
        ]
        high_score_criteria.sort(key=lambda x: x[1], reverse=True)
        
        # 低スコア基準（懸念点）
        low_score_criteria = [
            (criterion, score) for criterion, score in criteria_scores.items()
            if score is not None and score < 0.4
        ]
        low_score_criteria.sort(key=lambda x: x[1])
        
        # 推薦理由生成
        if compatibility_score.field_match_score > 0.7:
            reasons.append("研究分野の興味と非常によく一致しています")
        
        if len(high_score_criteria) >= 3:
            top_criteria = [self.CRITERIA_NAMES[c] for c, _ in high_score_criteria[:3]]
            reasons.append(f"{', '.join(top_criteria)}において高い適合性があります")
        
        if compatibility_score.overall_score > self.config.high_compatibility_threshold:
            reasons.append("総合的な適合度が非常に高い研究室です")
        
        # 懸念点生成
        if len(low_score_criteria) >= 2:
            concern_criteria = [self.CRITERIA_NAMES[c] for c, _ in low_score_criteria[:2]]
            concerns.append(f"{', '.join(concern_criteria)}において適合性が低い可能性があります")
        
        if compatibility_score.field_match_score < 0.3:
            concerns.append("研究分野の適合性が低い可能性があります")
        
        if compatibility_score.confidence < self.config.confidence_threshold:
            concerns.append("適合性の判定信頼度がやや低めです")
        
        # 詳細分析
        detailed_analysis = {
            "criteria_analysis": {
                criterion: {
                    "score": score,
                    "category": "high" if score and score > 0.7 else "low" if score and score < 0.4 else "medium",
                    "weight": self.DEFAULT_WEIGHTS.get(criterion, 1.0),
                    "importance": "critical" if criterion in self.COMPLETE_CRITERIA[:5] else "extended"
                }
                for criterion, score in criteria_scores.items()
                if score is not None
            },
            "field_analysis": {
                "field_match_score": compatibility_score.field_match_score,
                "field_bonus_applied": compatibility_score.field_match_score > 0
            },
            "score_breakdown": {
                "base_compatibility": sum(s for s in criteria_scores.values() if s) / len([s for s in criteria_scores.values() if s]),
                "field_bonus": compatibility_score.field_match_score,
                "final_score": compatibility_score.overall_score
            }
        }
        
        return {
            "reasons": reasons,
            "concerns": concerns,
            "detailed_analysis": detailed_analysis
        }
    
    def _determine_recommendation_level(self, overall_score: float) -> str:
        """推薦レベル決定"""
        
        if overall_score >= 0.85:
            return "強く推薦"
        elif overall_score >= 0.7:
            return "推薦"
        elif overall_score >= 0.5:
            return "検討可能"
        else:
            return "要慎重検討"
    
    def _calculate_recommendation_confidence(self, 
                                           lab_results: List[LabResult],
                                           completeness_info: Dict[str, Any]) -> float:
        """推薦信頼度計算"""
        
        if not lab_results:
            return 0.0
        
        # スコア分散による信頼度
        scores = [result.compatibility_score.overall_score for result in lab_results]
        score_variance = np.var(scores) if len(scores) > 1 else 0.0
        variance_factor = min(1.0, 1.0 - score_variance)
        
        # データ完全性による信頼度
        completeness_factor = completeness_info["completeness_ratio"]
        
        # 最高スコアによる信頼度
        max_score = max(scores) if scores else 0.0
        score_factor = max_score
        
        # 総合信頼度
        confidence = (variance_factor * 0.3 + completeness_factor * 0.4 + score_factor * 0.3)
        
        return min(1.0, max(0.0, confidence))
    
    def optimize_weights(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """重み最適化（13項目対応）"""
        
        if not self.genetic_optimizer:
            return {"success": False, "message": "遺伝的最適化器が利用できません"}
        
        try:
            logger.info(f"重み最適化開始: {len(training_data)}件のデータ")
            
            # 遺伝的アルゴリズムによる最適化実行
            optimization_result = self.genetic_optimizer.optimize(
                training_data, 
                criteria=self.COMPLETE_CRITERIA
            )
            
            self.optimized_weights = optimization_result["best_weights"]
            
            logger.info(f"重み最適化完了: 適応度={optimization_result['best_fitness']:.3f}")
            
            return {
                "success": True,
                "optimized_weights": self.optimized_weights,
                "optimization_fitness": optimization_result["best_fitness"],
                "generations_completed": optimization_result["generations"],
                "criteria_optimized": len(self.COMPLETE_CRITERIA)
            }
            
        except Exception as e:
            logger.error(f"重み最適化エラー: {e}")
            return {"success": False, "message": str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報取得"""
        
        return {
            "total_evaluations": self.total_evaluations,
            "successful_matches": self.successful_matches,
            "success_rate": self.successful_matches / max(1, self.total_evaluations),
            "average_processing_time": self.average_processing_time,
            "criteria_supported": len(self.COMPLETE_CRITERIA),
            "features_enabled": {
                "complete_13_criteria": True,
                "weighted_calculation": True,
                "genetic_optimization": self.config.enable_genetic_optimization,
                "fuzzy_inference": self.config.use_fuzzy_inference,
                "field_bonus": self.config.field_bonus_enabled
            },
            "cache_size": len(self.result_cache)
        }