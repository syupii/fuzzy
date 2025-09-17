# core/ai_system.py - 統合AIシステム

import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# 各AIコンポーネントのインポート
from core.fuzzy.inference import SimpleFuzzyInferenceEngine, FuzzyRule
from core.genetic.evolution import EvolutionEngine, EvolutionConfig, Individual, FitnessEvaluator
from core.decision_tree.tree import FuzzyDecisionTree

logger = logging.getLogger(__name__)

@dataclass
class AISystemConfig:
    """AI統合システム設定"""
    # ファジィシステム設定
    fuzzy_enabled: bool = True
    fuzzy_weight: float = 0.4
    
    # 遺伝的アルゴリズム設定
    genetic_enabled: bool = True
    genetic_weight: float = 0.3
    ga_population_size: int = 30
    ga_generations: int = 50
    ga_mutation_rate: float = 0.1
    
    # 決定木設定
    decision_tree_enabled: bool = True
    tree_weight: float = 0.3
    tree_max_depth: int = 5
    
    # 統合設定
    ensemble_method: str = "weighted_average"  # weighted_average, voting, stacking
    optimization_enabled: bool = True
    cache_results: bool = True

@dataclass
class EvaluationResult:
    """評価結果"""
    lab_id: str
    lab_name: str
    
    # 各システムの結果
    fuzzy_score: float
    genetic_score: float
    decision_tree_score: float
    
    # 統合結果
    final_score: float
    confidence: float
    
    # 詳細情報
    fuzzy_explanation: Dict[str, Any]
    genetic_explanation: Dict[str, Any]
    decision_tree_explanation: Dict[str, Any]
    
    # メタデータ
    processing_time: float
    timestamp: str

class IntegratedAISystem:
    """遺伝的アルゴリズムを用いたファジィ決定木統合システム"""
    
    def __init__(self, config: AISystemConfig = None):
        self.config = config or AISystemConfig()
        
        # AIコンポーネントの初期化
        self.fuzzy_engine: Optional[SimpleFuzzyInferenceEngine] = None
        self.genetic_engine: Optional[EvolutionEngine] = None
        self.decision_tree: Optional[FuzzyDecisionTree] = None
        
        # システム状態
        self.is_initialized = False
        self.is_optimized = False
        self.evaluation_count = 0
        
        # 最適化された重み
        self.optimized_weights: Optional[List[float]] = None
        
        # 結果キャッシュ
        self.result_cache: Dict[str, EvaluationResult] = {}
        
        # 統計情報
        self.stats = {
            "total_evaluations": 0,
            "average_processing_time": 0.0,
            "accuracy_history": [],
            "optimization_history": []
        }
        
        logger.info("統合AIシステムを初期化しました")
    
    def initialize(self, training_data: Optional[List[Dict[str, Any]]] = None):
        """システムの初期化"""
        logger.info("統合AIシステム初期化開始...")
        
        try:
            # ファジィ推論エンジン初期化
            if self.config.fuzzy_enabled:
                self.fuzzy_engine = SimpleFuzzyInferenceEngine()
                logger.info("ファジィ推論エンジンを初期化しました")
            
            # 遺伝的アルゴリズム初期化
            if self.config.genetic_enabled:
                ga_config = EvolutionConfig(
                    population_size=self.config.ga_population_size,
                    generations=self.config.ga_generations,
                    mutation_rate=self.config.ga_mutation_rate
                )
                self.genetic_engine = EvolutionEngine(ga_config, self.fuzzy_engine)
                logger.info("遺伝的アルゴリズムエンジンを初期化しました")
            
            # ファジィ決定木初期化
            if self.config.decision_tree_enabled:
                self.decision_tree = FuzzyDecisionTree(
                    max_depth=self.config.tree_max_depth
                )
                
                # 学習データがある場合は決定木を学習
                if training_data:
                    self._train_decision_tree(training_data)
                else:
                    self._generate_synthetic_training_data()
                
                logger.info("ファジィ決定木を初期化しました")
            
            self.is_initialized = True
            logger.info("✅ 統合AIシステム初期化完了")
            
        except Exception as e:
            logger.error(f"❌ 統合AIシステム初期化エラー: {e}")
            raise
    
    def _generate_synthetic_training_data(self):
        """合成学習データの生成"""
        logger.info("合成学習データを生成中...")
        
        # 13次元の評価基準で合成データを生成
        criteria = [
            "research_intensity", "advisor_style", "team_work", "workload",
            "theory_practice", "research_field_match", "skill_development",
            "lab_atmosphere", "flexibility", "publication_opportunity",
            "interdisciplinary", "communication_style", "innovation_risk"
        ]
        
        X = []
        y = []
        
        # 1000個の合成サンプルを生成
        for _ in range(1000):
            sample = {}
            
            # ランダムな評価基準値を生成
            for criterion in criteria:
                sample[criterion] = np.random.uniform(1.0, 10.0)
            
            # 適合度クラスを決定（5段階）
            # 高い値ほど高適合度となるようなルールベースの分類
            score = 0.0
            score += sample["research_field_match"] * 0.3  # 分野適合性が重要
            score += sample["research_intensity"] * 0.2
            score += sample["publication_opportunity"] * 0.2
            score += sample["advisor_style"] * 0.1
            score += sum(sample[c] for c in criteria[4:]) * 0.2 / len(criteria[4:])
            
            # 正規化してクラス分類
            normalized_score = (score - 1.0) / 9.0
            
            if normalized_score < 0.2:
                compatibility_class = "very_low"
            elif normalized_score < 0.4:
                compatibility_class = "low"
            elif normalized_score < 0.6:
                compatibility_class = "medium"
            elif normalized_score < 0.8:
                compatibility_class = "high"
            else:
                compatibility_class = "very_high"
            
            X.append(sample)
            y.append(compatibility_class)
        
        # 決定木を学習
        if self.decision_tree:
            self.decision_tree.fit(X, y)
            logger.info(f"合成データで決定木を学習しました（サンプル数: {len(X)}）")
    
    def _train_decision_tree(self, training_data: List[Dict[str, Any]]):
        """実データでの決定木学習"""
        X = []
        y = []
        
        for data in training_data:
            if "student_profile" in data and "compatibility" in data:
                X.append(data["student_profile"])
                y.append(data["compatibility"])
        
        if X and self.decision_tree:
            self.decision_tree.fit(X, y)
            logger.info(f"決定木を学習しました（サンプル数: {len(X)}）")
    
    def optimize_system(self, student_profiles: List[Dict[str, float]], 
                       lab_profiles: List[Dict[str, float]]):
        """遺伝的アルゴリズムによるシステム最適化"""
        if not self.config.optimization_enabled or not self.genetic_engine:
            logger.warning("最適化が無効または遺伝的アルゴリズムが利用できません")
            return
        
        logger.info("遺伝的アルゴリズムによる重み最適化を開始...")
        
        try:
            # 適応度評価器を設定
            fitness_evaluator = FitnessEvaluator(
                student_profiles, lab_profiles, self.fuzzy_engine
            )
            
            # 初期集団を生成（13次元：評価基準の重み）
            self.genetic_engine.initialize_population(chromosome_length=13)
            
            # 進化実行
            final_population = self.genetic_engine.evolve(
                fitness_evaluator, max_generations=self.config.ga_generations
            )
            
            # 最適重みを取得
            self.optimized_weights = self.genetic_engine.get_best_weights()
            self.is_optimized = True
            
            # 最適化履歴を記録
            evolution_summary = self.genetic_engine.get_evolution_summary()
            self.stats["optimization_history"].append(evolution_summary)
            
            logger.info(f"✅ 最適化完了: 最良適応度 {evolution_summary['best_fitness']:.4f}")
            
        except Exception as e:
            logger.error(f"❌ 最適化エラー: {e}")
            raise
    
    def evaluate_compatibility(self, student_profile: Dict[str, float], 
                             lab_profile: Dict[str, float]) -> EvaluationResult:
        """統合適合性評価"""
        
        if not self.is_initialized:
            raise ValueError("システムが初期化されていません")
        
        start_time = time.time()
        lab_id = lab_profile.get("lab_id", "unknown")
        lab_name = lab_profile.get("lab_name", "Unknown Lab")
        
        # キャッシュチェック
        cache_key = f"{hash(str(student_profile))}_{lab_id}"
        if self.config.cache_results and cache_key in self.result_cache:
            return self.result_cache[cache_key]
        
        # 各システムによる評価
        fuzzy_score = 0.0
        genetic_score = 0.0
        decision_tree_score = 0.0
        
        fuzzy_explanation = {}
        genetic_explanation = {}
        decision_tree_explanation = {}
        
        # 1. ファジィ推論による評価
        if self.config.fuzzy_enabled and self.fuzzy_engine:
            try:
                fuzzy_score = self.fuzzy_engine.infer_lab_compatibility(
                    student_profile, lab_profile
                )
                fuzzy_explanation = self.fuzzy_engine.explain_inference(student_profile)
                logger.debug(f"ファジィスコア: {fuzzy_score:.3f}")
            except Exception as e:
                logger.warning(f"ファジィ推論エラー: {e}")
                fuzzy_score = 0.5
        
        # 2. 遺伝的アルゴリズム最適化重みによる評価
        if self.config.genetic_enabled and self.optimized_weights:
            try:
                genetic_score = self._calculate_genetic_score(
                    student_profile, lab_profile, self.optimized_weights
                )
                genetic_explanation = {
                    "optimized_weights": self.optimized_weights,
                    "evolution_generations": self.genetic_engine.generation if self.genetic_engine else 0
                }
                logger.debug(f"遺伝的スコア: {genetic_score:.3f}")
            except Exception as e:
                logger.warning(f"遺伝的アルゴリズム評価エラー: {e}")
                genetic_score = 0.5
        
        # 3. ファジィ決定木による評価
        if self.config.decision_tree_enabled and self.decision_tree:
            try:
                tree_prediction = self.decision_tree.predict(student_profile)
                
                # クラス確率を数値スコアに変換
                class_mapping = {
                    "very_low": 0.1, "low": 0.3, "medium": 0.5, 
                    "high": 0.7, "very_high": 0.9
                }
                
                predicted_class = tree_prediction["predicted_class"]
                decision_tree_score = class_mapping.get(predicted_class, 0.5)
                decision_tree_score *= tree_prediction["confidence"]
                
                decision_tree_explanation = tree_prediction
                logger.debug(f"決定木スコア: {decision_tree_score:.3f}")
            except Exception as e:
                logger.warning(f"決定木評価エラー: {e}")
                decision_tree_score = 0.5
        
        # 4. 統合評価
        final_score, confidence = self._integrate_scores(
            fuzzy_score, genetic_score, decision_tree_score
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
            fuzzy_explanation=fuzzy_explanation,
            genetic_explanation=genetic_explanation,
            decision_tree_explanation=decision_tree_explanation,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        # キャッシュに保存
        if self.config.cache_results:
            self.result_cache[cache_key] = result
        
        # 統計更新
        self.evaluation_count += 1
        self.stats["total_evaluations"] += 1
        self._update_processing_time_stats(processing_time)
        
        return result
    
    def _calculate_genetic_score(self, student_profile: Dict[str, float], 
                                lab_profile: Dict[str, float], 
                                weights: List[float]) -> float:
        """遺伝的アルゴリズム最適化重みによるスコア計算"""
        
        criteria = [
            "research_intensity", "advisor_style", "team_work", "workload",
            "theory_practice", "research_field_match", "skill_development",
            "lab_atmosphere", "flexibility", "publication_opportunity",
            "interdisciplinary", "communication_style", "innovation_risk"
        ]
        
        if len(weights) < len(criteria):
            weights = weights + [1.0/len(criteria)] * (len(criteria) - len(weights))
        
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for i, criterion in enumerate(criteria):
            if criterion in student_profile and criterion in lab_profile:
                student_val = student_profile[criterion]
                lab_val = lab_profile[criterion]
                
                # 類似度計算
                similarity = 1.0 - abs(student_val - lab_val) / 9.0
                
                # 重み適用
                weight = weights[i] if i < len(weights) else 1.0/len(criteria)
                weighted_sum += similarity * weight
                weight_sum += weight
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    def _integrate_scores(self, fuzzy_score: float, genetic_score: float, 
                         decision_tree_score: float) -> Tuple[float, float]:
        """スコア統合"""
        
        if self.config.ensemble_method == "weighted_average":
            # 重み付き平均
            total_weight = 0.0
            weighted_sum = 0.0
            
            if self.config.fuzzy_enabled and fuzzy_score > 0:
                weighted_sum += fuzzy_score * self.config.fuzzy_weight
                total_weight += self.config.fuzzy_weight
            
            if self.config.genetic_enabled and genetic_score > 0:
                weighted_sum += genetic_score * self.config.genetic_weight
                total_weight += self.config.genetic_weight
            
            if self.config.decision_tree_enabled and decision_tree_score > 0:
                weighted_sum += decision_tree_score * self.config.tree_weight
                total_weight += self.config.tree_weight
            
            if total_weight > 0:
                final_score = weighted_sum / total_weight
                confidence = min(1.0, total_weight)  # 重みの合計が信頼度
            else:
                final_score = 0.5
                confidence = 0.1
        
        elif self.config.ensemble_method == "voting":
            # 多数決（閾値0.6で高/低を判定）
            votes = []
            if self.config.fuzzy_enabled:
                votes.append(1 if fuzzy_score > 0.6 else 0)
            if self.config.genetic_enabled:
                votes.append(1 if genetic_score > 0.6 else 0)
            if self.config.decision_tree_enabled:
                votes.append(1 if decision_tree_score > 0.6 else 0)
            
            if votes:
                vote_average = sum(votes) / len(votes)
                final_score = vote_average
                confidence = 1.0 - abs(0.5 - vote_average) * 2  # 0.5からの距離で信頼度
            else:
                final_score = 0.5
                confidence = 0.1
        
        else:
            # デフォルト：単純平均
            scores = []
            if self.config.fuzzy_enabled and fuzzy_score > 0:
                scores.append(fuzzy_score)
            if self.config.genetic_enabled and genetic_score > 0:
                scores.append(genetic_score)
            if self.config.decision_tree_enabled and decision_tree_score > 0:
                scores.append(decision_tree_score)
            
            if scores:
                final_score = sum(scores) / len(scores)
                confidence = len(scores) / 3  # アクティブなシステム数/総システム数
            else:
                final_score = 0.5
                confidence = 0.1
        
        return max(0.0, min(1.0, final_score)), max(0.0, min(1.0, confidence))
    
    def _update_processing_time_stats(self, processing_time: float):
        """処理時間統計の更新"""
        current_avg = self.stats["average_processing_time"]
        total_evals = self.stats["total_evaluations"]
        
        if total_evals == 1:
            self.stats["average_processing_time"] = processing_time
        else:
            self.stats["average_processing_time"] = (
                (current_avg * (total_evals - 1) + processing_time) / total_evals
            )
    
    def evaluate_multiple_labs(self, student_profile: Dict[str, float], 
                              lab_profiles: List[Dict[str, float]]) -> List[EvaluationResult]:
        """複数研究室の一括評価"""
        results = []
        
        for lab_profile in lab_profiles:
            try:
                result = self.evaluate_compatibility(student_profile, lab_profile)
                results.append(result)
            except Exception as e:
                logger.warning(f"研究室 {lab_profile.get('lab_name', 'Unknown')} の評価エラー: {e}")
        
        # スコア順にソート
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """システム状態取得"""
        return {
            "is_initialized": self.is_initialized,
            "is_optimized": self.is_optimized,
            "evaluation_count": self.evaluation_count,
            "components": {
                "fuzzy_engine": self.config.fuzzy_enabled and self.fuzzy_engine is not None,
                "genetic_engine": self.config.genetic_enabled and self.genetic_engine is not None,
                "decision_tree": self.config.decision_tree_enabled and self.decision_tree is not None
            },
            "configuration": {
                "fuzzy_weight": self.config.fuzzy_weight,
                "genetic_weight": self.config.genetic_weight,
                "tree_weight": self.config.tree_weight,
                "ensemble_method": self.config.ensemble_method
            },
            "statistics": self.stats,
            "optimized_weights": self.optimized_weights
        }
    
    def explain_evaluation(self, result: EvaluationResult) -> Dict[str, Any]:
        """評価結果の詳細説明"""
        return {
            "lab_info": {
                "lab_id": result.lab_id,
                "lab_name": result.lab_name
            },
            "scores": {
                "fuzzy_score": result.fuzzy_score,
                "genetic_score": result.genetic_score,
                "decision_tree_score": result.decision_tree_score,
                "final_score": result.final_score,
                "confidence": result.confidence
            },
            "explanations": {
                "fuzzy_explanation": result.fuzzy_explanation,
                "genetic_explanation": result.genetic_explanation,
                "decision_tree_explanation": result.decision_tree_explanation
            },
            "processing_info": {
                "processing_time": result.processing_time,
                "timestamp": result.timestamp,
                "ensemble_method": self.config.ensemble_method
            }
        }