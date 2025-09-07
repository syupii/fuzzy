# services/optimization.py - 最適化サービス（完全版）

import numpy as np
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from enum import Enum
import uuid

from models.schemas import (
    StudentProfile, Laboratory, OptimizationRequest, 
    OptimizationResult, EvaluationResponse
)
from core.genetic.evolution import EvolutionEngine, EvolutionConfig, EvolutionResult
from core.genetic.individual import Individual, WeightVector, FuzzyTreeIndividual
from core.genetic.population import Population, PopulationConfig
from core.genetic.operators import OperatorConfig, SelectionMethod, CrossoverMethod, MutationMethod
from core.fuzzy.inference import FuzzyInferenceEngine, SimpleFuzzyInferenceEngine
from core.decision_tree.tree import FuzzyDecisionTree
from core.decision_tree.builder import BuilderConfig
from services.lab_matching import LabMatchingService
from utils.metrics import CompatibilityMetrics, PredictionEvaluator
from models.storage import ModelStorage, SpecializedStorage
from config.settings import settings

logger = logging.getLogger(__name__)

class OptimizationType(str, Enum):
    """最適化タイプ"""
    WEIGHT_OPTIMIZATION = "weight_optimization"
    TREE_OPTIMIZATION = "tree_optimization"
    HYBRID_OPTIMIZATION = "hybrid_optimization"
    MULTI_OBJECTIVE = "multi_objective"

class OptimizationStatus(str, Enum):
    """最適化ステータス"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class OptimizationConfig:
    """最適化設定"""
    # 基本設定
    optimization_type: OptimizationType = OptimizationType.WEIGHT_OPTIMIZATION
    max_runtime_minutes: int = 30
    target_fitness: Optional[float] = None
    
    # 遺伝的アルゴリズム設定
    population_size: int = 50
    max_generations: int = 100
    elite_size: int = 5
    
    # 選択・交叉・変異設定
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    crossover_method: CrossoverMethod = CrossoverMethod.UNIFORM
    mutation_method: MutationMethod = MutationMethod.GAUSSIAN
    
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    
    # 適応設定
    adaptive_parameters: bool = True
    early_stopping: bool = True
    patience: int = 15
    min_improvement: float = 1e-6
    
    # 並列化設定
    parallel_evaluation: bool = True
    num_processes: int = 4
    
    # 多目的最適化設定
    objectives: List[str] = field(default_factory=lambda: ["accuracy", "diversity"])
    objective_weights: Dict[str, float] = field(default_factory=lambda: {"accuracy": 0.7, "diversity": 0.3})
    
    # 評価設定
    validation_split: float = 0.2
    cross_validation_folds: int = 3
    
    # その他
    random_seed: Optional[int] = None
    verbose: bool = True
    save_intermediate: bool = True

@dataclass
class OptimizationJob:
    """最適化ジョブ"""
    job_id: str
    optimization_type: OptimizationType
    config: OptimizationConfig
    training_data: List[Tuple[StudentProfile, Laboratory, float]]
    status: OptimizationStatus = OptimizationStatus.PENDING
    
    # 実行情報
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_generation: int = 0
    best_fitness: float = 0.0
    
    # 結果
    result: Optional[OptimizationResult] = None
    error_message: Optional[str] = None
    
    # 進行状況
    progress_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def get_progress(self) -> Dict[str, Any]:
        """進行状況の取得"""
        
        if self.status == OptimizationStatus.PENDING:
            return {"status": "pending", "progress": 0.0}
        
        if self.status == OptimizationStatus.RUNNING:
            progress = self.current_generation / self.config.max_generations if self.config.max_generations > 0 else 0.0
            return {
                "status": "running",
                "progress": min(progress, 1.0),
                "current_generation": self.current_generation,
                "best_fitness": self.best_fitness
            }
        
        return {"status": self.status.value, "progress": 1.0}

class OptimizationObjective:
    """最適化目的関数"""
    
    def __init__(self, name: str, weight: float = 1.0, maximize: bool = True):
        self.name = name
        self.weight = weight
        self.maximize = maximize
        
        # 評価用メトリクス
        self.compatibility_metrics = CompatibilityMetrics()
        self.evaluator = PredictionEvaluator()
    
    def evaluate(self, individual: Individual, 
                training_data: List[Tuple[StudentProfile, Laboratory, float]],
                context: Dict[str, Any] = None) -> float:
        """目的関数の評価"""
        
        try:
            if self.name == "accuracy":
                return self._evaluate_accuracy(individual, training_data)
            elif self.name == "diversity":
                return self._evaluate_diversity(individual, context)
            elif self.name == "complexity":
                return self._evaluate_complexity(individual)
            elif self.name == "robustness":
                return self._evaluate_robustness(individual, training_data)
            else:
                logger.warning(f"未知の目的関数: {self.name}")
                return 0.0
                
        except Exception as e:
            logger.error(f"目的関数評価エラー {self.name}: {e}")
            return 0.0
    
    def _evaluate_accuracy(self, individual: Individual, 
                          training_data: List[Tuple[StudentProfile, Laboratory, float]]) -> float:
        """精度の評価"""
        
        if not training_data:
            return 0.0
        
        predictions = []
        ground_truth = []
        
        # 簡易評価（実際の実装では詳細な予測を実行）
        for student, lab, target_score in training_data:
            try:
                # 個体による予測スコア計算
                if isinstance(individual, WeightVector):
                    predicted_score = self._calculate_weighted_score(student, lab, individual)
                else:
                    predicted_score = 0.5  # デフォルト値
                
                predictions.append(predicted_score)
                ground_truth.append(target_score)
                
            except Exception as e:
                logger.warning(f"予測エラー: {e}")
                continue
        
        if not predictions:
            return 0.0
        
        # 平均絶対誤差の逆数
        mae = self.compatibility_metrics.calculate_mae(predictions, ground_truth)
        accuracy = max(0.0, 1.0 - mae)
        
        return accuracy
    
    def _evaluate_diversity(self, individual: Individual, context: Dict[str, Any]) -> float:
        """多様性の評価"""
        
        if not context or "population" not in context:
            return 0.5
        
        population = context["population"]
        
        if len(population) < 2:
            return 1.0
        
        # 他個体との多様性を計算
        total_diversity = 0.0
        comparison_count = 0
        
        for other in population:
            if other != individual:
                diversity = individual.get_diversity_from(other)
                total_diversity += diversity
                comparison_count += 1
        
        return total_diversity / comparison_count if comparison_count > 0 else 0.0
    
    def _evaluate_complexity(self, individual: Individual) -> float:
        """複雑度の評価（低いほど良い）"""
        
        if isinstance(individual, WeightVector):
            # 重みベクトルの複雑度（非ゼロ要素数）
            genes = individual.get_genes()
            non_zero_count = sum(1 for value in genes.values() if abs(value) > 1e-6)
            complexity = non_zero_count / len(genes) if genes else 1.0
            
            return 1.0 - complexity  # 複雑度が低いほど高スコア
        
        elif isinstance(individual, FuzzyTreeIndividual):
            # 決定木の複雑度
            tree_params = individual.tree_parameters
            max_depth = tree_params.get("max_depth", 10)
            complexity = max_depth / 15.0  # 正規化
            
            return 1.0 - complexity
        
        return 0.5
    
    def _evaluate_robustness(self, individual: Individual,
                            training_data: List[Tuple[StudentProfile, Laboratory, float]]) -> float:
        """頑健性の評価"""
        
        if len(training_data) < 5:
            return 0.5
        
        # ノイズを加えたデータでの性能評価
        noise_levels = [0.05, 0.1, 0.15]
        robustness_scores = []
        
        for noise_level in noise_levels:
            noisy_predictions = []
            ground_truth = []
            
            for student, lab, target_score in training_data[:10]:  # サブサンプル
                try:
                    # ノイズを加えた予測
                    if isinstance(individual, WeightVector):
                        base_score = self._calculate_weighted_score(student, lab, individual)
                        noise = np.random.normal(0, noise_level)
                        noisy_score = np.clip(base_score + noise, 0, 1)
                        noisy_predictions.append(noisy_score)
                        ground_truth.append(target_score)
                        
                except Exception:
                    continue
            
            if noisy_predictions:
                mae = self.compatibility_metrics.calculate_mae(noisy_predictions, ground_truth)
                robustness_scores.append(max(0.0, 1.0 - mae))
        
        return np.mean(robustness_scores) if robustness_scores else 0.5
    
    def _calculate_weighted_score(self, student: StudentProfile, 
                                 laboratory: Laboratory, 
                                 weights: WeightVector) -> float:
        """重み付きスコアの計算"""
        
        # 基本的な適合性計算
        student_criteria = student.evaluation_criteria.dict()
        lab_criteria = laboratory.characteristics.dict()
        weight_genes = weights.get_genes()
        
        total_score = 0.0
        total_weight = 0.0
        
        for criterion, student_value in student_criteria.items():
            if student_value is not None:
                lab_value = lab_criteria.get(criterion, 5.0)
                if lab_value is not None:
                    # 差分ベースの適合性
                    diff = abs(student_value - lab_value)
                    similarity = max(0.0, 1.0 - diff / 9.0)
                    
                    # 重みの適用
                    weight = weight_genes.get(criterion, 0.1)
                    total_score += similarity * weight
                    total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0

class MultiObjectiveOptimizer:
    """多目的最適化器"""
    
    def __init__(self, objectives: List[OptimizationObjective]):
        self.objectives = objectives
    
    def evaluate_individual(self, individual: Individual,
                           training_data: List[Tuple[StudentProfile, Laboratory, float]],
                           context: Dict[str, Any] = None) -> float:
        """個体の多目的評価"""
        
        objective_scores = []
        
        for objective in self.objectives:
            score = objective.evaluate(individual, training_data, context)
            
            # 最大化/最小化の調整
            if not objective.maximize:
                score = 1.0 - score
            
            # 重み付き
            weighted_score = score * objective.weight
            objective_scores.append(weighted_score)
        
        # 重み付き平均
        total_weight = sum(obj.weight for obj in self.objectives)
        final_score = sum(objective_scores) / total_weight if total_weight > 0 else 0.0
        
        return final_score
    
    def get_pareto_front(self, population: List[Individual],
                        training_data: List[Tuple[StudentProfile, Laboratory, float]]) -> List[Individual]:
        """パレート最適解の取得"""
        
        # 各個体の目的関数値を計算
        objective_values = []
        
        for individual in population:
            values = []
            for objective in self.objectives:
                score = objective.evaluate(individual, training_data)
                values.append(score)
            objective_values.append(values)
        
        # パレート支配関係の計算
        pareto_front = []
        
        for i, individual in enumerate(population):
            is_dominated = False
            
            for j, other_values in enumerate(objective_values):
                if i != j:
                    # 支配関係のチェック
                    if self._dominates(other_values, objective_values[i]):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_front.append(individual)
        
        return pareto_front
    
    def _dominates(self, values1: List[float], values2: List[float]) -> bool:
        """支配関係のチェック"""
        
        better_in_any = False
        
        for v1, v2, obj in zip(values1, values2, self.objectives):
            if obj.maximize:
                if v1 < v2:
                    return False
                elif v1 > v2:
                    better_in_any = True
            else:
                if v1 > v2:
                    return False
                elif v1 < v2:
                    better_in_any = True
        
        return better_in_any

class OptimizationService:
    """最適化サービス"""
    
    def __init__(self, storage: Optional[SpecializedStorage] = None):
        self.storage = storage
        self.active_jobs: Dict[str, OptimizationJob] = {}
        self.completed_jobs: List[OptimizationJob] = []
        
        # 実行プール
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.process_pool = ProcessPoolExecutor(max_workers=2)
        
        # 統計情報
        self.total_optimizations = 0
        self.successful_optimizations = 0
        
        # サービス
        self.lab_matching_service: Optional[LabMatchingService] = None
    
    def submit_optimization(self, request: OptimizationRequest) -> str:
        """最適化ジョブの投入"""
        
        try:
            # ジョブIDの生成
            job_id = str(uuid.uuid4())[:12]
            
            # 設定の変換
            config = OptimizationConfig(
                optimization_type=OptimizationType.WEIGHT_OPTIMIZATION,
                population_size=request.population_size,
                max_generations=request.generations,
                crossover_rate=request.crossover_rate,
                mutation_rate=request.mutation_rate,
                max_runtime_minutes=request.timeout_seconds // 60,
                verbose=request.verbose
            )
            
            # 訓練データの準備
            training_data = self._prepare_training_data(request)
            
            # ジョブの作成
            job = OptimizationJob(
                job_id=job_id,
                optimization_type=config.optimization_type,
                config=config,
                training_data=training_data
            )
            
            self.active_jobs[job_id] = job
            
            # バックグラウンドで実行
            self.thread_pool.submit(self._run_optimization, job)
            
            logger.info(f"最適化ジョブ投入: {job_id}")
            return job_id
            
        except Exception as e:
            logger.error(f"最適化ジョブ投入エラー: {e}")
            raise
    
    def get_optimization_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """最適化ステータスの取得"""
        
        # アクティブジョブから検索
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            return {
                "job_id": job_id,
                "status": job.status.value,
                "progress": job.get_progress(),
                "start_time": job.start_time.isoformat() if job.start_time else None,
                "current_generation": job.current_generation,
                "best_fitness": job.best_fitness,
                "error_message": job.error_message
            }
        
        # 完了ジョブから検索
        for job in self.completed_jobs:
            if job.job_id == job_id:
                return {
                    "job_id": job_id,
                    "status": job.status.value,
                    "progress": job.get_progress(),
                    "start_time": job.start_time.isoformat() if job.start_time else None,
                    "end_time": job.end_time.isoformat() if job.end_time else None,
                    "result": job.result.dict() if job.result else None,
                    "error_message": job.error_message
                }
        
        return None
    
    def cancel_optimization(self, job_id: str) -> bool:
        """最適化のキャンセル"""
        
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            job.status = OptimizationStatus.CANCELLED
            logger.info(f"最適化キャンセル: {job_id}")
            return True
        
        return False
    
    def _prepare_training_data(self, request: OptimizationRequest) -> List[Tuple[StudentProfile, Laboratory, float]]:
        """訓練データの準備"""
        
        training_data = []
        
        # リクエストから学生と研究室のペアを作成
        for student in request.student_profiles:
            for lab in request.target_labs:
                # 簡易的な適合性スコアを計算（実際のアプリケーションではより詳細）
                target_score = self._calculate_target_score(student, lab)
                training_data.append((student, lab, target_score))
        
        return training_data
    
    def _calculate_target_score(self, student: StudentProfile, laboratory: Laboratory) -> float:
        """目標スコアの計算"""
        
        # 分野適合性
        field_match = 0.0
        for interest in student.field_interests:
            if interest.field == laboratory.research_field:
                field_match = interest.interest_level / 10.0
                break
        
        # 基本基準の適合性
        student_criteria = student.evaluation_criteria.dict()
        lab_criteria = laboratory.characteristics.dict()
        
        criterion_matches = []
        for criterion in ["research_intensity", "advisor_style", "team_work", "workload", "theory_practice"]:
            student_val = student_criteria.get(criterion, 5.0)
            lab_val = lab_criteria.get(criterion, 5.0)
            
            if student_val is not None and lab_val is not None:
                diff = abs(student_val - lab_val)
                match = max(0.0, 1.0 - diff / 9.0)
                criterion_matches.append(match)
        
        criterion_avg = np.mean(criterion_matches) if criterion_matches else 0.5
        
        # 総合スコア
        target_score = 0.6 * criterion_avg + 0.4 * field_match
        
        return target_score
    
    def _run_optimization(self, job: OptimizationJob):
        """最適化の実行"""
        
        try:
            job.status = OptimizationStatus.RUNNING
            job.start_time = datetime.now()
            
            logger.info(f"最適化開始: {job.job_id}")
            
            # 進化設定の構築
            evolution_config = EvolutionConfig(
                population_size=job.config.population_size,
                max_generations=job.config.max_generations,
                selection_method=job.config.selection_method,
                crossover_method=job.config.crossover_method,
                mutation_method=job.config.mutation_method,
                crossover_rate=job.config.crossover_rate,
                mutation_rate=job.config.mutation_rate,
                mutation_strength=job.config.mutation_strength,
                target_fitness=job.config.target_fitness,
                max_runtime_seconds=job.config.max_runtime_minutes * 60,
                adaptive_parameters=job.config.adaptive_parameters,
                early_stopping=job.config.early_stopping,
                convergence_generations=job.config.patience,
                min_improvement=job.config.min_improvement,
                verbose=job.config.verbose,
                random_seed=job.config.random_seed
            )
            
            # 最適化の実行
            if job.config.optimization_type == OptimizationType.WEIGHT_OPTIMIZATION:
                result = self._run_weight_optimization(job, evolution_config)
            elif job.config.optimization_type == OptimizationType.MULTI_OBJECTIVE:
                result = self._run_multi_objective_optimization(job, evolution_config)
            else:
                raise ValueError(f"未対応の最適化タイプ: {job.config.optimization_type}")
            
            # 結果の保存
            job.result = result
            job.status = OptimizationStatus.COMPLETED
            job.end_time = datetime.now()
            
            # 最適化重みの保存
            if self.storage and result.success:
                self._save_optimization_result(job)
            
            self.successful_optimizations += 1
            logger.info(f"最適化完了: {job.job_id} (適応度: {result.best_fitness:.6f})")
            
        except Exception as e:
            job.status = OptimizationStatus.FAILED
            job.error_message = str(e)
            job.end_time = datetime.now()
            
            logger.error(f"最適化エラー {job.job_id}: {e}")
        
        finally:
            # アクティブジョブから完了ジョブに移動
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
            
            self.completed_jobs.append(job)
            self.total_optimizations += 1
            
            # 完了ジョブの履歴制限
            if len(self.completed_jobs) > 100:
                self.completed_jobs = self.completed_jobs[-50:]
    
    def _run_weight_optimization(self, job: OptimizationJob, 
                                evolution_config: EvolutionConfig) -> OptimizationResult:
        """重み最適化の実行"""
        
        # 進化エンジンの初期化
        evolution_engine = EvolutionEngine(evolution_config, WeightVector)
        
        # 重み名の決定
        weight_names = list(settings.evaluation_criteria)
        evolution_engine.initialize_population(weight_names=weight_names)
        
        # 目的関数の設定
        objectives = [
            OptimizationObjective("accuracy", weight=0.7, maximize=True),
            OptimizationObjective("diversity", weight=0.2, maximize=True),
            OptimizationObjective("complexity", weight=0.1, maximize=True)
        ]
        
        multi_objective_optimizer = MultiObjectiveOptimizer(objectives)
        
        # 適応度関数
        def fitness_function(individual: WeightVector) -> float:
            # ジョブキャンセルチェック
            if job.status == OptimizationStatus.CANCELLED:
                return 0.0
            
            # 進行状況の更新
            job.current_generation = evolution_engine.current_generation
            
            # 多目的評価
            context = {"population": evolution_engine.population.individuals}
            fitness = multi_objective_optimizer.evaluate_individual(
                individual, job.training_data, context
            )
            
            # 最高適応度の更新
            job.best_fitness = max(job.best_fitness, fitness)
            
            return fitness
        
        # 進化実行
        evolution_result = evolution_engine.evolve(fitness_function)
        
        # 結果の構築
        optimization_result = OptimizationResult(
            request_id=job.job_id,
            best_weights=evolution_result.best_individual.get_genes(),
            best_fitness=evolution_result.best_fitness,
            generation_history=[
                {
                    "generation": i,
                    "best_fitness": fitness,
                    "average_fitness": avg_fitness,
                    "diversity": diversity
                }
                for i, (fitness, avg_fitness, diversity) in enumerate(
                    zip(evolution_result.fitness_history,
                        evolution_engine.average_fitness_history,
                        evolution_result.diversity_history)
                )
            ],
            convergence_generation=evolution_result.convergence_generation,
            execution_time=evolution_result.execution_time,
            total_evaluations=evolution_result.total_evaluations,
            success=evolution_result.success,
            algorithm_config={
                "population_size": evolution_config.population_size,
                "generations": evolution_config.max_generations,
                "crossover_rate": evolution_config.crossover_rate,
                "mutation_rate": evolution_config.mutation_rate
            }
        )
        
        return optimization_result
    
    def _run_multi_objective_optimization(self, job: OptimizationJob,
                                        evolution_config: EvolutionConfig) -> OptimizationResult:
        """多目的最適化の実行"""
        
        # 進化エンジンの初期化
        evolution_engine = EvolutionEngine(evolution_config, WeightVector)
        weight_names = list(settings.evaluation_criteria)
        evolution_engine.initialize_population(weight_names=weight_names)
        
        # 多目的最適化器の設定
        objectives = []
        for obj_name, weight in job.config.objective_weights.items():
            maximize = obj_name in ["accuracy", "diversity"]
            objectives.append(OptimizationObjective(obj_name, weight, maximize))
        
        multi_objective_optimizer = MultiObjectiveOptimizer(objectives)
        
        # 適応度関数
        def fitness_function(individual: WeightVector) -> float:
            if job.status == OptimizationStatus.CANCELLED:
                return 0.0
            
            job.current_generation = evolution_engine.current_generation
            
            context = {"population": evolution_engine.population.individuals}
            fitness = multi_objective_optimizer.evaluate_individual(
                individual, job.training_data, context
            )
            
            job.best_fitness = max(job.best_fitness, fitness)
            return fitness
        
        # 進化実行
        evolution_result = evolution_engine.evolve(fitness_function)
        
        # パレート最適解の取得
        pareto_front = multi_objective_optimizer.get_pareto_front(
            evolution_engine.population.individuals, job.training_data
        )
        
        # 結果の構築（パレート最適解から最良を選択）
        best_individual = max(pareto_front, key=lambda x: x.get_fitness()) if pareto_front else evolution_result.best_individual
        
        optimization_result = OptimizationResult(
            request_id=job.job_id,
            best_weights=best_individual.get_genes(),
            best_fitness=best_individual.get_fitness(),
            generation_history=[
                {
                    "generation": i,
                    "best_fitness": fitness,
                    "pareto_front_size": len(pareto_front) if i == len(evolution_result.fitness_history) - 1 else 0
                }
                for i, fitness in enumerate(evolution_result.fitness_history)
            ],
            convergence_generation=evolution_result.convergence_generation,
            execution_time=evolution_result.execution_time,
            total_evaluations=evolution_result.total_evaluations,
            success=evolution_result.success,
            algorithm_config={
                "optimization_type": "multi_objective",
                "objectives": [obj.name for obj in objectives],
                "objective_weights": {obj.name: obj.weight for obj in objectives}
            }
        )
        
        return optimization_result
    
    def _save_optimization_result(self, job: OptimizationJob):
        """最適化結果の保存"""
        
        try:
            if not job.result or not job.result.success:
                return
            
            # 最適化重みの保存
            best_weights = WeightVector(weight_names=list(job.result.best_weights.keys()))
            best_weights.set_genes(job.result.best_weights)
            best_weights.set_fitness(job.result.best_fitness)
            
            optimization_info = {
                "job_id": job.job_id,
                "optimization_type": job.optimization_type.value,
                "execution_time": job.result.execution_time,
                "total_evaluations": job.result.total_evaluations,
                "convergence_generation": job.result.convergence_generation,
                "algorithm_config": job.result.algorithm_config
            }
            
            weights_id = f"optimized_weights_{job.job_id}_{int(time.time())}"
            
            self.storage.save_optimized_weights(
                weights=best_weights,
                weights_id=weights_id,
                optimization_info=optimization_info,
                description=f"最適化ジョブ {job.job_id} の結果"
            )
            
            logger.info(f"最適化結果保存完了: {weights_id}")
            
        except Exception as e:
            logger.error(f"最適化結果保存エラー: {e}")
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """サービス統計の取得"""
        
        return {
            "total_optimizations": self.total_optimizations,
            "successful_optimizations": self.successful_optimizations,
            "success_rate": self.successful_optimizations / max(self.total_optimizations, 1),
            "active_jobs": len(self.active_jobs),
            "completed_jobs": len(self.completed_jobs),
            "active_job_ids": list(self.active_jobs.keys())
        }
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """古いジョブのクリーンアップ"""
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # 完了ジョブのクリーンアップ
        self.completed_jobs = [
            job for job in self.completed_jobs
            if job.end_time and job.end_time > cutoff_time
        ]
        
        logger.info(f"古いジョブクリーンアップ完了: {len(self.completed_jobs)}件残存")
    
    def shutdown(self):
        """サービスのシャットダウン"""
        
        # アクティブジョブのキャンセル
        for job_id in list(self.active_jobs.keys()):
            self.cancel_optimization(job_id)
        
        # 実行プールのシャットダウン
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        logger.info("最適化サービスシャットダウン完了")

# 使用例とテスト
def test_optimization_service():
    """最適化サービスのテスト"""
    
    print("🧬 最適化サービステスト開始")
    
    # サービスの初期化
    service = OptimizationService()
    
    # テスト用データの作成
    from models.schemas import (
        StudentProfile, EvaluationCriteria, FieldInterest, 
        Laboratory, Faculty, ResearchFieldEnum, OptimizationRequest
    )
    
    # テスト学生プロフィール
    test_students = [
        StudentProfile(
            student_id="test_001",
            evaluation_criteria=EvaluationCriteria(
                research_intensity=8.0, advisor_style=7.0, team_work=6.0,
                workload=7.0, theory_practice=8.0
            ),
            field_interests=[
                FieldInterest(field=ResearchFieldEnum.AI_MACHINE_LEARNING, 
                             interest_level=9.0, priority=1)
            ]
        )
    ]
    
    # テスト研究室
    test_labs = [
        Laboratory(
            lab_id="lab_001",
            faculty=Faculty(name="テスト教授", specialties=["AI", "ML"]),
            research_field=ResearchFieldEnum.AI_MACHINE_LEARNING,
            characteristics=EvaluationCriteria(
                research_intensity=8.5, advisor_style=7.5, team_work=6.5,
                workload=7.0, theory_practice=8.0
            )
        )
    ]
    
    # 最適化リクエスト
    request = OptimizationRequest(
        student_profiles=test_students,
        target_labs=test_labs,
        population_size=10,
        generations=5,
        timeout_seconds=60,
        verbose=True
    )
    
    # 最適化ジョブの投入
    job_id = service.submit_optimization(request)
    print(f"✅ 最適化ジョブ投入: {job_id}")
    
    # ステータス確認
    import time
    for i in range(10):
        status = service.get_optimization_status(job_id)
        if status:
            print(f"📊 ステータス: {status['status']} (進行率: {status['progress']['progress']:.1%})")
            
            if status['status'] in ['completed', 'failed']:
                break
        
        time.sleep(2)
    
    # 最終結果
    final_status = service.get_optimization_status(job_id)
    if final_status and final_status['status'] == 'completed':
        result = final_status['result']
        print(f"🎯 最適化完了:")
        print(f"  最高適応度: {result['best_fitness']:.6f}")
        print(f"  実行時間: {result['execution_time']:.2f}秒")
        print(f"  総評価回数: {result['total_evaluations']}")
        
        # 最適重みの表示
        best_weights = result['best_weights']
        print(f"📊 最適重み (上位5つ):")
        sorted_weights = sorted(best_weights.items(), key=lambda x: x[1], reverse=True)
        for name, weight in sorted_weights[:5]:
            print(f"  {name}: {weight:.3f}")
    
    # 統計情報
    stats = service.get_service_statistics()
    print(f"\n📈 サービス統計:")
    print(f"  総最適化数: {stats['total_optimizations']}")
    print(f"  成功率: {stats['success_rate']:.3f}")
    print(f"  アクティブジョブ数: {stats['active_jobs']}")
    
    # シャットダウン
    service.shutdown()
    
    print("✅ 最適化サービステスト完了")

if __name__ == "__main__":
    test_optimization_service()