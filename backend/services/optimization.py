# services/optimization.py - 最適化サービス

from typing import Dict, List, Any, Optional, Tuple, Callable
import numpy as np
import time
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from models.schemas import StudentProfile, Laboratory, EvaluationResponse
from core.genetic.evolution import GeneticAlgorithm
from core.genetic.individual import Individual
from core.genetic.population import Population
from core.genetic.operators import OperatorFactory, SelectionMethod, CrossoverMethod, MutationMethod
from core.fuzzy.inference import FuzzyInferenceEngine
from core.decision_tree.builder import FuzzyTreeBuilder, BuilderConfig
from config.settings import settings

@dataclass
class OptimizationConfig:
    """最適化設定"""
    # 遺伝的アルゴリズム設定
    population_size: int = 30
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    
    # 選択・交叉・変異手法
    selection_method: str = "tournament"
    crossover_method: str = "uniform"
    mutation_method: str = "gaussian"
    
    # 適応度関数設定
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.4,
        "diversity": 0.2,
        "consistency": 0.2,
        "complexity_penalty": 0.1,
        "convergence_bonus": 0.1
    })
    
    # 停止条件
    early_stopping: bool = True
    patience: int = 10
    min_fitness_improvement: float = 1e-6
    max_runtime_seconds: int = 300  # 5分
    
    # その他
    verbose: bool = True
    random_seed: Optional[int] = None

@dataclass
class OptimizationResult:
    """最適化結果"""
    best_individual: Individual
    final_population: Population
    optimization_history: List[Dict[str, Any]]
    execution_time: float
    convergence_generation: int
    final_fitness: float
    improvement_rate: float
    status: str  # "completed", "early_stopped", "timeout", "failed"

class OptimizationObjective(ABC):
    """最適化目的関数の抽象基底クラス"""
    
    @abstractmethod
    def evaluate(self, individual: Individual, student_profile: StudentProfile,
                labs: List[Laboratory], context: Dict[str, Any]) -> float:
        """個体の適応度を評価"""
        pass

class LabMatchingObjective(OptimizationObjective):
    """研究室マッチング用目的関数"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.fuzzy_engine = FuzzyInferenceEngine()
    
    def evaluate(self, individual: Individual, student_profile: StudentProfile,
                labs: List[Laboratory], context: Dict[str, Any]) -> float:
        """研究室マッチングの適応度評価"""
        
        total_fitness = 0.0
        
        # 1. マッチング精度
        accuracy_score = self._calculate_accuracy(individual, student_profile, labs)
        
        # 2. 多様性
        diversity_score = self._calculate_diversity(individual, context.get("population", []))
        
        # 3. 一貫性
        consistency_score = self._calculate_consistency(individual, student_profile)
        
        # 4. 複雑性ペナルティ
        complexity_penalty = self._calculate_complexity_penalty(individual)
        
        # 5. 収束ボーナス
        convergence_bonus = self._calculate_convergence_bonus(individual, context)
        
        # 重み付き統合
        weights = self.config.fitness_weights
        total_fitness = (
            accuracy_score * weights["accuracy"] +
            diversity_score * weights["diversity"] +
            consistency_score * weights["consistency"] -
            complexity_penalty * weights["complexity_penalty"] +
            convergence_bonus * weights["convergence_bonus"]
        )
        
        return max(0.0, min(1.0, total_fitness))
    
    def _calculate_accuracy(self, individual: Individual, student: StudentProfile,
                          labs: List[Laboratory]) -> float:
        """マッチング精度を計算"""
        
        total_score = 0.0
        lab_count = 0
        
        for lab in labs:
            # 分野適合性
            field_score = self._evaluate_field_matching(individual, student, lab)
            
            # 評価基準適合性
            criteria_score = self._evaluate_criteria_matching(individual, student, lab)
            
            # 総合マッチングスコア
            lab_score = field_score * 0.6 + criteria_score * 0.4
            total_score += lab_score
            lab_count += 1
        
        return total_score / lab_count if lab_count > 0 else 0.0
    
    def _evaluate_field_matching(self, individual: Individual, student: StudentProfile,
                                lab: Laboratory) -> float:
        """分野適合性評価"""
        
        score = 0.0
        matched_fields = 0
        
        student_fields = {fi.field_id: fi for fi in student.field_interests}
        
        for field_id in lab.research_fields:
            if field_id in student_fields:
                student_interest = student_fields[field_id]
                field_weight = individual.field_weights.get(field_id, 0.0)
                
                # 興味度・経験・重要度を統合
                interest_score = (
                    student_interest.interest_level * 0.5 +
                    student_interest.experience_level * 0.3 +
                    student_interest.importance_level * 0.2
                ) / 10.0
                
                score += interest_score * field_weight
                matched_fields += 1
        
        return score / matched_fields if matched_fields > 0 else 0.0
    
    def _evaluate_criteria_matching(self, individual: Individual, student: StudentProfile,
                                   lab: Laboratory) -> float:
        """評価基準適合性評価"""
        
        score = 0.0
        criteria_count = 0
        
        student_criteria = student.evaluation_criteria.dict()
        lab_features = lab.features.dict()
        
        for criterion, weight in individual.criteria_weights.items():
            if criterion in student_criteria and criterion in lab_features:
                student_val = student_criteria[criterion]
                lab_val = lab_features[criterion]
                
                # ガウシアン類似度
                distance = abs(student_val - lab_val)
                similarity = np.exp(-(distance ** 2) / (2 * 2.0 ** 2))
                
                score += similarity * weight
                criteria_count += 1
        
        return score / criteria_count if criteria_count > 0 else 0.0
    
    def _calculate_diversity(self, individual: Individual, population: List[Individual]) -> float:
        """個体の多様性を計算"""
        
        if len(population) < 2:
            return 0.5  # デフォルト値
        
        # 他の個体との平均距離
        total_distance = 0.0
        comparison_count = 0
        
        for other in population:
            if other.individual_id != individual.individual_id:
                distance = individual.calculate_diversity(other)
                total_distance += distance
                comparison_count += 1
        
        avg_distance = total_distance / comparison_count if comparison_count > 0 else 0.0
        
        return min(1.0, avg_distance)
    
    def _calculate_consistency(self, individual: Individual, student: StudentProfile) -> float:
        """一貫性を計算"""
        
        # 学生の選択と重みの一貫性
        student_fields = {fi.field_id: fi for fi in student.field_interests}
        
        consistency_sum = 0.0
        field_count = 0
        
        for field_id, weight in individual.field_weights.items():
            if field_id in student_fields:
                student_importance = student_fields[field_id].importance_level / 10.0
                consistency = 1.0 - abs(weight - student_importance)
                consistency_sum += consistency
                field_count += 1
        
        field_consistency = consistency_sum / field_count if field_count > 0 else 0.0
        
        # 評価基準の一貫性も計算（簡略化）
        criteria_consistency = 0.8  # 固定値として簡略化
        
        return (field_consistency + criteria_consistency) / 2.0
    
    def _calculate_complexity_penalty(self, individual: Individual) -> float:
        """複雑性ペナルティを計算"""
        
        # 重みの分散（過度に複雑な重み分布にペナルティ）
        field_weights = list(individual.field_weights.values())
        criteria_weights = list(individual.criteria_weights.values())
        
        field_std = np.std(field_weights) if len(field_weights) > 1 else 0.0
        criteria_std = np.std(criteria_weights) if len(criteria_weights) > 1 else 0.0
        
        # 標準偏差が大きいほどペナルティが大きい（但し適度な分散は許可）
        field_penalty = max(0, field_std - 0.2) * 2.0
        criteria_penalty = max(0, criteria_std - 0.2) * 2.0
        
        return min(1.0, (field_penalty + criteria_penalty) / 2.0)
    
    def _calculate_convergence_bonus(self, individual: Individual, context: Dict[str, Any]) -> float:
        """収束ボーナスを計算"""
        
        generation = context.get("generation", 0)
        max_generations = context.get("max_generations", 50)
        
        # 世代の進行に応じてボーナス
        progress = generation / max_generations if max_generations > 0 else 0.0
        
        # 適応度履歴があれば改善率を考慮
        if len(individual.evaluation_history) > 1:
            recent_improvement = individual._calculate_improvement_rate()
            improvement_bonus = max(0, recent_improvement) * 0.5
        else:
            improvement_bonus = 0.0
        
        return min(1.0, progress * 0.3 + improvement_bonus)

class OptimizationService:
    """最適化サービスメインクラス"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        
        # 演算子設定
        self.selection_operator = OperatorFactory.create_selection_operator(
            self.config.selection_method,
            tournament_size=3
        )
        self.crossover_operator = OperatorFactory.create_crossover_operator(
            self.config.crossover_method,
            crossover_rate=self.config.crossover_rate
        )
        self.mutation_operator = OperatorFactory.create_mutation_operator(
            self.config.mutation_method,
            mutation_rate=self.config.mutation_rate
        )
        
        # 目的関数
        self.objective = LabMatchingObjective(self.config)
        
        # 最適化履歴
        self.optimization_history: List[OptimizationResult] = []
    
    def optimize_lab_matching(self, student_profile: StudentProfile,
                            labs: List[Laboratory]) -> OptimizationResult:
        """研究室マッチングの最適化を実行"""
        
        start_time = time.time()
        
        if self.config.verbose:
            print(f"🔧 最適化開始")
            print(f"   集団サイズ: {self.config.population_size}")
            print(f"   世代数: {self.config.generations}")
            print(f"   選択手法: {self.config.selection_method}")
        
        try:
            # 集団初期化
            population = self._initialize_population(student_profile)
            
            # 最適化実行
            optimization_result = self._run_optimization(
                population, student_profile, labs, start_time
            )
            
            # 結果保存
            self.optimization_history.append(optimization_result)
            
            if self.config.verbose:
                print(f"✅ 最適化完了")
                print(f"   実行時間: {optimization_result.execution_time:.2f}秒")
                print(f"   最終適応度: {optimization_result.final_fitness:.4f}")
                print(f"   収束世代: {optimization_result.convergence_generation}")
            
            return optimization_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # エラー時のダミー結果
            error_result = OptimizationResult(
                best_individual=Individual(),
                final_population=Population(1),
                optimization_history=[],
                execution_time=execution_time,
                convergence_generation=-1,
                final_fitness=0.0,
                improvement_rate=0.0,
                status="failed"
            )
            
            if self.config.verbose:
                print(f"❌ 最適化失敗: {str(e)}")
            
            return error_result
    
    def _initialize_population(self, student_profile: StudentProfile) -> Population:
        """集団初期化"""
        
        population = Population(self.config.population_size)
        
        # シード初期化（学生プロフィールベース）
        population.initialize_with_seeding(
            research_fields=list(settings.research_fields.keys()),
            evaluation_criteria=settings.evaluation_criteria,
            student_profile=student_profile,
            seed_ratio=0.3
        )
        
        return population
    
    def _run_optimization(self, population: Population, student_profile: StudentProfile,
                         labs: List[Laboratory], start_time: float) -> OptimizationResult:
        """最適化実行"""
        
        optimization_history = []
        best_fitness_history = []
        no_improvement_count = 0
        
        for generation in range(self.config.generations):
            
            # タイムアウトチェック
            if time.time() - start_time > self.config.max_runtime_seconds:
                status = "timeout"
                break
            
            # 適応度評価
            self._evaluate_population(population, student_profile, labs, generation)
            
            # 統計記録
            stats = population.get_population_summary()
            optimization_history.append({
                "generation": generation,
                "best_fitness": stats["fitness"]["best"],
                "avg_fitness": stats["fitness"]["average"],
                "diversity": stats["diversity"]["diversity"],
                "convergence_rate": stats["convergence"]["rate"]
            })
            
            best_fitness_history.append(stats["fitness"]["best"])
            
            # 進捗表示
            if self.config.verbose and generation % 10 == 0:
                print(f"   世代 {generation:2d}: 最高={stats['fitness']['best']:.4f}, "
                      f"平均={stats['fitness']['average']:.4f}, "
                      f"多様性={stats['diversity']['diversity']:.3f}")
            
            # 早期停止チェック
            if self.config.early_stopping and self._check_early_stopping(
                best_fitness_history, generation
            ):
                status = "early_stopped"
                break
            
            # 次世代生成
            if generation < self.config.generations - 1:
                population = self._generate_next_generation(population)
        else:
            status = "completed"
        
        # 結果作成
        execution_time = time.time() - start_time
        convergence_generation = self._find_convergence_generation(best_fitness_history)
        improvement_rate = self._calculate_improvement_rate(best_fitness_history)
        
        return OptimizationResult(
            best_individual=population.best_individual,
            final_population=population,
            optimization_history=optimization_history,
            execution_time=execution_time,
            convergence_generation=convergence_generation,
            final_fitness=population.best_individual.fitness if population.best_individual else 0.0,
            improvement_rate=improvement_rate,
            status=status
        )
    
    def _evaluate_population(self, population: Population, student_profile: StudentProfile,
                           labs: List[Laboratory], generation: int) -> None:
        """集団の適応度評価"""
        
        context = {
            "generation": generation,
            "max_generations": self.config.generations,
            "population": population.individuals
        }
        
        for individual in population.individuals:
            if individual.fitness == 0.0:  # 未評価の場合のみ
                fitness = self.objective.evaluate(individual, student_profile, labs, context)
                individual.update_fitness(fitness)
        
        population._update_best_individual()
        population._update_statistics()
    
    def _generate_next_generation(self, population: Population) -> Population:
        """次世代生成"""
        
        # 選択
        selected = self.selection_operator.apply(population.individuals, self.config.population_size)
        
        # 交叉
        offspring = []
        elite = population.get_elite()
        offspring.extend([ind.clone() for ind in elite])
        
        while len(offspring) < self.config.population_size:
            parent1, parent2 = np.random.choice(selected, 2, replace=False)
            child1, child2 = self.crossover_operator.apply(parent1, parent2)
            offspring.extend([child1, child2])
        
        offspring = offspring[:self.config.population_size]
        
        # 変異
        for i in range(self.config.elite_size, len(offspring)):
            offspring[i] = self.mutation_operator.apply(offspring[i])
        
        # 新しい集団作成
        new_population = Population(self.config.population_size)
        new_population.advance_generation(offspring)
        
        return new_population
    
    def _check_early_stopping(self, fitness_history: List[float], generation: int) -> bool:
        """早期停止判定"""
        
        if generation < self.config.patience:
            return False
        
        # 最近の改善率をチェック
        recent_fitness = fitness_history[-self.config.patience:]
        improvement = max(recent_fitness) - min(recent_fitness)
        
        return improvement < self.config.min_fitness_improvement
    
    def _find_convergence_generation(self, fitness_history: List[float]) -> int:
        """収束世代を特定"""
        
        if len(fitness_history) < 5:
            return len(fitness_history) - 1
        
        # 適応度の変化が安定した世代を探す
        for i in range(5, len(fitness_history)):
            recent_std = np.std(fitness_history[i-5:i])
            if recent_std < self.config.min_fitness_improvement:
                return i
        
        return len(fitness_history) - 1
    
    def _calculate_improvement_rate(self, fitness_history: List[float]) -> float:
        """改善率を計算"""
        
        if len(fitness_history) < 2:
            return 0.0
        
        initial_fitness = fitness_history[0]
        final_fitness = fitness_history[-1]
        
        return (final_fitness - initial_fitness) / max(initial_fitness, 1e-10)
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """最適化履歴のサマリーを取得"""
        
        if not self.optimization_history:
            return {"message": "最適化履歴がありません"}
        
        recent_result = self.optimization_history[-1]
        
        return {
            "total_optimizations": len(self.optimization_history),
            "latest_result": {
                "status": recent_result.status,
                "execution_time": recent_result.execution_time,
                "final_fitness": recent_result.final_fitness,
                "convergence_generation": recent_result.convergence_generation,
                "improvement_rate": recent_result.improvement_rate
            },
            "config": {
                "population_size": self.config.population_size,
                "generations": self.config.generations,
                "selection_method": self.config.selection_method,
                "crossover_method": self.config.crossover_method,
                "mutation_method": self.config.mutation_method
            },
            "average_performance": {
                "avg_execution_time": np.mean([r.execution_time for r in self.optimization_history]),
                "avg_final_fitness": np.mean([r.final_fitness for r in self.optimization_history]),
                "success_rate": len([r for r in self.optimization_history if r.status == "completed"]) / len(self.optimization_history)
            }
        }
    
    def clear_history(self) -> None:
        """最適化履歴をクリア"""
        self.optimization_history.clear()

class HyperparameterOptimizer:
    """ハイパーパラメータ最適化"""
    
    def __init__(self, parameter_space: Dict[str, Any]):
        self.parameter_space = parameter_space
        self.optimization_results: List[Tuple[Dict, float]] = []
    
    def optimize_hyperparameters(self, student_profile: StudentProfile,
                                labs: List[Laboratory], n_trials: int = 20) -> Dict[str, Any]:
        """ハイパーパラメータ最適化実行"""
        
        print(f"🎛️ ハイパーパラメータ最適化開始 ({n_trials}試行)")
        
        best_params = None
        best_score = -1.0
        
        for trial in range(n_trials):
            # パラメータサンプリング
            params = self._sample_parameters()
            
            # 最適化実行
            config = OptimizationConfig(**params)
            config.verbose = False  # 詳細出力を抑制
            
            optimizer = OptimizationService(config)
            result = optimizer.optimize_lab_matching(student_profile, labs)
            
            score = result.final_fitness
            self.optimization_results.append((params, score))
            
            if score > best_score:
                best_score = score
                best_params = params
            
            if (trial + 1) % 5 == 0:
                print(f"   試行 {trial + 1}/{n_trials}: 最高スコア = {best_score:.4f}")
        
        print(f"✅ ハイパーパラメータ最適化完了")
        print(f"   最高スコア: {best_score:.4f}")
        
        return {
            "best_parameters": best_params,
            "best_score": best_score,
            "all_results": self.optimization_results
        }
    
    def _sample_parameters(self) -> Dict[str, Any]:
        """パラメータをサンプリング"""
        
        params = {}
        
        for param_name, param_range in self.parameter_space.items():
            if isinstance(param_range, dict):
                if param_range["type"] == "int":
                    params[param_name] = np.random.randint(param_range["min"], param_range["max"] + 1)
                elif param_range["type"] == "float":
                    params[param_name] = np.random.uniform(param_range["min"], param_range["max"])
                elif param_range["type"] == "choice":
                    params[param_name] = np.random.choice(param_range["choices"])
            elif isinstance(param_range, list):
                params[param_name] = np.random.choice(param_range)
        
        return params