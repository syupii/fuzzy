"""
進化アルゴリズム - core/genetic/evolution.py
遺伝的アルゴリズムのメイン進化エンジン
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np
import time
from dataclasses import dataclass
from enum import Enum

from .individual import GeneticIndividual, IndividualType
from .population import Population, SelectionMethod, ReplacementStrategy
from .operators import GeneticOperators, OperatorConfig, CrossoverType, MutationType


class EvolutionStrategy(Enum):
    """進化戦略"""
    STANDARD_GA = "standard_ga"
    ELITIST_GA = "elitist_ga"
    STEADY_STATE_GA = "steady_state_ga"
    ADAPTIVE_GA = "adaptive_ga"


@dataclass
class EvolutionConfig:
    """進化アルゴリズム設定"""
    # 基本パラメータ
    population_size: int = 50
    generations: int = 100
    elite_size: int = 5
    
    # 遺伝的操作
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    tournament_size: int = 3
    
    # 適応度重み
    fitness_weights: Dict[str, float] = None
    
    # 収束判定
    convergence_threshold: float = 0.001
    max_stagnant_generations: int = 20
    
    # 多様性管理
    diversity_threshold: float = 0.1
    diversity_injection_rate: float = 0.1
    
    # 進化戦略
    strategy: EvolutionStrategy = EvolutionStrategy.ELITIST_GA
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    replacement_strategy: ReplacementStrategy = ReplacementStrategy.ELITE_REPLACEMENT
    
    def __post_init__(self):
        if self.fitness_weights is None:
            self.fitness_weights = {
                'accuracy': 0.6,
                'complexity': 0.2,
                'interpretability': 0.2
            }


@dataclass
class EvolutionResult:
    """進化結果"""
    best_individual: Optional[GeneticIndividual]
    best_fitness: float
    final_population: List[GeneticIndividual]
    fitness_history: List[float]
    generation_count: int
    total_time: float
    convergence_generation: Optional[int]
    evaluation_count: int
    success: bool
    termination_reason: str


class EvolutionEngine:
    """進化エンジン"""
    
    def __init__(self, config: EvolutionConfig = None):
        self.config = config or EvolutionConfig()
        
        # 操作設定
        operator_config = OperatorConfig(
            crossover_rate=self.config.crossover_rate,
            mutation_rate=self.config.mutation_rate,
            mutation_strength=self.config.mutation_strength,
            tournament_size=self.config.tournament_size,
            elite_size=self.config.elite_size
        )
        
        self.operators = GeneticOperators(operator_config)
        self.population: Optional[Population] = None
        
        # 進化統計
        self.generation = 0
        self.evaluation_count = 0
        self.stagnation_counter = 0
        self.best_fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        
        # 適応的パラメータ
        self.adaptive_mutation_rate = self.config.mutation_rate
        self.adaptive_crossover_rate = self.config.crossover_rate
    
    def evolve(self, training_data: np.ndarray, test_data: np.ndarray,
               feature_names: List[str], target_name: str,
               fitness_function: Optional[Callable] = None) -> EvolutionResult:
        """進化の実行"""
        
        print(f"\n{'='*60}")
        print(f"遺伝的アルゴリズム進化開始")
        print(f"集団サイズ: {self.config.population_size}, 世代数: {self.config.generations}")
        print(f"戦略: {self.config.strategy.value}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # 初期化
            success = self._initialize_evolution(training_data, feature_names)
            if not success:
                return self._create_failed_result("Initialization failed")
            
            # 進化ループ
            termination_reason = self._evolution_loop(
                training_data, test_data, feature_names, target_name, fitness_function
            )
            
            # 結果作成
            total_time = time.time() - start_time
            result = self._create_evolution_result(termination_reason, total_time)
            
            print(f"\n{'='*60}")
            print(f"進化完了: {termination_reason}")
            print(f"最良適応度: {result.best_fitness:.6f}")
            print(f"総世代数: {result.generation_count}")
            print(f"実行時間: {total_time:.2f}秒")
            print(f"{'='*60}")
            
            return result
            
        except Exception as e:
            print(f"進化エラー: {e}")
            import traceback
            traceback.print_exc()
            return self._create_failed_result(f"Evolution error: {e}")
    
    def _initialize_evolution(self, training_data: np.ndarray, feature_names: List[str]) -> bool:
        """進化の初期化"""
        
        try:
            # 集団初期化
            self.population = Population(
                self.config.population_size,
                IndividualType.HYBRID
            )
            
            # ゲノム長計算
            genome_length = len(feature_names) * 10 + 20  # 基本長 + 木パラメータ用
            
            self.population.initialize_random(
                genome_length, 
                feature_names,
                max_depth=6,
                min_samples_leaf=5
            )
            
            print(f"集団初期化完了: {len(self.population.individuals)}個体")
            
            # 統計初期化
            self.generation = 0
            self.evaluation_count = 0
            self.stagnation_counter = 0
            self.best_fitness_history = []
            self.diversity_history = []
            
            return True
            
        except Exception as e:
            print(f"初期化エラー: {e}")
            return False
    
    def _evolution_loop(self, training_data: np.ndarray, test_data: np.ndarray,
                       feature_names: List[str], target_name: str,
                       fitness_function: Optional[Callable]) -> str:
        """進化ループの実行"""
        
        previous_best_fitness = 0.0
        
        for generation in range(self.config.generations):
            self.generation = generation
            
            # 集団評価
            eval_result = self.population.evaluate_population(
                training_data, test_data, feature_names, target_name, 
                self.config.fitness_weights
            )
            
            self.evaluation_count += eval_result['evaluated_count']
            
            # 統計更新
            current_best_fitness = self.population.best_individual.fitness if self.population.best_individual else 0.0
            self.best_fitness_history.append(current_best_fitness)
            
            # 進捗表示
            if generation % 10 == 0 or generation == self.config.generations - 1:
                avg_fitness = np.mean([ind.fitness for ind in self.population.individuals])
                print(f"世代 {generation:3d}: "
                      f"最良={current_best_fitness:.4f}, "
                      f"平均={avg_fitness:.4f}, "
                      f"評価数={eval_result['evaluated_count']}")
            
            # 収束判定
            if self._check_convergence(current_best_fitness, previous_best_fitness):
                return f"収束により終了 (世代 {generation})"
            
            # 多様性管理
            if self.population.is_low_diversity():
                injected = self.population.maintain_diversity(self.config.diversity_injection_rate)
                if injected > 0:
                    print(f"多様性注入: {injected}個体を置換")
            
            # 適応的パラメータ調整
            self._adapt_parameters(generation)
            
            # 次世代生成（最後の世代以外）
            if generation < self.config.generations - 1:
                self._create_next_generation()
            
            previous_best_fitness = current_best_fitness
        
        return f"最大世代数に到達 (世代 {self.config.generations})"
    
    def _check_convergence(self, current_fitness: float, previous_fitness: float) -> bool:
        """収束判定"""
        
        # 適応度の改善チェック
        improvement = current_fitness - previous_fitness
        
        if improvement < self.config.convergence_threshold:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        # 停滞による収束
        if self.stagnation_counter >= self.config.max_stagnant_generations:
            return True
        
        # 目標適応度達成
        if current_fitness >= 0.99:
            return True
        
        return False
    
    def _adapt_parameters(self, generation: int):
        """適応的パラメータ調整"""
        
        if self.config.strategy != EvolutionStrategy.ADAPTIVE_GA:
            return
        
        # 世代に基づく調整
        progress = generation / self.config.generations
        
        # 突然変異率の調整（初期は高く、後期は低く）
        if progress < 0.5:
            self.adaptive_mutation_rate = self.config.mutation_rate * (1.0 + progress)
        else:
            self.adaptive_mutation_rate = self.config.mutation_rate * (2.0 - progress)
        
        # 交叉率の調整
        if self.stagnation_counter > 5:
            self.adaptive_crossover_rate = min(0.95, self.config.crossover_rate * 1.1)
        else:
            self.adaptive_crossover_rate = self.config.crossover_rate
        
        # 操作設定更新
        self.operators.config.mutation_rate = self.adaptive_mutation_rate
        self.operators.config.crossover_rate = self.adaptive_crossover_rate
    
    def _create_next_generation(self):
        """次世代の作成"""
        
        if self.config.strategy == EvolutionStrategy.STANDARD_GA:
            self._standard_ga_generation()
        elif self.config.strategy == EvolutionStrategy.ELITIST_GA:
            self._elitist_ga_generation()
        elif self.config.strategy == EvolutionStrategy.STEADY_STATE_GA:
            self._steady_state_ga_generation()
        elif self.config.strategy == EvolutionStrategy.ADAPTIVE_GA:
            self._adaptive_ga_generation()
        else:
            self._elitist_ga_generation()
    
    def _standard_ga_generation(self):
        """標準遺伝的アルゴリズム世代交代"""
        new_individuals = self.operators.evolve_generation(self.population.individuals)
        self.population.individuals = new_individuals
        self.population.generation += 1
    
    def _elitist_ga_generation(self):
        """エリート保存遺伝的アルゴリズム世代交代"""
        # 親選択
        parents = self.population.select_parents(
            self.config.selection_method,
            self.config.tournament_size,
            self.config.population_size
        )
        
        # 子個体生成
        offspring = self.population.create_offspring(
            parents,
            self.adaptive_crossover_rate,
            self.adaptive_mutation_rate
        )
        
        # 世代交代
        self.population.replace_population(
            offspring,
            self.config.replacement_strategy,
            self.config.elite_size
        )
    
    def _steady_state_ga_generation(self):
        """定常状態遺伝的アルゴリズム世代交代"""
        # 少数の個体のみ置換
        num_replacements = max(2, self.config.population_size // 10)
        
        # 親選択
        parents = self.population.select_parents(
            self.config.selection_method,
            self.config.tournament_size,
            num_replacements * 2
        )
        
        # 子個体生成
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
            
            child1, child2 = self.operators.crossover_op.apply(parent1, parent2)
            child1 = self.operators.mutation_op.apply(child1)
            child2 = self.operators.mutation_op.apply(child2)
            
            offspring.extend([child1, child2])
        
        # 最悪個体を置換
        sorted_pop = sorted(self.population.individuals, key=lambda x: x.fitness)
        worst_individuals = sorted_pop[:num_replacements]
        
        for i, worst in enumerate(worst_individuals):
            if i < len(offspring):
                idx = self.population.individuals.index(worst)
                self.population.individuals[idx] = offspring[i]
        
        self.population.generation += 1
    
    def _adaptive_ga_generation(self):
        """適応的遺伝的アルゴリズム世代交代"""
        # 多様性に基づいて戦略を切り替え
        diversity = self.population._calculate_diversity()
        
        if diversity < self.config.diversity_threshold:
            # 多様性が低い場合：探索重視
            self._standard_ga_generation()
        else:
            # 多様性が高い場合：活用重視
            self._elitist_ga_generation()
    
    def _create_evolution_result(self, termination_reason: str, total_time: float) -> EvolutionResult:
        """進化結果の作成"""
        
        convergence_generation = None
        if "収束" in termination_reason:
            convergence_generation = self.generation
        
        return EvolutionResult(
            best_individual=self.population.best_individual.copy() if self.population.best_individual else None,
            best_fitness=self.population.best_individual.fitness if self.population.best_individual else 0.0,
            final_population=[ind.copy() for ind in self.population.individuals],
            fitness_history=self.best_fitness_history.copy(),
            generation_count=self.generation + 1,
            total_time=total_time,
            convergence_generation=convergence_generation,
            evaluation_count=self.evaluation_count,
            success=True,
            termination_reason=termination_reason
        )
    
    def _create_failed_result(self, error_message: str) -> EvolutionResult:
        """失敗結果の作成"""
        
        return EvolutionResult(
            best_individual=None,
            best_fitness=0.0,
            final_population=[],
            fitness_history=[],
            generation_count=0,
            total_time=0.0,
            convergence_generation=None,
            evaluation_count=0,
            success=False,
            termination_reason=error_message
        )
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """進化統計の取得"""
        
        pop_stats = self.population.get_population_summary() if self.population else {}
        
        return {
            'config': self.config.__dict__,
            'current_generation': self.generation,
            'evaluation_count': self.evaluation_count,
            'stagnation_counter': self.stagnation_counter,
            'best_fitness_history': self.best_fitness_history,
            'population_statistics': pop_stats,
            'operator_statistics': self.operators.get_operator_statistics(),
            'adaptive_parameters': {
                'mutation_rate': self.adaptive_mutation_rate,
                'crossover_rate': self.adaptive_crossover_rate
            }
        }
    
    def save_checkpoint(self, filepath: str) -> bool:
        """チェックポイント保存"""
        
        try:
            import pickle
            
            checkpoint_data = {
                'config': self.config,
                'generation': self.generation,
                'evaluation_count': self.evaluation_count,
                'best_fitness_history': self.best_fitness_history,
                'population': self.population.export_population() if self.population else None,
                'adaptive_parameters': {
                    'mutation_rate': self.adaptive_mutation_rate,
                    'crossover_rate': self.adaptive_crossover_rate
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            print(f"チェックポイント保存: {filepath}")
            return True
            
        except Exception as e:
            print(f"チェックポイント保存エラー: {e}")
            return False
    
    def load_checkpoint(self, filepath: str) -> bool:
        """チェックポイント読み込み"""
        
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.config = checkpoint_data['config']
            self.generation = checkpoint_data['generation']
            self.evaluation_count = checkpoint_data['evaluation_count']
            self.best_fitness_history = checkpoint_data['best_fitness_history']
            
            adaptive_params = checkpoint_data.get('adaptive_parameters', {})
            self.adaptive_mutation_rate = adaptive_params.get('mutation_rate', self.config.mutation_rate)
            self.adaptive_crossover_rate = adaptive_params.get('crossover_rate', self.config.crossover_rate)
            
            # TODO: 集団の復元（実装簡略化のため省略）
            
            print(f"チェックポイント読み込み: {filepath}")
            return True
            
        except Exception as e:
            print(f"チェックポイント読み込みエラー: {e}")
            return False


class MultiObjectiveEvolutionEngine(EvolutionEngine):
    """多目的進化エンジン"""
    
    def __init__(self, config: EvolutionConfig = None):
        super().__init__(config)
        self.pareto_front: List[GeneticIndividual] = []
    
    def _evaluate_multi_objective_fitness(self, individual: GeneticIndividual,
                                        test_data: np.ndarray, feature_names: List[str]) -> Tuple[float, List[float]]:
        """多目的適応度評価"""
        
        # 各目的の評価
        accuracy = individual._evaluate_accuracy(test_data, feature_names)
        complexity = individual._evaluate_complexity()
        interpretability = individual._evaluate_interpretability()
        
        objectives = [accuracy, 1.0 - complexity, interpretability]  # 最大化問題に統一
        
        # スカラー化（重み付き和）
        scalar_fitness = sum(w * obj for w, obj in zip(self.config.fitness_weights.values(), objectives))
        
        return scalar_fitness, objectives
    
    def _update_pareto_front(self, population: List[GeneticIndividual]):
        """パレートフロントの更新"""
        
        # 支配関係の計算
        dominated = set()
        
        for i, ind1 in enumerate(population):
            for j, ind2 in enumerate(population):
                if i != j and self._dominates(ind1, ind2):
                    dominated.add(j)
        
        # 非支配個体をパレートフロントに追加
        non_dominated = [population[i] for i in range(len(population)) if i not in dominated]
        
        self.pareto_front = non_dominated[:20]  # 最大20個体
    
    def _dominates(self, ind1: GeneticIndividual, ind2: GeneticIndividual) -> bool:
        """支配関係の判定"""
        
        # 簡易実装：適応度による支配関係
        return (ind1.fitness >= ind2.fitness and
                ind1.fitness_components.accuracy >= ind2.fitness_components.accuracy and
                ind1.fitness_components.complexity >= ind2.fitness_components.complexity and
                ind1.fitness_components.interpretability >= ind2.fitness_components.interpretability and
                (ind1.fitness > ind2.fitness or
                 ind1.fitness_components.accuracy > ind2.fitness_components.accuracy or
                 ind1.fitness_components.complexity > ind2.fitness_components.complexity or
                 ind1.fitness_components.interpretability > ind2.fitness_components.interpretability))


# ユーティリティ関数
def create_evolution_config(population_size: int = 30, generations: int = 50,
                           strategy: str = "elitist") -> EvolutionConfig:
    """進化設定の簡単作成"""
    
    strategy_enum = EvolutionStrategy.ELITIST_GA
    if strategy == "standard":
        strategy_enum = EvolutionStrategy.STANDARD_GA
    elif strategy == "steady_state":
        strategy_enum = EvolutionStrategy.STEADY_STATE_GA
    elif strategy == "adaptive":
        strategy_enum = EvolutionStrategy.ADAPTIVE_GA
    
    return EvolutionConfig(
        population_size=population_size,
        generations=generations,
        strategy=strategy_enum,
        elite_size=max(1, population_size // 10),
        tournament_size=min(5, max(2, population_size // 10))
    )