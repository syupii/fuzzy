from typing import List, Tuple, Dict, Any
import numpy as np
import random
from enum import Enum
from dataclasses import dataclass
from .individual import GeneticIndividual


class CrossoverType(Enum):
    """交叉の種類"""
    ONE_POINT = "one_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ARITHMETIC = "arithmetic"
    TREE_CROSSOVER = "tree_crossover"


class MutationType(Enum):
    """突然変異の種類"""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    SWAP = "swap"
    TREE_MUTATION = "tree_mutation"

@dataclass
class OperatorConfig:
    """遺伝的操作設定"""
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    tournament_size: int = 3
    elite_size: int = 3
    
    # 操作タイプ
    crossover_type: CrossoverType = CrossoverType.ONE_POINT
    mutation_type: MutationType = MutationType.GAUSSIAN
    
    # 適応制御
    adaptive_rates: bool = True
    diversity_threshold: float = 0.3


class GeneticOperators:
    """遺伝的操作集合"""
    
    def __init__(self, config: OperatorConfig = None):
        self.config = config or OperatorConfig()
        self.crossover_count = 0
        self.mutation_count = 0
    
    def crossover(self, parent1: GeneticIndividual, parent2: GeneticIndividual,
                 crossover_type: CrossoverType = CrossoverType.ONE_POINT,
                 crossover_rate: float = 0.8) -> Tuple[GeneticIndividual, GeneticIndividual]:
        """交叉操作"""
        
        self.crossover_count += 1
        
        if random.random() > crossover_rate:
            return parent1.clone(), parent2.clone()
        
        if crossover_type == CrossoverType.ONE_POINT:
            return self._one_point_crossover(parent1, parent2)
        elif crossover_type == CrossoverType.TWO_POINT:
            return self._two_point_crossover(parent1, parent2)
        elif crossover_type == CrossoverType.UNIFORM:
            return self._uniform_crossover(parent1, parent2)
        elif crossover_type == CrossoverType.ARITHMETIC:
            return self._arithmetic_crossover(parent1, parent2)
        else:
            return self._one_point_crossover(parent1, parent2)
    
    def _one_point_crossover(self, parent1: GeneticIndividual, 
                           parent2: GeneticIndividual) -> Tuple[GeneticIndividual, GeneticIndividual]:
        """一点交叉"""
        
        child1, child2 = parent1.crossover(parent2)
        return child1, child2
    
    def _two_point_crossover(self, parent1: GeneticIndividual, 
                           parent2: GeneticIndividual) -> Tuple[GeneticIndividual, GeneticIndividual]:
        """二点交叉"""
        
        if parent1.genome is None or parent2.genome is None:
            return parent1.clone(), parent2.clone()
        
        length = len(parent1.genome)
        if length < 3:
            return self._one_point_crossover(parent1, parent2)
        
        point1 = random.randint(1, length - 2)
        point2 = random.randint(point1 + 1, length - 1)
        
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        # 遺伝子交換
        child1.genome[point1:point2] = parent2.genome[point1:point2]
        child2.genome[point1:point2] = parent1.genome[point1:point2]
        
        return child1, child2
    
    def _uniform_crossover(self, parent1: GeneticIndividual, 
                          parent2: GeneticIndividual) -> Tuple[GeneticIndividual, GeneticIndividual]:
        """一様交叉"""
        
        if parent1.genome is None or parent2.genome is None:
            return parent1.clone(), parent2.clone()
        
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        for i in range(len(parent1.genome)):
            if random.random() < 0.5:
                child1.genome[i], child2.genome[i] = child2.genome[i], child1.genome[i]
        
        return child1, child2
    
    def _arithmetic_crossover(self, parent1: GeneticIndividual, 
                            parent2: GeneticIndividual) -> Tuple[GeneticIndividual, GeneticIndividual]:
        """算術交叉"""
        
        if parent1.genome is None or parent2.genome is None:
            return parent1.clone(), parent2.clone()
        
        alpha = random.uniform(0.2, 0.8)
        
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        child1.genome = alpha * parent1.genome + (1 - alpha) * parent2.genome
        child2.genome = alpha * parent2.genome + (1 - alpha) * parent1.genome
        
        return child1, child2
    
    def mutate(self, individual: GeneticIndividual,
              mutation_type: MutationType = MutationType.GAUSSIAN,
              mutation_rate: float = 0.1) -> GeneticIndividual:
        """突然変異操作"""
        
        self.mutation_count += 1
        
        if mutation_type == MutationType.GAUSSIAN:
            return self._gaussian_mutation(individual, mutation_rate)
        elif mutation_type == MutationType.UNIFORM:
            return self._uniform_mutation(individual, mutation_rate)
        elif mutation_type == MutationType.SWAP:
            return self._swap_mutation(individual, mutation_rate)
        else:
            return self._gaussian_mutation(individual, mutation_rate)
    
    def _gaussian_mutation(self, individual: GeneticIndividual, 
                          mutation_rate: float) -> GeneticIndividual:
        """ガウシアン突然変異"""
        
        mutated = individual.clone()
        
        if mutated.genome is not None:
            for i in range(len(mutated.genome)):
                if random.random() < mutation_rate:
                    noise = np.random.normal(0, 0.1)
                    mutated.genome[i] = np.clip(mutated.genome[i] + noise, 0.0, 1.0)
        
        return mutated
    
    def _uniform_mutation(self, individual: GeneticIndividual, 
                         mutation_rate: float) -> GeneticIndividual:
        """一様突然変異"""
        
        return individual.mutate(mutation_rate)
    
    def _swap_mutation(self, individual: GeneticIndividual, 
                      mutation_rate: float) -> GeneticIndividual:
        """スワップ突然変異"""
        
        mutated = individual.clone()
        
        if mutated.genome is not None and len(mutated.genome) >= 2:
            if random.random() < mutation_rate:
                i, j = random.sample(range(len(mutated.genome)), 2)
                mutated.genome[i], mutated.genome[j] = mutated.genome[j], mutated.genome[i]
        
        return mutated
    
    def get_operator_statistics(self) -> Dict[str, Any]:
        """操作統計の取得"""
        
        return {
            'total_crossovers': self.crossover_count,
            'total_mutations': self.mutation_count
        }


# core/genetic/evolution.py
"""
進化アルゴリズム - core/genetic/evolution.py
"""

from typing import Dict, Any, List, Optional, Callable
import numpy as np
import time
from dataclasses import dataclass

from .individual import GeneticIndividual, IndividualType
from .population import Population, PopulationConfig, SelectionMethod
from .operators import GeneticOperators, CrossoverType, MutationType


@dataclass
class EvolutionConfig:
    """進化設定"""
    population_size: int = 30
    max_generations: int = 20
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elite_size: int = 3
    tournament_size: int = 3
    convergence_threshold: float = 0.001
    max_stagnation: int = 5
    
    # 操作設定
    crossover_type: CrossoverType = CrossoverType.ONE_POINT
    mutation_type: MutationType = MutationType.GAUSSIAN
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    
    # 適応制御
    adaptive_parameters: bool = True
    diversity_maintenance: bool = True


class EvolutionEngine:
    """進化エンジン"""
    
    def __init__(self, config: EvolutionConfig = None):
        self.config = config or EvolutionConfig()
        
        # 集団設定
        pop_config = PopulationConfig(
            population_size=self.config.population_size,
            elite_size=self.config.elite_size,
            tournament_size=self.config.tournament_size,
            max_generations=self.config.max_generations,
            convergence_threshold=self.config.convergence_threshold,
            diversity_maintenance=self.config.diversity_maintenance
        )
        
        self.population = Population(pop_config)
        self.operators = GeneticOperators()
        
        # 実行統計
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.total_runtime: float = 0.0
        self.convergence_generation: int = -1
        
        # 適応制御
        self.current_crossover_rate = self.config.crossover_rate
        self.current_mutation_rate = self.config.mutation_rate
    
    def evolve(self, training_data: np.ndarray, feature_names: List[str],
              target_name: str, genome_length: int = 10) -> Dict[str, Any]:
        """進化実行"""
        
        print(f"🧬 進化開始: 世代数={self.config.max_generations}, 集団サイズ={self.config.population_size}")
        
        self.start_time = time.time()
        
        # 初期集団作成
        self.population.initialize_random(genome_length, feature_names, IndividualType.HYBRID)
        
        # 進化ループ
        for generation in range(self.config.max_generations):
            print(f"\n--- 世代 {generation} ---")
            
            # 適応度評価
            self.population.evaluate_fitness(training_data, feature_names, target_name)
            
            # 統計表示
            stats = self.population.get_statistics()
            print(f"最良適応度: {stats['best_fitness']:.4f}")
            print(f"平均適応度: {stats['average_fitness']:.4f}")
            print(f"多様性: {stats['diversity']:.4f}")
            
            # 収束判定
            if self.population.is_converged():
                print(f"収束検出: 世代{generation}")
                self.convergence_generation = generation
                break
            
            # 適応制御
            if self.config.adaptive_parameters:
                self._adapt_parameters(stats)
            
            # 次世代作成
            parents = self.population.select_parents(self.config.selection_method)
            next_generation = self._create_next_generation_advanced(parents)
            self.population.advance_generation(next_generation)
        
        self.end_time = time.time()
        self.total_runtime = self.end_time - self.start_time
        
        # 結果まとめ
        final_stats = self.population.get_statistics()
        
        result = {
            'evolution_completed': True,
            'total_runtime': self.total_runtime,
            'total_generations': generation + 1,
            'convergence_generation': self.convergence_generation,
            'best_individual': self.population.best_individual.to_dict() if self.population.best_individual else None,
            'final_population_stats': final_stats,
            'fitness_history': self.population.fitness_history,
            'diversity_history': self.population.diversity_history,
            'operator_stats': self.operators.get_operator_statistics(),
            'configuration': {
                'population_size': self.config.population_size,
                'max_generations': self.config.max_generations,
                'crossover_rate': self.config.crossover_rate,
                'mutation_rate': self.config.mutation_rate,
                'crossover_type': self.config.crossover_type.value,
                'mutation_type': self.config.mutation_type.value
            }
        }
        
        print(f"\n🎉 進化完了!")
        print(f"実行時間: {self.total_runtime:.3f}秒")
        print(f"最良適応度: {final_stats['best_fitness']:.4f}")
        
        return result
    
    def _create_next_generation_advanced(self, parents: List[GeneticIndividual]) -> List[GeneticIndividual]:
        """高度な次世代作成"""
        
        next_generation = []
        
        # エリート保存
        elite_count = self.config.elite_size
        for i in range(min(elite_count, len(self.population.individuals))):
            elite = self.population.individuals[i].clone()
            elite.generation = self.population.generation + 1
            next_generation.append(elite)
        
        # 交叉と突然変異
        while len(next_generation) < self.config.population_size:
            # 親選択
            parent1 = np.random.choice(parents)
            parent2 = np.random.choice(parents)
            
            # 交叉
            child1, child2 = self.operators.crossover(
                parent1, parent2, 
                self.config.crossover_type, 
                self.current_crossover_rate
            )
            
            # 突然変異
            child1 = self.operators.mutate(
                child1, 
                self.config.mutation_type, 
                self.current_mutation_rate
            )
            child2 = self.operators.mutate(
                child2, 
                self.config.mutation_type, 
                self.current_mutation_rate
            )
            
            # 世代設定
            child1.generation = self.population.generation + 1
            child2.generation = self.population.generation + 1
            
            next_generation.append(child1)
            if len(next_generation) < self.config.population_size:
                next_generation.append(child2)
        
        return next_generation[:self.config.population_size]
    
    def _adapt_parameters(self, stats: Dict[str, Any]):
        """パラメータ適応制御"""
        
        diversity = stats.get('diversity', 0.5)
        stagnation = stats.get('stagnation_count', 0)
        
        # 多様性が低い場合は突然変異率を上げる
        if diversity < 0.3:
            self.current_mutation_rate = min(0.3, self.current_mutation_rate * 1.1)
            print(f"  多様性低下 → 突然変異率上昇: {self.current_mutation_rate:.3f}")
        elif diversity > 0.7:
            self.current_mutation_rate = max(0.05, self.current_mutation_rate * 0.9)
            print(f"  多様性高 → 突然変異率低下: {self.current_mutation_rate:.3f}")
        
        # 停滞が続く場合は交叉率を調整
        if stagnation > 3:
            self.current_crossover_rate = min(0.95, self.current_crossover_rate * 1.05)
            print(f"  停滞検出 → 交叉率上昇: {self.current_crossover_rate:.3f}")
    
    def get_best_individual(self) -> Optional[GeneticIndividual]:
        """最良個体取得"""
        
        return self.population.best_individual
    
    def get_evolution_report(self) -> Dict[str, Any]:
        """進化レポート生成"""
        
        if not self.population.individuals:
            return {'error': 'No evolution data available'}
        
        final_stats = self.population.get_statistics()
        
        return {
            'summary': {
                'evolution_status': 'completed' if self.end_time else 'running',
                'total_runtime': self.total_runtime,
                'generations_completed': self.population.generation,
                'convergence_generation': self.convergence_generation,
                'total_evaluations': self.population.total_evaluations
            },
            'performance': {
                'best_fitness': final_stats.get('best_fitness', 0.0),
                'average_fitness': final_stats.get('average_fitness', 0.0),
                'fitness_improvement': self._calculate_fitness_improvement(),
                'diversity_maintained': final_stats.get('diversity', 0.0),
                'convergence_rate': self._calculate_convergence_rate()
            },
            'population_analysis': final_stats,
            'parameter_adaptation': {
                'final_crossover_rate': self.current_crossover_rate,
                'final_mutation_rate': self.current_mutation_rate,
                'adaptive_control_enabled': self.config.adaptive_parameters
            },
            'best_individual_analysis': self._analyze_best_individual()
        }
    
    def _calculate_fitness_improvement(self) -> float:
        """適応度改善率計算"""
        
        if len(self.population.fitness_history) < 2:
            return 0.0
        
        initial = self.population.fitness_history[0]
        final = self.population.fitness_history[-1]
        
        if initial == 0:
            return 0.0
        
        return (final - initial) / initial
    
    def _calculate_convergence_rate(self) -> float:
        """収束速度計算"""
        
        if self.convergence_generation == -1:
            return 0.0
        
        return 1.0 - (self.convergence_generation / self.config.max_generations)
    
    def _analyze_best_individual(self) -> Dict[str, Any]:
        """最良個体分析"""
        
        if not self.population.best_individual:
            return {'error': 'No best individual available'}
        
        best = self.population.best_individual
        
        return {
            'fitness_components': {
                'accuracy': best.fitness_components.accuracy,
                'complexity': best.fitness_components.complexity,
                'interpretability': best.fitness_components.interpretability,
                'generalization': best.fitness_components.generalization
            },
            'evolution_info': {
                'generation_born': best.generation,
                'evaluation_count': best.evaluation_count,
                'creation_method': best.creation_method,
                'mutation_count': best.mutation_count,
                'crossover_count': best.crossover_count
            },
            'genetic_diversity': len(set(best.parents)) if best.parents else 0
        }