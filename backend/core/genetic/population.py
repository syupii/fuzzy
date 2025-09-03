"""
集団管理 - core/genetic/population.py
遺伝的アルゴリズムの集団管理クラス
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np
import random
from dataclasses import dataclass
from enum import Enum

from .individual import GeneticIndividual, IndividualType, FitnessComponents


class SelectionMethod(Enum):
    """選択手法"""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    ELITE = "elite"


class ReplacementStrategy(Enum):
    """世代交代戦略"""
    GENERATIONAL = "generational"
    STEADY_STATE = "steady_state"
    ELITE_REPLACEMENT = "elite_replacement"


@dataclass
class PopulationStatistics:
    """集団統計"""
    generation: int
    population_size: int
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    fitness_std: float
    diversity_score: float
    convergence_rate: float


class Population:
    """遺伝的アルゴリズムの集団"""
    
    def __init__(self, population_size: int, individual_type: IndividualType = IndividualType.HYBRID):
        self.population_size = population_size
        self.individual_type = individual_type
        self.individuals: List[GeneticIndividual] = []
        self.generation = 0
        
        # 統計情報
        self.statistics_history: List[PopulationStatistics] = []
        self.best_individual: Optional[GeneticIndividual] = None
        self.best_fitness_history: List[float] = []
        
        # 多様性管理
        self.diversity_threshold = 0.1
        self.stagnation_counter = 0
        self.max_stagnation = 10
    
    def initialize_random(self, genome_length: int, feature_names: List[str], 
                         max_depth: int = 5, min_samples_leaf: int = 5):
        """集団をランダム初期化"""
        
        self.individuals = []
        
        for i in range(self.population_size):
            individual = GeneticIndividual(self.individual_type)
            individual.initialize_random(genome_length, feature_names, max_depth, min_samples_leaf)
            individual.generation = 0
            self.individuals.append(individual)
        
        print(f"Initialized population with {len(self.individuals)} individuals")
    
    def evaluate_population(self, training_data: np.ndarray, test_data: np.ndarray,
                          feature_names: List[str], target_name: str,
                          fitness_weights: Dict[str, float] = None) -> Dict[str, Any]:
        """集団全体の適応度評価"""
        
        evaluation_results = {
            'evaluated_count': 0,
            'build_failures': 0,
            'evaluation_failures': 0,
            'fitness_scores': []
        }
        
        for individual in self.individuals:
            try:
                # 決定木構築
                if individual.tree is None:
                    success = individual.build_tree(training_data, feature_names, target_name)
                    if not success:
                        evaluation_results['build_failures'] += 1
                        individual.fitness = 0.0
                        continue
                
                # 適応度評価
                fitness = individual.evaluate_fitness(test_data, feature_names, target_name, fitness_weights)
                evaluation_results['fitness_scores'].append(fitness)
                evaluation_results['evaluated_count'] += 1
                
                # 世代更新
                individual.generation = self.generation
                
            except Exception as e:
                print(f"Evaluation error for individual {individual.id}: {e}")
                evaluation_results['evaluation_failures'] += 1
                individual.fitness = 0.0
        
        # 最良個体更新
        self._update_best_individual()
        
        # 統計計算
        self._calculate_statistics()
        
        return evaluation_results
    
    def _update_best_individual(self):
        """最良個体の更新"""
        if not self.individuals:
            return
        
        current_best = max(self.individuals, key=lambda x: x.fitness)
        
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best.copy()
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        
        self.best_fitness_history.append(current_best.fitness)
    
    def _calculate_statistics(self):
        """集団統計の計算"""
        if not self.individuals:
            return
        
        fitness_values = [ind.fitness for ind in self.individuals]
        
        stats = PopulationStatistics(
            generation=self.generation,
            population_size=len(self.individuals),
            best_fitness=max(fitness_values),
            average_fitness=np.mean(fitness_values),
            worst_fitness=min(fitness_values),
            fitness_std=np.std(fitness_values),
            diversity_score=self._calculate_diversity(),
            convergence_rate=self._calculate_convergence_rate()
        )
        
        self.statistics_history.append(stats)
    
    def _calculate_diversity(self) -> float:
        """集団の多様性計算"""
        if len(self.individuals) < 2:
            return 0.0
        
        # 適応度の分散を基準とした多様性
        fitness_values = [ind.fitness for ind in self.individuals]
        fitness_std = np.std(fitness_values)
        fitness_range = max(fitness_values) - min(fitness_values)
        
        # 正規化された多様性スコア
        diversity = fitness_std / max(0.001, fitness_range)
        
        return min(1.0, diversity)
    
    def _calculate_convergence_rate(self) -> float:
        """収束率の計算"""
        if len(self.best_fitness_history) < 5:
            return 0.0
        
        recent_improvements = 0
        for i in range(-4, 0):
            if self.best_fitness_history[i] > self.best_fitness_history[i-1]:
                recent_improvements += 1
        
        return recent_improvements / 4.0
    
    def select_parents(self, selection_method: SelectionMethod = SelectionMethod.TOURNAMENT,
                      tournament_size: int = 3, num_parents: int = None) -> List[GeneticIndividual]:
        """親選択"""
        
        if num_parents is None:
            num_parents = self.population_size
        
        parents = []
        
        for _ in range(num_parents):
            if selection_method == SelectionMethod.TOURNAMENT:
                parent = self._tournament_selection(tournament_size)
            elif selection_method == SelectionMethod.ROULETTE:
                parent = self._roulette_selection()
            elif selection_method == SelectionMethod.RANK:
                parent = self._rank_selection()
            elif selection_method == SelectionMethod.ELITE:
                parent = self._elite_selection()
            else:
                parent = self._tournament_selection(tournament_size)
            
            parents.append(parent)
        
        return parents
    
    def _tournament_selection(self, tournament_size: int) -> GeneticIndividual:
        """トーナメント選択"""
        tournament = random.sample(self.individuals, min(tournament_size, len(self.individuals)))
        return max(tournament, key=lambda x: x.fitness)
    
    def _roulette_selection(self) -> GeneticIndividual:
        """ルーレット選択"""
        fitness_values = [max(0.001, ind.fitness) for ind in self.individuals]  # 負の適応度を回避
        total_fitness = sum(fitness_values)
        
        if total_fitness == 0:
            return random.choice(self.individuals)
        
        selection_prob = random.uniform(0, total_fitness)
        cumulative_fitness = 0
        
        for i, individual in enumerate(self.individuals):
            cumulative_fitness += fitness_values[i]
            if cumulative_fitness >= selection_prob:
                return individual
        
        return self.individuals[-1]  # フォールバック
    
    def _rank_selection(self) -> GeneticIndividual:
        """ランク選択"""
        sorted_individuals = sorted(self.individuals, key=lambda x: x.fitness)
        ranks = list(range(1, len(sorted_individuals) + 1))
        total_rank = sum(ranks)
        
        selection_prob = random.uniform(0, total_rank)
        cumulative_rank = 0
        
        for i, individual in enumerate(sorted_individuals):
            cumulative_rank += ranks[i]
            if cumulative_rank >= selection_prob:
                return individual
        
        return sorted_individuals[-1]
    
    def _elite_selection(self) -> GeneticIndividual:
        """エリート選択"""
        sorted_individuals = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)
        elite_size = max(1, len(self.individuals) // 10)  # 上位10%
        return random.choice(sorted_individuals[:elite_size])
    
    def create_offspring(self, parents: List[GeneticIndividual], crossover_rate: float = 0.8,
                        mutation_rate: float = 0.1) -> List[GeneticIndividual]:
        """子個体の生成"""
        
        offspring = []
        
        # ペアを作って交叉
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
            
            if random.random() < crossover_rate:
                child1, child2 = parent1.crossover(parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # 突然変異
            if random.random() < mutation_rate:
                child1.mutate(mutation_rate)
            if random.random() < mutation_rate:
                child2.mutate(mutation_rate)
            
            offspring.extend([child1, child2])
        
        # 集団サイズに調整
        return offspring[:self.population_size]
    
    def replace_population(self, offspring: List[GeneticIndividual], 
                          strategy: ReplacementStrategy = ReplacementStrategy.ELITE_REPLACEMENT,
                          elite_size: int = None) -> None:
        """世代交代"""
        
        if elite_size is None:
            elite_size = max(1, self.population_size // 10)
        
        if strategy == ReplacementStrategy.GENERATIONAL:
            # 完全世代交代
            self.individuals = offspring[:self.population_size]
            
        elif strategy == ReplacementStrategy.ELITE_REPLACEMENT:
            # エリート保存戦略
            
            # 現世代のエリート選択
            sorted_current = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)
            elites = sorted_current[:elite_size]
            
            # 子個体と統合
            combined = elites + offspring
            
            # 上位個体を選択
            sorted_combined = sorted(combined, key=lambda x: x.fitness, reverse=True)
            self.individuals = sorted_combined[:self.population_size]
            
        elif strategy == ReplacementStrategy.STEADY_STATE:
            # 定常状態モデル
            
            # 最悪個体を子個体と置換
            combined = self.individuals + offspring
            sorted_combined = sorted(combined, key=lambda x: x.fitness, reverse=True)
            self.individuals = sorted_combined[:self.population_size]
        
        self.generation += 1
    
    def is_converged(self, convergence_threshold: float = 0.001) -> bool:
        """収束判定"""
        
        # 停滞による収束
        if self.stagnation_counter >= self.max_stagnation:
            return True
        
        # 適応度の収束
        if len(self.statistics_history) >= 5:
            recent_best = [stats.best_fitness for stats in self.statistics_history[-5:]]
            fitness_improvement = max(recent_best) - min(recent_best)
            
            if fitness_improvement < convergence_threshold:
                return True
        
        # 多様性の低下
        current_stats = self.statistics_history[-1] if self.statistics_history else None
        if current_stats and current_stats.diversity_score < self.diversity_threshold:
            return True
        
        return False
    
    def maintain_diversity(self, diversity_injection_rate: float = 0.1) -> int:
        """多様性維持"""
        
        if not self.is_low_diversity():
            return 0
        
        # 最悪個体の一部をランダム個体で置換
        num_to_replace = max(1, int(self.population_size * diversity_injection_rate))
        
        sorted_individuals = sorted(self.individuals, key=lambda x: x.fitness)
        worst_individuals = sorted_individuals[:num_to_replace]
        
        # 新しいランダム個体を生成
        feature_names = []
        if self.individuals and hasattr(self.individuals[0], 'tree_genes'):
            feature_selection_probs = self.individuals[0].tree_genes.get('feature_selection_probs', [])
            feature_names = [f'feature_{i}' for i in range(len(feature_selection_probs))]
        
        new_individuals = []
        for individual in worst_individuals:
            new_individual = GeneticIndividual(self.individual_type)
            new_individual.initialize_random(
                len(individual.genome) if individual.genome is not None else 20,
                feature_names
            )
            new_individual.generation = self.generation
            new_individuals.append(new_individual)
        
        # 置換
        for i, old_individual in enumerate(worst_individuals):
            idx = self.individuals.index(old_individual)
            self.individuals[idx] = new_individuals[i]
        
        print(f"Diversity injection: replaced {num_to_replace} individuals")
        return num_to_replace
    
    def is_low_diversity(self) -> bool:
        """多様性の低下判定"""
        if not self.statistics_history:
            return False
        
        current_stats = self.statistics_history[-1]
        return current_stats.diversity_score < self.diversity_threshold
    
    def get_population_summary(self) -> Dict[str, Any]:
        """集団概要の取得"""
        
        if not self.individuals:
            return {'status': 'empty', 'population_size': 0}
        
        current_stats = self.statistics_history[-1] if self.statistics_history else None
        
        summary = {
            'generation': self.generation,
            'population_size': len(self.individuals),
            'best_individual_info': self.best_individual.get_info() if self.best_individual else None,
            'current_statistics': {
                'best_fitness': current_stats.best_fitness if current_stats else 0,
                'average_fitness': current_stats.average_fitness if current_stats else 0,
                'diversity_score': current_stats.diversity_score if current_stats else 0
            },
            'convergence_info': {
                'stagnation_counter': self.stagnation_counter,
                'max_stagnation': self.max_stagnation,
                'is_converged': self.is_converged(),
                'is_low_diversity': self.is_low_diversity()
            },
            'fitness_history': self.best_fitness_history[-10:],  # 最新10世代
            'individual_summary': [
                {
                    'id': ind.id[:8],
                    'fitness': ind.fitness,
                    'generation': ind.generation,
                    'evaluation_count': ind.evaluation_count
                }
                for ind in sorted(self.individuals, key=lambda x: x.fitness, reverse=True)[:5]
            ]
        }
        
        return summary
    
    def export_population(self) -> Dict[str, Any]:
        """集団のエクスポート"""
        
        return {
            'generation': self.generation,
            'population_size': self.population_size,
            'individual_type': self.individual_type.value,
            'individuals': [ind.get_info() for ind in self.individuals],
            'best_individual': self.best_individual.get_info() if self.best_individual else None,
            'statistics_history': [
                {
                    'generation': stats.generation,
                    'best_fitness': stats.best_fitness,
                    'average_fitness': stats.average_fitness,
                    'diversity_score': stats.diversity_score,
                    'convergence_rate': stats.convergence_rate
                }
                for stats in self.statistics_history
            ],
            'fitness_history': self.best_fitness_history,
            'convergence_info': {
                'stagnation_counter': self.stagnation_counter,
                'is_converged': self.is_converged()
            }
        }