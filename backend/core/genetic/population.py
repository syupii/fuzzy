from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np
import random
from dataclasses import dataclass
from enum import Enum

from .individual import GeneticIndividual, IndividualType


@dataclass
class PopulationConfig:
    """集団設定"""
    population_size: int = 30
    elite_size: int = 3
    tournament_size: int = 3
    max_generations: int = 20
    convergence_threshold: float = 0.001
    diversity_maintenance: bool = True
    aging_enabled: bool = False
    max_age: int = 10


class SelectionMethod(Enum):
    """選択方法"""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    ELITE = "elite"


class Population:
    """遺伝的アルゴリズムの集団"""
    
    def __init__(self, config: PopulationConfig = None):
        self.config = config or PopulationConfig()
        self.individuals: List[GeneticIndividual] = []
        self.generation = 0
        self.best_individual: Optional[GeneticIndividual] = None
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        
        # 統計情報
        self.total_evaluations = 0
        self.stagnation_count = 0
        self.last_improvement_generation = 0
    
    def initialize_random(self, genome_length: int, feature_names: List[str],
                         individual_type: IndividualType = IndividualType.HYBRID):
        """ランダム初期化"""
        
        print(f"集団ランダム初期化: サイズ={self.config.population_size}")
        
        self.individuals = []
        
        for i in range(self.config.population_size):
            individual = GeneticIndividual(individual_type)
            individual.initialize_random(genome_length, feature_names)
            individual.generation = 0
            self.individuals.append(individual)
        
        print(f"初期集団作成完了: {len(self.individuals)}個体")
    
    def evaluate_fitness(self, training_data: np.ndarray, 
                        feature_names: List[str], target_name: str):
        """集団の適応度評価"""
        
        print(f"世代{self.generation}の適応度評価中...")
        
        for individual in self.individuals:
            individual.evaluate_fitness(training_data, feature_names, target_name)
            self.total_evaluations += 1
        
        # 適応度でソート
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        
        # 最良個体更新
        current_best = self.individuals[0]
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best.clone()
            self.last_improvement_generation = self.generation
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
        
        # 統計更新
        fitness_values = [ind.fitness for ind in self.individuals]
        avg_fitness = np.mean(fitness_values)
        self.fitness_history.append(avg_fitness)
        
        # 多様性計算
        diversity = self._calculate_diversity()
        self.diversity_history.append(diversity)
        
        print(f"  最良適応度: {current_best.fitness:.4f}")
        print(f"  平均適応度: {avg_fitness:.4f}")
        print(f"  多様性: {diversity:.4f}")
    
    def select_parents(self, selection_method: SelectionMethod = SelectionMethod.TOURNAMENT,
                      num_parents: int = None) -> List[GeneticIndividual]:
        """親個体選択"""
        
        if num_parents is None:
            num_parents = self.config.population_size - self.config.elite_size
        
        parents = []
        
        if selection_method == SelectionMethod.TOURNAMENT:
            parents = self._tournament_selection(num_parents)
        elif selection_method == SelectionMethod.ROULETTE:
            parents = self._roulette_selection(num_parents)
        elif selection_method == SelectionMethod.RANK:
            parents = self._rank_selection(num_parents)
        else:
            # デフォルトはトーナメント選択
            parents = self._tournament_selection(num_parents)
        
        return parents
    
    def _tournament_selection(self, num_parents: int) -> List[GeneticIndividual]:
        """トーナメント選択"""
        
        parents = []
        
        for _ in range(num_parents):
            tournament = random.sample(self.individuals, self.config.tournament_size)
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        
        return parents
    
    def _roulette_selection(self, num_parents: int) -> List[GeneticIndividual]:
        """ルーレット選択"""
        
        # 適応度を正の値に調整
        min_fitness = min(ind.fitness for ind in self.individuals)
        if min_fitness < 0:
            adjusted_fitness = [ind.fitness - min_fitness + 0.1 for ind in self.individuals]
        else:
            adjusted_fitness = [ind.fitness for ind in self.individuals]
        
        total_fitness = sum(adjusted_fitness)
        if total_fitness == 0:
            return random.sample(self.individuals, num_parents)
        
        parents = []
        
        for _ in range(num_parents):
            spin = random.uniform(0, total_fitness)
            cumulative = 0
            
            for i, fitness in enumerate(adjusted_fitness):
                cumulative += fitness
                if cumulative >= spin:
                    parents.append(self.individuals[i])
                    break
        
        return parents
    
    def _rank_selection(self, num_parents: int) -> List[GeneticIndividual]:
        """ランク選択"""
        
        # 既にソートされているのでランクを利用
        ranks = list(range(len(self.individuals), 0, -1))  # 降順ランク
        total_rank = sum(ranks)
        
        parents = []
        
        for _ in range(num_parents):
            spin = random.uniform(0, total_rank)
            cumulative = 0
            
            for i, rank in enumerate(ranks):
                cumulative += rank
                if cumulative >= spin:
                    parents.append(self.individuals[i])
                    break
        
        return parents
    
    def create_next_generation(self, parents: List[GeneticIndividual],
                             crossover_rate: float = 0.8,
                             mutation_rate: float = 0.1) -> List[GeneticIndividual]:
        """次世代作成"""
        
        next_generation = []
        
        # エリート保存
        elite_count = min(self.config.elite_size, len(self.individuals))
        for i in range(elite_count):
            elite = self.individuals[i].clone()
            elite.generation = self.generation + 1
            next_generation.append(elite)
        
        # 交叉と突然変異で残りを生成
        while len(next_generation) < self.config.population_size:
            # 親選択
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            # 交叉
            if random.random() < crossover_rate and parent1 != parent2:
                child1, child2 = parent1.crossover(parent2, crossover_rate)
            else:
                child1 = parent1.clone()
                child2 = parent2.clone()
            
            # 突然変異
            child1 = child1.mutate(mutation_rate)
            child2 = child2.mutate(mutation_rate)
            
            # 次世代に追加
            child1.generation = self.generation + 1
            child2.generation = self.generation + 1
            
            next_generation.append(child1)
            if len(next_generation) < self.config.population_size:
                next_generation.append(child2)
        
        # サイズ調整
        next_generation = next_generation[:self.config.population_size]
        
        return next_generation
    
    def advance_generation(self, new_individuals: List[GeneticIndividual]):
        """世代進行"""
        
        self.individuals = new_individuals
        self.generation += 1
        
        # 年齢更新（有効な場合）
        if self.config.aging_enabled:
            for individual in self.individuals:
                individual.age += 1
        
        # 多様性維持（有効な場合）
        if self.config.diversity_maintenance:
            self._maintain_diversity()
    
    def _maintain_diversity(self):
        """多様性維持"""
        
        # 類似個体の検出と置換
        similarity_threshold = 0.9
        replaced_count = 0
        
        for i in range(len(self.individuals)):
            for j in range(i + 1, len(self.individuals)):
                similarity = self._calculate_similarity(self.individuals[i], self.individuals[j])
                
                if similarity > similarity_threshold:
                    # より適応度の低い個体をランダム個体で置換
                    if self.individuals[i].fitness < self.individuals[j].fitness:
                        self.individuals[i] = self._create_random_individual()
                        replaced_count += 1
                    else:
                        self.individuals[j] = self._create_random_individual()
                        replaced_count += 1
                    
                    if replaced_count >= 3:  # 過度な置換を防ぐ
                        break
            
            if replaced_count >= 3:
                break
        
        if replaced_count > 0:
            print(f"  多様性維持: {replaced_count}個体を置換")
    
    def _create_random_individual(self) -> GeneticIndividual:
        """ランダム個体作成"""
        
        individual = GeneticIndividual(IndividualType.HYBRID)
        # 既存の個体から設定を推定
        if self.individuals:
            example = self.individuals[0]
            if example.genome is not None:
                genome_length = len(example.genome)
                feature_names = list(example.tree_genes.get('membership_params', {}).keys())
                individual.initialize_random(genome_length, feature_names)
        
        individual.generation = self.generation
        return individual
    
    def _calculate_similarity(self, ind1: GeneticIndividual, ind2: GeneticIndividual) -> float:
        """個体間類似度計算"""
        
        if ind1.genome is None or ind2.genome is None:
            return 0.0
        
        # ゲノム類似度（コサイン類似度）
        dot_product = np.dot(ind1.genome, ind2.genome)
        norm1 = np.linalg.norm(ind1.genome)
        norm2 = np.linalg.norm(ind2.genome)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _calculate_diversity(self) -> float:
        """集団多様性計算"""
        
        if len(self.individuals) < 2:
            return 0.0
        
        total_similarity = 0.0
        count = 0
        
        for i in range(len(self.individuals)):
            for j in range(i + 1, len(self.individuals)):
                similarity = self._calculate_similarity(self.individuals[i], self.individuals[j])
                total_similarity += similarity
                count += 1
        
        if count == 0:
            return 1.0
        
        avg_similarity = total_similarity / count
        diversity = 1.0 - avg_similarity  # 類似度が低いほど多様性が高い
        
        return max(0.0, min(1.0, diversity))
    
    def is_converged(self) -> bool:
        """収束判定"""
        
        # 停滞回数による判定
        if self.stagnation_count >= 5:
            return True
        
        # 適応度履歴による判定
        if len(self.fitness_history) >= 5:
            recent_fitness = self.fitness_history[-5:]
            fitness_variance = np.var(recent_fitness)
            
            if fitness_variance < self.config.convergence_threshold:
                return True
        
        # 多様性による判定
        if len(self.diversity_history) >= 3:
            recent_diversity = self.diversity_history[-3:]
            if all(d < 0.1 for d in recent_diversity):
                return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """集団統計情報"""
        
        if not self.individuals:
            return {'error': 'No individuals in population'}
        
        fitness_values = [ind.fitness for ind in self.individuals]
        
        return {
            'generation': self.generation,
            'population_size': len(self.individuals),
            'best_fitness': max(fitness_values),
            'worst_fitness': min(fitness_values),
            'average_fitness': np.mean(fitness_values),
            'fitness_std': np.std(fitness_values),
            'diversity': self.diversity_history[-1] if self.diversity_history else 0.0,
            'total_evaluations': self.total_evaluations,
            'stagnation_count': self.stagnation_count,
            'last_improvement_generation': self.last_improvement_generation,
            'convergence_status': self.is_converged(),
            'elite_fitness': [ind.fitness for ind in self.individuals[:self.config.elite_size]],
            'fitness_components': {
                'accuracy': np.mean([ind.fitness_components.accuracy for ind in self.individuals]),
                'complexity': np.mean([ind.fitness_components.complexity for ind in self.individuals]),
                'interpretability': np.mean([ind.fitness_components.interpretability for ind in self.individuals]),
                'generalization': np.mean([ind.fitness_components.generalization for ind in self.individuals])
            }
        }
    
    def export_population(self) -> Dict[str, Any]:
        """集団のエクスポート"""
        
        return {
            'config': {
                'population_size': self.config.population_size,
                'elite_size': self.config.elite_size,
                'tournament_size': self.config.tournament_size,
                'max_generations': self.config.max_generations
            },
            'generation': self.generation,
            'individuals': [ind.to_dict() for ind in self.individuals],
            'best_individual': self.best_individual.to_dict() if self.best_individual else None,
            'fitness_history': self.fitness_history,
            'diversity_history': self.diversity_history,
            'statistics': self.get_statistics()
        }
    
    @classmethod
    def import_population(cls, data: Dict[str, Any]) -> 'Population':
        """集団のインポート"""
        
        config_data = data.get('config', {})
        config = PopulationConfig(
            population_size=config_data.get('population_size', 30),
            elite_size=config_data.get('elite_size', 3),
            tournament_size=config_data.get('tournament_size', 3),
            max_generations=config_data.get('max_generations', 20)
        )
        
        population = cls(config)
        population.generation = data.get('generation', 0)
        
        # 個体復元
        individuals_data = data.get('individuals', [])
        population.individuals = [
            GeneticIndividual.from_dict(ind_data) 
            for ind_data in individuals_data
        ]
        
        # 最良個体復元
        best_data = data.get('best_individual')
        if best_data:
            population.best_individual = GeneticIndividual.from_dict(best_data)
        
        # 履歴復元
        population.fitness_history = data.get('fitness_history', [])
        population.diversity_history = data.get('diversity_history', [])
        
        return population

