"""
遺伝的操作 - core/genetic/operators.py
遺伝的アルゴリズムの選択、交叉、突然変異操作
"""

from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
import random
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass

from .individual import GeneticIndividual


class CrossoverType(Enum):
    """交叉の種類"""
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    UNIFORM = "uniform"
    ARITHMETIC = "arithmetic"
    SIMULATED_BINARY = "simulated_binary"


class MutationType(Enum):
    """突然変異の種類"""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    POLYNOMIAL = "polynomial"
    ADAPTIVE = "adaptive"


@dataclass
class OperatorConfig:
    """遺伝的操作の設定"""
    crossover_type: CrossoverType = CrossoverType.UNIFORM
    mutation_type: MutationType = MutationType.GAUSSIAN
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    tournament_size: int = 3
    elite_size: int = 2


class GeneticOperator(ABC):
    """遺伝的操作の抽象基底クラス"""
    
    def __init__(self, config: OperatorConfig):
        self.config = config
        self.operation_count = 0
    
    @abstractmethod
    def apply(self, *args, **kwargs):
        """操作を適用"""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """操作統計を取得"""
        return {
            'operation_count': self.operation_count,
            'config': self.config.__dict__
        }


class SelectionOperator(GeneticOperator):
    """選択操作"""
    
    def apply(self, population: List[GeneticIndividual], num_parents: int) -> List[GeneticIndividual]:
        """親選択を実行"""
        self.operation_count += 1
        
        parents = []
        for _ in range(num_parents):
            parent = self._tournament_selection(population)
            parents.append(parent)
        
        return parents
    
    def _tournament_selection(self, population: List[GeneticIndividual]) -> GeneticIndividual:
        """トーナメント選択"""
        tournament = random.sample(population, min(self.config.tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)
    
    def _roulette_selection(self, population: List[GeneticIndividual]) -> GeneticIndividual:
        """ルーレット選択"""
        fitness_values = [max(0.001, ind.fitness) for ind in population]
        total_fitness = sum(fitness_values)
        
        if total_fitness == 0:
            return random.choice(population)
        
        selection_prob = random.uniform(0, total_fitness)
        cumulative_fitness = 0
        
        for i, individual in enumerate(population):
            cumulative_fitness += fitness_values[i]
            if cumulative_fitness >= selection_prob:
                return individual
        
        return population[-1]


class CrossoverOperator(GeneticOperator):
    """交叉操作"""
    
    def apply(self, parent1: GeneticIndividual, parent2: GeneticIndividual) -> Tuple[GeneticIndividual, GeneticIndividual]:
        """交叉を実行"""
        self.operation_count += 1
        
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        if self.config.crossover_type == CrossoverType.SINGLE_POINT:
            return self._single_point_crossover(parent1, parent2)
        elif self.config.crossover_type == CrossoverType.TWO_POINT:
            return self._two_point_crossover(parent1, parent2)
        elif self.config.crossover_type == CrossoverType.UNIFORM:
            return self._uniform_crossover(parent1, parent2)
        elif self.config.crossover_type == CrossoverType.ARITHMETIC:
            return self._arithmetic_crossover(parent1, parent2)
        elif self.config.crossover_type == CrossoverType.SIMULATED_BINARY:
            return self._simulated_binary_crossover(parent1, parent2)
        else:
            return self._uniform_crossover(parent1, parent2)
    
    def _single_point_crossover(self, parent1: GeneticIndividual, parent2: GeneticIndividual) -> Tuple[GeneticIndividual, GeneticIndividual]:
        """一点交叉"""
        child1, child2 = parent1.copy(), parent2.copy()
        
        if parent1.genome is not None and parent2.genome is not None:
            crossover_point = random.randint(1, len(parent1.genome) - 1)
            
            child1.genome = np.concatenate([
                parent1.genome[:crossover_point],
                parent2.genome[crossover_point:]
            ])
            
            child2.genome = np.concatenate([
                parent2.genome[:crossover_point],
                parent1.genome[crossover_point:]
            ])
        
        self._crossover_tree_genes(child1, child2, parent1, parent2)
        self._update_child_metadata(child1, child2, parent1, parent2)
        
        return child1, child2
    
    def _two_point_crossover(self, parent1: GeneticIndividual, parent2: GeneticIndividual) -> Tuple[GeneticIndividual, GeneticIndividual]:
        """二点交叉"""
        child1, child2 = parent1.copy(), parent2.copy()
        
        if parent1.genome is not None and parent2.genome is not None:
            genome_length = len(parent1.genome)
            point1 = random.randint(1, genome_length - 2)
            point2 = random.randint(point1 + 1, genome_length - 1)
            
            child1.genome = np.concatenate([
                parent1.genome[:point1],
                parent2.genome[point1:point2],
                parent1.genome[point2:]
            ])
            
            child2.genome = np.concatenate([
                parent2.genome[:point1],
                parent1.genome[point1:point2],
                parent2.genome[point2:]
            ])
        
        self._crossover_tree_genes(child1, child2, parent1, parent2)
        self._update_child_metadata(child1, child2, parent1, parent2)
        
        return child1, child2
    
    def _uniform_crossover(self, parent1: GeneticIndividual, parent2: GeneticIndividual) -> Tuple[GeneticIndividual, GeneticIndividual]:
        """一様交叉"""
        child1, child2 = parent1.copy(), parent2.copy()
        
        if parent1.genome is not None and parent2.genome is not None:
            mask = np.random.random(len(parent1.genome)) < 0.5
            
            child1.genome = parent1.genome.copy()
            child2.genome = parent2.genome.copy()
            
            child1.genome[mask] = parent2.genome[mask]
            child2.genome[mask] = parent1.genome[mask]
        
        self._crossover_tree_genes(child1, child2, parent1, parent2)
        self._update_child_metadata(child1, child2, parent1, parent2)
        
        return child1, child2
    
    def _arithmetic_crossover(self, parent1: GeneticIndividual, parent2: GeneticIndividual) -> Tuple[GeneticIndividual, GeneticIndividual]:
        """算術交叉"""
        child1, child2 = parent1.copy(), parent2.copy()
        
        if parent1.genome is not None and parent2.genome is not None:
            alpha = random.uniform(0.3, 0.7)
            
            child1.genome = alpha * parent1.genome + (1 - alpha) * parent2.genome
            child2.genome = (1 - alpha) * parent1.genome + alpha * parent2.genome
            
            # [0, 1]の範囲にクリッピング
            child1.genome = np.clip(child1.genome, 0, 1)
            child2.genome = np.clip(child2.genome, 0, 1)
        
        self._crossover_tree_genes(child1, child2, parent1, parent2)
        self._update_child_metadata(child1, child2, parent1, parent2)
        
        return child1, child2
    
    def _simulated_binary_crossover(self, parent1: GeneticIndividual, parent2: GeneticIndividual) -> Tuple[GeneticIndividual, GeneticIndividual]:
        """模擬二進交叉（SBX）"""
        child1, child2 = parent1.copy(), parent2.copy()
        
        if parent1.genome is not None and parent2.genome is not None:
            eta_c = 20.0  # 分布指数
            
            child1.genome = parent1.genome.copy()
            child2.genome = parent2.genome.copy()
            
            for i in range(len(parent1.genome)):
                if random.random() <= 0.5:
                    y1, y2 = parent1.genome[i], parent2.genome[i]
                    
                    if abs(y1 - y2) > 1e-14:
                        if y1 > y2:
                            y1, y2 = y2, y1
                        
                        # ベータ値の計算
                        rand = random.random()
                        if rand <= 0.5:
                            beta = (2.0 * rand) ** (1.0 / (eta_c + 1.0))
                        else:
                            beta = (1.0 / (2.0 * (1.0 - rand))) ** (1.0 / (eta_c + 1.0))
                        
                        # 子個体の遺伝子値計算
                        c1 = 0.5 * ((1.0 + beta) * y1 + (1.0 - beta) * y2)
                        c2 = 0.5 * ((1.0 - beta) * y1 + (1.0 + beta) * y2)
                        
                        child1.genome[i] = np.clip(c1, 0, 1)
                        child2.genome[i] = np.clip(c2, 0, 1)
        
        self._crossover_tree_genes(child1, child2, parent1, parent2)
        self._update_child_metadata(child1, child2, parent1, parent2)
        
        return child1, child2
    
    def _crossover_tree_genes(self, child1: GeneticIndividual, child2: GeneticIndividual,
                            parent1: GeneticIndividual, parent2: GeneticIndividual):
        """木構造遺伝子の交叉"""
        # 確率的に親の木構造遺伝子を選択
        if random.random() < 0.5:
            child1.tree_genes = self._deep_copy_dict(parent1.tree_genes)
            child2.tree_genes = self._deep_copy_dict(parent2.tree_genes)
        else:
            child1.tree_genes = self._deep_copy_dict(parent2.tree_genes)
            child2.tree_genes = self._deep_copy_dict(parent1.tree_genes)
        
        # 一部のパラメータをブレンド
        self._blend_tree_parameters(child1, child2, parent1, parent2)
    
    def _blend_tree_parameters(self, child1: GeneticIndividual, child2: GeneticIndividual,
                             parent1: GeneticIndividual, parent2: GeneticIndividual):
        """木パラメータのブレンド"""
        # 特徴量選択確率のブレンド
        if ('feature_selection_probs' in parent1.tree_genes and 
            'feature_selection_probs' in parent2.tree_genes):
            
            alpha = random.uniform(0.3, 0.7)
            child1.tree_genes['feature_selection_probs'] = (
                alpha * parent1.tree_genes['feature_selection_probs'] +
                (1 - alpha) * parent2.tree_genes['feature_selection_probs']
            )
            child2.tree_genes['feature_selection_probs'] = (
                (1 - alpha) * parent1.tree_genes['feature_selection_probs'] +
                alpha * parent2.tree_genes['feature_selection_probs']
            )
    
    def _update_child_metadata(self, child1: GeneticIndividual, child2: GeneticIndividual,
                             parent1: GeneticIndividual, parent2: GeneticIndividual):
        """子個体のメタデータ更新"""
        child1.parents = [parent1.id, parent2.id]
        child2.parents = [parent2.id, parent1.id]
        child1.creation_method = "crossover"
        child2.creation_method = "crossover"
        child1.crossover_count = max(parent1.crossover_count, parent2.crossover_count) + 1
        child2.crossover_count = max(parent1.crossover_count, parent2.crossover_count) + 1
        child1.tree = None  # 再構築が必要
        child2.tree = None
        child1.fitness = 0.0
        child2.fitness = 0.0
    
    def _deep_copy_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """辞書の深いコピー"""
        if isinstance(d, dict):
            return {k: self._deep_copy_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [self._deep_copy_dict(item) for item in d]
        elif isinstance(d, np.ndarray):
            return d.copy()
        else:
            return d


class MutationOperator(GeneticOperator):
    """突然変異操作"""
    
    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        self.adaptive_strength = config.mutation_strength
        self.success_count = 0
        self.total_count = 0
    
    def apply(self, individual: GeneticIndividual) -> GeneticIndividual:
        """突然変異を実行"""
        self.operation_count += 1
        self.total_count += 1
        
        if random.random() > self.config.mutation_rate:
            return individual
        
        mutant = individual.copy()
        
        if self.config.mutation_type == MutationType.GAUSSIAN:
            self._gaussian_mutation(mutant)
        elif self.config.mutation_type == MutationType.UNIFORM:
            self._uniform_mutation(mutant)
        elif self.config.mutation_type == MutationType.POLYNOMIAL:
            self._polynomial_mutation(mutant)
        elif self.config.mutation_type == MutationType.ADAPTIVE:
            self._adaptive_mutation(mutant)
        else:
            self._gaussian_mutation(mutant)
        
        # 木構造遺伝子の突然変異
        self._mutate_tree_genes(mutant)
        
        # メタデータ更新
        mutant.mutation_count += 1
        mutant.tree = None
        mutant.fitness = 0.0
        mutant.creation_method = "mutation"
        
        return mutant
    
    def _gaussian_mutation(self, individual: GeneticIndividual):
        """ガウシアン突然変異"""
        if individual.genome is None:
            return
        
        mutation_mask = np.random.random(len(individual.genome)) < self.config.mutation_rate
        mutation_values = np.random.normal(0, self.config.mutation_strength, len(individual.genome))
        
        individual.genome[mutation_mask] += mutation_values[mutation_mask]
        individual.genome = np.clip(individual.genome, 0, 1)
    
    def _uniform_mutation(self, individual: GeneticIndividual):
        """一様突然変異"""
        if individual.genome is None:
            return
        
        mutation_mask = np.random.random(len(individual.genome)) < self.config.mutation_rate
        
        for i in range(len(individual.genome)):
            if mutation_mask[i]:
                individual.genome[i] = random.random()
    
    def _polynomial_mutation(self, individual: GeneticIndividual):
        """多項式突然変異"""
        if individual.genome is None:
            return
        
        eta_m = 20.0  # 分布指数
        
        for i in range(len(individual.genome)):
            if random.random() < self.config.mutation_rate:
                y = individual.genome[i]
                rand = random.random()
                
                if rand < 0.5:
                    delta = (2.0 * rand) ** (1.0 / (eta_m + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - rand)) ** (1.0 / (eta_m + 1.0))
                
                individual.genome[i] = np.clip(y + delta * self.config.mutation_strength, 0, 1)
    
    def _adaptive_mutation(self, individual: GeneticIndividual):
        """適応的突然変異"""
        # 成功率に基づいて変異強度を調整
        if self.total_count > 10:
            success_rate = self.success_count / self.total_count
            if success_rate > 0.2:  # 成功率が高い場合
                self.adaptive_strength = min(0.5, self.adaptive_strength * 1.1)
            else:  # 成功率が低い場合
                self.adaptive_strength = max(0.01, self.adaptive_strength * 0.9)
        
        # ガウシアン突然変異を適応強度で実行
        if individual.genome is None:
            return
        
        mutation_mask = np.random.random(len(individual.genome)) < self.config.mutation_rate
        mutation_values = np.random.normal(0, self.adaptive_strength, len(individual.genome))
        
        individual.genome[mutation_mask] += mutation_values[mutation_mask]
        individual.genome = np.clip(individual.genome, 0, 1)
    
    def _mutate_tree_genes(self, individual: GeneticIndividual):
        """木構造遺伝子の突然変異"""
        # 最大深度の変更
        if random.random() < self.config.mutation_rate * 0.3:
            current_depth = individual.tree_genes.get('max_depth', 5)
            individual.tree_genes['max_depth'] = max(2, min(8, current_depth + random.choice([-1, 0, 1])))
        
        # メンバーシップ関数パラメータの変更
        if 'membership_params' in individual.tree_genes:
            self._mutate_membership_params(individual)
        
        # 特徴量選択確率の変更
        if 'feature_selection_probs' in individual.tree_genes:
            self._mutate_feature_selection_probs(individual)
    
    def _mutate_membership_params(self, individual: GeneticIndividual):
        """メンバーシップ関数パラメータの突然変異"""
        for feature, fuzzy_sets in individual.tree_genes['membership_params'].items():
            for fuzzy_set, params in fuzzy_sets.items():
                if params['type'] == 'triangular':
                    for param in ['a', 'b', 'c']:
                        if random.random() < self.config.mutation_rate * 0.5:
                            params[param] += random.gauss(0, 0.5)
                            params[param] = max(0, min(10, params[param]))
                    
                    # 順序制約の維持
                    params_list = [params['a'], params['b'], params['c']]
                    params_list.sort()
                    params['a'], params['b'], params['c'] = params_list
    
    def _mutate_feature_selection_probs(self, individual: GeneticIndividual):
        """特徴量選択確率の突然変異"""
        probs = individual.tree_genes['feature_selection_probs']
        noise = np.random.normal(0, 0.1, len(probs))
        
        mask = np.random.random(len(probs)) < self.config.mutation_rate * 0.5
        probs[mask] += noise[mask]
        probs = np.abs(probs)
        probs /= np.sum(probs)  # 正規化
        
        individual.tree_genes['feature_selection_probs'] = probs
    
    def update_success_rate(self, improved: bool):
        """成功率の更新（適応的突然変異用）"""
        if improved:
            self.success_count += 1


class GeneticOperators:
    """遺伝的操作の統合クラス"""
    
    def __init__(self, config: OperatorConfig):
        self.config = config
        self.selection_op = SelectionOperator(config)
        self.crossover_op = CrossoverOperator(config)
        self.mutation_op = MutationOperator(config)
    
    def evolve_generation(self, population: List[GeneticIndividual]) -> List[GeneticIndividual]:
        """1世代の進化"""
        
        # エリート保存
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        elites = sorted_pop[:self.config.elite_size]
        
        # 親選択
        num_offspring_needed = len(population) - self.config.elite_size
        parents = self.selection_op.apply(population, num_offspring_needed)
        
        # 子個体生成
        offspring = []
        
        for i in range(0, len(parents) - 1, 2):
            parent1 = parents[i]
            parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]
            
            # 交叉
            child1, child2 = self.crossover_op.apply(parent1, parent2)
            
            # 突然変異
            child1 = self.mutation_op.apply(child1)
            child2 = self.mutation_op.apply(child2)
            
            offspring.extend([child1, child2])
        
        # 新世代作成
        new_generation = list(elites) + offspring[:num_offspring_needed]
        
        return new_generation[:len(population)]
    
    def get_operator_statistics(self) -> Dict[str, Any]:
        """操作統計の取得"""
        return {
            'selection': self.selection_op.get_statistics(),
            'crossover': self.crossover_op.get_statistics(),
            'mutation': self.mutation_op.get_statistics(),
            'config': self.config.__dict__
        }