# core/genetic/operators.py - 遺伝的演算子

import random
import numpy as np
from typing import List, Tuple, Dict, Callable, Optional
from enum import Enum
from abc import ABC, abstractmethod

from core.genetic.individual import Individual

class SelectionMethod(Enum):
    """選択手法"""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    STOCHASTIC_UNIVERSAL = "stochastic_universal"
    TRUNCATION = "truncation"

class CrossoverMethod(Enum):
    """交叉手法"""
    UNIFORM = "uniform"
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    BLEND = "blend"
    SIMULATED_BINARY = "simulated_binary"

class MutationMethod(Enum):
    """変異手法"""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    POLYNOMIAL = "polynomial"
    ADAPTIVE = "adaptive"

class GeneticOperator(ABC):
    """遺伝的演算子の抽象基底クラス"""
    
    def __init__(self, **kwargs):
        self.parameters = kwargs
    
    @abstractmethod
    def apply(self, *args, **kwargs):
        """演算子を適用"""
        pass

class SelectionOperator(GeneticOperator):
    """選択演算子"""
    
    def __init__(self, method: SelectionMethod, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.tournament_size = kwargs.get('tournament_size', 3)
        self.truncation_ratio = kwargs.get('truncation_ratio', 0.5)
    
    def apply(self, population: List[Individual], selection_size: int) -> List[Individual]:
        """選択を実行"""
        
        if self.method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection(population, selection_size)
        elif self.method == SelectionMethod.ROULETTE:
            return self._roulette_selection(population, selection_size)
        elif self.method == SelectionMethod.RANK:
            return self._rank_selection(population, selection_size)
        elif self.method == SelectionMethod.STOCHASTIC_UNIVERSAL:
            return self._stochastic_universal_selection(population, selection_size)
        elif self.method == SelectionMethod.TRUNCATION:
            return self._truncation_selection(population, selection_size)
        else:
            raise ValueError(f"未知の選択手法: {self.method}")
    
    def _tournament_selection(self, population: List[Individual], 
                            selection_size: int) -> List[Individual]:
        """トーナメント選択"""
        selected = []
        
        for _ in range(selection_size):
            tournament = random.sample(population, min(self.tournament_size, len(population)))
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)
        
        return selected
    
    def _roulette_selection(self, population: List[Individual], 
                          selection_size: int) -> List[Individual]:
        """ルーレット選択"""
        fitness_values = [ind.fitness for ind in population]
        
        # 適応度を正の値に調整
        min_fitness = min(fitness_values)
        if min_fitness <= 0:
            adjusted_fitness = [f - min_fitness + 1e-6 for f in fitness_values]
        else:
            adjusted_fitness = fitness_values
        
        total_fitness = sum(adjusted_fitness)
        
        if total_fitness == 0:
            return random.choices(population, k=selection_size)
        
        # 確率計算
        probabilities = [f / total_fitness for f in adjusted_fitness]
        
        return random.choices(population, weights=probabilities, k=selection_size)
    
    def _rank_selection(self, population: List[Individual], 
                       selection_size: int) -> List[Individual]:
        """ランク選択"""
        # 適応度でソート
        sorted_population = sorted(population, key=lambda x: x.fitness)
        
        # ランクに基づく重み（線形ランキング）
        n = len(population)
        weights = [i + 1 for i in range(n)]  # 1, 2, 3, ..., n
        
        return random.choices(sorted_population, weights=weights, k=selection_size)
    
    def _stochastic_universal_selection(self, population: List[Individual], 
                                      selection_size: int) -> List[Individual]:
        """確率的ユニバーサル選択"""
        fitness_values = [ind.fitness for ind in population]
        
        # 適応度を正の値に調整
        min_fitness = min(fitness_values)
        if min_fitness <= 0:
            adjusted_fitness = [f - min_fitness + 1e-6 for f in fitness_values]
        else:
            adjusted_fitness = fitness_values
        
        total_fitness = sum(adjusted_fitness)
        
        if total_fitness == 0:
            return random.choices(population, k=selection_size)
        
        # 確率的ユニバーサル選択の実装
        step_size = total_fitness / selection_size
        start_point = random.uniform(0, step_size)
        
        selected = []
        cumulative_fitness = 0
        current_index = 0
        
        for i in range(selection_size):
            selection_point = start_point + i * step_size
            
            while cumulative_fitness < selection_point and current_index < len(population):
                cumulative_fitness += adjusted_fitness[current_index]
                current_index += 1
            
            selected.append(population[current_index - 1])
        
        return selected
    
    def _truncation_selection(self, population: List[Individual], 
                            selection_size: int) -> List[Individual]:
        """切り捨て選択"""
        # 上位一定割合を選択
        truncation_count = max(1, int(len(population) * self.truncation_ratio))
        sorted_population = sorted(population, key=lambda x: x.fitness, reverse=True)
        elite = sorted_population[:truncation_count]
        
        # 選択サイズまで復元抽出
        return random.choices(elite, k=selection_size)

class CrossoverOperator(GeneticOperator):
    """交叉演算子"""
    
    def __init__(self, method: CrossoverMethod, crossover_rate: float = 0.8, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.crossover_rate = crossover_rate
        self.blend_alpha = kwargs.get('blend_alpha', 0.5)
        self.sbx_eta = kwargs.get('sbx_eta', 20.0)  # SBX分布指数
    
    def apply(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """交叉を実行"""
        
        if random.random() > self.crossover_rate:
            return parent1.clone(), parent2.clone()
        
        if self.method == CrossoverMethod.UNIFORM:
            return self._uniform_crossover(parent1, parent2)
        elif self.method == CrossoverMethod.SINGLE_POINT:
            return self._single_point_crossover(parent1, parent2)
        elif self.method == CrossoverMethod.TWO_POINT:
            return self._two_point_crossover(parent1, parent2)
        elif self.method == CrossoverMethod.BLEND:
            return self._blend_crossover(parent1, parent2)
        elif self.method == CrossoverMethod.SIMULATED_BINARY:
            return self._simulated_binary_crossover(parent1, parent2)
        else:
            raise ValueError(f"未知の交叉手法: {self.method}")
    
    def _uniform_crossover(self, parent1: Individual, 
                          parent2: Individual) -> Tuple[Individual, Individual]:
        """一様交叉"""
        child1 = Individual()
        child2 = Individual()
        
        self._set_offspring_metadata(child1, child2, parent1, parent2)
        
        # 分野重み交叉
        child1.field_weights = {}
        child2.field_weights = {}
        
        for field_id in parent1.field_weights:
            if random.random() < 0.5:
                child1.field_weights[field_id] = parent1.field_weights[field_id]
                child2.field_weights[field_id] = parent2.field_weights[field_id]
            else:
                child1.field_weights[field_id] = parent2.field_weights[field_id]
                child2.field_weights[field_id] = parent1.field_weights[field_id]
        
        # 評価基準重み交叉
        child1.criteria_weights = {}
        child2.criteria_weights = {}
        
        for criterion in parent1.criteria_weights:
            if random.random() < 0.5:
                child1.criteria_weights[criterion] = parent1.criteria_weights[criterion]
                child2.criteria_weights[criterion] = parent2.criteria_weights[criterion]
            else:
                child1.criteria_weights[criterion] = parent2.criteria_weights[criterion]
                child2.criteria_weights[criterion] = parent1.criteria_weights[criterion]
        
        child1._normalize_weights()
        child2._normalize_weights()
        
        return child1, child2
    
    def _single_point_crossover(self, parent1: Individual, 
                               parent2: Individual) -> Tuple[Individual, Individual]:
        """一点交叉"""
        child1 = Individual()
        child2 = Individual()
        
        self._set_offspring_metadata(child1, child2, parent1, parent2)
        
        # 全遺伝子を統合
        all_genes1 = {**parent1.field_weights, **parent1.criteria_weights}
        all_genes2 = {**parent2.field_weights, **parent2.criteria_weights}
        
        gene_keys = list(all_genes1.keys())
        crossover_point = random.randint(1, len(gene_keys) - 1)
        
        # 交叉実行
        child1_genes = {}
        child2_genes = {}
        
        for i, key in enumerate(gene_keys):
            if i < crossover_point:
                child1_genes[key] = all_genes1[key]
                child2_genes[key] = all_genes2[key]
            else:
                child1_genes[key] = all_genes2[key]
                child2_genes[key] = all_genes1[key]
        
        # 分野重みと評価基準重みに分離
        self._split_genes(child1, child1_genes, parent1)
        self._split_genes(child2, child2_genes, parent2)
        
        child1._normalize_weights()
        child2._normalize_weights()
        
        return child1, child2
    
    def _two_point_crossover(self, parent1: Individual, 
                            parent2: Individual) -> Tuple[Individual, Individual]:
        """二点交叉"""
        child1 = Individual()
        child2 = Individual()
        
        self._set_offspring_metadata(child1, child2, parent1, parent2)
        
        # 全遺伝子を統合
        all_genes1 = {**parent1.field_weights, **parent1.criteria_weights}
        all_genes2 = {**parent2.field_weights, **parent2.criteria_weights}
        
        gene_keys = list(all_genes1.keys())
        point1 = random.randint(1, len(gene_keys) - 2)
        point2 = random.randint(point1 + 1, len(gene_keys) - 1)
        
        # 交叉実行
        child1_genes = {}
        child2_genes = {}
        
        for i, key in enumerate(gene_keys):
            if point1 <= i < point2:
                # 中間部分を交換
                child1_genes[key] = all_genes2[key]
                child2_genes[key] = all_genes1[key]
            else:
                child1_genes[key] = all_genes1[key]
                child2_genes[key] = all_genes2[key]
        
        # 分野重みと評価基準重みに分離
        self._split_genes(child1, child1_genes, parent1)
        self._split_genes(child2, child2_genes, parent2)
        
        child1._normalize_weights()
        child2._normalize_weights()
        
        return child1, child2
    
    def _blend_crossover(self, parent1: Individual, 
                        parent2: Individual) -> Tuple[Individual, Individual]:
        """ブレンド交叉（BLX-α）"""
        return parent1.blend_crossover_with(parent2, self.blend_alpha)
    
    def _simulated_binary_crossover(self, parent1: Individual, 
                                   parent2: Individual) -> Tuple[Individual, Individual]:
        """シミュレート二進交叉（SBX）"""
        child1 = Individual()
        child2 = Individual()
        
        self._set_offspring_metadata(child1, child2, parent1, parent2)
        
        # 分野重みのSBX
        child1.field_weights = {}
        child2.field_weights = {}
        
        for field_id in parent1.field_weights:
            val1, val2 = self._sbx_crossover_values(
                parent1.field_weights[field_id], 
                parent2.field_weights[field_id]
            )
            child1.field_weights[field_id] = val1
            child2.field_weights[field_id] = val2
        
        # 評価基準重みのSBX
        child1.criteria_weights = {}
        child2.criteria_weights = {}
        
        for criterion in parent1.criteria_weights:
            val1, val2 = self._sbx_crossover_values(
                parent1.criteria_weights[criterion], 
                parent2.criteria_weights[criterion]
            )
            child1.criteria_weights[criterion] = val1
            child2.criteria_weights[criterion] = val2
        
        child1._normalize_weights()
        child2._normalize_weights()
        
        return child1, child2
    
    def _sbx_crossover_values(self, parent1_val: float, parent2_val: float) -> Tuple[float, float]:
        """SBX交叉での値計算"""
        if abs(parent1_val - parent2_val) < 1e-14:
            return parent1_val, parent2_val
        
        # SBXアルゴリズム
        u = random.random()
        
        if u <= 0.5:
            beta = (2 * u) ** (1.0 / (self.sbx_eta + 1))
        else:
            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (self.sbx_eta + 1))
        
        child1_val = 0.5 * ((1 + beta) * parent1_val + (1 - beta) * parent2_val)
        child2_val = 0.5 * ((1 - beta) * parent1_val + (1 + beta) * parent2_val)
        
        # 範囲制限
        child1_val = max(0.01, min(1.0, child1_val))
        child2_val = max(0.01, min(1.0, child2_val))
        
        return child1_val, child2_val
    
    def _set_offspring_metadata(self, child1: Individual, child2: Individual,
                               parent1: Individual, parent2: Individual) -> None:
        """子個体のメタデータを設定"""
        max_generation = max(parent1.generation, parent2.generation)
        
        child1.generation = max_generation + 1
        child2.generation = max_generation + 1
        
        child1.parent_ids = [parent1.individual_id, parent2.individual_id]
        child2.parent_ids = [parent1.individual_id, parent2.individual_id]
    
    def _split_genes(self, child: Individual, genes: Dict[str, float], 
                    parent: Individual) -> None:
        """統合された遺伝子を分野重みと評価基準重みに分離"""
        child.field_weights = {}
        child.criteria_weights = {}
        
        for key, value in genes.items():
            if key in parent.field_weights:
                child.field_weights[key] = value
            elif key in parent.criteria_weights:
                child.criteria_weights[key] = value

class MutationOperator(GeneticOperator):
    """変異演算子"""
    
    def __init__(self, method: MutationMethod, mutation_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.mutation_rate = mutation_rate
        self.gaussian_sigma = kwargs.get('gaussian_sigma', 0.1)
        self.uniform_range = kwargs.get('uniform_range', 0.2)
        self.polynomial_eta = kwargs.get('polynomial_eta', 20.0)
        self.adaptive_factor = kwargs.get('adaptive_factor', 1.0)
    
    def apply(self, individual: Individual) -> Individual:
        """変異を実行"""
        
        if self.method == MutationMethod.GAUSSIAN:
            return self._gaussian_mutation(individual)
        elif self.method == MutationMethod.UNIFORM:
            return self._uniform_mutation(individual)
        elif self.method == MutationMethod.POLYNOMIAL:
            return self._polynomial_mutation(individual)
        elif self.method == MutationMethod.ADAPTIVE:
            return self._adaptive_mutation(individual)
        else:
            raise ValueError(f"未知の変異手法: {self.method}")
    
    def _gaussian_mutation(self, individual: Individual) -> Individual:
        """ガウス変異"""
        # 分野重み変異
        for field_id in individual.field_weights:
            if random.random() < self.mutation_rate:
                noise = random.gauss(0, self.gaussian_sigma)
                individual.field_weights[field_id] += noise
                individual.field_weights[field_id] = max(0.01, min(1.0, individual.field_weights[field_id]))
        
        # 評価基準重み変異
        for criterion in individual.criteria_weights:
            if random.random() < self.mutation_rate:
                noise = random.gauss(0, self.gaussian_sigma)
                individual.criteria_weights[criterion] += noise
                individual.criteria_weights[criterion] = max(0.01, min(1.0, individual.criteria_weights[criterion]))
        
        individual._normalize_weights()
        return individual
    
    def _uniform_mutation(self, individual: Individual) -> Individual:
        """一様変異"""
        # 分野重み変異
        for field_id in individual.field_weights:
            if random.random() < self.mutation_rate:
                noise = random.uniform(-self.uniform_range, self.uniform_range)
                individual.field_weights[field_id] += noise
                individual.field_weights[field_id] = max(0.01, min(1.0, individual.field_weights[field_id]))
        
        # 評価基準重み変異
        for criterion in individual.criteria_weights:
            if random.random() < self.mutation_rate:
                noise = random.uniform(-self.uniform_range, self.uniform_range)
                individual.criteria_weights[criterion] += noise
                individual.criteria_weights[criterion] = max(0.01, min(1.0, individual.criteria_weights[criterion]))
        
        individual._normalize_weights()
        return individual
    
    def _polynomial_mutation(self, individual: Individual) -> Individual:
        """多項式変異"""
        # 分野重み変異
        for field_id in individual.field_weights:
            if random.random() < self.mutation_rate:
                mutated_value = self._polynomial_mutate_value(individual.field_weights[field_id])
                individual.field_weights[field_id] = mutated_value
        
        # 評価基準重み変異
        for criterion in individual.criteria_weights:
            if random.random() < self.mutation_rate:
                mutated_value = self._polynomial_mutate_value(individual.criteria_weights[criterion])
                individual.criteria_weights[criterion] = mutated_value
        
        individual._normalize_weights()
        return individual
    
    def _polynomial_mutate_value(self, value: float) -> float:
        """多項式変異での値変更"""
        u = random.random()
        
        if u < 0.5:
            delta = (2 * u) ** (1.0 / (self.polynomial_eta + 1)) - 1.0
        else:
            delta = 1.0 - (2 * (1 - u)) ** (1.0 / (self.polynomial_eta + 1))
        
        mutated_value = value + delta
        return max(0.01, min(1.0, mutated_value))
    
    def _adaptive_mutation(self, individual: Individual) -> Individual:
        """適応的変異"""
        # 個体の適応度に基づいて変異強度を調整
        adaptive_rate = self.mutation_rate * self.adaptive_factor
        
        # 適応度が低い個体ほど強く変異
        if individual.fitness < 0.5:
            adaptive_rate *= 2.0
        elif individual.fitness > 0.8:
            adaptive_rate *= 0.5
        
        # ガウス変異を適応的変異率で実行
        original_rate = self.mutation_rate
        self.mutation_rate = adaptive_rate
        individual = self._gaussian_mutation(individual)
        self.mutation_rate = original_rate
        
        return individual

class OperatorFactory:
    """演算子ファクトリクラス"""
    
    @staticmethod
    def create_selection_operator(method: str, **kwargs) -> SelectionOperator:
        """選択演算子を作成"""
        method_enum = SelectionMethod(method)
        return SelectionOperator(method_enum, **kwargs)
    
    @staticmethod
    def create_crossover_operator(method: str, **kwargs) -> CrossoverOperator:
        """交叉演算子を作成"""
        method_enum = CrossoverMethod(method)
        return CrossoverOperator(method_enum, **kwargs)
    
    @staticmethod
    def create_mutation_operator(method: str, **kwargs) -> MutationOperator:
        """変異演算子を作成"""
        method_enum = MutationMethod(method)
        return MutationOperator(method_enum, **kwargs)