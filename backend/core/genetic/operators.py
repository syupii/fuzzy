# core/genetic/operators.py - 遺伝的操作

import numpy as np
import random
import math
from typing import Dict, List, Any, Optional, Tuple, Callable, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import logging

from core.genetic.individual import Individual, WeightVector, FuzzyTreeIndividual

logger = logging.getLogger(__name__)

class SelectionMethod(str, Enum):
    """選択手法"""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    STOCHASTIC_UNIVERSAL = "stochastic_universal"
    ELITIST = "elitist"

class CrossoverMethod(str, Enum):
    """交叉手法"""
    UNIFORM = "uniform"
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    ARITHMETIC = "arithmetic"
    BLEND_ALPHA = "blend_alpha"
    SBX = "sbx"  # Simulated Binary Crossover

class MutationMethod(str, Enum):
    """変異手法"""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    POLYNOMIAL = "polynomial"
    CREEP = "creep"
    ADAPTIVE = "adaptive"

@dataclass
class OperatorConfig:
    """遺伝的操作設定"""
    # 選択設定
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    tournament_size: int = 3
    selection_pressure: float = 2.0
    
    # 交叉設定
    crossover_method: CrossoverMethod = CrossoverMethod.UNIFORM
    crossover_rate: float = 0.8
    crossover_alpha: float = 0.5  # BLX-α用
    
    # 変異設定
    mutation_method: MutationMethod = MutationMethod.GAUSSIAN
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    adaptive_mutation: bool = False
    
    # エリート設定
    elite_preservation: bool = True
    elite_size: int = 2

class GeneticOperator(ABC):
    """遺伝的操作の抽象基底クラス"""
    
    def __init__(self, config: OperatorConfig):
        self.config = config
        self.operation_count = 0
        self.success_count = 0
    
    @abstractmethod
    def apply(self, *args, **kwargs) -> Any:
        """操作を適用"""
        pass
    
    def get_success_rate(self) -> float:
        """成功率を取得"""
        return self.success_count / max(self.operation_count, 1)

class SelectionOperator(GeneticOperator):
    """選択操作"""
    
    def apply(self, population: List[Individual], num_selections: int = 1) -> List[Individual]:
        """選択を実行"""
        
        self.operation_count += 1
        
        try:
            if self.config.selection_method == SelectionMethod.TOURNAMENT:
                selected = self._tournament_selection(population, num_selections)
            elif self.config.selection_method == SelectionMethod.ROULETTE:
                selected = self._roulette_selection(population, num_selections)
            elif self.config.selection_method == SelectionMethod.RANK:
                selected = self._rank_selection(population, num_selections)
            elif self.config.selection_method == SelectionMethod.STOCHASTIC_UNIVERSAL:
                selected = self._stochastic_universal_selection(population, num_selections)
            else:
                selected = self._tournament_selection(population, num_selections)
            
            self.success_count += 1
            return selected
            
        except Exception as e:
            logger.error(f"選択操作エラー: {e}")
            return random.sample(population, min(num_selections, len(population)))
    
    def _tournament_selection(self, population: List[Individual], num_selections: int) -> List[Individual]:
        """トーナメント選択"""
        selected = []
        
        for _ in range(num_selections):
            tournament_size = min(self.config.tournament_size, len(population))
            contestants = random.sample(population, tournament_size)
            
            # 適応度でソート
            contestants.sort(key=lambda x: x.get_fitness() or 0.0, reverse=True)
            
            # 選択圧を適用
            if self.config.selection_pressure > 1.0:
                # 指数的な選択確率
                probabilities = [
                    (self.config.selection_pressure ** (tournament_size - i - 1))
                    for i in range(tournament_size)
                ]
                total_prob = sum(probabilities)
                probabilities = [p / total_prob for p in probabilities]
                
                winner = np.random.choice(contestants, p=probabilities)
            else:
                winner = contestants[0]
            
            selected.append(winner)
        
        return selected
    
    def _roulette_selection(self, population: List[Individual], num_selections: int) -> List[Individual]:
        """ルーレット選択"""
        fitness_values = [individual.get_fitness() or 0.0 for individual in population]
        
        # 負の適応度の処理
        min_fitness = min(fitness_values)
        if min_fitness < 0:
            fitness_values = [f - min_fitness + 1e-6 for f in fitness_values]
        
        total_fitness = sum(fitness_values)
        if total_fitness == 0:
            return random.sample(population, min(num_selections, len(population)))
        
        # 正規化
        probabilities = [f / total_fitness for f in fitness_values]
        
        selected = []
        for _ in range(num_selections):
            selection_point = random.random()
            cumulative_prob = 0.0
            
            for i, prob in enumerate(probabilities):
                cumulative_prob += prob
                if cumulative_prob >= selection_point:
                    selected.append(population[i])
                    break
        
        return selected
    
    def _rank_selection(self, population: List[Individual], num_selections: int) -> List[Individual]:
        """ランク選択"""
        # 適応度でソート
        sorted_population = sorted(population, key=lambda x: x.get_fitness() or 0.0, reverse=True)
        
        n = len(sorted_population)
        ranks = list(range(n, 0, -1))  # n, n-1, ..., 1
        
        # 選択圧を適用
        adjusted_ranks = [rank ** self.config.selection_pressure for rank in ranks]
        total_rank = sum(adjusted_ranks)
        
        probabilities = [rank / total_rank for rank in adjusted_ranks]
        
        selected = []
        for _ in range(num_selections):
            selected_index = np.random.choice(len(sorted_population), p=probabilities)
            selected.append(sorted_population[selected_index])
        
        return selected
    
    def _stochastic_universal_selection(self, population: List[Individual], num_selections: int) -> List[Individual]:
        """確率的ユニバーサル選択"""
        fitness_values = [individual.get_fitness() or 0.0 for individual in population]
        
        # 負の適応度の処理
        min_fitness = min(fitness_values)
        if min_fitness < 0:
            fitness_values = [f - min_fitness + 1e-6 for f in fitness_values]
        
        total_fitness = sum(fitness_values)
        if total_fitness == 0:
            return random.sample(population, min(num_selections, len(population)))
        
        # ポインタ間隔
        pointer_distance = total_fitness / num_selections
        start_point = random.uniform(0, pointer_distance)
        
        selected = []
        cumulative_fitness = 0.0
        current_individual = 0
        
        for i in range(num_selections):
            pointer = start_point + i * pointer_distance
            
            while cumulative_fitness < pointer and current_individual < len(population):
                cumulative_fitness += fitness_values[current_individual]
                current_individual += 1
            
            if current_individual > 0:
                selected.append(population[current_individual - 1])
            else:
                selected.append(population[0])
        
        return selected

class CrossoverOperator(GeneticOperator):
    """交叉操作"""
    
    def apply(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """交叉を実行"""
        
        self.operation_count += 1
        
        try:
            if random.random() > self.config.crossover_rate:
                # 交叉しない場合は複製を返す
                return parent1.clone(), parent2.clone()
            
            if self.config.crossover_method == CrossoverMethod.UNIFORM:
                offspring = self._uniform_crossover(parent1, parent2)
            elif self.config.crossover_method == CrossoverMethod.SINGLE_POINT:
                offspring = self._single_point_crossover(parent1, parent2)
            elif self.config.crossover_method == CrossoverMethod.TWO_POINT:
                offspring = self._two_point_crossover(parent1, parent2)
            elif self.config.crossover_method == CrossoverMethod.ARITHMETIC:
                offspring = self._arithmetic_crossover(parent1, parent2)
            elif self.config.crossover_method == CrossoverMethod.BLEND_ALPHA:
                offspring = self._blend_alpha_crossover(parent1, parent2)
            elif self.config.crossover_method == CrossoverMethod.SBX:
                offspring = self._sbx_crossover(parent1, parent2)
            else:
                offspring = self._uniform_crossover(parent1, parent2)
            
            self.success_count += 1
            return offspring
            
        except Exception as e:
            logger.error(f"交叉操作エラー: {e}")
            return parent1.clone(), parent2.clone()
    
    def _uniform_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """一様交叉"""
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        genes1 = parent1.get_genes()
        genes2 = parent2.get_genes()
        
        # 共通の遺伝子に対して一様交叉
        common_genes = set(genes1.keys()) & set(genes2.keys())
        
        for gene_name in common_genes:
            if random.random() < 0.5:
                # 遺伝子を交換
                new_genes1 = child1.get_genes()
                new_genes2 = child2.get_genes()
                
                new_genes1[gene_name] = genes2[gene_name]
                new_genes2[gene_name] = genes1[gene_name]
                
                child1.set_genes(new_genes1)
                child2.set_genes(new_genes2)
        
        return child1, child2
    
    def _single_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """一点交叉"""
        genes1 = list(parent1.get_genes().items())
        genes2 = list(parent2.get_genes().items())
        
        if len(genes1) <= 1:
            return parent1.clone(), parent2.clone()
        
        # 交叉点の選択
        crossover_point = random.randint(1, len(genes1) - 1)
        
        # 交叉の実行
        child1_genes = dict(genes1[:crossover_point] + genes2[crossover_point:])
        child2_genes = dict(genes2[:crossover_point] + genes1[crossover_point:])
        
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        child1.set_genes(child1_genes)
        child2.set_genes(child2_genes)
        
        return child1, child2
    
    def _two_point_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """二点交叉"""
        genes1 = list(parent1.get_genes().items())
        genes2 = list(parent2.get_genes().items())
        
        if len(genes1) <= 2:
            return self._single_point_crossover(parent1, parent2)
        
        # 交叉点の選択
        point1 = random.randint(1, len(genes1) - 2)
        point2 = random.randint(point1 + 1, len(genes1) - 1)
        
        # 交叉の実行
        child1_genes = dict(genes1[:point1] + genes2[point1:point2] + genes1[point2:])
        child2_genes = dict(genes2[:point1] + genes1[point1:point2] + genes2[point2:])
        
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        child1.set_genes(child1_genes)
        child2.set_genes(child2_genes)
        
        return child1, child2
    
    def _arithmetic_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """算術交叉"""
        alpha = self.config.crossover_alpha
        
        genes1 = parent1.get_genes()
        genes2 = parent2.get_genes()
        
        child1_genes = {}
        child2_genes = {}
        
        common_genes = set(genes1.keys()) & set(genes2.keys())
        
        for gene_name in common_genes:
            val1 = genes1[gene_name]
            val2 = genes2[gene_name]
            
            # 算術交叉
            child1_genes[gene_name] = alpha * val1 + (1 - alpha) * val2
            child2_genes[gene_name] = (1 - alpha) * val1 + alpha * val2
        
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        child1.set_genes(child1_genes)
        child2.set_genes(child2_genes)
        
        return child1, child2
    
    def _blend_alpha_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """BLX-α交叉"""
        alpha = self.config.crossover_alpha
        
        genes1 = parent1.get_genes()
        genes2 = parent2.get_genes()
        
        child1_genes = {}
        child2_genes = {}
        
        common_genes = set(genes1.keys()) & set(genes2.keys())
        
        for gene_name in common_genes:
            val1 = genes1[gene_name]
            val2 = genes2[gene_name]
            
            # 範囲の計算
            min_val = min(val1, val2)
            max_val = max(val1, val2)
            range_val = max_val - min_val
            
            # 拡張範囲
            extended_min = min_val - alpha * range_val
            extended_max = max_val + alpha * range_val
            
            # 子の値を生成
            child1_genes[gene_name] = random.uniform(extended_min, extended_max)
            child2_genes[gene_name] = random.uniform(extended_min, extended_max)
        
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        child1.set_genes(child1_genes)
        child2.set_genes(child2_genes)
        
        return child1, child2
    
    def _sbx_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Simulated Binary Crossover (SBX)"""
        eta = 2.0  # 分布指標
        
        genes1 = parent1.get_genes()
        genes2 = parent2.get_genes()
        
        child1_genes = {}
        child2_genes = {}
        
        common_genes = set(genes1.keys()) & set(genes2.keys())
        
        for gene_name in common_genes:
            val1 = genes1[gene_name]
            val2 = genes2[gene_name]
            
            if abs(val1 - val2) < 1e-14:
                child1_genes[gene_name] = val1
                child2_genes[gene_name] = val2
                continue
            
            u = random.random()
            
            if u <= 0.5:
                beta = (2 * u) ** (1.0 / (eta + 1))
            else:
                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
            
            child1_genes[gene_name] = 0.5 * ((1 + beta) * val1 + (1 - beta) * val2)
            child2_genes[gene_name] = 0.5 * ((1 - beta) * val1 + (1 + beta) * val2)
        
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        child1.set_genes(child1_genes)
        child2.set_genes(child2_genes)
        
        return child1, child2

class MutationOperator(GeneticOperator):
    """変異操作"""
    
    def __init__(self, config: OperatorConfig):
        super().__init__(config)
        self.adaptive_parameters = {
            "current_mutation_rate": config.mutation_rate,
            "current_mutation_strength": config.mutation_strength,
            "generation_without_improvement": 0
        }
    
    def apply(self, individual: Individual) -> Individual:
        """変異を実行"""
        
        self.operation_count += 1
        
        try:
            mutated = individual.clone()
            
            # 適応的変異率の調整
            if self.config.adaptive_mutation:
                self._adjust_adaptive_parameters()
            
            current_rate = self.adaptive_parameters["current_mutation_rate"]
            current_strength = self.adaptive_parameters["current_mutation_strength"]
            
            if self.config.mutation_method == MutationMethod.GAUSSIAN:
                self._gaussian_mutation(mutated, current_rate, current_strength)
            elif self.config.mutation_method == MutationMethod.UNIFORM:
                self._uniform_mutation(mutated, current_rate, current_strength)
            elif self.config.mutation_method == MutationMethod.POLYNOMIAL:
                self._polynomial_mutation(mutated, current_rate, current_strength)
            elif self.config.mutation_method == MutationMethod.CREEP:
                self._creep_mutation(mutated, current_rate, current_strength)
            else:
                self._gaussian_mutation(mutated, current_rate, current_strength)
            
            self.success_count += 1
            return mutated
            
        except Exception as e:
            logger.error(f"変異操作エラー: {e}")
            return individual.clone()
    
    def _gaussian_mutation(self, individual: Individual, mutation_rate: float, mutation_strength: float):
        """ガウシアン変異"""
        genes = individual.get_genes()
        
        for gene_name, value in genes.items():
            if random.random() < mutation_rate:
                noise = np.random.normal(0, mutation_strength)
                genes[gene_name] = max(0.0, min(1.0, value + noise))
        
        individual.set_genes(genes)
    
    def _uniform_mutation(self, individual: Individual, mutation_rate: float, mutation_strength: float):
        """一様変異"""
        genes = individual.get_genes()
        
        for gene_name, value in genes.items():
            if random.random() < mutation_rate:
                noise = random.uniform(-mutation_strength, mutation_strength)
                genes[gene_name] = max(0.0, min(1.0, value + noise))
        
        individual.set_genes(genes)
    
    def _polynomial_mutation(self, individual: Individual, mutation_rate: float, mutation_strength: float):
        """多項式変異"""
        eta = 20.0  # 分布指標
        genes = individual.get_genes()
        
        for gene_name, value in genes.items():
            if random.random() < mutation_rate:
                u = random.random()
                
                if u < 0.5:
                    delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                
                mutated_value = value + delta * mutation_strength
                genes[gene_name] = max(0.0, min(1.0, mutated_value))
        
        individual.set_genes(genes)
    
    def _creep_mutation(self, individual: Individual, mutation_rate: float, mutation_strength: float):
        """クリープ変異"""
        genes = individual.get_genes()
        
        for gene_name, value in genes.items():
            if random.random() < mutation_rate:
                # 小さな変化のみ
                direction = random.choice([-1, 1])
                change = direction * mutation_strength * 0.1
                genes[gene_name] = max(0.0, min(1.0, value + change))
        
        individual.set_genes(genes)
    
    def _adjust_adaptive_parameters(self):
        """適応的パラメータの調整"""
        # 改善がない世代が続く場合は変異率を上げる
        if self.adaptive_parameters["generation_without_improvement"] > 5:
            self.adaptive_parameters["current_mutation_rate"] = min(
                0.5, 
                self.adaptive_parameters["current_mutation_rate"] * 1.1
            )
            self.adaptive_parameters["current_mutation_strength"] = min(
                0.3,
                self.adaptive_parameters["current_mutation_strength"] * 1.05
            )
        else:
            # 改善がある場合は元に戻す
            self.adaptive_parameters["current_mutation_rate"] = self.config.mutation_rate
            self.adaptive_parameters["current_mutation_strength"] = self.config.mutation_strength
    
    def update_improvement_status(self, has_improved: bool):
        """改善状況の更新"""
        if has_improved:
            self.adaptive_parameters["generation_without_improvement"] = 0
        else:
            self.adaptive_parameters["generation_without_improvement"] += 1

class OperatorFactory:
    """遺伝的操作ファクトリ"""
    
    @staticmethod
    def create_selection_operator(config: OperatorConfig) -> SelectionOperator:
        """選択操作の作成"""
        return SelectionOperator(config)
    
    @staticmethod
    def create_crossover_operator(config: OperatorConfig) -> CrossoverOperator:
        """交叉操作の作成"""
        return CrossoverOperator(config)
    
    @staticmethod
    def create_mutation_operator(config: OperatorConfig) -> MutationOperator:
        """変異操作の作成"""
        return MutationOperator(config)
    
    @staticmethod
    def create_all_operators(config: OperatorConfig) -> Dict[str, GeneticOperator]:
        """全操作の作成"""
        return {
            "selection": OperatorFactory.create_selection_operator(config),
            "crossover": OperatorFactory.create_crossover_operator(config),
            "mutation": OperatorFactory.create_mutation_operator(config)
        }

# 使用例とテスト
def test_genetic_operators():
    """遺伝的操作のテスト"""
    
    print("🧬 遺伝的操作テスト開始")
    
    # 設定の作成
    config = OperatorConfig(
        selection_method=SelectionMethod.TOURNAMENT,
        crossover_method=CrossoverMethod.UNIFORM,
        mutation_method=MutationMethod.GAUSSIAN,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    
    # 操作オブジェクトの作成
    operators = OperatorFactory.create_all_operators(config)
    
    # テスト用個体の作成
    parent1 = WeightVector(weight_names=["w1", "w2", "w3"])
    parent2 = WeightVector(weight_names=["w1", "w2", "w3"])
    
    # 適応度設定
    parent1.set_fitness(0.8)
    parent2.set_fitness(0.6)
    
    population = [parent1, parent2]
    
    print(f"👨‍👩‍👧‍👦 親1の遺伝子: {parent1.get_genes()}")
    print(f"👨‍👩‍👧‍👦 親2の遺伝子: {parent2.get_genes()}")
    
    # 選択テスト
    selected = operators["selection"].apply(population, 1)
    print(f"\n🎯 選択結果: 個体{selected[0].individual_id} (適応度: {selected[0].get_fitness()})")
    
    # 交叉テスト
    child1, child2 = operators["crossover"].apply(parent1, parent2)
    print(f"\n👶 子1の遺伝子: {child1.get_genes()}")
    print(f"👶 子2の遺伝子: {child2.get_genes()}")
    
    # 変異テスト
    mutated = operators["mutation"].apply(child1)
    print(f"\n🔄 変異後の遺伝子: {mutated.get_genes()}")
    
    # 成功率の表示
    print(f"\n📊 操作成功率:")
    for name, operator in operators.items():
        print(f"  {name}: {operator.get_success_rate():.3f}")
    
    print("✅ 遺伝的操作テスト完了")

if __name__ == "__main__":
    test_genetic_operators()