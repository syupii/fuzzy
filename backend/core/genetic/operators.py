# core/genetic/operators.py - 遺伝的操作（完全版）

import numpy as np
import random
import math
from typing import Dict, List, Any, Optional, Tuple, Callable, Type
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class SelectionMethod(str, Enum):
    """選択手法"""
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    ELITE = "elite"
    STOCHASTIC_UNIVERSAL = "stochastic_universal"

class CrossoverMethod(str, Enum):
    """交叉手法"""
    UNIFORM = "uniform"
    SINGLE_POINT = "single_point"
    TWO_POINT = "two_point"
    ARITHMETIC = "arithmetic"
    BLEND_ALPHA = "blend_alpha"

class MutationMethod(str, Enum):
    """変異手法"""
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    POLYNOMIAL = "polynomial"
    CREEP = "creep"
    RANDOM_RESETTING = "random_resetting"

@dataclass
class OperatorConfig:
    """遺伝的操作設定"""
    # 選択設定
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    tournament_size: int = 3
    selection_pressure: float = 1.5
    
    # 交叉設定
    crossover_method: CrossoverMethod = CrossoverMethod.UNIFORM
    crossover_rate: float = 0.8
    crossover_alpha: float = 0.5
    
    # 変異設定
    mutation_method: MutationMethod = MutationMethod.GAUSSIAN
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    polynomial_eta: float = 20.0
    
    # 適応設定
    adaptive_operators: bool = False
    operator_probabilities: Dict[str, float] = None

class GeneticOperator(ABC):
    """遺伝的操作の抽象基底クラス"""
    
    def __init__(self, config: OperatorConfig):
        self.config = config
        self.usage_count = 0
        self.success_count = 0
        
    @abstractmethod
    def apply(self, *args, **kwargs) -> Any:
        """操作を適用"""
        pass
    
    def get_success_rate(self) -> float:
        """成功率を取得"""
        return self.success_count / self.usage_count if self.usage_count > 0 else 0.0

class SelectionOperator(GeneticOperator):
    """選択操作"""
    
    def apply(self, population: List[Any], num_parents: int) -> List[Any]:
        """選択を実行"""
        
        self.usage_count += 1
        
        try:
            if self.config.selection_method == SelectionMethod.TOURNAMENT:
                return self._tournament_selection(population, num_parents)
            elif self.config.selection_method == SelectionMethod.ROULETTE:
                return self._roulette_selection(population, num_parents)
            elif self.config.selection_method == SelectionMethod.RANK:
                return self._rank_selection(population, num_parents)
            elif self.config.selection_method == SelectionMethod.ELITE:
                return self._elite_selection(population, num_parents)
            else:
                return self._tournament_selection(population, num_parents)
                
        except Exception as e:
            logger.warning(f"選択操作エラー: {e}")
            return population[:num_parents]
    
    def _tournament_selection(self, population: List[Any], num_parents: int) -> List[Any]:
        """トーナメント選択"""
        
        selected = []
        tournament_size = min(self.config.tournament_size, len(population))
        
        for _ in range(num_parents):
            # トーナメント参加者をランダム選択
            tournament = random.sample(population, tournament_size)
            
            # 最高適応度の個体を選択
            winner = max(tournament, key=lambda x: getattr(x, 'fitness_value', 0.0))
            selected.append(winner)
        
        self.success_count += 1
        return selected
    
    def _roulette_selection(self, population: List[Any], num_parents: int) -> List[Any]:
        """ルーレット選択"""
        
        # 適応度の取得
        fitness_values = [getattr(ind, 'fitness_value', 0.0) for ind in population]
        
        # 負の適応度を正に変換
        min_fitness = min(fitness_values)
        if min_fitness < 0:
            fitness_values = [f - min_fitness + 1e-6 for f in fitness_values]
        
        # 累積確率の計算
        total_fitness = sum(fitness_values)
        if total_fitness == 0:
            return random.sample(population, min(num_parents, len(population)))
        
        probabilities = [f / total_fitness for f in fitness_values]
        cumulative_probs = []
        cumulative = 0.0
        for prob in probabilities:
            cumulative += prob
            cumulative_probs.append(cumulative)
        
        # 選択実行
        selected = []
        for _ in range(num_parents):
            r = random.random()
            for i, cum_prob in enumerate(cumulative_probs):
                if r <= cum_prob:
                    selected.append(population[i])
                    break
        
        self.success_count += 1
        return selected
    
    def _rank_selection(self, population: List[Any], num_parents: int) -> List[Any]:
        """ランク選択"""
        
        # 適応度順にソート
        sorted_pop = sorted(population, key=lambda x: getattr(x, 'fitness_value', 0.0))
        
        # ランクベースの選択確率
        n = len(sorted_pop)
        selection_probs = []
        
        for i in range(n):
            rank = i + 1
            prob = (2 - self.config.selection_pressure) / n + \
                   (2 * rank * (self.config.selection_pressure - 1)) / (n * (n - 1))
            selection_probs.append(prob)
        
        # 累積確率による選択
        cumulative_probs = []
        cumulative = 0.0
        for prob in selection_probs:
            cumulative += prob
            cumulative_probs.append(cumulative)
        
        selected = []
        for _ in range(num_parents):
            r = random.random()
            for i, cum_prob in enumerate(cumulative_probs):
                if r <= cum_prob:
                    selected.append(sorted_pop[i])
                    break
        
        self.success_count += 1
        return selected
    
    def _elite_selection(self, population: List[Any], num_parents: int) -> List[Any]:
        """エリート選択"""
        
        # 適応度順にソートして上位を選択
        sorted_pop = sorted(population, 
                          key=lambda x: getattr(x, 'fitness_value', 0.0), 
                          reverse=True)
        
        selected = sorted_pop[:num_parents]
        self.success_count += 1
        return selected

class CrossoverOperator(GeneticOperator):
    """交叉操作"""
    
    def apply(self, parent1: Any, parent2: Any) -> Tuple[Any, Any]:
        """交叉を実行"""
        
        self.usage_count += 1
        
        # 交叉率による実行判定
        if random.random() > self.config.crossover_rate:
            return self._clone_individuals(parent1, parent2)
        
        try:
            if self.config.crossover_method == CrossoverMethod.UNIFORM:
                return self._uniform_crossover(parent1, parent2)
            elif self.config.crossover_method == CrossoverMethod.SINGLE_POINT:
                return self._single_point_crossover(parent1, parent2)
            elif self.config.crossover_method == CrossoverMethod.TWO_POINT:
                return self._two_point_crossover(parent1, parent2)
            elif self.config.crossover_method == CrossoverMethod.ARITHMETIC:
                return self._arithmetic_crossover(parent1, parent2)
            else:
                return self._uniform_crossover(parent1, parent2)
                
        except Exception as e:
            logger.warning(f"交叉操作エラー: {e}")
            return self._clone_individuals(parent1, parent2)
    
    def _clone_individuals(self, parent1: Any, parent2: Any) -> Tuple[Any, Any]:
        """個体の複製"""
        
        if hasattr(parent1, 'clone'):
            child1 = parent1.clone()
        else:
            child1 = parent1
            
        if hasattr(parent2, 'clone'):
            child2 = parent2.clone()
        else:
            child2 = parent2
            
        return child1, child2
    
    def _uniform_crossover(self, parent1: Any, parent2: Any) -> Tuple[Any, Any]:
        """一様交叉"""
        
        child1, child2 = self._clone_individuals(parent1, parent2)
        
        # 遺伝子レベルでの交叉
        if hasattr(parent1, 'get_genes') and hasattr(parent2, 'get_genes'):
            genes1 = parent1.get_genes()
            genes2 = parent2.get_genes()
            
            new_genes1 = {}
            new_genes2 = {}
            
            all_keys = set(genes1.keys()) | set(genes2.keys())
            
            for key in all_keys:
                if random.random() < 0.5:
                    new_genes1[key] = genes1.get(key, 0.0)
                    new_genes2[key] = genes2.get(key, 0.0)
                else:
                    new_genes1[key] = genes2.get(key, 0.0)
                    new_genes2[key] = genes1.get(key, 0.0)
            
            if hasattr(child1, 'set_genes'):
                child1.set_genes(new_genes1)
            if hasattr(child2, 'set_genes'):
                child2.set_genes(new_genes2)
        
        # 適応度をリセット
        if hasattr(child1, 'fitness_value'):
            child1.fitness_value = None
        if hasattr(child2, 'fitness_value'):
            child2.fitness_value = None
            
        self.success_count += 1
        return child1, child2
    
    def _single_point_crossover(self, parent1: Any, parent2: Any) -> Tuple[Any, Any]:
        """一点交叉"""
        
        child1, child2 = self._clone_individuals(parent1, parent2)
        
        if hasattr(parent1, 'get_genes') and hasattr(parent2, 'get_genes'):
            genes1 = parent1.get_genes()
            genes2 = parent2.get_genes()
            
            if genes1 and genes2:
                keys = list(genes1.keys())
                if len(keys) > 1:
                    crossover_point = random.randint(1, len(keys) - 1)
                    
                    new_genes1 = {}
                    new_genes2 = {}
                    
                    for i, key in enumerate(keys):
                        if i < crossover_point:
                            new_genes1[key] = genes1.get(key, 0.0)
                            new_genes2[key] = genes2.get(key, 0.0)
                        else:
                            new_genes1[key] = genes2.get(key, 0.0)
                            new_genes2[key] = genes1.get(key, 0.0)
                    
                    if hasattr(child1, 'set_genes'):
                        child1.set_genes(new_genes1)
                    if hasattr(child2, 'set_genes'):
                        child2.set_genes(new_genes2)
        
        # 適応度をリセット
        if hasattr(child1, 'fitness_value'):
            child1.fitness_value = None
        if hasattr(child2, 'fitness_value'):
            child2.fitness_value = None
            
        self.success_count += 1
        return child1, child2
    
    def _two_point_crossover(self, parent1: Any, parent2: Any) -> Tuple[Any, Any]:
        """二点交叉"""
        
        child1, child2 = self._clone_individuals(parent1, parent2)
        
        if hasattr(parent1, 'get_genes') and hasattr(parent2, 'get_genes'):
            genes1 = parent1.get_genes()
            genes2 = parent2.get_genes()
            
            if genes1 and genes2:
                keys = list(genes1.keys())
                if len(keys) > 2:
                    point1 = random.randint(1, len(keys) - 2)
                    point2 = random.randint(point1 + 1, len(keys) - 1)
                    
                    new_genes1 = {}
                    new_genes2 = {}
                    
                    for i, key in enumerate(keys):
                        if point1 <= i < point2:
                            new_genes1[key] = genes2.get(key, 0.0)
                            new_genes2[key] = genes1.get(key, 0.0)
                        else:
                            new_genes1[key] = genes1.get(key, 0.0)
                            new_genes2[key] = genes2.get(key, 0.0)
                    
                    if hasattr(child1, 'set_genes'):
                        child1.set_genes(new_genes1)
                    if hasattr(child2, 'set_genes'):
                        child2.set_genes(new_genes2)
        
        # 適応度をリセット
        if hasattr(child1, 'fitness_value'):
            child1.fitness_value = None
        if hasattr(child2, 'fitness_value'):
            child2.fitness_value = None
            
        self.success_count += 1
        return child1, child2
    
    def _arithmetic_crossover(self, parent1: Any, parent2: Any) -> Tuple[Any, Any]:
        """算術交叉"""
        
        child1, child2 = self._clone_individuals(parent1, parent2)
        alpha = self.config.crossover_alpha
        
        if hasattr(parent1, 'get_genes') and hasattr(parent2, 'get_genes'):
            genes1 = parent1.get_genes()
            genes2 = parent2.get_genes()
            
            new_genes1 = {}
            new_genes2 = {}
            
            all_keys = set(genes1.keys()) | set(genes2.keys())
            
            for key in all_keys:
                val1 = genes1.get(key, 0.0)
                val2 = genes2.get(key, 0.0)
                
                new_genes1[key] = alpha * val1 + (1 - alpha) * val2
                new_genes2[key] = alpha * val2 + (1 - alpha) * val1
            
            if hasattr(child1, 'set_genes'):
                child1.set_genes(new_genes1)
            if hasattr(child2, 'set_genes'):
                child2.set_genes(new_genes2)
        
        # 適応度をリセット
        if hasattr(child1, 'fitness_value'):
            child1.fitness_value = None
        if hasattr(child2, 'fitness_value'):
            child2.fitness_value = None
            
        self.success_count += 1
        return child1, child2

class MutationOperator(GeneticOperator):
    """変異操作"""
    
    def apply(self, individual: Any) -> Any:
        """変異を実行"""
        
        self.usage_count += 1
        
        try:
            if self.config.mutation_method == MutationMethod.GAUSSIAN:
                return self._gaussian_mutation(individual)
            elif self.config.mutation_method == MutationMethod.UNIFORM:
                return self._uniform_mutation(individual)
            elif self.config.mutation_method == MutationMethod.POLYNOMIAL:
                return self._polynomial_mutation(individual)
            elif self.config.mutation_method == MutationMethod.CREEP:
                return self._creep_mutation(individual)
            else:
                return self._gaussian_mutation(individual)
                
        except Exception as e:
            logger.warning(f"変異操作エラー: {e}")
            return individual
    
    def _gaussian_mutation(self, individual: Any) -> Any:
        """ガウシアン変異"""
        
        if hasattr(individual, 'get_genes') and hasattr(individual, 'set_genes'):
            genes = individual.get_genes()
            new_genes = {}
            
            for key, value in genes.items():
                if random.random() < self.config.mutation_rate:
                    # ガウシアンノイズを追加
                    noise = np.random.normal(0, self.config.mutation_strength)
                    new_value = value + noise
                    
                    # 値の範囲制限（0-1）
                    new_value = max(0.0, min(1.0, new_value))
                    new_genes[key] = new_value
                else:
                    new_genes[key] = value
            
            individual.set_genes(new_genes)
        elif hasattr(individual, 'mutate'):
            individual.mutate(self.config.mutation_rate, self.config.mutation_strength)
        
        # 適応度をリセット
        if hasattr(individual, 'fitness_value'):
            individual.fitness_value = None
            
        self.success_count += 1
        return individual
    
    def _uniform_mutation(self, individual: Any) -> Any:
        """一様変異"""
        
        if hasattr(individual, 'get_genes') and hasattr(individual, 'set_genes'):
            genes = individual.get_genes()
            new_genes = {}
            
            for key, value in genes.items():
                if random.random() < self.config.mutation_rate:
                    # 一様分布からランダム値を生成
                    new_value = random.uniform(0.0, 1.0)
                    new_genes[key] = new_value
                else:
                    new_genes[key] = value
            
            individual.set_genes(new_genes)
        
        # 適応度をリセット
        if hasattr(individual, 'fitness_value'):
            individual.fitness_value = None
            
        self.success_count += 1
        return individual
    
    def _polynomial_mutation(self, individual: Any) -> Any:
        """多項式変異"""
        
        if hasattr(individual, 'get_genes') and hasattr(individual, 'set_genes'):
            genes = individual.get_genes()
            new_genes = {}
            eta = self.config.polynomial_eta
            
            for key, value in genes.items():
                if random.random() < self.config.mutation_rate:
                    u = random.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    
                    new_value = value + delta * self.config.mutation_strength
                    new_value = max(0.0, min(1.0, new_value))
                    new_genes[key] = new_value
                else:
                    new_genes[key] = value
            
            individual.set_genes(new_genes)
        
        # 適応度をリセット
        if hasattr(individual, 'fitness_value'):
            individual.fitness_value = None
            
        self.success_count += 1
        return individual
    
    def _creep_mutation(self, individual: Any) -> Any:
        """クリープ変異（小さな変化）"""
        
        if hasattr(individual, 'get_genes') and hasattr(individual, 'set_genes'):
            genes = individual.get_genes()
            new_genes = {}
            
            for key, value in genes.items():
                if random.random() < self.config.mutation_rate:
                    # 小さなランダム変化
                    change = random.uniform(-0.1, 0.1) * self.config.mutation_strength
                    new_value = value + change
                    new_value = max(0.0, min(1.0, new_value))
                    new_genes[key] = new_value
                else:
                    new_genes[key] = value
            
            individual.set_genes(new_genes)
        
        # 適応度をリセット
        if hasattr(individual, 'fitness_value'):
            individual.fitness_value = None
            
        self.success_count += 1
        return individual

class OperatorFactory:
    """遺伝的操作ファクトリ"""
    
    @staticmethod
    def create_selection_operator(config: OperatorConfig) -> SelectionOperator:
        """選択操作を作成"""
        return SelectionOperator(config)
    
    @staticmethod
    def create_crossover_operator(config: OperatorConfig) -> CrossoverOperator:
        """交叉操作を作成"""
        return CrossoverOperator(config)
    
    @staticmethod
    def create_mutation_operator(config: OperatorConfig) -> MutationOperator:
        """変異操作を作成"""
        return MutationOperator(config)
    
    @staticmethod
    def create_all_operators(config: OperatorConfig) -> Dict[str, GeneticOperator]:
        """全操作を作成"""
        return {
            "selection": OperatorFactory.create_selection_operator(config),
            "crossover": OperatorFactory.create_crossover_operator(config),
            "mutation": OperatorFactory.create_mutation_operator(config)
        }

# 使用例とテスト
def test_genetic_operators():
    """遺伝的操作のテスト"""
    
    print("🧬 遺伝的操作テスト開始")
    
    # 設定作成
    config = OperatorConfig(
        selection_method=SelectionMethod.TOURNAMENT,
        crossover_method=CrossoverMethod.UNIFORM,
        mutation_method=MutationMethod.GAUSSIAN,
        tournament_size=3,
        crossover_rate=0.8,
        mutation_rate=0.1,
        mutation_strength=0.1
    )
    
    # 操作作成
    operators = OperatorFactory.create_all_operators(config)
    
    print(f"✅ 遺伝的操作作成完了")
    print(f"  選択: {config.selection_method}")
    print(f"  交叉: {config.crossover_method}")
    print(f"  変異: {config.mutation_method}")
    
    # 簡易個体クラス（テスト用）
    class TestIndividual:
        def __init__(self):
            self.genes = {"w1": random.random(), "w2": random.random(), "w3": random.random()}
            self.fitness_value = sum(self.genes.values())
        
        def get_genes(self):
            return self.genes
        
        def set_genes(self, genes):
            self.genes = genes
        
        def clone(self):
            clone = TestIndividual()
            clone.genes = self.genes.copy()
            clone.fitness_value = self.fitness_value
            return clone
    
    # テスト集団作成
    population = [TestIndividual() for _ in range(10)]
    
    # 選択テスト
    print(f"\n🎯 選択テスト:")
    parents = operators["selection"].apply(population, 2)
    print(f"  選択された親: {len(parents)}個体")
    print(f"  親1適応度: {parents[0].fitness_value:.3f}")
    print(f"  親2適応度: {parents[1].fitness_value:.3f}")
    
    # 交叉テスト
    print(f"\n🔀 交叉テスト:")
    parent1, parent2 = parents[0], parents[1]
    print(f"  親1遺伝子: {parent1.get_genes()}")
    print(f"  親2遺伝子: {parent2.get_genes()}")
    
    child1, child2 = operators["crossover"].apply(parent1, parent2)
    print(f"  子1遺伝子: {child1.get_genes()}")
    print(f"  子2遺伝子: {child2.get_genes()}")
    
    # 変異テスト
    print(f"\n🔄 変異テスト:")
    original_genes = child1.get_genes().copy()
    print(f"  変異前: {original_genes}")
    
    mutated = operators["mutation"].apply(child1)
    print(f"  変異後: {mutated.get_genes()}")
    
    print("✅ 遺伝的操作テスト完了")

if __name__ == "__main__":
    test_genetic_operators()