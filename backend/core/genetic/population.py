"""
遺伝的アルゴリズム - 集団管理クラス
個体群の生成、選択、交叉、突然変異を管理
"""

import random
import numpy as np
from typing import List, Tuple, Callable
from dataclasses import dataclass


@dataclass
class PopulationConfig:
    """集団設定"""
    population_size: int = 50
    elite_size: int = 5
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    tournament_size: int = 5


class Population:
    """遺伝的アルゴリズムの集団"""
    
    def __init__(self, config: PopulationConfig = None):
        """
        Args:
            config: 集団設定
        """
        self.config = config or PopulationConfig()
        self.individuals = []
        self.generation = 0
        self.best_individual = None
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def initialize(self):
        """初期集団を生成"""
        from .individual import Individual
        
        self.individuals = [
            Individual() 
            for _ in range(self.config.population_size)
        ]
        self.generation = 0
        print(f"✅ 初期集団生成完了: {self.config.population_size}個体")
    
    def evaluate(self, fitness_function: Callable):
        """全個体の適合度を評価
        
        Args:
            fitness_function: 適合度計算関数
        """
        for individual in self.individuals:
            individual.fitness = fitness_function(individual)
        
        self._update_statistics()
    
    def _update_statistics(self):
        """統計情報を更新"""
        if not self.individuals:
            return
        
        current_best = max(self.individuals, key=lambda ind: ind.fitness)
        
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best.copy()
        
        self.best_fitness_history.append(self.best_individual.fitness)
        avg_fitness = np.mean([ind.fitness for ind in self.individuals])
        self.avg_fitness_history.append(avg_fitness)
    
    def selection_tournament(self, k: int = None):
        """トーナメント選択
        
        Args:
            k: トーナメントサイズ
            
        Returns:
            選択された個体
        """
        if k is None:
            k = self.config.tournament_size
        
        tournament = random.sample(self.individuals, k)
        return max(tournament, key=lambda ind: ind.fitness)
    
    def selection_roulette(self):
        """ルーレット選択
        
        Returns:
            選択された個体
        """
        total_fitness = sum(ind.fitness for ind in self.individuals)
        
        if total_fitness == 0:
            return random.choice(self.individuals)
        
        pick = random.uniform(0, total_fitness)
        current = 0
        
        for individual in self.individuals:
            current += individual.fitness
            if current >= pick:
                return individual
        
        return self.individuals[-1]
    
    def evolve(self, selection_method: str = "tournament"):
        """次世代を生成
        
        Args:
            selection_method: 選択方法
            
        Returns:
            次世代の個体群
        """
        if selection_method == "tournament":
            select = self.selection_tournament
        elif selection_method == "roulette":
            select = self.selection_roulette
        else:
            select = self.selection_tournament
        
        next_generation = []
        
        # エリート保存
        elite_individuals = sorted(
            self.individuals, 
            key=lambda ind: ind.fitness, 
            reverse=True
        )[:self.config.elite_size]
        
        for elite in elite_individuals:
            next_generation.append(elite.copy())
        
        # 残りを生成
        while len(next_generation) < self.config.population_size:
            parent1 = select()
            parent2 = select()
            
            # 交叉
            if random.random() < self.config.crossover_rate:
                child1, child2 = parent1.crossover(parent2)
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()
            
            # 突然変異
            child1.mutate(self.config.mutation_rate)
            child2.mutate(self.config.mutation_rate)
            
            next_generation.append(child1)
            if len(next_generation) < self.config.population_size:
                next_generation.append(child2)
        
        for individual in next_generation:
            individual.age += 1
        
        self.individuals = next_generation
        self.generation += 1
        
        return self.individuals
    
    def get_best(self, n: int = 1):
        """上位n個体を取得
        
        Args:
            n: 取得する個体数
            
        Returns:
            上位n個体のリスト
        """
        sorted_individuals = sorted(
            self.individuals,
            key=lambda ind: ind.fitness,
            reverse=True
        )
        return sorted_individuals[:n]
    
    def get_statistics(self) -> dict:
        """統計情報を取得
        
        Returns:
            統計情報の辞書
        """
        if not self.individuals:
            return {}
        
        fitnesses = [ind.fitness for ind in self.individuals]
        
        return {
            "generation": self.generation,
            "population_size": len(self.individuals),
            "best_fitness": max(fitnesses),
            "worst_fitness": min(fitnesses),
            "avg_fitness": np.mean(fitnesses),
            "std_fitness": np.std(fitnesses),
            "median_fitness": np.median(fitnesses),
            "best_individual": self.best_individual,
            "diversity": self._calculate_diversity()
        }
    
    def _calculate_diversity(self) -> float:
        """集団の多様性を計算
        
        Returns:
            多様性スコア（0〜1）
        """
        if len(self.individuals) < 2:
            return 0.0
        
        from .individual import Individual
        
        level1_features = [ind.gene.level1_feature for ind in self.individuals]
        unique_level1 = len(set(level1_features))
        diversity_level1 = unique_level1 / len(Individual.AVAILABLE_FEATURES)
        
        fitnesses = [ind.fitness for ind in self.individuals]
        fitness_std = np.std(fitnesses)
        fitness_range = max(fitnesses) - min(fitnesses)
        
        if fitness_range > 0:
            diversity_fitness = min(1.0, fitness_std / fitness_range)
        else:
            diversity_fitness = 0.0
        
        diversity = (diversity_level1 + diversity_fitness) / 2
        
        return diversity
    
    def has_converged(self, threshold: float = 0.001, patience: int = 10) -> bool:
        """収束判定
        
        Args:
            threshold: 改善の閾値
            patience: 改善がない世代数の許容値
            
        Returns:
            収束したかどうか
        """
        if len(self.best_fitness_history) < patience + 1:
            return False
        
        recent_history = self.best_fitness_history[-patience-1:]
        improvement = recent_history[-1] - recent_history[0]
        
        return improvement < threshold


# 使用例とテスト
if __name__ == "__main__":
    print("=" * 70)
    print("遺伝的アルゴリズム - 集団クラステスト")
    print("=" * 70)
    
    from individual import Individual
    
    # 集団生成
    print("\n1. 初期集団生成")
    config = PopulationConfig(
        population_size=20,
        elite_size=2,
        crossover_rate=0.8,
        mutation_rate=0.1
    )
    population = Population(config)
    population.initialize()
    
    # ダミー適合度関数
    def dummy_fitness(individual: Individual) -> float:
        return random.random()
    
    # 評価
    print("\n2. 集団評価")
    population.evaluate(dummy_fitness)
    stats = population.get_statistics()
    print(f"世代: {stats['generation']}")
    print(f"最良適合度: {stats['best_fitness']:.4f}")
    print(f"平均適合度: {stats['avg_fitness']:.4f}")
    print(f"多様性: {stats['diversity']:.4f}")
    
    # 進化
    print("\n3. 5世代進化")
    for gen in range(5):
        population.evolve(selection_method="tournament")
        population.evaluate(dummy_fitness)
        stats = population.get_statistics()
        print(f"世代{stats['generation']}: "
              f"最良={stats['best_fitness']:.4f}, "
              f"平均={stats['avg_fitness']:.4f}")
    
    print("\n" + "=" * 70)