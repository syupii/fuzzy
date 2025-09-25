# core/genetic/evolution.py - 遺伝的アルゴリズム完全実装版

import random
import math
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import copy

# ===== データクラス定義 =====

@dataclass
class EvolutionConfig:
    """遺伝的アルゴリズム設定"""
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism_rate: float = 0.1
    tournament_size: int = 3
    convergence_threshold: float = 1e-6
    max_stagnation: int = 20

@dataclass
class Individual:
    """個体クラス"""
    chromosome: List[float] = field(default_factory=list)
    fitness: float = 0.0
    age: int = 0
    
    def __post_init__(self):
        if not self.chromosome:
            # デフォルトの染色体（13項目の重み）
            self.chromosome = [random.uniform(0.5, 2.0) for _ in range(13)]
    
    def __str__(self):
        chromosome_preview = [f'{x:.2f}' for x in self.chromosome[:5]]
        return f"Individual(fitness={self.fitness:.4f}, chromosome={chromosome_preview}...)"
    
    def copy(self):
        """個体のコピーを作成"""
        return Individual(
            chromosome=self.chromosome.copy(),
            fitness=self.fitness,
            age=self.age
        )

@dataclass
class EvolutionResult:
    """進化結果"""
    best_individual: Individual
    best_fitness: float
    generation: int
    population: List[Individual]
    fitness_history: List[float]
    convergence_achieved: bool
    processing_time: float
    parameters_used: EvolutionConfig

# ===== 遺伝的アルゴリズムエンジン =====

class EvolutionEngine:
    """遺伝的アルゴリズムエンジン（完全版）"""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.population: List[Individual] = []
        self.generation = 0
        self.fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
        self.stagnation_count = 0
        self.best_fitness = float('-inf')
        
    def initialize_population(self) -> None:
        """初期集団を生成"""
        self.population = []
        for _ in range(self.config.population_size):
            individual = Individual()
            # 染色体の正規化（重みが適切な範囲になるように）
            total = sum(individual.chromosome)
            individual.chromosome = [x / total * 13.0 for x in individual.chromosome]
            self.population.append(individual)
        
        print(f"✅ 初期集団生成完了: {len(self.population)}個体")
    
    def evaluate_fitness(self, individual: Individual, fitness_function: Callable) -> float:
        """個体の適応度を評価"""
        try:
            fitness = fitness_function(individual.chromosome)
            individual.fitness = fitness
            return fitness
        except Exception as e:
            print(f"⚠️ 適応度評価エラー: {e}")
            individual.fitness = 0.0
            return 0.0
    
    def tournament_selection(self, tournament_size: int = None) -> Individual:
        """トーナメント選択"""
        if tournament_size is None:
            tournament_size = self.config.tournament_size
        
        # ランダムにトーナメント参加者を選択
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        # 最も適応度の高い個体を返す
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """交叉操作（一様交叉）"""
        if random.random() > self.config.crossover_rate:
            # 交叉しない場合は親をそのままコピー
            return parent1.copy(), parent2.copy()
        
        child1_chromosome = []
        child2_chromosome = []
        
        # 遺伝子ごとにランダムに親から継承
        for i in range(len(parent1.chromosome)):
            if random.random() < 0.5:
                child1_chromosome.append(parent1.chromosome[i])
                child2_chromosome.append(parent2.chromosome[i])
            else:
                child1_chromosome.append(parent2.chromosome[i])
                child2_chromosome.append(parent1.chromosome[i])
        
        child1 = Individual(chromosome=child1_chromosome)
        child2 = Individual(chromosome=child2_chromosome)
        
        return child1, child2
    
    def mutate(self, individual: Individual) -> Individual:
        """突然変異操作（ガウシアン変異）"""
        mutated = individual.copy()
        
        for i in range(len(mutated.chromosome)):
            if random.random() < self.config.mutation_rate:
                # ガウシアン変異: 正規分布に従うランダムな値を加算
                mutation_strength = 0.1
                mutated.chromosome[i] += random.gauss(0, mutation_strength)
                # 範囲制限（0.1〜3.0の範囲に収める）
                mutated.chromosome[i] = max(0.1, min(3.0, mutated.chromosome[i]))
        
        # 染色体の正規化
        total = sum(mutated.chromosome)
        if total > 0:
            mutated.chromosome = [x / total * 13.0 for x in mutated.chromosome]
        
        return mutated
    
    def select_survivors(self, population: List[Individual]) -> List[Individual]:
        """生存者選択（エリート保存 + ルーレット選択）"""
        
        # エリート個体の保存
        elite_count = max(1, int(self.config.population_size * self.config.elitism_rate))
        population.sort(key=lambda x: x.fitness, reverse=True)
        survivors = [ind.copy() for ind in population[:elite_count]]
        
        # 残りをルーレット選択で決定
        remaining_count = self.config.population_size - elite_count
        
        if remaining_count > 0:
            # 適応度の正規化（負の値を避ける）
            min_fitness = min(ind.fitness for ind in population)
            adjusted_fitnesses = [ind.fitness - min_fitness + 1e-6 for ind in population]
            total_fitness = sum(adjusted_fitnesses)
            
            if total_fitness > 0:
                probabilities = [f / total_fitness for f in adjusted_fitnesses]
                
                # ルーレット選択
                for _ in range(remaining_count):
                    r = random.random()
                    cumulative = 0.0
                    for i, prob in enumerate(probabilities):
                        cumulative += prob
                        if r <= cumulative:
                            survivors.append(population[i].copy())
                            break
            else:
                # フォールバック：ランダム選択
                survivors.extend([ind.copy() for ind in random.choices(population, k=remaining_count)])
        
        return survivors[:self.config.population_size]
    
    def evolve_generation(self) -> None:
        """1世代の進化プロセス"""
        
        # 新しい子個体を生成
        offspring = []
        
        # エリート個体を保存
        elite_count = max(1, int(self.config.population_size * self.config.elitism_rate))
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        elites = [ind.copy() for ind in self.population[:elite_count]]
        
        # 残りの個体数を生成
        offspring_count = self.config.population_size - elite_count
        
        while len(offspring) < offspring_count:
            # 親の選択（トーナメント選択）
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            # 交叉
            child1, child2 = self.crossover(parent1, parent2)
            
            # 変異
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            offspring.append(child1)
            if len(offspring) < offspring_count:
                offspring.append(child2)
        
        # 新世代を作成（エリート + 子個体）
        self.population = elites + offspring[:offspring_count]
        
        # 年齢を更新
        for ind in self.population:
            ind.age += 1
    
    def calculate_statistics(self) -> Tuple[float, float, float, float]:
        """集団の統計情報を計算"""
        fitnesses = [ind.fitness for ind in self.population]
        
        best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        min_fitness = min(fitnesses)
        std_fitness = np.std(fitnesses) if len(fitnesses) > 1 else 0.0
        
        return best_fitness, avg_fitness, min_fitness, std_fitness
    
    def evolve(self, fitness_function: Callable, verbose: bool = True) -> EvolutionResult:
        """進化プロセスの実行（完全版）"""
        start_time = time.time()
        
        # 初期化
        if not self.population:
            self.initialize_population()
        
        self.generation = 0
        self.fitness_history = []
        self.avg_fitness_history = []
        self.stagnation_count = 0
        self.best_fitness = float('-inf')
        
        if verbose:
            print(f"\n🧬 遺伝的アルゴリズム開始")
            print(f"  集団サイズ: {self.config.population_size}")
            print(f"  最大世代数: {self.config.generations}")
            print(f"  交叉率: {self.config.crossover_rate}")
            print(f"  変異率: {self.config.mutation_rate}")
        
        # 進化ループ
        for generation in range(self.config.generations):
            self.generation = generation
            
            # 全個体の適応度を評価
            for individual in self.population:
                self.evaluate_fitness(individual, fitness_function)
            
            # 統計情報の計算
            best_fitness, avg_fitness, min_fitness, std_fitness = self.calculate_statistics()
            
            # 履歴に記録
            self.fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            
            # 進捗表示
            if verbose and generation % 10 == 0:
                print(f"  世代 {generation:3d}: "
                      f"最良={best_fitness:.6f}, "
                      f"平均={avg_fitness:.6f}, "
                      f"標準偏差={std_fitness:.6f}")
            
            # 収束判定
            if generation > 0:
                improvement = best_fitness - self.best_fitness
                if abs(improvement) < self.config.convergence_threshold:
                    self.stagnation_count += 1
                    if self.stagnation_count >= self.config.max_stagnation:
                        if verbose:
                            print(f"\n✅ 収束検出（世代{generation}）")
                            print(f"  改善が{self.config.max_stagnation}世代間で"
                                  f"{self.config.convergence_threshold}以下")
                        break
                else:
                    self.stagnation_count = 0
                    self.best_fitness = best_fitness
            else:
                self.best_fitness = best_fitness
            
            # 次世代の生成
            self.evolve_generation()
        
        processing_time = time.time() - start_time
        
        # 最終的な最良個体を取得
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        best_individual = self.population[0]
        
        if verbose:
            print(f"\n✅ 遺伝的アルゴリズム完了")
            print(f"  最終世代: {self.generation + 1}")
            print(f"  最良適応度: {best_individual.fitness:.6f}")
            print(f"  処理時間: {processing_time:.2f}秒")
            print(f"  収束判定: {'達成' if self.stagnation_count >= self.config.max_stagnation else '未達成'}")
        
        # 結果を返す
        return EvolutionResult(
            best_individual=best_individual,
            best_fitness=best_individual.fitness,
            generation=self.generation + 1,
            population=self.population,
            fitness_history=self.fitness_history,
            convergence_achieved=self.stagnation_count >= self.config.max_stagnation,
            processing_time=processing_time,
            parameters_used=self.config
        )

# ===== ヘルパー関数 =====

def create_evolution_config(
    population_size: int = 50,
    generations: int = 100,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.8,
    elitism_rate: float = 0.1,
    tournament_size: int = 3,
    convergence_threshold: float = 1e-6,
    max_stagnation: int = 20
) -> EvolutionConfig:
    """進化設定の作成"""
    return EvolutionConfig(
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        elitism_rate=elitism_rate,
        tournament_size=tournament_size,
        convergence_threshold=convergence_threshold,
        max_stagnation=max_stagnation
    )

# ===== テスト関数 =====

def test_evolution_engine():
    """進化エンジンのテスト"""
    
    print("=" * 60)
    print("🧬 遺伝的アルゴリズムエンジンテスト")
    print("=" * 60)
    
    # テスト用の適応度関数（簡単な最適化問題）
    def test_fitness_function(chromosome: List[float]) -> float:
        """テスト適応度関数: 全要素の二乗和の逆数を最大化"""
        # 目標: 全ての要素を1.0に近づける
        target = 1.0
        total_error = sum((x - target) ** 2 for x in chromosome)
        fitness = 1.0 / (total_error + 1e-6)
        return fitness
    
    # 設定作成
    config = create_evolution_config(
        population_size=20,
        generations=50,
        mutation_rate=0.15,
        crossover_rate=0.8,
        elitism_rate=0.1,
        max_stagnation=10
    )
    
    print(f"\n📋 設定:")
    print(f"  集団サイズ: {config.population_size}")
    print(f"  世代数: {config.generations}")
    print(f"  変異率: {config.mutation_rate}")
    print(f"  交叉率: {config.crossover_rate}")
    
    # エンジン作成と実行
    engine = EvolutionEngine(config)
    result = engine.evolve(test_fitness_function, verbose=True)
    
    # 結果表示
    print(f"\n📊 最終結果:")
    print(f"  最良適応度: {result.best_fitness:.6f}")
    print(f"  世代数: {result.generation}")
    print(f"  処理時間: {result.processing_time:.2f}秒")
    print(f"  収束: {'✅ 達成' if result.convergence_achieved else '❌ 未達成'}")
    print(f"  最良個体の染色体（先頭5要素）: {[f'{x:.3f}' for x in result.best_individual.chromosome[:5]]}")
    
    # 適応度の推移をグラフ的に表示
    print(f"\n📈 適応度推移（簡易版）:")
    history_sample = result.fitness_history[::max(1, len(result.fitness_history) // 10)]
    for i, fitness in enumerate(history_sample):
        bar_length = int(fitness / max(history_sample) * 40)
        print(f"  {'█' * bar_length} {fitness:.4f}")
    
    print("\n✅ テスト完了")
    print("=" * 60)

if __name__ == "__main__":
    # テスト実行
    test_evolution_engine()