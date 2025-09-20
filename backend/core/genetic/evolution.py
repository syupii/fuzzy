# core/genetic/evolution.py - 遺伝的アルゴリズム実装（SyntaxError修正版）

import random
import math
import numpy as np
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
        # SyntaxError修正：f-stringの中のリスト内包表記を分離
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
    """遺伝的アルゴリズムエンジン"""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.population: List[Individual] = []
        self.generation = 0
        self.fitness_history: List[float] = []
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
        
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """交叉操作（一様交叉）"""
        if random.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1_chromosome = []
        child2_chromosome = []
        
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
        """突然変異操作"""
        mutated = individual.copy()
        
        for i in range(len(mutated.chromosome)):
            if random.random() < self.config.mutation_rate:
                # ガウシアン変異
                mutation_strength = 0.1
                mutated.chromosome[i] += random.gauss(0, mutation_strength)
                # 範囲制限
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
        survivors = population[:elite_count].copy()
        
        # 残りをルーレット選択で決定
        remaining_count = self.config.population_size - elite_count
        
        if remaining_count > 0:
            # 適応度の正規化（負の値を避ける）
            min_fitness = min(ind.fitness for ind in population)
            adjusted_fitnesses = [ind.fitness - min_fitness + 1e-6 for ind in population]
            total_fitness = sum(adjusted_fitnesses)
            
            if total_fitness > 0:
                probabilities = [f / total_fitness for f in adjusted_fitnesses]
                
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
                survivors.extend(random.choices(population, k=remaining_count))
        
        return survivors[:self.config.population_size]
    
    def evolve(self, fitness_function: Callable, verbose: bool = True) -> EvolutionResult:
        """進化プロセスの実行"""
        start_time = time.time()
        
        # 初期化
        self.initialize_population()
        self.generation = 0
        self.fitness_history = []
        self.stagnation_count = 0
        self.best_fitness = float('-inf')
        
        if verbose:
            print(f"🧬 遺伝的アルゴリズム開始: {self.config.generations}世代")
        
        for generation in range(self.config.generations):
            self.generation = generation
            
            # 適応度評価
            for individual in self.population:
                self.evaluate_fitness(individual, fitness_function)
            
            # 統計情報の更新
            current_best_fitness = max(ind.fitness for ind in self.population)
            avg_fitness = sum(ind.fitness for ind in self.population) / len(self.population)
            self.fitness_history.append(current_best_fitness)
            
            # 収束判定
            if abs(current_best_fitness - self.best_fitness) < self.config.convergence_threshold:
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0
                self.best_fitness = current_best_fitness
            
            if verbose and generation % 10 == 0:
                print(f"  世代 {generation}: 最高適応度={current_best_fitness:.4f}, 平均={avg_fitness:.4f}")
            
            # 早期終了条件
            if self.stagnation_count >= self.config.max_stagnation:
                if verbose:
                    print(f"📊 収束達成: 世代 {generation} (停滞: {self.stagnation_count})")
                break
            
            # 次世代の生成
            new_population = []
            
            while len(new_population) < self.config.population_size:
                # 親選択
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                # 交叉
                child1, child2 = self.crossover(parent1, parent2)
                
                # 突然変異
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # 集団サイズの調整
            new_population = new_population[:self.config.population_size]
            
            # 適応度評価
            for individual in new_population:
                self.evaluate_fitness(individual, fitness_function)
            
            # 生存者選択（エリート保存）
            combined_population = self.population + new_population
            self.population = self.select_survivors(combined_population)
            
            # 年齢の更新
            for individual in self.population:
                individual.age += 1
        
        # 最終結果の準備
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        best_individual = self.population[0]
        
        processing_time = time.time() - start_time
        convergence_achieved = self.stagnation_count >= self.config.max_stagnation
        
        if verbose:
            print(f"🎉 進化完了: {self.generation + 1}世代, 処理時間={processing_time:.2f}秒")
            print(f"📈 最高適応度: {best_individual.fitness:.4f}")
        
        return EvolutionResult(
            best_individual=best_individual,
            best_fitness=best_individual.fitness,
            generation=self.generation,
            population=self.population,
            fitness_history=self.fitness_history,
            convergence_achieved=convergence_achieved,
            processing_time=processing_time,
            parameters_used=self.config
        )
    
    def optimize_weights_for_student(self, student_profile: Dict[str, float], 
                                   lab_database: List[Dict[str, Any]]) -> EvolutionResult:
        """学生プロフィールに対する重み最適化"""
        
        def fitness_function(weights: List[float]) -> float:
            """適応度関数：学生の希望に合う研究室との適合度を最大化"""
            total_score = 0.0
            count = 0
            
            # 重みを辞書形式に変換
            criteria = [
                "research_intensity", "advisor_style", "team_work", "workload", "theory_practice",
                "research_field_match", "skill_development", "lab_atmosphere", "flexibility", 
                "publication_opportunity", "interdisciplinary", "communication_style", "innovation_risk"
            ]
            
            weight_dict = {criteria[i]: weights[i] for i in range(min(len(criteria), len(weights)))}
            
            for lab in lab_database:
                try:
                    # 各基準での類似度計算
                    criterion_scores = []
                    for criterion in criteria:
                        if criterion in student_profile and criterion in lab:
                            student_val = student_profile[criterion]
                            lab_val = lab.get(criterion, 5.5)
                            
                            # 類似度計算
                            diff = abs(student_val - lab_val)
                            if diff <= 1.0:
                                similarity = 1.0
                            elif diff <= 2.0:
                                similarity = 0.9
                            elif diff <= 3.0:
                                similarity = 0.7
                            else:
                                similarity = max(0.1, 0.7 - (diff - 3.0) * 0.1)
                            
                            # 重み適用
                            weight = weight_dict.get(criterion, 1.0)
                            weighted_score = similarity * weight
                            criterion_scores.append(weighted_score)
                    
                    if criterion_scores:
                        lab_score = sum(criterion_scores) / len(criterion_scores)
                        total_score += lab_score
                        count += 1
                        
                except Exception as e:
                    continue
            
            return total_score / max(count, 1)
        
        # 進化実行
        return self.evolve(fitness_function, verbose=True)

# ===== ユーティリティ関数 =====

def create_default_weights() -> Dict[str, float]:
    """デフォルト重みを作成"""
    return {
        "research_intensity": 1.3,
        "advisor_style": 1.2,
        "team_work": 1.1,
        "workload": 1.0,
        "theory_practice": 1.1,
        "research_field_match": 1.4,
        "skill_development": 1.1,
        "lab_atmosphere": 1.0,
        "flexibility": 0.9,
        "publication_opportunity": 1.2,
        "interdisciplinary": 0.8,
        "communication_style": 0.9,
        "innovation_risk": 1.0
    }

def weights_list_to_dict(weights_list: List[float]) -> Dict[str, float]:
    """重みリストを辞書に変換"""
    criteria = [
        "research_intensity", "advisor_style", "team_work", "workload", "theory_practice",
        "research_field_match", "skill_development", "lab_atmosphere", "flexibility", 
        "publication_opportunity", "interdisciplinary", "communication_style", "innovation_risk"
    ]
    
    return {criteria[i]: weights_list[i] for i in range(min(len(criteria), len(weights_list)))}

def weights_dict_to_list(weights_dict: Dict[str, float]) -> List[float]:
    """重み辞書をリストに変換"""
    criteria = [
        "research_intensity", "advisor_style", "team_work", "workload", "theory_practice",
        "research_field_match", "skill_development", "lab_atmosphere", "flexibility", 
        "publication_opportunity", "interdisciplinary", "communication_style", "innovation_risk"
    ]
    
    return [weights_dict.get(criterion, 1.0) for criterion in criteria]