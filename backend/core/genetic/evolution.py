# core/genetic/evolution.py - 遺伝的アルゴリズムエンジン

import numpy as np
import random
import math
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from copy import deepcopy
import logging
import time

logger = logging.getLogger(__name__)

@dataclass
class EvolutionConfig:
    """遺伝的アルゴリズム設定"""
    population_size: int = 50
    generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elitism_rate: float = 0.1
    selection_method: str = "tournament"  # tournament, roulette, rank
    tournament_size: int = 3
    random_seed: Optional[int] = None

class Individual:
    """個体クラス"""
    
    def __init__(self, chromosome: List[float], fitness: float = 0.0):
        self.chromosome = chromosome  # 遺伝子（評価基準の重み）
        self.fitness = fitness
        self.age = 0
        self.birth_generation = 0
        
    def copy(self) -> 'Individual':
        """個体のコピーを作成"""
        new_individual = Individual(self.chromosome.copy(), self.fitness)
        new_individual.age = self.age
        new_individual.birth_generation = self.birth_generation
        return new_individual
    
    def __str__(self):
        return f"Individual(fitness={self.fitness:.4f}, chromosome={[f'{x:.2f}' for x in self.chromosome[:5]]...})"

class Population:
    """集団クラス"""
    
    def __init__(self, individuals: List[Individual] = None):
        self.individuals = individuals or []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.worst_individual: Optional[Individual] = None
        self.average_fitness = 0.0
        self.diversity_score = 0.0
    
    def add_individual(self, individual: Individual):
        """個体を追加"""
        self.individuals.append(individual)
    
    def update_statistics(self):
        """統計情報を更新"""
        if not self.individuals:
            return
        
        fitnesses = [ind.fitness for ind in self.individuals]
        self.average_fitness = np.mean(fitnesses)
        
        best_idx = np.argmax(fitnesses)
        worst_idx = np.argmin(fitnesses)
        
        self.best_individual = self.individuals[best_idx]
        self.worst_individual = self.individuals[worst_idx]
        
        # 多様性スコア（個体間の平均距離）
        if len(self.individuals) > 1:
            distances = []
            for i in range(len(self.individuals)):
                for j in range(i + 1, len(self.individuals)):
                    dist = np.linalg.norm(
                        np.array(self.individuals[i].chromosome) - 
                        np.array(self.individuals[j].chromosome)
                    )
                    distances.append(dist)
            self.diversity_score = np.mean(distances)
    
    def sort_by_fitness(self, descending=True):
        """適応度順にソート"""
        self.individuals.sort(key=lambda x: x.fitness, reverse=descending)

class GeneticOperators:
    """遺伝的操作クラス"""
    
    @staticmethod
    def tournament_selection(population: Population, tournament_size: int) -> Individual:
        """トーナメント選択"""
        tournament = random.sample(population.individuals, min(tournament_size, len(population.individuals)))
        return max(tournament, key=lambda x: x.fitness)
    
    @staticmethod
    def roulette_wheel_selection(population: Population) -> Individual:
        """ルーレット選択"""
        total_fitness = sum(ind.fitness for ind in population.individuals)
        if total_fitness <= 0:
            return random.choice(population.individuals)
        
        spin = random.uniform(0, total_fitness)
        current = 0
        for individual in population.individuals:
            current += individual.fitness
            if current >= spin:
                return individual
        return population.individuals[-1]
    
    @staticmethod
    def uniform_crossover(parent1: Individual, parent2: Individual, crossover_rate: float = 0.5) -> Tuple[Individual, Individual]:
        """一様交叉"""
        if random.random() > crossover_rate:
            return parent1.copy(), parent2.copy()
        
        chromosome1 = []
        chromosome2 = []
        
        for i in range(len(parent1.chromosome)):
            if random.random() < 0.5:
                chromosome1.append(parent1.chromosome[i])
                chromosome2.append(parent2.chromosome[i])
            else:
                chromosome1.append(parent2.chromosome[i])
                chromosome2.append(parent1.chromosome[i])
        
        return Individual(chromosome1), Individual(chromosome2)
    
    @staticmethod
    def blend_crossover(parent1: Individual, parent2: Individual, alpha: float = 0.5) -> Tuple[Individual, Individual]:
        """ブレンド交叉（BLX-α）"""
        chromosome1 = []
        chromosome2 = []
        
        for i in range(len(parent1.chromosome)):
            x1, x2 = parent1.chromosome[i], parent2.chromosome[i]
            
            # ブレンド範囲の計算
            min_val = min(x1, x2)
            max_val = max(x1, x2)
            range_val = max_val - min_val
            
            lower = min_val - alpha * range_val
            upper = max_val + alpha * range_val
            
            # 制約内でクランプ
            lower = max(0.0, lower)
            upper = min(1.0, upper)
            
            chromosome1.append(random.uniform(lower, upper))
            chromosome2.append(random.uniform(lower, upper))
        
        return Individual(chromosome1), Individual(chromosome2)
    
    @staticmethod
    def gaussian_mutation(individual: Individual, mutation_rate: float, sigma: float = 0.1) -> Individual:
        """ガウス変異"""
        mutated_chromosome = []
        
        for gene in individual.chromosome:
            if random.random() < mutation_rate:
                # ガウス分布から変異値を生成
                mutation = random.gauss(0, sigma)
                new_gene = gene + mutation
                # [0, 1]の範囲にクランプ
                new_gene = max(0.0, min(1.0, new_gene))
                mutated_chromosome.append(new_gene)
            else:
                mutated_chromosome.append(gene)
        
        return Individual(mutated_chromosome)
    
    @staticmethod
    def polynomial_mutation(individual: Individual, mutation_rate: float, eta: float = 20.0) -> Individual:
        """多項式変異"""
        mutated_chromosome = []
        
        for gene in individual.chromosome:
            if random.random() < mutation_rate:
                # 多項式変異の適用
                u = random.random()
                if u <= 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                
                new_gene = gene + delta
                new_gene = max(0.0, min(1.0, new_gene))
                mutated_chromosome.append(new_gene)
            else:
                mutated_chromosome.append(gene)
        
        return Individual(mutated_chromosome)

class FitnessEvaluator:
    """適応度評価クラス"""
    
    def __init__(self, student_profiles: List[Dict[str, float]], 
                 lab_profiles: List[Dict[str, float]], 
                 fuzzy_engine):
        self.student_profiles = student_profiles
        self.lab_profiles = lab_profiles
        self.fuzzy_engine = fuzzy_engine
        self.evaluation_count = 0
    
    def evaluate_individual(self, individual: Individual) -> float:
        """個体の適応度を評価"""
        self.evaluation_count += 1
        
        # 個体の染色体を評価基準の重みとして使用
        weights = individual.chromosome
        
        total_satisfaction = 0.0
        num_evaluations = 0
        
        # すべての学生-研究室ペアについて評価
        for student in self.student_profiles:
            best_match_score = 0.0
            
            for lab in self.lab_profiles:
                # 重み付き適合度を計算
                weighted_score = self.calculate_weighted_compatibility(
                    student, lab, weights
                )
                best_match_score = max(best_match_score, weighted_score)
            
            total_satisfaction += best_match_score
            num_evaluations += 1
        
        # 平均満足度を適応度とする
        fitness = total_satisfaction / num_evaluations if num_evaluations > 0 else 0.0
        
        # 多様性ボーナス（重みの均等性を評価）
        diversity_penalty = np.std(weights) * 0.1  # 分散が大きいほどペナルティ
        fitness = fitness - diversity_penalty
        
        return max(0.0, fitness)
    
    def calculate_weighted_compatibility(self, student: Dict[str, float], 
                                       lab: Dict[str, float], 
                                       weights: List[float]) -> float:
        """重み付き適合度計算"""
        
        criteria = [
            "research_intensity", "advisor_style", "team_work", "workload",
            "theory_practice", "research_field_match", "skill_development",
            "lab_atmosphere", "flexibility", "publication_opportunity",
            "interdisciplinary", "communication_style", "innovation_risk"
        ]
        
        if len(weights) != len(criteria):
            weights = weights[:len(criteria)] + [0.5] * (len(criteria) - len(weights))
        
        # 基本適合度（重み付き）
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for i, criterion in enumerate(criteria):
            if criterion in student and criterion in lab:
                student_val = student[criterion]
                lab_val = lab[criterion]
                
                # 類似度計算
                similarity = 1.0 - abs(student_val - lab_val) / 9.0
                
                # 重み適用
                weight = weights[i] if i < len(weights) else 0.5
                weighted_sum += similarity * weight
                weight_sum += weight
        
        base_compatibility = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        
        # ファジィ推論による調整
        try:
            fuzzy_adjustment = self.fuzzy_engine.infer_lab_compatibility(student, lab)
            final_score = 0.7 * base_compatibility + 0.3 * fuzzy_adjustment
        except:
            final_score = base_compatibility
        
        return final_score

class EvolutionEngine:
    """進化エンジン"""
    
    def __init__(self, config: EvolutionConfig, fuzzy_engine=None):
        self.config = config
        self.fuzzy_engine = fuzzy_engine
        self.population = Population()
        self.generation = 0
        self.operators = GeneticOperators()
        
        # 進化履歴
        self.evolution_history = []
        self.convergence_threshold = 1e-6
        self.stagnation_count = 0
        self.max_stagnation = 20
        
        # 乱数シード設定
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
    
    def initialize_population(self, chromosome_length: int) -> Population:
        """初期集団を生成"""
        individuals = []
        
        for _ in range(self.config.population_size):
            # ランダムな重みを生成（0-1の範囲）
            chromosome = [random.random() for _ in range(chromosome_length)]
            
            # 重みの正規化
            weight_sum = sum(chromosome)
            if weight_sum > 0:
                chromosome = [w / weight_sum for w in chromosome]
            
            individual = Individual(chromosome)
            individual.birth_generation = 0
            individuals.append(individual)
        
        self.population = Population(individuals)
        logger.info(f"初期集団を生成しました: {self.config.population_size}個体")
        return self.population
    
    def evolve(self, fitness_evaluator: FitnessEvaluator, max_generations: Optional[int] = None) -> Population:
        """進化実行"""
        
        max_gen = max_generations or self.config.generations
        start_time = time.time()
        
        logger.info(f"進化開始: {max_gen}世代, 集団サイズ{self.config.population_size}")
        
        # 初期集団の評価
        self.evaluate_population(fitness_evaluator)
        
        for generation in range(max_gen):
            self.generation = generation
            
            # 新世代の生成
            new_population = self.create_next_generation()
            
            # 評価
            self.evaluate_population_with_evaluator(new_population, fitness_evaluator)
            
            # エリート保存
            self.apply_elitism(new_population)
            
            # 統計更新
            self.population = new_population
            self.population.update_statistics()
            
            # 進化履歴記録
            self.record_generation_statistics()
            
            # 収束判定
            if self.check_convergence():
                logger.info(f"収束により進化終了: 世代 {generation}")
                break
            
            # 進捗報告
            if generation % 10 == 0:
                logger.info(f"世代 {generation}: 最良適応度 {self.population.best_individual.fitness:.4f}")
        
        execution_time = time.time() - start_time
        logger.info(f"進化完了: 実行時間 {execution_time:.2f}秒")
        
        return self.population
    
    def create_next_generation(self) -> Population:
        """次世代を生成"""
        new_individuals = []
        
        # 選択・交叉・変異
        while len(new_individuals) < self.config.population_size:
            # 親選択
            if self.config.selection_method == "tournament":
                parent1 = self.operators.tournament_selection(self.population, self.config.tournament_size)
                parent2 = self.operators.tournament_selection(self.population, self.config.tournament_size)
            else:
                parent1 = self.operators.roulette_wheel_selection(self.population)
                parent2 = self.operators.roulette_wheel_selection(self.population)
            
            # 交叉
            child1, child2 = self.operators.blend_crossover(parent1, parent2, alpha=0.3)
            
            # 変異
            child1 = self.operators.gaussian_mutation(child1, self.config.mutation_rate, sigma=0.05)
            child2 = self.operators.gaussian_mutation(child2, self.config.mutation_rate, sigma=0.05)
            
            child1.birth_generation = self.generation + 1
            child2.birth_generation = self.generation + 1
            
            new_individuals.extend([child1, child2])
        
        # 集団サイズに調整
        new_individuals = new_individuals[:self.config.population_size]
        
        return Population(new_individuals)
    
    def evaluate_population(self, fitness_evaluator: FitnessEvaluator):
        """集団の適応度評価"""
        for individual in self.population.individuals:
            if individual.fitness == 0.0:  # 未評価の個体のみ
                individual.fitness = fitness_evaluator.evaluate_individual(individual)
    
    def evaluate_population_with_evaluator(self, population: Population, fitness_evaluator: FitnessEvaluator):
        """指定した集団の適応度評価"""
        for individual in population.individuals:
            individual.fitness = fitness_evaluator.evaluate_individual(individual)
    
    def apply_elitism(self, new_population: Population):
        """エリート保存"""
        if self.config.elitism_rate <= 0:
            return
        
        num_elites = max(1, int(self.config.population_size * self.config.elitism_rate))
        
        # 現世代のエリートを取得
        self.population.sort_by_fitness(descending=True)
        elites = self.population.individuals[:num_elites]
        
        # 新世代の下位個体をエリートに置き換え
        new_population.sort_by_fitness(descending=True)
        new_population.individuals[-num_elites:] = [elite.copy() for elite in elites]
    
    def record_generation_statistics(self):
        """世代統計を記録"""
        stats = {
            "generation": self.generation,
            "best_fitness": self.population.best_individual.fitness,
            "worst_fitness": self.population.worst_individual.fitness,
            "average_fitness": self.population.average_fitness,
            "diversity": self.population.diversity_score
        }
        self.evolution_history.append(stats)
    
    def check_convergence(self) -> bool:
        """収束判定"""
        if len(self.evolution_history) < 5:
            return False
        
        # 最近5世代の最良適応度の変化を確認
        recent_best = [h["best_fitness"] for h in self.evolution_history[-5:]]
        fitness_change = max(recent_best) - min(recent_best)
        
        if fitness_change < self.convergence_threshold:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
        
        return self.stagnation_count >= self.max_stagnation
    
    def get_best_weights(self) -> List[float]:
        """最良個体の重みを取得"""
        if self.population.best_individual:
            return self.population.best_individual.chromosome.copy()
        return [1.0/13] * 13  # 均等重み
    
    def get_evolution_summary(self) -> Dict[str, Any]:
        """進化結果のサマリーを取得"""
        return {
            "generations_completed": self.generation,
            "best_fitness": self.population.best_individual.fitness if self.population.best_individual else 0.0,
            "convergence_achieved": self.stagnation_count >= self.max_stagnation,
            "final_diversity": self.population.diversity_score,
            "evolution_history": self.evolution_history,
            "best_weights": self.get_best_weights()
        }