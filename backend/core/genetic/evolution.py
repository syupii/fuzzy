# backend/core/genetic/evolution.py - 遺伝的アルゴリズム実装
# 研究室選択最適化用

import random
import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import copy

@dataclass
class EvolutionConfig:
    """進化アルゴリズム設定"""
    population_size: int = 30
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    tournament_size: int = 3
    convergence_threshold: float = 0.001
    max_stagnation: int = 10

@dataclass
class PopulationConfig:
    """集団設定"""
    chromosome_length: int = 13  # 13項目の評価基準
    gene_min: float = 1.0
    gene_max: float = 10.0
    initialization_method: str = "random"  # "random", "uniform", "normal"

class Individual:
    """個体クラス"""
    
    def __init__(self, chromosome: List[float] = None, chromosome_length: int = 13):
        if chromosome is None:
            self.chromosome = [random.uniform(1.0, 10.0) for _ in range(chromosome_length)]
        else:
            self.chromosome = chromosome.copy()
        
        self.fitness: float = 0.0
        self.rank: int = 0
        self.age: int = 0
        self.metadata: Dict[str, Any] = {}
    
    def __len__(self):
        return len(self.chromosome)
    
    def __getitem__(self, index):
        return self.chromosome[index]
    
    def __setitem__(self, index, value):
        self.chromosome[index] = value
    
    def copy(self) -> 'Individual':
        """個体のコピーを作成"""
        new_individual = Individual(self.chromosome)
        new_individual.fitness = self.fitness
        new_individual.rank = self.rank
        new_individual.age = self.age
        new_individual.metadata = self.metadata.copy()
        return new_individual
    
    def mutate(self, mutation_rate: float, gene_min: float = 1.0, gene_max: float = 10.0):
        """突然変異"""
        for i in range(len(self.chromosome)):
            if random.random() < mutation_rate:
                # ガウシアン突然変異
                mutation_strength = 0.5
                new_value = self.chromosome[i] + random.gauss(0, mutation_strength)
                self.chromosome[i] = max(gene_min, min(gene_max, new_value))
    
    def to_dict(self) -> Dict[str, float]:
        """辞書形式に変換"""
        criteria_names = [
            "research_intensity", "advisor_style", "team_work", "workload", "theory_practice",
            "research_field_match", "skill_development", "lab_atmosphere", "flexibility", 
            "publication_opportunity", "interdisciplinary", "communication_style", "innovation_risk"
        ]
        
        return {name: value for name, value in zip(criteria_names[:len(self.chromosome)], self.chromosome)}

class FitnessEvaluator(ABC):
    """適応度評価の抽象基底クラス"""
    
    @abstractmethod
    def evaluate(self, individual: Individual, context: Dict[str, Any]) -> float:
        """適応度を評価"""
        pass

class LabMatchingFitnessEvaluator(FitnessEvaluator):
    """研究室マッチング用適応度評価"""
    
    def __init__(self, lab_data: List[Dict[str, Any]], target_preferences: Dict[str, float] = None):
        self.lab_data = lab_data
        self.target_preferences = target_preferences or {}
        
        # 評価基準の重み
        self.weights = {
            "research_intensity": 1.2,
            "advisor_style": 1.1,
            "team_work": 1.0,
            "workload": 0.9,
            "theory_practice": 1.0,
            "research_field_match": 1.3,
            "skill_development": 1.1,
            "lab_atmosphere": 0.8,
            "flexibility": 0.9,
            "publication_opportunity": 1.2,
            "interdisciplinary": 0.8,
            "communication_style": 0.7,
            "innovation_risk": 0.9
        }
    
    def evaluate(self, individual: Individual, context: Dict[str, Any] = None) -> float:
        """適応度評価（研究室との適合性）"""
        
        if not self.lab_data:
            return 0.0
        
        individual_preferences = individual.to_dict()
        
        # 最適研究室との適合性を計算
        max_compatibility = 0.0
        total_compatibility = 0.0
        
        for lab in self.lab_data:
            lab_features = lab.get("features", {})
            if not lab_features:
                continue
            
            compatibility = self._calculate_compatibility(individual_preferences, lab_features)
            max_compatibility = max(max_compatibility, compatibility)
            total_compatibility += compatibility
        
        # 平均適合性も考慮
        avg_compatibility = total_compatibility / len(self.lab_data) if self.lab_data else 0.0
        
        # 目標設定との一致度
        target_match = 0.0
        if self.target_preferences:
            target_match = self._calculate_compatibility(individual_preferences, self.target_preferences)
        
        # 多様性ボーナス（遺伝子の分散）
        diversity_bonus = self._calculate_diversity_bonus(individual)
        
        # 総合適応度
        fitness = (
            max_compatibility * 0.4 +      # 最高適合性
            avg_compatibility * 0.3 +      # 平均適合性
            target_match * 0.2 +           # 目標一致度
            diversity_bonus * 0.1          # 多様性ボーナス
        )
        
        return min(1.0, fitness)
    
    def _calculate_compatibility(self, preferences: Dict[str, float], features: Dict[str, float]) -> float:
        """適合性計算"""
        total_score = 0.0
        total_weight = 0.0
        
        for criterion in preferences:
            if criterion in features:
                pref_val = preferences[criterion]
                feat_val = features[criterion]
                
                # 正規化（1-10 → 0-1）
                pref_norm = (pref_val - 1) / 9
                feat_norm = (feat_val - 1) / 9
                
                # 類似度計算
                similarity = 1.0 - abs(pref_norm - feat_norm)
                
                # 重み付け
                weight = self.weights.get(criterion, 1.0)
                total_score += similarity * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_diversity_bonus(self, individual: Individual) -> float:
        """多様性ボーナス計算"""
        if len(individual.chromosome) < 2:
            return 0.0
        
        # 遺伝子値の分散を計算
        mean_val = sum(individual.chromosome) / len(individual.chromosome)
        variance = sum((x - mean_val) ** 2 for x in individual.chromosome) / len(individual.chromosome)
        
        # 正規化された分散をボーナスとして使用
        max_variance = (10 - 1) ** 2 / 4  # 理論的最大分散
        return min(1.0, variance / max_variance)

class GeneticOperators:
    """遺伝的操作クラス"""
    
    @staticmethod
    def tournament_selection(population: List[Individual], tournament_size: int = 3) -> Individual:
        """トーナメント選択"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness)
    
    @staticmethod
    def roulette_selection(population: List[Individual]) -> Individual:
        """ルーレット選択"""
        if not population:
            raise ValueError("Population is empty")
        
        # 適応度の総和
        total_fitness = sum(ind.fitness for ind in population)
        
        if total_fitness == 0:
            return random.choice(population)
        
        # ルーレット回転
        spin = random.uniform(0, total_fitness)
        current = 0
        
        for individual in population:
            current += individual.fitness
            if current >= spin:
                return individual
        
        return population[-1]  # フォールバック
    
    @staticmethod
    def uniform_crossover(parent1: Individual, parent2: Individual, crossover_rate: float = 0.5) -> Tuple[Individual, Individual]:
        """一様交叉"""
        if random.random() > crossover_rate:
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
        
        child1 = Individual(child1_chromosome)
        child2 = Individual(child2_chromosome)
        
        return child1, child2
    
    @staticmethod
    def arithmetic_crossover(parent1: Individual, parent2: Individual, alpha: float = 0.5) -> Tuple[Individual, Individual]:
        """算術交叉"""
        child1_chromosome = []
        child2_chromosome = []
        
        for i in range(len(parent1.chromosome)):
            gene1 = alpha * parent1.chromosome[i] + (1 - alpha) * parent2.chromosome[i]
            gene2 = (1 - alpha) * parent1.chromosome[i] + alpha * parent2.chromosome[i]
            
            child1_chromosome.append(gene1)
            child2_chromosome.append(gene2)
        
        child1 = Individual(child1_chromosome)
        child2 = Individual(child2_chromosome)
        
        return child1, child2
    
    @staticmethod
    def adaptive_mutation(individual: Individual, generation: int, max_generations: int, 
                         initial_rate: float = 0.1, final_rate: float = 0.01):
        """適応的突然変異"""
        # 世代に応じて突然変異率を調整
        progress = generation / max_generations
        current_rate = initial_rate * (1 - progress) + final_rate * progress
        
        individual.mutate(current_rate)

class Population:
    """集団クラス"""
    
    def __init__(self, config: PopulationConfig, initial_population: List[Individual] = None):
        self.config = config
        
        if initial_population:
            self.individuals = initial_population
        else:
            self.individuals = self._initialize_population()
        
        self.generation = 0
        self.best_fitness_history: List[float] = []
        self.avg_fitness_history: List[float] = []
    
    def _initialize_population(self) -> List[Individual]:
        """初期集団の生成"""
        population = []
        
        for _ in range(self.config.chromosome_length):
            if self.config.initialization_method == "random":
                individual = Individual(chromosome_length=self.config.chromosome_length)
            elif self.config.initialization_method == "uniform":
                chromosome = [5.5] * self.config.chromosome_length  # 中央値
                individual = Individual(chromosome)
                individual.mutate(0.3)  # 多様性のため軽い突然変異
            elif self.config.initialization_method == "normal":
                chromosome = [random.gauss(5.5, 1.5) for _ in range(self.config.chromosome_length)]
                chromosome = [max(1.0, min(10.0, x)) for x in chromosome]  # 範囲制限
                individual = Individual(chromosome)
            else:
                individual = Individual(chromosome_length=self.config.chromosome_length)
            
            population.append(individual)
        
        return population
    
    def evaluate_fitness(self, evaluator: FitnessEvaluator, context: Dict[str, Any] = None):
        """集団の適応度評価"""
        for individual in self.individuals:
            individual.fitness = evaluator.evaluate(individual, context)
        
        # ソート（適応度降順）
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        
        # ランク付け
        for i, individual in enumerate(self.individuals):
            individual.rank = i + 1
    
    def get_statistics(self) -> Dict[str, float]:
        """統計情報取得"""
        fitnesses = [ind.fitness for ind in self.individuals]
        
        return {
            "best_fitness": max(fitnesses) if fitnesses else 0.0,
            "worst_fitness": min(fitnesses) if fitnesses else 0.0,
            "average_fitness": sum(fitnesses) / len(fitnesses) if fitnesses else 0.0,
            "fitness_std": np.std(fitnesses) if fitnesses else 0.0,
            "diversity": self._calculate_diversity()
        }
    
    def _calculate_diversity(self) -> float:
        """集団の多様性計算"""
        if len(self.individuals) < 2:
            return 0.0
        
        total_distance = 0.0
        count = 0
        
        for i in range(len(self.individuals)):
            for j in range(i + 1, len(self.individuals)):
                distance = self._euclidean_distance(
                    self.individuals[i].chromosome,
                    self.individuals[j].chromosome
                )
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def _euclidean_distance(self, chromosome1: List[float], chromosome2: List[float]) -> float:
        """ユークリッド距離計算"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(chromosome1, chromosome2)))

class EvolutionEngine:
    """進化エンジン"""
    
    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.population: Optional[Population] = None
        self.evaluator: Optional[FitnessEvaluator] = None
        self.evolution_history: List[Dict[str, Any]] = []
    
    def initialize(self, population_config: PopulationConfig, evaluator: FitnessEvaluator):
        """初期化"""
        self.population = Population(population_config)
        self.evaluator = evaluator
    
    def evolve(self, lab_data: List[Dict[str, Any]], target_preferences: Dict[str, float] = None) -> Dict[str, Any]:
        """進化実行"""
        
        if not self.population or not self.evaluator:
            # 自動初期化
            pop_config = PopulationConfig(chromosome_length=13)
            evaluator = LabMatchingFitnessEvaluator(lab_data, target_preferences)
            self.initialize(pop_config, evaluator)
        
        self.evolution_history = []
        stagnation_count = 0
        best_fitness = 0.0
        
        for generation in range(self.config.generations):
            # 適応度評価
            self.population.evaluate_fitness(self.evaluator)
            
            # 統計情報
            stats = self.population.get_statistics()
            self.evolution_history.append({
                "generation": generation,
                **stats
            })
            
            # 収束判定
            if abs(stats["best_fitness"] - best_fitness) < self.config.convergence_threshold:
                stagnation_count += 1
            else:
                stagnation_count = 0
                best_fitness = stats["best_fitness"]
            
            if stagnation_count >= self.config.max_stagnation:
                print(f"🔄 収束により進化を終了: 世代 {generation}")
                break
            
            # 次世代生成
            if generation < self.config.generations - 1:
                self._create_next_generation(generation)
        
        # 最終評価
        self.population.evaluate_fitness(self.evaluator)
        best_individual = self.population.individuals[0]
        
        return {
            "best_individual": best_individual.to_dict(),
            "best_fitness": best_individual.fitness,
            "generations_completed": len(self.evolution_history),
            "evolution_history": self.evolution_history,
            "final_population_size": len(self.population.individuals),
            "convergence_achieved": stagnation_count >= self.config.max_stagnation
        }
    
    def _create_next_generation(self, generation: int):
        """次世代の生成"""
        new_individuals = []
        
        # エリート保存
        elite_count = min(self.config.elite_size, len(self.population.individuals))
        elites = self.population.individuals[:elite_count]
        new_individuals.extend([elite.copy() for elite in elites])
        
        # 交叉と突然変異による新個体生成
        while len(new_individuals) < self.config.population_size:
            # 親選択
            parent1 = GeneticOperators.tournament_selection(
                self.population.individuals, self.config.tournament_size
            )
            parent2 = GeneticOperators.tournament_selection(
                self.population.individuals, self.config.tournament_size
            )
            
            # 交叉
            if random.random() < self.config.crossover_rate:
                child1, child2 = GeneticOperators.arithmetic_crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # 適応的突然変異
            GeneticOperators.adaptive_mutation(
                child1, generation, self.config.generations,
                self.config.mutation_rate, self.config.mutation_rate * 0.1
            )
            GeneticOperators.adaptive_mutation(
                child2, generation, self.config.generations,
                self.config.mutation_rate, self.config.mutation_rate * 0.1
            )
            
            new_individuals.extend([child1, child2])
        
        # 集団サイズ調整
        self.population.individuals = new_individuals[:self.config.population_size]
        self.population.generation += 1

# テスト用のメイン関数
if __name__ == "__main__":
    print("🧪 遺伝的アルゴリズムテスト開始...")
    
    # テスト用の研究室データ
    test_labs = [
        {
            "id": "lab_1",
            "name": "AI研究室",
            "features": {
                "research_intensity": 8.5,
                "advisor_style": 7.0,
                "team_work": 8.0,
                "workload": 8.0,
                "theory_practice": 6.5,
                "research_field_match": 9.0,
                "skill_development": 8.0,
                "lab_atmosphere": 7.5,
                "flexibility": 7.0,
                "publication_opportunity": 8.0,
                "interdisciplinary": 7.0,
                "communication_style": 7.0,
                "innovation_risk": 7.5
            }
        },
        {
            "id": "lab_2",
            "name": "デザイン研究室",
            "features": {
                "research_intensity": 6.0,
                "advisor_style": 8.5,
                "team_work": 9.0,
                "workload": 6.5,
                "theory_practice": 8.0,
                "research_field_match": 7.0,
                "skill_development": 8.5,
                "lab_atmosphere": 9.0,
                "flexibility": 9.0,
                "publication_opportunity": 6.0,
                "interdisciplinary": 8.0,
                "communication_style": 9.0,
                "innovation_risk": 7.5
            }
        }
    ]
    
    # 進化設定
    evolution_config = EvolutionConfig(
        population_size=20,
        generations=30,
        mutation_rate=0.1,
        crossover_rate=0.8,
        elite_size=3
    )
    
    # 進化エンジン初期化
    engine = EvolutionEngine(evolution_config)
    
    # 進化実行
    result = engine.evolve(test_labs)
    
    print(f"🏆 最適化結果:")
    print(f"  最適個体: {result['best_individual']}")
    print(f"  最高適応度: {result['best_fitness']:.4f}")
    print(f"  完了世代数: {result['generations_completed']}")
    print(f"  収束達成: {result['convergence_achieved']}")
    
    print("✅ テスト完了")