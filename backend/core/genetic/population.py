# core/genetic/population.py - 遺伝的アルゴリズムの集団管理

import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from collections import defaultdict
import statistics

from core.genetic.individual import Individual

@dataclass
class PopulationStats:
    """集団統計情報"""
    generation: int
    population_size: int
    best_fitness: float
    worst_fitness: float
    average_fitness: float
    median_fitness: float
    fitness_std: float
    diversity_score: float
    convergence_rate: float

class Population:
    """遺伝的アルゴリズムの集団クラス"""
    
    def __init__(self, population_size: int):
        self.population_size = population_size
        self.individuals: List[Individual] = []
        self.generation = 0
        self.best_individual: Optional[Individual] = None
        self.statistics_history: List[PopulationStats] = []
        
        # 集団管理パラメータ
        self.elite_size = max(1, population_size // 10)  # エリートサイズ（10%）
        self.tournament_size = 3                         # トーナメントサイズ
        
    def initialize_random(self, research_fields: List[str], 
                         evaluation_criteria: List[str]) -> None:
        """ランダムに集団を初期化"""
        
        self.individuals = []
        
        for i in range(self.population_size):
            individual = Individual()
            individual.initialize_random(research_fields, evaluation_criteria)
            individual.generation = 0
            self.individuals.append(individual)
        
        print(f"🎲 ランダム集団初期化完了: {self.population_size}個体")
    
    def initialize_with_seeding(self, research_fields: List[str],
                               evaluation_criteria: List[str],
                               student_profile: 'StudentProfile',
                               seed_ratio: float = 0.3) -> None:
        """学生プロフィールを考慮した集団初期化"""
        
        self.individuals = []
        seed_count = int(self.population_size * seed_ratio)
        
        # シード個体（学生プロフィールベース）
        for i in range(seed_count):
            individual = Individual()
            individual.initialize_from_profile(student_profile, research_fields, evaluation_criteria)
            individual.generation = 0
            self.individuals.append(individual)
        
        # ランダム個体
        for i in range(seed_count, self.population_size):
            individual = Individual()
            individual.initialize_random(research_fields, evaluation_criteria)
            individual.generation = 0
            self.individuals.append(individual)
        
        print(f"🌱 シード集団初期化完了: シード{seed_count}個体 + ランダム{self.population_size - seed_count}個体")
    
    def evaluate_population(self, fitness_function: Callable[[Individual], float]) -> None:
        """集団全体の適応度を評価"""
        
        for individual in self.individuals:
            if individual.fitness == 0.0:  # 未評価の個体のみ
                fitness = fitness_function(individual)
                individual.update_fitness(fitness)
        
        # 最良個体を更新
        self._update_best_individual()
        
        # 統計情報を更新
        self._update_statistics()
    
    def _update_best_individual(self) -> None:
        """最良個体を更新"""
        if not self.individuals:
            return
        
        current_best = max(self.individuals, key=lambda x: x.fitness)
        
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best.clone()
            print(f"🏆 新しい最良個体: {self.best_individual.fitness:.4f} (世代{self.generation})")
    
    def _update_statistics(self) -> None:
        """統計情報を更新"""
        if not self.individuals:
            return
        
        fitness_values = [ind.fitness for ind in self.individuals]
        
        stats = PopulationStats(
            generation=self.generation,
            population_size=len(self.individuals),
            best_fitness=max(fitness_values),
            worst_fitness=min(fitness_values),
            average_fitness=statistics.mean(fitness_values),
            median_fitness=statistics.median(fitness_values),
            fitness_std=statistics.stdev(fitness_values) if len(fitness_values) > 1 else 0,
            diversity_score=self._calculate_population_diversity(),
            convergence_rate=self._calculate_convergence_rate()
        )
        
        self.statistics_history.append(stats)
    
    def _calculate_population_diversity(self) -> float:
        """集団の多様性を計算"""
        if len(self.individuals) < 2:
            return 0.0
        
        diversity_sum = 0.0
        comparison_count = 0
        
        # すべてのペアの多様性を計算
        for i in range(len(self.individuals)):
            for j in range(i + 1, len(self.individuals)):
                diversity = self.individuals[i].calculate_diversity(self.individuals[j])
                diversity_sum += diversity
                comparison_count += 1
        
        return diversity_sum / comparison_count if comparison_count > 0 else 0.0
    
    def _calculate_convergence_rate(self) -> float:
        """収束率を計算"""
        if len(self.statistics_history) < 2:
            return 0.0
        
        # 最近5世代の最良適応度の変化率
        recent_stats = self.statistics_history[-5:]
        if len(recent_stats) < 2:
            return 0.0
        
        fitness_changes = []
        for i in range(1, len(recent_stats)):
            change = recent_stats[i].best_fitness - recent_stats[i-1].best_fitness
            fitness_changes.append(change)
        
        return statistics.mean(fitness_changes)
    
    def selection_tournament(self, tournament_size: Optional[int] = None) -> List[Individual]:
        """トーナメント選択"""
        if tournament_size is None:
            tournament_size = self.tournament_size
        
        selected = []
        
        for _ in range(self.population_size):
            # トーナメント参加者をランダム選択
            tournament = random.sample(self.individuals, min(tournament_size, len(self.individuals)))
            
            # 最良個体を選択
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)
        
        return selected
    
    def selection_roulette(self) -> List[Individual]:
        """ルーレット選択"""
        fitness_values = [ind.fitness for ind in self.individuals]
        
        # 負の適応度を0に調整
        min_fitness = min(fitness_values)
        if min_fitness < 0:
            adjusted_fitness = [f - min_fitness + 0.1 for f in fitness_values]
        else:
            adjusted_fitness = [f + 0.1 for f in fitness_values]  # 最低値0.1を保証
        
        total_fitness = sum(adjusted_fitness)
        
        if total_fitness == 0:
            # 全個体の適応度が0の場合はランダム選択
            return random.choices(self.individuals, k=self.population_size)
        
        # 確率に基づく選択
        probabilities = [f / total_fitness for f in adjusted_fitness]
        selected = random.choices(self.individuals, weights=probabilities, k=self.population_size)
        
        return selected
    
    def selection_rank(self) -> List[Individual]:
        """ランク選択"""
        # 適応度でソート
        sorted_individuals = sorted(self.individuals, key=lambda x: x.fitness)
        
        # ランクに基づく重み（線形ランキング）
        ranks = list(range(1, len(sorted_individuals) + 1))
        total_rank = sum(ranks)
        
        probabilities = [rank / total_rank for rank in ranks]
        
        selected = random.choices(sorted_individuals, weights=probabilities, k=self.population_size)
        
        return selected
    
    def crossover_population(self, selected: List[Individual], 
                           crossover_rate: float = 0.8) -> List[Individual]:
        """集団全体での交叉"""
        offspring = []
        
        # エリートを保存
        elite = self.get_elite()
        offspring.extend([ind.clone() for ind in elite])
        
        # 残りを交叉で生成
        while len(offspring) < self.population_size:
            parent1, parent2 = random.sample(selected, 2)
            
            if random.random() < crossover_rate:
                child1, child2 = parent1.crossover_with(parent2, crossover_rate)
            else:
                child1, child2 = parent1.clone(), parent2.clone()
            
            offspring.extend([child1, child2])
        
        # サイズ調整
        return offspring[:self.population_size]
    
    def mutate_population(self, offspring: List[Individual], 
                         mutation_rate: float = 0.1) -> List[Individual]:
        """集団全体での変異"""
        
        # エリートは変異しない
        elite_count = self.elite_size
        
        for i in range(elite_count, len(offspring)):
            offspring[i].mutate(mutation_rate)
        
        return offspring
    
    def get_elite(self) -> List[Individual]:
        """エリート個体を取得"""
        sorted_individuals = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)
        return sorted_individuals[:self.elite_size]
    
    def replace_worst_with_immigrants(self, immigrant_ratio: float = 0.05,
                                    research_fields: List[str] = None,
                                    evaluation_criteria: List[str] = None) -> None:
        """最悪個体を移民で置換（多様性維持）"""
        
        if research_fields is None or evaluation_criteria is None:
            return
        
        immigrant_count = max(1, int(self.population_size * immigrant_ratio))
        
        # 最悪個体を特定
        sorted_individuals = sorted(self.individuals, key=lambda x: x.fitness)
        worst_individuals = sorted_individuals[:immigrant_count]
        
        # 移民個体を生成
        immigrants = []
        for _ in range(immigrant_count):
            immigrant = Individual()
            immigrant.initialize_random(research_fields, evaluation_criteria)
            immigrant.generation = self.generation
            immigrants.append(immigrant)
        
        # 置換
        for i, worst_ind in enumerate(worst_individuals):
            idx = self.individuals.index(worst_ind)
            self.individuals[idx] = immigrants[i]
        
        print(f"🌍 移民導入: {immigrant_count}個体を置換")
    
    def advance_generation(self, new_population: List[Individual]) -> None:
        """次世代に進む"""
        self.individuals = new_population
        self.generation += 1
        
        # 世代番号を更新
        for individual in self.individuals:
            if individual.generation < self.generation:
                individual.generation = self.generation
    
    def is_converged(self, convergence_threshold: float = 1e-6,
                    stagnation_generations: int = 10) -> bool:
        """収束判定"""
        
        if len(self.statistics_history) < stagnation_generations:
            return False
        
        # 最近の世代での適応度変化をチェック
        recent_best_fitness = [stats.best_fitness for stats in self.statistics_history[-stagnation_generations:]]
        
        # 適応度の標準偏差が閾値以下なら収束
        if len(recent_best_fitness) > 1:
            fitness_std = statistics.stdev(recent_best_fitness)
            return fitness_std < convergence_threshold
        
        return False
    
    def get_diversity_statistics(self) -> Dict[str, float]:
        """多様性統計を取得"""
        
        if len(self.individuals) < 2:
            return {"diversity": 0.0, "avg_distance": 0.0, "cluster_count": 1}
        
        # 平均遺伝的距離
        total_distance = 0.0
        comparison_count = 0
        
        for i in range(len(self.individuals)):
            for j in range(i + 1, len(self.individuals)):
                distance = self.individuals[i].distance_to(self.individuals[j])
                total_distance += distance
                comparison_count += 1
        
        avg_distance = total_distance / comparison_count if comparison_count > 0 else 0.0
        
        # クラスター数の推定（類似個体のグループ数）
        clusters = self._estimate_clusters()
        
        return {
            "diversity": self._calculate_population_diversity(),
            "avg_distance": avg_distance,
            "cluster_count": len(clusters),
            "largest_cluster_size": max(len(cluster) for cluster in clusters) if clusters else 0
        }
    
    def _estimate_clusters(self, similarity_threshold: float = 0.1) -> List[List[Individual]]:
        """類似個体のクラスターを推定"""
        
        clusters = []
        unassigned = self.individuals[:]
        
        while unassigned:
            # 新しいクラスター開始
            cluster = [unassigned.pop(0)]
            
            # 類似個体を同じクラスターに追加
            i = 0
            while i < len(unassigned):
                is_similar = any(
                    cluster_member.is_similar_to(unassigned[i], similarity_threshold)
                    for cluster_member in cluster
                )
                
                if is_similar:
                    cluster.append(unassigned.pop(i))
                else:
                    i += 1
            
            clusters.append(cluster)
        
        return clusters
    
    def get_population_summary(self) -> Dict[str, Any]:
        """集団の統計サマリーを取得"""
        
        if not self.statistics_history:
            return {"message": "統計データがありません"}
        
        current_stats = self.statistics_history[-1]
        diversity_stats = self.get_diversity_statistics()
        
        # 適応度推移
        fitness_trend = [stats.best_fitness for stats in self.statistics_history[-10:]]
        
        # エリート分析
        elite = self.get_elite()
        elite_analysis = {
            "count": len(elite),
            "avg_fitness": statistics.mean([ind.fitness for ind in elite]),
            "fitness_range": max([ind.fitness for ind in elite]) - min([ind.fitness for ind in elite]) if len(elite) > 1 else 0
        }
        
        return {
            "generation": current_stats.generation,
            "population_size": current_stats.population_size,
            "fitness": {
                "best": current_stats.best_fitness,
                "worst": current_stats.worst_fitness,
                "average": current_stats.average_fitness,
                "median": current_stats.median_fitness,
                "std": current_stats.fitness_std
            },
            "diversity": diversity_stats,
            "convergence": {
                "rate": current_stats.convergence_rate,
                "is_converged": self.is_converged()
            },
            "elite": elite_analysis,
            "fitness_trend": fitness_trend
        }
    
    def export_population_data(self) -> Dict[str, Any]:
        """集団データをエクスポート"""
        
        return {
            "generation": self.generation,
            "population_size": self.population_size,
            "individuals": [ind.to_dict() for ind in self.individuals],
            "best_individual": self.best_individual.to_dict() if self.best_individual else None,
            "statistics_history": [
                {
                    "generation": stats.generation,
                    "best_fitness": stats.best_fitness,
                    "average_fitness": stats.average_fitness,
                    "diversity_score": stats.diversity_score
                }
                for stats in self.statistics_history
            ]
        }
    
    def clear_statistics(self) -> None:
        """統計履歴をクリア"""
        self.statistics_history.clear()
    
    def __len__(self) -> int:
        return len(self.individuals)
    
    def __getitem__(self, index: int) -> Individual:
        return self.individuals[index]
    
    def __iter__(self):
        return iter(self.individuals)
    
    def __str__(self) -> str:
        return f"Population(Gen:{self.generation}, Size:{len(self.individuals)}, Best:{self.best_individual.fitness:.4f if self.best_individual else 0:.4f})"