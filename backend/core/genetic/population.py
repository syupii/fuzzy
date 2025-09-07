# core/genetic/population.py - 集団管理

import numpy as np
import random
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple, Callable, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict
import logging

from core.genetic.individual import Individual, WeightVector, FuzzyTreeIndividual

logger = logging.getLogger(__name__)

@dataclass
class PopulationConfig:
    """集団設定"""
    population_size: int = 50
    elite_size: int = 5
    max_generations: int = 100
    
    # 選択設定
    selection_pressure: float = 2.0
    tournament_size: int = 3
    
    # 多様性設定
    diversity_threshold: float = 0.1
    max_diversity_age: int = 10
    
    # 統計設定
    track_genealogy: bool = True
    save_best_individual: bool = True
    
    def validate(self) -> List[str]:
        """設定値の検証"""
        errors = []
        
        if self.population_size < 2:
            errors.append("集団サイズは2以上である必要があります")
        
        if self.elite_size >= self.population_size:
            errors.append("エリートサイズは集団サイズ未満である必要があります")
        
        if self.tournament_size > self.population_size:
            errors.append("トーナメントサイズは集団サイズ以下である必要があります")
        
        if not 0 < self.selection_pressure <= 10:
            errors.append("選択圧は0-10の範囲で指定してください")
        
        return errors

@dataclass
class PopulationStatistics:
    """集団統計情報"""
    generation: int = 0
    population_size: int = 0
    
    # 適応度統計
    best_fitness: float = 0.0
    worst_fitness: float = 0.0
    average_fitness: float = 0.0
    median_fitness: float = 0.0
    fitness_variance: float = 0.0
    fitness_std: float = 0.0
    
    # 多様性統計
    average_diversity: float = 0.0
    genetic_diversity: float = 0.0
    phenotypic_diversity: float = 0.0
    
    # 進化統計
    improvement_rate: float = 0.0
    convergence_indicator: float = 0.0
    stagnation_generations: int = 0
    
    # その他
    elite_count: int = 0
    unique_individuals: int = 0
    evaluation_count: int = 0

class Population:
    """遺伝的アルゴリズム集団クラス"""
    
    def __init__(self, config: PopulationConfig, 
                 individual_type: Type[Individual] = WeightVector):
        
        self.config = config
        self.individual_type = individual_type
        
        # 個体群
        self.individuals: List[Individual] = []
        self.elite_individuals: List[Individual] = []
        
        # 世代管理
        self.current_generation = 0
        self.generation_history: List[PopulationStatistics] = []
        
        # 統計情報
        self.best_individual: Optional[Individual] = None
        self.best_fitness_history: List[float] = []
        self.average_fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        
        # 多様性管理
        self.diversity_matrix: Optional[np.ndarray] = None
        self.genetic_clusters: Dict[str, List[Individual]] = {}
        
        # 評価管理
        self.total_evaluations = 0
        self.fitness_function: Optional[Callable] = None
        
        # ファイル管理
        self.save_directory = "./data/populations"
        
        # 設定検証
        validation_errors = self.config.validate()
        if validation_errors:
            raise ValueError(f"集団設定エラー: {', '.join(validation_errors)}")
    
    def initialize_random(self, **kwargs) -> None:
        """ランダムな個体群で初期化"""
        
        self.individuals = []
        
        for i in range(self.config.population_size):
            if self.individual_type == WeightVector:
                individual = WeightVector(
                    individual_id=f"gen0_ind{i}",
                    weight_names=kwargs.get('weight_names')
                )
            elif self.individual_type == FuzzyTreeIndividual:
                individual = FuzzyTreeIndividual(
                    individual_id=f"gen0_ind{i}"
                )
            else:
                individual = self.individual_type(individual_id=f"gen0_ind{i}")
            
            individual.generation = self.current_generation
            self.individuals.append(individual)
        
        logger.info(f"集団を{self.config.population_size}個体でランダム初期化完了")
    
    def add_individual(self, individual: Individual) -> None:
        """個体を追加"""
        individual.generation = self.current_generation
        self.individuals.append(individual)
    
    def remove_individual(self, individual_id: str) -> bool:
        """個体を削除"""
        for i, individual in enumerate(self.individuals):
            if individual.individual_id == individual_id:
                del self.individuals[i]
                return True
        return False
    
    def get_individual(self, individual_id: str) -> Optional[Individual]:
        """個体を取得"""
        for individual in self.individuals:
            if individual.individual_id == individual_id:
                return individual
        return None
    
    def evaluate_population(self, fitness_function: Callable[[Individual], float]) -> None:
        """集団全体の適応度評価"""
        
        self.fitness_function = fitness_function
        evaluated_count = 0
        
        for individual in self.individuals:
            if not individual.is_evaluated():
                try:
                    fitness = fitness_function(individual)
                    individual.set_fitness(fitness)
                    self.total_evaluations += 1
                    evaluated_count += 1
                except Exception as e:
                    logger.error(f"個体評価エラー {individual.individual_id}: {e}")
                    individual.set_fitness(0.0)
        
        # 統計更新
        self._update_statistics()
        
        logger.info(f"集団評価完了: {evaluated_count}個体を新規評価")
    
    def sort_by_fitness(self, reverse: bool = True) -> None:
        """適応度でソート"""
        self.individuals.sort(
            key=lambda x: x.get_fitness() or 0.0, 
            reverse=reverse
        )
    
    def select_elite(self) -> List[Individual]:
        """エリート個体の選択"""
        self.sort_by_fitness(reverse=True)
        elite_size = min(self.config.elite_size, len(self.individuals))
        
        self.elite_individuals = []
        for i in range(elite_size):
            elite = self.individuals[i].clone()
            elite.is_elite = True
            self.elite_individuals.append(elite)
        
        return self.elite_individuals.copy()
    
    def tournament_selection(self, tournament_size: Optional[int] = None) -> Individual:
        """トーナメント選択"""
        if tournament_size is None:
            tournament_size = self.config.tournament_size
        
        tournament_size = min(tournament_size, len(self.individuals))
        
        # ランダムに個体を選択
        contestants = random.sample(self.individuals, tournament_size)
        
        # 最高適応度の個体を選択
        winner = max(contestants, key=lambda x: x.get_fitness() or 0.0)
        
        return winner
    
    def roulette_selection(self) -> Individual:
        """ルーレット選択"""
        fitness_values = [individual.get_fitness() or 0.0 for individual in self.individuals]
        
        # 負の適応度の処理
        min_fitness = min(fitness_values)
        if min_fitness < 0:
            fitness_values = [f - min_fitness + 1e-6 for f in fitness_values]
        
        total_fitness = sum(fitness_values)
        if total_fitness == 0:
            return random.choice(self.individuals)
        
        # ルーレット回転
        selection_point = random.uniform(0, total_fitness)
        cumulative_fitness = 0.0
        
        for i, fitness in enumerate(fitness_values):
            cumulative_fitness += fitness
            if cumulative_fitness >= selection_point:
                return self.individuals[i]
        
        # フォールバック
        return self.individuals[-1]
    
    def rank_selection(self) -> Individual:
        """ランク選択"""
        self.sort_by_fitness(reverse=True)
        
        n = len(self.individuals)
        ranks = list(range(n, 0, -1))  # n, n-1, ..., 1
        
        # 選択圧を適用
        adjusted_ranks = [rank ** self.config.selection_pressure for rank in ranks]
        total_rank = sum(adjusted_ranks)
        
        if total_rank == 0:
            return random.choice(self.individuals)
        
        selection_point = random.uniform(0, total_rank)
        cumulative_rank = 0.0
        
        for i, rank in enumerate(adjusted_ranks):
            cumulative_rank += rank
            if cumulative_rank >= selection_point:
                return self.individuals[i]
        
        return self.individuals[-1]
    
    def calculate_diversity(self) -> float:
        """集団の多様性を計算"""
        if len(self.individuals) < 2:
            return 0.0
        
        n = len(self.individuals)
        total_diversity = 0.0
        comparison_count = 0
        
        # 全ペアの多様性を計算
        for i in range(n):
            for j in range(i + 1, n):
                diversity = self.individuals[i].get_diversity_from(self.individuals[j])
                total_diversity += diversity
                comparison_count += 1
        
        average_diversity = total_diversity / comparison_count if comparison_count > 0 else 0.0
        
        # 多様性行列の更新
        self.diversity_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                diversity = self.individuals[i].get_diversity_from(self.individuals[j])
                self.diversity_matrix[i][j] = diversity
                self.diversity_matrix[j][i] = diversity
        
        return average_diversity
    
    def maintain_diversity(self) -> None:
        """多様性の維持"""
        if len(self.individuals) < 2:
            return
        
        # 多様性の計算
        current_diversity = self.calculate_diversity()
        
        # 多様性が低い場合の処理
        if current_diversity < self.config.diversity_threshold:
            self._apply_diversity_maintenance()
    
    def _apply_diversity_maintenance(self) -> None:
        """多様性維持の適用"""
        
        # 類似個体の特定
        similar_pairs = []
        n = len(self.individuals)
        
        if self.diversity_matrix is not None:
            for i in range(n):
                for j in range(i + 1, n):
                    if self.diversity_matrix[i][j] < self.config.diversity_threshold:
                        similar_pairs.append((i, j, self.diversity_matrix[i][j]))
        
        # 類似度順にソート
        similar_pairs.sort(key=lambda x: x[2])
        
        # 類似個体の一方を突然変異
        diversity_maintained = 0
        for i, j, similarity in similar_pairs[:len(self.individuals) // 4]:
            # 適応度の低い方を変異
            if (self.individuals[i].get_fitness() or 0.0) < (self.individuals[j].get_fitness() or 0.0):
                target_individual = self.individuals[i]
            else:
                target_individual = self.individuals[j]
            
            # 強い突然変異を適用
            target_individual.mutate(0.5, 0.3)
            target_individual.fitness_value = None  # 再評価が必要
            diversity_maintained += 1
        
        if diversity_maintained > 0:
            logger.info(f"多様性維持: {diversity_maintained}個体を変異")
    
    def _update_statistics(self) -> None:
        """統計情報の更新"""
        
        if not self.individuals:
            return
        
        # 適応度リストの取得
        fitness_values = [individual.get_fitness() or 0.0 for individual in self.individuals]
        
        # 基本統計
        stats = PopulationStatistics()
        stats.generation = self.current_generation
        stats.population_size = len(self.individuals)
        stats.best_fitness = max(fitness_values)
        stats.worst_fitness = min(fitness_values)
        stats.average_fitness = sum(fitness_values) / len(fitness_values)
        stats.median_fitness = np.median(fitness_values)
        stats.fitness_variance = np.var(fitness_values)
        stats.fitness_std = np.std(fitness_values)
        
        # 多様性統計
        stats.average_diversity = self.calculate_diversity()
        
        # 進化統計
        if len(self.best_fitness_history) > 0:
            stats.improvement_rate = (stats.best_fitness - self.best_fitness_history[-1]) / max(abs(self.best_fitness_history[-1]), 1e-6)
        
        # 収束指標
        stats.convergence_indicator = 1.0 - (stats.fitness_std / (abs(stats.average_fitness) + 1e-6))
        
        # エリート統計
        stats.elite_count = len(self.elite_individuals)
        
        # ユニーク個体数
        unique_genomes = set()
        for individual in self.individuals:
            genome_str = json.dumps(individual.get_genes(), sort_keys=True)
            unique_genomes.add(genome_str)
        stats.unique_individuals = len(unique_genomes)
        
        # 評価回数
        stats.evaluation_count = self.total_evaluations
        
        # 履歴の更新
        self.generation_history.append(stats)
        self.best_fitness_history.append(stats.best_fitness)
        self.average_fitness_history.append(stats.average_fitness)
        self.diversity_history.append(stats.average_diversity)
        
        # 最良個体の更新
        if self.best_individual is None or stats.best_fitness > self.best_individual.get_fitness():
            best_individual = max(self.individuals, key=lambda x: x.get_fitness() or 0.0)
            self.best_individual = best_individual.clone()
    
    def advance_generation(self) -> None:
        """世代を進める"""
        self.current_generation += 1
        
        # 個体の年齢を更新
        for individual in self.individuals:
            individual.age += 1
    
    def get_statistics(self) -> Optional[PopulationStatistics]:
        """最新の統計情報を取得"""
        return self.generation_history[-1] if self.generation_history else None
    
    def get_best_individual(self) -> Optional[Individual]:
        """最良個体を取得"""
        return self.best_individual
    
    def get_population_summary(self) -> Dict[str, Any]:
        """集団の要約情報を取得"""
        
        current_stats = self.get_statistics()
        
        return {
            "generation": self.current_generation,
            "population_size": len(self.individuals),
            "total_evaluations": self.total_evaluations,
            "best_fitness": current_stats.best_fitness if current_stats else 0.0,
            "average_fitness": current_stats.average_fitness if current_stats else 0.0,
            "diversity": current_stats.average_diversity if current_stats else 0.0,
            "elite_count": len(self.elite_individuals),
            "convergence": current_stats.convergence_indicator if current_stats else 0.0,
            "unique_individuals": current_stats.unique_individuals if current_stats else 0
        }
    
    def save_population(self, filename: str = None) -> str:
        """集団をファイルに保存"""
        
        if filename is None:
            filename = f"population_gen{self.current_generation}.pkl"
        
        import os
        os.makedirs(self.save_directory, exist_ok=True)
        filepath = os.path.join(self.save_directory, filename)
        
        save_data = {
            "config": self.config,
            "individuals": [ind.to_dict() for ind in self.individuals],
            "elite_individuals": [ind.to_dict() for ind in self.elite_individuals],
            "current_generation": self.current_generation,
            "generation_history": self.generation_history,
            "best_individual": self.best_individual.to_dict() if self.best_individual else None,
            "total_evaluations": self.total_evaluations
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)
            
            logger.info(f"集団保存完了: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"集団保存エラー: {e}")
            raise
    
    def load_population(self, filepath: str) -> None:
        """ファイルから集団を読み込み"""
        
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            # 基本情報の復元
            self.config = save_data.get("config", self.config)
            self.current_generation = save_data.get("current_generation", 0)
            self.generation_history = save_data.get("generation_history", [])
            self.total_evaluations = save_data.get("total_evaluations", 0)
            
            # 個体の復元
            self.individuals = []
            for ind_data in save_data.get("individuals", []):
                individual = self.individual_type()
                individual.from_dict(ind_data)
                self.individuals.append(individual)
            
            # エリート個体の復元
            self.elite_individuals = []
            for elite_data in save_data.get("elite_individuals", []):
                elite = self.individual_type()
                elite.from_dict(elite_data)
                self.elite_individuals.append(elite)
            
            # 最良個体の復元
            best_data = save_data.get("best_individual")
            if best_data:
                self.best_individual = self.individual_type()
                self.best_individual.from_dict(best_data)
            
            logger.info(f"集団読み込み完了: {filepath}")
            
        except Exception as e:
            logger.error(f"集団読み込みエラー: {e}")
            raise

# 使用例とテスト
def test_population():
    """集団クラスのテスト"""
    
    print("👥 集団管理テスト開始")
    
    # 設定の作成
    config = PopulationConfig(
        population_size=20,
        elite_size=3,
        tournament_size=3
    )
    
    # 集団の初期化
    population = Population(config, WeightVector)
    population.initialize_random(weight_names=["w1", "w2", "w3"])
    
    print(f"✅ 集団初期化完了: {len(population.individuals)}個体")
    
    # 簡易適応度関数
    def simple_fitness(individual):
        genes = individual.get_genes()
        return sum(genes.values())
    
    # 集団評価
    population.evaluate_population(simple_fitness)
    
    # 統計情報
    stats = population.get_statistics()
    print(f"📊 統計情報:")
    print(f"  最高適応度: {stats.best_fitness:.3f}")
    print(f"  平均適応度: {stats.average_fitness:.3f}")
    print(f"  多様性: {stats.average_diversity:.3f}")
    
    # エリート選択
    elite = population.select_elite()
    print(f"👑 エリート選択完了: {len(elite)}個体")
    
    # 選択テスト
    selected = population.tournament_selection()
    print(f"🎯 トーナメント選択: 個体{selected.individual_id} (適応度: {selected.get_fitness():.3f})")
    
    print("✅ 集団管理テスト完了")

if __name__ == "__main__":
    test_population()