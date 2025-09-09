# core/genetic/population.py - 集団管理（完全版）

import numpy as np
import random
import json
import pickle
from typing import Dict, List, Any, Optional, Tuple, Callable, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict
import logging

# 個体クラスのインポート（循環インポート回避）
try:
    from core.genetic.individual import Individual, WeightVector, FuzzyTreeIndividual
except ImportError:
    # フォールバック実装
    class Individual:
        def __init__(self, individual_id: str = None):
            self.individual_id = individual_id or f"ind_{random.randint(1000, 9999)}"
            self.fitness_value = None
            self.generation = 0
        
        def get_fitness(self):
            return self.fitness_value or 0.0
        
        def is_evaluated(self):
            return self.fitness_value is not None
        
        def clone(self):
            return Individual(self.individual_id + "_clone")
    
    class WeightVector(Individual):
        pass
    
    class FuzzyTreeIndividual(Individual):
        pass

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
            logger.warning(f"集団設定警告: {', '.join(validation_errors)}")
    
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
    
    def get_best_individual(self) -> Optional[Individual]:
        """最良個体を取得"""
        
        if not self.individuals:
            return None
        
        # 適応度が設定されている個体のみを対象
        evaluated_individuals = [ind for ind in self.individuals if ind.is_evaluated()]
        
        if not evaluated_individuals:
            return None
        
        return max(evaluated_individuals, key=lambda x: x.get_fitness())
    
    def get_worst_individual(self) -> Optional[Individual]:
        """最悪個体を取得"""
        
        if not self.individuals:
            return None
        
        evaluated_individuals = [ind for ind in self.individuals if ind.is_evaluated()]
        
        if not evaluated_individuals:
            return None
        
        return min(evaluated_individuals, key=lambda x: x.get_fitness())
    
    def select_elite(self, elite_size: Optional[int] = None) -> List[Individual]:
        """エリート個体を選択"""
        
        if elite_size is None:
            elite_size = self.config.elite_size
        
        if not self.individuals:
            return []
        
        # 適応度順にソート
        sorted_individuals = sorted(
            [ind for ind in self.individuals if ind.is_evaluated()],
            key=lambda x: x.get_fitness(),
            reverse=True
        )
        
        elite_count = min(elite_size, len(sorted_individuals))
        self.elite_individuals = sorted_individuals[:elite_count]
        
        return self.elite_individuals
    
    def evaluate_population(self, fitness_function: Callable[[Individual], float]) -> None:
        """集団全体を評価"""
        
        self.fitness_function = fitness_function
        
        for individual in self.individuals:
            if not individual.is_evaluated():
                try:
                    fitness = fitness_function(individual)
                    individual.fitness_value = fitness
                    self.total_evaluations += 1
                except Exception as e:
                    logger.warning(f"個体評価エラー {individual.individual_id}: {e}")
                    individual.fitness_value = 0.0
        
        # 最良個体の更新
        self.best_individual = self.get_best_individual()
    
    def calculate_diversity(self) -> float:
        """集団の多様性を計算"""
        
        if len(self.individuals) < 2:
            return 0.0
        
        # 適応度の分散による多様性計算
        fitness_values = [ind.get_fitness() for ind in self.individuals if ind.is_evaluated()]
        
        if len(fitness_values) < 2:
            return 0.0
        
        return float(np.std(fitness_values))
    
    def calculate_genetic_diversity(self) -> float:
        """遺伝的多様性を計算"""
        
        if len(self.individuals) < 2:
            return 0.0
        
        # 遺伝子レベルでの多様性
        genetic_distances = []
        
        for i in range(len(self.individuals)):
            for j in range(i + 1, len(self.individuals)):
                distance = self._calculate_genetic_distance(
                    self.individuals[i], self.individuals[j]
                )
                genetic_distances.append(distance)
        
        return float(np.mean(genetic_distances)) if genetic_distances else 0.0
    
    def _calculate_genetic_distance(self, ind1: Individual, ind2: Individual) -> float:
        """2個体間の遺伝的距離を計算"""
        
        if not (hasattr(ind1, 'get_genes') and hasattr(ind2, 'get_genes')):
            return 0.0
        
        genes1 = ind1.get_genes()
        genes2 = ind2.get_genes()
        
        if not genes1 or not genes2:
            return 0.0
        
        # ユークリッド距離
        all_keys = set(genes1.keys()) | set(genes2.keys())
        distance = 0.0
        
        for key in all_keys:
            val1 = genes1.get(key, 0.0)
            val2 = genes2.get(key, 0.0)
            distance += (val1 - val2) ** 2
        
        return math.sqrt(distance)
    
    def maintain_diversity(self) -> None:
        """多様性を維持"""
        
        if len(self.individuals) < 2:
            return
        
        # 多様性が閾値を下回る場合の処理
        diversity = self.calculate_diversity()
        
        if diversity < self.config.diversity_threshold:
            # 最悪個体を突然変異個体で置換
            worst_individual = self.get_worst_individual()
            if worst_individual:
                # 新しいランダム個体を生成
                new_individual = self.individual_type(
                    individual_id=f"div_gen{self.current_generation}_ind{random.randint(1000, 9999)}"
                )
                new_individual.generation = self.current_generation
                
                # 置換
                for i, ind in enumerate(self.individuals):
                    if ind.individual_id == worst_individual.individual_id:
                        self.individuals[i] = new_individual
                        break
                
                logger.debug(f"多様性維持のため個体置換実行")
    
    def get_statistics(self) -> PopulationStatistics:
        """集団統計を計算"""
        
        stats = PopulationStatistics()
        stats.generation = self.current_generation
        stats.population_size = len(self.individuals)
        
        # 適応度統計
        evaluated_individuals = [ind for ind in self.individuals if ind.is_evaluated()]
        
        if evaluated_individuals:
            fitness_values = [ind.get_fitness() for ind in evaluated_individuals]
            
            stats.best_fitness = max(fitness_values)
            stats.worst_fitness = min(fitness_values)
            stats.average_fitness = np.mean(fitness_values)
            stats.median_fitness = np.median(fitness_values)
            stats.fitness_variance = np.var(fitness_values)
            stats.fitness_std = np.std(fitness_values)
        
        # 多様性統計
        stats.average_diversity = self.calculate_diversity()
        stats.genetic_diversity = self.calculate_genetic_diversity()
        
        # その他統計
        stats.elite_count = len(self.elite_individuals)
        stats.evaluation_count = self.total_evaluations
        
        # 改善率計算
        if len(self.best_fitness_history) > 1:
            previous_best = self.best_fitness_history[-2]
            current_best = self.best_fitness_history[-1]
            stats.improvement_rate = (current_best - previous_best) / abs(previous_best) if previous_best != 0 else 0.0
        
        # 収束指標
        if len(self.best_fitness_history) >= 5:
            recent_variance = np.var(self.best_fitness_history[-5:])
            stats.convergence_indicator = 1.0 / (1.0 + recent_variance)
        
        return stats
    
    def update_generation(self) -> None:
        """世代を更新"""
        
        self.current_generation += 1
        
        # 統計の記録
        stats = self.get_statistics()
        self.generation_history.append(stats)
        
        # 履歴の更新
        if stats.best_fitness:
            self.best_fitness_history.append(stats.best_fitness)
        if stats.average_fitness:
            self.average_fitness_history.append(stats.average_fitness)
        if stats.average_diversity:
            self.diversity_history.append(stats.average_diversity)
        
        logger.debug(f"世代更新: {self.current_generation}")
    
    def save_population(self, filepath: str) -> None:
        """集団をファイルに保存"""
        
        try:
            data = {
                "generation": self.current_generation,
                "config": self.config.__dict__,
                "individuals": [ind.__dict__ for ind in self.individuals],
                "statistics": [stats.__dict__ for stats in self.generation_history],
                "best_fitness_history": self.best_fitness_history,
                "average_fitness_history": self.average_fitness_history,
                "diversity_history": self.diversity_history
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"集団保存完了: {filepath}")
            
        except Exception as e:
            logger.error(f"集団保存エラー: {e}")
    
    def load_population(self, filepath: str) -> None:
        """集団をファイルから読み込み"""
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.current_generation = data["generation"]
            self.best_fitness_history = data["best_fitness_history"]
            self.average_fitness_history = data["average_fitness_history"]
            self.diversity_history = data["diversity_history"]
            
            # 個体の復元（簡易版）
            self.individuals = []
            for ind_data in data["individuals"]:
                individual = self.individual_type(ind_data["individual_id"])
                individual.fitness_value = ind_data.get("fitness_value")
                individual.generation = ind_data.get("generation", 0)
                self.individuals.append(individual)
            
            logger.info(f"集団読み込み完了: {filepath}")
            
        except Exception as e:
            logger.error(f"集団読み込みエラー: {e}")
    
    def get_population_summary(self) -> Dict[str, Any]:
        """集団の要約情報を取得"""
        
        stats = self.get_statistics()
        
        return {
            "generation": self.current_generation,
            "population_size": len(self.individuals),
            "best_fitness": stats.best_fitness,
            "average_fitness": stats.average_fitness,
            "diversity": stats.average_diversity,
            "elite_count": stats.elite_count,
            "evaluation_count": self.total_evaluations,
            "convergence": stats.convergence_indicator
        }

# 使用例とテスト
def test_population():
    """集団クラスのテスト"""
    
    print("🧬 集団管理テスト開始")
    
    # 設定作成
    config = PopulationConfig(
        population_size=20,
        elite_size=3,
        selection_pressure=2.0,
        diversity_threshold=0.1
    )
    
    # 集団初期化
    population = Population(config, WeightVector)
    population.initialize_random(weight_names=["w1", "w2", "w3"])
    
    print(f"✅ 集団初期化完了: {len(population.individuals)}個体")
    
    # 適応度関数（テスト用）
    def test_fitness(individual):
        if hasattr(individual, 'get_genes'):
            genes = individual.get_genes()
            return sum(genes.values()) if genes else random.random()
        return random.random()
    
    # 集団評価
    population.evaluate_population(test_fitness)
    
    # 統計計算
    stats = population.get_statistics()
    
    print(f"\n📊 集団統計:")
    print(f"  最高適応度: {stats.best_fitness:.3f}")
    print(f"  平均適応度: {stats.average_fitness:.3f}")
    print(f"  多様性: {stats.average_diversity:.3f}")
    print(f"  評価回数: {stats.evaluation_count}")
    
    # エリート選択
    elite = population.select_elite()
    print(f"\n🏆 エリート選択:")
    print(f"  エリート数: {len(elite)}")
    
    if elite:
        print(f"  最良個体適応度: {elite[0].get_fitness():.3f}")
    
    # 多様性維持
    population.maintain_diversity()
    
    # 世代更新
    population.update_generation()
    
    print(f"\n📈 集団要約:")
    summary = population.get_population_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("✅ 集団管理テスト完了")

if __name__ == "__main__":
    import math
    test_population()