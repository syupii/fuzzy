# core/genetic/evolution.py - 進化アルゴリズム（完全版）

import numpy as np
import random
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Callable, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

# 必要なクラスのインポート（循環インポート回避）
try:
    from core.genetic.individual import Individual, WeightVector, FuzzyTreeIndividual
    from core.genetic.population import Population, PopulationConfig, PopulationStatistics
except ImportError:
    # フォールバック実装
    class Individual:
        def __init__(self, individual_id: str = None):
            self.individual_id = individual_id or f"ind_{random.randint(1000, 9999)}"
            self.fitness_value = None
            self.generation = 0
            
        def get_fitness(self):
            return self.fitness_value or 0.0
            
        def get_genes(self):
            return {}
    
    class WeightVector(Individual):
        pass
    
    class FuzzyTreeIndividual(Individual):
        pass
    
    class Population:
        def __init__(self, config=None, individual_type=Individual):
            self.individuals = []
            self.config = config
            self.individual_type = individual_type
            self.total_evaluations = 0
            self.generation_history = []
        
        def initialize_random(self, **kwargs):
            pass
        
        def get_best_individual(self):
            return Individual()
    
    class PopulationConfig:
        def __init__(self, population_size=50, elite_size=5):
            self.population_size = population_size
            self.elite_size = elite_size
    
    class PopulationStatistics:
        def __init__(self):
            self.best_fitness = 0.0
            self.average_fitness = 0.0

# 遺伝的操作のための簡易実装
try:
    from core.genetic.operators import (
        OperatorFactory, OperatorConfig, SelectionMethod, 
        CrossoverMethod, MutationMethod
    )
except ImportError:
    # フォールバック実装
    from enum import Enum
    
    class SelectionMethod(str, Enum):
        TOURNAMENT = "tournament"
        ROULETTE = "roulette"
        RANK = "rank"
    
    class CrossoverMethod(str, Enum):
        UNIFORM = "uniform"
        SINGLE_POINT = "single_point"
        TWO_POINT = "two_point"
    
    class MutationMethod(str, Enum):
        GAUSSIAN = "gaussian"
        UNIFORM = "uniform"
        POLYNOMIAL = "polynomial"
    
    class OperatorConfig:
        def __init__(self, **kwargs):
            self.selection_method = kwargs.get('selection_method', SelectionMethod.TOURNAMENT)
            self.crossover_method = kwargs.get('crossover_method', CrossoverMethod.UNIFORM)
            self.mutation_method = kwargs.get('mutation_method', MutationMethod.GAUSSIAN)
            self.crossover_rate = kwargs.get('crossover_rate', 0.8)
            self.mutation_rate = kwargs.get('mutation_rate', 0.1)
            self.mutation_strength = kwargs.get('mutation_strength', 0.1)
    
    class OperatorFactory:
        @staticmethod
        def create_all_operators(config):
            return {
                "selection": lambda individuals, n: individuals[:n],
                "crossover": lambda p1, p2: (p1, p2),
                "mutation": lambda ind: ind
            }

logger = logging.getLogger(__name__)

@dataclass
class EvolutionConfig:
    """進化アルゴリズム設定"""
    # 基本設定
    population_size: int = 50
    max_generations: int = 100
    elite_size: int = 5
    
    # 遺伝的操作設定
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    crossover_method: CrossoverMethod = CrossoverMethod.UNIFORM
    mutation_method: MutationMethod = MutationMethod.GAUSSIAN
    
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    
    # 停止条件
    target_fitness: Optional[float] = None
    max_runtime_seconds: int = 3600  # 1時間
    convergence_generations: int = 20
    min_improvement: float = 1e-6
    
    # 多様性設定
    diversity_threshold: float = 0.05
    maintain_diversity: bool = True
    
    # 適応設定
    adaptive_parameters: bool = True
    adaptive_interval: int = 10
    
    # 保存設定
    save_interval: int = 10
    save_best_individual: bool = True
    checkpoint_enabled: bool = True
    
    # 並列化設定
    parallel_evaluation: bool = False
    num_processes: int = 4
    
    # ログ設定
    verbose: bool = True
    log_interval: int = 1

@dataclass
class EvolutionResult:
    """進化結果"""
    best_individual: Individual
    best_fitness: float
    final_population: Population
    
    # 実行統計
    total_generations: int
    total_evaluations: int
    execution_time: float
    convergence_generation: int
    
    # 進化履歴
    fitness_history: List[float]
    diversity_history: List[float]
    generation_statistics: List[PopulationStatistics]
    
    # 終了理由
    termination_reason: str
    success: bool

class EvolutionEngine:
    """進化エンジンクラス"""
    
    def __init__(self, config: EvolutionConfig, 
                 individual_type: Type[Individual] = WeightVector):
        
        self.config = config
        self.individual_type = individual_type
        
        # 集団管理
        pop_config = PopulationConfig(
            population_size=config.population_size,
            elite_size=config.elite_size
        )
        self.population = Population(pop_config, individual_type)
        
        # 遺伝的操作
        operator_config = OperatorConfig(
            selection_method=config.selection_method,
            crossover_method=config.crossover_method,
            mutation_method=config.mutation_method,
            crossover_rate=config.crossover_rate,
            mutation_rate=config.mutation_rate,
            mutation_strength=config.mutation_strength
        )
        self.operators = OperatorFactory.create_all_operators(operator_config)
        
        # 進化状態
        self.current_generation = 0
        self.total_evaluations = 0
        self.start_time: Optional[float] = None
        self.best_fitness_history: List[float] = []
        self.average_fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        
        # 停止条件管理
        self.generations_without_improvement = 0
        self.last_best_fitness = -float('inf')
        self.convergence_detected = False
        
        # 適応パラメータ
        self.adaptive_mutation_rate = config.mutation_rate
        self.adaptive_crossover_rate = config.crossover_rate
        
        # 統計情報
        self.evolution_statistics: List[Dict[str, Any]] = []
        
    def initialize_population(self, **kwargs) -> None:
        """集団の初期化"""
        self.population.initialize_random(**kwargs)
        self.current_generation = 0
        self.total_evaluations = 0
        logger.info(f"集団初期化完了: {self.config.population_size}個体")
    
    def evolve(self, fitness_function: Callable[[Individual], float], **kwargs) -> EvolutionResult:
        """進化実行"""
        
        self.start_time = time.time()
        termination_reason = "max_generations_reached"
        
        try:
            # 初期評価
            if not any(getattr(ind, 'fitness_value', None) is not None for ind in self.population.individuals):
                logger.info("初期集団の評価開始...")
                self._evaluate_population(fitness_function)
                self._update_evolution_statistics()
            
            logger.info(f"進化開始: {self.config.max_generations}世代, 集団サイズ{self.config.population_size}")
            
            # 進化ループ
            for generation in range(self.config.max_generations):
                self.current_generation = generation
                
                # 停止条件チェック
                if self._check_termination_conditions():
                    termination_reason = self._get_termination_reason()
                    break
                
                # 世代進化
                self._evolve_generation(fitness_function)
                
                # 統計更新
                self._update_evolution_statistics()
                
                # 適応パラメータ調整
                if self.config.adaptive_parameters and generation % self.config.adaptive_interval == 0:
                    self._adjust_adaptive_parameters()
                
                # 定期保存
                if self.config.checkpoint_enabled and generation % self.config.save_interval == 0:
                    self._save_checkpoint(generation)
                
                # ログ出力
                if self.config.verbose and generation % self.config.log_interval == 0:
                    self._log_generation_info(generation)
            
            # 結果の構築
            result = self._build_evolution_result(termination_reason)
            
            logger.info(f"進化完了: {result.total_generations}世代, 最高適応度{result.best_fitness:.6f}")
            
            return result
            
        except KeyboardInterrupt:
            logger.info("進化が中断されました")
            return self._build_evolution_result("user_interrupted")
        
        except Exception as e:
            logger.error(f"進化中にエラー発生: {e}")
            return self._build_evolution_result("error_occurred")
    
    def _evaluate_population(self, fitness_function: Callable[[Individual], float]) -> None:
        """集団の評価"""
        for individual in self.population.individuals:
            if getattr(individual, 'fitness_value', None) is None:
                try:
                    fitness = fitness_function(individual)
                    individual.fitness_value = fitness
                    self.total_evaluations += 1
                except Exception as e:
                    logger.warning(f"個体評価エラー: {e}")
                    individual.fitness_value = 0.0
    
    def _evolve_generation(self, fitness_function: Callable[[Individual], float]) -> None:
        """1世代の進化"""
        
        # 現在の個体群を適応度順にソート
        self.population.individuals.sort(key=lambda x: getattr(x, 'fitness_value', 0.0), reverse=True)
        
        # エリート保存
        elite_count = min(self.config.elite_size, len(self.population.individuals))
        elite_individuals = self.population.individuals[:elite_count]
        
        # 新世代の生成
        new_population = []
        
        # エリートをそのまま追加
        for elite in elite_individuals:
            new_individual = self._clone_individual(elite)
            new_individual.generation = self.current_generation + 1
            new_population.append(new_individual)
        
        # 残りの個体を生成
        while len(new_population) < self.config.population_size:
            # 親選択（簡易実装：上位個体から選択）
            parent1 = random.choice(self.population.individuals[:max(1, len(self.population.individuals)//2)])
            parent2 = random.choice(self.population.individuals[:max(1, len(self.population.individuals)//2)])
            
            # 交叉（簡易実装）
            child1, child2 = self._crossover(parent1, parent2)
            
            # 変異
            child1 = self._mutate(child1)
            if len(new_population) + 1 < self.config.population_size:
                child2 = self._mutate(child2)
                new_population.extend([child1, child2])
            else:
                new_population.append(child1)
        
        # 新個体の評価
        for individual in new_population[elite_count:]:  # エリート以外を評価
            individual.generation = self.current_generation + 1
            try:
                fitness = fitness_function(individual)
                individual.fitness_value = fitness
                self.total_evaluations += 1
            except Exception as e:
                logger.warning(f"新個体評価エラー: {e}")
                individual.fitness_value = 0.0
        
        # 集団の更新
        self.population.individuals = new_population
    
    def _clone_individual(self, individual: Individual) -> Individual:
        """個体を複製"""
        if hasattr(individual, 'clone'):
            return individual.clone()
        else:
            # 簡易複製
            new_individual = self.individual_type()
            new_individual.fitness_value = individual.fitness_value
            if hasattr(individual, 'get_genes') and hasattr(new_individual, 'set_genes'):
                new_individual.set_genes(individual.get_genes())
            return new_individual
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """交叉操作"""
        if random.random() > self.config.crossover_rate:
            return self._clone_individual(parent1), self._clone_individual(parent2)
        
        # 簡易実装
        if hasattr(parent1, 'crossover'):
            return parent1.crossover(parent2, self.config.crossover_rate)
        else:
            return self._clone_individual(parent1), self._clone_individual(parent2)
    
    def _mutate(self, individual: Individual) -> Individual:
        """変異操作"""
        if hasattr(individual, 'mutate'):
            individual.mutate(self.config.mutation_rate, self.config.mutation_strength)
        individual.fitness_value = None  # 評価をリセット
        return individual
    
    def _check_termination_conditions(self) -> bool:
        """停止条件のチェック"""
        
        # 最大世代数
        if self.current_generation >= self.config.max_generations:
            return True
        
        # 実行時間制限
        if self.start_time and (time.time() - self.start_time) > self.config.max_runtime_seconds:
            return True
        
        # 目標適応度到達
        if self.config.target_fitness and self.best_fitness_history:
            if self.best_fitness_history[-1] >= self.config.target_fitness:
                return True
        
        # 収束判定
        if len(self.best_fitness_history) >= self.config.convergence_generations:
            recent_improvements = []
            for i in range(1, min(self.config.convergence_generations, len(self.best_fitness_history))):
                improvement = self.best_fitness_history[-i] - self.best_fitness_history[-i-1]
                recent_improvements.append(improvement)
            
            if all(imp < self.config.min_improvement for imp in recent_improvements):
                self.convergence_detected = True
                return True
        
        return False
    
    def _get_termination_reason(self) -> str:
        """停止理由を取得"""
        if self.current_generation >= self.config.max_generations:
            return "max_generations_reached"
        elif self.start_time and (time.time() - self.start_time) > self.config.max_runtime_seconds:
            return "time_limit_reached"
        elif self.config.target_fitness and self.best_fitness_history and self.best_fitness_history[-1] >= self.config.target_fitness:
            return "target_fitness_reached"
        elif self.convergence_detected:
            return "convergence_detected"
        else:
            return "unknown"
    
    def _update_evolution_statistics(self) -> None:
        """進化統計の更新"""
        
        if not self.population.individuals:
            return
        
        # 適応度統計
        fitness_values = [getattr(ind, 'fitness_value', 0.0) for ind in self.population.individuals]
        fitness_values = [f for f in fitness_values if f is not None]
        
        if fitness_values:
            best_fitness = max(fitness_values)
            average_fitness = np.mean(fitness_values)
            
            self.best_fitness_history.append(best_fitness)
            self.average_fitness_history.append(average_fitness)
            
            # 改善判定
            if best_fitness > self.last_best_fitness + self.config.min_improvement:
                self.generations_without_improvement = 0
                self.last_best_fitness = best_fitness
            else:
                self.generations_without_improvement += 1
            
            # 多様性計算（簡易版）
            diversity = self._calculate_diversity()
            self.diversity_history.append(diversity)
            
            # 統計情報の記録
            stats = PopulationStatistics()
            stats.generation = self.current_generation
            stats.best_fitness = best_fitness
            stats.average_fitness = average_fitness
            stats.average_diversity = diversity
            
            self.evolution_statistics.append({
                "generation": self.current_generation,
                "best_fitness": best_fitness,
                "average_fitness": average_fitness,
                "diversity": diversity,
                "evaluations": self.total_evaluations
            })
    
    def _calculate_diversity(self) -> float:
        """多様性の計算"""
        if len(self.population.individuals) < 2:
            return 0.0
        
        # 簡易多様性計算（適応度分散）
        fitness_values = [getattr(ind, 'fitness_value', 0.0) for ind in self.population.individuals]
        fitness_values = [f for f in fitness_values if f is not None]
        
        if len(fitness_values) < 2:
            return 0.0
        
        return float(np.std(fitness_values))
    
    def _adjust_adaptive_parameters(self) -> None:
        """適応パラメータの調整"""
        
        # 改善が停滞している場合は変異率を上げる
        if self.generations_without_improvement > 5:
            self.adaptive_mutation_rate = min(0.5, self.adaptive_mutation_rate * 1.1)
        else:
            self.adaptive_mutation_rate = max(0.01, self.adaptive_mutation_rate * 0.95)
        
        # 多様性が低い場合は交叉率を調整
        if self.diversity_history and self.diversity_history[-1] < self.config.diversity_threshold:
            self.adaptive_crossover_rate = min(0.95, self.adaptive_crossover_rate * 1.05)
        else:
            self.adaptive_crossover_rate = max(0.5, self.adaptive_crossover_rate * 0.98)
    
    def _save_checkpoint(self, generation: int) -> None:
        """チェックポイントの保存"""
        
        try:
            checkpoint_dir = "./data/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_data = {
                "generation": generation,
                "config": self.config.__dict__,
                "statistics": self.evolution_statistics,
                "best_fitness_history": self.best_fitness_history,
                "adaptive_parameters": {
                    "mutation_rate": self.adaptive_mutation_rate,
                    "crossover_rate": self.adaptive_crossover_rate
                }
            }
            
            checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_gen{generation}.json")
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.debug(f"チェックポイント保存: {checkpoint_file}")
            
        except Exception as e:
            logger.warning(f"チェックポイント保存エラー: {e}")
    
    def _log_generation_info(self, generation: int) -> None:
        """世代情報のログ出力"""
        
        if not self.best_fitness_history or not self.average_fitness_history:
            return
        
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        log_message = (
            f"世代 {generation:4d}: "
            f"最高適応度 {self.best_fitness_history[-1]:.6f}, "
            f"平均適応度 {self.average_fitness_history[-1]:.6f}, "
            f"多様性 {self.diversity_history[-1]:.4f}, "
            f"停滞世代 {self.generations_without_improvement}, "
            f"経過時間 {elapsed_time:.1f}s"
        )
        
        if self.config.verbose:
            print(log_message)
        
        logger.info(log_message)
    
    def _build_evolution_result(self, termination_reason: str) -> EvolutionResult:
        """進化結果の構築"""
        
        best_individual = self.population.get_best_individual()
        if not best_individual and self.population.individuals:
            # フォールバック：最高適応度の個体を選択
            best_individual = max(self.population.individuals, 
                                key=lambda x: getattr(x, 'fitness_value', 0.0))
        
        best_fitness = getattr(best_individual, 'fitness_value', 0.0) if best_individual else 0.0
        
        execution_time = time.time() - self.start_time if self.start_time else 0.0
        
        # 収束世代の検出
        convergence_generation = self.current_generation
        if len(self.best_fitness_history) > 1:
            for i in range(len(self.best_fitness_history) - 1, 0, -1):
                if (self.best_fitness_history[i] - self.best_fitness_history[i-1]) > self.config.min_improvement:
                    convergence_generation = i
                    break
        
        return EvolutionResult(
            best_individual=best_individual or Individual(),
            best_fitness=best_fitness,
            final_population=self.population,
            total_generations=self.current_generation + 1,
            total_evaluations=self.total_evaluations,
            execution_time=execution_time,
            convergence_generation=convergence_generation,
            fitness_history=self.best_fitness_history.copy(),
            diversity_history=self.diversity_history.copy(),
            generation_statistics=[],  # 簡略化
            termination_reason=termination_reason,
            success=termination_reason in ["target_fitness_reached", "convergence_detected"]
        )

# 使用例とテスト
def test_evolution_engine():
    """進化エンジンのテスト"""
    
    print("🧬 進化エンジンテスト開始")
    
    # 設定の作成
    config = EvolutionConfig(
        population_size=20,
        max_generations=10,
        elite_size=2,
        crossover_rate=0.8,
        mutation_rate=0.1,
        verbose=True
    )
    
    # 進化エンジンの初期化
    engine = EvolutionEngine(config, WeightVector)
    engine.initialize_population(weight_names=["w1", "w2", "w3"])
    
    # 簡易適応度関数（重みの合計値）
    def simple_fitness(individual):
        genes = individual.get_genes()
        return sum(genes.values()) if genes else random.random()
    
    # 進化実行
    result = engine.evolve(simple_fitness)
    
    print(f"\n📊 進化結果:")
    print(f"  最高適応度: {result.best_fitness:.6f}")
    print(f"  総世代数: {result.total_generations}")
    print(f"  実行時間: {result.execution_time:.2f}秒")
    print(f"  終了理由: {result.termination_reason}")
    
    if hasattr(result.best_individual, 'get_genes'):
        print(f"  最良個体の遺伝子: {result.best_individual.get_genes()}")
    
    print("✅ 進化エンジンテスト完了")

if __name__ == "__main__":
    test_evolution_engine()