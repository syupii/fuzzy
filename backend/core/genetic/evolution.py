# core/genetic/evolution.py - 進化アルゴリズム

import numpy as np
import random
import time
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Callable, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

from core.genetic.individual import Individual, WeightVector, FuzzyTreeIndividual
from core.genetic.population import Population, PopulationConfig, PopulationStatistics
from core.genetic.operators import (
    OperatorFactory, OperatorConfig, SelectionMethod, 
    CrossoverMethod, MutationMethod
)

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
            if not any(ind.is_evaluated() for ind in self.population.individuals):
                logger.info("初期集団の評価開始...")
                self.population.evaluate_population(fitness_function)
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
                
                # 多様性維持
                if self.config.maintain_diversity:
                    self.population.maintain_diversity()
                
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
            raise
    
    def _evolve_generation(self, fitness_function: Callable[[Individual], float]) -> None:
        """1世代の進化"""
        
        # エリート保存
        elite_individuals = self.population.select_elite()
        
        # 新世代の生成
        new_population = []
        
        # エリートの追加
        new_population.extend([elite.clone() for elite in elite_individuals])
        
        # 残りの個体を生成
        remaining_size = self.config.population_size - len(elite_individuals)
        
        while len(new_population) < self.config.population_size:
            # 親選択
            parents = self.operators["selection"].apply(self.population.individuals, 2)
            
            if len(parents) >= 2:
                parent1, parent2 = parents[0], parents[1]
                
                # 交叉
                child1, child2 = self.operators["crossover"].apply(parent1, parent2)
                
                # 変異
                child1 = self.operators["mutation"].apply(child1)
                child2 = self.operators["mutation"].apply(child2)
                
                # 世代とID更新
                child1.generation = self.current_generation + 1
                child2.generation = self.current_generation + 1
                child1.individual_id = f"gen{self.current_generation + 1}_ind{len(new_population)}"
                child2.individual_id = f"gen{self.current_generation + 1}_ind{len(new_population) + 1}"
                
                new_population.append(child1)
                if len(new_population) < self.config.population_size:
                    new_population.append(child2)
        
        # 集団の更新
        self.population.individuals = new_population[:self.config.population_size]
        self.population.advance_generation()
        
        # 新個体の評価
        self.population.evaluate_population(fitness_function)
    
    def _check_termination_conditions(self) -> bool:
        """停止条件のチェック"""
        
        # 実行時間チェック
        if self.start_time and time.time() - self.start_time > self.config.max_runtime_seconds:
            return True
        
        # 目標適応度チェック
        if self.config.target_fitness is not None:
            current_best = self.population.get_best_individual()
            if current_best and current_best.get_fitness() >= self.config.target_fitness:
                return True
        
        # 収束チェック
        if self.generations_without_improvement >= self.config.convergence_generations:
            self.convergence_detected = True
            return True
        
        return False
    
    def _get_termination_reason(self) -> str:
        """終了理由の取得"""
        
        if self.start_time and time.time() - self.start_time > self.config.max_runtime_seconds:
            return "timeout"
        
        if self.config.target_fitness is not None:
            current_best = self.population.get_best_individual()
            if current_best and current_best.get_fitness() >= self.config.target_fitness:
                return "target_fitness_reached"
        
        if self.convergence_detected:
            return "convergence_detected"
        
        return "max_generations_reached"
    
    def _update_evolution_statistics(self) -> None:
        """進化統計の更新"""
        
        stats = self.population.get_statistics()
        if not stats:
            return
        
        # 履歴の更新
        self.best_fitness_history.append(stats.best_fitness)
        self.average_fitness_history.append(stats.average_fitness)
        self.diversity_history.append(stats.average_diversity)
        
        # 改善チェック
        if stats.best_fitness > self.last_best_fitness + self.config.min_improvement:
            self.generations_without_improvement = 0
            self.last_best_fitness = stats.best_fitness
        else:
            self.generations_without_improvement += 1
        
        # 詳細統計
        evolution_stat = {
            "generation": self.current_generation,
            "best_fitness": stats.best_fitness,
            "average_fitness": stats.average_fitness,
            "diversity": stats.average_diversity,
            "evaluations": self.total_evaluations,
            "convergence_indicator": stats.convergence_indicator,
            "improvement_rate": stats.improvement_rate,
            "generations_without_improvement": self.generations_without_improvement
        }
        
        self.evolution_statistics.append(evolution_stat)
    
    def _adjust_adaptive_parameters(self) -> None:
        """適応パラメータの調整"""
        
        if not self.config.adaptive_parameters:
            return
        
        # 収束傾向の場合は変異率を上げる
        if self.generations_without_improvement > 5:
            self.adaptive_mutation_rate = min(0.3, self.adaptive_mutation_rate * 1.1)
            
            # 変異操作の更新
            if "mutation" in self.operators:
                self.operators["mutation"].config.mutation_rate = self.adaptive_mutation_rate
        
        # 改善がある場合は元に戻す
        else:
            self.adaptive_mutation_rate = self.config.mutation_rate
            if "mutation" in self.operators:
                self.operators["mutation"].config.mutation_rate = self.adaptive_mutation_rate
        
        # 多様性が低い場合は交叉率を調整
        current_diversity = self.diversity_history[-1] if self.diversity_history else 0.0
        if current_diversity < self.config.diversity_threshold:
            self.adaptive_crossover_rate = max(0.5, self.adaptive_crossover_rate * 0.95)
        else:
            self.adaptive_crossover_rate = self.config.crossover_rate
        
        if "crossover" in self.operators:
            self.operators["crossover"].config.crossover_rate = self.adaptive_crossover_rate
    
    def _save_checkpoint(self, generation: int) -> None:
        """チェックポイント保存"""
        
        try:
            checkpoint_dir = "./data/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_data = {
                "generation": generation,
                "config": self.config,
                "population": self.population.save_population(),
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
        
        stats = self.population.get_statistics()
        if not stats:
            return
        
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        log_message = (
            f"世代 {generation:4d}: "
            f"最高適応度 {stats.best_fitness:.6f}, "
            f"平均適応度 {stats.average_fitness:.6f}, "
            f"多様性 {stats.average_diversity:.4f}, "
            f"収束指標 {stats.convergence_indicator:.4f}, "
            f"停滞世代 {self.generations_without_improvement}, "
            f"経過時間 {elapsed_time:.1f}s"
        )
        
        if self.config.verbose:
            print(log_message)
        
        logger.info(log_message)
    
    def _build_evolution_result(self, termination_reason: str) -> EvolutionResult:
        """進化結果の構築"""
        
        best_individual = self.population.get_best_individual()
        best_fitness = best_individual.get_fitness() if best_individual else 0.0
        
        execution_time = time.time() - self.start_time if self.start_time else 0.0
        
        # 収束世代の検出
        convergence_generation = self.current_generation
        if len(self.best_fitness_history) > 1:
            for i in range(len(self.best_fitness_history) - 1, 0, -1):
                if (self.best_fitness_history[i] - self.best_fitness_history[i-1]) > self.config.min_improvement:
                    convergence_generation = i
                    break
        
        return EvolutionResult(
            best_individual=best_individual,
            best_fitness=best_fitness,
            final_population=self.population,
            total_generations=self.current_generation + 1,
            total_evaluations=self.population.total_evaluations,
            execution_time=execution_time,
            convergence_generation=convergence_generation,
            fitness_history=self.best_fitness_history.copy(),
            diversity_history=self.diversity_history.copy(),
            generation_statistics=self.population.generation_history.copy(),
            termination_reason=termination_reason,
            success=termination_reason in ["target_fitness_reached", "convergence_detected"]
        )
    
    def load_checkpoint(self, checkpoint_file: str) -> None:
        """チェックポイントの読み込み"""
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            self.current_generation = checkpoint_data["generation"]
            self.evolution_statistics = checkpoint_data["statistics"]
            self.best_fitness_history = checkpoint_data["best_fitness_history"]
            
            # 適応パラメータの復元
            adaptive_params = checkpoint_data.get("adaptive_parameters", {})
            self.adaptive_mutation_rate = adaptive_params.get("mutation_rate", self.config.mutation_rate)
            self.adaptive_crossover_rate = adaptive_params.get("crossover_rate", self.config.crossover_rate)
            
            # 集団の復元
            population_file = checkpoint_data["population"]
            self.population.load_population(population_file)
            
            logger.info(f"チェックポイント読み込み完了: 世代{self.current_generation}")
            
        except Exception as e:
            logger.error(f"チェックポイント読み込みエラー: {e}")
            raise

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
        return sum(genes.values())
    
    # 進化実行
    result = engine.evolve(simple_fitness)
    
    print(f"\n📊 進化結果:")
    print(f"  最高適応度: {result.best_fitness:.6f}")
    print(f"  総世代数: {result.total_generations}")
    print(f"  実行時間: {result.execution_time:.2f}秒")
    print(f"  終了理由: {result.termination_reason}")
    print(f"  最良個体の遺伝子: {result.best_individual.get_genes()}")
    
    print("✅ 進化エンジンテスト完了")

if __name__ == "__main__":
    test_evolution_engine()