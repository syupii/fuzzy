# backend/optimization_tracker.py
import json
import time
import pickle
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import threading
import queue


@dataclass
class GenerationStats:
    """世代統計"""
    generation: int
    timestamp: datetime

    # 適応度統計
    best_fitness: float = 0.0
    worst_fitness: float = 0.0
    avg_fitness: float = 0.0
    median_fitness: float = 0.0
    std_fitness: float = 0.0

    # 多目的統計
    best_accuracy: float = 0.0
    best_simplicity: float = 0.0
    best_interpretability: float = 0.0
    best_generalization: float = 0.0
    best_validity: float = 0.0

    avg_accuracy: float = 0.0
    avg_simplicity: float = 0.0
    avg_interpretability: float = 0.0
    avg_generalization: float = 0.0
    avg_validity: float = 0.0

    # 多様性統計
    diversity_score: float = 0.0
    unique_individuals: int = 0

    # 進化統計
    mutation_success_rate: float = 0.0
    crossover_success_rate: float = 0.0

    # 性能統計
    evaluation_time: float = 0.0
    memory_usage: float = 0.0


@dataclass
class IndividualRecord:
    """個体記録"""
    individual_id: str
    generation: int
    timestamp: datetime

    # 遺伝子情報
    genome: Dict[str, Any] = field(default_factory=dict)

    # 適応度情報
    fitness_components: Dict[str, float] = field(default_factory=dict)
    overall_fitness: float = 0.0

    # 系譜情報
    parents: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)

    # 性能情報
    prediction_accuracy: float = 0.0
    model_complexity: int = 0
    evaluation_time: float = 0.0

    # メタデータ
    notes: str = ""
    tags: List[str] = field(default_factory=list)


class OptimizationTracker:
    """最適化プロセス追跡システム"""

    def __init__(self, run_id: str = None, save_dir: str = "optimization_logs"):
        self.run_id = run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = save_dir

        # ディレクトリ作成
        os.makedirs(save_dir, exist_ok=True)

        # データ構造
        self.generation_stats: List[GenerationStats] = []
        self.individual_records: Dict[str, IndividualRecord] = {}
        self.best_individuals_history: List[str] = []

        # 実行情報
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.current_generation: int = 0
        self.is_running: bool = False

        # 設定情報
        self.config: Dict[str, Any] = {}

        # リアルタイム監視
        self.monitoring_queue: queue.Queue = queue.Queue()
        self.monitoring_thread: Optional[threading.Thread] = None
        self.enable_realtime_monitoring: bool = True

        # パレート最適解追跡
        self.pareto_front_history: List[List[str]] = []

        print(f"🎯 OptimizationTracker initialized: {self.run_id}")

    def start_optimization(self, config: Dict[str, Any]):
        """最適化開始"""
        self.start_time = datetime.now()
        self.is_running = True
        self.config = config.copy()
        self.current_generation = 0

        # 設定保存
        self._save_config()

        # リアルタイム監視開始
        if self.enable_realtime_monitoring:
            self._start_monitoring_thread()

        print(f"🚀 Optimization started at {self.start_time}")
        print(f"📁 Logs saved to: {self.save_dir}")

    def end_optimization(self):
        """最適化終了"""
        self.end_time = datetime.now()
        self.is_running = False

        # リアルタイム監視停止
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_queue.put(('STOP', None))
            self.monitoring_thread.join(timeout=5.0)

        # 最終レポート生成
        self._generate_final_report()

        duration = self.end_time - self.start_time if self.start_time else None
        print(f"🏁 Optimization completed at {self.end_time}")
        if duration:
            print(f"⏱️ Total duration: {duration}")

    def record_generation(self, generation: int, population: List[Any],
                          evaluation_results: List[Dict[str, Any]],
                          generation_metrics: Dict[str, Any] = None):
        """世代記録"""

        start_time = time.time()

        # 適応度抽出
        fitness_values = []
        accuracy_values = []
        simplicity_values = []
        interpretability_values = []
        generalization_values = []
        validity_values = []

        for result in evaluation_results:
            if 'overall_fitness' in result:
                fitness_values.append(result['overall_fitness'])
            if 'accuracy' in result:
                accuracy_values.append(result['accuracy'])
            if 'simplicity' in result:
                simplicity_values.append(result['simplicity'])
            if 'interpretability' in result:
                interpretability_values.append(result['interpretability'])
            if 'generalization' in result:
                generalization_values.append(result['generalization'])
            if 'validity' in result:
                validity_values.append(result['validity'])

        # 統計計算
        fitness_stats = self._calculate_stats(fitness_values)
        accuracy_stats = self._calculate_stats(accuracy_values)
        simplicity_stats = self._calculate_stats(simplicity_values)
        interpretability_stats = self._calculate_stats(interpretability_values)
        generalization_stats = self._calculate_stats(generalization_values)
        validity_stats = self._calculate_stats(validity_values)

        # 多様性計算
        diversity_score = self._calculate_diversity(population)

        # 世代統計作成
        stats = GenerationStats(
            generation=generation,
            timestamp=datetime.now(),
            best_fitness=fitness_stats['max'],
            worst_fitness=fitness_stats['min'],
            avg_fitness=fitness_stats['mean'],
            median_fitness=fitness_stats['median'],
            std_fitness=fitness_stats['std'],
            best_accuracy=accuracy_stats['max'],
            best_simplicity=simplicity_stats['max'],
            best_interpretability=interpretability_stats['max'],
            best_generalization=generalization_stats['max'],
            best_validity=validity_stats['max'],
            avg_accuracy=accuracy_stats['mean'],
            avg_simplicity=simplicity_stats['mean'],
            avg_interpretability=interpretability_stats['mean'],
            avg_generalization=generalization_stats['mean'],
            avg_validity=validity_stats['mean'],
            diversity_score=diversity_score,
            unique_individuals=len(set(id(ind) for ind in population)),
            evaluation_time=time.time() - start_time,
            memory_usage=self._get_memory_usage()
        )

        self.generation_stats.append(stats)
        self.current_generation = generation

        # 最良個体記録
        if fitness_values:
            best_idx = fitness_values.index(max(fitness_values))
            best_individual = population[best_idx]
            individual_id = f"gen{generation}_ind{best_idx}"
            self.best_individuals_history.append(individual_id)

            # 個体記録
            self.record_individual(
                individual=best_individual,
                individual_id=individual_id,
                generation=generation,
                fitness_components=evaluation_results[best_idx] if best_idx < len(
                    evaluation_results) else {},
                overall_fitness=fitness_values[best_idx]
            )

        # パレート最適解計算
        pareto_front = self._calculate_pareto_front(evaluation_results)
        self.pareto_front_history.append(pareto_front)

        # リアルタイム監視
        if self.enable_realtime_monitoring:
            self.monitoring_queue.put(('GENERATION', stats))

        print(f"📊 Generation {generation}: Best={stats.best_fitness:.4f}, "
              f"Avg={stats.avg_fitness:.4f}, Diversity={stats.diversity_score:.3f}")

        return stats

    def record_individual(self, individual: Any, individual_id: str, generation: int,
                          fitness_components: Dict[str, float], overall_fitness: float,
                          parents: List[str] = None, notes: str = ""):
        """個体記録"""

        record = IndividualRecord(
            individual_id=individual_id,
            generation=generation,
            timestamp=datetime.now(),
            fitness_components=fitness_components,
            overall_fitness=overall_fitness,
            parents=parents or [],
            notes=notes
        )

        # ゲノム情報（可能な場合）
        if hasattr(individual, 'genome'):
            record.genome = individual.genome
        elif hasattr(individual, '__dict__'):
            record.genome = {k: str(v) for k, v in individual.__dict__.items()}

        # 複雑度情報
        if hasattr(individual, 'calculate_complexity'):
            record.model_complexity = individual.calculate_complexity()

        self.individual_records[individual_id] = record

    def generate_performance_plots(self) -> List[str]:
        """性能プロット生成"""

        plot_files = []

        if not self.generation_stats:
            return plot_files

        # データ準備
        generations = [stats.generation for stats in self.generation_stats]
        best_fitness = [stats.best_fitness for stats in self.generation_stats]
        avg_fitness = [stats.avg_fitness for stats in self.generation_stats]
        diversity = [stats.diversity_score for stats in self.generation_stats]

        # プロット1: 適応度推移
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(generations, best_fitness, 'b-',
                 linewidth=2, label='Best Fitness')
        plt.plot(generations, avg_fitness, 'g--',
                 linewidth=1, label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # プロット2: 多様性推移
        plt.subplot(2, 2, 2)
        plt.plot(generations, diversity, 'r-', linewidth=2, label='Diversity')
        plt.xlabel('Generation')
        plt.ylabel('Diversity Score')
        plt.title('Population Diversity')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # プロット3: 多目的統計
        plt.subplot(2, 2, 3)
        accuracy = [stats.best_accuracy for stats in self.generation_stats]
        simplicity = [stats.best_simplicity for stats in self.generation_stats]
        interpretability = [
            stats.best_interpretability for stats in self.generation_stats]

        plt.plot(generations, accuracy, label='Accuracy', linewidth=1)
        plt.plot(generations, simplicity, label='Simplicity', linewidth=1)
        plt.plot(generations, interpretability,
                 label='Interpretability', linewidth=1)
        plt.xlabel('Generation')
        plt.ylabel('Score')
        plt.title('Multi-objective Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # プロット4: 評価時間
        plt.subplot(2, 2, 4)
        eval_times = [stats.evaluation_time for stats in self.generation_stats]
        plt.plot(generations, eval_times, 'm-',
                 linewidth=2, label='Evaluation Time')
        plt.xlabel('Generation')
        plt.ylabel('Time (seconds)')
        plt.title('Evaluation Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存
        fitness_plot_path = os.path.join(
            self.save_dir, f"{self.run_id}_fitness_evolution.png")
        plt.savefig(fitness_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        plot_files.append(fitness_plot_path)

        # プロット2: パレート最適解推移
        if self.pareto_front_history:
            plt.figure(figsize=(10, 6))
            pareto_sizes = [len(front) for front in self.pareto_front_history]
            plt.plot(generations[:len(pareto_sizes)],
                     pareto_sizes, 'o-', linewidth=2)
            plt.xlabel('Generation')
            plt.ylabel('Pareto Front Size')
            plt.title('Pareto Front Evolution')
            plt.grid(True, alpha=0.3)

            pareto_plot_path = os.path.join(
                self.save_dir, f"{self.run_id}_pareto_evolution.png")
            plt.savefig(pareto_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_files.append(pareto_plot_path)

        # プロット3: ヒートマップ（世代×目的関数）
        if len(self.generation_stats) > 1:
            objectives = ['best_accuracy', 'best_simplicity', 'best_interpretability',
                          'best_generalization', 'best_validity']

            heatmap_data = []
            for stats in self.generation_stats:
                row = [getattr(stats, obj, 0.0) for obj in objectives]
                heatmap_data.append(row)

            plt.figure(figsize=(10, 8))
            sns.heatmap(np.array(heatmap_data).T,
                        xticklabels=[f"Gen {i}" for i in generations],
                        yticklabels=[
                            'Accuracy', 'Simplicity', 'Interpretability', 'Generalization', 'Validity'],
                        cmap='viridis', annot=False, cbar=True)
            plt.title('Multi-objective Evolution Heatmap')
            plt.xlabel('Generation')
            plt.ylabel('Objective')

            heatmap_plot_path = os.path.join(
                self.save_dir, f"{self.run_id}_objectives_heatmap.png")
            plt.savefig(heatmap_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plot_files.append(heatmap_plot_path)

        print(f"📊 Performance plots generated: {len(plot_files)} files")
        return plot_files

    def get_convergence_analysis(self) -> Dict[str, Any]:
        """収束分析"""

        if len(self.generation_stats) < 3:
            return {
                'convergence_detected': False,
                'stagnation_generations': 0,
                'max_fitness_achieved': 0.0,
                'convergence_rate': 0.0
            }

        fitness_values = [
            stats.best_fitness for stats in self.generation_stats]

        # 停滞世代数計算
        stagnation_threshold = 1e-6
        stagnation_count = 0
        max_stagnation = 0

        for i in range(1, len(fitness_values)):
            if abs(fitness_values[i] - fitness_values[i-1]) < stagnation_threshold:
                stagnation_count += 1
            else:
                max_stagnation = max(max_stagnation, stagnation_count)
                stagnation_count = 0

        max_stagnation = max(max_stagnation, stagnation_count)

        # 収束判定
        convergence_detected = max_stagnation >= 5  # 5世代連続停滞で収束判定

        # 収束率計算
        convergence_rate = 0.0
        if len(fitness_values) > 1:
            total_improvement = fitness_values[-1] - fitness_values[0]
            convergence_rate = total_improvement / len(fitness_values)

        return {
            'convergence_detected': bool(convergence_detected),
            'stagnation_generations': int(max_stagnation),
            'max_fitness_achieved': float(max(fitness_values)),
            'convergence_rate': float(convergence_rate),
            'final_fitness': float(fitness_values[-1]),
            'initial_fitness': float(fitness_values[0]),
            'total_improvement': float(total_improvement) if len(fitness_values) > 1 else 0.0
        }

    def get_diversity_analysis(self) -> Dict[str, Any]:
        """多様性分析"""

        if not self.generation_stats:
            return {
                'final_diversity': 0.0,
                'max_diversity': 0.0,
                'min_diversity': 0.0,
                'avg_diversity': 0.0,
                'diversity_trend': 'unknown'
            }

        diversity_values = [
            stats.diversity_score for stats in self.generation_stats]

        # 多様性傾向分析
        if len(diversity_values) > 1:
            slope = (diversity_values[-1] -
                     diversity_values[0]) / len(diversity_values)
            if slope > 0.01:
                trend = 'increasing'
            elif slope < -0.01:
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'unknown'

        return {
            'final_diversity': float(diversity_values[-1]),
            'max_diversity': float(max(diversity_values)),
            'min_diversity': float(min(diversity_values)),
            'avg_diversity': float(sum(diversity_values) / len(diversity_values)),
            'diversity_trend': trend,
            'diversity_variance': float(np.var(diversity_values)) if len(diversity_values) > 1 else 0.0
        }

    def get_best_individual_history(self) -> List[Dict[str, Any]]:
        """最良個体履歴取得"""

        history = []
        for individual_id in self.best_individuals_history:
            if individual_id in self.individual_records:
                record = self.individual_records[individual_id]
                history.append({
                    'generation': int(record.generation),
                    'individual_id': individual_id,
                    'fitness': float(record.overall_fitness),
                    'fitness_components': record.fitness_components,
                    'complexity': int(record.model_complexity)
                })

        return history

    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """統計値計算"""

        if not values:
            return {
                'min': 0.0,
                'max': 0.0,
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0
            }

        values = np.array(values)
        return {
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'mean': float(np.mean(values)),
            'median': float(np.median(values)),
            'std': float(np.std(values))
        }

    def _calculate_diversity(self, population: List[Any]) -> float:
        """個体群多様性計算"""

        if len(population) <= 1:
            return 0.0

        # 遺伝的距離ベース多様性計算（簡易版）
        try:
            distances = []
            for i in range(len(population)):
                for j in range(i + 1, len(population)):
                    distance = self._genetic_distance(
                        population[i], population[j])
                    distances.append(distance)

            return float(np.mean(distances)) if distances else 0.0

        except Exception:
            # フォールバック: ユニーク個体数ベース
            unique_count = len(set(str(ind) for ind in population))
            return float(unique_count / len(population))

    def _genetic_distance(self, ind1: Any, ind2: Any) -> float:
        """2個体間の遺伝的距離"""

        try:
            # ゲノムベース距離
            if hasattr(ind1, 'genome') and hasattr(ind2, 'genome'):
                genome1 = ind1.genome
                genome2 = ind2.genome

                if isinstance(genome1, dict) and isinstance(genome2, dict):
                    common_keys = set(genome1.keys()) & set(genome2.keys())
                    if common_keys:
                        diffs = []
                        for key in common_keys:
                            try:
                                diff = abs(
                                    float(genome1[key]) - float(genome2[key]))
                                diffs.append(diff)
                            except (ValueError, TypeError):
                                # 文字列比較
                                diff = 0.0 if genome1[key] == genome2[key] else 1.0
                                diffs.append(diff)
                        return float(np.mean(diffs)) if diffs else 1.0

            # 適応度ベース距離
            if hasattr(ind1, 'fitness') and hasattr(ind2, 'fitness'):
                if hasattr(ind1.fitness, 'values') and hasattr(ind2.fitness, 'values'):
                    values1 = ind1.fitness.values
                    values2 = ind2.fitness.values
                    return float(np.linalg.norm(np.array(values1) - np.array(values2)))

            # デフォルト
            return 1.0

        except Exception:
            return 1.0

    def _calculate_pareto_front(self, evaluation_results: List[Dict[str, Any]]) -> List[str]:
        """パレート最適解計算"""

        if not evaluation_results:
            return []

        pareto_front = []
        for i, result_i in enumerate(evaluation_results):
            individual_id_i = f"ind_{i}"
            is_dominated = False

            for j, result_j in enumerate(evaluation_results):
                if i != j and self._dominates(result_j, result_i):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append(individual_id_i)

        return pareto_front

    def _dominates(self, components_a: Dict[str, float], components_b: Dict[str, float]) -> bool:
        """支配関係チェック"""

        objectives = ['accuracy', 'simplicity',
                      'interpretability', 'generalization', 'validity']

        all_better_or_equal = True
        at_least_one_better = False

        for obj in objectives:
            val_a = components_a.get(obj, 0.0)
            val_b = components_b.get(obj, 0.0)

            if val_a < val_b:
                all_better_or_equal = False
                break
            elif val_a > val_b:
                at_least_one_better = True

        return all_better_or_equal and at_least_one_better

    def _get_memory_usage(self) -> float:
        """メモリ使用量取得（MB）"""

        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # MB
        except:
            return 0.0

    def _save_config(self):
        """設定保存"""

        config_path = os.path.join(self.save_dir, f"{self.run_id}_config.json")

        def custom_json_serializer(obj):
            """カスタムJSONシリアライザー"""
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return str(obj)

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2,
                      ensure_ascii=False, default=custom_json_serializer)

    def _save_intermediate_results(self):
        """中間結果保存"""

        def custom_json_serializer(obj):
            """カスタムJSONシリアライザー"""
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return str(obj)

        # 世代統計保存
        stats_path = os.path.join(
            self.save_dir, f"{self.run_id}_generation_stats.json")

        stats_data = []
        for stats in self.generation_stats:
            stats_dict = asdict(stats)
            stats_dict['timestamp'] = stats.timestamp.isoformat()
            stats_data.append(stats_dict)

        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False,
                      default=custom_json_serializer)

        # 個体記録保存
        individuals_path = os.path.join(
            self.save_dir, f"{self.run_id}_individuals.json")

        individuals_data = {}
        for individual_id, record in self.individual_records.items():
            record_dict = asdict(record)
            record_dict['timestamp'] = record.timestamp.isoformat()
            individuals_data[individual_id] = record_dict

        with open(individuals_path, 'w', encoding='utf-8') as f:
            json.dump(individuals_data, f, indent=2,
                      ensure_ascii=False, default=custom_json_serializer)

    def _generate_final_report(self):
        """最終レポート生成（JSON安全版）"""

        def custom_json_serializer(obj):
            """カスタムJSONシリアライザー"""
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.integer, int)):
                return int(obj)
            elif isinstance(obj, (np.floating, float)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return str(obj)

        def make_json_safe(data):
            """データをJSON安全にする再帰関数"""
            if isinstance(data, dict):
                return {k: make_json_safe(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [make_json_safe(item) for item in data]
            elif isinstance(data, tuple):
                return [make_json_safe(item) for item in data]
            elif isinstance(data, (np.bool_, bool)):
                return bool(data)
            elif isinstance(data, (np.integer, int)):
                return int(data)
            elif isinstance(data, (np.floating, float)):
                return float(data)
            elif isinstance(data, np.ndarray):
                return data.tolist()
            elif isinstance(data, datetime):
                return data.isoformat()
            elif hasattr(data, '__dict__'):
                return make_json_safe(data.__dict__)
            else:
                return str(data) if data is not None else None

        # 全データ保存
        self._save_intermediate_results()

        # 性能プロット生成
        plot_files = self.generate_performance_plots()

        # 収束分析
        convergence_analysis = self.get_convergence_analysis()

        # 多様性分析
        diversity_analysis = self.get_diversity_analysis()

        # 最良個体履歴
        best_history = self.get_best_individual_history()

        # 最終レポート
        report = {
            'run_info': {
                'run_id': self.run_id,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'end_time': self.end_time.isoformat() if self.end_time else None,
                'duration_seconds': (self.end_time - self.start_time).total_seconds() if self.start_time and self.end_time else None,
                'total_generations': len(self.generation_stats),
                'total_individuals': len(self.individual_records)
            },
            'configuration': self.config,
            'convergence_analysis': convergence_analysis,
            'diversity_analysis': diversity_analysis,
            'best_individual_history': best_history,
            'final_statistics': {
                'best_fitness': float(self.generation_stats[-1].best_fitness) if self.generation_stats else 0.0,
                'final_diversity': float(self.generation_stats[-1].diversity_score) if self.generation_stats else 0.0,
                'pareto_front_size': len(self.pareto_front_history[-1]) if self.pareto_front_history else 0
            },
            'plot_files': plot_files
        }

        # データをJSON安全にする
        safe_report = make_json_safe(report)

        # レポート保存
        report_path = os.path.join(
            self.save_dir, f"{self.run_id}_final_report.json")

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(safe_report, f, indent=2, ensure_ascii=False,
                          default=custom_json_serializer)

            print(f"📊 Final report saved: {report_path}")
            return safe_report

        except Exception as e:
            print(f"⚠️ JSON保存エラー: {e}")
            # テキスト形式でバックアップ保存
            text_report_path = report_path.replace('.json', '_backup.txt')
            with open(text_report_path, 'w', encoding='utf-8') as f:
                f.write(str(safe_report))
            print(f"📝 テキスト形式でバックアップ保存: {text_report_path}")
            return safe_report

    def _start_monitoring_thread(self):
        """リアルタイム監視スレッド開始"""

        def monitoring_worker():
            while True:
                try:
                    message_type, data = self.monitoring_queue.get(timeout=1.0)

                    if message_type == 'STOP':
                        break
                    elif message_type == 'GENERATION':
                        self._handle_realtime_generation_update(data)

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"⚠️ Monitoring error: {e}")

        self.monitoring_thread = threading.Thread(
            target=monitoring_worker, daemon=True)
        self.monitoring_thread.start()

    def _handle_realtime_generation_update(self, stats: GenerationStats):
        """リアルタイム世代更新処理"""

        # 簡易的なリアルタイム出力
        print(f"⚡ Real-time update - Gen {stats.generation}: "
              f"Best={stats.best_fitness:.4f}, "
              f"Diversity={stats.diversity_score:.3f}, "
              f"Time={stats.evaluation_time:.2f}s")

        # 必要に応じて外部システムへの通知などを追加可能


class OptimizationReporter:
    """最適化結果レポート生成"""

    @staticmethod
    def generate_html_report(tracker: OptimizationTracker) -> str:
        """HTML形式レポート生成"""

        convergence_analysis = tracker.get_convergence_analysis()
        diversity_analysis = tracker.get_diversity_analysis()

        html_content = f"""
        <!DOCTYPE html>
        <html lang="ja">
        <head>
            <meta charset="UTF-8">
            <title>Optimization Report - {tracker.run_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 8px; }}
                .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #007acc; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #f9f9f9; border-radius: 4px; }}
                .chart {{ text-align: center; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Fuzzy Decision Tree Optimization Report</h1>
                <p>Run ID: {tracker.run_id}</p>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Execution Summary</h2>
                <div class="metric">
                    <strong>Generations:</strong> {len(tracker.generation_stats)}
                </div>
                <div class="metric">
                    <strong>Best Fitness:</strong> {tracker.generation_stats[-1].best_fitness:.4f if tracker.generation_stats else 'N/A'}
                </div>
                <div class="metric">
                    <strong>Final Diversity:</strong> {tracker.generation_stats[-1].diversity_score:.3f if tracker.generation_stats else 'N/A'}
                </div>
            </div>
            
            <div class="section">
                <h2>Convergence Analysis</h2>
                <div class="metric">
                    <strong>Convergence Detected:</strong> {'Yes' if convergence_analysis.get('convergence_detected', False) else 'No'}
                </div>
                <div class="metric">
                    <strong>Stagnation Generations:</strong> {convergence_analysis.get('stagnation_generations', 0)}
                </div>
                <div class="metric">
                    <strong>Max Fitness:</strong> {convergence_analysis.get('max_fitness_achieved', 0.0):.4f}
                </div>
            </div>
            
            <div class="section">
                <h2>Diversity Analysis</h2>
                <div class="metric">
                    <strong>Final Diversity:</strong> {diversity_analysis.get('final_diversity', 0.0):.3f}
                </div>
                <div class="metric">
                    <strong>Max Diversity:</strong> {diversity_analysis.get('max_diversity', 0.0):.3f}
                </div>
                <div class="metric">
                    <strong>Trend:</strong> {diversity_analysis.get('diversity_trend', 'Unknown')}
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Charts</h2>
                <div class="chart">
                    <p>Performance visualization charts have been generated.</p>
                    <p>Check the optimization logs directory for detailed plots.</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Best Individual History</h2>
                <table border="1" style="border-collapse: collapse; width: 100%;">
                    <thead>
                        <tr>
                            <th>Generation</th>
                            <th>Fitness</th>
                            <th>Complexity</th>
                            <th>Individual ID</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        # 最良個体履歴テーブル
        best_history = tracker.get_best_individual_history()
        for entry in best_history[-10:]:  # 最新10世代のみ表示
            html_content += f"""
                        <tr>
                            <td>{entry['generation']}</td>
                            <td>{entry['fitness']:.4f}</td>
                            <td>{entry['complexity']}</td>
                            <td>{entry['individual_id']}</td>
                        </tr>
            """

        html_content += """
                    </tbody>
                </table>
            </div>
        </body>
        </html>
        """

        # HTMLファイル保存
        html_path = os.path.join(
            tracker.save_dir, f"{tracker.run_id}_report.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"📊 HTML report generated: {html_path}")
        return html_path

    @staticmethod
    def export_results_to_csv(tracker: OptimizationTracker) -> List[str]:
        """CSV形式でのデータエクスポート"""

        csv_files = []

        # 世代統計CSV
        if tracker.generation_stats:
            import pandas as pd

            stats_data = []
            for stats in tracker.generation_stats:
                stats_dict = asdict(stats)
                stats_dict['timestamp'] = stats.timestamp.isoformat()
                stats_data.append(stats_dict)

            df_stats = pd.DataFrame(stats_data)
            stats_csv_path = os.path.join(
                tracker.save_dir, f"{tracker.run_id}_generation_stats.csv")
            df_stats.to_csv(stats_csv_path, index=False, encoding='utf-8')
            csv_files.append(stats_csv_path)

        # 個体記録CSV
        if tracker.individual_records:
            individual_data = []
            for individual_id, record in tracker.individual_records.items():
                record_dict = asdict(record)
                record_dict['individual_id'] = individual_id
                record_dict['timestamp'] = record.timestamp.isoformat()

                # fitness_componentsを展開
                for component, value in record.fitness_components.items():
                    record_dict[f'fitness_{component}'] = value

                individual_data.append(record_dict)

            df_individuals = pd.DataFrame(individual_data)
            individuals_csv_path = os.path.join(
                tracker.save_dir, f"{tracker.run_id}_individuals.csv")
            df_individuals.to_csv(individuals_csv_path,
                                  index=False, encoding='utf-8')
            csv_files.append(individuals_csv_path)

        print(f"📁 CSV files exported: {len(csv_files)} files")
        return csv_files

    @staticmethod
    def generate_summary_statistics(tracker: OptimizationTracker) -> Dict[str, Any]:
        """要約統計生成"""

        if not tracker.generation_stats:
            return {'error': 'No generation statistics available'}

        fitness_values = [
            stats.best_fitness for stats in tracker.generation_stats]
        diversity_values = [
            stats.diversity_score for stats in tracker.generation_stats]

        # 基本統計
        summary = {
            'run_info': {
                'run_id': tracker.run_id,
                'total_generations': len(tracker.generation_stats),
                'total_individuals_evaluated': len(tracker.individual_records),
                'duration_seconds': (tracker.end_time - tracker.start_time).total_seconds() if tracker.start_time and tracker.end_time else None
            },
            'fitness_statistics': {
                'initial_best': float(fitness_values[0]) if fitness_values else 0.0,
                'final_best': float(fitness_values[-1]) if fitness_values else 0.0,
                'max_achieved': float(max(fitness_values)) if fitness_values else 0.0,
                'improvement': float(fitness_values[-1] - fitness_values[0]) if len(fitness_values) > 1 else 0.0,
                'improvement_percentage': float((fitness_values[-1] - fitness_values[0]) / max(fitness_values[0], 1e-8) * 100) if len(fitness_values) > 1 else 0.0
            },
            'diversity_statistics': {
                'initial_diversity': float(diversity_values[0]) if diversity_values else 0.0,
                'final_diversity': float(diversity_values[-1]) if diversity_values else 0.0,
                'max_diversity': float(max(diversity_values)) if diversity_values else 0.0,
                'min_diversity': float(min(diversity_values)) if diversity_values else 0.0,
                'avg_diversity': float(sum(diversity_values) / len(diversity_values)) if diversity_values else 0.0
            },
            'convergence_info': tracker.get_convergence_analysis(),
            'performance_metrics': {
                'avg_evaluation_time': float(sum(stats.evaluation_time for stats in tracker.generation_stats) / len(tracker.generation_stats)),
                'total_evaluation_time': float(sum(stats.evaluation_time for stats in tracker.generation_stats)),
                'avg_memory_usage': float(sum(stats.memory_usage for stats in tracker.generation_stats) / len(tracker.generation_stats)),
                'max_memory_usage': float(max(stats.memory_usage for stats in tracker.generation_stats))
            }
        }

        return summary


# ユーティリティ関数
def create_optimization_tracker(run_id: str = None, save_dir: str = "optimization_logs") -> OptimizationTracker:
    """最適化トラッカー作成"""
    return OptimizationTracker(run_id=run_id, save_dir=save_dir)


def generate_optimization_report(tracker: OptimizationTracker, format: str = 'html') -> str:
    """最適化レポート生成"""

    if format.lower() == 'html':
        return OptimizationReporter.generate_html_report(tracker)
    elif format.lower() == 'csv':
        files = OptimizationReporter.export_results_to_csv(tracker)
        return f"CSV files exported: {', '.join(files)}"
    else:
        raise ValueError(f"Unsupported format: {format}")


if __name__ == "__main__":
    # デモンストレーション用テストコード
    print("🧬 OptimizationTracker Demo")

    # トラッカー作成
    tracker = OptimizationTracker("demo_test")

    # 設定例
    config = {
        'population_size': 30,
        'generations': 10,
        'mutation_rate': 0.15,
        'crossover_rate': 0.8
    }

    # 最適化開始
    tracker.start_optimization(config)

    # ダミーデータで世代記録
    import random
    for gen in range(5):
        population = [f"individual_{i}" for i in range(30)]
        evaluation_results = [
            {
                'overall_fitness': random.uniform(0.5, 0.9),
                'accuracy': random.uniform(0.6, 0.9),
                'simplicity': random.uniform(0.4, 0.8),
                'interpretability': random.uniform(0.5, 0.9),
                'generalization': random.uniform(0.4, 0.8),
                'validity': random.uniform(0.6, 0.95)
            }
            for _ in range(30)
        ]

        tracker.record_generation(gen, population, evaluation_results)
        time.sleep(0.1)  # 短い待機

    # 最適化終了
    tracker.end_optimization()

    # 統計情報表示
    summary = OptimizationReporter.generate_summary_statistics(tracker)
    print("\n📊 Summary Statistics:")
    print(
        f"   Final Best Fitness: {summary['fitness_statistics']['final_best']:.4f}")
    print(
        f"   Total Improvement: {summary['fitness_statistics']['improvement']:.4f}")
    print(
        f"   Convergence Detected: {summary['convergence_info']['convergence_detected']}")

    print("✅ Demo completed!")
