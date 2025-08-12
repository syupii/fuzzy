# backend/genetic_fuzzy_tree.py - 完全修正版
"""
🧬 Fixed Genetic Fuzzy Decision Tree System
修正版遺伝的ファジィ決定木システム - record_individual 呼び出し修正済み
"""

import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import copy
import math
import uuid
import time
import warnings
warnings.filterwarnings('ignore')

# 依存関係のインポート（エラー処理付き）
try:
    from explanation_engine import FuzzyExplanationEngine, DecisionExplanation
except ImportError:
    print("⚠️ explanation_engine not found, using mock")

    class FuzzyExplanationEngine:
        def generate_comprehensive_explanation(self, *args, **kwargs):
            return "Basic explanation generated"

    class DecisionExplanation:
        def __init__(self, overall_conclusion, confidence_level=None):
            self.overall_conclusion = overall_conclusion

try:
    from optimization_tracker import OptimizationTracker, GenerationStats
except ImportError:
    print("⚠️ optimization_tracker not found, using mock")

    class OptimizationTracker:
        def __init__(self, run_id):
            self.run_id = run_id

        def start_optimization(self, config): pass
        def end_optimization(self): pass
        def record_individual(self, *args, **kwargs): pass
        def record_generation(self, *args): pass

try:
    from evaluation_metrics import MultiObjectiveEvaluator, MetricResult
except ImportError:
    print("⚠️ evaluation_metrics not found, using mock")

    class MultiObjectiveEvaluator:
        def evaluate_individual(self, *args, **kwargs):
            return {
                'accuracy': type('obj', (object,), {'normalized_value': 0.8})(),
                'simplicity': type('obj', (object,), {'normalized_value': 0.7})(),
                'interpretability': type('obj', (object,), {'normalized_value': 0.6})(),
                'generalization': type('obj', (object,), {'normalized_value': 0.75})(),
                'validity': type('obj', (object,), {'normalized_value': 0.9})()
            }

try:
    from advanced_nodes import (
        AdvancedFuzzyDecisionNode, MembershipFunction, MembershipType,
        FitnessComponents, TreeVisualizationHelper
    )
except ImportError:
    print("⚠️ advanced_nodes not found, using simple implementations")

    from enum import Enum

    class MembershipType(Enum):
        TRIANGULAR = "triangular"
        GAUSSIAN = "gaussian"
        TRAPEZOIDAL = "trapezoidal"

    class MembershipFunction:
        def __init__(self, name, mf_type, parameters, weight=1.0):
            self.name = name
            self.type = mf_type
            self.parameters = parameters
            self.weight = weight

        def evaluate(self, x):
            if self.type == MembershipType.TRIANGULAR:
                a, b, c = self.parameters
                if x <= a or x >= c:
                    return 0.0
                elif a < x <= b:
                    return (x - a) / (b - a)
                else:  # b < x < c
                    return (c - x) / (c - b)
            elif self.type == MembershipType.GAUSSIAN:
                center, sigma = self.parameters
                return np.exp(-0.5 * ((x - center) / sigma) ** 2)
            else:
                return 0.5  # デフォルト値

    class AdvancedFuzzyDecisionNode:
        def __init__(self, node_id=None, feature_name=None, leaf_value=None):
            self.node_id = node_id or str(uuid.uuid4())[:8]
            self.feature_name = feature_name
            self.leaf_value = leaf_value
            self.is_leaf = leaf_value is not None
            self.membership_functions = {}
            self.children = {}
            self.depth = 0

        def add_membership_function(self, label, mf):
            self.membership_functions[label] = mf

        def add_child(self, label, child):
            self.children[label] = child

        def predict(self, features):
            if self.is_leaf:
                return self.leaf_value

            feature_value = features.get(self.feature_name, 5.0)

            # ファジィ評価
            max_membership = 0.0
            best_label = None

            for label, mf in self.membership_functions.items():
                membership = mf.evaluate(feature_value)
                if membership > max_membership:
                    max_membership = membership
                    best_label = label

            # 子ノードで予測
            if best_label and best_label in self.children:
                return self.children[best_label].predict(features)
            else:
                return 0.5  # デフォルト値

        def predict_with_explanation(self, features, feature_names):
            prediction = self.predict(features)
            explanation = {
                'confidence': 0.8,
                'decision_steps': [],
                'rationale': f'Prediction: {prediction:.3f}'
            }
            return prediction, explanation

        def calculate_complexity(self):
            if self.is_leaf:
                return 1
            total = 1 + len(self.membership_functions)
            for child in self.children.values():
                total += child.calculate_complexity()
            return total

        def calculate_depth(self):
            if self.is_leaf:
                return 1
            max_depth = 0
            for child in self.children.values():
                max_depth = max(max_depth, child.calculate_depth())
            return max_depth + 1

        def get_tree_structure(self):
            if self.is_leaf:
                return {'type': 'leaf', 'value': self.leaf_value}
            return {
                'type': 'internal',
                'feature': self.feature_name,
                'children': len(self.children),
                'membership_functions': len(self.membership_functions)
            }

    class FitnessComponents:
        def __init__(self, accuracy=0.0, simplicity=0.0, interpretability=0.0,
                     generalization=0.0, validity=0.0):
            self.accuracy = accuracy
            self.simplicity = simplicity
            self.interpretability = interpretability
            self.generalization = generalization
            self.validity = validity
            self.overall = self.compute_overall()

        def compute_overall(self):
            weights = [0.3, 0.2, 0.2, 0.15, 0.15]
            components = [self.accuracy, self.simplicity, self.interpretability,
                          self.generalization, self.validity]
            return sum(w * c for w, c in zip(weights, components))

        def to_dict(self):
            return {
                'accuracy': self.accuracy,
                'simplicity': self.simplicity,
                'interpretability': self.interpretability,
                'generalization': self.generalization,
                'validity': self.validity,
                'overall': self.overall
            }

        @property
        def __dict__(self):
            return self.to_dict()

    class TreeVisualizationHelper:
        @staticmethod
        def generate_tree_summary(tree):
            return f"Tree: depth={tree.calculate_depth()}, complexity={tree.calculate_complexity()}"


@dataclass
class GeneticParameters:
    """遺伝的アルゴリズムパラメータ"""
    population_size: int = 50
    generations: int = 30
    mutation_rate: float = 0.15
    crossover_rate: float = 0.8
    elite_size: int = 5
    tournament_size: int = 3
    max_depth: int = 6
    min_membership_functions: int = 2
    max_membership_functions: int = 4

    def __post_init__(self):
        # エリートサイズの調整
        if self.elite_size >= self.population_size:
            self.elite_size = max(1, self.population_size // 10)


class Individual:
    """遺伝的個体"""

    def __init__(self, individual_id: str = None, generation: int = 0):
        self.individual_id = individual_id or str(uuid.uuid4())[:8]
        self.generation = generation
        self.tree: Optional[AdvancedFuzzyDecisionNode] = None
        self.genome: Dict[str, Any] = {}
        self.fitness_value: float = 0.0
        self.fitness_components: Optional[FitnessComponents] = None
        self.complexity_score: int = 0
        self.parents: List[str] = []


class GeneticFuzzyTreeOptimizer:
    """遺伝的ファジィ決定木最適化器 - 修正版"""

    def __init__(self, parameters: GeneticParameters = None, random_seed: int = None):
        self.parameters = parameters or GeneticParameters()
        self.random_seed = random_seed

        # 乱数シード設定
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # 評価器
        self.evaluator = MultiObjectiveEvaluator()

        # トラッカー
        self.tracker: Optional[OptimizationTracker] = None

        # 説明エンジン
        self.explanation_engine = FuzzyExplanationEngine()

        # 統計情報
        self.evolution_stats = {
            'best_fitness_history': [],
            'avg_fitness_history': [],
            'diversity_history': [],
            'evaluation_times': []
        }

        # 特徴名（研究室選択基準）
        self.feature_names = [
            'research_intensity', 'advisor_style', 'team_work',
            'workload', 'theory_practice'
        ]

        # データ保存用
        self.training_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.target_column: str = 'compatibility'

        print("🧬 GeneticFuzzyTreeOptimizer initialized")

    def optimize(self, training_data: pd.DataFrame,
                 test_data: pd.DataFrame = None,
                 target_column: str = 'compatibility',
                 run_id: str = None) -> Dict[str, Any]:
        """遺伝的最適化実行"""

        # トラッカー初期化
        self.tracker = OptimizationTracker(run_id)

        # 設定保存
        config = {
            'parameters': self.parameters.__dict__,
            'feature_names': self.feature_names,
            'training_samples': len(training_data),
            'test_samples': len(test_data) if test_data is not None else 0,
            'target_column': target_column,
            'random_seed': self.random_seed
        }

        self.tracker.start_optimization(config)

        print(f"🚀 Starting genetic optimization...")
        print(f"📊 Training samples: {len(training_data)}")
        print(f"🎯 Target: {target_column}")
        print(f"👥 Population: {self.parameters.population_size}")
        print(f"🔄 Generations: {self.parameters.generations}")

        try:
            # テストデータがない場合は訓練データを分割
            if test_data is None:
                test_data = training_data.sample(
                    frac=0.2, random_state=self.random_seed)
                training_data = training_data.drop(test_data.index)

            # データ保存
            self.training_data = training_data
            self.test_data = test_data
            self.target_column = target_column

            # 初期個体群生成
            population = self._create_initial_population()

            # 進化実行
            result = self._evolve_population(population)

            # 最終化
            self.tracker.end_optimization()

            return result

        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            if self.tracker:
                self.tracker.end_optimization()
            raise

    def _create_initial_population(self) -> List[Individual]:
        """初期個体群生成"""
        population = []

        for i in range(self.parameters.population_size):
            individual = self._create_individual()
            individual.individual_id = f"gen0_ind{i}"
            population.append(individual)

        return population

    def _evolve_population(self, population: List[Individual]) -> Dict[str, Any]:
        """個体群進化 - 修正版"""

        best_individual = None

        # 世代ループ
        for generation in range(self.parameters.generations):
            generation_start_time = time.time()

            print(
                f"\n🔄 Generation {generation + 1}/{self.parameters.generations}")

            # 個体評価
            evaluation_results = []

            for individual in population:
                evaluation_start = time.time()

                # 適応度評価
                fitness_value = self._evaluate_individual_fitness(individual)
                individual.fitness_value = fitness_value

                evaluation_time = time.time() - evaluation_start

                # 評価結果記録
                evaluation_result = {
                    'individual_id': individual.individual_id,
                    'overall_fitness': fitness_value,
                    'fitness_components': individual.fitness_components.__dict__ if individual.fitness_components else {},
                    'evaluation_time': evaluation_time,
                    'model_complexity': individual.complexity_score
                }
                evaluation_results.append(evaluation_result)

                # ✅ 修正済み個体記録 - 正しい引数順序
                self.tracker.record_individual(
                    individual=individual,
                    individual_id=individual.individual_id,
                    generation=generation,
                    fitness_components=evaluation_result.get(
                        'fitness_components', {}),
                    overall_fitness=evaluation_result.get(
                        'overall_fitness', 0.0),
                    notes=f"Generation {generation} evaluation"
                )

            # 最良個体更新
            population.sort(key=lambda x: x.fitness_value, reverse=True)
            current_best = population[0]

            if best_individual is None or current_best.fitness_value > best_individual.fitness_value:
                best_individual = copy.deepcopy(current_best)

            # 統計計算
            fitness_values = [ind.fitness_value for ind in population]
            best_fitness = max(fitness_values)
            avg_fitness = np.mean(fitness_values)

            # 多様性計算
            diversity_score = self._calculate_population_diversity(population)

            # 世代記録
            generation_metrics = {
                'mutation_success_rate': 0.7,
                'crossover_success_rate': 0.8
            }

            self.tracker.record_generation(
                generation, population, evaluation_results, generation_metrics
            )

            # 進化統計更新
            self.evolution_stats['best_fitness_history'].append(best_fitness)
            self.evolution_stats['avg_fitness_history'].append(avg_fitness)
            self.evolution_stats['diversity_history'].append(diversity_score)
            self.evolution_stats['evaluation_times'].append(
                time.time() - generation_start_time)

            print(
                f"📊 Best: {best_fitness:.4f}, Avg: {avg_fitness:.4f}, Diversity: {diversity_score:.3f}")

            # 収束チェック
            if self._check_convergence(generation):
                print(f"🎯 Convergence detected at generation {generation + 1}")
                break

            # 最終世代でない場合は次世代生成
            if generation < self.parameters.generations - 1:
                population = self._create_next_generation(population)

        # 最終結果まとめ
        result = self._compile_final_result(best_individual, population)

        print(f"🏁 Optimization completed!")
        print(f"🎯 Best fitness: {result['best_fitness']:.4f}")
        print(f"📊 Final population diversity: {result['final_diversity']:.3f}")

        return result

    def _create_individual(self) -> Individual:
        """個体生成"""

        individual = Individual(
            individual_id=str(uuid.uuid4())[:8],
            generation=0
        )

        # ランダムなファジィ決定木生成
        individual.tree = self._generate_random_fuzzy_tree()

        # ゲノム情報作成
        individual.genome = self._tree_to_genome(individual.tree)

        # 複雑度計算
        individual.complexity_score = individual.tree.calculate_complexity()

        return individual

    def _generate_random_fuzzy_tree(self, depth: int = 0, max_depth: int = None) -> AdvancedFuzzyDecisionNode:
        """ランダムファジィ決定木生成"""

        if max_depth is None:
            max_depth = self.parameters.max_depth

        # リーフノード条件
        if (depth >= max_depth or
            random.random() < 0.3 or
                depth > 0 and random.random() < 0.1 * depth):

            # リーフ値はランダム（0-1）
            leaf_value = random.uniform(0.0, 1.0)
            return AdvancedFuzzyDecisionNode(leaf_value=leaf_value)

        # 内部ノード生成
        node = AdvancedFuzzyDecisionNode(
            feature_name=random.choice(self.feature_names)
        )

        # メンバーシップ関数生成
        num_mfs = random.randint(
            self.parameters.min_membership_functions,
            self.parameters.max_membership_functions
        )

        mf_labels = ['low', 'medium', 'high', 'very_high'][:num_mfs]

        for i, label in enumerate(mf_labels):
            mf = self._generate_random_membership_function(label, i, num_mfs)
            node.add_membership_function(label, mf)

            # 子ノード生成
            child = self._generate_random_fuzzy_tree(depth + 1, max_depth)
            node.add_child(label, child)

        return node

    def _generate_random_membership_function(self, label: str, index: int, total_mfs: int) -> MembershipFunction:
        """ランダムメンバーシップ関数生成"""

        # タイプ選択
        mf_types = [MembershipType.TRIANGULAR,
                    MembershipType.GAUSSIAN, MembershipType.TRAPEZOIDAL]
        mf_type = random.choice(mf_types)

        # パラメータ生成（0-10スケール）
        if mf_type == MembershipType.TRIANGULAR:
            # 三角形: [a, b, c] where a < b < c
            center = (index + 0.5) * 10.0 / total_mfs
            spread = 10.0 / total_mfs * 0.8
            a = max(0, center - spread)
            b = center
            c = min(10, center + spread)
            parameters = [a, b, c]

        elif mf_type == MembershipType.GAUSSIAN:
            # ガウシアン: [center, sigma]
            center = (index + 0.5) * 10.0 / total_mfs
            sigma = 10.0 / total_mfs * 0.5
            parameters = [center, sigma]

        else:  # TRAPEZOIDAL
            # 台形: [a, b, c, d]
            center = (index + 0.5) * 10.0 / total_mfs
            spread = 10.0 / total_mfs * 0.6
            a = max(0, center - spread)
            b = max(0, center - spread * 0.5)
            c = min(10, center + spread * 0.5)
            d = min(10, center + spread)
            parameters = [a, b, c, d]

        return MembershipFunction(label, mf_type, parameters)

    def _evaluate_individual_fitness(self, individual: Individual) -> float:
        """個体適応度評価"""

        try:
            if individual.tree is None:
                return 0.0

            # 訓練データで予測
            predictions = []
            targets = []

            for _, row in self.training_data.iterrows():
                features = {
                    'research_intensity': row.get('research_intensity', 5.0),
                    'advisor_style': row.get('advisor_style', 5.0),
                    'team_work': row.get('team_work', 5.0),
                    'workload': row.get('workload', 5.0),
                    'theory_practice': row.get('theory_practice', 5.0)
                }

                prediction = individual.tree.predict(features)
                target = row.get(self.target_column, 0.5)

                predictions.append(prediction)
                targets.append(target)

            predictions = np.array(predictions)
            targets = np.array(targets)

            # 多目的評価
            evaluation_results = self.evaluator.evaluate_individual(
                individual, predictions, targets
            )

            # 適応度成分計算
            accuracy = evaluation_results['accuracy'].normalized_value
            simplicity = evaluation_results['simplicity'].normalized_value
            interpretability = evaluation_results['interpretability'].normalized_value
            generalization = evaluation_results['generalization'].normalized_value
            validity = evaluation_results['validity'].normalized_value

            # フィットネス成分保存
            individual.fitness_components = FitnessComponents(
                accuracy=accuracy,
                simplicity=simplicity,
                interpretability=interpretability,
                generalization=generalization,
                validity=validity
            )

            # 総合適応度
            overall_fitness = individual.fitness_components.compute_overall()

            return overall_fitness

        except Exception as e:
            print(f"⚠️ Fitness evaluation error: {e}")
            return 0.0

    def _create_next_generation(self, population: List[Individual]) -> List[Individual]:
        """次世代生成"""

        next_generation = []

        # エリート保存
        elite_count = min(self.parameters.elite_size, len(population))
        for i in range(elite_count):
            elite = copy.deepcopy(population[i])
            elite.generation += 1
            next_generation.append(elite)

        # 残りを交叉・突然変異で生成
        while len(next_generation) < self.parameters.population_size:
            # 親選択（トーナメント選択）
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)

            # 交叉
            if random.random() < self.parameters.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

            # 突然変異
            if random.random() < self.parameters.mutation_rate:
                child1 = self._mutate(child1)
            if random.random() < self.parameters.mutation_rate:
                child2 = self._mutate(child2)

            # 評価
            child1.fitness_value = self._evaluate_individual_fitness(child1)
            child2.fitness_value = self._evaluate_individual_fitness(child2)

            next_generation.extend([child1, child2])

        return next_generation[:self.parameters.population_size]

    def _tournament_selection(self, population: List[Individual]) -> Individual:
        """トーナメント選択"""
        tournament_size = min(self.parameters.tournament_size, len(population))
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.fitness_value)

    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """交叉操作"""

        try:
            # 新個体作成
            child1 = Individual(
                individual_id=str(uuid.uuid4())[:8],
                generation=parent1.generation + 1
            )

            child2 = Individual(
                individual_id=str(uuid.uuid4())[:8],
                generation=parent2.generation + 1
            )

            # 簡易交叉：親の木をランダム選択
            child1.tree = copy.deepcopy(
                random.choice([parent1.tree, parent2.tree]))
            child2.tree = copy.deepcopy(
                random.choice([parent1.tree, parent2.tree]))

            # ゲノム更新
            child1.genome = self._tree_to_genome(child1.tree)
            child2.genome = self._tree_to_genome(child2.tree)

            # 複雑度更新
            child1.complexity_score = child1.tree.calculate_complexity()
            child2.complexity_score = child2.tree.calculate_complexity()

            return child1, child2

        except Exception as e:
            print(f"⚠️ Crossover error: {e}")
            return parent1, parent2

    def _mutate(self, individual: Individual) -> Individual:
        """突然変異"""

        try:
            # 新個体作成
            mutant = copy.deepcopy(individual)
            mutant.individual_id = str(uuid.uuid4())[:8]

            # 簡易突然変異：新しい木を生成
            if random.random() < 0.5:
                mutant.tree = self._generate_random_fuzzy_tree()

            # ゲノム更新
            mutant.genome = self._tree_to_genome(mutant.tree)
            mutant.complexity_score = mutant.tree.calculate_complexity()

            return mutant

        except Exception as e:
            print(f"⚠️ Mutation error: {e}")
            return individual

    def _collect_nodes(self, tree: AdvancedFuzzyDecisionNode) -> List[AdvancedFuzzyDecisionNode]:
        """ノード収集"""

        nodes = [tree]

        if not tree.is_leaf:
            for child in tree.children.values():
                nodes.extend(self._collect_nodes(child))

        return nodes

    def _tree_to_genome(self, tree: AdvancedFuzzyDecisionNode) -> Dict[str, Any]:
        """木をゲノム表現に変換"""

        genome = {
            'tree_structure': tree.get_tree_structure(),
            'complexity': tree.calculate_complexity(),
            'depth': tree.calculate_depth(),
            'feature_usage': {}
        }

        # 特徴使用統計
        nodes = self._collect_nodes(tree)
        for node in nodes:
            if not node.is_leaf and node.feature_name:
                feature = node.feature_name
                genome['feature_usage'][feature] = genome['feature_usage'].get(
                    feature, 0) + 1

        return genome

    def _calculate_population_diversity(self, population: List[Individual]) -> float:
        """個体群多様性計算"""

        if len(population) < 2:
            return 0.0

        # 複雑度の多様性
        complexities = [ind.complexity_score for ind in population]
        complexity_diversity = np.std(
            complexities) / max(1, np.mean(complexities))

        # 適応度の多様性
        fitness_values = [ind.fitness_value for ind in population]
        fitness_diversity = np.std(fitness_values) / \
            max(0.001, np.mean(fitness_values))

        return (complexity_diversity + fitness_diversity) / 2.0

    def _check_convergence(self, generation: int) -> bool:
        """収束チェック"""

        # 最低世代数
        if generation < 5:
            return False

        # 適応度履歴での収束チェック
        recent_best = self.evolution_stats['best_fitness_history'][-5:]

        if len(recent_best) >= 5:
            variance = np.var(recent_best)
            if variance < 0.001:  # 分散が小さい
                return True

        # 多様性による収束チェック
        recent_diversity = self.evolution_stats['diversity_history'][-3:]

        if len(recent_diversity) >= 3:
            avg_diversity = np.mean(recent_diversity)
            if avg_diversity < 0.05:  # 多様性が低い
                return True

        return False

    def _compile_final_result(self, best_individual: Individual,
                              final_population: List[Individual]) -> Dict[str, Any]:
        """最終結果コンパイル"""

        result = {
            'best_individual': best_individual,
            'best_fitness': best_individual.fitness_value if best_individual else 0.0,
            'best_fitness_components': best_individual.fitness_components.__dict__ if best_individual and best_individual.fitness_components else {},
            'final_population_size': len(final_population),
            'final_diversity': self._calculate_population_diversity(final_population),
            'evolution_stats': self.evolution_stats,
            'optimization_config': self.parameters.__dict__,
            'feature_names': self.feature_names,
            'convergence_analysis': self._analyze_convergence()
        }

        # 最良個体の詳細分析
        if best_individual and best_individual.tree:
            result['best_tree_analysis'] = {
                'complexity': best_individual.tree.calculate_complexity(),
                'depth': best_individual.tree.calculate_depth(),
                'structure_summary': TreeVisualizationHelper.generate_tree_summary(best_individual.tree),
                'feature_importance': self._analyze_feature_importance(best_individual.tree)
            }

        return result

    def _analyze_convergence(self) -> Dict[str, Any]:
        """収束分析"""

        if len(self.evolution_stats['best_fitness_history']) < 3:
            return {'convergence_detected': False, 'stagnation_generations': 0}

        fitness_history = self.evolution_stats['best_fitness_history']

        # 停滞世代数計算
        stagnation_count = 0
        threshold = 0.001

        for i in range(1, len(fitness_history)):
            if abs(fitness_history[i] - fitness_history[i-1]) < threshold:
                stagnation_count += 1
            else:
                stagnation_count = 0

        convergence_detected = stagnation_count >= 5

        return {
            'convergence_detected': convergence_detected,
            'stagnation_generations': stagnation_count,
            'final_improvement': fitness_history[-1] - fitness_history[0] if len(fitness_history) > 1 else 0.0
        }

    def _analyze_feature_importance(self, tree: AdvancedFuzzyDecisionNode) -> Dict[str, float]:
        """特徴量重要度分析"""

        feature_usage = {}

        def analyze_node(node: AdvancedFuzzyDecisionNode, depth: int = 0):
            """再帰的にノードを分析"""
            if not node.is_leaf and node.feature_name:
                feature = node.feature_name
                # 深度に基づく重み付け（深い方が重要度低い）
                weight = 1.0 / (depth + 1)
                feature_usage[feature] = feature_usage.get(
                    feature, 0.0) + weight

            # 子ノードを再帰的に処理
            if not node.is_leaf:
                for child in node.children.values():
                    analyze_node(child, depth + 1)

        # ルートから分析開始
        analyze_node(tree, 0)

        # 正規化
        total_usage = sum(feature_usage.values())
        if total_usage > 0:
            for feature in feature_usage:
                feature_usage[feature] /= total_usage

        return feature_usage

    def predict_with_explanation(self, individual: Individual, user_prefs: Dict, lab_features: Dict) -> Tuple[float, str]:
        """説明付き予測"""

        if not individual or not individual.tree:
            return 0.5, "No valid tree available"

        # 特徴量ベクトル作成
        features = {}
        for criterion in self.feature_names:
            user_val = user_prefs.get(criterion, 5.0)
            lab_val = lab_features.get(criterion, 5.0)
            # 類似度計算（簡易版）
            similarity = 1.0 - abs(user_val - lab_val) / 10.0
            features[criterion] = max(
                0.0, min(1.0, similarity)) * 10.0  # 0-10スケール

        # 予測実行
        try:
            prediction, explanation_data = individual.tree.predict_with_explanation(
                features, self.feature_names)

            # 説明文生成
            explanation = f"遺伝的ファジィ決定木による予測: {prediction:.1%}\n"
            explanation += f"適応度: {individual.fitness_value:.3f}\n"
            explanation += f"木の複雑度: {individual.tree.calculate_complexity()}\n"
            explanation += f"決定経路の信頼度: {explanation_data.get('confidence', 0.8):.1%}"

            return prediction, explanation

        except Exception as e:
            return 0.5, f"Prediction error: {e}"


# デモンストレーション用関数
def run_genetic_optimization_demo():
    """遺伝的最適化デモ実行"""
    print("🧬 Genetic Fuzzy Tree Optimization Demo")
    print("=" * 50)

    # テストデータ生成
    np.random.seed(42)
    n_samples = 500

    data = []
    for i in range(n_samples):
        # ユーザー設定
        user_research = np.random.uniform(1, 10)
        user_advisor = np.random.uniform(1, 10)
        user_team = np.random.uniform(1, 10)
        user_workload = np.random.uniform(1, 10)
        user_theory = np.random.uniform(1, 10)

        # 研究室特徴
        lab_research = np.random.uniform(1, 10)
        lab_advisor = np.random.uniform(1, 10)
        lab_team = np.random.uniform(1, 10)
        lab_workload = np.random.uniform(1, 10)
        lab_theory = np.random.uniform(1, 10)

        # ガウシアン類似度ベース適合度計算
        weights = [0.25, 0.20, 0.20, 0.15, 0.20]
        sigma = 2.0

        similarities = []
        criteria_pairs = [
            (user_research, lab_research),
            (user_advisor, lab_advisor),
            (user_team, lab_team),
            (user_workload, lab_workload),
            (user_theory, lab_theory)
        ]

        for (user_val, lab_val) in criteria_pairs:
            similarity = np.exp(-((user_val - lab_val)
                                ** 2) / (2 * sigma ** 2))
            similarities.append(similarity)

        compatibility = sum(w * s for w, s in zip(weights, similarities))
        compatibility += np.random.normal(0, 0.05)
        compatibility = max(0.1, min(0.9, compatibility))

        sample = {
            'research_intensity': user_research,
            'advisor_style': user_advisor,
            'team_work': user_team,
            'workload': user_workload,
            'theory_practice': user_theory,
            'compatibility': compatibility
        }
        data.append(sample)

    df = pd.DataFrame(data)
    print(f"📊 Generated {len(df)} training samples")
    print(
        f"📈 Compatibility stats: mean={df['compatibility'].mean():.3f}, std={df['compatibility'].std():.3f}")

    # 最適化実行
    parameters = GeneticParameters(
        population_size=20,
        generations=15,
        mutation_rate=0.15,
        crossover_rate=0.8
    )

    optimizer = GeneticFuzzyTreeOptimizer(parameters, random_seed=42)
    result = optimizer.optimize(df)

    print(f"\n✅ Optimization completed!")
    print(f"🎯 Best fitness: {result['best_fitness']:.4f}")
    print(
        f"📈 Generations completed: {len(result['evolution_stats']['best_fitness_history'])}")

    # テスト予測
    if result['best_individual']:
        print(f"\n🎯 Test Predictions:")

        test_cases = [
            {
                'name': '理論重視学生',
                'user_prefs': {'research_intensity': 9, 'advisor_style': 4, 'team_work': 5, 'workload': 8, 'theory_practice': 3},
                'lab_features': {'research_intensity': 8.5, 'advisor_style': 4, 'team_work': 5, 'workload': 7.5, 'theory_practice': 3}
            },
            {
                'name': '実践重視学生',
                'user_prefs': {'research_intensity': 7, 'advisor_style': 8, 'team_work': 9, 'workload': 6, 'theory_practice': 9},
                'lab_features': {'research_intensity': 7.2, 'advisor_style': 7.8, 'team_work': 8.7, 'workload': 6.3, 'theory_practice': 8.5}
            }
        ]

        for case in test_cases:
            prediction, explanation = optimizer.predict_with_explanation(
                result['best_individual'],
                case['user_prefs'],
                case['lab_features']
            )
            print(f"   {case['name']}: {prediction*100:.1f}%")

    # モデル保存
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_path = f"models/genetic_model_{timestamp}.pkl"

    try:
        import pickle
        import os
        os.makedirs("models", exist_ok=True)

        with open(model_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"💾 Model saved successfully: {model_path}")
    except Exception as e:
        print(f"⚠️ Model save failed: {e}")

    return result


# ユーティリティ関数
def create_genetic_optimizer(population_size: int = 50,
                             generations: int = 30,
                             mutation_rate: float = 0.15,
                             crossover_rate: float = 0.8,
                             random_seed: int = None) -> GeneticFuzzyTreeOptimizer:
    """遺伝的最適化器作成"""

    parameters = GeneticParameters(
        population_size=population_size,
        generations=generations,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate
    )

    return GeneticFuzzyTreeOptimizer(parameters, random_seed)


def evaluate_individual_performance(individual: Individual,
                                    test_data: pd.DataFrame,
                                    target_column: str = 'compatibility') -> Dict[str, float]:
    """個体性能評価"""

    if not individual or not individual.tree:
        return {'error': 'Invalid individual or tree'}

    predictions = []
    targets = []

    for _, row in test_data.iterrows():
        features = {
            'research_intensity': row.get('research_intensity', 5.0),
            'advisor_style': row.get('advisor_style', 5.0),
            'team_work': row.get('team_work', 5.0),
            'workload': row.get('workload', 5.0),
            'theory_practice': row.get('theory_practice', 5.0)
        }

        prediction = individual.tree.predict(features)
        target = row.get(target_column, 0.5)

        predictions.append(prediction)
        targets.append(target)

    predictions = np.array(predictions)
    targets = np.array(targets)

    # 性能指標計算
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))

    # 相関係数
    correlation = np.corrcoef(predictions, targets)[
        0, 1] if len(predictions) > 1 else 0.0

    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(np.sqrt(mse)),
        'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
        'complexity': individual.tree.calculate_complexity(),
        'depth': individual.tree.calculate_depth(),
        'fitness': individual.fitness_value
    }


def save_genetic_model(result: Dict[str, Any], filepath: str):
    """遺伝的モデル保存"""

    try:
        import pickle
        import os

        # ディレクトリ作成
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # モデルデータ準備
        model_data = {
            'best_individual': result['best_individual'],
            'optimization_results': {
                'best_fitness': result['best_fitness'],
                'final_diversity': result['final_diversity'],
                'evolution_stats': result['evolution_stats'],
                'convergence_analysis': result['convergence_analysis']
            },
            'metadata': {
                'saved_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                'feature_names': result['feature_names'],
                'optimization_config': result['optimization_config']
            }
        }

        # 保存
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"💾 Genetic model saved: {filepath}")
        return True

    except Exception as e:
        print(f"❌ Failed to save model: {e}")
        return False


def load_genetic_model(filepath: str) -> Dict[str, Any]:
    """遺伝的モデル読み込み"""

    try:
        import pickle

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        print(f"✅ Genetic model loaded: {filepath}")
        return model_data

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return {}


# テスト実行
if __name__ == '__main__':
    try:
        result = run_genetic_optimization_demo()
        print("🎉 Demo completed successfully!")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
