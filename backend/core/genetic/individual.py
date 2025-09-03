"""
遺伝的アルゴリズムの個体 - core/genetic/individual.py
ファジィ決定木を表現する個体クラス
"""

from typing import Dict, Any, Optional, List, Tuple
import uuid
import numpy as np
import random
from dataclasses import dataclass
from enum import Enum

from ..decision_tree.node import FuzzyDecisionNode
from ..fuzzy.membership import MembershipFunction, TriangularMF, MembershipFunctionFactory, MembershipType


class IndividualType(Enum):
    """個体の種類"""
    TREE_STRUCTURE = "tree_structure"
    PARAMETER_VECTOR = "parameter_vector" 
    HYBRID = "hybrid"


@dataclass
class FitnessComponents:
    """適応度成分"""
    accuracy: float = 0.0
    complexity: float = 0.0
    interpretability: float = 0.0
    generalization: float = 0.0
    total: float = 0.0


class GeneticIndividual:
    """遺伝的アルゴリズムの個体"""
    
    def __init__(self, individual_type: IndividualType = IndividualType.HYBRID):
        self.id = str(uuid.uuid4())
        self.individual_type = individual_type
        self.generation = 0
        self.age = 0
        
        # 遺伝子表現
        self.genome: np.ndarray = None
        self.tree_genes: Dict[str, Any] = {}
        
        # 表現型（ファジィ決定木）
        self.tree: Optional[FuzzyDecisionNode] = None
        
        # 適応度
        self.fitness_components = FitnessComponents()
        self.fitness: float = 0.0
        self.raw_fitness: float = 0.0
        
        # 統計情報
        self.evaluation_count = 0
        self.mutation_count = 0
        self.crossover_count = 0
        
        # メタ情報
        self.parents: List[str] = []
        self.creation_method = "random"
    
    def initialize_random(self, genome_length: int, feature_names: List[str], 
                         max_depth: int = 5, min_samples_leaf: int = 5):
        """ランダム初期化"""
        
        # パラメータベクトル初期化
        self.genome = np.random.random(genome_length)
        
        # 木構造遺伝子初期化
        self.tree_genes = {
            'max_depth': random.randint(2, max_depth),
            'min_samples_leaf': min_samples_leaf,
            'feature_selection_probs': np.random.dirichlet(np.ones(len(feature_names))),
            'membership_params': self._generate_random_membership_params(feature_names),
            'split_strategies': [random.choice(['information_gain', 'gini', 'fuzzy_entropy']) 
                               for _ in range(max_depth)]
        }
        
        self.creation_method = "random_initialization"
    
    def _generate_random_membership_params(self, feature_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """ランダムなメンバーシップ関数パラメータ生成"""
        membership_params = {}
        
        for feature in feature_names:
            # 各特徴量に対してLow, Medium, Highのパラメータ
            membership_params[feature] = {
                'Low': {
                    'type': 'triangular',
                    'a': 0.0,
                    'b': random.uniform(0, 3),
                    'c': random.uniform(2, 5)
                },
                'Medium': {
                    'type': 'triangular', 
                    'a': random.uniform(2, 4),
                    'b': random.uniform(4, 6),
                    'c': random.uniform(6, 8)
                },
                'High': {
                    'type': 'triangular',
                    'a': random.uniform(5, 8),
                    'b': random.uniform(7, 10),
                    'c': 10.0
                }
            }
        
        return membership_params
    
    def build_tree(self, training_data: np.ndarray, feature_names: List[str], 
                  target_name: str) -> bool:
        """遺伝子から決定木を構築"""
        
        try:
            from ..decision_tree.builder import FuzzyTreeBuilder
            
            builder = FuzzyTreeBuilder(
                max_depth=self.tree_genes.get('max_depth', 5),
                min_samples_leaf=self.tree_genes.get('min_samples_leaf', 5),
                membership_params=self.tree_genes.get('membership_params', {}),
                feature_selection_probs=self.tree_genes.get('feature_selection_probs', None)
            )
            
            self.tree = builder.build_from_genes(
                training_data, feature_names, target_name, self.genome
            )
            
            return self.tree is not None
            
        except Exception as e:
            print(f"Tree building error for individual {self.id}: {e}")
            self.tree = None
            return False
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1):
        """突然変異"""
        
        self.mutation_count += 1
        
        # パラメータベクトルの突然変異
        if self.genome is not None:
            mask = np.random.random(len(self.genome)) < mutation_rate
            self.genome[mask] += np.random.normal(0, mutation_strength, np.sum(mask))
            self.genome = np.clip(self.genome, 0, 1)
        
        # 木構造遺伝子の突然変異
        self._mutate_tree_genes(mutation_rate)
        
        # 木を無効化（再構築が必要）
        self.tree = None
        self.fitness = 0.0
    
    def _mutate_tree_genes(self, mutation_rate: float):
        """木構造遺伝子の突然変異"""
        
        # 最大深度の変更
        if random.random() < mutation_rate:
            self.tree_genes['max_depth'] = max(2, self.tree_genes.get('max_depth', 5) + random.choice([-1, 0, 1]))
        
        # メンバーシップ関数パラメータの変更
        if random.random() < mutation_rate and 'membership_params' in self.tree_genes:
            self._mutate_membership_params(mutation_rate * 0.5)
        
        # 特徴量選択確率の変更
        if random.random() < mutation_rate and 'feature_selection_probs' in self.tree_genes:
            noise = np.random.normal(0, 0.1, len(self.tree_genes['feature_selection_probs']))
            self.tree_genes['feature_selection_probs'] += noise
            self.tree_genes['feature_selection_probs'] = np.abs(self.tree_genes['feature_selection_probs'])
            self.tree_genes['feature_selection_probs'] /= np.sum(self.tree_genes['feature_selection_probs'])
    
    def _mutate_membership_params(self, mutation_rate: float):
        """メンバーシップ関数パラメータの突然変異"""
        
        for feature, fuzzy_sets in self.tree_genes['membership_params'].items():
            for fuzzy_set, params in fuzzy_sets.items():
                if params['type'] == 'triangular':
                    for param in ['a', 'b', 'c']:
                        if random.random() < mutation_rate:
                            params[param] += random.gauss(0, 0.5)
                            params[param] = max(0, min(10, params[param]))
                    
                    # 順序制約の維持
                    if params['a'] > params['b']:
                        params['a'], params['b'] = params['b'], params['a']
                    if params['b'] > params['c']:
                        params['b'], params['c'] = params['c'], params['b']
                    if params['a'] > params['b']:
                        params['a'], params['b'] = params['b'], params['a']
    
    def crossover(self, other: 'GeneticIndividual') -> Tuple['GeneticIndividual', 'GeneticIndividual']:
        """交叉"""
        
        child1 = GeneticIndividual(self.individual_type)
        child2 = GeneticIndividual(self.individual_type)
        
        child1.parents = [self.id, other.id]
        child2.parents = [other.id, self.id]
        child1.creation_method = "crossover"
        child2.creation_method = "crossover"
        child1.crossover_count = self.crossover_count + 1
        child2.crossover_count = other.crossover_count + 1
        
        # パラメータベクトルの交叉
        if self.genome is not None and other.genome is not None:
            child1.genome, child2.genome = self._crossover_genomes(self.genome, other.genome)
        
        # 木構造遺伝子の交叉
        child1.tree_genes, child2.tree_genes = self._crossover_tree_genes(
            self.tree_genes, other.tree_genes)
        
        return child1, child2
    
    def _crossover_genomes(self, genome1: np.ndarray, genome2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """パラメータベクトルの交叉"""
        
        # 一様交叉
        mask = np.random.random(len(genome1)) < 0.5
        
        child1_genome = genome1.copy()
        child2_genome = genome2.copy()
        
        child1_genome[mask] = genome2[mask]
        child2_genome[mask] = genome1[mask]
        
        return child1_genome, child2_genome
    
    def _crossover_tree_genes(self, genes1: Dict[str, Any], genes2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """木構造遺伝子の交叉"""
        
        child1_genes = genes1.copy()
        child2_genes = genes2.copy()
        
        # 各パラメータを確率的に交換
        crossover_prob = 0.5
        
        for key in ['max_depth', 'min_samples_leaf']:
            if random.random() < crossover_prob:
                child1_genes[key], child2_genes[key] = genes2.get(key, genes1.get(key, 5)), genes1.get(key, 5)
        
        # 特徴量選択確率のブレンド交叉
        if 'feature_selection_probs' in genes1 and 'feature_selection_probs' in genes2:
            alpha = random.uniform(0.3, 0.7)
            child1_genes['feature_selection_probs'] = alpha * genes1['feature_selection_probs'] + (1-alpha) * genes2['feature_selection_probs']
            child2_genes['feature_selection_probs'] = (1-alpha) * genes1['feature_selection_probs'] + alpha * genes2['feature_selection_probs']
        
        # メンバーシップ関数パラメータの交叉
        if 'membership_params' in genes1 and 'membership_params' in genes2:
            child1_genes['membership_params'] = self._crossover_membership_params(
                genes1['membership_params'], genes2['membership_params'])
            child2_genes['membership_params'] = self._crossover_membership_params(
                genes2['membership_params'], genes1['membership_params'])
        
        return child1_genes, child2_genes
    
    def _crossover_membership_params(self, params1: Dict[str, Dict[str, Any]], 
                                   params2: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """メンバーシップ関数パラメータの交叉"""
        
        result = {}
        
        for feature in params1.keys():
            if feature in params2:
                result[feature] = {}
                
                for fuzzy_set in params1[feature].keys():
                    if fuzzy_set in params2[feature]:
                        # パラメータをランダムに選択または平均
                        if random.random() < 0.5:
                            result[feature][fuzzy_set] = params1[feature][fuzzy_set].copy()
                        else:
                            # パラメータの平均
                            result[feature][fuzzy_set] = {}
                            for param_key in params1[feature][fuzzy_set]:
                                if param_key != 'type':
                                    val1 = params1[feature][fuzzy_set][param_key]
                                    val2 = params2[feature][fuzzy_set].get(param_key, val1)
                                    result[feature][fuzzy_set][param_key] = (val1 + val2) / 2
                                else:
                                    result[feature][fuzzy_set][param_key] = params1[feature][fuzzy_set][param_key]
                    else:
                        result[feature][fuzzy_set] = params1[feature][fuzzy_set].copy()
            else:
                result[feature] = params1[feature].copy()
        
        return result
    
    def evaluate_fitness(self, test_data: np.ndarray, feature_names: List[str], 
                        target_name: str, fitness_weights: Dict[str, float] = None) -> float:
        """適応度評価"""
        
        if fitness_weights is None:
            fitness_weights = {'accuracy': 0.6, 'complexity': 0.2, 'interpretability': 0.2}
        
        self.evaluation_count += 1
        
        if self.tree is None:
            self.fitness = 0.0
            return self.fitness
        
        try:
            # 精度評価
            accuracy = self._evaluate_accuracy(test_data, feature_names)
            
            # 複雑度評価（単純さ）
            complexity_score = self._evaluate_complexity()
            
            # 解釈可能性評価
            interpretability_score = self._evaluate_interpretability()
            
            # 適応度成分更新
            self.fitness_components.accuracy = accuracy
            self.fitness_components.complexity = complexity_score
            self.fitness_components.interpretability = interpretability_score
            
            # 総合適応度計算
            self.fitness = (
                fitness_weights.get('accuracy', 0.6) * accuracy +
                fitness_weights.get('complexity', 0.2) * complexity_score +
                fitness_weights.get('interpretability', 0.2) * interpretability_score
            )
            
            self.fitness_components.total = self.fitness
            self.raw_fitness = self.fitness
            
            return self.fitness
            
        except Exception as e:
            print(f"Fitness evaluation error for individual {self.id}: {e}")
            self.fitness = 0.0
            return self.fitness
    
    def _evaluate_accuracy(self, test_data: np.ndarray, feature_names: List[str]) -> float:
        """精度評価"""
        
        if len(test_data) == 0:
            return 0.0
        
        predictions = []
        targets = []
        
        for row in test_data:
            features = {feature_names[i]: row[i] for i in range(len(feature_names))}
            target = row[-1]  # 最後の列が目標値
            
            pred = self.tree.predict(features)
            predictions.append(pred)
            targets.append(target)
        
        # RMSE計算
        mse = np.mean([(p - t) ** 2 for p, t in zip(predictions, targets)])
        rmse = np.sqrt(mse)
        
        # 正規化された精度スコア（0-1）
        max_error = 1.0  # 最大想定誤差
        accuracy = max(0.0, 1.0 - (rmse / max_error))
        
        return accuracy
    
    def _evaluate_complexity(self) -> float:
        """複雑度評価（単純さスコア）"""
        
        if self.tree is None:
            return 0.0
        
        # ノード数
        node_count = self._count_nodes(self.tree)
        max_nodes = 31  # 完全5分木の最大ノード数
        
        # 深度
        depth = self._calculate_depth(self.tree)
        max_depth = self.tree_genes.get('max_depth', 5)
        
        # 単純さスコア（複雑度の逆数）
        node_simplicity = 1.0 - min(node_count / max_nodes, 1.0)
        depth_simplicity = 1.0 - min(depth / max_depth, 1.0)
        
        return 0.6 * node_simplicity + 0.4 * depth_simplicity
    
    def _evaluate_interpretability(self) -> float:
        """解釈可能性評価"""
        
        if self.tree is None:
            return 0.0
        
        # 特徴量の多様性
        feature_diversity = self._calculate_feature_diversity()
        
        # メンバーシップ関数の単純さ
        membership_simplicity = self._calculate_membership_simplicity()
        
        # 分岐の一貫性
        branch_consistency = self._calculate_branch_consistency()
        
        return 0.4 * feature_diversity + 0.3 * membership_simplicity + 0.3 * branch_consistency
    
    def _count_nodes(self, node: FuzzyDecisionNode) -> int:
        """ノード数をカウント"""
        if node is None:
            return 0
        
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        
        return count
    
    def _calculate_depth(self, node: FuzzyDecisionNode) -> int:
        """木の深度計算"""
        if node is None or node.is_leaf:
            return 1
        
        max_child_depth = 0
        for child in node.children.values():
            child_depth = self._calculate_depth(child)
            max_child_depth = max(max_child_depth, child_depth)
        
        return 1 + max_child_depth
    
    def _calculate_feature_diversity(self) -> float:
        """特徴量多様性計算"""
        used_features = set()
        self._collect_features(self.tree, used_features)
        
        total_features = len(self.tree_genes.get('feature_selection_probs', []))
        if total_features == 0:
            return 0.0
        
        return len(used_features) / total_features
    
    def _collect_features(self, node: FuzzyDecisionNode, used_features: set):
        """使用されている特徴量を収集"""
        if node is None or node.is_leaf:
            return
        
        if node.feature_name:
            used_features.add(node.feature_name)
        
        for child in node.children.values():
            self._collect_features(child, used_features)
    
    def _calculate_membership_simplicity(self) -> float:
        """メンバーシップ関数の単純さ計算"""
        # 三角形メンバーシップ関数の使用率を基準
        triangular_count = 0
        total_count = 0
        
        for feature_params in self.tree_genes.get('membership_params', {}).values():
            for fuzzy_set_params in feature_params.values():
                total_count += 1
                if fuzzy_set_params.get('type') == 'triangular':
                    triangular_count += 1
        
        return triangular_count / max(1, total_count)
    
    def _calculate_branch_consistency(self) -> float:
        """分岐の一貫性計算"""
        # 簡易実装：深度の均一性
        depths = []
        self._collect_leaf_depths(self.tree, 0, depths)
        
        if len(depths) <= 1:
            return 1.0
        
        depth_std = np.std(depths)
        max_std = max(depths) - min(depths)
        
        return 1.0 - (depth_std / max(1, max_std))
    
    def _collect_leaf_depths(self, node: FuzzyDecisionNode, current_depth: int, depths: List[int]):
        """葉ノードの深度を収集"""
        if node is None:
            return
        
        if node.is_leaf:
            depths.append(current_depth)
        else:
            for child in node.children.values():
                self._collect_leaf_depths(child, current_depth + 1, depths)
    
    def copy(self) -> 'GeneticIndividual':
        """個体のコピー"""
        new_individual = GeneticIndividual(self.individual_type)
        
        # 基本属性
        new_individual.generation = self.generation
        new_individual.age = self.age + 1
        
        # 遺伝子
        if self.genome is not None:
            new_individual.genome = self.genome.copy()
        new_individual.tree_genes = self._deep_copy_dict(self.tree_genes)
        
        # 適応度
        new_individual.fitness = self.fitness
        new_individual.fitness_components = FitnessComponents(
            accuracy=self.fitness_components.accuracy,
            complexity=self.fitness_components.complexity,
            interpretability=self.fitness_components.interpretability,
            total=self.fitness_components.total
        )
        
        # 統計
        new_individual.evaluation_count = self.evaluation_count
        new_individual.parents = self.parents.copy()
        new_individual.creation_method = "copy"
        
        return new_individual
    
    def _deep_copy_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """辞書の深いコピー"""
        if isinstance(d, dict):
            return {k: self._deep_copy_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [self._deep_copy_dict(item) for item in d]
        elif isinstance(d, np.ndarray):
            return d.copy()
        else:
            return d
    
    def get_info(self) -> Dict[str, Any]:
        """個体情報取得"""
        return {
            'id': self.id,
            'generation': self.generation,
            'age': self.age,
            'fitness': self.fitness,
            'fitness_components': {
                'accuracy': self.fitness_components.accuracy,
                'complexity': self.fitness_components.complexity,
                'interpretability': self.fitness_components.interpretability,
                'total': self.fitness_components.total
            },
            'genome_length': len(self.genome) if self.genome is not None else 0,
            'tree_exists': self.tree is not None,
            'evaluation_count': self.evaluation_count,
            'mutation_count': self.mutation_count,
            'crossover_count': self.crossover_count,
            'parents': self.parents,
            'creation_method': self.creation_method,
            'tree_depth': self._calculate_depth(self.tree) if self.tree else 0,
            'tree_nodes': self._count_nodes(self.tree) if self.tree else 0
        }