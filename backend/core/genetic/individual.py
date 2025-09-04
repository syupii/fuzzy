from typing import Dict, Any, Optional, List, Tuple
import uuid
import numpy as np
import random
from dataclasses import dataclass
from enum import Enum

try:
    from ..decision_tree.node import FuzzyDecisionNode
except ImportError:
    # フォールバック
    class FuzzyDecisionNode:
        pass


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
        
        params = {}
        
        for feature in feature_names:
            # 各特徴量に対して3つのメンバーシップ関数（Low, Medium, High）
            params[feature] = {
                'low': {
                    'type': 'triangular',
                    'a': random.uniform(0.0, 0.3),
                    'b': random.uniform(0.1, 0.4),
                    'c': random.uniform(0.2, 0.5)
                },
                'medium': {
                    'type': 'triangular', 
                    'a': random.uniform(0.2, 0.5),
                    'b': random.uniform(0.4, 0.6),
                    'c': random.uniform(0.5, 0.8)
                },
                'high': {
                    'type': 'triangular',
                    'a': random.uniform(0.5, 0.8),
                    'b': random.uniform(0.6, 0.9),
                    'c': random.uniform(0.7, 1.0)
                }
            }
        
        return params
    
    def evaluate_fitness(self, training_data: np.ndarray, 
                        feature_names: List[str], target_name: str) -> float:
        """適応度評価"""
        
        self.evaluation_count += 1
        
        try:
            # 決定木構築（簡略版）
            if self.genome is not None:
                # 遺伝子から予測性能を推定
                complexity_penalty = len(self.genome) / 100.0
                accuracy_estimate = np.mean(self.genome) - complexity_penalty * 0.1
                
                self.fitness_components.accuracy = max(0.0, min(1.0, accuracy_estimate))
                self.fitness_components.complexity = 1.0 - complexity_penalty
                self.fitness_components.interpretability = 0.8 - complexity_penalty * 0.2
                self.fitness_components.generalization = self.fitness_components.accuracy * 0.9
            else:
                # フォールバック
                self.fitness_components.accuracy = 0.5
                self.fitness_components.complexity = 0.5
                self.fitness_components.interpretability = 0.5
                self.fitness_components.generalization = 0.5
            
            # 総合適応度計算
            self.fitness_components.total = (
                self.fitness_components.accuracy * 0.5 +
                self.fitness_components.complexity * 0.2 +
                self.fitness_components.interpretability * 0.2 +
                self.fitness_components.generalization * 0.1
            )
            
            self.fitness = self.fitness_components.total
            self.raw_fitness = self.fitness
            
            return self.fitness
            
        except Exception as e:
            print(f"適応度評価エラー: {e}")
            self.fitness = 0.1
            return self.fitness
    
    def mutate(self, mutation_rate: float = 0.1) -> 'GeneticIndividual':
        """突然変異"""
        
        mutated = GeneticIndividual(self.individual_type)
        mutated.parents = [self.id]
        mutated.generation = self.generation + 1
        mutated.creation_method = "mutation"
        
        # ゲノム複製
        if self.genome is not None:
            mutated.genome = self.genome.copy()
            
            # 突然変異適用
            for i in range(len(mutated.genome)):
                if random.random() < mutation_rate:
                    mutated.genome[i] = random.random()
        
        # 木遺伝子複製と突然変異
        mutated.tree_genes = self._mutate_tree_genes(self.tree_genes, mutation_rate)
        
        mutated.mutation_count = self.mutation_count + 1
        
        return mutated
    
    def _mutate_tree_genes(self, original_genes: Dict[str, Any], 
                          mutation_rate: float) -> Dict[str, Any]:
        """木遺伝子の突然変異"""
        
        mutated_genes = {}
        
        for key, value in original_genes.items():
            if key == 'max_depth':
                if random.random() < mutation_rate:
                    mutated_genes[key] = random.randint(2, 8)
                else:
                    mutated_genes[key] = value
                    
            elif key == 'feature_selection_probs':
                if random.random() < mutation_rate:
                    mutated_genes[key] = np.random.dirichlet(np.ones(len(value)))
                else:
                    mutated_genes[key] = value.copy()
                    
            else:
                mutated_genes[key] = value
        
        return mutated_genes
    
    def crossover(self, other: 'GeneticIndividual', 
                 crossover_rate: float = 0.8) -> Tuple['GeneticIndividual', 'GeneticIndividual']:
        """交叉"""
        
        if random.random() > crossover_rate:
            return self, other
        
        # 子個体作成
        child1 = GeneticIndividual(self.individual_type)
        child2 = GeneticIndividual(self.individual_type)
        
        child1.parents = [self.id, other.id]
        child2.parents = [self.id, other.id]
        child1.generation = max(self.generation, other.generation) + 1
        child2.generation = max(self.generation, other.generation) + 1
        child1.creation_method = "crossover"
        child2.creation_method = "crossover"
        
        # ゲノム交叉（一点交叉）
        if self.genome is not None and other.genome is not None:
            crossover_point = random.randint(1, len(self.genome) - 1)
            
            child1.genome = np.concatenate([
                self.genome[:crossover_point],
                other.genome[crossover_point:]
            ])
            
            child2.genome = np.concatenate([
                other.genome[:crossover_point], 
                self.genome[crossover_point:]
            ])
        
        # 木遺伝子交叉
        child1.tree_genes = self._crossover_tree_genes(self.tree_genes, other.tree_genes)
        child2.tree_genes = self._crossover_tree_genes(other.tree_genes, self.tree_genes)
        
        child1.crossover_count = max(self.crossover_count, other.crossover_count) + 1
        child2.crossover_count = max(self.crossover_count, other.crossover_count) + 1
        
        return child1, child2
    
    def _crossover_tree_genes(self, genes1: Dict[str, Any], 
                             genes2: Dict[str, Any]) -> Dict[str, Any]:
        """木遺伝子の交叉"""
        
        offspring_genes = {}
        
        for key in genes1.keys():
            if key in genes2:
                # ランダムに親から選択
                if random.random() < 0.5:
                    offspring_genes[key] = genes1[key]
                else:
                    offspring_genes[key] = genes2[key]
            else:
                offspring_genes[key] = genes1[key]
        
        return offspring_genes
    
    def clone(self) -> 'GeneticIndividual':
        """個体のクローン作成"""
        
        clone = GeneticIndividual(self.individual_type)
        clone.parents = [self.id]
        clone.generation = self.generation
        clone.creation_method = "clone"
        
        if self.genome is not None:
            clone.genome = self.genome.copy()
        
        clone.tree_genes = {k: v for k, v in self.tree_genes.items()}
        clone.fitness_components = FitnessComponents(
            accuracy=self.fitness_components.accuracy,
            complexity=self.fitness_components.complexity,
            interpretability=self.fitness_components.interpretability,
            generalization=self.fitness_components.generalization,
            total=self.fitness_components.total
        )
        clone.fitness = self.fitness
        
        return clone
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式への変換"""
        
        return {
            'id': self.id,
            'individual_type': self.individual_type.value,
            'generation': self.generation,
            'age': self.age,
            'genome': self.genome.tolist() if self.genome is not None else None,
            'tree_genes': self.tree_genes,
            'fitness_components': {
                'accuracy': self.fitness_components.accuracy,
                'complexity': self.fitness_components.complexity,
                'interpretability': self.fitness_components.interpretability,
                'generalization': self.fitness_components.generalization,
                'total': self.fitness_components.total
            },
            'fitness': self.fitness,
            'evaluation_count': self.evaluation_count,
            'mutation_count': self.mutation_count,
            'crossover_count': self.crossover_count,
            'parents': self.parents,
            'creation_method': self.creation_method
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeneticIndividual':
        """辞書から個体を復元"""
        
        individual = cls(IndividualType(data.get('individual_type', 'hybrid')))
        individual.id = data.get('id', str(uuid.uuid4()))
        individual.generation = data.get('generation', 0)
        individual.age = data.get('age', 0)
        
        genome_data = data.get('genome')
        if genome_data:
            individual.genome = np.array(genome_data)
        
        individual.tree_genes = data.get('tree_genes', {})
        
        fitness_data = data.get('fitness_components', {})
        individual.fitness_components = FitnessComponents(
            accuracy=fitness_data.get('accuracy', 0.0),
            complexity=fitness_data.get('complexity', 0.0),
            interpretability=fitness_data.get('interpretability', 0.0),
            generalization=fitness_data.get('generalization', 0.0),
            total=fitness_data.get('total', 0.0)
        )
        
        individual.fitness = data.get('fitness', 0.0)
        individual.evaluation_count = data.get('evaluation_count', 0)
        individual.mutation_count = data.get('mutation_count', 0)
        individual.crossover_count = data.get('crossover_count', 0)
        individual.parents = data.get('parents', [])
        individual.creation_method = data.get('creation_method', 'unknown')
        
        return individual
