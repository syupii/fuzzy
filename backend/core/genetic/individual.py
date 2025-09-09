# core/genetic/individual.py - 個体クラス（完全版）

import numpy as np
import random
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

@dataclass
class GeneInfo:
    """遺伝子情報"""
    name: str
    value: float
    min_value: float = 0.0
    max_value: float = 1.0
    precision: int = 3
    mutation_strength: float = 0.1
    
    def normalize(self) -> float:
        """値を正規化"""
        if self.max_value == self.min_value:
            return 0.0
        normalized = (self.value - self.min_value) / (self.max_value - self.min_value)
        return max(0.0, min(1.0, normalized))
    
    def denormalize(self, normalized_value: float) -> float:
        """正規化値から実際の値に変換"""
        return self.min_value + normalized_value * (self.max_value - self.min_value)
    
    def mutate(self, mutation_rate: float = 0.1, strength: float = None) -> None:
        """遺伝子を変異"""
        if random.random() < mutation_rate:
            mut_strength = strength or self.mutation_strength
            noise = np.random.normal(0, mut_strength)
            self.value = max(self.min_value, min(self.max_value, self.value + noise))
            self.value = round(self.value, self.precision)

class Individual(ABC):
    """個体の抽象基底クラス"""
    
    def __init__(self, individual_id: str = None):
        self.individual_id = individual_id or f"ind_{int(time.time() * 1000000) % 1000000}"
        
        # 適応度関連
        self.fitness_value: Optional[float] = None
        self.raw_fitness: Optional[float] = None
        self.rank: Optional[int] = None
        
        # 世代管理
        self.generation: int = 0
        self.age: int = 0
        self.birth_time: float = time.time()
        
        # 系譜情報
        self.parent_ids: List[str] = []
        self.offspring_count: int = 0
        
        # 統計情報
        self.evaluation_count: int = 0
        self.last_evaluation_time: float = 0.0
        self.improvement_history: List[float] = []
        
        # フラグ
        self.is_elite: bool = False
        self.is_feasible: bool = True
        
        # 多様性関連
        self.diversity_score: Optional[float] = None
        self.cluster_id: Optional[str] = None
        
        # メタデータ
        self.metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def get_genes(self) -> Dict[str, float]:
        """遺伝子を取得"""
        pass
    
    @abstractmethod
    def set_genes(self, genes: Dict[str, float]) -> None:
        """遺伝子を設定"""
        pass
    
    @abstractmethod
    def clone(self) -> 'Individual':
        """個体を複製"""
        pass
    
    @abstractmethod
    def crossover(self, other: 'Individual', crossover_rate: float = 0.8) -> Tuple['Individual', 'Individual']:
        """交叉"""
        pass
    
    @abstractmethod
    def mutate(self, mutation_rate: float, mutation_strength: float = 0.1) -> None:
        """変異"""
        pass
    
    def get_fitness(self) -> float:
        """適応度を取得"""
        return self.fitness_value if self.fitness_value is not None else 0.0
    
    def set_fitness(self, fitness: float) -> None:
        """適応度を設定"""
        if self.fitness_value is not None:
            self.improvement_history.append(fitness - self.fitness_value)
        
        self.fitness_value = fitness
        self.evaluation_count += 1
        self.last_evaluation_time = time.time()
        
        # 改善履歴の制限（最新100件）
        if len(self.improvement_history) > 100:
            self.improvement_history = self.improvement_history[-100:]
    
    def is_evaluated(self) -> bool:
        """評価済みかどうか"""
        return self.fitness_value is not None
    
    def get_age(self) -> int:
        """年齢を取得"""
        return self.age
    
    def age_increment(self) -> None:
        """年齢を増加"""
        self.age += 1
    
    def get_diversity_score(self) -> float:
        """多様性スコアを取得"""
        return self.diversity_score if self.diversity_score is not None else 0.0
    
    def calculate_similarity(self, other: 'Individual') -> float:
        """他個体との類似度を計算"""
        genes1 = self.get_genes()
        genes2 = other.get_genes()
        
        if not genes1 or not genes2:
            return 0.0
        
        all_keys = set(genes1.keys()) | set(genes2.keys())
        if not all_keys:
            return 1.0
        
        distance = 0.0
        for key in all_keys:
            val1 = genes1.get(key, 0.0)
            val2 = genes2.get(key, 0.0)
            distance += (val1 - val2) ** 2
        
        similarity = 1.0 / (1.0 + np.sqrt(distance))
        return similarity
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "individual_id": self.individual_id,
            "fitness_value": self.fitness_value,
            "raw_fitness": self.raw_fitness,
            "rank": self.rank,
            "generation": self.generation,
            "age": self.age,
            "parent_ids": self.parent_ids,
            "offspring_count": self.offspring_count,
            "evaluation_count": self.evaluation_count,
            "is_elite": self.is_elite,
            "is_feasible": self.is_feasible,
            "diversity_score": self.diversity_score,
            "cluster_id": self.cluster_id,
            "genes": self.get_genes(),
            "metadata": self.metadata
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """辞書から復元"""
        self.individual_id = data.get("individual_id", self.individual_id)
        self.fitness_value = data.get("fitness_value")
        self.raw_fitness = data.get("raw_fitness")
        self.rank = data.get("rank")
        self.generation = data.get("generation", 0)
        self.age = data.get("age", 0)
        self.parent_ids = data.get("parent_ids", [])
        self.offspring_count = data.get("offspring_count", 0)
        self.evaluation_count = data.get("evaluation_count", 0)
        self.is_elite = data.get("is_elite", False)
        self.is_feasible = data.get("is_feasible", True)
        self.diversity_score = data.get("diversity_score")
        self.cluster_id = data.get("cluster_id")
        self.metadata = data.get("metadata", {})
        
        if "genes" in data:
            self.set_genes(data["genes"])

class WeightVector(Individual):
    """重みベクトル個体（研究室マッチング用）"""
    
    def __init__(self, individual_id: str = None, 
                 weight_names: List[str] = None):
        super().__init__(individual_id)
        
        # 重み名のデフォルト設定
        if weight_names is None:
            weight_names = [
                "research_intensity", "advisor_style", "team_work",
                "workload", "theory_practice", "research_field_match",
                "skill_development", "lab_atmosphere", "flexibility",
                "publication_opportunity", "interdisciplinary",
                "communication_style", "innovation_risk"
            ]
        
        self.weight_names = weight_names
        self.weights: Dict[str, GeneInfo] = {}
        
        # 重みの初期化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """重みを初期化"""
        for name in self.weight_names:
            # ランダムな初期重みを設定（0.0-1.0）
            initial_value = random.uniform(0.1, 1.0)
            self.weights[name] = GeneInfo(
                name=name,
                value=initial_value,
                min_value=0.0,
                max_value=1.0,
                precision=3
            )
        
        # 重みの正規化
        self._normalize_weights()
    
    def _normalize_weights(self):
        """重みを正規化（合計が1になるよう調整）"""
        total_weight = sum(gene.value for gene in self.weights.values())
        
        if total_weight > 0:
            for gene in self.weights.values():
                gene.value = gene.value / total_weight
                gene.value = round(gene.value, gene.precision)
    
    def get_genes(self) -> Dict[str, float]:
        """遺伝子（重み）を取得"""
        return {name: gene.value for name, gene in self.weights.items()}
    
    def set_genes(self, genes: Dict[str, float]):
        """遺伝子（重み）を設定"""
        for name, value in genes.items():
            if name in self.weights:
                self.weights[name].value = max(0.0, min(1.0, value))
        
        self._normalize_weights()
    
    def clone(self) -> 'WeightVector':
        """個体の複製を作成"""
        clone = WeightVector(weight_names=self.weight_names)
        clone.weights = {name: GeneInfo(
            name=gene.name,
            value=gene.value,
            min_value=gene.min_value,
            max_value=gene.max_value,
            precision=gene.precision,
            mutation_strength=gene.mutation_strength
        ) for name, gene in self.weights.items()}
        
        clone.fitness_value = self.fitness_value
        clone.raw_fitness = self.raw_fitness
        clone.generation = self.generation
        clone.parent_ids = [self.individual_id]
        
        return clone
    
    def crossover(self, other: 'WeightVector', crossover_rate: float = 0.8) -> Tuple['WeightVector', 'WeightVector']:
        """交叉"""
        if random.random() > crossover_rate:
            return self.clone(), other.clone()
        
        child1 = self.clone()
        child2 = other.clone()
        
        # 一様交叉
        for weight_name in self.weight_names:
            if weight_name in other.weights:
                if random.random() < 0.5:
                    # 重みを交換
                    temp_value = child1.weights[weight_name].value
                    child1.weights[weight_name].value = child2.weights[weight_name].value
                    child2.weights[weight_name].value = temp_value
        
        # 正規化
        child1._normalize_weights()
        child2._normalize_weights()
        
        # 親IDの設定
        child1.parent_ids = [self.individual_id, other.individual_id]
        child2.parent_ids = [self.individual_id, other.individual_id]
        
        # 適応度をリセット
        child1.fitness_value = None
        child1.raw_fitness = None
        child2.fitness_value = None
        child2.raw_fitness = None
        
        return child1, child2
    
    def mutate(self, mutation_rate: float, mutation_strength: float = 0.1):
        """変異"""
        for gene in self.weights.values():
            gene.mutate(mutation_rate, mutation_strength)
        
        # 正規化
        self._normalize_weights()
        
        # 適応度をリセット
        self.fitness_value = None
        self.raw_fitness = None
    
    def get_weight(self, weight_name: str) -> float:
        """特定の重みを取得"""
        if weight_name in self.weights:
            return self.weights[weight_name].value
        return 0.0
    
    def set_weight(self, weight_name: str, value: float):
        """特定の重みを設定"""
        if weight_name in self.weights:
            self.weights[weight_name].value = max(0.0, min(1.0, value))
            self._normalize_weights()
    
    def get_top_weights(self, n: int = 3) -> List[Tuple[str, float]]:
        """上位N個の重みを取得"""
        sorted_weights = sorted(self.weights.items(), key=lambda x: x[1].value, reverse=True)
        return [(name, gene.value) for name, gene in sorted_weights[:n]]
    
    def calculate_weight_diversity(self) -> float:
        """重みの多様性を計算"""
        values = [gene.value for gene in self.weights.values()]
        return float(np.std(values)) if values else 0.0

class FuzzyTreeIndividual(Individual):
    """ファジィ決定木個体"""
    
    def __init__(self, individual_id: str = None):
        super().__init__(individual_id)
        
        # 決定木関連の遅延インポート
        self.tree: Optional[Any] = None
        self.tree_parameters: Dict[str, Any] = {
            "max_depth": random.randint(3, 10),
            "min_samples_split": random.randint(2, 10),
            "min_samples_leaf": random.randint(1, 5),
            "fuzzy_threshold": random.uniform(0.05, 0.2)
        }
        
        # ルール重み
        self.rule_weights: Dict[str, float] = {}
        
        # 決定木特有の統計
        self.tree_complexity: float = 0.0
        self.tree_accuracy: float = 0.0
        self.rule_count: int = 0
    
    def get_genes(self) -> Dict[str, float]:
        """遺伝子（木パラメータ + ルール重み）を取得"""
        genes = {}
        
        # 木パラメータ（正規化）
        genes.update({
            "max_depth": self.tree_parameters["max_depth"] / 15.0,  # 正規化
            "min_samples_split": self.tree_parameters["min_samples_split"] / 20.0,
            "min_samples_leaf": self.tree_parameters["min_samples_leaf"] / 10.0,
            "fuzzy_threshold": self.tree_parameters["fuzzy_threshold"] / 0.5
        })
        
        # ルール重み
        genes.update(self.rule_weights)
        
        return genes
    
    def set_genes(self, genes: Dict[str, float]):
        """遺伝子を設定"""
        # 木パラメータの復元
        if "max_depth" in genes:
            self.tree_parameters["max_depth"] = max(1, int(genes["max_depth"] * 15))
        if "min_samples_split" in genes:
            self.tree_parameters["min_samples_split"] = max(2, int(genes["min_samples_split"] * 20))
        if "min_samples_leaf" in genes:
            self.tree_parameters["min_samples_leaf"] = max(1, int(genes["min_samples_leaf"] * 10))
        if "fuzzy_threshold" in genes:
            self.tree_parameters["fuzzy_threshold"] = max(0.01, genes["fuzzy_threshold"] * 0.5)
        
        # ルール重みの設定
        for key, value in genes.items():
            if key not in ["max_depth", "min_samples_split", "min_samples_leaf", "fuzzy_threshold"]:
                self.rule_weights[key] = max(0.0, min(1.0, value))
    
    def clone(self) -> 'FuzzyTreeIndividual':
        """個体の複製を作成"""
        clone = FuzzyTreeIndividual()
        clone.tree_parameters = self.tree_parameters.copy()
        clone.rule_weights = self.rule_weights.copy()
        clone.fitness_value = self.fitness_value
        clone.raw_fitness = self.raw_fitness
        clone.generation = self.generation
        clone.parent_ids = [self.individual_id]
        clone.tree_complexity = self.tree_complexity
        clone.tree_accuracy = self.tree_accuracy
        clone.rule_count = self.rule_count
        
        return clone
    
    def crossover(self, other: 'FuzzyTreeIndividual', crossover_rate: float = 0.8) -> Tuple['FuzzyTreeIndividual', 'FuzzyTreeIndividual']:
        """交叉"""
        if random.random() > crossover_rate:
            return self.clone(), other.clone()
        
        child1 = self.clone()
        child2 = other.clone()
        
        # パラメータの交叉
        for param_name in self.tree_parameters.keys():
            if random.random() < 0.5:
                temp = child1.tree_parameters[param_name]
                child1.tree_parameters[param_name] = child2.tree_parameters[param_name]
                child2.tree_parameters[param_name] = temp
        
        # ルール重みの交叉
        all_weights = set(self.rule_weights.keys()) | set(other.rule_weights.keys())
        for weight_name in all_weights:
            if random.random() < 0.5:
                weight1 = child1.rule_weights.get(weight_name, 0.5)
                weight2 = child2.rule_weights.get(weight_name, 0.5)
                child1.rule_weights[weight_name] = weight2
                child2.rule_weights[weight_name] = weight1
        
        # 親IDの設定
        child1.parent_ids = [self.individual_id, other.individual_id]
        child2.parent_ids = [self.individual_id, other.individual_id]
        
        # 適応度をリセット
        child1.fitness_value = None
        child1.raw_fitness = None
        child2.fitness_value = None
        child2.raw_fitness = None
        
        return child1, child2
    
    def mutate(self, mutation_rate: float, mutation_strength: float = 0.1):
        """変異"""
        # 木パラメータの変異
        if random.random() < mutation_rate:
            param_to_mutate = random.choice(list(self.tree_parameters.keys()))
            
            if param_to_mutate == "max_depth":
                self.tree_parameters["max_depth"] = max(1, min(15, 
                    self.tree_parameters["max_depth"] + random.randint(-2, 2)))
            elif param_to_mutate == "min_samples_split":
                self.tree_parameters["min_samples_split"] = max(2, min(20,
                    self.tree_parameters["min_samples_split"] + random.randint(-2, 2)))
            elif param_to_mutate == "min_samples_leaf":
                self.tree_parameters["min_samples_leaf"] = max(1, min(10,
                    self.tree_parameters["min_samples_leaf"] + random.randint(-1, 1)))
            elif param_to_mutate == "fuzzy_threshold":
                noise = random.uniform(-0.05, 0.05)
                self.tree_parameters["fuzzy_threshold"] = max(0.01, min(0.5,
                    self.tree_parameters["fuzzy_threshold"] + noise))
        
        # ルール重みの変異
        for weight_name in list(self.rule_weights.keys()):
            if random.random() < mutation_rate:
                noise = random.uniform(-mutation_strength, mutation_strength)
                self.rule_weights[weight_name] = max(0.0, min(1.0,
                    self.rule_weights[weight_name] + noise))
        
        # 適応度をリセット
        self.fitness_value = None
        self.raw_fitness = None
    
    def build_tree(self, training_data: Any) -> None:
        """決定木を構築"""
        try:
            from core.decision_tree.tree import FuzzyDecisionTree
            
            self.tree = FuzzyDecisionTree(
                max_depth=self.tree_parameters["max_depth"]
            )
            
            # 構築実行（実装依存）
            # self.tree.fit(training_data)
            
            # 統計更新
            if self.tree:
                self.tree_complexity = getattr(self.tree, 'complexity_score', 0.0)
                self.rule_count = len(getattr(self.tree, 'get_rules', lambda: [])())
                
        except ImportError:
            logger.warning("決定木モジュールが利用できません")
    
    def predict(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """予測を実行"""
        if self.tree and hasattr(self.tree, 'predict'):
            return self.tree.predict(sample)
        else:
            # フォールバック
            return {
                "predicted_class": "unknown",
                "confidence": 0.5,
                "class_probabilities": {"unknown": 0.5}
            }
    
    def get_tree_metrics(self) -> Dict[str, float]:
        """決定木メトリクスを取得"""
        return {
            "complexity": self.tree_complexity,
            "accuracy": self.tree_accuracy,
            "rule_count": self.rule_count,
            "max_depth": self.tree_parameters["max_depth"],
            "fuzzy_threshold": self.tree_parameters["fuzzy_threshold"]
        }

# 使用例とテスト
def test_individuals():
    """個体クラスのテスト"""
    
    print("🧬 遺伝的アルゴリズム個体テスト開始")
    
    # 重みベクトル個体のテスト
    print("\n📊 重みベクトル個体テスト:")
    weight_individual = WeightVector()
    print(f"  初期重み: {weight_individual.get_genes()}")
    
    # 適応度設定
    weight_individual.set_fitness(0.75)
    print(f"  適応度: {weight_individual.get_fitness()}")
    
    # 突然変異テスト
    weight_individual.mutate(0.5, 0.2)
    print(f"  変異後重み: {weight_individual.get_genes()}")
    
    # 交叉テスト
    other_individual = WeightVector()
    child1, child2 = weight_individual.crossover(other_individual, 0.8)
    print(f"  子1重み: {child1.get_genes()}")
    print(f"  子2重み: {child2.get_genes()}")
    
    # 上位重み表示
    top_weights = weight_individual.get_top_weights(3)
    print(f"  上位3重み: {top_weights}")
    
    # ファジィ決定木個体のテスト
    print("\n🌳 ファジィ決定木個体テスト:")
    tree_individual = FuzzyTreeIndividual()
    print(f"  初期パラメータ: {tree_individual.tree_parameters}")
    
    # 突然変異テスト
    tree_individual.mutate(0.3, 0.1)
    print(f"  変異後パラメータ: {tree_individual.tree_parameters}")
    
    # メトリクス表示
    metrics = tree_individual.get_tree_metrics()
    print(f"  木メトリクス: {metrics}")
    
    # 個体情報表示
    print(f"\n📋 個体情報:")
    info = weight_individual.to_dict()
    print(f"  個体ID: {info['individual_id']}")
    print(f"  世代: {info['generation']}")
    print(f"  評価回数: {info['evaluation_count']}")
    
    print("✅ 遺伝的アルゴリズム個体テスト完了")

if __name__ == "__main__":
    test_individuals()