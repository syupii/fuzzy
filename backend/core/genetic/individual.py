# core/genetic/individual.py - 遺伝的アルゴリズム個体クラス

import numpy as np
import random
import copy
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import uuid
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
    
    def normalize(self) -> float:
        """値を0-1に正規化"""
        if self.max_value == self.min_value:
            return 0.5
        return (self.value - self.min_value) / (self.max_value - self.min_value)
    
    def denormalize(self, normalized_value: float) -> float:
        """正規化値を元の範囲に戻す"""
        return self.min_value + normalized_value * (self.max_value - self.min_value)
    
    def mutate(self, mutation_rate: float, mutation_strength: float = 0.1):
        """遺伝子の突然変異"""
        if random.random() < mutation_rate:
            # ガウシアン変異
            noise = np.random.normal(0, mutation_strength)
            self.value = np.clip(
                self.value + noise * (self.max_value - self.min_value),
                self.min_value, 
                self.max_value
            )
            self.value = round(self.value, self.precision)

class Individual(ABC):
    """遺伝的アルゴリズム個体の抽象基底クラス"""
    
    def __init__(self, individual_id: str = None):
        self.individual_id = individual_id or str(uuid.uuid4())[:8]
        self.fitness_value: Optional[float] = None
        self.raw_fitness: Optional[float] = None
        self.generation: int = 0
        self.age: int = 0
        
        # 遺伝的操作履歴
        self.parent_ids: List[str] = []
        self.mutation_history: List[Dict[str, Any]] = []
        self.crossover_history: List[Dict[str, Any]] = []
        
        # 評価統計
        self.evaluation_count: int = 0
        self.last_evaluation_time: Optional[float] = None
        
        # メタデータ
        self.creation_time: float = 0.0
        self.is_elite: bool = False
        self.diversity_score: Optional[float] = None
    
    @abstractmethod
    def get_genes(self) -> Dict[str, float]:
        """遺伝子を取得"""
        pass
    
    @abstractmethod
    def set_genes(self, genes: Dict[str, float]):
        """遺伝子を設定"""
        pass
    
    @abstractmethod
    def clone(self) -> 'Individual':
        """個体の複製を作成"""
        pass
    
    @abstractmethod
    def mutate(self, mutation_rate: float, mutation_strength: float = 0.1):
        """突然変異"""
        pass
    
    @abstractmethod
    def crossover(self, other: 'Individual', crossover_rate: float = 0.8) -> Tuple['Individual', 'Individual']:
        """交叉"""
        pass
    
    def get_fitness(self) -> Optional[float]:
        """適応度を取得"""
        return self.fitness_value
    
    def set_fitness(self, fitness: float):
        """適応度を設定"""
        self.fitness_value = fitness
        self.evaluation_count += 1
    
    def get_raw_fitness(self) -> Optional[float]:
        """生の適応度を取得"""
        return self.raw_fitness
    
    def set_raw_fitness(self, raw_fitness: float):
        """生の適応度を設定"""
        self.raw_fitness = raw_fitness
    
    def is_evaluated(self) -> bool:
        """評価済みかどうか"""
        return self.fitness_value is not None
    
    def get_genome_size(self) -> int:
        """ゲノムサイズを取得"""
        return len(self.get_genes())
    
    def get_diversity_from(self, other: 'Individual') -> float:
        """他個体との多様性を計算"""
        self_genes = self.get_genes()
        other_genes = other.get_genes()
        
        if not self_genes or not other_genes:
            return 0.0
        
        # ユークリッド距離ベースの多様性
        total_distance = 0.0
        common_genes = set(self_genes.keys()) & set(other_genes.keys())
        
        if not common_genes:
            return 1.0  # 全く異なる遺伝子構成
        
        for gene_name in common_genes:
            distance = abs(self_genes[gene_name] - other_genes[gene_name])
            total_distance += distance ** 2
        
        return (total_distance / len(common_genes)) ** 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "individual_id": self.individual_id,
            "fitness_value": self.fitness_value,
            "raw_fitness": self.raw_fitness,
            "generation": self.generation,
            "age": self.age,
            "genes": self.get_genes(),
            "evaluation_count": self.evaluation_count,
            "is_elite": self.is_elite,
            "diversity_score": self.diversity_score
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """辞書から復元"""
        self.individual_id = data.get("individual_id", self.individual_id)
        self.fitness_value = data.get("fitness_value")
        self.raw_fitness = data.get("raw_fitness")
        self.generation = data.get("generation", 0)
        self.age = data.get("age", 0)
        self.evaluation_count = data.get("evaluation_count", 0)
        self.is_elite = data.get("is_elite", False)
        self.diversity_score = data.get("diversity_score")
        
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
        clone = WeightVector(weight_names=self.weight_names.copy())
        clone.set_genes(self.get_genes())
        
        # メタデータのコピー
        clone.fitness_value = self.fitness_value
        clone.raw_fitness = self.raw_fitness
        clone.generation = self.generation
        clone.parent_ids = [self.individual_id]
        
        return clone
    
    def mutate(self, mutation_rate: float, mutation_strength: float = 0.1):
        """突然変異"""
        mutation_applied = False
        
        for gene in self.weights.values():
            if random.random() < mutation_rate:
                old_value = gene.value
                gene.mutate(1.0, mutation_strength)  # 選択された遺伝子は必ず変異
                mutation_applied = True
                
                # 変異履歴記録
                self.mutation_history.append({
                    "gene": gene.name,
                    "old_value": old_value,
                    "new_value": gene.value,
                    "generation": self.generation
                })
        
        if mutation_applied:
            self._normalize_weights()
            # 適応度をリセット
            self.fitness_value = None
            self.raw_fitness = None
    
    def crossover(self, other: 'WeightVector', crossover_rate: float = 0.8) -> Tuple['WeightVector', 'WeightVector']:
        """一様交叉"""
        if random.random() > crossover_rate:
            return self.clone(), other.clone()
        
        child1 = self.clone()
        child2 = other.clone()
        
        # 一様交叉を実行
        for gene_name in self.weight_names:
            if random.random() < 0.5:
                # 遺伝子を交換
                if gene_name in self.weights and gene_name in other.weights:
                    temp_value = child1.weights[gene_name].value
                    child1.weights[gene_name].value = child2.weights[gene_name].value
                    child2.weights[gene_name].value = temp_value
        
        # 正規化
        child1._normalize_weights()
        child2._normalize_weights()
        
        # 交叉履歴記録
        crossover_record = {
            "other_parent": other.individual_id,
            "generation": self.generation,
            "method": "uniform"
        }
        child1.crossover_history.append(crossover_record)
        child2.crossover_history.append(crossover_record)
        
        # 親IDの設定
        child1.parent_ids = [self.individual_id, other.individual_id]
        child2.parent_ids = [self.individual_id, other.individual_id]
        
        # 適応度をリセット
        child1.fitness_value = None
        child1.raw_fitness = None
        child2.fitness_value = None
        child2.raw_fitness = None
        
        return child1, child2
    
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

class FuzzyTreeIndividual(Individual):
    """ファジィ決定木個体"""
    
    def __init__(self, individual_id: str = None):
        super().__init__(individual_id)
        
        from core.decision_tree.tree import FuzzyDecisionTree
        self.tree: Optional[FuzzyDecisionTree] = None
        self.tree_parameters: Dict[str, Any] = {
            "max_depth": random.randint(3, 10),
            "min_samples_split": random.randint(2, 10),
            "min_samples_leaf": random.randint(1, 5),
            "fuzzy_threshold": random.uniform(0.05, 0.2)
        }
        
        # ルール重み
        self.rule_weights: Dict[str, float] = {}
    
    def get_genes(self) -> Dict[str, float]:
        """遺伝子（木パラメータ + ルール重み）を取得"""
        genes = {}
        
        # 木パラメータ
        genes.update({
            "max_depth": self.tree_parameters["max_depth"] / 10.0,  # 正規化
            "min_samples_split": self.tree_parameters["min_samples_split"] / 10.0,
            "min_samples_leaf": self.tree_parameters["min_samples_leaf"] / 5.0,
            "fuzzy_threshold": self.tree_parameters["fuzzy_threshold"] / 0.2
        })
        
        # ルール重み
        genes.update(self.rule_weights)
        
        return genes
    
    def set_genes(self, genes: Dict[str, float]):
        """遺伝子を設定"""
        # 木パラメータの復元
        if "max_depth" in genes:
            self.tree_parameters["max_depth"] = max(1, int(genes["max_depth"] * 10))
        if "min_samples_split" in genes:
            self.tree_parameters["min_samples_split"] = max(2, int(genes["min_samples_split"] * 10))
        if "min_samples_leaf" in genes:
            self.tree_parameters["min_samples_leaf"] = max(1, int(genes["min_samples_leaf"] * 5))
        if "fuzzy_threshold" in genes:
            self.tree_parameters["fuzzy_threshold"] = max(0.01, genes["fuzzy_threshold"] * 0.2)
        
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
        
        return clone
    
    def mutate(self, mutation_rate: float, mutation_strength: float = 0.1):
        """突然変異"""
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

# 使用例とテスト
def test_individuals():
    """個体クラスのテスト"""
    
    print("🧬 遺伝的アルゴリズム個体テスト開始")
    
    # 重みベクトル個体のテスト
    print("\n📊 重みベクトル個体テスト:")
    weight_individual = WeightVector()
    print(f"  初期重み: {weight_individual.get_genes()}")
    
    # 突然変異テスト
    weight_individual.mutate(0.5, 0.2)
    print(f"  変異後重み: {weight_individual.get_genes()}")
    
    # 交叉テスト
    other_individual = WeightVector()
    child1, child2 = weight_individual.crossover(other_individual, 0.8)
    print(f"  子1重み: {child1.get_genes()}")
    print(f"  子2重み: {child2.get_genes()}")
    
    # ファジィ決定木個体のテスト
    print("\n🌳 ファジィ決定木個体テスト:")
    tree_individual = FuzzyTreeIndividual()
    print(f"  初期パラメータ: {tree_individual.tree_parameters}")
    
    # 突然変異テスト
    tree_individual.mutate(0.3, 0.1)
    print(f"  変異後パラメータ: {tree_individual.tree_parameters}")
    
    print("✅ 遺伝的アルゴリズム個体テスト完了")

if __name__ == "__main__":
    test_individuals()