# core/genetic/individual.py - 遺伝的アルゴリズムの個体クラス

import random
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from copy import deepcopy

@dataclass
class Individual:
    """遺伝的アルゴリズムの個体"""
    
    # 遺伝子（重み）
    field_weights: Dict[str, float] = field(default_factory=dict)      # 研究分野重み
    criteria_weights: Dict[str, float] = field(default_factory=dict)   # 評価基準重み
    
    # 適応度情報
    fitness: float = 0.0                                               # 適応度
    raw_fitness: float = 0.0                                          # 生の適応度
    normalized_fitness: float = 0.0                                   # 正規化適応度
    
    # 遺伝的情報
    generation: int = 0                                                # 世代数
    parent_ids: List[str] = field(default_factory=list)              # 親のID
    individual_id: str = ""                                           # 個体ID
    
    # 評価履歴
    evaluation_history: List[float] = field(default_factory=list)    # 適応度履歴
    
    def __post_init__(self):
        """初期化後の処理"""
        if not self.individual_id:
            self.individual_id = self._generate_id()
    
    def _generate_id(self) -> str:
        """個体IDを生成"""
        import uuid
        return f"IND_{str(uuid.uuid4())[:8]}"
    
    def initialize_random(self, research_fields: List[str], 
                         evaluation_criteria: List[str]) -> None:
        """ランダムに遺伝子を初期化"""
        
        # 研究分野重みの初期化
        self.field_weights = {
            field_id: random.uniform(0.1, 1.0) 
            for field_id in research_fields
        }
        
        # 評価基準重みの初期化
        self.criteria_weights = {
            criterion: random.uniform(0.1, 1.0)
            for criterion in evaluation_criteria
        }
        
        # 重みを正規化
        self._normalize_weights()
    
    def initialize_from_profile(self, student_profile: 'StudentProfile',
                              research_fields: List[str],
                              evaluation_criteria: List[str]) -> None:
        """学生プロフィールから遺伝子を初期化"""
        
        # 学生の分野興味から重みを設定
        student_fields = {fi.field_id: fi for fi in student_profile.field_interests}
        
        for field_id in research_fields:
            if field_id in student_fields:
                # 興味度と重要度を組み合わせて重み計算
                interest = student_fields[field_id].interest_level / 10.0
                importance = student_fields[field_id].importance_level / 10.0
                self.field_weights[field_id] = (interest * 0.7 + importance * 0.3)
            else:
                # 未選択分野は低い重み
                self.field_weights[field_id] = random.uniform(0.1, 0.3)
        
        # 評価基準重みは学生の値を基準に設定
        student_criteria = student_profile.evaluation_criteria.dict()
        
        for criterion in evaluation_criteria:
            if criterion in student_criteria:
                # 学生の重視度を重みとして使用（正規化）
                base_weight = student_criteria[criterion] / 10.0
                # 少しランダム性を加える
                noise = random.uniform(-0.1, 0.1)
                self.criteria_weights[criterion] = max(0.1, min(1.0, base_weight + noise))
            else:
                self.criteria_weights[criterion] = random.uniform(0.3, 0.7)
        
        # 重みを正規化
        self._normalize_weights()
    
    def _normalize_weights(self) -> None:
        """重みを正規化"""
        
        # 分野重みの正規化
        field_sum = sum(self.field_weights.values())
        if field_sum > 0:
            for field_id in self.field_weights:
                self.field_weights[field_id] /= field_sum
        
        # 評価基準重みの正規化
        criteria_sum = sum(self.criteria_weights.values())
        if criteria_sum > 0:
            for criterion in self.criteria_weights:
                self.criteria_weights[criterion] /= criteria_sum
    
    def mutate(self, mutation_rate: float, mutation_strength: float = 0.1) -> None:
        """変異操作"""
        
        # 分野重み変異
        for field_id in self.field_weights:
            if random.random() < mutation_rate:
                noise = random.uniform(-mutation_strength, mutation_strength)
                self.field_weights[field_id] += noise
                self.field_weights[field_id] = max(0.01, min(1.0, self.field_weights[field_id]))
        
        # 評価基準重み変異
        for criterion in self.criteria_weights:
            if random.random() < mutation_rate:
                noise = random.uniform(-mutation_strength, mutation_strength)
                self.criteria_weights[criterion] += noise
                self.criteria_weights[criterion] = max(0.01, min(1.0, self.criteria_weights[criterion]))
        
        # 重みを再正規化
        self._normalize_weights()
    
    def crossover_with(self, other: 'Individual', crossover_rate: float = 0.8) -> Tuple['Individual', 'Individual']:
        """他の個体との交叉"""
        
        if random.random() > crossover_rate:
            return deepcopy(self), deepcopy(other)
        
        # 子個体を作成
        child1 = Individual()
        child2 = Individual()
        
        child1.generation = max(self.generation, other.generation) + 1
        child2.generation = max(self.generation, other.generation) + 1
        
        child1.parent_ids = [self.individual_id, other.individual_id]
        child2.parent_ids = [self.individual_id, other.individual_id]
        
        # 分野重み交叉（一様交叉）
        child1.field_weights = {}
        child2.field_weights = {}
        
        for field_id in self.field_weights:
            if random.random() < 0.5:
                child1.field_weights[field_id] = self.field_weights[field_id]
                child2.field_weights[field_id] = other.field_weights[field_id]
            else:
                child1.field_weights[field_id] = other.field_weights[field_id]
                child2.field_weights[field_id] = self.field_weights[field_id]
        
        # 評価基準重み交叉（一様交叉）
        child1.criteria_weights = {}
        child2.criteria_weights = {}
        
        for criterion in self.criteria_weights:
            if random.random() < 0.5:
                child1.criteria_weights[criterion] = self.criteria_weights[criterion]
                child2.criteria_weights[criterion] = other.criteria_weights[criterion]
            else:
                child1.criteria_weights[criterion] = other.criteria_weights[criterion]
                child2.criteria_weights[criterion] = self.criteria_weights[criterion]
        
        # 重みを正規化
        child1._normalize_weights()
        child2._normalize_weights()
        
        return child1, child2
    
    def blend_crossover_with(self, other: 'Individual', alpha: float = 0.5) -> Tuple['Individual', 'Individual']:
        """ブレンド交叉（連続値用）"""
        
        child1 = Individual()
        child2 = Individual()
        
        child1.generation = max(self.generation, other.generation) + 1
        child2.generation = max(self.generation, other.generation) + 1
        
        child1.parent_ids = [self.individual_id, other.individual_id]
        child2.parent_ids = [self.individual_id, other.individual_id]
        
        # 分野重みのブレンド交叉
        child1.field_weights = {}
        child2.field_weights = {}
        
        for field_id in self.field_weights:
            val1 = self.field_weights[field_id]
            val2 = other.field_weights[field_id]
            
            # ブレンド範囲計算
            min_val = min(val1, val2)
            max_val = max(val1, val2)
            range_val = max_val - min_val
            
            lower_bound = max(0.01, min_val - alpha * range_val)
            upper_bound = min(1.0, max_val + alpha * range_val)
            
            child1.field_weights[field_id] = random.uniform(lower_bound, upper_bound)
            child2.field_weights[field_id] = random.uniform(lower_bound, upper_bound)
        
        # 評価基準重みのブレンド交叉
        child1.criteria_weights = {}
        child2.criteria_weights = {}
        
        for criterion in self.criteria_weights:
            val1 = self.criteria_weights[criterion]
            val2 = other.criteria_weights[criterion]
            
            min_val = min(val1, val2)
            max_val = max(val1, val2)
            range_val = max_val - min_val
            
            lower_bound = max(0.01, min_val - alpha * range_val)
            upper_bound = min(1.0, max_val + alpha * range_val)
            
            child1.criteria_weights[criterion] = random.uniform(lower_bound, upper_bound)
            child2.criteria_weights[criterion] = random.uniform(lower_bound, upper_bound)
        
        # 重みを正規化
        child1._normalize_weights()
        child2._normalize_weights()
        
        return child1, child2
    
    def update_fitness(self, new_fitness: float) -> None:
        """適応度を更新"""
        self.evaluation_history.append(new_fitness)
        self.raw_fitness = new_fitness
        self.fitness = new_fitness
    
    def calculate_diversity(self, other: 'Individual') -> float:
        """他の個体との多様性を計算"""
        
        # 分野重みの差分
        field_diff = sum(
            abs(self.field_weights.get(field_id, 0) - other.field_weights.get(field_id, 0))
            for field_id in set(self.field_weights.keys()) | set(other.field_weights.keys())
        )
        
        # 評価基準重みの差分
        criteria_diff = sum(
            abs(self.criteria_weights.get(criterion, 0) - other.criteria_weights.get(criterion, 0))
            for criterion in set(self.criteria_weights.keys()) | set(other.criteria_weights.keys())
        )
        
        # 正規化された多様性
        total_variables = len(self.field_weights) + len(self.criteria_weights)
        diversity = (field_diff + criteria_diff) / (2 * total_variables) if total_variables > 0 else 0
        
        return diversity
    
    def get_dominant_fields(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """重みの高い分野を取得"""
        sorted_fields = sorted(
            self.field_weights.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_fields[:top_k]
    
    def get_dominant_criteria(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """重みの高い評価基準を取得"""
        sorted_criteria = sorted(
            self.criteria_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_criteria[:top_k]
    
    def get_weight_distribution_stats(self) -> Dict[str, Any]:
        """重み分布の統計情報を取得"""
        
        field_weights_list = list(self.field_weights.values())
        criteria_weights_list = list(self.criteria_weights.values())
        
        stats = {
            "field_weights": {
                "mean": np.mean(field_weights_list),
                "std": np.std(field_weights_list),
                "min": np.min(field_weights_list),
                "max": np.max(field_weights_list),
                "entropy": self._calculate_entropy(field_weights_list)
            },
            "criteria_weights": {
                "mean": np.mean(criteria_weights_list),
                "std": np.std(criteria_weights_list),
                "min": np.min(criteria_weights_list),
                "max": np.max(criteria_weights_list),
                "entropy": self._calculate_entropy(criteria_weights_list)
            },
            "fitness_trend": {
                "current": self.fitness,
                "best_ever": max(self.evaluation_history) if self.evaluation_history else 0,
                "evaluations": len(self.evaluation_history),
                "improvement": self._calculate_improvement_rate()
            }
        }
        
        return stats
    
    def _calculate_entropy(self, weights: List[float]) -> float:
        """重み分布のエントロピーを計算（多様性指標）"""
        if not weights or sum(weights) == 0:
            return 0.0
        
        # 正規化
        total = sum(weights)
        probabilities = [w / total for w in weights]
        
        # エントロピー計算
        entropy = -sum(p * np.log(p + 1e-10) for p in probabilities if p > 0)
        
        # 最大エントロピーで正規化
        max_entropy = np.log(len(weights))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_improvement_rate(self) -> float:
        """適応度改善率を計算"""
        if len(self.evaluation_history) < 2:
            return 0.0
        
        recent_evals = self.evaluation_history[-5:]  # 最近5回の評価
        if len(recent_evals) < 2:
            return 0.0
        
        # 線形回帰で傾きを計算
        x = np.arange(len(recent_evals))
        y = np.array(recent_evals)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return slope
        
        return 0.0
    
    def clone(self) -> 'Individual':
        """個体を複製"""
        clone = deepcopy(self)
        clone.individual_id = clone._generate_id()
        clone.parent_ids = [self.individual_id]
        return clone
    
    def distance_to(self, other: 'Individual') -> float:
        """他の個体との遺伝的距離"""
        return 1.0 - self.calculate_diversity(other)
    
    def is_similar_to(self, other: 'Individual', threshold: float = 0.1) -> bool:
        """他の個体との類似性判定"""
        return self.calculate_diversity(other) < threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式で出力"""
        return {
            "individual_id": self.individual_id,
            "generation": self.generation,
            "fitness": self.fitness,
            "field_weights": self.field_weights,
            "criteria_weights": self.criteria_weights,
            "stats": self.get_weight_distribution_stats()
        }
    
    def __str__(self) -> str:
        """文字列表現"""
        return f"Individual({self.individual_id}, Gen:{self.generation}, Fitness:{self.fitness:.4f})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __lt__(self, other: 'Individual') -> bool:
        """適応度での比較（ソート用）"""
        return self.fitness < other.fitness
    
    def __eq__(self, other: 'Individual') -> bool:
        """個体IDでの等価性判定"""
        if not isinstance(other, Individual):
            return False
        return self.individual_id == other.individual_id
    
    def __hash__(self) -> int:
        """ハッシュ値（セット操作用）"""
        return hash(self.individual_id)