"""
遺伝的アルゴリズム - 個体（染色体）クラス
決定木構造とパラメータを遺伝子としてエンコード
12項目完全対応版
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
import copy


@dataclass
class FuzzyTreeGene:
    """ファジィ決定木の遺伝子（12項目対応）"""
    
    # Level 1 の特徴と閾値
    level1_feature: str = "research_intensity"
    level1_threshold_low: float = 0.4
    level1_threshold_high: float = 0.7
    
    # Level 2 の特徴（各ブランチ）
    level2_features: Dict[str, str] = field(default_factory=lambda: {
        "high": "team_work",
        "medium": "flexibility",
        "low": "lab_atmosphere"
    })
    
    # Level 2 の閾値
    level2_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "high": 0.7,
        "medium": 0.6,
        "low": 0.6
    })
    
    # メンバーシップ関数パラメータ（三角型）
    membership_params: Dict[str, Tuple[float, float, float]] = field(default_factory=lambda: {
        "low": (0.0, 0.0, 0.5),
        "medium": (0.3, 0.5, 0.7),
        "high": (0.5, 1.0, 1.0)
    })
    
    # 12項目の重要度（合計が1.0になるように正規化される）
    importance_weights: Dict[str, float] = field(default_factory=lambda: {
        "research_intensity": 0.13,
        "advisor_style": 0.09,
        "team_work": 0.09,
        "workload": 0.06,
        "theory_practice": 0.05,
        "research_field_match": 0.11,
        "skill_development": 0.07,
        "lab_atmosphere": 0.07,
        "flexibility": 0.06,
        "publication_opportunity": 0.05,
        "interdisciplinary": 0.11,
        "communication_style": 0.11
    })
    
    def normalize_weights(self):
        """重要度を正規化（合計を1.0に）"""
        total = sum(self.importance_weights.values())
        if total > 0:
            for key in self.importance_weights:
                self.importance_weights[key] /= total
    
    def validate(self) -> bool:
        """遺伝子の妥当性チェック"""
        if not (0 <= self.level1_threshold_low < self.level1_threshold_high <= 1.0):
            return False
        
        for label, (a, b, c) in self.membership_params.items():
            if not (a <= b <= c):
                return False
        
        if any(w < 0 for w in self.importance_weights.values()):
            return False
        
        return True
    
    def to_dict(self) -> Dict:
        """辞書形式に変換"""
        return {
            "level1_feature": self.level1_feature,
            "level1_threshold_low": self.level1_threshold_low,
            "level1_threshold_high": self.level1_threshold_high,
            "level2_features": self.level2_features.copy(),
            "level2_thresholds": self.level2_thresholds.copy(),
            "membership_params": self.membership_params.copy(),
            "importance_weights": self.importance_weights.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FuzzyTreeGene':
        """辞書から生成"""
        return cls(**data)


class Individual:
    """遺伝的アルゴリズムの個体（12項目対応）"""
    
    # 利用可能な特徴リスト（12項目）
    AVAILABLE_FEATURES = [
        "research_intensity", "advisor_style", "team_work",
        "workload", "theory_practice", "research_field_match",
        "skill_development", "lab_atmosphere", "flexibility",
        "publication_opportunity", "interdisciplinary",
        "communication_style"
    ]
    
    def __init__(self, gene: FuzzyTreeGene = None):
        """
        Args:
            gene: ファジィ決定木の遺伝子（Noneの場合はランダム生成）
        """
        if gene is None:
            self.gene = self.random_gene()
        else:
            self.gene = gene
        
        self.fitness: float = 0.0
        self.age: int = 0
    
    @classmethod
    def random_gene(cls) -> FuzzyTreeGene:
        """ランダムな遺伝子を生成"""
        gene = FuzzyTreeGene()
        
        # Level 1 特徴をランダム選択
        gene.level1_feature = random.choice(cls.AVAILABLE_FEATURES)
        
        # Level 1 閾値をランダム生成
        low = random.uniform(0.2, 0.5)
        high = random.uniform(0.6, 0.8)
        gene.level1_threshold_low = low
        gene.level1_threshold_high = high
        
        # Level 2 特徴をランダム選択（重複なし）
        available = [f for f in cls.AVAILABLE_FEATURES if f != gene.level1_feature]
        selected = random.sample(available, 3)
        gene.level2_features = {
            "high": selected[0],
            "medium": selected[1],
            "low": selected[2]
        }
        
        # Level 2 閾値をランダム生成
        gene.level2_thresholds = {
            "high": random.uniform(0.5, 0.8),
            "medium": random.uniform(0.4, 0.7),
            "low": random.uniform(0.4, 0.7)
        }
        
        # メンバーシップパラメータをランダム生成
        gene.membership_params = {
            "low": (0.0, 0.0, random.uniform(0.4, 0.6)),
            "medium": (
                random.uniform(0.2, 0.4),
                random.uniform(0.4, 0.6),
                random.uniform(0.6, 0.8)
            ),
            "high": (random.uniform(0.4, 0.6), 1.0, 1.0)
        }
        
        # 重要度をランダム生成（12項目）
        weights = {feature: random.uniform(0.01, 0.15) 
                   for feature in cls.AVAILABLE_FEATURES}
        gene.importance_weights = weights
        gene.normalize_weights()
        
        return gene
    
    def mutate(self, mutation_rate: float = 0.1):
        """突然変異
        
        Args:
            mutation_rate: 突然変異率
        """
        gene = self.gene
        
        # Level 1 特徴の突然変異
        if random.random() < mutation_rate:
            gene.level1_feature = random.choice(self.AVAILABLE_FEATURES)
        
        # Level 1 閾値の突然変異
        if random.random() < mutation_rate:
            gene.level1_threshold_low += random.gauss(0, 0.1)
            gene.level1_threshold_low = np.clip(gene.level1_threshold_low, 0.2, 0.5)
        
        if random.random() < mutation_rate:
            gene.level1_threshold_high += random.gauss(0, 0.1)
            gene.level1_threshold_high = np.clip(gene.level1_threshold_high, 0.6, 0.8)
        
        # Level 2 特徴の突然変異
        for branch in ["high", "medium", "low"]:
            if random.random() < mutation_rate:
                available = [f for f in self.AVAILABLE_FEATURES 
                           if f != gene.level1_feature]
                gene.level2_features[branch] = random.choice(available)
        
        # Level 2 閾値の突然変異
        for branch in ["high", "medium", "low"]:
            if random.random() < mutation_rate:
                gene.level2_thresholds[branch] += random.gauss(0, 0.1)
                gene.level2_thresholds[branch] = np.clip(
                    gene.level2_thresholds[branch], 0.3, 0.8
                )
        
        # メンバーシップパラメータの突然変異
        if random.random() < mutation_rate:
            for label in ["low", "medium", "high"]:
                a, b, c = gene.membership_params[label]
                
                if label == "low":
                    c = np.clip(c + random.gauss(0, 0.05), 0.4, 0.6)
                    gene.membership_params[label] = (0.0, 0.0, c)
                elif label == "medium":
                    a = np.clip(a + random.gauss(0, 0.05), 0.2, 0.4)
                    b = np.clip(b + random.gauss(0, 0.05), 0.4, 0.6)
                    c = np.clip(c + random.gauss(0, 0.05), 0.6, 0.8)
                    a, b, c = sorted([a, b, c])
                    gene.membership_params[label] = (a, b, c)
                else:  # high
                    a = np.clip(a + random.gauss(0, 0.05), 0.4, 0.6)
                    gene.membership_params[label] = (a, 1.0, 1.0)
        
        # 重要度の突然変異（12項目）
        if random.random() < mutation_rate:
            feature = random.choice(self.AVAILABLE_FEATURES)
            gene.importance_weights[feature] += random.gauss(0, 0.02)
            gene.importance_weights[feature] = max(0.01, gene.importance_weights[feature])
            gene.normalize_weights()
        
        if not gene.validate():
            self.gene = self.random_gene()
    
    def crossover(self, other: 'Individual') -> Tuple['Individual', 'Individual']:
        """交叉（一点交叉と一様交叉の組み合わせ）
        
        Args:
            other: 交叉相手の個体
            
        Returns:
            2つの子個体
        """
        child1_gene = FuzzyTreeGene()
        child2_gene = FuzzyTreeGene()
        
        # Level 1 特徴（一様交叉）
        if random.random() < 0.5:
            child1_gene.level1_feature = self.gene.level1_feature
            child2_gene.level1_feature = other.gene.level1_feature
        else:
            child1_gene.level1_feature = other.gene.level1_feature
            child2_gene.level1_feature = self.gene.level1_feature
        
        # Level 1 閾値（算術交叉）
        alpha = random.random()
        child1_gene.level1_threshold_low = (
            alpha * self.gene.level1_threshold_low +
            (1 - alpha) * other.gene.level1_threshold_low
        )
        child2_gene.level1_threshold_low = (
            (1 - alpha) * self.gene.level1_threshold_low +
            alpha * other.gene.level1_threshold_low
        )
        
        child1_gene.level1_threshold_high = (
            alpha * self.gene.level1_threshold_high +
            (1 - alpha) * other.gene.level1_threshold_high
        )
        child2_gene.level1_threshold_high = (
            (1 - alpha) * self.gene.level1_threshold_high +
            alpha * other.gene.level1_threshold_high
        )
        
        # Level 2 特徴（一様交叉）
        for branch in ["high", "medium", "low"]:
            if random.random() < 0.5:
                child1_gene.level2_features[branch] = self.gene.level2_features[branch]
                child2_gene.level2_features[branch] = other.gene.level2_features[branch]
            else:
                child1_gene.level2_features[branch] = other.gene.level2_features[branch]
                child2_gene.level2_features[branch] = self.gene.level2_features[branch]
            
            # Level 2 閾値（算術交叉）
            child1_gene.level2_thresholds[branch] = (
                alpha * self.gene.level2_thresholds[branch] +
                (1 - alpha) * other.gene.level2_thresholds[branch]
            )
            child2_gene.level2_thresholds[branch] = (
                (1 - alpha) * self.gene.level2_thresholds[branch] +
                alpha * other.gene.level2_thresholds[branch]
            )
        
        # メンバーシップパラメータ（算術交叉）
        for label in ["low", "medium", "high"]:
            a1, b1, c1 = self.gene.membership_params[label]
            a2, b2, c2 = other.gene.membership_params[label]
            
            child1_gene.membership_params[label] = (
                alpha * a1 + (1 - alpha) * a2,
                alpha * b1 + (1 - alpha) * b2,
                alpha * c1 + (1 - alpha) * c2
            )
            child2_gene.membership_params[label] = (
                (1 - alpha) * a1 + alpha * a2,
                (1 - alpha) * b1 + alpha * b2,
                (1 - alpha) * c1 + alpha * c2
            )
        
        # 重要度（一様交叉）- 12項目
        for feature in self.AVAILABLE_FEATURES:
            if random.random() < 0.5:
                child1_gene.importance_weights[feature] = self.gene.importance_weights[feature]
                child2_gene.importance_weights[feature] = other.gene.importance_weights[feature]
            else:
                child1_gene.importance_weights[feature] = other.gene.importance_weights[feature]
                child2_gene.importance_weights[feature] = self.gene.importance_weights[feature]
        
        # 正規化
        child1_gene.normalize_weights()
        child2_gene.normalize_weights()
        
        # 子個体を作成
        child1 = Individual(child1_gene)
        child2 = Individual(child2_gene)
        
        return child1, child2
    
    def copy(self) -> 'Individual':
        """個体のコピーを作成"""
        new_gene = FuzzyTreeGene.from_dict(self.gene.to_dict())
        new_individual = Individual(new_gene)
        new_individual.fitness = self.fitness
        new_individual.age = self.age
        return new_individual
    
    def __repr__(self) -> str:
        return f"Individual(fitness={self.fitness:.4f}, age={self.age})"
    
    def __lt__(self, other: 'Individual') -> bool:
        """比較演算子（適合度で比較）"""
        return self.fitness < other.fitness


# 使用例とテスト
if __name__ == "__main__":
    print("=" * 70)
    print("遺伝的アルゴリズム - 個体クラステスト（12項目対応）")
    print("=" * 70)
    
    # ランダム個体生成
    print("\n1. ランダム個体生成")
    ind1 = Individual()
    print(f"個体1: {ind1}")
    print(f"Level1特徴: {ind1.gene.level1_feature}")
    print(f"Level2特徴: {ind1.gene.level2_features}")
    print(f"重要度（トップ3）:")
    sorted_weights = sorted(
        ind1.gene.importance_weights.items(),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    for feature, weight in sorted_weights:
        print(f"  {feature}: {weight:.4f}")
    
    # 突然変異
    print("\n2. 突然変異")
    ind2 = ind1.copy()
    print(f"変異前: Level1={ind2.gene.level1_feature}")
    ind2.mutate(mutation_rate=0.5)
    print(f"変異後: Level1={ind2.gene.level1_feature}")
    
    # 交叉
    print("\n3. 交叉")
    ind3 = Individual()
    child1, child2 = ind1.crossover(ind3)
    print(f"親1: {ind1.gene.level1_feature}")
    print(f"親2: {ind3.gene.level1_feature}")
    print(f"子1: {child1.gene.level1_feature}")
    print(f"子2: {child2.gene.level1_feature}")
    
    print("\n" + "=" * 70)
    print(f"✅ 12項目対応個体クラス - テスト完了")