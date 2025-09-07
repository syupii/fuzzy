# core/fuzzy/membership.py - メンバーシップ関数

import numpy as np
import math
from typing import Dict, List, Tuple, Callable, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class MembershipType(str, Enum):
    """メンバーシップ関数の種類"""
    TRIANGULAR = "triangular"
    TRAPEZOIDAL = "trapezoidal"
    GAUSSIAN = "gaussian"
    SIGMOID = "sigmoid"
    BELL = "bell"

@dataclass
class MembershipParams:
    """メンバーシップ関数のパラメータ"""
    a: float  # 左端または中心
    b: Optional[float] = None  # 右端または幅
    c: Optional[float] = None  # 第3パラメータ
    d: Optional[float] = None  # 第4パラメータ

class MembershipFunction(ABC):
    """メンバーシップ関数の抽象基底クラス"""
    
    def __init__(self, name: str, params: MembershipParams):
        self.name = name
        self.params = params
    
    @abstractmethod
    def membership(self, x: float) -> float:
        """メンバーシップ度を計算"""
        pass
    
    def batch_membership(self, x_values: List[float]) -> List[float]:
        """複数値のメンバーシップ度を一括計算"""
        return [self.membership(x) for x in x_values]

class TriangularMF(MembershipFunction):
    """三角形メンバーシップ関数"""
    
    def __init__(self, name: str, a: float, b: float, c: float):
        """
        Args:
            a: 左端
            b: 頂点
            c: 右端
        """
        super().__init__(name, MembershipParams(a, b, c))
        self.a, self.b, self.c = a, b, c
    
    def membership(self, x: float) -> float:
        """三角形メンバーシップ度計算"""
        if x <= self.a or x >= self.c:
            return 0.0
        elif self.a < x <= self.b:
            return (x - self.a) / (self.b - self.a) if self.b != self.a else 1.0
        else:  # self.b < x < self.c
            return (self.c - x) / (self.c - self.b) if self.c != self.b else 1.0

class TrapezoidalMF(MembershipFunction):
    """台形メンバーシップ関数"""
    
    def __init__(self, name: str, a: float, b: float, c: float, d: float):
        """
        Args:
            a: 左端
            b: 左上端
            c: 右上端
            d: 右端
        """
        super().__init__(name, MembershipParams(a, b, c, d))
        self.a, self.b, self.c, self.d = a, b, c, d
    
    def membership(self, x: float) -> float:
        """台形メンバーシップ度計算"""
        if x <= self.a or x >= self.d:
            return 0.0
        elif self.a < x <= self.b:
            return (x - self.a) / (self.b - self.a) if self.b != self.a else 1.0
        elif self.b < x <= self.c:
            return 1.0
        else:  # self.c < x < self.d
            return (self.d - x) / (self.d - self.c) if self.d != self.c else 1.0

class GaussianMF(MembershipFunction):
    """ガウシアンメンバーシップ関数"""
    
    def __init__(self, name: str, center: float, sigma: float):
        """
        Args:
            center: 中心
            sigma: 標準偏差
        """
        super().__init__(name, MembershipParams(center, sigma))
        self.center = center
        self.sigma = sigma
    
    def membership(self, x: float) -> float:
        """ガウシアンメンバーシップ度計算"""
        return math.exp(-0.5 * ((x - self.center) / self.sigma) ** 2)

class SigmoidMF(MembershipFunction):
    """シグモイドメンバーシップ関数"""
    
    def __init__(self, name: str, a: float, c: float):
        """
        Args:
            a: 傾き
            c: 中心点
        """
        super().__init__(name, MembershipParams(a, c))
        self.a = a
        self.c = c
    
    def membership(self, x: float) -> float:
        """シグモイドメンバーシップ度計算"""
        return 1.0 / (1.0 + math.exp(-self.a * (x - self.c)))

class BellMF(MembershipFunction):
    """ベル型メンバーシップ関数"""
    
    def __init__(self, name: str, a: float, b: float, c: float):
        """
        Args:
            a: 幅パラメータ
            b: 形状パラメータ
            c: 中心
        """
        super().__init__(name, MembershipParams(a, b, c))
        self.a, self.b, self.c = a, b, c
    
    def membership(self, x: float) -> float:
        """ベル型メンバーシップ度計算"""
        return 1.0 / (1.0 + abs((x - self.c) / self.a) ** (2 * self.b))

class FuzzySet:
    """ファジィ集合クラス"""
    
    def __init__(self, name: str, membership_function: MembershipFunction, 
                 universe: Tuple[float, float] = (0.0, 10.0)):
        self.name = name
        self.membership_function = membership_function
        self.universe = universe  # 議論域
        
    def membership(self, x: float) -> float:
        """メンバーシップ度を取得"""
        return self.membership_function.membership(x)
    
    def alpha_cut(self, alpha: float, num_points: int = 1000) -> List[float]:
        """α-カットを計算"""
        x_range = np.linspace(self.universe[0], self.universe[1], num_points)
        return [x for x in x_range if self.membership(x) >= alpha]
    
    def support(self, threshold: float = 0.001) -> List[float]:
        """サポート（台）を計算"""
        return self.alpha_cut(threshold)
    
    def height(self) -> float:
        """高さ（最大メンバーシップ度）を計算"""
        x_range = np.linspace(self.universe[0], self.universe[1], 1000)
        return max(self.membership(x) for x in x_range)
    
    def centroid(self, num_points: int = 1000) -> float:
        """重心を計算"""
        x_range = np.linspace(self.universe[0], self.universe[1], num_points)
        numerator = sum(x * self.membership(x) for x in x_range)
        denominator = sum(self.membership(x) for x in x_range)
        
        return numerator / denominator if denominator > 0 else 0.0

class FuzzyVariable:
    """ファジィ変数クラス"""
    
    def __init__(self, name: str, universe: Tuple[float, float] = (0.0, 10.0)):
        self.name = name
        self.universe = universe
        self.sets: Dict[str, FuzzySet] = {}
    
    def add_set(self, fuzzy_set: FuzzySet):
        """ファジィ集合を追加"""
        self.sets[fuzzy_set.name] = fuzzy_set
    
    def get_membership(self, value: float) -> Dict[str, float]:
        """全ての集合に対するメンバーシップ度を取得"""
        return {name: fs.membership(value) for name, fs in self.sets.items()}
    
    def fuzzify(self, value: float) -> Dict[str, float]:
        """ファジィ化"""
        return self.get_membership(value)
    
    def defuzzify(self, membership_values: Dict[str, float], 
                  method: str = "centroid") -> float:
        """非ファジィ化"""
        
        if method == "centroid":
            return self._centroid_defuzzify(membership_values)
        elif method == "max":
            return self._max_defuzzify(membership_values)
        elif method == "mean_of_max":
            return self._mean_of_max_defuzzify(membership_values)
        else:
            raise ValueError(f"未知の非ファジィ化手法: {method}")
    
    def _centroid_defuzzify(self, membership_values: Dict[str, float]) -> float:
        """重心法による非ファジィ化"""
        x_range = np.linspace(self.universe[0], self.universe[1], 1000)
        
        # 各点での結合メンバーシップ度を計算
        combined_membership = []
        for x in x_range:
            max_membership = 0.0
            for set_name, activation in membership_values.items():
                if set_name in self.sets:
                    set_membership = self.sets[set_name].membership(x)
                    combined = min(activation, set_membership)
                    max_membership = max(max_membership, combined)
            combined_membership.append(max_membership)
        
        # 重心計算
        numerator = sum(x * m for x, m in zip(x_range, combined_membership))
        denominator = sum(combined_membership)
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def _max_defuzzify(self, membership_values: Dict[str, float]) -> float:
        """最大値法による非ファジィ化"""
        max_activation = max(membership_values.values())
        max_sets = [name for name, val in membership_values.items() 
                   if val == max_activation]
        
        if len(max_sets) == 1:
            return self.sets[max_sets[0]].centroid()
        else:
            # 複数の最大値がある場合は平均
            centroids = [self.sets[name].centroid() for name in max_sets]
            return sum(centroids) / len(centroids)
    
    def _mean_of_max_defuzzify(self, membership_values: Dict[str, float]) -> float:
        """最大値平均法による非ファジィ化"""
        return self._max_defuzzify(membership_values)

class MembershipFunctionFactory:
    """メンバーシップ関数ファクトリ"""
    
    @staticmethod
    def create_standard_sets(variable_name: str, 
                           universe: Tuple[float, float] = (1.0, 10.0)) -> FuzzyVariable:
        """標準的な3分割ファジィ集合を作成"""
        
        var = FuzzyVariable(variable_name, universe)
        min_val, max_val = universe
        range_val = max_val - min_val
        
        # Low, Medium, High の3つの集合
        low_set = FuzzySet(
            "low",
            TriangularMF("low", min_val, min_val, min_val + range_val * 0.5),
            universe
        )
        
        medium_set = FuzzySet(
            "medium", 
            TriangularMF("medium", min_val + range_val * 0.25, 
                        min_val + range_val * 0.5, min_val + range_val * 0.75),
            universe
        )
        
        high_set = FuzzySet(
            "high",
            TriangularMF("high", min_val + range_val * 0.5, max_val, max_val),
            universe
        )
        
        var.add_set(low_set)
        var.add_set(medium_set)
        var.add_set(high_set)
        
        return var
    
    @staticmethod
    def create_five_level_sets(variable_name: str,
                             universe: Tuple[float, float] = (1.0, 10.0)) -> FuzzyVariable:
        """5段階ファジィ集合を作成"""
        
        var = FuzzyVariable(variable_name, universe)
        min_val, max_val = universe
        range_val = max_val - min_val
        
        # Very Low, Low, Medium, High, Very High の5つの集合
        sets_config = [
            ("very_low", 0.0, 0.0, 0.25),
            ("low", 0.0, 0.25, 0.5), 
            ("medium", 0.25, 0.5, 0.75),
            ("high", 0.5, 0.75, 1.0),
            ("very_high", 0.75, 1.0, 1.0)
        ]
        
        for name, start, peak, end in sets_config:
            fuzzy_set = FuzzySet(
                name,
                TriangularMF(
                    name,
                    min_val + range_val * start,
                    min_val + range_val * peak, 
                    min_val + range_val * end
                ),
                universe
            )
            var.add_set(fuzzy_set)
        
        return var
    
    @staticmethod
    def create_compatibility_variable() -> FuzzyVariable:
        """適合性用ファジィ変数を作成"""
        
        var = FuzzyVariable("compatibility", (0.0, 1.0))
        
        # 不適合, 低適合, 中適合, 高適合, 完全適合
        sets_config = [
            ("incompatible", 0.0, 0.0, 0.2),
            ("low_match", 0.0, 0.2, 0.4),
            ("medium_match", 0.2, 0.5, 0.8),
            ("high_match", 0.6, 0.8, 1.0),
            ("perfect_match", 0.8, 1.0, 1.0)
        ]
        
        for name, start, peak, end in sets_config:
            fuzzy_set = FuzzySet(
                name,
                TriangularMF(name, start, peak, end),
                (0.0, 1.0)
            )
            var.add_set(fuzzy_set)
        
        return var

# 使用例とテスト関数
def test_membership_functions():
    """メンバーシップ関数のテスト"""
    
    print("🧪 メンバーシップ関数テスト開始")
    
    # 三角形関数テスト
    tri_mf = TriangularMF("test_tri", 2, 5, 8)
    test_values = [1, 2, 3.5, 5, 6.5, 8, 9]
    
    print("\n📐 三角形メンバーシップ関数 (2, 5, 8):")
    for val in test_values:
        membership = tri_mf.membership(val)
        print(f"  x={val}: μ={membership:.3f}")
    
    # ガウシアン関数テスト
    gauss_mf = GaussianMF("test_gauss", 5, 1.5)
    
    print("\n🔔 ガウシアンメンバーシップ関数 (center=5, σ=1.5):")
    for val in test_values:
        membership = gauss_mf.membership(val)
        print(f"  x={val}: μ={membership:.3f}")
    
    # ファジィ変数テスト
    print("\n🔢 標準ファジィ変数テスト:")
    research_intensity = MembershipFunctionFactory.create_standard_sets(
        "research_intensity", (1.0, 10.0)
    )
    
    test_value = 7.5
    memberships = research_intensity.fuzzify(test_value)
    print(f"  値 {test_value} のファジィ化結果:")
    for set_name, membership in memberships.items():
        print(f"    {set_name}: {membership:.3f}")
    
    print("✅ メンバーシップ関数テスト完了")

if __name__ == "__main__":
    test_membership_functions()