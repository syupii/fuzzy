# core/fuzzy/membership.py - メンバーシップ関数（完全版）

import numpy as np
import math
from typing import Dict, List, Tuple, Callable, Any, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

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
    
    def __init__(self, name: str, params: MembershipParams = None):
        self.name = name
        self.params = params or MembershipParams(0.0)
    
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
        if self.sigma == 0:
            return 1.0 if x == self.center else 0.0
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
        if self.a == 0:
            return 1.0 if x == self.c else 0.0
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
            numerator = 0.0
            denominator = 0.0
            
            for set_name, membership in membership_values.items():
                if set_name in self.sets and membership > 0:
                    centroid = self.sets[set_name].centroid()
                    numerator += membership * centroid
                    denominator += membership
            
            return numerator / denominator if denominator > 0 else 0.0
        
        elif method == "weighted_average":
            total_weight = sum(membership_values.values())
            if total_weight == 0:
                return 0.0
            
            weighted_sum = 0.0
            for set_name, membership in membership_values.items():
                if set_name in self.sets:
                    # 集合の代表値（重心）を使用
                    centroid = self.sets[set_name].centroid()
                    weighted_sum += membership * centroid
            
            return weighted_sum / total_weight
        
        elif method == "max_membership":
            # 最大メンバーシップ度を持つ集合の重心
            max_membership = max(membership_values.values())
            if max_membership == 0:
                return 0.0
            
            for set_name, membership in membership_values.items():
                if membership == max_membership and set_name in self.sets:
                    return self.sets[set_name].centroid()
            
            return 0.0
        
        else:
            raise ValueError(f"未知の非ファジィ化手法: {method}")
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式で出力"""
        return {
            "name": self.name,
            "universe": self.universe,
            "sets": {name: {
                "name": fs.name,
                "membership_function": fs.membership_function.name,
                "universe": fs.universe
            } for name, fs in self.sets.items()}
        }

class MembershipFunctionFactory:
    """メンバーシップ関数ファクトリ"""
    
    @staticmethod
    def create_standard_sets(variable_name: str, 
                           universe: Tuple[float, float] = (1.0, 10.0)) -> FuzzyVariable:
        """標準的な3分割ファジィ集合を作成（低、中、高）"""
        
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
        """5段階ファジィ集合を作成（極低、低、中、高、極高）"""
        
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
    
    @staticmethod
    def create_research_intensity_variable() -> FuzzyVariable:
        """研究強度用ファジィ変数を作成"""
        
        var = FuzzyVariable("research_intensity", (1.0, 10.0))
        
        # 軽い研究, 普通の研究, 集中研究
        sets_config = [
            ("light", 1.0, 1.0, 4.0),
            ("moderate", 2.0, 5.5, 8.0),
            ("intensive", 6.0, 10.0, 10.0)
        ]
        
        for name, start, peak, end in sets_config:
            fuzzy_set = FuzzySet(
                name,
                TriangularMF(name, start, peak, end),
                (1.0, 10.0)
            )
            var.add_set(fuzzy_set)
        
        return var
    
    @staticmethod
    def create_gaussian_sets(variable_name: str, 
                           universe: Tuple[float, float] = (1.0, 10.0),
                           num_sets: int = 3) -> FuzzyVariable:
        """ガウシアンメンバーシップ関数を使用したファジィ集合を作成"""
        
        var = FuzzyVariable(variable_name, universe)
        min_val, max_val = universe
        range_val = max_val - min_val
        
        for i in range(num_sets):
            center = min_val + (i / (num_sets - 1)) * range_val
            sigma = range_val / (num_sets * 2)  # 適度な重複
            
            set_name = f"gauss_{i}"
            if num_sets == 3:
                set_names = ["low", "medium", "high"]
                set_name = set_names[i]
            elif num_sets == 5:
                set_names = ["very_low", "low", "medium", "high", "very_high"]
                set_name = set_names[i]
            
            fuzzy_set = FuzzySet(
                set_name,
                GaussianMF(set_name, center, sigma),
                universe
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
    
    # 非ファジィ化テスト
    defuzzified = research_intensity.defuzzify(memberships, "centroid")
    print(f"  非ファジィ化結果 (重心法): {defuzzified:.3f}")
    
    print("✅ メンバーシップ関数テスト完了")

def create_evaluation_criteria_variables() -> Dict[str, FuzzyVariable]:
    """評価基準用のファジィ変数群を作成"""
    
    variables = {}
    
    # 基本項目（5項目）
    basic_criteria = [
        "research_intensity", "advisor_style", "team_work", 
        "workload", "theory_practice"
    ]
    
    for criterion in basic_criteria:
        variables[criterion] = MembershipFunctionFactory.create_standard_sets(
            criterion, (1.0, 10.0)
        )
    
    # 拡張項目（5項目）
    extended_criteria = [
        "research_field_match", "skill_development", "lab_atmosphere",
        "flexibility", "publication_opportunity"
    ]
    
    for criterion in extended_criteria:
        variables[criterion] = MembershipFunctionFactory.create_standard_sets(
            criterion, (1.0, 10.0)
        )
    
    # 特殊項目（3項目）
    special_criteria = [
        "interdisciplinary", "communication_style", "innovation_risk"
    ]
    
    for criterion in special_criteria:
        variables[criterion] = MembershipFunctionFactory.create_standard_sets(
            criterion, (1.0, 10.0)
        )
    
    # 適合性変数
    variables["compatibility"] = MembershipFunctionFactory.create_compatibility_variable()
    
    return variables

if __name__ == "__main__":
    test_membership_functions()