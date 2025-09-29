"""
ファジィメンバーシップ関数モジュール
各種メンバーシップ関数の実装
"""

import numpy as np
from typing import List, Tuple, Dict, Callable
import math


class MembershipFunction:
    """メンバーシップ関数の基底クラス"""
    
    def __call__(self, x: float) -> float:
        """メンバーシップ度を計算"""
        raise NotImplementedError
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class TriangularMF(MembershipFunction):
    """三角型メンバーシップ関数
    
    最も基本的なファジィメンバーシップ関数
    3つのパラメータ(a, b, c)で定義される
    """
    
    def __init__(self, a: float, b: float, c: float):
        """
        Args:
            a: 左端（メンバーシップ度=0）
            b: 中心（メンバーシップ度=1のピーク）
            c: 右端（メンバーシップ度=0）
        """
        if not (a <= b <= c):
            raise ValueError(f"Parameters must satisfy a <= b <= c, got a={a}, b={b}, c={c}")
        
        self.a = a
        self.b = b
        self.c = c
    
    def __call__(self, x: float) -> float:
        """メンバーシップ度を計算
        
        Args:
            x: 入力値
            
        Returns:
            メンバーシップ度 [0, 1]
        """
        if x <= self.a or x >= self.c:
            return 0.0
        elif x == self.b:
            return 1.0
        elif x < self.b:
            return (x - self.a) / (self.b - self.a)
        else:  # x > self.b
            return (self.c - x) / (self.c - self.b)
    
    def __repr__(self) -> str:
        return f"TriangularMF(a={self.a}, b={self.b}, c={self.c})"


class TrapezoidalMF(MembershipFunction):
    """台形型メンバーシップ関数
    
    4つのパラメータで定義される
    平坦な頂上を持つため、より柔軟な表現が可能
    """
    
    def __init__(self, a: float, b: float, c: float, d: float):
        """
        Args:
            a: 左端（メンバーシップ度=0）
            b: 左上端（メンバーシップ度=1の開始）
            c: 右上端（メンバーシップ度=1の終了）
            d: 右端（メンバーシップ度=0）
        """
        if not (a <= b <= c <= d):
            raise ValueError(f"Parameters must satisfy a <= b <= c <= d")
        
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    def __call__(self, x: float) -> float:
        """メンバーシップ度を計算"""
        if x <= self.a or x >= self.d:
            return 0.0
        elif self.b <= x <= self.c:
            return 1.0
        elif self.a < x < self.b:
            return (x - self.a) / (self.b - self.a)
        else:  # self.c < x < self.d
            return (self.d - x) / (self.d - self.c)
    
    def __repr__(self) -> str:
        return f"TrapezoidalMF(a={self.a}, b={self.b}, c={self.c}, d={self.d})"


class GaussianMF(MembershipFunction):
    """ガウス型（正規分布型）メンバーシップ関数
    
    滑らかな曲線を持つメンバーシップ関数
    統計的な解釈が可能
    """
    
    def __init__(self, mean: float, sigma: float):
        """
        Args:
            mean: 平均（中心、メンバーシップ度=1のピーク）
            sigma: 標準偏差（広がりを制御）
        """
        if sigma <= 0:
            raise ValueError(f"Sigma must be positive, got {sigma}")
        
        self.mean = mean
        self.sigma = sigma
    
    def __call__(self, x: float) -> float:
        """メンバーシップ度を計算"""
        return math.exp(-0.5 * ((x - self.mean) / self.sigma) ** 2)
    
    def __repr__(self) -> str:
        return f"GaussianMF(mean={self.mean}, sigma={self.sigma})"


class GeneralizedBellMF(MembershipFunction):
    """一般化ベル型メンバーシップ関数
    
    3つのパラメータで形状を柔軟に制御できる
    ガウス関数よりも形状の制御が容易
    """
    
    def __init__(self, a: float, b: float, c: float):
        """
        Args:
            a: 広がり（大きいほど広い）
            b: 傾き（大きいほど急峻）
            c: 中心
        """
        if a <= 0:
            raise ValueError(f"Parameter 'a' must be positive, got {a}")
        if b <= 0:
            raise ValueError(f"Parameter 'b' must be positive, got {b}")
        
        self.a = a
        self.b = b
        self.c = c
    
    def __call__(self, x: float) -> float:
        """メンバーシップ度を計算"""
        return 1.0 / (1.0 + abs((x - self.c) / self.a) ** (2 * self.b))
    
    def __repr__(self) -> str:
        return f"GeneralizedBellMF(a={self.a}, b={self.b}, c={self.c})"


class SigmoidMF(MembershipFunction):
    """シグモイド型メンバーシップ関数
    
    S字カーブを描くメンバーシップ関数
    単調増加または単調減少の表現に適している
    """
    
    def __init__(self, a: float, c: float):
        """
        Args:
            a: 傾き（正で増加、負で減少）
            c: 変曲点（S字の中心）
        """
        self.a = a
        self.c = c
    
    def __call__(self, x: float) -> float:
        """メンバーシップ度を計算"""
        return 1.0 / (1.0 + math.exp(-self.a * (x - self.c)))
    
    def __repr__(self) -> str:
        return f"SigmoidMF(a={self.a}, c={self.c})"


class FuzzySet:
    """ファジィ集合
    
    メンバーシップ関数と言語ラベルを持つ
    """
    
    def __init__(self, label: str, mf: MembershipFunction):
        """
        Args:
            label: 言語ラベル（例: "低い", "中程度", "高い"）
            mf: メンバーシップ関数
        """
        self.label = label
        self.mf = mf
    
    def membership(self, x: float) -> float:
        """メンバーシップ度を計算"""
        return self.mf(x)
    
    def __call__(self, x: float) -> float:
        """メンバーシップ度を計算（簡略記法）"""
        return self.membership(x)
    
    def __repr__(self) -> str:
        return f"FuzzySet(label='{self.label}', mf={self.mf})"


class FuzzyVariable:
    """ファジィ変数
    
    複数のファジィ集合を持つ言語変数
    例: 研究強度 = {低い, 中程度, 高い}
    """
    
    def __init__(self, name: str, universe: Tuple[float, float]):
        """
        Args:
            name: 変数名
            universe: 論理領域 (min, max)
        """
        self.name = name
        self.universe = universe
        self.fuzzy_sets: Dict[str, FuzzySet] = {}
    
    def add_fuzzy_set(self, fuzzy_set: FuzzySet):
        """ファジィ集合を追加"""
        self.fuzzy_sets[fuzzy_set.label] = fuzzy_set
    
    def fuzzify(self, x: float) -> Dict[str, float]:
        """ファジィ化: crisp値を各ファジィ集合のメンバーシップ度に変換
        
        Args:
            x: crisp値
            
        Returns:
            各言語ラベルに対するメンバーシップ度の辞書
        """
        return {
            label: fuzzy_set.membership(x)
            for label, fuzzy_set in self.fuzzy_sets.items()
        }
    
    def defuzzify(self, memberships: Dict[str, float], method: str = "centroid") -> float:
        """非ファジィ化: メンバーシップ度をcrisp値に変換
        
        Args:
            memberships: 各言語ラベルのメンバーシップ度
            method: 非ファジィ化手法 ("centroid", "max", "mean")
            
        Returns:
            crisp値
        """
        if method == "centroid":
            # 重心法
            numerator = 0.0
            denominator = 0.0
            
            # 論理領域をサンプリング
            samples = np.linspace(self.universe[0], self.universe[1], 100)
            
            for x in samples:
                # 各点での総合メンバーシップ度を計算
                max_membership = max(
                    min(memberships.get(label, 0), fuzzy_set.membership(x))
                    for label, fuzzy_set in self.fuzzy_sets.items()
                )
                numerator += x * max_membership
                denominator += max_membership
            
            return numerator / denominator if denominator > 0 else self.universe[0]
        
        elif method == "max":
            # 最大値法（最大メンバーシップ度を持つラベルの中心値）
            max_label = max(memberships.items(), key=lambda x: x[1])[0]
            fuzzy_set = self.fuzzy_sets[max_label]
            
            # メンバーシップ関数のピーク位置を返す
            if isinstance(fuzzy_set.mf, TriangularMF):
                return fuzzy_set.mf.b
            elif isinstance(fuzzy_set.mf, GaussianMF):
                return fuzzy_set.mf.mean
            else:
                # その他の場合は重心法にフォールバック
                return self.defuzzify(memberships, method="centroid")
        
        elif method == "mean":
            # 平均値法
            total = sum(memberships.values())
            if total == 0:
                return (self.universe[0] + self.universe[1]) / 2
            
            weighted_sum = 0.0
            for label, membership in memberships.items():
                fuzzy_set = self.fuzzy_sets[label]
                
                # 各ファジィ集合の代表値を取得
                if isinstance(fuzzy_set.mf, TriangularMF):
                    rep_value = fuzzy_set.mf.b
                elif isinstance(fuzzy_set.mf, GaussianMF):
                    rep_value = fuzzy_set.mf.mean
                else:
                    rep_value = (self.universe[0] + self.universe[1]) / 2
                
                weighted_sum += rep_value * membership
            
            return weighted_sum / total
        
        else:
            raise ValueError(f"Unknown defuzzification method: {method}")
    
    def __repr__(self) -> str:
        return f"FuzzyVariable(name='{self.name}', sets={list(self.fuzzy_sets.keys())})"


def create_standard_fuzzy_variable(name: str, 
                                   universe: Tuple[float, float] = (0.0, 1.0),
                                   n_sets: int = 3) -> FuzzyVariable:
    """標準的なファジィ変数を作成
    
    Args:
        name: 変数名
        universe: 論理領域
        n_sets: ファジィ集合の数（3, 5, 7のいずれか）
        
    Returns:
        設定済みのファジィ変数
    """
    var = FuzzyVariable(name, universe)
    min_val, max_val = universe
    
    if n_sets == 3:
        # 3分割: 低い、中程度、高い
        labels = ["low", "medium", "high"]
        
        var.add_fuzzy_set(FuzzySet(
            "low",
            TriangularMF(min_val, min_val, (min_val + max_val) / 2)
        ))
        var.add_fuzzy_set(FuzzySet(
            "medium",
            TriangularMF(min_val, (min_val + max_val) / 2, max_val)
        ))
        var.add_fuzzy_set(FuzzySet(
            "high",
            TriangularMF((min_val + max_val) / 2, max_val, max_val)
        ))
    
    elif n_sets == 5:
        # 5分割: 非常に低い、低い、中程度、高い、非常に高い
        step = (max_val - min_val) / 4
        
        var.add_fuzzy_set(FuzzySet(
            "very_low",
            TriangularMF(min_val, min_val, min_val + step)
        ))
        var.add_fuzzy_set(FuzzySet(
            "low",
            TriangularMF(min_val, min_val + step, min_val + 2*step)
        ))
        var.add_fuzzy_set(FuzzySet(
            "medium",
            TriangularMF(min_val + step, min_val + 2*step, min_val + 3*step)
        ))
        var.add_fuzzy_set(FuzzySet(
            "high",
            TriangularMF(min_val + 2*step, min_val + 3*step, max_val)
        ))
        var.add_fuzzy_set(FuzzySet(
            "very_high",
            TriangularMF(min_val + 3*step, max_val, max_val)
        ))
    
    elif n_sets == 7:
        # 7分割
        step = (max_val - min_val) / 6
        labels = ["very_low", "low", "rather_low", "medium", 
                 "rather_high", "high", "very_high"]
        
        for i, label in enumerate(labels):
            if i == 0:
                mf = TriangularMF(min_val, min_val, min_val + step)
            elif i == len(labels) - 1:
                mf = TriangularMF(min_val + (i-1)*step, max_val, max_val)
            else:
                mf = TriangularMF(min_val + (i-1)*step, 
                                 min_val + i*step, 
                                 min_val + (i+1)*step)
            
            var.add_fuzzy_set(FuzzySet(label, mf))
    
    else:
        raise ValueError(f"n_sets must be 3, 5, or 7, got {n_sets}")
    
    return var


# 使用例とテスト
if __name__ == "__main__":
    print("=" * 60)
    print("ファジィメンバーシップ関数テスト")
    print("=" * 60)
    
    # 三角型メンバーシップ関数
    print("\n1. 三角型メンバーシップ関数")
    tri_mf = TriangularMF(0.0, 0.5, 1.0)
    test_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    for x in test_values:
        print(f"  μ({x:.2f}) = {tri_mf(x):.3f}")
    
    # ガウス型メンバーシップ関数
    print("\n2. ガウス型メンバーシップ関数")
    gauss_mf = GaussianMF(0.5, 0.2)
    for x in test_values:
        print(f"  μ({x:.2f}) = {gauss_mf(x):.3f}")
    
    # ファジィ変数の作成
    print("\n3. ファジィ変数（研究強度）")
    research_intensity = create_standard_fuzzy_variable(
        "research_intensity", 
        universe=(0.0, 1.0),
        n_sets=3
    )
    print(f"  変数: {research_intensity}")
    
    # ファジィ化
    print("\n4. ファジィ化（x = 0.65）")
    memberships = research_intensity.fuzzify(0.65)
    for label, degree in memberships.items():
        print(f"  {label}: {degree:.3f}")
    
    # 非ファジィ化
    print("\n5. 非ファジィ化")
    test_memberships = {"low": 0.2, "medium": 0.7, "high": 0.5}
    result = research_intensity.defuzzify(test_memberships, method="centroid")
    print(f"  重心法: {result:.3f}")
    
    print("\n" + "=" * 60)