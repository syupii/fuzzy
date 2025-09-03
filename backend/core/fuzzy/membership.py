"""
メンバーシップ関数 - core/fuzzy/membership.py
ファジィ論理のメンバーシップ関数実装
"""

from abc import ABC, abstractmethod
from typing import Union, List, Dict, Any
import numpy as np
from enum import Enum


class MembershipType(Enum):
    """メンバーシップ関数の種類"""
    TRIANGULAR = "triangular"
    GAUSSIAN = "gaussian"  
    TRAPEZOIDAL = "trapezoidal"


class MembershipFunction(ABC):
    """メンバーシップ関数の抽象基底クラス"""
    
    def __init__(self, name: str):
        self.name = name
        self.activation_count = 0
        self.total_membership = 0.0
    
    @abstractmethod
    def membership(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """メンバーシップ度を計算"""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, float]:
        """パラメータを取得"""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        return {
            'name': self.name,
            'activation_count': self.activation_count,
            'average_membership': self.total_membership / max(1, self.activation_count),
            'utilization_rate': self.activation_count / max(1, self.activation_count + 100)
        }


class TriangularMF(MembershipFunction):
    """三角形メンバーシップ関数"""
    
    def __init__(self, name: str, a: float, b: float, c: float):
        super().__init__(name)
        self.a = a  # 左端
        self.b = b  # 頂点
        self.c = c  # 右端
        
        # パラメータ検証
        if not (a <= b <= c):
            raise ValueError(f"Invalid triangular parameters: a={a}, b={b}, c={c}")
    
    def membership(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """三角形メンバーシップ度計算"""
        if isinstance(x, np.ndarray):
            result = np.zeros_like(x)
            
            # 左側の傾斜
            mask1 = (x >= self.a) & (x <= self.b)
            if self.b != self.a:
                result[mask1] = (x[mask1] - self.a) / (self.b - self.a)
            
            # 右側の傾斜  
            mask2 = (x > self.b) & (x <= self.c)
            if self.c != self.b:
                result[mask2] = (self.c - x[mask2]) / (self.c - self.b)
            
            # 頂点
            result[x == self.b] = 1.0
            
        else:
            if x <= self.a or x >= self.c:
                result = 0.0
            elif x == self.b:
                result = 1.0
            elif x < self.b:
                result = (x - self.a) / (self.b - self.a) if self.b != self.a else 0.0
            else:
                result = (self.c - x) / (self.c - self.b) if self.c != self.b else 0.0
        
        # 統計更新
        if isinstance(result, np.ndarray):
            active_count = np.sum(result > 0.1)
            self.activation_count += active_count
            self.total_membership += np.sum(result)
        else:
            if result > 0.1:
                self.activation_count += 1
                self.total_membership += result
        
        return np.clip(result, 0, 1)
    
    def get_params(self) -> Dict[str, float]:
        return {'a': self.a, 'b': self.b, 'c': self.c}


class GaussianMF(MembershipFunction):
    """ガウシアンメンバーシップ関数"""
    
    def __init__(self, name: str, center: float, sigma: float):
        super().__init__(name)
        self.center = center
        self.sigma = max(sigma, 0.01)  # シグマは正の値
    
    def membership(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """ガウシアンメンバーシップ度計算"""
        result = np.exp(-0.5 * ((x - self.center) / self.sigma) ** 2)
        
        # 統計更新
        if isinstance(result, np.ndarray):
            active_count = np.sum(result > 0.1)
            self.activation_count += active_count
            self.total_membership += np.sum(result)
        else:
            if result > 0.1:
                self.activation_count += 1
                self.total_membership += result
        
        return result
    
    def get_params(self) -> Dict[str, float]:
        return {'center': self.center, 'sigma': self.sigma}


class TrapezoidalMF(MembershipFunction):
    """台形メンバーシップ関数"""
    
    def __init__(self, name: str, a: float, b: float, c: float, d: float):
        super().__init__(name)
        self.a = a  # 左下端
        self.b = b  # 左上端
        self.c = c  # 右上端
        self.d = d  # 右下端
        
        # パラメータ検証
        if not (a <= b <= c <= d):
            raise ValueError(f"Invalid trapezoidal parameters: a={a}, b={b}, c={c}, d={d}")
    
    def membership(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """台形メンバーシップ度計算"""
        if isinstance(x, np.ndarray):
            result = np.zeros_like(x)
            
            # 左側の傾斜
            mask1 = (x >= self.a) & (x <= self.b)
            if self.b != self.a:
                result[mask1] = (x[mask1] - self.a) / (self.b - self.a)
            
            # 平坦部分
            mask2 = (x >= self.b) & (x <= self.c)
            result[mask2] = 1.0
            
            # 右側の傾斜
            mask3 = (x >= self.c) & (x <= self.d)
            if self.d != self.c:
                result[mask3] = (self.d - x[mask3]) / (self.d - self.c)
                
        else:
            if x <= self.a or x >= self.d:
                result = 0.0
            elif self.b <= x <= self.c:
                result = 1.0
            elif self.a < x < self.b:
                result = (x - self.a) / (self.b - self.a) if self.b != self.a else 0.0
            else:  # self.c < x < self.d
                result = (self.d - x) / (self.d - self.c) if self.d != self.c else 0.0
        
        # 統計更新
        if isinstance(result, np.ndarray):
            active_count = np.sum(result > 0.1)
            self.activation_count += active_count
            self.total_membership += np.sum(result)
        else:
            if result > 0.1:
                self.activation_count += 1
                self.total_membership += result
        
        return np.clip(result, 0, 1)
    
    def get_params(self) -> Dict[str, float]:
        return {'a': self.a, 'b': self.b, 'c': self.c, 'd': self.d}


class MembershipFunctionFactory:
    """メンバーシップ関数ファクトリー"""
    
    @staticmethod
    def create_membership_function(mf_type: MembershipType, name: str, 
                                 **params) -> MembershipFunction:
        """メンバーシップ関数を作成"""
        
        if mf_type == MembershipType.TRIANGULAR:
            return TriangularMF(name, params['a'], params['b'], params['c'])
        elif mf_type == MembershipType.GAUSSIAN:
            return GaussianMF(name, params['center'], params['sigma'])
        elif mf_type == MembershipType.TRAPEZOIDAL:
            return TrapezoidalMF(name, params['a'], params['b'], params['c'], params['d'])
        else:
            raise ValueError(f"Unsupported membership function type: {mf_type}")
    
    @staticmethod
    def create_fuzzy_sets(variable_name: str, domain_range: tuple, 
                         num_sets: int = 3, mf_type: MembershipType = MembershipType.TRIANGULAR,
                         labels: List[str] = None) -> Dict[str, MembershipFunction]:
        """変数に対するファジィ集合を自動生成"""
        
        min_val, max_val = domain_range
        if labels is None:
            if num_sets == 3:
                labels = ['Low', 'Medium', 'High']
            elif num_sets == 5:
                labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
            else:
                labels = [f'Set_{i+1}' for i in range(num_sets)]
        
        fuzzy_sets = {}
        
        if mf_type == MembershipType.TRIANGULAR:
            # 三角形メンバーシップ関数の均等分割
            step = (max_val - min_val) / (num_sets - 1)
            overlap = step * 0.5  # 重複度
            
            for i, label in enumerate(labels[:num_sets]):
                center = min_val + i * step
                left = center - overlap
                right = center + overlap
                
                # 境界調整
                if i == 0:
                    left = min_val
                if i == num_sets - 1:
                    right = max_val
                
                fuzzy_sets[label] = TriangularMF(
                    f"{variable_name}_{label}",
                    left, center, right
                )
        
        elif mf_type == MembershipType.GAUSSIAN:
            # ガウシアンメンバーシップ関数
            step = (max_val - min_val) / (num_sets - 1)
            sigma = step * 0.3  # 標準偏差
            
            for i, label in enumerate(labels[:num_sets]):
                center = min_val + i * step
                fuzzy_sets[label] = GaussianMF(
                    f"{variable_name}_{label}",
                    center, sigma
                )
        
        return fuzzy_sets
    
    @staticmethod
    def create_adaptive_fuzzy_sets(data: np.ndarray, variable_name: str,
                                 num_sets: int = 3, labels: List[str] = None) -> Dict[str, MembershipFunction]:
        """データに基づく適応的ファジィ集合生成"""
        
        if labels is None:
            labels = ['Low', 'Medium', 'High'] if num_sets == 3 else [f'Set_{i+1}' for i in range(num_sets)]
        
        # データの分位点に基づいてメンバーシップ関数を配置
        percentiles = np.linspace(0, 100, num_sets + 2)[1:-1]  # 境界を除く
        split_points = np.percentile(data, percentiles)
        
        min_val, max_val = data.min(), data.max()
        fuzzy_sets = {}
        
        # 三角形メンバーシップ関数で構築
        for i, label in enumerate(labels[:num_sets]):
            if i == 0:
                # 最初の集合
                left = min_val
                center = split_points[0] if len(split_points) > 0 else min_val
                right = split_points[1] if len(split_points) > 1 else max_val
            elif i == num_sets - 1:
                # 最後の集合
                left = split_points[i-1] if i-1 < len(split_points) else min_val
                center = split_points[i] if i < len(split_points) else max_val
                right = max_val
            else:
                # 中間の集合
                left = split_points[i-1]
                center = split_points[i]
                right = split_points[i+1] if i+1 < len(split_points) else max_val
            
            fuzzy_sets[label] = TriangularMF(
                f"{variable_name}_{label}",
                left, center, right
            )
        
        return fuzzy_sets