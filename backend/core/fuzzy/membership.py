from abc import ABC, abstractmethod
from typing import Union, List
import numpy as np

class MembershipFunction(ABC):
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def membership(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, float]:
        pass

class TriangularMF(MembershipFunction):
    def __init__(self, name: str, a: float, b: float, c: float):
        super().__init__(name)
        self.a = a  # 左端
        self.b = b  # 頂点
        self.c = c  # 右端
    
    def membership(self, x):
        return np.maximum(0, np.minimum((x - self.a) / (self.b - self.a), 
                                       (self.c - x) / (self.c - self.b)))