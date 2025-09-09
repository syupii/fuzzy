# core/fuzzy/inference.py - ファジィ推論エンジン（シンプル動作版）

import numpy as np
import random
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class DefuzzificationMethod(str, Enum):
    """非ファジィ化手法"""
    CENTROID = "centroid"
    WEIGHTED_AVERAGE = "weighted_average"
    MAX_MEMBERSHIP = "max_membership"
    MEAN_OF_MAXIMA = "mean_of_maxima"

@dataclass
class InferenceResult:
    """推論結果"""
    output_values: Dict[str, float]
    rule_activations: Dict[str, float]
    confidence: float
    inference_time: float = 0.0
    details: Dict[str, Any] = None

class FuzzyInferenceEngine(ABC):
    """ファジィ推論エンジンの抽象基底クラス"""
    
    @abstractmethod
    def infer(self, inputs: Dict[str, float]) -> InferenceResult:
        """推論実行"""
        pass
    
    def evaluate_compatibility(self, student_profile: Dict, lab_data: Dict) -> float:
        """適合性評価（後方互換性）"""
        result = self.infer(student_profile)
        return result.output_values.get("compatibility", 0.5)

class SimpleFuzzyInferenceEngine(FuzzyInferenceEngine):
    """シンプルなファジィ推論エンジン"""
    
    def __init__(self, defuzzification_method: DefuzzificationMethod = DefuzzificationMethod.WEIGHTED_AVERAGE):
        self.defuzzification_method = defuzzification_method
        self.initialized = True
        
        # 統計情報
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        logger.info("SimpleFuzzyInferenceEngine初期化完了")
    
    def infer(self, inputs: Dict[str, float]) -> InferenceResult:
        """推論を実行"""
        import time
        start_time = time.time()
        
        try:
            # 基本的な適合性計算
            output_values = self._compute_compatibility(inputs)
            
            # ルール活性化（簡易版）
            rule_activations = self._evaluate_rules(inputs)
            
            # 信頼度計算
            confidence = self._calculate_confidence(output_values, rule_activations)
            
            inference_time = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += inference_time
            
            return InferenceResult(
                output_values=output_values,
                rule_activations=rule_activations,
                confidence=confidence,
                inference_time=inference_time,
                details={
                    "method": self.defuzzification_method.value,
                    "input_features": list(inputs.keys())
                }
            )
            
        except Exception as e:
            logger.error(f"推論エラー: {e}")
            return InferenceResult(
                output_values={"compatibility": 0.5},
                rule_activations={},
                confidence=0.0,
                inference_time=time.time() - start_time
            )
    
    def _compute_compatibility(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """適合性を計算"""
        
        # 基本的な5項目での評価
        core_features = [
            "research_intensity", "advisor_style", "team_work", 
            "workload", "theory_practice"
        ]
        
        # 理想的な研究室プロファイル（例）
        ideal_lab_profile = {
            "research_intensity": 7.5,
            "advisor_style": 6.0,
            "team_work": 7.0,
            "workload": 6.5,
            "theory_practice": 7.0
        }
        
        total_score = 0.0
        weight_sum = 0.0
        
        # 重み設定（重要度に基づく）
        feature_weights = {
            "research_intensity": 0.25,
            "advisor_style": 0.20,
            "team_work": 0.20,
            "workload": 0.15,
            "theory_practice": 0.20
        }
        
        for feature in core_features:
            if feature in inputs:
                student_val = inputs[feature]
                lab_val = ideal_lab_profile.get(feature, 5.0)
                
                # ガウシアン類似度関数
                similarity = np.exp(-0.5 * ((student_val - lab_val) / 2.0) ** 2)
                
                weight = feature_weights.get(feature, 0.1)
                total_score += similarity * weight
                weight_sum += weight
        
        compatibility = total_score / weight_sum if weight_sum > 0 else 0.5
        
        return {
            "compatibility": compatibility,
            "raw_score": total_score,
            "normalized_score": min(1.0, max(0.0, compatibility))
        }
    
    def _evaluate_rules(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """ルールを評価"""
        
        rule_activations = {}
        
        # 簡易ルール評価
        research_intensity = inputs.get("research_intensity", 5.0)
        
        # デフォルトルール
        rule_activations["high_research"] = 0.8 if research_intensity > 7 else 0.3
        rule_activations["medium_research"] = 0.6 if 4 <= research_intensity <= 7 else 0.2
        rule_activations["low_research"] = 0.4 if research_intensity < 4 else 0.1
        
        return rule_activations
    
    def _calculate_confidence(self, outputs: Dict[str, float], 
                            rule_activations: Dict[str, float]) -> float:
        """信頼度を計算"""
        
        # 出力値の分散から信頼度を計算
        output_values = list(outputs.values())
        if len(output_values) == 0:
            return 0.0
        
        mean_output = np.mean(output_values)
        
        # ルール活性化の強さ
        if rule_activations:
            max_activation = max(rule_activations.values())
            mean_activation = np.mean(list(rule_activations.values()))
            
            # 活性化度が高く、一貫性があるほど信頼度が高い
            consistency = 1.0 - np.std(list(rule_activations.values()))
            confidence = (max_activation + mean_activation + consistency) / 3.0
        else:
            confidence = mean_output
        
        return min(1.0, max(0.0, confidence))
    
    def evaluate_compatibility(self, student_profile: Dict, lab_data: Dict) -> float:
        """後方互換性のための適合性評価メソッド"""
        
        # 学生プロファイルを入力として使用
        result = self.infer(student_profile)
        return result.output_values.get("compatibility", 0.5)
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        
        avg_inference_time = (self.total_inference_time / self.inference_count 
                             if self.inference_count > 0 else 0.0)
        
        return {
            "engine_type": "SimpleFuzzyInferenceEngine",
            "defuzzification_method": self.defuzzification_method.value,
            "total_inferences": self.inference_count,
            "total_inference_time": self.total_inference_time,
            "average_inference_time": avg_inference_time
        }

# 使用例とテスト
def test_inference_engine():
    """推論エンジンのテスト"""
    
    print("🔧 ファジィ推論エンジンテスト開始")
    
    # エンジン初期化
    engine = SimpleFuzzyInferenceEngine(DefuzzificationMethod.WEIGHTED_AVERAGE)
    
    # テスト入力
    test_inputs = {
        "research_intensity": 8.0,
        "advisor_style": 6.5,
        "team_work": 7.0,
        "workload": 5.5,
        "theory_practice": 8.5
    }
    
    # 推論実行
    result = engine.infer(test_inputs)
    
    print(f"📊 推論結果:")
    print(f"  適合性: {result.output_values.get('compatibility', 0):.3f}")
    print(f"  信頼度: {result.confidence:.3f}")
    print(f"  推論時間: {result.inference_time*1000:.2f}ms")
    
    # 後方互換性テスト
    compatibility = engine.evaluate_compatibility(test_inputs, {})
    print(f"  後方互換性: {compatibility:.3f}")
    
    print("✅ ファジィ推論エンジンテスト完了")

if __name__ == "__main__":
    test_inference_engine()