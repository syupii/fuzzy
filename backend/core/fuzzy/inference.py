# core/fuzzy/inference.py - ファジィ推論エンジン

import numpy as np
import math
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MembershipFunction:
    """メンバーシップ関数"""
    name: str
    function_type: str  # triangular, trapezoidal, gaussian
    parameters: List[float]
    
    def evaluate(self, x: float) -> float:
        """メンバーシップ値を計算"""
        if self.function_type == "triangular":
            a, b, c = self.parameters
            if x <= a or x >= c:
                return 0.0
            elif x == b:
                return 1.0
            elif x < b:
                return (x - a) / (b - a)
            else:
                return (c - x) / (c - b)
                
        elif self.function_type == "trapezoidal":
            a, b, c, d = self.parameters
            if x <= a or x >= d:
                return 0.0
            elif b <= x <= c:
                return 1.0
            elif a < x < b:
                return (x - a) / (b - a)
            elif c < x < d:
                return (d - x) / (d - a)
                
        elif self.function_type == "gaussian":
            mean, sigma = self.parameters
            return math.exp(-0.5 * ((x - mean) / sigma) ** 2)
            
        return 0.0

@dataclass
class FuzzyVariable:
    """ファジィ変数"""
    name: str
    domain: Tuple[float, float]  # (min, max)
    membership_functions: Dict[str, MembershipFunction]
    
    def fuzzify(self, value: float) -> Dict[str, float]:
        """ファジィ化 - 入力値に対する各言語値のメンバーシップ度を計算"""
        result = {}
        for term, mf in self.membership_functions.items():
            result[term] = mf.evaluate(value)
        return result
    
    def defuzzify(self, fuzzy_output: Dict[str, float], method: str = "centroid") -> float:
        """非ファジィ化"""
        if method == "centroid":
            numerator = 0.0
            denominator = 0.0
            
            # 重心法による非ファジィ化
            for i in range(int(self.domain[0] * 10), int(self.domain[1] * 10) + 1):
                x = i / 10.0
                membership_sum = 0.0
                
                for term, weight in fuzzy_output.items():
                    if term in self.membership_functions:
                        membership_sum += weight * self.membership_functions[term].evaluate(x)
                
                numerator += x * membership_sum
                denominator += membership_sum
            
            return numerator / denominator if denominator > 0 else 0.0
        
        elif method == "max":
            # 最大値法
            max_term = max(fuzzy_output, key=fuzzy_output.get)
            mf = self.membership_functions.get(max_term)
            if mf and mf.function_type == "triangular":
                return mf.parameters[1]  # ピーク値
            
        return 0.0

@dataclass
class FuzzyRule:
    """ファジィルール"""
    antecedent: Dict[str, str]  # 変数名 -> 言語値
    consequent: Dict[str, str]  # 変数名 -> 言語値
    weight: float = 1.0
    
    def evaluate_antecedent(self, input_memberships: Dict[str, Dict[str, float]]) -> float:
        """前件部の評価（MIN演算）"""
        min_membership = 1.0
        
        for var_name, linguistic_value in self.antecedent.items():
            if var_name in input_memberships:
                membership = input_memberships[var_name].get(linguistic_value, 0.0)
                min_membership = min(min_membership, membership)
        
        return min_membership * self.weight

class SimpleFuzzyInferenceEngine:
    """ファジィ推論エンジン"""
    
    def __init__(self):
        self.variables: Dict[str, FuzzyVariable] = {}
        self.rules: List[FuzzyRule] = []
        self.setup_research_lab_fuzzy_system()
    
    def setup_research_lab_fuzzy_system(self):
        """研究室選択用のファジィシステムをセットアップ"""
        
        # 入力変数の定義
        input_vars = [
            "research_intensity", "advisor_style", "team_work", "workload", 
            "theory_practice", "research_field_match", "skill_development",
            "lab_atmosphere", "flexibility", "publication_opportunity",
            "interdisciplinary", "communication_style", "innovation_risk"
        ]
        
        # 各入力変数に対してファジィセットを定義
        for var_name in input_vars:
            self.variables[var_name] = FuzzyVariable(
                name=var_name,
                domain=(1.0, 10.0),
                membership_functions={
                    "low": MembershipFunction("low", "triangular", [1.0, 1.0, 5.0]),
                    "medium": MembershipFunction("medium", "triangular", [2.0, 5.5, 9.0]),
                    "high": MembershipFunction("high", "triangular", [5.0, 10.0, 10.0])
                }
            )
        
        # 出力変数（適合度）の定義
        self.variables["compatibility"] = FuzzyVariable(
            name="compatibility",
            domain=(0.0, 1.0),
            membership_functions={
                "very_low": MembershipFunction("very_low", "triangular", [0.0, 0.0, 0.2]),
                "low": MembershipFunction("low", "triangular", [0.0, 0.2, 0.4]),
                "medium": MembershipFunction("medium", "triangular", [0.2, 0.5, 0.8]),
                "high": MembershipFunction("high", "triangular", [0.6, 0.8, 1.0]),
                "very_high": MembershipFunction("very_high", "triangular", [0.8, 1.0, 1.0])
            }
        )
        
        # ファジィルールの定義
        self.setup_fuzzy_rules()
    
    def setup_fuzzy_rules(self):
        """ファジィルールの設定"""
        
        # 研究室選択に関するファジィルール
        rules = [
            # 高研究強度 + 高分野適合性 = 高適合度
            FuzzyRule(
                {"research_intensity": "high", "research_field_match": "high"},
                {"compatibility": "very_high"},
                weight=0.9
            ),
            
            # 中研究強度 + 高論文機会 = 高適合度
            FuzzyRule(
                {"research_intensity": "medium", "publication_opportunity": "high"},
                {"compatibility": "high"},
                weight=0.8
            ),
            
            # 高チームワーク + 高コミュニケーション = 高適合度
            FuzzyRule(
                {"team_work": "high", "communication_style": "high"},
                {"compatibility": "high"},
                weight=0.7
            ),
            
            # 低ワークロード + 高柔軟性 = 中適合度
            FuzzyRule(
                {"workload": "low", "flexibility": "high"},
                {"compatibility": "medium"},
                weight=0.6
            ),
            
            # 高革新性 + 高学際性 = 高適合度
            FuzzyRule(
                {"innovation_risk": "high", "interdisciplinary": "high"},
                {"compatibility": "high"},
                weight=0.8
            ),
            
            # 低分野適合性 = 低適合度
            FuzzyRule(
                {"research_field_match": "low"},
                {"compatibility": "low"},
                weight=0.9
            ),
            
            # 理論重視 + 高スキル開発 = 高適合度
            FuzzyRule(
                {"theory_practice": "low", "skill_development": "high"},
                {"compatibility": "high"},
                weight=0.7
            ),
            
            # 実践重視 + 高革新性 = 高適合度
            FuzzyRule(
                {"theory_practice": "high", "innovation_risk": "high"},
                {"compatibility": "high"},
                weight=0.7
            ),
            
            # バランス型評価
            FuzzyRule(
                {"research_intensity": "medium", "advisor_style": "medium", "team_work": "medium"},
                {"compatibility": "medium"},
                weight=0.5
            ),
            
            # ミスマッチパターン
            FuzzyRule(
                {"workload": "high", "flexibility": "low"},
                {"compatibility": "low"},
                weight=0.6
            )
        ]
        
        self.rules.extend(rules)
        logger.info(f"ファジィルール {len(self.rules)} 個を設定しました")
    
    def infer(self, inputs: Dict[str, float]) -> float:
        """ファジィ推論実行"""
        
        # Step 1: ファジィ化
        input_memberships = {}
        for var_name, value in inputs.items():
            if var_name in self.variables:
                input_memberships[var_name] = self.variables[var_name].fuzzify(value)
        
        # Step 2: ルール評価
        output_memberships = {"very_low": 0.0, "low": 0.0, "medium": 0.0, "high": 0.0, "very_high": 0.0}
        
        for rule in self.rules:
            # 前件部の評価
            antecedent_strength = rule.evaluate_antecedent(input_memberships)
            
            if antecedent_strength > 0:
                # 後件部への伝播
                for var_name, linguistic_value in rule.consequent.items():
                    if var_name == "compatibility":
                        current_strength = output_memberships.get(linguistic_value, 0.0)
                        output_memberships[linguistic_value] = max(current_strength, antecedent_strength)
        
        # Step 3: 非ファジィ化
        compatibility_var = self.variables["compatibility"]
        crisp_output = compatibility_var.defuzzify(output_memberships, method="centroid")
        
        return max(0.0, min(1.0, crisp_output))
    
    def infer_lab_compatibility(self, student_profile: Dict[str, float], 
                               lab_profile: Dict[str, float]) -> float:
        """研究室適合性推論"""
        
        # 学生プロファイルと研究室プロファイルの差分を基にした推論
        adjusted_inputs = {}
        
        for criterion in self.variables.keys():
            if criterion == "compatibility":
                continue
                
            student_val = student_profile.get(criterion, 5.0)
            lab_val = lab_profile.get(criterion, 5.0)
            
            # 類似度に基づく調整値を計算
            similarity = 1.0 - abs(student_val - lab_val) / 9.0
            adjusted_inputs[criterion] = student_val * similarity
        
        return self.infer(adjusted_inputs)
    
    def explain_inference(self, inputs: Dict[str, float]) -> Dict[str, Any]:
        """推論過程の説明"""
        
        # ファジィ化結果
        input_memberships = {}
        for var_name, value in inputs.items():
            if var_name in self.variables:
                input_memberships[var_name] = self.variables[var_name].fuzzify(value)
        
        # 活性化されたルールの特定
        activated_rules = []
        for i, rule in enumerate(self.rules):
            strength = rule.evaluate_antecedent(input_memberships)
            if strength > 0.1:  # 閾値以上のルールのみ
                activated_rules.append({
                    "rule_id": i,
                    "antecedent": rule.antecedent,
                    "consequent": rule.consequent,
                    "strength": strength,
                    "weight": rule.weight
                })
        
        # 推論結果
        compatibility = self.infer(inputs)
        
        return {
            "compatibility_score": compatibility,
            "input_memberships": input_memberships,
            "activated_rules": activated_rules,
            "total_rules": len(self.rules),
            "rule_coverage": len(activated_rules) / len(self.rules)
        }