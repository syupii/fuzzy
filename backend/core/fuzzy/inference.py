# backend/core/fuzzy/inference.py - ファジィ推論エンジン
# 研究室選択支援システム用

import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

class MembershipFunction(Enum):
    """メンバーシップ関数の種類"""
    TRIANGULAR = "triangular"
    TRAPEZOIDAL = "trapezoidal"
    GAUSSIAN = "gaussian"
    LINEAR = "linear"

class FuzzySet:
    """ファジィ集合クラス"""
    
    def __init__(self, name: str, function_type: MembershipFunction, parameters: List[float]):
        self.name = name
        self.function_type = function_type
        self.parameters = parameters
    
    def membership(self, x: float) -> float:
        """メンバーシップ値を計算"""
        if self.function_type == MembershipFunction.TRIANGULAR:
            return self._triangular_membership(x)
        elif self.function_type == MembershipFunction.TRAPEZOIDAL:
            return self._trapezoidal_membership(x)
        elif self.function_type == MembershipFunction.GAUSSIAN:
            return self._gaussian_membership(x)
        elif self.function_type == MembershipFunction.LINEAR:
            return self._linear_membership(x)
        else:
            return 0.0
    
    def _triangular_membership(self, x: float) -> float:
        """三角メンバーシップ関数"""
        if len(self.parameters) != 3:
            return 0.0
        
        a, b, c = self.parameters
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x < c:
            return (c - x) / (c - b)
        else:
            return 0.0
    
    def _trapezoidal_membership(self, x: float) -> float:
        """台形メンバーシップ関数"""
        if len(self.parameters) != 4:
            return 0.0
        
        a, b, c, d = self.parameters
        if x <= a or x >= d:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        elif b < x <= c:
            return 1.0
        elif c < x < d:
            return (d - x) / (d - c)
        else:
            return 0.0
    
    def _gaussian_membership(self, x: float) -> float:
        """ガウシアンメンバーシップ関数"""
        if len(self.parameters) != 2:
            return 0.0
        
        center, sigma = self.parameters
        return math.exp(-0.5 * ((x - center) / sigma) ** 2)
    
    def _linear_membership(self, x: float) -> float:
        """線形メンバーシップ関数"""
        if len(self.parameters) != 2:
            return 0.0
        
        a, b = self.parameters
        if b == a:
            return 1.0 if x == a else 0.0
        
        result = (x - a) / (b - a)
        return max(0.0, min(1.0, result))

class FuzzyVariable:
    """ファジィ変数クラス"""
    
    def __init__(self, name: str, range_min: float = 0.0, range_max: float = 10.0):
        self.name = name
        self.range_min = range_min
        self.range_max = range_max
        self.fuzzy_sets: Dict[str, FuzzySet] = {}
    
    def add_fuzzy_set(self, fuzzy_set: FuzzySet):
        """ファジィ集合を追加"""
        self.fuzzy_sets[fuzzy_set.name] = fuzzy_set
    
    def get_membership_values(self, x: float) -> Dict[str, float]:
        """すべてのファジィ集合のメンバーシップ値を取得"""
        return {name: fuzzy_set.membership(x) for name, fuzzy_set in self.fuzzy_sets.items()}

class FuzzyRule:
    """ファジィルールクラス"""
    
    def __init__(self, rule_id: str, antecedent: Dict[str, str], consequent: Dict[str, str], weight: float = 1.0):
        self.rule_id = rule_id
        self.antecedent = antecedent  # {"variable_name": "fuzzy_set_name"}
        self.consequent = consequent
        self.weight = weight
    
    def evaluate(self, input_values: Dict[str, float], variables: Dict[str, FuzzyVariable]) -> Dict[str, float]:
        """ルールを評価"""
        # 前件部の評価（AND演算 = 最小値）
        antecedent_strength = 1.0
        
        for var_name, fuzzy_set_name in self.antecedent.items():
            if var_name in input_values and var_name in variables:
                membership = variables[var_name].fuzzy_sets[fuzzy_set_name].membership(input_values[var_name])
                antecedent_strength = min(antecedent_strength, membership)
        
        # 後件部への適用
        result = {}
        for var_name, fuzzy_set_name in self.consequent.items():
            result[f"{var_name}_{fuzzy_set_name}"] = antecedent_strength * self.weight
        
        return result

class SimpleFuzzyInferenceEngine:
    """シンプルなファジィ推論エンジン"""
    
    def __init__(self):
        self.variables: Dict[str, FuzzyVariable] = {}
        self.rules: List[FuzzyRule] = []
        self._initialize_default_variables()
    
    def _initialize_default_variables(self):
        """デフォルトの変数とファジィ集合を初期化"""
        
        # 研究強度の変数
        research_intensity = FuzzyVariable("research_intensity", 1.0, 10.0)
        research_intensity.add_fuzzy_set(FuzzySet("low", MembershipFunction.TRIANGULAR, [1, 1, 4]))
        research_intensity.add_fuzzy_set(FuzzySet("medium", MembershipFunction.TRIANGULAR, [2, 5, 8]))
        research_intensity.add_fuzzy_set(FuzzySet("high", MembershipFunction.TRIANGULAR, [6, 10, 10]))
        self.variables["research_intensity"] = research_intensity
        
        # 指導スタイル
        advisor_style = FuzzyVariable("advisor_style", 1.0, 10.0)
        advisor_style.add_fuzzy_set(FuzzySet("strict", MembershipFunction.TRIANGULAR, [1, 1, 4]))
        advisor_style.add_fuzzy_set(FuzzySet("balanced", MembershipFunction.TRIANGULAR, [3, 5, 7]))
        advisor_style.add_fuzzy_set(FuzzySet("flexible", MembershipFunction.TRIANGULAR, [6, 10, 10]))
        self.variables["advisor_style"] = advisor_style
        
        # チームワーク
        team_work = FuzzyVariable("team_work", 1.0, 10.0)
        team_work.add_fuzzy_set(FuzzySet("individual", MembershipFunction.TRIANGULAR, [1, 1, 4]))
        team_work.add_fuzzy_set(FuzzySet("mixed", MembershipFunction.TRIANGULAR, [3, 5, 7]))
        team_work.add_fuzzy_set(FuzzySet("collaborative", MembershipFunction.TRIANGULAR, [6, 10, 10]))
        self.variables["team_work"] = team_work
        
        # 適合性出力変数
        compatibility = FuzzyVariable("compatibility", 0.0, 1.0)
        compatibility.add_fuzzy_set(FuzzySet("low", MembershipFunction.TRIANGULAR, [0, 0, 0.4]))
        compatibility.add_fuzzy_set(FuzzySet("medium", MembershipFunction.TRIANGULAR, [0.2, 0.5, 0.8]))
        compatibility.add_fuzzy_set(FuzzySet("high", MembershipFunction.TRIANGULAR, [0.6, 1.0, 1.0]))
        self.variables["compatibility"] = compatibility
        
        # デフォルトルールの追加
        self._add_default_rules()
    
    def _add_default_rules(self):
        """デフォルトのファジィルールを追加"""
        
        # ルール1: 高い研究強度 + 柔軟な指導 + 協調的 → 高い適合性
        rule1 = FuzzyRule(
            "rule_1",
            {"research_intensity": "high", "advisor_style": "flexible", "team_work": "collaborative"},
            {"compatibility": "high"},
            weight=1.0
        )
        self.rules.append(rule1)
        
        # ルール2: 低い研究強度 + 厳格な指導 → 低い適合性
        rule2 = FuzzyRule(
            "rule_2",
            {"research_intensity": "low", "advisor_style": "strict"},
            {"compatibility": "low"},
            weight=0.8
        )
        self.rules.append(rule2)
        
        # ルール3: 中程度の研究強度 + バランス型指導 → 中程度の適合性
        rule3 = FuzzyRule(
            "rule_3",
            {"research_intensity": "medium", "advisor_style": "balanced"},
            {"compatibility": "medium"},
            weight=0.9
        )
        self.rules.append(rule3)
        
        # ルール4: 個人研究志向 + 厳格指導 → 中程度の適合性
        rule4 = FuzzyRule(
            "rule_4",
            {"team_work": "individual", "advisor_style": "strict"},
            {"compatibility": "medium"},
            weight=0.7
        )
        self.rules.append(rule4)
    
    def add_variable(self, variable: FuzzyVariable):
        """ファジィ変数を追加"""
        self.variables[variable.name] = variable
    
    def add_rule(self, rule: FuzzyRule):
        """ファジィルールを追加"""
        self.rules.append(rule)
    
    def infer(self, input_values: Dict[str, float]) -> Dict[str, float]:
        """ファジィ推論を実行"""
        
        # すべてのルールを評価
        rule_outputs = []
        
        for rule in self.rules:
            try:
                output = rule.evaluate(input_values, self.variables)
                rule_outputs.append(output)
            except Exception as e:
                print(f"⚠️ ルール評価エラー: {rule.rule_id} - {e}")
                continue
        
        # 出力の集約（最大値法）
        aggregated_output = {}
        
        for output in rule_outputs:
            for key, value in output.items():
                if key in aggregated_output:
                    aggregated_output[key] = max(aggregated_output[key], value)
                else:
                    aggregated_output[key] = value
        
        # 非ファジィ化（重心法）
        final_output = {}
        
        for var_name in self.variables:
            if var_name == "compatibility":  # 出力変数の場合
                fuzzy_outputs = {k: v for k, v in aggregated_output.items() if k.startswith(var_name)}
                
                if fuzzy_outputs:
                    final_output[var_name] = self._defuzzify(var_name, fuzzy_outputs)
                else:
                    final_output[var_name] = 0.5  # デフォルト値
        
        return final_output
    
    def _defuzzify(self, var_name: str, fuzzy_outputs: Dict[str, float]) -> float:
        """非ファジィ化（重心法）"""
        
        variable = self.variables[var_name]
        
        # 離散化された値での重心計算
        step = (variable.range_max - variable.range_min) / 100
        numerator = 0.0
        denominator = 0.0
        
        for i in range(101):
            x = variable.range_min + i * step
            membership_sum = 0.0
            
            # 各ファジィ集合の影響を計算
            for fuzzy_key, strength in fuzzy_outputs.items():
                fuzzy_set_name = fuzzy_key.split('_')[-1]
                if fuzzy_set_name in variable.fuzzy_sets:
                    membership = variable.fuzzy_sets[fuzzy_set_name].membership(x)
                    membership_sum += min(membership, strength)
            
            numerator += x * membership_sum
            denominator += membership_sum
        
        if denominator == 0:
            return (variable.range_min + variable.range_max) / 2
        
        return numerator / denominator

class AdvancedFuzzyInferenceEngine(SimpleFuzzyInferenceEngine):
    """高度なファジィ推論エンジン（13項目評価対応）"""
    
    def __init__(self):
        super().__init__()
        self._initialize_extended_variables()
        self._add_extended_rules()
    
    def _initialize_extended_variables(self):
        """拡張変数の初期化"""
        
        # 研究分野適合性
        field_match = FuzzyVariable("research_field_match", 1.0, 10.0)
        field_match.add_fuzzy_set(FuzzySet("poor", MembershipFunction.TRIANGULAR, [1, 1, 3]))
        field_match.add_fuzzy_set(FuzzySet("good", MembershipFunction.TRIANGULAR, [2, 5, 8]))
        field_match.add_fuzzy_set(FuzzySet("excellent", MembershipFunction.TRIANGULAR, [7, 10, 10]))
        self.variables["research_field_match"] = field_match
        
        # スキル開発
        skill_dev = FuzzyVariable("skill_development", 1.0, 10.0)
        skill_dev.add_fuzzy_set(FuzzySet("specialized", MembershipFunction.TRIANGULAR, [1, 1, 4]))
        skill_dev.add_fuzzy_set(FuzzySet("balanced", MembershipFunction.TRIANGULAR, [3, 5, 7]))
        skill_dev.add_fuzzy_set(FuzzySet("broad", MembershipFunction.TRIANGULAR, [6, 10, 10]))
        self.variables["skill_development"] = skill_dev
        
        # 研究室雰囲気
        atmosphere = FuzzyVariable("lab_atmosphere", 1.0, 10.0)
        atmosphere.add_fuzzy_set(FuzzySet("quiet", MembershipFunction.TRIANGULAR, [1, 1, 4]))
        atmosphere.add_fuzzy_set(FuzzySet("moderate", MembershipFunction.TRIANGULAR, [3, 5, 7]))
        atmosphere.add_fuzzy_set(FuzzySet("active", MembershipFunction.TRIANGULAR, [6, 10, 10]))
        self.variables["lab_atmosphere"] = atmosphere
        
        # 柔軟性
        flexibility = FuzzyVariable("flexibility", 1.0, 10.0)
        flexibility.add_fuzzy_set(FuzzySet("fixed", MembershipFunction.TRIANGULAR, [1, 1, 4]))
        flexibility.add_fuzzy_set(FuzzySet("somewhat_flexible", MembershipFunction.TRIANGULAR, [3, 5, 7]))
        flexibility.add_fuzzy_set(FuzzySet("very_flexible", MembershipFunction.TRIANGULAR, [6, 10, 10]))
        self.variables["flexibility"] = flexibility
        
        # 論文発表機会
        publication = FuzzyVariable("publication_opportunity", 1.0, 10.0)
        publication.add_fuzzy_set(FuzzySet("limited", MembershipFunction.TRIANGULAR, [1, 1, 4]))
        publication.add_fuzzy_set(FuzzySet("moderate", MembershipFunction.TRIANGULAR, [3, 5, 7]))
        publication.add_fuzzy_set(FuzzySet("abundant", MembershipFunction.TRIANGULAR, [6, 10, 10]))
        self.variables["publication_opportunity"] = publication
    
    def _add_extended_rules(self):
        """拡張ルールの追加"""
        
        # 研究分野適合性が重要なルール
        rule_field = FuzzyRule(
            "rule_field_match",
            {"research_field_match": "excellent", "research_intensity": "high"},
            {"compatibility": "high"},
            weight=1.2
        )
        self.rules.append(rule_field)
        
        # スキル開発重視のルール
        rule_skill = FuzzyRule(
            "rule_skill_broad",
            {"skill_development": "broad", "flexibility": "very_flexible"},
            {"compatibility": "high"},
            weight=1.0
        )
        self.rules.append(rule_skill)
        
        # 論文発表機会重視のルール
        rule_publication = FuzzyRule(
            "rule_publication",
            {"publication_opportunity": "abundant", "research_intensity": "high"},
            {"compatibility": "high"},
            weight=1.1
        )
        self.rules.append(rule_publication)
    
    def calculate_detailed_compatibility(self, student_profile: Dict[str, float], lab_features: Dict[str, float]) -> Dict[str, Any]:
        """詳細適合性計算"""
        
        # 各項目の類似度計算
        similarities = {}
        
        for criterion in student_profile:
            if criterion in lab_features:
                student_val = student_profile[criterion]
                lab_val = lab_features[criterion]
                
                # 正規化（1-10 → 0-1）
                student_norm = (student_val - 1) / 9
                lab_norm = (lab_val - 1) / 9
                
                # 類似度（距離ベース）
                similarity = 1.0 - abs(student_norm - lab_norm)
                similarities[criterion] = similarity
        
        # ファジィ推論による総合評価
        fuzzy_result = self.infer(student_profile)
        
        # 重要度重み付け
        weights = {
            "research_intensity": 1.2,
            "research_field_match": 1.3,
            "advisor_style": 1.1,
            "team_work": 1.0,
            "workload": 0.9,
            "theory_practice": 1.0,
            "skill_development": 1.1,
            "lab_atmosphere": 0.8,
            "flexibility": 0.9,
            "publication_opportunity": 1.2,
            "interdisciplinary": 0.8,
            "communication_style": 0.7,
            "innovation_risk": 0.9
        }
        
        # 重み付き類似度
        weighted_score = 0.0
        total_weight = 0.0
        
        for criterion, similarity in similarities.items():
            weight = weights.get(criterion, 1.0)
            weighted_score += similarity * weight
            total_weight += weight
        
        final_compatibility = weighted_score / total_weight if total_weight > 0 else 0.5
        
        # ファジィ推論結果と組み合わせ
        fuzzy_compatibility = fuzzy_result.get("compatibility", 0.5)
        combined_compatibility = (final_compatibility * 0.7 + fuzzy_compatibility * 0.3)
        
        return {
            "total_compatibility": min(1.0, combined_compatibility),
            "fuzzy_inference_result": fuzzy_compatibility,
            "weighted_similarity": final_compatibility,
            "criterion_similarities": similarities,
            "applied_weights": {k: v for k, v in weights.items() if k in similarities}
        }

# テスト用のメイン関数
if __name__ == "__main__":
    print("🧪 ファジィ推論エンジンテスト開始...")
    
    # シンプルエンジンのテスト
    simple_engine = SimpleFuzzyInferenceEngine()
    
    test_input = {
        "research_intensity": 8.5,
        "advisor_style": 7.0,
        "team_work": 8.0
    }
    
    result = simple_engine.infer(test_input)
    print(f"📊 シンプル推論結果: {result}")
    
    # 高度エンジンのテスト
    advanced_engine = AdvancedFuzzyInferenceEngine()
    
    student_profile = {
        "research_intensity": 8.5,
        "advisor_style": 7.0,
        "team_work": 8.0,
        "research_field_match": 9.0,
        "skill_development": 7.5,
        "publication_opportunity": 8.0
    }
    
    lab_features = {
        "research_intensity": 8.0,
        "advisor_style": 7.5,
        "team_work": 8.5,
        "research_field_match": 8.5,
        "skill_development": 8.0,
        "publication_opportunity": 7.5
    }
    
    detailed_result = advanced_engine.calculate_detailed_compatibility(student_profile, lab_features)
    print(f"📈 詳細適合性結果:")
    for key, value in detailed_result.items():
        print(f"  {key}: {value}")
    
    print("✅ テスト完了")