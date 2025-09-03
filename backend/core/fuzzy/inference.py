"""
ファジィ推論エンジン - core/fuzzy/inference.py
ファジィ推論の実行と非ファジィ化
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from enum import Enum

from .membership import MembershipFunction, MembershipFunctionFactory, MembershipType
from .rules import FuzzyRule, RuleBase, AggregationMethod


class DefuzzificationMethod(Enum):
    """非ファジィ化手法"""
    CENTROID = "centroid"           # 重心法
    MAX_MEMBERSHIP = "max_membership"  # 最大メンバーシップ法
    WEIGHTED_AVERAGE = "weighted_average"  # 重み付き平均法
    FIRST_MAX = "first_max"        # 最初の最大値


class FuzzyInferenceEngine:
    """ファジィ推論エンジン"""
    
    def __init__(self):
        self.input_variables: Dict[str, List[MembershipFunction]] = {}
        self.output_variables: Dict[str, List[MembershipFunction]] = {}
        self.rule_base = RuleBase()
        self.inference_count = 0
        self.total_inference_time = 0.0
    
    def add_input_variable(self, var_name: str, mfs: List[MembershipFunction]):
        """入力変数とメンバーシップ関数を追加"""
        self.input_variables[var_name] = mfs
    
    def add_output_variable(self, var_name: str, mfs: List[MembershipFunction]):
        """出力変数とメンバーシップ関数を追加"""
        self.output_variables[var_name] = mfs
    
    def add_rule(self, rule: FuzzyRule):
        """ルールを追加"""
        self.rule_base.rules.append(rule)
    
    def infer(self, inputs: Dict[str, float], 
             defuzzification: DefuzzificationMethod = DefuzzificationMethod.WEIGHTED_AVERAGE) -> Dict[str, float]:
        """ファジィ推論実行"""
        
        import time
        start_time = time.time()
        
        try:
            # Step 1: ファジィ化 (Fuzzification)
            membership_values = self._fuzzify(inputs)
            
            # Step 2: ルール評価
            rule_outputs = self._evaluate_rules(membership_values)
            
            # Step 3: 集約
            aggregated_outputs = self._aggregate_outputs(rule_outputs)
            
            # Step 4: 非ファジィ化
            crisp_outputs = self._defuzzify(aggregated_outputs, defuzzification)
            
            # 統計更新
            self.inference_count += 1
            self.total_inference_time += time.time() - start_time
            
            return crisp_outputs
            
        except Exception as e:
            print(f"Inference error: {e}")
            return {var: 0.5 for var in self.output_variables.keys()}
    
    def _fuzzify(self, inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """ファジィ化：クリスプ値をファジィ値に変換"""
        membership_values = {}
        
        for var_name, value in inputs.items():
            if var_name in self.input_variables:
                membership_values[var_name] = {}
                
                for mf in self.input_variables[var_name]:
                    membership_degree = mf.membership(value)
                    membership_values[var_name][mf.name.split('_')[-1]] = membership_degree
        
        return membership_values
    
    def _evaluate_rules(self, membership_values: Dict[str, Dict[str, float]]) -> Dict[str, List[Tuple[str, float, str]]]:
        """ルール評価"""
        return self.rule_base.evaluate_rules(membership_values)
    
    def _aggregate_outputs(self, rule_outputs: Dict[str, List[Tuple[str, float, str]]]) -> Dict[str, Dict[str, float]]:
        """出力の集約"""
        aggregated = {}
        
        for output_var, outputs in rule_outputs.items():
            aggregated[output_var] = {}
            
            # 同じファジィ集合に対する複数のルール出力を集約
            fuzzy_set_outputs = {}
            for fuzzy_set, strength, rule_id in outputs:
                if fuzzy_set not in fuzzy_set_outputs:
                    fuzzy_set_outputs[fuzzy_set] = []
                fuzzy_set_outputs[fuzzy_set].append(strength)
            
            # 最大値で集約（Mamdani推論）
            for fuzzy_set, strengths in fuzzy_set_outputs.items():
                aggregated[output_var][fuzzy_set] = max(strengths) if strengths else 0.0
        
        return aggregated
    
    def _defuzzify(self, aggregated_outputs: Dict[str, Dict[str, float]], 
                  method: DefuzzificationMethod) -> Dict[str, float]:
        """非ファジィ化：ファジィ値をクリスプ値に変換"""
        crisp_outputs = {}
        
        for output_var, fuzzy_values in aggregated_outputs.items():
            if output_var not in self.output_variables:
                crisp_outputs[output_var] = 0.5
                continue
            
            if method == DefuzzificationMethod.WEIGHTED_AVERAGE:
                crisp_outputs[output_var] = self._weighted_average_defuzzify(
                    output_var, fuzzy_values)
            elif method == DefuzzificationMethod.CENTROID:
                crisp_outputs[output_var] = self._centroid_defuzzify(
                    output_var, fuzzy_values)
            elif method == DefuzzificationMethod.MAX_MEMBERSHIP:
                crisp_outputs[output_var] = self._max_membership_defuzzify(
                    output_var, fuzzy_values)
            else:
                crisp_outputs[output_var] = self._weighted_average_defuzzify(
                    output_var, fuzzy_values)
        
        return crisp_outputs
    
    def _weighted_average_defuzzify(self, output_var: str, fuzzy_values: Dict[str, float]) -> float:
        """重み付き平均法による非ファジィ化"""
        numerator = 0.0
        denominator = 0.0
        
        for mf in self.output_variables[output_var]:
            fuzzy_set_name = mf.name.split('_')[-1]
            
            if fuzzy_set_name in fuzzy_values:
                strength = fuzzy_values[fuzzy_set_name]
                
                if strength > 0:
                    # メンバーシップ関数の代表値を取得
                    representative_value = self._get_representative_value(mf)
                    numerator += strength * representative_value
                    denominator += strength
        
        return numerator / denominator if denominator > 0 else 0.5
    
    def _centroid_defuzzify(self, output_var: str, fuzzy_values: Dict[str, float]) -> float:
        """重心法による非ファジィ化"""
        # 簡易実装：離散的な重心計算
        x_values = np.linspace(0, 10, 101)  # 0-10の範囲を101点でサンプリング
        y_values = np.zeros_like(x_values)
        
        for mf in self.output_variables[output_var]:
            fuzzy_set_name = mf.name.split('_')[-1]
            
            if fuzzy_set_name in fuzzy_values:
                strength = fuzzy_values[fuzzy_set_name]
                
                if strength > 0:
                    # 各点でのメンバーシップ度を計算し、強度でクリッピング
                    mf_values = np.array([min(mf.membership(x), strength) for x in x_values])
                    y_values = np.maximum(y_values, mf_values)
        
        # 重心計算
        if np.sum(y_values) > 0:
            return np.sum(x_values * y_values) / np.sum(y_values)
        else:
            return 5.0  # デフォルト値
    
    def _max_membership_defuzzify(self, output_var: str, fuzzy_values: Dict[str, float]) -> float:
        """最大メンバーシップ法による非ファジィ化"""
        max_strength = 0.0
        max_value = 5.0
        
        for mf in self.output_variables[output_var]:
            fuzzy_set_name = mf.name.split('_')[-1]
            
            if fuzzy_set_name in fuzzy_values:
                strength = fuzzy_values[fuzzy_set_name]
                
                if strength > max_strength:
                    max_strength = strength
                    max_value = self._get_representative_value(mf)
        
        return max_value
    
    def _get_representative_value(self, mf: MembershipFunction) -> float:
        """メンバーシップ関数の代表値を取得"""
        params = mf.get_params()
        
        if 'b' in params:  # 三角形
            return params['b']
        elif 'center' in params:  # ガウシアン
            return params['center']
        elif 'c' in params and 'd' in params:  # 台形
            return (params['c'] + params['d']) / 2
        else:
            return 5.0  # デフォルト値
    
    def get_explanation(self, inputs: Dict[str, float]) -> Dict[str, Any]:
        """推論過程の説明生成"""
        
        # ファジィ化結果
        membership_values = self._fuzzify(inputs)
        
        # 活性化ルール
        active_rules = self.rule_base.get_active_rules(membership_values, threshold=0.1)
        
        # ルール評価結果
        rule_outputs = self._evaluate_rules(membership_values)
        
        explanation = {
            'inputs': inputs,
            'fuzzification': membership_values,
            'active_rules': [
                {
                    'rule_id': rule.rule_id,
                    'rule_text': str(rule),
                    'activation_strength': strength,
                    'conditions': [
                        {
                            'variable': cond.variable_name,
                            'fuzzy_set': cond.fuzzy_set_name,
                            'membership_degree': membership_values.get(cond.variable_name, {}).get(cond.fuzzy_set_name, 0.0)
                        }
                        for cond in rule.conditions
                    ]
                }
                for rule, strength in active_rules[:5]  # 上位5つのルール
            ],
            'rule_outputs': rule_outputs,
            'inference_statistics': {
                'total_rules': len(self.rule_base.rules),
                'active_rules': len(active_rules),
                'inference_count': self.inference_count
            }
        }
        
        return explanation
    
    def get_engine_statistics(self) -> Dict[str, Any]:
        """エンジン統計情報"""
        return {
            'input_variables': list(self.input_variables.keys()),
            'output_variables': list(self.output_variables.keys()),
            'total_rules': len(self.rule_base.rules),
            'inference_count': self.inference_count,
            'average_inference_time': self.total_inference_time / max(1, self.inference_count),
            'rule_base_statistics': self.rule_base.get_statistics()
        }


class SimpleFuzzyInferenceEngine:
    """簡易ファジィ推論エンジン（軽量版）"""
    
    def __init__(self, feature_names: List[str], output_name: str = "compatibility"):
        self.feature_names = feature_names
        self.output_name = output_name
        
        # 各特徴量に対してファジィ集合を自動生成
        self.input_fuzzy_sets = {}
        for feature in feature_names:
            self.input_fuzzy_sets[feature] = MembershipFunctionFactory.create_fuzzy_sets(
                feature, (0, 10), num_sets=3, mf_type=MembershipType.TRIANGULAR
            )
        
        # 出力ファジィ集合
        self.output_fuzzy_sets = MembershipFunctionFactory.create_fuzzy_sets(
            output_name, (0, 1), num_sets=3, mf_type=MembershipType.TRIANGULAR
        )
    
    def infer(self, inputs: Dict[str, float]) -> float:
        """簡易推論"""
        
        # 各特徴量のファジィ化
        fuzzy_inputs = {}
        for feature, value in inputs.items():
            if feature in self.input_fuzzy_sets:
                fuzzy_inputs[feature] = {}
                for fuzzy_set_name, mf in self.input_fuzzy_sets[feature].items():
                    fuzzy_inputs[feature][fuzzy_set_name] = mf.membership(value)
        
        # 簡易ルール：全特徴量の平均的な適合度
        output_strengths = {'Low': 0.0, 'Medium': 0.0, 'High': 0.0}
        
        for feature, fuzzy_values in fuzzy_inputs.items():
            for fuzzy_set, strength in fuzzy_values.items():
                output_strengths[fuzzy_set] += strength
        
        # 正規化
        total_features = len(fuzzy_inputs)
        for fuzzy_set in output_strengths:
            output_strengths[fuzzy_set] /= max(1, total_features)
        
        # 重み付き平均による非ファジィ化
        output_values = {'Low': 0.2, 'Medium': 0.5, 'High': 0.8}
        
        numerator = sum(output_strengths[fs] * output_values[fs] for fs in output_strengths)
        denominator = sum(output_strengths.values())
        
        return numerator / denominator if denominator > 0 else 0.5