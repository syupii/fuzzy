"""
ファジィルール - core/fuzzy/rules.py
ファジィ推論のためのルール定義と管理
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np


class LogicalOperator(Enum):
    """論理演算子"""
    AND = "and"
    OR = "or"
    NOT = "not"


class AggregationMethod(Enum):
    """集約手法"""
    MIN = "min"          # Mamdani推論
    PRODUCT = "product"  # 積集合
    MAX = "max"          # 最大値
    SUM = "sum"          # 和集合


@dataclass
class FuzzyCondition:
    """ファジィ条件"""
    variable_name: str
    fuzzy_set_name: str
    operator: LogicalOperator = LogicalOperator.AND
    weight: float = 1.0


@dataclass
class FuzzyConclusion:
    """ファジィ結論"""
    variable_name: str
    fuzzy_set_name: str
    weight: float = 1.0


class FuzzyRule:
    """ファジィルール"""
    
    def __init__(self, rule_id: str, conditions: List[FuzzyCondition], 
                 conclusion: FuzzyConclusion, confidence: float = 1.0):
        self.rule_id = rule_id
        self.conditions = conditions
        self.conclusion = conclusion
        self.confidence = confidence
        self.activation_count = 0
        self.total_activation_strength = 0.0
    
    def evaluate_antecedent(self, membership_values: Dict[str, Dict[str, float]],
                          aggregation: AggregationMethod = AggregationMethod.MIN) -> float:
        """前件部の評価"""
        
        condition_strengths = []
        
        for condition in self.conditions:
            var_name = condition.variable_name
            fuzzy_set = condition.fuzzy_set_name
            
            if var_name in membership_values and fuzzy_set in membership_values[var_name]:
                strength = membership_values[var_name][fuzzy_set] * condition.weight
                condition_strengths.append(strength)
            else:
                condition_strengths.append(0.0)
        
        if not condition_strengths:
            return 0.0
        
        # 集約手法による前件部強度計算
        if aggregation == AggregationMethod.MIN:
            antecedent_strength = min(condition_strengths)
        elif aggregation == AggregationMethod.PRODUCT:
            antecedent_strength = np.prod(condition_strengths)
        elif aggregation == AggregationMethod.MAX:
            antecedent_strength = max(condition_strengths)
        elif aggregation == AggregationMethod.SUM:
            antecedent_strength = min(1.0, sum(condition_strengths))
        else:
            antecedent_strength = min(condition_strengths)  # デフォルト
        
        # 統計更新
        if antecedent_strength > 0.1:
            self.activation_count += 1
            self.total_activation_strength += antecedent_strength
        
        return antecedent_strength * self.confidence
    
    def get_statistics(self) -> Dict[str, Any]:
        """ルール統計情報"""
        return {
            'rule_id': self.rule_id,
            'activation_count': self.activation_count,
            'average_strength': self.total_activation_strength / max(1, self.activation_count),
            'confidence': self.confidence,
            'num_conditions': len(self.conditions)
        }
    
    def __str__(self) -> str:
        """ルール文字列表現"""
        condition_strs = []
        for i, cond in enumerate(self.conditions):
            if i > 0 and cond.operator == LogicalOperator.AND:
                condition_strs.append(" AND ")
            elif i > 0 and cond.operator == LogicalOperator.OR:
                condition_strs.append(" OR ")
            
            condition_strs.append(f"{cond.variable_name} is {cond.fuzzy_set_name}")
        
        return (f"IF {''.join(condition_strs)} "
                f"THEN {self.conclusion.variable_name} is {self.conclusion.fuzzy_set_name} "
                f"(CF: {self.confidence:.2f})")


class RuleBase:
    """ファジィルールベース"""
    
    def __init__(self):
        self.rules: List[FuzzyRule] = []
        self.rule_id_counter = 0
    
    def add_rule(self, conditions: List[FuzzyCondition], conclusion: FuzzyConclusion,
                confidence: float = 1.0) -> str:
        """ルール追加"""
        rule_id = f"rule_{self.rule_id_counter:04d}"
        self.rule_id_counter += 1
        
        rule = FuzzyRule(rule_id, conditions, conclusion, confidence)
        self.rules.append(rule)
        
        return rule_id
    
    def remove_rule(self, rule_id: str) -> bool:
        """ルール削除"""
        for i, rule in enumerate(self.rules):
            if rule.rule_id == rule_id:
                del self.rules[i]
                return True
        return False
    
    def get_active_rules(self, membership_values: Dict[str, Dict[str, float]], 
                        threshold: float = 0.1) -> List[tuple]:
        """活性化ルール取得"""
        active_rules = []
        
        for rule in self.rules:
            strength = rule.evaluate_antecedent(membership_values)
            if strength > threshold:
                active_rules.append((rule, strength))
        
        return sorted(active_rules, key=lambda x: x[1], reverse=True)
    
    def evaluate_rules(self, membership_values: Dict[str, Dict[str, float]]) -> Dict[str, List[tuple]]:
        """全ルール評価"""
        results = {}
        
        for rule in self.rules:
            strength = rule.evaluate_antecedent(membership_values)
            
            conclusion_var = rule.conclusion.variable_name
            conclusion_set = rule.conclusion.fuzzy_set_name
            
            if conclusion_var not in results:
                results[conclusion_var] = []
            
            results[conclusion_var].append((conclusion_set, strength, rule.rule_id))
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """ルールベース統計"""
        if not self.rules:
            return {
                'total_rules': 0,
                'average_conditions_per_rule': 0,
                'most_active_rule': None,
                'rule_statistics': []
            }
        
        rule_stats = [rule.get_statistics() for rule in self.rules]
        most_active = max(self.rules, key=lambda r: r.activation_count)
        
        return {
            'total_rules': len(self.rules),
            'average_conditions_per_rule': np.mean([len(r.conditions) for r in self.rules]),
            'most_active_rule': most_active.rule_id,
            'rule_statistics': rule_stats
        }
    
    def __len__(self) -> int:
        return len(self.rules)
    
    def __iter__(self):
        return iter(self.rules)


class RuleGenerator:
    """ファジィルール自動生成器"""
    
    @staticmethod
    def generate_basic_rules(input_variables: List[str], output_variable: str,
                           fuzzy_sets: Dict[str, List[str]]) -> RuleBase:
        """基本的なルール生成"""
        rule_base = RuleBase()
        
        # 各入力変数の組み合わせでルールを生成
        input_sets = [fuzzy_sets.get(var, ['Low', 'Medium', 'High']) for var in input_variables]
        output_sets = fuzzy_sets.get(output_variable, ['Low', 'Medium', 'High'])
        
        # 簡単なヒューリスティック: 入力の平均に基づく出力決定
        for combo_idx, input_combo in enumerate(RuleGenerator._generate_combinations(input_sets)):
            conditions = []
            for var_idx, var_name in enumerate(input_variables):
                condition = FuzzyCondition(var_name, input_combo[var_idx])
                conditions.append(condition)
            
            # 出力決定（簡単なルール）
            output_level = RuleGenerator._determine_output_level(input_combo, output_sets)
            conclusion = FuzzyConclusion(output_variable, output_level)
            
            rule_base.add_rule(conditions, conclusion, confidence=0.8)
        
        return rule_base
    
    @staticmethod
    def _generate_combinations(sets_list: List[List[str]]) -> List[List[str]]:
        """集合の直積生成"""
        if not sets_list:
            return [[]]
        
        result = []
        for item in sets_list[0]:
            for combo in RuleGenerator._generate_combinations(sets_list[1:]):
                result.append([item] + combo)
        
        return result
    
    @staticmethod
    def _determine_output_level(input_combo: List[str], output_sets: List[str]) -> str:
        """入力組み合わせから出力レベル決定"""
        # 簡単なヒューリスティック
        level_mapping = {'Low': 0, 'Medium': 1, 'High': 2, 'Very Low': -1, 'Very High': 3}
        
        input_levels = [level_mapping.get(level, 1) for level in input_combo]
        avg_level = np.mean(input_levels)
        
        if avg_level <= 0.5:
            return output_sets[0] if len(output_sets) > 0 else 'Low'
        elif avg_level >= 1.5:
            return output_sets[-1] if len(output_sets) > 0 else 'High'
        else:
            return output_sets[len(output_sets)//2] if len(output_sets) > 0 else 'Medium'
    
    @staticmethod
    def generate_expert_rules() -> RuleBase:
        """専門家知識に基づくルール生成（研究室選択向け）"""
        rule_base = RuleBase()
        
        # 研究室選択の専門知識ルール
        expert_rules = [
            # 研究集約度が高く、理論志向なら高適合
            ([("research_intensity", "High"), ("theory_practice", "High")], 
             ("compatibility", "High"), 0.9),
            
            # チームワークとアドバイザースタイルが合致
            ([("team_work", "High"), ("advisor_style", "High")], 
             ("compatibility", "High"), 0.8),
            
            # 負荷が高すぎる場合は適合度低下
            ([("workload", "High"), ("theory_practice", "Low")], 
             ("compatibility", "Low"), 0.7),
            
            # バランス型
            ([("research_intensity", "Medium"), ("team_work", "Medium"), ("workload", "Medium")],
             ("compatibility", "Medium"), 0.6)
        ]
        
        for conditions_data, conclusion_data, confidence in expert_rules:
            conditions = [FuzzyCondition(var, fuzzy_set) for var, fuzzy_set in conditions_data]
            conclusion = FuzzyConclusion(conclusion_data[0], conclusion_data[1])
            
            rule_base.add_rule(conditions, conclusion, confidence)
        
        return rule_base
    
    @staticmethod
    def generate_data_driven_rules(training_data: np.ndarray, input_names: List[str],
                                 output_name: str, min_support: float = 0.1) -> RuleBase:
        """データ駆動型ルール生成"""
        rule_base = RuleBase()
        
        # 簡単な関連ルール生成アルゴリズム
        # 実装は簡略化版
        
        # データからパターンを抽出してルール化
        # ここでは基本的な相関関係に基づくルール生成
        
        n_samples = len(training_data)
        min_samples = int(n_samples * min_support)
        
        # 各入力変数と出力の関係を分析
        for i, input_name in enumerate(input_names):
            input_data = training_data[:, i]
            output_data = training_data[:, -1]  # 出力は最後の列と仮定
            
            # 高い相関がある場合のルール生成
            correlation = np.corrcoef(input_data, output_data)[0, 1]
            
            if abs(correlation) > 0.5:  # 相関が強い場合
                # データを3分割してルール作成
                input_percentiles = np.percentile(input_data, [33, 67])
                output_percentiles = np.percentile(output_data, [33, 67])
                
                # Low-Low, Medium-Medium, High-High パターン
                if correlation > 0:  # 正の相関
                    rules_data = [
                        ([(input_name, "Low")], ("compatibility", "Low"), 0.6),
                        ([(input_name, "Medium")], ("compatibility", "Medium"), 0.6),
                        ([(input_name, "High")], ("compatibility", "High"), 0.6)
                    ]
                else:  # 負の相関
                    rules_data = [
                        ([(input_name, "Low")], ("compatibility", "High"), 0.6),
                        ([(input_name, "Medium")], ("compatibility", "Medium"), 0.6),
                        ([(input_name, "High")], ("compatibility", "Low"), 0.6)
                    ]
                
                for conditions_data, conclusion_data, confidence in rules_data:
                    conditions = [FuzzyCondition(var, fuzzy_set) for var, fuzzy_set in conditions_data]
                    conclusion = FuzzyConclusion(conclusion_data[0], conclusion_data[1])
                    rule_base.add_rule(conditions, conclusion, confidence)
        
        return rule_base


class RuleOptimizer:
    """ルール最適化器"""
    
    def __init__(self, rule_base: RuleBase):
        self.rule_base = rule_base
    
    def optimize_confidence_values(self, validation_data: Dict[str, Any]) -> None:
        """信頼度値の最適化"""
        # 検証データに基づいてルールの信頼度を調整
        for rule in self.rule_base.rules:
            # ルールの性能評価
            performance = self._evaluate_rule_performance(rule, validation_data)
            
            # 性能に基づく信頼度調整
            rule.confidence = min(1.0, rule.confidence * performance)
    
    def prune_weak_rules(self, threshold: float = 0.1) -> int:
        """弱いルールの除去"""
        initial_count = len(self.rule_base.rules)
        
        # 活性化頻度や性能の低いルールを除去
        self.rule_base.rules = [
            rule for rule in self.rule_base.rules
            if rule.confidence > threshold and rule.activation_count > 0
        ]
        
        return initial_count - len(self.rule_base.rules)
    
    def _evaluate_rule_performance(self, rule: FuzzyRule, validation_data: Dict[str, Any]) -> float:
        """個別ルールの性能評価"""
        # 簡単な性能評価（実装簡略化）
        if rule.activation_count == 0:
            return 0.1
        
        average_strength = rule.total_activation_strength / rule.activation_count
        return min(1.0, average_strength * 1.2)


class FuzzyRuleSystem:
    """ファジィルールシステム統合クラス"""
    
    def __init__(self):
        self.rule_base = RuleBase()
        self.optimizer = RuleOptimizer(self.rule_base)
        self.input_variables: List[str] = []
        self.output_variables: List[str] = []
        
    def add_expert_knowledge_rules(self):
        """専門家知識ルールの追加"""
        expert_rule_base = RuleGenerator.generate_expert_rules()
        self.rule_base.rules.extend(expert_rule_base.rules)
    
    def add_data_driven_rules(self, training_data: np.ndarray, 
                             input_names: List[str], output_name: str):
        """データ駆動型ルールの追加"""
        data_rule_base = RuleGenerator.generate_data_driven_rules(
            training_data, input_names, output_name
        )
        self.rule_base.rules.extend(data_rule_base.rules)
    
    def optimize_rules(self, validation_data: Dict[str, Any]):
        """ルールの最適化"""
        self.optimizer.optimize_confidence_values(validation_data)
        pruned_count = self.optimizer.prune_weak_rules()
        return pruned_count
    
    def evaluate_input(self, membership_values: Dict[str, Dict[str, float]]) -> Dict[str, List[tuple]]:
        """入力に対するルール評価"""
        return self.rule_base.evaluate_rules(membership_values)
    
    def get_system_info(self) -> Dict[str, Any]:
        """システム情報の取得"""
        return {
            'total_rules': len(self.rule_base.rules),
            'input_variables': self.input_variables,
            'output_variables': self.output_variables,
            'rule_statistics': self.rule_base.get_statistics()
        }ール生成
            correlation = np.corrcoef(input_data, output_data)[0, 1]
            
            if abs(correlation) > 0.5:  # 相関が強い場合
                # データを3分割してルール作成
                input_percentiles = np.percentile(input_data, [33, 67])
                output_percentiles = np.percentile(output_data, [33, 67])
                
                # Low-Low, Medium-Medium, High-High パターン
                if correlation > 0:  # 正の相関
                    rules_data = [
                        ([(input_name, "Low")], ("compatibility", "Low"), 0.6),
                        ([(input_name, "Medium")], ("compatibility", "Medium"), 0.6),
                        ([(input_name, "High")], ("compatibility", "High"), 0.6)
                    ]
                else:  # 負の相関
                    rules_data = [
                        ([(input_name, "Low")], ("compatibility", "High"), 0.6),
                        ([(input_name, "Medium")], ("compatibility", "Medium"), 0.6),
                        ([(input_name, "High")], ("compatibility", "Low"), 0.6)
                    ]
                
                for conditions_data, conclusion_data, confidence in rules_data:
                    conditions = [FuzzyCondition(var, fuzzy_set) for var, fuzzy_set in conditions_data]
                    conclusion = FuzzyConclusion(conclusion_data[0], conclusion_data[1])
                    rule_base.add_rule(conditions, conclusion, confidence)
        
        return rule_base


class RuleOptimizer:
    """ルール最適化器"""
    
    def __init__(self, rule_base: RuleBase):
        self.rule_base = rule_base
    
    def optimize_confidence_values(self, validation_data: Dict[str, Any]) -> None:
        """信頼度値の最適化"""
        # 検証データに基づいてルールの信頼度を調整
        for rule in self.rule_base.rules:
            # ルールの性能評価
            performance = self._evaluate_rule_performance(rule, validation_data)
            
            # 性能に基づく信頼度調整
            rule.confidence = min(1.0, rule.confidence * performance)
    
    def prune_weak_rules(self, threshold: float = 0.1) -> int:
        """弱いルールの除去"""
        initial_count = len(self.rule_base.rules)
        
        # 活性化頻度や性能の低いルールを除去
        self.rule_base.rules = [
            rule for rule in self.rule_base.rules
            if rule.confidence > threshold and rule.activation_count > 0
        ]
        
        return initial_count - len(self.rule_base.rules)
    
    def _evaluate_rule_performance(self, rule: FuzzyRule, validation_data: Dict[str, Any]) -> float:
        """個別ルールの性能評価"""
        # 簡単な性能評価（実装簡略化）
        if rule.activation_count == 0:
            return 0.1
        
        average_strength = rule.total_activation_strength / rule.activation_count
        return min(1.0, average_strength * 1.2)