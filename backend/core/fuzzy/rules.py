# core/fuzzy/rules.py - ファジィルール定義

from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)

class LogicalOperator(str, Enum):
    """論理演算子"""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"

class ComparisonOperator(str, Enum):
    """比較演算子"""
    IS = "IS"
    IS_NOT = "IS_NOT"
    GREATER_THAN = ">"
    LESS_THAN = "<"
    EQUAL = "="

@dataclass
class Condition:
    """ファジィルールの条件"""
    variable: str
    operator: ComparisonOperator
    linguistic_value: str
    weight: float = 1.0
    negated: bool = False
    
    def __str__(self) -> str:
        neg_str = "NOT " if self.negated else ""
        return f"{neg_str}{self.variable} {self.operator.value} {self.linguistic_value}"

@dataclass
class Conclusion:
    """ファジィルールの結論"""
    variable: str
    linguistic_value: str
    certainty_factor: float = 1.0
    
    def __str__(self) -> str:
        return f"{self.variable} IS {self.linguistic_value}"

class RuleOperator:
    """ルール演算子（t-norm, t-conorm等）"""
    
    @staticmethod
    def min_and(a: float, b: float) -> float:
        """最小値AND演算"""
        return min(a, b)
    
    @staticmethod
    def product_and(a: float, b: float) -> float:
        """積AND演算"""
        return a * b
    
    @staticmethod
    def lukasiewicz_and(a: float, b: float) -> float:
        """ルカシェヴィッチAND演算"""
        return max(0, a + b - 1)
    
    @staticmethod
    def max_or(a: float, b: float) -> float:
        """最大値OR演算"""
        return max(a, b)
    
    @staticmethod
    def probabilistic_or(a: float, b: float) -> float:
        """確率的OR演算"""
        return a + b - a * b
    
    @staticmethod
    def lukasiewicz_or(a: float, b: float) -> float:
        """ルカシェヴィッチOR演算"""
        return min(1, a + b)
    
    @staticmethod
    def complement(a: float) -> float:
        """補集合（NOT演算）"""
        return 1.0 - a

class FuzzyRule:
    """ファジィルール"""
    
    def __init__(self, name: str, conditions: List[Condition], 
                 conclusion: Conclusion, weight: float = 1.0):
        self.name = name
        self.conditions = conditions
        self.conclusion = conclusion
        self.weight = weight
        
        # 論理演算子の設定
        self.and_operator = RuleOperator.min_and
        self.or_operator = RuleOperator.max_or
        self.not_operator = RuleOperator.complement
        
        # 統計情報
        self.activation_count = 0
        self.total_activation = 0.0
        self.average_activation = 0.0
    
    def evaluate(self, memberships: Dict[str, Dict[str, float]]) -> float:
        """ルールの活性化度を評価"""
        
        if not self.conditions:
            return 0.0
        
        # 全ての条件を評価
        condition_values = []
        
        for condition in self.conditions:
            var_name = condition.variable
            set_name = condition.linguistic_value
            
            if var_name in memberships and set_name in memberships[var_name]:
                value = memberships[var_name][set_name]
                
                # 重み適用
                if condition.weight != 1.0:
                    value = value * condition.weight
                
                # 否定処理
                if condition.negated:
                    value = self.not_operator(value)
                
                condition_values.append(value)
            else:
                # 条件が満たせない場合は0
                condition_values.append(0.0)
        
        # 条件の結合（現在はAND結合のみ）
        activation = condition_values[0]
        for value in condition_values[1:]:
            activation = self.and_operator(activation, value)
        
        # ルール重み適用
        activation *= self.weight
        
        # 統計更新
        self.activation_count += 1
        self.total_activation += activation
        self.average_activation = self.total_activation / self.activation_count
        
        return activation
    
    def set_operators(self, and_op: Callable[[float, float], float] = None,
                     or_op: Callable[[float, float], float] = None,
                     not_op: Callable[[float], float] = None):
        """論理演算子を設定"""
        if and_op:
            self.and_operator = and_op
        if or_op:
            self.or_operator = or_op
        if not_op:
            self.not_operator = not_op
    
    def __str__(self) -> str:
        conditions_str = " AND ".join(str(cond) for cond in self.conditions)
        return f"IF {conditions_str} THEN {self.conclusion}"

class FuzzyRuleSet:
    """ファジィルール集合"""
    
    def __init__(self, name: str):
        self.name = name
        self.rules: List[FuzzyRule] = []
        
        # 統計情報
        self.total_inferences = 0
        self.successful_inferences = 0
        
    def add_rule(self, rule: FuzzyRule):
        """ルールを追加"""
        self.rules.append(rule)
    
    def remove_rule(self, rule_name: str) -> bool:
        """ルールを削除"""
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                del self.rules[i]
                return True
        return False
    
    def get_rule(self, rule_name: str) -> Optional[FuzzyRule]:
        """ルールを取得"""
        for rule in self.rules:
            if rule.name == rule_name:
                return rule
        return None
    
    def evaluate_all(self, memberships: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """全ルールを評価"""
        
        rule_activations = {}
        
        for rule in self.rules:
            try:
                activation = rule.evaluate(memberships)
                rule_activations[rule.name] = activation
            except Exception as e:
                logger.warning(f"ルール評価エラー {rule.name}: {e}")
                rule_activations[rule.name] = 0.0
        
        self.total_inferences += 1
        if any(act > 0 for act in rule_activations.values()):
            self.successful_inferences += 1
            
        return rule_activations
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報を取得"""
        
        success_rate = (self.successful_inferences / self.total_inferences 
                       if self.total_inferences > 0 else 0.0)
        
        rule_stats = []
        for rule in self.rules:
            rule_stats.append({
                "name": rule.name,
                "activation_count": rule.activation_count,
                "average_activation": rule.average_activation,
                "weight": rule.weight
            })
        
        return {
            "rule_set_name": self.name,
            "total_rules": len(self.rules),
            "total_inferences": self.total_inferences,
            "successful_inferences": self.successful_inferences,
            "success_rate": success_rate,
            "rule_statistics": rule_stats
        }
    
    def __len__(self) -> int:
        return len(self.rules)

class RuleParser:
    """ファジィルール文字列パーサー"""
    
    # 正規表現パターン
    RULE_PATTERN = re.compile(
        r'IF\s+(.+?)\s+THEN\s+(.+)',
        re.IGNORECASE
    )
    
    CONDITION_PATTERN = re.compile(
        r'(\w+)\s+(IS|IS_NOT|>|<|=)\s+(\w+)',
        re.IGNORECASE
    )
    
    @classmethod
    def parse_rule(cls, rule_string: str, rule_name: str = None) -> Optional[FuzzyRule]:
        """ルール文字列をパースしてFuzzyRuleを作成"""
        
        try:
            # ルール全体のマッチング
            rule_match = cls.RULE_PATTERN.match(rule_string.strip())
            if not rule_match:
                logger.error(f"ルール形式が無効: {rule_string}")
                return None
            
            antecedent_str = rule_match.group(1).strip()
            consequent_str = rule_match.group(2).strip()
            
            # 条件部のパース
            conditions = cls._parse_conditions(antecedent_str)
            if not conditions:
                logger.error(f"条件部の解析に失敗: {antecedent_str}")
                return None
            
            # 結論部のパース
            conclusion = cls._parse_conclusion(consequent_str)
            if not conclusion:
                logger.error(f"結論部の解析に失敗: {consequent_str}")
                return None
            
            # ルール名の生成
            if not rule_name:
                rule_name = f"rule_{len(conditions)}_{hash(rule_string) % 1000}"
            
            return FuzzyRule(rule_name, conditions, conclusion)
            
        except Exception as e:
            logger.error(f"ルール解析エラー: {e}")
            return None
    
    @classmethod
    def _parse_conditions(cls, antecedent_str: str) -> List[Condition]:
        """条件部をパース"""
        
        conditions = []
        
        # ANDで分割（簡易版）
        condition_parts = re.split(r'\s+AND\s+', antecedent_str, flags=re.IGNORECASE)
        
        for part in condition_parts:
            part = part.strip()
            
            # NOT処理
            negated = False
            if part.upper().startswith('NOT '):
                negated = True
                part = part[4:].strip()
            
            # 条件のマッチング
            condition_match = cls.CONDITION_PATTERN.match(part)
            if condition_match:
                variable = condition_match.group(1)
                operator_str = condition_match.group(2).upper()
                linguistic_value = condition_match.group(3)
                
                # 演算子の変換
                try:
                    operator = ComparisonOperator(operator_str)
                except ValueError:
                    operator = ComparisonOperator.IS
                
                condition = Condition(
                    variable=variable,
                    operator=operator,
                    linguistic_value=linguistic_value,
                    negated=negated
                )
                conditions.append(condition)
            else:
                logger.warning(f"条件の解析に失敗: {part}")
        
        return conditions
    
    @classmethod
    def _parse_conclusion(cls, consequent_str: str) -> Optional[Conclusion]:
        """結論部をパース"""
        
        conclusion_match = cls.CONDITION_PATTERN.match(consequent_str)
        if conclusion_match:
            variable = conclusion_match.group(1)
            linguistic_value = conclusion_match.group(3)
            
            return Conclusion(
                variable=variable,
                linguistic_value=linguistic_value
            )
        
        return None

class LabMatchingRuleBuilder:
    """研究室マッチング用ルール生成器"""
    
    @staticmethod
    def create_basic_rules() -> FuzzyRuleSet:
        """基本的な研究室マッチングルールを作成"""
        
        rule_set = FuzzyRuleSet("lab_matching_basic")
        
        # 基本ルール定義
        basic_rule_strings = [
            # 高適合ルール
            "IF research_intensity IS high AND advisor_style IS high THEN compatibility IS high_match",
            "IF research_field_match IS high THEN compatibility IS high_match",
            "IF team_work IS high AND communication_style IS high THEN compatibility IS high_match",
            
            # 中適合ルール
            "IF research_intensity IS medium AND advisor_style IS medium THEN compatibility IS medium_match",
            "IF workload IS medium AND flexibility IS medium THEN compatibility IS medium_match",
            
            # 低適合ルール
            "IF research_intensity IS low AND advisor_style IS high THEN compatibility IS low_match",
            "IF workload IS high AND flexibility IS low THEN compatibility IS low_match",
            
            # 特殊ケース
            "IF theory_practice IS high AND innovation_risk IS high THEN compatibility IS high_match",
            "IF publication_opportunity IS high AND skill_development IS high THEN compatibility IS high_match",
        ]
        
        # ルールを解析して追加
        for i, rule_str in enumerate(basic_rule_strings):
            rule = RuleParser.parse_rule(rule_str, f"basic_rule_{i+1}")
            if rule:
                rule_set.add_rule(rule)
        
        return rule_set
    
    @staticmethod
    def create_advanced_rules() -> FuzzyRuleSet:
        """高度な研究室マッチングルールを作成"""
        
        rule_set = FuzzyRuleSet("lab_matching_advanced")
        
        # 高度ルール定義
        advanced_rule_strings = [
            # 分野別特化ルール
            "IF research_field_match IS high AND interdisciplinary IS low THEN compatibility IS high_match",
            "IF research_field_match IS medium AND interdisciplinary IS high THEN compatibility IS medium_match",
            
            # 指導スタイル適合ルール
            "IF advisor_style IS high AND communication_style IS high AND flexibility IS high THEN compatibility IS high_match",
            "IF advisor_style IS low AND team_work IS low THEN compatibility IS medium_match",
            
            # スキル開発ルール
            "IF skill_development IS high AND publication_opportunity IS high AND innovation_risk IS medium THEN compatibility IS high_match",
            
            # 研究環境ルール
            "IF lab_atmosphere IS high AND workload IS medium AND flexibility IS high THEN compatibility IS high_match",
        ]
        
        for i, rule_str in enumerate(advanced_rule_strings):
            rule = RuleParser.parse_rule(rule_str, f"advanced_rule_{i+1}")
            if rule:
                rule_set.add_rule(rule)
        
        return rule_set

# 使用例とテスト
def test_rule_system():
    """ルールシステムのテスト"""
    
    print("📋 ファジィルールシステムテスト開始")
    
    # 基本ルールセット作成
    basic_rules = LabMatchingRuleBuilder.create_basic_rules()
    print(f"✅ 基本ルール数: {len(basic_rules)}")
    
    # 高度ルールセット作成
    advanced_rules = LabMatchingRuleBuilder.create_advanced_rules()
    print(f"✅ 高度ルール数: {len(advanced_rules)}")
    
    # テスト用メンバーシップ値
    test_memberships = {
        "research_intensity": {"low": 0.1, "medium": 0.3, "high": 0.8},
        "advisor_style": {"low": 0.2, "medium": 0.4, "high": 0.7},
        "team_work": {"low": 0.0, "medium": 0.6, "high": 0.9},
        "research_field_match": {"low": 0.1, "medium": 0.2, "high": 0.9}
    }
    
    # ルール評価テスト
    activations = basic_rules.evaluate_all(test_memberships)
    
    print(f"\n📊 ルール活性化結果:")
    for rule_name, activation in activations.items():
        if activation > 0.1:
            print(f"  {rule_name}: {activation:.3f}")
    
    # 統計情報表示
    stats = basic_rules.get_statistics()
    print(f"\n📈 統計情報:")
    print(f"  成功率: {stats['success_rate']:.3f}")
    print(f"  実行回数: {stats['total_inferences']}")
    
    print("✅ ファジィルールシステムテスト完了")

if __name__ == "__main__":
    test_rule_system()