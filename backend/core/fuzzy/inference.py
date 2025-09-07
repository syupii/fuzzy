# core/fuzzy/inference.py - ファジィ推論エンジン

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

from core.fuzzy.membership import MembershipFunction
from core.fuzzy.rules import FuzzyRuleBase, FuzzyRule, RuleOperator
from models.schemas import StudentProfile, Laboratory

@dataclass
class InferenceResult:
    """推論結果"""
    variable: str                    # 推論変数名
    crisp_value: float              # 明確化された値
    fuzzy_values: Dict[str, float]  # 各言語値のメンバーシップ度
    activated_rules: List[str]      # 発火したルールID
    confidence: float               # 推論の信頼度

class FuzzyInferenceEngine:
    """ファジィ推論エンジン"""
    
    def __init__(self):
        self.membership_func = MembershipFunction()
        self.rule_base = FuzzyRuleBase()
        self.inference_history: List[InferenceResult] = []
    
    def infer(self, input_data: Dict[str, float], 
              target_variable: str) -> InferenceResult:
        """
        ファジィ推論を実行
        
        Args:
            input_data: 入力データ {変数名: 値}
            target_variable: 推論対象変数
            
        Returns:
            InferenceResult: 推論結果
        """
        
        # 1. ファジィ化（入力値を言語値に変換）
        fuzzified_inputs = self._fuzzify_inputs(input_data)
        
        # 2. 適用可能なルールを取得
        applicable_rules = self._get_applicable_rules(target_variable, input_data)
        
        # 3. ルール評価（前件部の評価）
        rule_activations = self._evaluate_rules(applicable_rules, fuzzified_inputs)
        
        # 4. 推論実行（後件部の計算）
        fuzzy_output = self._execute_inference(rule_activations)
        
        # 5. 非ファジィ化（明確化）
        crisp_value = self._defuzzify(fuzzy_output)
        
        # 6. 信頼度計算
        confidence = self._calculate_confidence(rule_activations, applicable_rules)
        
        # 推論結果作成
        result = InferenceResult(
            variable=target_variable,
            crisp_value=crisp_value,
            fuzzy_values=fuzzy_output,
            activated_rules=[rule.rule_id for rule, _ in rule_activations if _ > 0],
            confidence=confidence
        )
        
        self.inference_history.append(result)
        return result
    
    def _fuzzify_inputs(self, input_data: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """入力値をファジィ化"""
        
        fuzzified = {}
        
        for variable, value in input_data.items():
            fuzzified[variable] = {}
            
            # 各言語値に対するメンバーシップ度を計算
            for linguistic_value in ["very_low", "low", "medium", "high", "very_high"]:
                membership = self.membership_func.evaluate(value, linguistic_value)
                fuzzified[variable][linguistic_value] = membership
        
        return fuzzified
    
    def _get_applicable_rules(self, target_variable: str, 
                            input_data: Dict[str, float]) -> List[FuzzyRule]:
        """対象変数に適用可能なルールを取得"""
        
        # 対象変数を後件部に持つルールを取得
        target_rules = self.rule_base.get_rules_by_consequent(target_variable)
        
        # 入力データに必要な変数を持つルールのみを選択
        applicable_rules = []
        
        for rule in target_rules:
            has_all_inputs = all(
                condition.variable in input_data
                for condition in rule.antecedent
            )
            
            if has_all_inputs:
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def _evaluate_rules(self, rules: List[FuzzyRule], 
                       fuzzified_inputs: Dict[str, Dict[str, float]]) -> List[Tuple[FuzzyRule, float]]:
        """ルールの前件部を評価"""
        
        rule_activations = []
        
        for rule in rules:
            activation = self._evaluate_antecedent(rule, fuzzified_inputs)
            rule_activations.append((rule, activation))
        
        return rule_activations
    
    def _evaluate_antecedent(self, rule: FuzzyRule, 
                           fuzzified_inputs: Dict[str, Dict[str, float]]) -> float:
        """ルールの前件部を評価"""
        
        condition_values = []
        
        # 各条件のメンバーシップ度を取得
        for condition in rule.antecedent:
            variable = condition.variable
            linguistic_value = condition.linguistic_value
            weight = condition.weight
            
            if variable in fuzzified_inputs and linguistic_value in fuzzified_inputs[variable]:
                membership = fuzzified_inputs[variable][linguistic_value]
                weighted_membership = membership * weight
                condition_values.append(weighted_membership)
        
        if not condition_values:
            return 0.0
        
        # 演算子に基づいて結合
        if rule.operator == RuleOperator.AND:
            # AND演算：最小値
            activation = min(condition_values)
        elif rule.operator == RuleOperator.OR:
            # OR演算：最大値
            activation = max(condition_values)
        else:
            # デフォルトはAND
            activation = min(condition_values)
        
        # ルールの信頼度を適用
        return activation * rule.confidence
    
    def _execute_inference(self, rule_activations: List[Tuple[FuzzyRule, float]]) -> Dict[str, float]:
        """推論を実行"""
        
        fuzzy_output = {
            "very_low": 0.0,
            "low": 0.0,
            "medium": 0.0,
            "high": 0.0,
            "very_high": 0.0
        }
        
        # 各ルールの後件部を適用
        for rule, activation in rule_activations:
            if activation > 0:
                consequent_value = rule.consequent.linguistic_value
                
                # 最大値合成（Max-Min合成）
                if consequent_value in fuzzy_output:
                    fuzzy_output[consequent_value] = max(
                        fuzzy_output[consequent_value], 
                        activation
                    )
        
        return fuzzy_output
    
    def _defuzzify(self, fuzzy_output: Dict[str, float]) -> float:
        """重心法による非ファジィ化"""
        
        # 言語値を数値に対応付け
        linguistic_to_numeric = {
            "very_low": 1.0,
            "low": 3.0,
            "medium": 5.0,
            "high": 7.0,
            "very_high": 9.0
        }
        
        numerator = 0.0
        denominator = 0.0
        
        for linguistic_value, membership in fuzzy_output.items():
            if membership > 0:
                numeric_value = linguistic_to_numeric[linguistic_value]
                numerator += numeric_value * membership
                denominator += membership
        
        if denominator > 0:
            crisp_value = numerator / denominator
        else:
            crisp_value = 5.0  # デフォルト値（中間値）
        
        return crisp_value
    
    def _calculate_confidence(self, rule_activations: List[Tuple[FuzzyRule, float]], 
                            applicable_rules: List[FuzzyRule]) -> float:
        """推論の信頼度を計算"""
        
        if not applicable_rules:
            return 0.0
        
        # 発火したルールの重み付き平均
        total_activation = 0.0
        total_weight = 0.0
        
        for rule, activation in rule_activations:
            if activation > 0:
                total_activation += activation * rule.confidence
                total_weight += rule.confidence
        
        if total_weight > 0:
            confidence = total_activation / total_weight
        else:
            confidence = 0.0
        
        # ルール適用率も考慮
        activation_rate = len([a for _, a in rule_activations if a > 0]) / len(applicable_rules)
        
        # 最終信頼度
        final_confidence = confidence * 0.8 + activation_rate * 0.2
        
        return min(1.0, max(0.0, final_confidence))
    
    def infer_lab_compatibility(self, student: StudentProfile, 
                              lab: Laboratory) -> Dict[str, InferenceResult]:
        """研究室適合性を総合的に推論"""
        
        # 入力データ準備
        input_data = self._prepare_lab_input_data(student, lab)
        
        # 各側面を段階的に推論
        results = {}
        
        # 1. 分野適合性
        field_compatibility = self.infer(input_data, "field_compatibility")
        results["field_compatibility"] = field_compatibility
        
        # 2. 経験マッチング
        experience_match = self.infer(input_data, "experience_match")
        results["experience_match"] = experience_match
        
        # 3. 研究スタイルマッチング
        research_style_match = self.infer(input_data, "research_style_match")
        results["research_style_match"] = research_style_match
        
        # 4. 指導スタイルマッチング
        advisor_match = self.infer(input_data, "advisor_match")
        results["advisor_match"] = advisor_match
        
        # 中間結果を入力データに追加
        input_data.update({
            "field_compatibility": field_compatibility.crisp_value,
            "experience_match": experience_match.crisp_value,
            "research_style_match": research_style_match.crisp_value,
            "advisor_match": advisor_match.crisp_value
        })
        
        # 5. 総合適合性
        overall_compatibility = self.infer(input_data, "overall_compatibility")
        results["overall_compatibility"] = overall_compatibility
        
        return results
    
    def _prepare_lab_input_data(self, student: StudentProfile, 
                              lab: Laboratory) -> Dict[str, float]:
        """研究室マッチング用の入力データを準備"""
        
        input_data = {}
        
        # 学生の評価基準
        student_criteria = student.evaluation_criteria.dict()
        for key, value in student_criteria.items():
            input_data[f"student_{key}"] = float(value)
        
        # 研究室特徴
        lab_features = lab.features.dict()
        for key, value in lab_features.items():
            input_data[f"lab_{key}"] = float(value)
        
        # 分野興味度（選択した分野の平均）
        student_fields = {fi.field_id: fi for fi in student.field_interests}
        
        field_interest_scores = []
        field_experience_scores = []
        field_importance_scores = []
        
        for field_id in lab.research_fields:
            if field_id in student_fields:
                field_interest = student_fields[field_id]
                field_interest_scores.append(field_interest.interest_level)
                field_experience_scores.append(field_interest.experience_level)
                field_importance_scores.append(field_interest.importance_level)
        
        if field_interest_scores:
            input_data["interest_level"] = np.mean(field_interest_scores)
            input_data["experience_level"] = np.mean(field_experience_scores)
            input_data["importance_level"] = np.mean(field_importance_scores)
        else:
            input_data["interest_level"] = 0.0
            input_data["experience_level"] = 0.0
            input_data["importance_level"] = 0.0
        
        # 分野難易度（研究室の分野の平均難易度）
        from config.settings import settings
        
        difficulty_scores = []
        for field_id in lab.research_fields:
            field_info = settings.research_fields.get(field_id, {})
            difficulty = field_info.get("difficulty", "intermediate")
            
            difficulty_map = {"beginner": 3, "intermediate": 6, "advanced": 9}
            difficulty_scores.append(difficulty_map[difficulty])
        
        if difficulty_scores:
            input_data["field_difficulty"] = np.mean(difficulty_scores)
        else:
            input_data["field_difficulty"] = 6.0  # デフォルト：中級
        
        return input_data
    
    def explain_inference(self, result: InferenceResult) -> str:
        """推論過程の説明を生成"""
        
        explanation = f"📊 {result.variable}の推論結果\n"
        explanation += f"   明確化値: {result.crisp_value:.2f}\n"
        explanation += f"   信頼度: {result.confidence:.3f}\n\n"
        
        explanation += "🔥 発火したルール:\n"
        for rule_id in result.activated_rules:
            rule = self.rule_base.get_rule(rule_id)
            if rule:
                explanation += f"   - {rule_id}: {rule.description}\n"
        
        explanation += "\n🎯 ファジィ値:\n"
        for linguistic_value, membership in result.fuzzy_values.items():
            if membership > 0:
                explanation += f"   - {linguistic_value}: {membership:.3f}\n"
        
        return explanation
    
    def get_inference_summary(self) -> Dict[str, Any]:
        """推論の統計サマリーを取得"""
        
        if not self.inference_history:
            return {"message": "推論履歴がありません"}
        
        summary = {
            "total_inferences": len(self.inference_history),
            "variables_inferred": list(set(r.variable for r in self.inference_history)),
            "average_confidence": np.mean([r.confidence for r in self.inference_history]),
            "rule_usage": {}
        }
        
        # ルール使用統計
        all_activated_rules = []
        for result in self.inference_history:
            all_activated_rules.extend(result.activated_rules)
        
        from collections import Counter
        rule_counts = Counter(all_activated_rules)
        summary["rule_usage"] = dict(rule_counts.most_common(10))
        
        return summary
    
    def clear_history(self):
        """推論履歴をクリア"""
        self.inference_history.clear()
    
    def validate_inference_system(self) -> List[str]:
        """推論システムの妥当性チェック"""
        
        issues = []
        
        # ルールベースの検証
        rule_issues = self.rule_base.validate_rules()
        issues.extend(rule_issues)
        
        # メンバーシップ関数の検証
        test_values = [0, 2.5, 5, 7.5, 10]
        
        for value in test_values:
            for linguistic_value in ["very_low", "low", "medium", "high", "very_high"]:
                try:
                    membership = self.membership_func.evaluate(value, linguistic_value)
                    if membership < 0 or membership > 1:
                        issues.append(f"メンバーシップ値が範囲外: {linguistic_value}({value}) = {membership}")
                except Exception as e:
                    issues.append(f"メンバーシップ関数エラー: {str(e)}")
        
        return issues