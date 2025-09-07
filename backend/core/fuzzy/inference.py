# core/fuzzy/inference.py - ファジィ推論エンジン

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

from core.fuzzy.membership import (
    FuzzyVariable, FuzzySet, MembershipFunctionFactory,
    TriangularMF, TrapezoidalMF, GaussianMF
)
from core.fuzzy.rules import FuzzyRule, FuzzyRuleSet, RuleOperator
from models.schemas import StudentProfile, Laboratory, EvaluationCriteria

logger = logging.getLogger(__name__)

@dataclass
class InferenceResult:
    """推論結果"""
    output_value: float
    confidence: float
    activated_rules: List[str]
    rule_activations: Dict[str, float]
    intermediate_values: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "output_value": self.output_value,
            "confidence": self.confidence,
            "activated_rules": self.activated_rules,
            "rule_activations": self.rule_activations,
            "intermediate_values": self.intermediate_values
        }

class InferenceMethod(str):
    """推論手法"""
    MAMDANI = "mamdani"
    SUGENO = "sugeno"
    TSUKAMOTO = "tsukamoto"

class DefuzzificationMethod(str):
    """非ファジィ化手法"""
    CENTROID = "centroid"
    BISECTOR = "bisector"
    MOM = "mean_of_maximum"
    SOM = "smallest_of_maximum"
    LOM = "largest_of_maximum"

class FuzzyInferenceEngine:
    """ファジィ推論エンジン"""
    
    def __init__(self, inference_method: str = InferenceMethod.MAMDANI,
                 defuzzification_method: str = DefuzzificationMethod.CENTROID):
        
        self.inference_method = inference_method
        self.defuzzification_method = defuzzification_method
        
        # ファジィ変数群
        self.input_variables: Dict[str, FuzzyVariable] = {}
        self.output_variables: Dict[str, FuzzyVariable] = {}
        
        # ルール集合
        self.rule_sets: Dict[str, FuzzyRuleSet] = {}
        
        # 設定
        self.confidence_threshold = 0.1
        self.min_activation = 1e-6
        
        # 統計情報
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        # 研究室選択支援用の初期化
        self._initialize_lab_matching_system()
    
    def _initialize_lab_matching_system(self):
        """研究室選択支援システム用の初期化"""
        
        # 入力変数の定義（13の評価基準）
        evaluation_criteria = [
            "research_intensity", "advisor_style", "team_work", "workload", 
            "theory_practice", "research_field_match", "skill_development",
            "lab_atmosphere", "flexibility", "publication_opportunity",
            "interdisciplinary", "communication_style", "innovation_risk"
        ]
        
        for criterion in evaluation_criteria:
            var = MembershipFunctionFactory.create_standard_sets(criterion, (1.0, 10.0))
            self.add_input_variable(var)
        
        # 出力変数の定義（適合性）
        compatibility_var = MembershipFunctionFactory.create_compatibility_variable()
        self.add_output_variable(compatibility_var)
        
        # 基本ルールセットの作成
        self._create_basic_rule_set()
    
    def add_input_variable(self, variable: FuzzyVariable):
        """入力変数を追加"""
        self.input_variables[variable.name] = variable
    
    def add_output_variable(self, variable: FuzzyVariable):
        """出力変数を追加"""
        self.output_variables[variable.name] = variable
    
    def add_rule_set(self, rule_set: FuzzyRuleSet):
        """ルールセットを追加"""
        self.rule_sets[rule_set.name] = rule_set
    
    def _create_basic_rule_set(self):
        """基本ルールセットの作成"""
        
        try:
            from core.fuzzy.rules import FuzzyRuleSet, FuzzyRule, Condition, Conclusion
            
            # 基本適合性ルールセット
            basic_rules = FuzzyRuleSet("basic_compatibility")
            
            # 高適合性ルール群
            high_compatibility_rules = [
                # 研究強度と指導スタイルの組み合わせ
                "IF research_intensity IS high AND advisor_style IS high THEN compatibility IS high_match",
                "IF research_intensity IS medium AND advisor_style IS medium THEN compatibility IS medium_match",
                
                # チームワークとコミュニケーションの組み合わせ
                "IF team_work IS high AND communication_style IS high THEN compatibility IS high_match",
                
                # 理論実践バランス
                "IF theory_practice IS medium THEN compatibility IS medium_match",
                
                # ワークロードとスキル開発
                "IF workload IS high AND skill_development IS high THEN compatibility IS high_match",
                
                # 分野適合性重視
                "IF research_field_match IS high THEN compatibility IS high_match",
            ]
            
            # ルールを文字列からパース（簡易版）
            for rule_str in high_compatibility_rules:
                rule = self._parse_rule_string(rule_str)
                if rule:
                    basic_rules.add_rule(rule)
            
            self.add_rule_set(basic_rules)
            
        except ImportError:
            # rulesモジュールが利用できない場合は後で追加
            logger.warning("ルールモジュールが利用できません。後で追加してください。")
    
    def _parse_rule_string(self, rule_str: str) -> Optional['FuzzyRule']:
        """ルール文字列の簡易パース"""
        
        try:
            # 簡易パーサー（実装を簡略化）
            parts = rule_str.split(" THEN ")
            if len(parts) != 2:
                return None
            
            antecedent_str = parts[0].replace("IF ", "")
            consequent_str = parts[1]
            
            # 実際の実装では適切なパーサーを使用
            # ここでは None を返して後で手動で追加
            return None
            
        except Exception as e:
            logger.warning(f"ルール解析エラー: {rule_str} - {e}")
            return None
    
    def infer(self, inputs: Dict[str, float], 
              output_variable: str = "compatibility") -> InferenceResult:
        """ファジィ推論を実行"""
        
        start_time = time.time()
        
        try:
            # 1. ファジィ化
            fuzzified_inputs = {}
            for var_name, value in inputs.items():
                if var_name in self.input_variables:
                    fuzzified_inputs[var_name] = self.input_variables[var_name].fuzzify(value)
            
            # 2. ルール評価
            rule_activations = {}
            activated_rules = []
            
            if output_variable in self.rule_sets:
                rule_set = self.rule_sets[output_variable]
                
                for rule in rule_set.rules:
                    activation = self._evaluate_rule(rule, fuzzified_inputs)
                    if activation > self.min_activation:
                        rule_activations[rule.name] = activation
                        activated_rules.append(rule.name)
            
            # 3. 含意と統合
            output_memberships = self._aggregate_outputs(
                rule_activations, output_variable
            )
            
            # 4. 非ファジィ化
            if output_variable in self.output_variables:
                output_value = self.output_variables[output_variable].defuzzify(
                    output_memberships, self.defuzzification_method
                )
            else:
                output_value = 0.5  # デフォルト値
            
            # 5. 信頼度計算
            confidence = self._calculate_confidence(
                rule_activations, output_memberships
            )
            
            # 統計更新
            self.inference_count += 1
            self.total_inference_time += time.time() - start_time
            
            return InferenceResult(
                output_value=output_value,
                confidence=confidence,
                activated_rules=activated_rules,
                rule_activations=rule_activations,
                intermediate_values={
                    "fuzzified_inputs": fuzzified_inputs,
                    "output_memberships": output_memberships
                }
            )
            
        except Exception as e:
            logger.error(f"推論エラー: {e}")
            return InferenceResult(
                output_value=0.0,
                confidence=0.0,
                activated_rules=[],
                rule_activations={},
                intermediate_values={}
            )
    
    def _evaluate_rule(self, rule: 'FuzzyRule', 
                      fuzzified_inputs: Dict[str, Dict[str, float]]) -> float:
        """ルールの活性化度を評価"""
        
        # 簡易版：全ての条件の最小値を取る（AND結合）
        activation = 1.0
        
        try:
            for condition in rule.conditions:
                var_name = condition.variable
                set_name = condition.linguistic_value
                
                if var_name in fuzzified_inputs:
                    membership = fuzzified_inputs[var_name].get(set_name, 0.0)
                    activation = min(activation, membership)
                else:
                    activation = 0.0
                    break
            
            return activation
            
        except Exception:
            return 0.0
    
    def _aggregate_outputs(self, rule_activations: Dict[str, float],
                          output_variable: str) -> Dict[str, float]:
        """出力の統合"""
        
        output_memberships = {}
        
        try:
            if output_variable in self.rule_sets:
                rule_set = self.rule_sets[output_variable]
                
                for rule in rule_set.rules:
                    if rule.name in rule_activations:
                        activation = rule_activations[rule.name]
                        conclusion_set = rule.conclusion.linguistic_value
                        
                        # 最大値結合
                        current_membership = output_memberships.get(conclusion_set, 0.0)
                        output_memberships[conclusion_set] = max(current_membership, activation)
            
        except Exception as e:
            logger.warning(f"出力統合エラー: {e}")
        
        return output_memberships
    
    def _calculate_confidence(self, rule_activations: Dict[str, float],
                             output_memberships: Dict[str, float]) -> float:
        """信頼度を計算"""
        
        if not rule_activations:
            return 0.0
        
        # 活性化されたルール数と活性化度から信頼度を計算
        avg_activation = sum(rule_activations.values()) / len(rule_activations)
        rule_coverage = len(rule_activations) / max(len(self.rule_sets), 1)
        
        return min(avg_activation * rule_coverage, 1.0)
    
    def infer_lab_compatibility(self, student_profile: StudentProfile,
                               laboratory: Laboratory) -> InferenceResult:
        """研究室適合性の推論"""
        
        # 学生と研究室の特性値を入力として準備
        inputs = {}
        
        # 学生の評価基準
        student_criteria = student_profile.evaluation_criteria.dict()
        lab_criteria = laboratory.characteristics.dict()
        
        # 適合性計算のため、学生と研究室の特性の差分や組み合わせを考慮
        for criterion, student_value in student_criteria.items():
            if student_value is not None:
                lab_value = lab_criteria.get(criterion, 5.0)  # デフォルト値
                
                if lab_value is not None:
                    # 差分ベースの適合性
                    diff = abs(student_value - lab_value)
                    compatibility_score = max(0, 10 - diff * 2)  # 差が小さいほど高得点
                    inputs[criterion] = compatibility_score
                else:
                    inputs[criterion] = student_value
        
        # 分野適合性の特別処理
        field_match_score = self._calculate_field_match(
            student_profile, laboratory
        )
        inputs["research_field_match"] = field_match_score
        
        # ファジィ推論実行
        return self.infer(inputs, "compatibility")
    
    def _calculate_field_match(self, student_profile: StudentProfile,
                              laboratory: Laboratory) -> float:
        """研究分野適合性を計算"""
        
        lab_field = laboratory.research_field.value
        
        # 学生の興味分野から適合度を計算
        for interest in student_profile.field_interests:
            if interest.field.value == lab_field:
                return interest.interest_level
        
        # 該当なしの場合は低めの値
        return 3.0

class SimpleFuzzyInferenceEngine:
    """簡易ファジィ推論エンジン（フォールバック用）"""
    
    def __init__(self):
        self.inference_count = 0
    
    def infer_lab_compatibility(self, student_profile: StudentProfile,
                               laboratory: Laboratory) -> InferenceResult:
        """簡易適合性推論"""
        
        # 基本5項目の重み付き平均を計算
        basic_criteria = [
            "research_intensity", "advisor_style", "team_work", 
            "workload", "theory_practice"
        ]
        
        total_score = 0.0
        total_weight = 0.0
        matched_criteria = 0
        
        student_criteria = student_profile.evaluation_criteria.dict()
        lab_criteria = laboratory.characteristics.dict()
        
        for criterion in basic_criteria:
            student_val = student_criteria.get(criterion)
            lab_val = lab_criteria.get(criterion)
            
            if student_val is not None and lab_val is not None:
                # 差分ベースの適合性計算
                diff = abs(student_val - lab_val)
                compatibility = max(0, 1 - diff / 9.0)  # 0-1の範囲に正規化
                
                weight = 1.0  # 基本重み
                total_score += compatibility * weight
                total_weight += weight
                matched_criteria += 1
        
        # 分野適合性ボーナス
        field_bonus = self._simple_field_match(student_profile, laboratory)
        total_score += field_bonus * 0.5
        total_weight += 0.5
        
        # 最終スコア計算
        final_score = total_score / total_weight if total_weight > 0 else 0.0
        confidence = matched_criteria / len(basic_criteria)
        
        self.inference_count += 1
        
        return InferenceResult(
            output_value=final_score,
            confidence=confidence,
            activated_rules=[f"simple_rule_{i}" for i in range(matched_criteria)],
            rule_activations={f"rule_{i}": final_score for i in range(matched_criteria)},
            intermediate_values={
                "matched_criteria": matched_criteria,
                "field_bonus": field_bonus
            }
        )
    
    def _simple_field_match(self, student_profile: StudentProfile,
                           laboratory: Laboratory) -> float:
        """簡易分野適合性"""
        
        lab_field = laboratory.research_field.value
        
        for interest in student_profile.field_interests:
            if interest.field.value == lab_field:
                return interest.interest_level / 10.0  # 0-1に正規化
        
        return 0.2  # デフォルト低値


# 使用例とテスト
def test_inference_engine():
    """推論エンジンのテスト"""
    
    print("🧠 ファジィ推論エンジンテスト開始")
    
    # エンジンの初期化
    engine = SimpleFuzzyInferenceEngine()
    
    # テスト用データ
    from models.schemas import (
        StudentProfile, EvaluationCriteria, FieldInterest, 
        Laboratory, Faculty, ResearchFieldEnum
    )
    
    # テスト学生プロフィール
    test_student = StudentProfile(
        student_id="test_001",
        evaluation_criteria=EvaluationCriteria(
            research_intensity=8.0,
            advisor_style=7.0,
            team_work=6.0,
            workload=7.0,
            theory_practice=8.0
        ),
        field_interests=[
            FieldInterest(
                field=ResearchFieldEnum.AI_MACHINE_LEARNING,
                interest_level=9.0,
                priority=1
            )
        ]
    )
    
    # テスト研究室
    test_lab = Laboratory(
        lab_id="lab_001",
        faculty=Faculty(
            name="テスト教授",
            specialties=["機械学習", "データ解析"]
        ),
        research_field=ResearchFieldEnum.AI_MACHINE_LEARNING,
        characteristics=EvaluationCriteria(
            research_intensity=8.5,
            advisor_style=7.5,
            team_work=6.5,
            workload=7.0,
            theory_practice=8.0
        )
    )
    
    # 推論実行
    result = engine.infer_lab_compatibility(test_student, test_lab)
    
    print(f"📊 推論結果:")
    print(f"  適合性スコア: {result.output_value:.3f}")
    print(f"  信頼度: {result.confidence:.3f}")
    print(f"  活性化ルール数: {len(result.activated_rules)}")
    
    print("✅ ファジィ推論エンジンテスト完了")

if __name__ == "__main__":
    import time
    test_inference_engine()