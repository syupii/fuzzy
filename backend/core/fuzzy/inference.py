# core/fuzzy/inference.py - 完全13項目対応 ファジィ推論システム

import math
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)

class MembershipFunctionType(str, Enum):
    """メンバーシップ関数タイプ"""
    TRIANGULAR = "triangular"
    TRAPEZOIDAL = "trapezoidal"
    GAUSSIAN = "gaussian"
    SIGMOID = "sigmoid"

class InferenceMethod(str, Enum):
    """推論手法"""
    MAMDANI = "mamdani"
    SUGENO = "sugeno"
    TSUKAMOTO = "tsukamoto"

class DefuzzificationMethod(str, Enum):
    """非ファジィ化手法"""
    CENTROID = "centroid"
    BISECTOR = "bisector"
    MOM = "mean_of_maximum"
    SOM = "smallest_of_maximum"
    LOM = "largest_of_maximum"

@dataclass
class FuzzySet:
    """ファジィ集合"""
    
    name: str
    membership_function: Callable[[float], float]
    parameters: List[float]
    function_type: MembershipFunctionType
    
    def membership_degree(self, x: float) -> float:
        """メンバーシップ度計算"""
        try:
            return max(0.0, min(1.0, self.membership_function(x)))
        except Exception as e:
            logger.warning(f"メンバーシップ度計算エラー ({self.name}): {e}")
            return 0.0
    
    def __str__(self) -> str:
        return f"FuzzySet({self.name}, {self.function_type.value}, {self.parameters})"

@dataclass
class FuzzyVariable:
    """ファジィ変数（13項目対応）"""
    
    name: str
    range: Tuple[float, float] = (1.0, 10.0)  # 評価基準の範囲
    fuzzy_sets: Dict[str, FuzzySet] = field(default_factory=dict)
    importance_weight: float = 1.0
    
    def add_fuzzy_set(self, linguistic_value: str, fuzzy_set: FuzzySet):
        """ファジィ集合追加"""
        self.fuzzy_sets[linguistic_value] = fuzzy_set
    
    def fuzzify(self, crisp_value: float) -> Dict[str, float]:
        """ファジィ化"""
        if not self.range[0] <= crisp_value <= self.range[1]:
            logger.warning(f"値が範囲外: {crisp_value} not in {self.range}")
        
        membership_degrees = {}
        for linguistic_value, fuzzy_set in self.fuzzy_sets.items():
            degree = fuzzy_set.membership_degree(crisp_value)
            membership_degrees[linguistic_value] = degree
        
        return membership_degrees
    
    def get_linguistic_values(self) -> List[str]:
        """言語値一覧取得"""
        return list(self.fuzzy_sets.keys())

@dataclass
class FuzzyRule:
    """ファジィルール（13項目対応）"""
    
    rule_id: int
    antecedents: Dict[str, str]  # criterion_name -> linguistic_value
    consequent: str  # 結論の言語値
    confidence: float = 1.0
    rule_weight: float = 1.0
    
    def evaluate(self, fuzzified_inputs: Dict[str, Dict[str, float]]) -> float:
        """ルール評価（前件部の適合度計算）"""
        
        antecedent_degrees = []
        
        for criterion, linguistic_value in self.antecedents.items():
            if criterion in fuzzified_inputs:
                degree = fuzzified_inputs[criterion].get(linguistic_value, 0.0)
                antecedent_degrees.append(degree)
            else:
                # データ不足の場合はニュートラル値
                antecedent_degrees.append(0.5)
        
        # 前件部の結合（最小値演算）
        if antecedent_degrees:
            rule_strength = min(antecedent_degrees) * self.confidence * self.rule_weight
        else:
            rule_strength = 0.0
        
        return rule_strength
    
    def __str__(self) -> str:
        antecedent_str = " AND ".join([f"{k} is {v}" for k, v in self.antecedents.items()])
        return f"Rule {self.rule_id}: IF {antecedent_str} THEN compatibility is {self.consequent}"

class Complete13CriteriaFuzzyEngine:
    """完全13項目対応ファジィ推論エンジン"""
    
    # 13項目評価基準
    CRITERIA_NAMES = [
        "research_intensity", "advisor_style", "team_work", "workload", "theory_practice",
        "research_field_match", "skill_development", "lab_atmosphere", "flexibility",
        "publication_opportunity", "interdisciplinary", "communication_style", "innovation_risk"
    ]
    
    # 基準別重要度重み
    CRITERIA_WEIGHTS = {
        # 基本項目（高重要度）
        "research_intensity": 1.3,
        "advisor_style": 1.2,
        "team_work": 1.1, 
        "workload": 1.1,
        "theory_practice": 1.2,
        
        # 拡張項目（中〜高重要度）
        "research_field_match": 1.5,  # 最重要
        "skill_development": 1.0,
        "lab_atmosphere": 0.9,
        "flexibility": 0.9,
        "publication_opportunity": 1.1,
        
        # 特殊項目（調整重要度）
        "interdisciplinary": 0.8,
        "communication_style": 0.9,
    }
    
    def __init__(self, 
                 inference_method: InferenceMethod = InferenceMethod.MAMDANI,
                 defuzzification_method: DefuzzificationMethod = DefuzzificationMethod.CENTROID):
        
        self.inference_method = inference_method
        self.defuzzification_method = defuzzification_method
        
        # ファジィ変数（13項目分）
        self.input_variables: Dict[str, FuzzyVariable] = {}
        self.output_variable: Optional[FuzzyVariable] = None
        
        # ファジィルール
        self.rules: List[FuzzyRule] = []
        
        # 推論結果キャッシュ
        self.inference_cache: Dict[str, Dict[str, Any]] = {}
        
        # 統計情報
        self.stats = {
            "total_inferences": 0,
            "cache_hits": 0,
            "rule_activations": {},
            "average_inference_time": 0.0
        }
        
        # システム初期化
        self._initialize_13_criteria_system()
        
        logger.info(f"完全13項目対応ファジィ推論エンジン初期化完了 (手法: {inference_method.value})")
    
    def _initialize_13_criteria_system(self):
        """13項目対応システム初期化"""
        
        # 入力変数初期化（13項目）
        for criterion in self.CRITERIA_NAMES:
            weight = self.CRITERIA_WEIGHTS.get(criterion, 1.0)
            self.input_variables[criterion] = self._create_input_variable(criterion, weight)
        
        # 出力変数初期化（適合性）
        self.output_variable = self._create_output_variable()
        
        # ファジィルール生成
        self._generate_complete_rule_base()
        
        logger.info(f"システム初期化完了: {len(self.input_variables)}入力変数, {len(self.rules)}ルール")
    
    def _create_input_variable(self, criterion_name: str, importance_weight: float) -> FuzzyVariable:
        """入力変数作成（各評価基準用）"""
        
        variable = FuzzyVariable(
            name=criterion_name,
            range=(1.0, 10.0),
            importance_weight=importance_weight
        )
        
        # 三角型メンバーシップ関数による言語値定義
        linguistic_values = ["low", "medium", "high"]
        
        # Low (1-4)
        low_mf = self._create_triangular_mf([1, 1, 4])
        variable.add_fuzzy_set("low", FuzzySet(
            name=f"{criterion_name}_low",
            membership_function=low_mf,
            parameters=[1, 1, 4],
            function_type=MembershipFunctionType.TRIANGULAR
        ))
        
        # Medium (3-7)
        medium_mf = self._create_triangular_mf([3, 5.5, 8])
        variable.add_fuzzy_set("medium", FuzzySet(
            name=f"{criterion_name}_medium", 
            membership_function=medium_mf,
            parameters=[3, 5.5, 8],
            function_type=MembershipFunctionType.TRIANGULAR
        ))
        
        # High (6-10)
        high_mf = self._create_triangular_mf([6, 10, 10])
        variable.add_fuzzy_set("high", FuzzySet(
            name=f"{criterion_name}_high",
            membership_function=high_mf,
            parameters=[6, 10, 10],
            function_type=MembershipFunctionType.TRIANGULAR
        ))
        
        return variable
    
    def _create_output_variable(self) -> FuzzyVariable:
        """出力変数作成（適合性用）"""
        
        variable = FuzzyVariable(
            name="compatibility",
            range=(0.0, 1.0)
        )
        
        # 5段階の適合性レベル
        levels = [
            ("very_low", [0.0, 0.0, 0.2]),
            ("low", [0.1, 0.25, 0.4]),
            ("medium", [0.3, 0.5, 0.7]),
            ("high", [0.6, 0.75, 0.9]),
            ("very_high", [0.8, 1.0, 1.0])
        ]
        
        for level_name, params in levels:
            mf = self._create_triangular_mf(params)
            variable.add_fuzzy_set(level_name, FuzzySet(
                name=f"compatibility_{level_name}",
                membership_function=mf,
                parameters=params,
                function_type=MembershipFunctionType.TRIANGULAR
            ))
        
        return variable
    
    def _create_triangular_mf(self, params: List[float]) -> Callable[[float], float]:
        """三角型メンバーシップ関数作成"""
        
        a, b, c = params
        
        def triangular_membership(x: float) -> float:
            if x <= a or x >= c:
                return 0.0
            elif a < x <= b:
                return (x - a) / (b - a) if b != a else 1.0
            else:  # b < x < c
                return (c - x) / (c - b) if c != b else 1.0
        
        return triangular_membership
    
    def _generate_complete_rule_base(self):
        """完全ルールベース生成（13項目対応）"""
        
        # 基本的な適合性パターンルール
        self._add_high_compatibility_rules()
        self._add_medium_compatibility_rules() 
        self._add_low_compatibility_rules()
        
        # 特殊パターンルール
        self._add_special_pattern_rules()
        
        # 分野特化ルール
        self._add_field_specific_rules()
        
        logger.info(f"ルールベース生成完了: {len(self.rules)}ルール")
    
    def _add_high_compatibility_rules(self):
        """高適合性ルール追加"""
        
        rule_id = 1
        
        # 基本項目が全て高い場合
        self.rules.append(FuzzyRule(
            rule_id=rule_id,
            antecedents={
                "research_intensity": "high",
                "advisor_style": "high", 
                "team_work": "high",
                "workload": "high",
                "theory_practice": "high"
            },
            consequent="very_high",
            confidence=0.95,
            rule_weight=1.2
        ))
        rule_id += 1
        
        # 研究分野適合性が非常に重要
        self.rules.append(FuzzyRule(
            rule_id=rule_id,
            antecedents={
                "research_field_match": "high",
                "research_intensity": "high",
                "publication_opportunity": "high"
            },
            consequent="very_high", 
            confidence=0.9,
            rule_weight=1.4
        ))
        rule_id += 1
        
        # バランス型高適合
        self.rules.append(FuzzyRule(
            rule_id=rule_id,
            antecedents={
                "research_field_match": "high",
                "skill_development": "high",
                "lab_atmosphere": "high",
                "flexibility": "medium"
            },
            consequent="high",
            confidence=0.85,
            rule_weight=1.1
        ))
        rule_id += 1
        
        self._next_rule_id = rule_id
    
    def _add_medium_compatibility_rules(self):
        """中程度適合性ルール追加"""
        
        rule_id = getattr(self, '_next_rule_id', 10)
        
        # 基本項目が中程度
        self.rules.append(FuzzyRule(
            rule_id=rule_id,
            antecedents={
                "research_intensity": "medium",
                "advisor_style": "medium",
                "team_work": "medium",
                "workload": "medium"
            },
            consequent="medium",
            confidence=0.8,
            rule_weight=1.0
        ))
        rule_id += 1
        
        # 分野適合は高いが他が中程度
        self.rules.append(FuzzyRule(
            rule_id=rule_id,
            antecedents={
                "research_field_match": "high",
                "research_intensity": "medium",
                "advisor_style": "medium"
            },
            consequent="high",
            confidence=0.75,
            rule_weight=1.2
        ))
        rule_id += 1
        
        # コミュニケーション重視パターン
        self.rules.append(FuzzyRule(
            rule_id=rule_id,
            antecedents={
                "communication_style": "high",
                "team_work": "high",
                "lab_atmosphere": "high"
            },
            consequent="high",
            confidence=0.7,
            rule_weight=0.9
        ))
        rule_id += 1
        
        self._next_rule_id = rule_id
    
    def _add_low_compatibility_rules(self):
        """低適合性ルール追加"""
        
        rule_id = getattr(self, '_next_rule_id', 20)
        
        # 基本項目が低い場合
        self.rules.append(FuzzyRule(
            rule_id=rule_id,
            antecedents={
                "research_intensity": "low",
                "advisor_style": "low",
                "workload": "low"
            },
            consequent="low",
            confidence=0.85,
            rule_weight=1.1
        ))
        rule_id += 1
        
        # 分野不適合
        self.rules.append(FuzzyRule(
            rule_id=rule_id,
            antecedents={
                "research_field_match": "low"
            },
            consequent="low",
            confidence=0.9,
            rule_weight=1.3
        ))
        rule_id += 1
        
        # 理論・実践の大きなミスマッチ
        self.rules.append(FuzzyRule(
            rule_id=rule_id,
            antecedents={
                "theory_practice": "low",
                "skill_development": "low"
            },
            consequent="very_low",
            confidence=0.8,
            rule_weight=1.0
        ))
        rule_id += 1
        
        self._next_rule_id = rule_id
    
    def _add_special_pattern_rules(self):
        """特殊パターンルール追加"""
        
        rule_id = getattr(self, '_next_rule_id', 30)


        
        rule_id = getattr(self, '_next_rule_id', 30)
    
        # 挑戦的研究パターン（革新性の代替）
        self.rules.append(FuzzyRule(
            rule_id=rule_id,
            antecedents={
                "research_intensity": "high",
                "interdisciplinary": "high",
                "publication_opportunity": "high"
            },
            consequent="high", 
            confidence=0.8,
            rule_weight=1.0
        ))
        rule_id += 1
    
        # 安定志向パターン（安定性の代替）
        self.rules.append(FuzzyRule(
            rule_id=rule_id,
            antecedents={
                "workload": "medium",
                "flexibility": "medium",
                "advisor_style": "low"  # 指導重視
            },
            consequent="medium",
            confidence=0.7,
            rule_weight=0.9
        ))
        rule_id += 1
    
        self._next_rule_id = rule_id
        
        # 学際性重視パターン
        self.rules.append(FuzzyRule(
            rule_id=rule_id,
            antecedents={
                "interdisciplinary": "high",
                "communication_style": "high",
                "skill_development": "high"
            },
            consequent="high",
            confidence=0.8,
            rule_weight=1.1
        ))
        rule_id += 1
        
        self._next_rule_id = rule_id
    
    def _add_field_specific_rules(self):
        """分野特化ルール追加"""
        
        rule_id = getattr(self, '_next_rule_id', 40)
        
        # 理論研究重視（数理・アルゴリズム系）
        self.rules.append(FuzzyRule(
            rule_id=rule_id,
            antecedents={
                "theory_practice": "low",  # 理論重視
                "research_intensity": "high",
                "publication_opportunity": "high"
            },
            consequent="high",
            confidence=0.8,
            rule_weight=1.0
        ))
        rule_id += 1
        
        # 実践研究重視（工学・システム系）
        self.rules.append(FuzzyRule(
            rule_id=rule_id,
            antecedents={
                "theory_practice": "high",  # 実践重視
                "skill_development": "high",
                "team_work": "high"
            },
            consequent="high",
            confidence=0.8,
            rule_weight=1.0
        ))
        rule_id += 1
        
        # クリエイティブ系
       self.rules.append(FuzzyRule(
            rule_id=rule_id,
            antecedents={
                "flexibility": "high",
                "lab_atmosphere": "high",
                "skill_development": "high"  # 創造性スキル重視
            },
            consequent="high",
            confidence=0.75,
            rule_weight=1.0
        ))
        
        self._next_rule_id = rule_id
    
    def infer_lab_compatibility(self, student_profile: Dict[str, Any], 
                              lab_profile: Dict[str, Any]) -> Dict[str, Any]:
        """研究室適合性推論（13項目完全対応）"""
        
        import time
        start_time = time.time()
        
        # キャッシュキー生成
        cache_key = self._generate_cache_key(student_profile, lab_profile)
        
        if cache_key in self.inference_cache:
            self.stats["cache_hits"] += 1
            return self.inference_cache[cache_key]
        
        try:
            # 1. ファジィ化（13項目）
            fuzzified_inputs = self._fuzzify_inputs(student_profile, lab_profile)
            
            # 2. ルール評価
            rule_activations = self._evaluate_rules(fuzzified_inputs)
            
            # 3. 推論実行
            if self.inference_method == InferenceMethod.MAMDANI:
                inference_result = self._mamdani_inference(rule_activations)
            elif self.inference_method == InferenceMethod.SUGENO:
                inference_result = self._sugeno_inference(rule_activations, fuzzified_inputs)
            else:  # TSUKAMOTO
                inference_result = self._tsukamoto_inference(rule_activations)
            
            # 4. 非ファジィ化
            crisp_output = self._defuzzify(inference_result)
            
            # 5. 信頼度計算
            confidence = self._calculate_inference_confidence(
                fuzzified_inputs, rule_activations, crisp_output
            )
            
            # 6. 詳細分析
            detailed_analysis = self._generate_detailed_analysis(
                student_profile, lab_profile, fuzzified_inputs, rule_activations
            )
            
            processing_time = time.time() - start_time
            
            # 結果構築
            result = {
                "output_value": crisp_output,
                "confidence": confidence,
                "processing_time": processing_time,
                "method": self.inference_method.value,
                "rule_activations": len([r for r in rule_activations if r["strength"] > 0]),
                "data_completeness": self._calculate_data_completeness(student_profile, lab_profile),
                "detailed_analysis": detailed_analysis,
                "fuzzified_inputs": self._format_fuzzified_inputs(fuzzified_inputs),
                "explanation": self._generate_inference_explanation(
                    fuzzified_inputs, rule_activations, crisp_output
                )
            }
            
            # キャッシュ保存
            self.inference_cache[cache_key] = result
            
            # 統計更新
            self.stats["total_inferences"] += 1
            self._update_rule_activation_stats(rule_activations)
            self._update_processing_time_stats(processing_time)
            
            return result
            
        except Exception as e:
            logger.error(f"ファジィ推論エラー: {e}")
            return {
                "output_value": 0.5,
                "confidence": 0.0,
                "error": str(e),
                "method": self.inference_method.value
            }
    
    def _fuzzify_inputs(self, student_profile: Dict[str, Any], 
                       lab_profile: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """入力ファジィ化（13項目対応）"""
        
        fuzzified = {}
        
        for criterion in self.CRITERIA_NAMES:
            if criterion in self.input_variables:
                variable = self.input_variables[criterion]
                
                # 学生と研究室の値から適合度を計算
                student_val = student_profile.get(criterion, 5.0)  # デフォルト値
                lab_val = lab_profile.get(criterion, 5.0)
                
                # 類似度計算（距離ベース）
                diff = abs(float(student_val) - float(lab_val))
                similarity = max(0.0, min(10.0, 10.0 - diff))  # 1-10スケールに正規化
                
                # ファジィ化
                fuzzified[criterion] = variable.fuzzify(similarity)
        
        return fuzzified
    
    def _evaluate_rules(self, fuzzified_inputs: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """ルール評価"""
        
        rule_activations = []
        
        for rule in self.rules:
            strength = rule.evaluate(fuzzified_inputs)
            
            rule_activations.append({
                "rule_id": rule.rule_id,
                "rule": rule,
                "strength": strength,
                "consequent": rule.consequent
            })
        
        return rule_activations
    
    def _mamdani_inference(self, rule_activations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Mamdani推論"""
        
        # 結論部の言語値別に最大強度を計算
        consequent_strengths = {}
        
        for activation in rule_activations:
            consequent = activation["consequent"]
            strength = activation["strength"]
            
            if consequent not in consequent_strengths:
                consequent_strengths[consequent] = strength
            else:
                consequent_strengths[consequent] = max(consequent_strengths[consequent], strength)
        
        return consequent_strengths
    
    def _sugeno_inference(self, rule_activations: List[Dict[str, Any]], 
                         fuzzified_inputs: Dict[str, Dict[str, float]]) -> float:
        """Sugeno推論"""
        
        weighted_sum = 0.0
        total_strength = 0.0
        
        for activation in rule_activations:
            strength = activation["strength"]
            if strength > 0:
                # 結論部の線形関数（簡易版：定数項）
                consequent_mapping = {
                    "very_low": 0.1,
                    "low": 0.3,
                    "medium": 0.5,
                    "high": 0.7,
                    "very_high": 0.9
                }
                
                consequent_value = consequent_mapping.get(activation["consequent"], 0.5)
                weighted_sum += strength * consequent_value
                total_strength += strength
        
        return weighted_sum / total_strength if total_strength > 0 else 0.5
    
    def _tsukamoto_inference(self, rule_activations: List[Dict[str, Any]]) -> float:
        """Tsukamoto推論"""
        
        # 簡易実装（Mamdaniに近い処理）
        return self._sugeno_inference(rule_activations, {})
    
    def _defuzzify(self, inference_result: Any) -> float:
        """非ファジィ化"""
        
        if isinstance(inference_result, dict):
            # Mamdani推論結果の場合
            if self.defuzzification_method == DefuzzificationMethod.CENTROID:
                return self._centroid_defuzzification(inference_result)
            elif self.defuzzification_method == DefuzzificationMethod.MOM:
                return self._mom_defuzzification(inference_result)
            else:
                return self._centroid_defuzzification(inference_result)
        
        else:
            # Sugeno/Tsukamoto推論結果の場合
            return float(inference_result)
    
    def _centroid_defuzzification(self, consequent_strengths: Dict[str, float]) -> float:
        """重心非ファジィ化"""
        
        if not consequent_strengths or not self.output_variable:
            return 0.5
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        # 各結論言語値の代表値と強度から重心計算
        consequent_centers = {
            "very_low": 0.1,
            "low": 0.25,
            "medium": 0.5,
            "high": 0.75,
            "very_high": 0.9
        }
        
        for consequent, strength in consequent_strengths.items():
            if strength > 0 and consequent in consequent_centers:
                center = consequent_centers[consequent]
                weighted_sum += center * strength
                total_weight += strength
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def _mom_defuzzification(self, consequent_strengths: Dict[str, float]) -> float:
        """最大値平均非ファジィ化"""
        
        if not consequent_strengths:
            return 0.5
        
        max_strength = max(consequent_strengths.values())
        max_consequents = [k for k, v in consequent_strengths.items() if v == max_strength]
        
        consequent_centers = {
            "very_low": 0.1,
            "low": 0.25,
            "medium": 0.5,
            "high": 0.75,
            "very_high": 0.9
        }
        
        centers = [consequent_centers.get(c, 0.5) for c in max_consequents]
        return sum(centers) / len(centers) if centers else 0.5
    
    def _calculate_inference_confidence(self, 
                                      fuzzified_inputs: Dict[str, Dict[str, float]],
                                      rule_activations: List[Dict[str, Any]],
                                      output_value: float) -> float:
        """推論信頼度計算"""
        
        # 1. ルール活性化度による信頼度
        active_rules = [r for r in rule_activations if r["strength"] > 0]
        rule_confidence = len(active_rules) / len(self.rules) if self.rules else 0.0
        
        # 2. データ完全性による信頼度
        complete_criteria = sum(1 for inputs in fuzzified_inputs.values() if max(inputs.values()) > 0)
        data_confidence = complete_criteria / len(self.CRITERIA_NAMES)
        
        # 3. 出力値の安定性
        if active_rules:
            max_strength = max(r["strength"] for r in active_rules)
            strength_confidence = max_strength
        else:
            strength_confidence = 0.0
        
        # 総合信頼度
        confidence = (rule_confidence * 0.4 + data_confidence * 0.4 + strength_confidence * 0.2)
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_data_completeness(self, student_profile: Dict[str, Any], 
                                   lab_profile: Dict[str, Any]) -> float:
        """データ完全性計算"""
        
        total_criteria = len(self.CRITERIA_NAMES)
        complete_criteria = 0
        
        for criterion in self.CRITERIA_NAMES:
            if (criterion in student_profile and student_profile[criterion] is not None and
                criterion in lab_profile and lab_profile[criterion] is not None):
                complete_criteria += 1
        
        return complete_criteria / total_criteria
    
    def _generate_detailed_analysis(self, 
                                  student_profile: Dict[str, Any],
                                  lab_profile: Dict[str, Any],
                                  fuzzified_inputs: Dict[str, Dict[str, float]],
                                  rule_activations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """詳細分析生成"""
        
        analysis = {
            "criteria_analysis": {},
            "rule_analysis": {
                "total_rules": len(self.rules),
                "active_rules": len([r for r in rule_activations if r["strength"] > 0]),
                "top_active_rules": []
            },
            "strength_distribution": {},
            "recommendation_factors": []
        }
        
        # 基準別分析
        for criterion, memberships in fuzzified_inputs.items():
            dominant_linguistic = max(memberships.items(), key=lambda x: x[1])
            analysis["criteria_analysis"][criterion] = {
                "dominant_linguistic_value": dominant_linguistic[0],
                "membership_degree": dominant_linguistic[1],
                "all_memberships": memberships,
                "importance_weight": self.CRITERIA_WEIGHTS.get(criterion, 1.0)
            }
        
        # 活性ルール分析
        active_rules = sorted(
            [r for r in rule_activations if r["strength"] > 0],
            key=lambda x: x["strength"],
            reverse=True
        )[:5]  # 上位5ルール
        
        analysis["rule_analysis"]["top_active_rules"] = [
            {
                "rule_id": r["rule_id"],
                "strength": r["strength"],
                "consequent": r["consequent"],
                "antecedents": r["rule"].antecedents
            }
            for r in active_rules
        ]
        
        # 強度分布
        consequent_strengths = {}
        for activation in rule_activations:
            consequent = activation["consequent"]
            strength = activation["strength"]
            if consequent not in consequent_strengths:
                consequent_strengths[consequent] = []
            consequent_strengths[consequent].append(strength)
        
        for consequent, strengths in consequent_strengths.items():
            analysis["strength_distribution"][consequent] = {
                "max": max(strengths),
                "avg": sum(strengths) / len(strengths),
                "count": len([s for s in strengths if s > 0])
            }
        
        return analysis
    
    def _format_fuzzified_inputs(self, fuzzified_inputs: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """ファジィ化入力のフォーマット"""
        
        formatted = {}
        for criterion, memberships in fuzzified_inputs.items():
            dominant = max(memberships.items(), key=lambda x: x[1])
            formatted[criterion] = {
                "dominant_value": dominant[0],
                "dominant_degree": dominant[1],
                "all_degrees": memberships
            }
        
        return formatted
    
    def _generate_inference_explanation(self, 
                                      fuzzified_inputs: Dict[str, Dict[str, float]],
                                      rule_activations: List[Dict[str, Any]],
                                      output_value: float) -> str:
        """推論説明文生成"""
        
        # 最も影響の大きいルールを特定
        most_active_rule = max(rule_activations, key=lambda x: x["strength"])
        
        # 支配的な基準を特定
        dominant_criteria = []
        for criterion, memberships in fuzzified_inputs.items():
            max_membership = max(memberships.values())
            if max_membership > 0.7:
                dominant_value = max(memberships.items(), key=lambda x: x[1])[0]
                criterion_name = criterion.replace("_", " ").title()
                dominant_criteria.append(f"{criterion_name}: {dominant_value}")
        
        explanation_parts = []
        
        # 適合性レベル
        if output_value >= 0.8:
            explanation_parts.append("非常に高い適合性が推論されました。")
        elif output_value >= 0.6:
            explanation_parts.append("高い適合性が推論されました。")
        elif output_value >= 0.4:
            explanation_parts.append("中程度の適合性が推論されました。")
        else:
            explanation_parts.append("低い適合性が推論されました。")
        
        # 主要要因
        if most_active_rule["strength"] > 0.5:
            explanation_parts.append(f"主要な判定要因はルール{most_active_rule['rule_id']}（強度: {most_active_rule['strength']:.2f}）によるものです。")
        
        # 支配的基準
        if dominant_criteria:
            criteria_text = "、".join(dominant_criteria[:3])
            explanation_parts.append(f"特に{criteria_text}が判定に大きく影響しています。")
        
        # 活性ルール数
        active_count = len([r for r in rule_activations if r["strength"] > 0])
        explanation_parts.append(f"合計{active_count}個のルールが活性化されました。")
        
        return " ".join(explanation_parts)
    
    def _generate_cache_key(self, student_profile: Dict[str, Any], 
                          lab_profile: Dict[str, Any]) -> str:
        """キャッシュキー生成"""
        
        # 関連する値のみを使用してキーを生成
        key_values = []
        for criterion in self.CRITERIA_NAMES:
            student_val = student_profile.get(criterion, "N/A")
            lab_val = lab_profile.get(criterion, "N/A")
            key_values.append(f"{criterion}:{student_val}-{lab_val}")
        
        return hash("|".join(key_values))
    
    def _update_rule_activation_stats(self, rule_activations: List[Dict[str, Any]]):
        """ルール活性化統計更新"""
        
        for activation in rule_activations:
            if activation["strength"] > 0:
                rule_id = activation["rule_id"]
                if rule_id not in self.stats["rule_activations"]:
                    self.stats["rule_activations"][rule_id] = 0
                self.stats["rule_activations"][rule_id] += 1
    
    def _update_processing_time_stats(self, processing_time: float):
        """処理時間統計更新"""
        
        current_avg = self.stats["average_inference_time"]
        total_inferences = self.stats["total_inferences"]
        
        if total_inferences == 1:
            self.stats["average_inference_time"] = processing_time
        else:
            self.stats["average_inference_time"] = (
                (current_avg * (total_inferences - 1) + processing_time) / total_inferences
            )
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """システム統計取得"""
        
        return {
            "inference_engine": {
                "method": self.inference_method.value,
                "defuzzification": self.defuzzification_method.value,
                "criteria_count": len(self.CRITERIA_NAMES),
                "rules_count": len(self.rules),
                "input_variables": len(self.input_variables)
            },
            "performance": self.stats,
            "rule_usage": {
                "most_used_rules": sorted(
                    self.stats["rule_activations"].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10],
                "unused_rules": len(self.rules) - len(self.stats["rule_activations"])
            },
            "criteria_weights": self.CRITERIA_WEIGHTS
        }
    
    def clear_cache(self):
        """キャッシュクリア"""
        self.inference_cache.clear()
        logger.info("推論結果キャッシュをクリアしました")

# ファクトリー関数と使用インターフェース
def create_complete_fuzzy_engine(inference_method: str = "mamdani") -> Complete13CriteriaFuzzyEngine:
    """完全13項目対応ファジィエンジン生成"""
    
    method_map = {
        "mamdani": InferenceMethod.MAMDANI,
        "sugeno": InferenceMethod.SUGENO,
        "tsukamoto": InferenceMethod.TSUKAMOTO
    }
    
    method = method_map.get(inference_method.lower(), InferenceMethod.MAMDANI)
    return Complete13CriteriaFuzzyEngine(inference_method=method)

# 簡易使用インターフェース
def evaluate_compatibility_fuzzy_13(student_profile: Dict[str, Any],
                                  lab_profile: Dict[str, Any],
                                  **kwargs) -> Dict[str, Any]:
    """13項目ファジィ適合性評価の簡易インターフェース"""
    
    engine = create_complete_fuzzy_engine(**kwargs)
    return engine.infer_lab_compatibility(student_profile, lab_profile)

# バックワード互換性のためのクラス別名
SimpleFuzzyInferenceEngine = Complete13CriteriaFuzzyEngine
CompleteTriangularFuzzyEngine = Complete13CriteriaFuzzyEngine
CompleteFuzzyInferenceEngine = Complete13CriteriaFuzzyEngine

# エクスポート用リスト
__all__ = [
    "Complete13CriteriaFuzzyEngine",
    "SimpleFuzzyInferenceEngine", 
    "CompleteTriangularFuzzyEngine",
    "CompleteFuzzyInferenceEngine",
    "FuzzySet",
    "FuzzyVariable", 
    "FuzzyRule",
    "MembershipFunctionType",
    "InferenceMethod", 
    "DefuzzificationMethod",
    "create_complete_fuzzy_engine",
    "evaluate_compatibility_fuzzy_13"
]