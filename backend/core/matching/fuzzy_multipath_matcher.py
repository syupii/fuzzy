# backend/core/matching/fuzzy_multipath_matcher.py
"""
ファジィ決定木マッチャー - 統合版 v2.2

【改善点】
- 正規化: value / 10
- 3分岐メンバーシップ関数: 低=三角、中=台形、高=三角
- 2分岐メンバーシップ関数: 低=三角、高=三角（別途定義）
- 重なり領域を適切に設計し、枝刈りのバランスを改善
- 分野スコア: 重み付き平均方式
- ボーナス/ペナルティ: 削除
- 分野重視度: λ = 分野重視度 / 10

【基本スコアの計算方法】
S_basic = γ × S_fuzzy + (1 - γ) × S_gaussian

【分野スコアの計算方法】
S_field = Σ(match_score_i × interest_i) / Σ(interest_i)

【最終適合度の計算方法】
λ = 分野重視度 / 10
S = (1 - λ) × S_basic + λ × S_field
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class FuzzyPath:
    """ファジィパス"""
    path_id: int
    layers: List[Tuple[str, str, float]]  # [(criterion, label, membership), ...]
    total_membership: float  # パス全体の所属度（研究室から計算）
    leaf_value: float  # リーフ値（学生から計算）


@dataclass
class CompatibilityResult:
    """適合度計算結果"""
    total_compatibility: float
    basic_score: float
    fuzzy_score: float
    gaussian_score: float
    field_score: float
    field_weight_lambda: float
    basic_weight: float
    
    criteria_scores: Dict[str, float]
    fuzzy_paths: List[FuzzyPath]
    field_detail: Dict[str, Any]
    
    explanation: str
    explanation_detailed: Optional[str] = ""
    explanation_short: Optional[str] = ""
    recommendation: str = ""


class MembershipFunctions:
    """
    メンバーシップ関数（改善版 v2.1）
    
    【3分岐（優先度 ≥ 8）】
    - 低（三角）: ピーク 0.1、終了 0.5
    - 中（台形）: 開始 0.2、上辺 0.4-0.7、終了 0.9
    - 高（三角）: 開始 0.6、ピーク 1.0
    
    【2分岐（優先度 5〜7）】
    - 低（三角）: ピーク 0.1、終了 0.6
    - 高（三角）: 開始 0.4、ピーク 1.0
    """
    
    # ========================================
    # 3分岐用メンバーシップ関数
    # ========================================
    
    @staticmethod
    def low_3level(x: float) -> float:
        """
        低（三角形）- 3分岐用
        
        ピーク: x = 0.1 で μ = 1.0
        終了: x = 0.5 で μ = 0.0
        """
        if x <= 0.1:
            return 1.0
        elif x < 0.5:
            return (0.5 - x) / 0.4
        else:
            return 0.0
    
    @staticmethod
    def medium_3level(x: float) -> float:
        """
        中（台形）- 3分岐用
        
        開始: x = 0.2 で μ = 0.0
        上辺: x = 0.4〜0.7 で μ = 1.0
        終了: x = 0.9 で μ = 0.0
        """
        if x <= 0.2:
            return 0.0
        elif x < 0.4:
            return (x - 0.2) / 0.2
        elif x <= 0.7:
            return 1.0
        elif x < 0.9:
            return (0.9 - x) / 0.2
        else:
            return 0.0
    
    @staticmethod
    def high_3level(x: float) -> float:
        """
        高（三角形）- 3分岐用
        
        開始: x = 0.6 で μ = 0.0
        ピーク: x = 1.0 で μ = 1.0
        """
        if x <= 0.6:
            return 0.0
        elif x < 1.0:
            return (x - 0.6) / 0.4
        else:
            return 1.0
    
    # ========================================
    # 2分岐用メンバーシップ関数
    # ========================================
    
    @staticmethod
    def low_2level(x: float) -> float:
        """
        低（三角形）- 2分岐用
        
        ピーク: x = 0.1 で μ = 1.0
        終了: x = 0.6 で μ = 0.0
        """
        if x <= 0.1:
            return 1.0
        elif x < 0.6:
            return (0.6 - x) / 0.5
        else:
            return 0.0
    
    @staticmethod
    def high_2level(x: float) -> float:
        """
        高（三角形）- 2分岐用
        
        開始: x = 0.4 で μ = 0.0
        ピーク: x = 1.0 で μ = 1.0
        """
        if x <= 0.4:
            return 0.0
        elif x < 1.0:
            return (x - 0.4) / 0.6
        else:
            return 1.0
    
    # ========================================
    # ファジィ化関数
    # ========================================
    
    @staticmethod
    def fuzzify_3level(x: float) -> Dict[str, float]:
        """
        値をファジィ化（3段階: 低・中・高）
        
        ※ 合計が1になるように正規化
        """
        raw_low = MembershipFunctions.low_3level(x)
        raw_medium = MembershipFunctions.medium_3level(x)
        raw_high = MembershipFunctions.high_3level(x)
        
        total = raw_low + raw_medium + raw_high
        
        if total > 0:
            return {
                "low": raw_low / total,
                "medium": raw_medium / total,
                "high": raw_high / total
            }
        else:
            return {"low": 0.0, "medium": 1.0, "high": 0.0}
    
    @staticmethod
    def fuzzify_2level(x: float) -> Dict[str, float]:
        """
        値をファジィ化（2段階: 低・高）
        
        ※ 合計が1になるように正規化
        """
        raw_low = MembershipFunctions.low_2level(x)
        raw_high = MembershipFunctions.high_2level(x)
        
        total = raw_low + raw_high
        
        if total > 0:
            return {
                "low": raw_low / total,
                "high": raw_high / total
            }
        else:
            return {"low": 0.5, "high": 0.5}
    
    # ========================================
    # 事前計算用テーブル
    # ========================================
    
    @staticmethod
    def get_membership_table_3level() -> Dict[int, Dict[str, float]]:
        """
        入力値1-10に対する3分岐メンバーシップ値のテーブル
        """
        table = {}
        for value in range(1, 11):
            normalized = value / 10.0
            table[value] = MembershipFunctions.fuzzify_3level(normalized)
        return table
    
    @staticmethod
    def get_membership_table_2level() -> Dict[int, Dict[str, float]]:
        """
        入力値1-10に対する2分岐メンバーシップ値のテーブル
        """
        table = {}
        for value in range(1, 11):
            normalized = value / 10.0
            table[value] = MembershipFunctions.fuzzify_2level(normalized)
        return table


class FuzzyMultiPathMatcher:
    """
    ファジィ決定木マッチャー（統合版 v2.2）
    
    【変更点 v2.2】
    - ボーナス/ペナルティを削除
    - 分野重視度の計算を変更: λ = 分野重視度 / 10
    - 最終適合度: S = (1 - λ) × S_basic + λ × S_field
    """
    
    CRITERIA = [
        "research_intensity", "advisor_style", "team_work",
        "workload", "theory_practice", "skill_development",
        "lab_atmosphere", "flexibility", "publication_opportunity",
        "interdisciplinary", "communication_style"
    ]
    
    # パラメータ
    SIMILARITY_SIGMA = 0.3
    PRUNING_THRESHOLD = 0.01
    HIGH_PRIORITY_THRESHOLD = 8.0
    MID_PRIORITY_THRESHOLD = 5.0
    FUZZY_GAUSSIAN_GAMMA = 0.5
    
    # 分野マッチング
    FIELD_EXACT_MATCH_SCORE = 1.0
    FIELD_CATEGORY_DECAY = 0.7
    FIELD_NO_MATCH_SCORE = 0.3
    
    def __init__(self):
        print("=" * 70)
        print("FuzzyMultiPathMatcher v2.2 初期化")
        print("=" * 70)
        print(f"  正規化方式: value / 10")
        print(f"  3分岐メンバーシップ: 低=三角(~0.5), 中=台形(0.2~0.9), 高=三角(0.6~)")
        print(f"  2分岐メンバーシップ: 低=三角(~0.6), 高=三角(0.4~)")
        print(f"  σ (ガウス類似度) = {self.SIMILARITY_SIGMA}")
        print(f"  γ (S_fuzzy比重) = {self.FUZZY_GAUSSIAN_GAMMA}")
        print(f"  枝刈り閾値 = {self.PRUNING_THRESHOLD}")
        print(f"  分野重視度計算: λ = 分野重視度 / 10")
        print(f"  最終適合度: S = (1-λ)×S_basic + λ×S_field")
        print("=" * 70)
        
        # メンバーシップテーブルを事前計算
        self._membership_table_3level = MembershipFunctions.get_membership_table_3level()
        self._membership_table_2level = MembershipFunctions.get_membership_table_2level()
        
        print("\n【3分岐メンバーシップ値テーブル】")
        print("-" * 55)
        print(f"{'入力値':>6} | {'μ_低':>8} | {'μ_中':>8} | {'μ_高':>8}")
        print("-" * 55)
        for value, memberships in self._membership_table_3level.items():
            print(f"{value:>6} | {memberships['low']:>8.4f} | {memberships['medium']:>8.4f} | {memberships['high']:>8.4f}")
        print("-" * 55)
        
        print("\n【2分岐メンバーシップ値テーブル】")
        print("-" * 40)
        print(f"{'入力値':>6} | {'μ_低':>8} | {'μ_高':>8}")
        print("-" * 40)
        for value, memberships in self._membership_table_2level.items():
            print(f"{value:>6} | {memberships['low']:>8.4f} | {memberships['high']:>8.4f}")
        print("-" * 40)
    
    def _normalize_value(self, value: float) -> float:
        """
        値を0-1に正規化
        
        入力: [1, 10] → 出力: [0.1, 1.0]
        """
        if value >= 1.0 and value <= 10.0:
            return value / 10.0
        elif value > 1.0:
            return min(value / 10.0, 1.0)
        return value
    
    def calculate_compatibility(
        self,
        student: Dict[str, Any],
        lab: Dict[str, Any]
    ) -> CompatibilityResult:
        """適合度計算（メイン関数）"""
        
        print(f"\n{'#'*70}")
        print(f"### 適合度計算開始 ###")
        print(f"### 研究室: {lab.get('name', 'Unknown')} ###")
        print(f"{'#'*70}")
        
        # 第1段階: 決定木構築
        priorities = self._get_sorted_priorities(student)
        tree_layers = self._build_fuzzy_tree(priorities)
        
        print(f"\n【第1段階】決定木構築")
        print(f"  レイヤー数: {len(tree_layers)}")
        for i, layer in enumerate(tree_layers, 1):
            print(f"  Layer {i}: {layer['criterion']} ({layer['branches']}分岐, 優先度{layer['priority']})")
        
        # 第2段階: 研究室のパス所属度
        lab_fuzzified = self._fuzzify_lab(lab, tree_layers)
        
        print(f"\n【第2段階】研究室のファジィ化")
        for criterion, memberships in lab_fuzzified.items():
            layer = next((l for l in tree_layers if l['criterion'] == criterion), None)
            if layer:
                if layer['branches'] == 3:
                    print(f"  {criterion}: 低={memberships['low']:.4f}, 中={memberships['medium']:.4f}, 高={memberships['high']:.4f}")
                else:
                    print(f"  {criterion}: 低={memberships['low']:.4f}, 高={memberships['high']:.4f}")
        
        fuzzy_paths = self._generate_paths_from_lab(tree_layers, lab_fuzzified)
        fuzzy_paths = self._prune_and_normalize_paths(fuzzy_paths)
        
        print(f"\n  有効パス数: {len(fuzzy_paths)}")
        for path in fuzzy_paths:
            labels = "-".join([l[1][0].upper() for l in path.layers])
            print(f"    パス「{labels}」: 所属度 = {path.total_membership:.4f}")
        
        # 第3段階: リーフ値計算
        student_fuzzified = self._fuzzify_student(student, tree_layers)
        
        print(f"\n【第3段階】学生のファジィ化")
        for criterion, memberships in student_fuzzified.items():
            layer = next((l for l in tree_layers if l['criterion'] == criterion), None)
            if layer:
                if layer['branches'] == 3:
                    print(f"  {criterion}: 低={memberships['low']:.4f}, 中={memberships['medium']:.4f}, 高={memberships['high']:.4f}")
                else:
                    print(f"  {criterion}: 低={memberships['low']:.4f}, 高={memberships['high']:.4f}")
        
        fuzzy_paths = self._calculate_leaf_values(
            fuzzy_paths, student_fuzzified, tree_layers, student
        )
        
        print(f"\n  リーフ値計算結果:")
        for path in fuzzy_paths:
            labels = "-".join([l[1][0].upper() for l in path.layers])
            print(f"    パス「{labels}」: 所属度={path.total_membership:.4f}, リーフ値={path.leaf_value:.4f}")
        
        # 第4段階: S_fuzzy
        fuzzy_score = self._calculate_fuzzy_score(fuzzy_paths)
        print(f"\n【第4段階】S_fuzzy = {fuzzy_score:.4f}")
        
        # S_gaussian
        gaussian_score, criteria_scores = self._calculate_gaussian_score(student, lab)
        print(f"\n【ガウス類似度】S_gaussian = {gaussian_score:.4f}")
        
        # 基本スコア統合
        gamma = self.FUZZY_GAUSSIAN_GAMMA
        basic_score = gamma * fuzzy_score + (1 - gamma) * gaussian_score
        print(f"\n【基本スコア統合】")
        print(f"  S_basic = {gamma} × {fuzzy_score:.4f} + {1-gamma} × {gaussian_score:.4f} = {basic_score:.4f}")
        
        # 分野マッチング
        field_interests = student.get("field_interests", {})
        lab_field = lab.get("field_id", "")
        field_score, field_detail = self._calculate_field_match_weighted_average(
            field_interests, lab_field
        )
        print(f"\n【分野マッチング】S_field = {field_score:.4f}")
        
        # 最終スコア計算
        # λ = 分野重視度 / 10
        field_priority = student.get("field_priority", 5.0)
        lambda_weight = field_priority / 10.0
        
        # S = (1 - λ) × S_basic + λ × S_field
        total = (1 - lambda_weight) * basic_score + lambda_weight * field_score
        
        # 0〜1の範囲にクリップ
        total = max(0.0, min(1.0, total))
        
        print(f"\n【最終スコア】")
        print(f"  分野重視度 = {field_priority}")
        print(f"  λ = {field_priority} / 10 = {lambda_weight:.2f}")
        print(f"  S = (1 - {lambda_weight:.2f}) × {basic_score:.4f} + {lambda_weight:.2f} × {field_score:.4f}")
        print(f"    = {1 - lambda_weight:.2f} × {basic_score:.4f} + {lambda_weight:.2f} × {field_score:.4f}")
        print(f"    = {(1 - lambda_weight) * basic_score:.4f} + {lambda_weight * field_score:.4f}")
        print(f"    = {total:.4f}")
        print(f"\n★★★ 最終適合度 = {total:.4f} ({total*100:.2f}%) ★★★")
        
        explanation, explanation_detailed, explanation_short = self._generate_explanation(
            total, basic_score, field_score, lambda_weight,
            field_detail, len(fuzzy_paths),
            student=student, lab=lab, criteria_scores=criteria_scores
        )
        
        recommendation = self._get_recommendation(total)
        
        return CompatibilityResult(
            total_compatibility=total,
            basic_score=basic_score,
            fuzzy_score=fuzzy_score,
            gaussian_score=gaussian_score,
            field_score=field_score,
            field_weight_lambda=lambda_weight,
            basic_weight=1 - lambda_weight,
            criteria_scores=criteria_scores,
            fuzzy_paths=fuzzy_paths,
            field_detail=field_detail,
            explanation=explanation,
            explanation_detailed=explanation_detailed,
            explanation_short=explanation_short,
            recommendation=recommendation
        )
    
    # ========================================
    # 第1段階: 決定木構築
    # ========================================
    
    def _get_sorted_priorities(self, student: Dict[str, Any]) -> List[Dict[str, Any]]:
        priorities = []
        for criterion in self.CRITERIA:
            priority_key = f"{criterion}_priority"
            priority = student.get(priority_key, 5.0)
            priorities.append({"criterion": criterion, "priority": priority})
        priorities.sort(key=lambda x: x["priority"], reverse=True)
        return priorities
    
    def _build_fuzzy_tree(self, priorities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tree_layers = []
        for item in priorities:
            priority = item["priority"]
            criterion = item["criterion"]
            if priority >= self.HIGH_PRIORITY_THRESHOLD:
                tree_layers.append({
                    "criterion": criterion,
                    "priority": priority,
                    "branches": 3,
                    "labels": ["low", "medium", "high"]
                })
            elif priority >= self.MID_PRIORITY_THRESHOLD:
                tree_layers.append({
                    "criterion": criterion,
                    "priority": priority,
                    "branches": 2,
                    "labels": ["low", "high"]
                })
        return tree_layers
    
    # ========================================
    # 第2段階: 研究室のパス所属度
    # ========================================
    
    def _fuzzify_lab(
        self,
        lab: Dict[str, Any],
        tree_layers: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """研究室の各項目をファジィ化"""
        lab_fuzzified = {}
        for layer in tree_layers:
            criterion = layer["criterion"]
            branches = layer["branches"]
            raw_value = lab.get(criterion, 5.0)
            normalized = self._normalize_value(raw_value)
            
            if branches == 3:
                memberships = MembershipFunctions.fuzzify_3level(normalized)
            else:
                memberships = MembershipFunctions.fuzzify_2level(normalized)
            
            lab_fuzzified[criterion] = memberships
        return lab_fuzzified
    
    def _generate_paths_from_lab(
        self,
        tree_layers: List[Dict[str, Any]],
        lab_fuzzified: Dict[str, Dict[str, float]]
    ) -> List[FuzzyPath]:
        if not tree_layers:
            return [FuzzyPath(path_id=0, layers=[], total_membership=1.0, leaf_value=0.0)]
        
        all_paths = []
        self._generate_paths_recursive(tree_layers, lab_fuzzified, 0, [], 1.0, all_paths)
        return all_paths
    
    def _generate_paths_recursive(
        self,
        tree_layers: List[Dict[str, Any]],
        lab_fuzzified: Dict[str, Dict[str, float]],
        layer_idx: int,
        current_path: List[Tuple[str, str, float]],
        cumulative_membership: float,
        all_paths: List[FuzzyPath]
    ):
        if layer_idx >= len(tree_layers):
            all_paths.append(FuzzyPath(
                path_id=len(all_paths),
                layers=current_path.copy(),
                total_membership=cumulative_membership,
                leaf_value=0.0
            ))
            return
        
        layer = tree_layers[layer_idx]
        criterion = layer["criterion"]
        labels = layer["labels"]
        memberships = lab_fuzzified[criterion]
        
        for label in labels:
            membership = memberships[label]
            new_membership = cumulative_membership * membership
            
            if new_membership < self.PRUNING_THRESHOLD:
                continue
            
            new_path = current_path + [(criterion, label, membership)]
            self._generate_paths_recursive(
                tree_layers, lab_fuzzified, layer_idx + 1,
                new_path, new_membership, all_paths
            )
    
    def _prune_and_normalize_paths(self, paths: List[FuzzyPath]) -> List[FuzzyPath]:
        paths = [p for p in paths if p.total_membership >= self.PRUNING_THRESHOLD]
        if not paths:
            return paths
        total = sum(p.total_membership for p in paths)
        if total > 0:
            for path in paths:
                path.total_membership = path.total_membership / total
        return paths
    
    # ========================================
    # 第3段階: リーフ値計算
    # ========================================
    
    def _fuzzify_student(
        self,
        student: Dict[str, Any],
        tree_layers: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        student_fuzzified = {}
        for layer in tree_layers:
            criterion = layer["criterion"]
            branches = layer["branches"]
            raw_value = student.get(criterion, 5.0)
            normalized = self._normalize_value(raw_value)
            
            if branches == 3:
                memberships = MembershipFunctions.fuzzify_3level(normalized)
            else:
                memberships = MembershipFunctions.fuzzify_2level(normalized)
            
            student_fuzzified[criterion] = memberships
        return student_fuzzified
    
    def _calculate_leaf_values(
        self,
        paths: List[FuzzyPath],
        student_fuzzified: Dict[str, Dict[str, float]],
        tree_layers: List[Dict[str, Any]],
        student: Dict[str, Any]
    ) -> List[FuzzyPath]:
        for path in paths:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for criterion, label, _ in path.layers:
                student_membership = student_fuzzified[criterion][label]
                priority_key = f"{criterion}_priority"
                priority = student.get(priority_key, 5.0)
                weight = self._calculate_priority_weight(priority)
                
                weighted_sum += student_membership * weight
                total_weight += weight
            
            path.leaf_value = weighted_sum / total_weight if total_weight > 0 else 0.5
        return paths
    
    # ========================================
    # 第4段階: S_fuzzy
    # ========================================
    
    def _calculate_fuzzy_score(self, paths: List[FuzzyPath]) -> float:
        if not paths:
            return 0.5
        return sum(path.total_membership * path.leaf_value for path in paths)
    
    # ========================================
    # ガウス類似度
    # ========================================
    
    def _calculate_gaussian_score(
        self,
        student: Dict[str, Any],
        lab: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        criteria_scores = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for criterion in self.CRITERIA:
            student_val = self._normalize_value(student.get(criterion, 5.0))
            lab_val = self._normalize_value(lab.get(criterion, 5.0))
            similarity = self._calculate_gaussian_similarity(student_val, lab_val)
            priority_key = f"{criterion}_priority"
            priority = student.get(priority_key, 5.0)
            weight = self._calculate_priority_weight(priority)
            
            criteria_scores[criterion] = similarity
            weighted_sum += similarity * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5, criteria_scores
    
    def _calculate_gaussian_similarity(self, student_val: float, lab_val: float) -> float:
        d = abs(student_val - lab_val)
        return math.exp(-(d ** 2) / (2 * self.SIMILARITY_SIGMA ** 2))
    
    def _calculate_priority_weight(self, priority: float) -> float:
        return (priority / 10.0) ** 1.5
    
    # ========================================
    # 分野マッチング（重み付き平均）
    # ========================================
    
    def _calculate_field_match_weighted_average(
        self,
        field_interests: Dict[str, float],
        lab_field: str
    ) -> Tuple[float, Dict[str, Any]]:
        
        if not field_interests:
            return 0.5, {
                "match_type": "unknown",
                "primary_match_type": "unknown",
                "lab_field": lab_field,
                "message": "興味分野が指定されていません",
                "field_details": []
            }
        
        if not lab_field:
            return 0.5, {
                "match_type": "unknown",
                "primary_match_type": "unknown",
                "lab_field": lab_field,
                "message": "研究室の分野が指定されていません",
                "field_details": []
            }
        
        weighted_sum = 0.0
        total_weight = 0.0
        field_details = []
        has_exact_match = False
        
        for interest_field, interest_level in field_interests.items():
            if interest_field == lab_field:
                match_type = "exact"
                match_score = self.FIELD_EXACT_MATCH_SCORE
                has_exact_match = True
            elif self._is_same_category(interest_field, lab_field):
                match_type = "category"
                match_score = self.FIELD_CATEGORY_DECAY
            else:
                match_type = "no_match"
                match_score = self.FIELD_NO_MATCH_SCORE
            
            weighted_sum += match_score * interest_level
            total_weight += interest_level
            
            field_details.append({
                "field": interest_field,
                "interest_level": interest_level,
                "match_type": match_type,
                "match_score": match_score,
                "contribution": match_score * interest_level
            })
        
        field_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        if has_exact_match:
            primary_match_type = "exact"
        elif any(d["match_type"] == "category" for d in field_details):
            primary_match_type = "category"
        else:
            primary_match_type = "no_match"
        
        return field_score, {
            "match_type": "weighted_average",
            "primary_match_type": primary_match_type,
            "lab_field": lab_field,
            "weighted_sum": weighted_sum,
            "total_weight": total_weight,
            "field_details": field_details,
            "message": self._generate_field_match_message(primary_match_type, field_details)
        }
    
    def _generate_field_match_message(
        self,
        primary_match_type: str,
        field_details: List[Dict[str, Any]]
    ) -> str:
        exact = [d for d in field_details if d["match_type"] == "exact"]
        category = [d for d in field_details if d["match_type"] == "category"]
        
        if exact:
            return f"興味分野「{exact[0]['field']}」と完全一致（興味度: {exact[0]['interest_level']}/10）"
        elif category:
            return f"興味分野「{category[0]['field']}」と同じカテゴリ"
        else:
            return "興味分野との一致なし"
    
    def _is_same_category(self, field1: str, field2: str) -> bool:
        FIELD_CATEGORIES = {
            "テクノロジー・システム": [
                "ai_ml", "image_processing", "network_security",
                "database_systems", "embedded_iot", "education_linguistics",
                "natural_science_math", "tourism_regional", "business_decision",
                "audio_processing", "system_ethics", "medical_healthcare"
            ],
            "クリエイティブ": [
                "web_design", "design_visual", "video_animation", "computer_music"
            ],
            "エンターテイメント": [
                "game_esports", "vr_ar_media"
            ],
            "人文・社会・体育": [
                "philosophy_humanities", "sports_science"
            ]
        }
        
        cat1 = cat2 = None
        for category, fields in FIELD_CATEGORIES.items():
            if field1 in fields:
                cat1 = category
            if field2 in fields:
                cat2 = category
        
        return cat1 and cat2 and cat1 == cat2
    
    # ========================================
    # ユーティリティ
    # ========================================
    
    # 項目名の日本語マッピング
    CRITERIA_NAMES = {
        "research_intensity": "研究強度",
        "advisor_style": "指導スタイル",
        "team_work": "チームワーク",
        "workload": "ワークロード",
        "theory_practice": "理論・実践バランス",
        "skill_development": "スキル開発",
        "lab_atmosphere": "研究室雰囲気",
        "flexibility": "柔軟性",
        "publication_opportunity": "論文発表機会",
        "interdisciplinary": "学際性",
        "communication_style": "コミュニケーション"
    }
    
    # 項目の説明（低い値と高い値の意味）
    CRITERIA_DESCRIPTIONS = {
        "research_intensity": {"low": "軽い研究", "high": "集中的な研究"},
        "advisor_style": {"low": "厳格な指導", "high": "自由な指導"},
        "team_work": {"low": "個人研究中心", "high": "チーム研究中心"},
        "workload": {"low": "軽い負荷", "high": "重い負荷"},
        "theory_practice": {"low": "理論重視", "high": "実践重視"},
        "skill_development": {"low": "専門特化", "high": "幅広いスキル"},
        "lab_atmosphere": {"low": "静かで集中", "high": "活発な議論"},
        "flexibility": {"low": "固定スケジュール", "high": "柔軟なスケジュール"},
        "publication_opportunity": {"low": "論文発表少なめ", "high": "論文発表多め"},
        "interdisciplinary": {"low": "単一分野", "high": "学際的連携"},
        "communication_style": {"low": "少人数で密接", "high": "オープンな交流"}
    }
    
    # 分野IDの日本語マッピング
    FIELD_NAMES = {
        "ai_ml": "AI・機械学習",
        "image_processing": "画像処理",
        "network_security": "ネットワーク・セキュリティ",
        "database_systems": "データベースシステム",
        "embedded_iot": "組込み・IoT",
        "web_design": "Webデザイン",
        "design_visual": "グラフィックデザイン",
        "video_animation": "映像・アニメーション",
        "computer_music": "コンピュータ音楽",
        "game_esports": "ゲーム開発",
        "vr_ar_media": "VR/AR・メディア",
        "education_linguistics": "教育・言語学",
        "philosophy_humanities": "哲学・人文学",
        "sports_science": "スポーツ科学",
        "tourism_regional": "観光・地域情報",
        "business_decision": "経営・意思決定",
        "natural_science_math": "自然科学・数学",
        "audio_processing": "音声処理",
        "system_ethics": "システム倫理",
        "medical_healthcare": "医療・ヘルスケア"
    }
    
    def _get_field_name(self, field_id: str) -> str:
        """分野IDを日本語名に変換"""
        return self.FIELD_NAMES.get(field_id, field_id)
    
    def _value_to_display(self, normalized_value: float) -> int:
        """正規化値を1-10表示に変換"""
        if normalized_value <= 1.0:
            return max(1, min(10, round(normalized_value * 10)))
        return max(1, min(10, round(normalized_value)))
    
    def _get_value_description(self, criterion: str, value: float) -> str:
        """値に応じた説明を返す"""
        display_val = self._value_to_display(value)
        desc = self.CRITERIA_DESCRIPTIONS.get(criterion, {"low": "低い", "high": "高い"})
        
        if display_val <= 3:
            return desc["low"]
        elif display_val >= 8:
            return desc["high"]
        else:
            return f"{desc['low']}と{desc['high']}の中間"
    
    def _generate_explanation(
        self,
        total: float,
        basic_score: float,
        field_score: float,
        lambda_weight: float,
        field_detail: Dict,
        num_paths: int,
        student: Dict[str, Any] = None,
        lab: Dict[str, Any] = None,
        criteria_scores: Dict[str, float] = None
    ) -> Tuple[str, str, str]:
        """
        説明文を生成（3種類）
        
        Returns:
            Tuple[str, str, str]: (従来版, 詳細版, 短縮版)
        """
        # 従来版（シンプルな閾値ベース）
        if total >= 0.8:
            legacy = "非常に高い適合性があります。あなたの希望条件を高い水準で満たしています。"
        elif total >= 0.6:
            legacy = "条件によく合致しています。大きな欠点がなく、安定して研究に取り組める環境です。"
        elif total >= 0.4:
            legacy = "一部の条件で妥協が必要ですが、検討候補として有力です。"
        else:
            legacy = "希望条件との差異が大きいため、慎重に検討してください。"
        
        # criteria_scoresがない場合は従来版のみ返す
        if not criteria_scores or not student or not lab:
            return legacy, "", ""
        
        # ========================================
        # 優先度情報を収集
        # ========================================
        priority_info = []
        for criterion in self.CRITERIA:
            priority = student.get(f"{criterion}_priority", 5.0)
            student_val = student.get(criterion, 0.5)
            lab_val = lab.get(criterion, 0.5)
            similarity = criteria_scores.get(criterion, 0.5)
            
            priority_info.append({
                "criterion": criterion,
                "name": self.CRITERIA_NAMES.get(criterion, criterion),
                "priority": priority,
                "student_val": student_val,
                "lab_val": lab_val,
                "similarity": similarity,
                "student_display": self._value_to_display(student_val),
                "lab_display": self._value_to_display(lab_val)
            })
        
        # 優先度でソート
        priority_info.sort(key=lambda x: x["priority"], reverse=True)
        
        # 高優先度項目（優先度8以上）
        high_priority = [p for p in priority_info if p["priority"] >= 8]
        # 中優先度項目（優先度5-7）
        mid_priority = [p for p in priority_info if 5 <= p["priority"] < 8]
        
        # ========================================
        # 詳細版の生成
        # ========================================
        detailed_parts = []
        
        # 1. 総合評価サマリー
        lab_name = lab.get("name", "この研究室")
        if total >= 0.8:
            detailed_parts.append(
                f"◆ 総合適合度 {total*100:.1f}%\n"
                f"{lab_name}は、あなたの希望条件を非常に高い水準で満たしています。"
            )
        elif total >= 0.6:
            detailed_parts.append(
                f"◆ 総合適合度 {total*100:.1f}%\n"
                f"{lab_name}は、あなたの希望条件によく合致しています。"
            )
        elif total >= 0.4:
            detailed_parts.append(
                f"◆ 総合適合度 {total*100:.1f}%\n"
                f"{lab_name}は、一部の条件で妥協が必要ですが、検討に値する研究室です。"
            )
        else:
            detailed_parts.append(
                f"◆ 総合適合度 {total*100:.1f}%\n"
                f"{lab_name}は、希望条件との差異が見られます。慎重にご検討ください。"
            )
        
        # 2. 分野マッチングの詳細説明
        if field_detail and lambda_weight > 0.3:
            primary_match = field_detail.get("primary_match_type", "unknown")
            lab_field = field_detail.get("lab_field", "")
            lab_field_name = self._get_field_name(lab_field)
            
            field_text = f"\n◆ 研究分野について（適合度への寄与: {lambda_weight*100:.0f}%）\n"
            
            if primary_match == "exact":
                field_details = field_detail.get("field_details", [])
                exact_fields = [d for d in field_details if d.get("match_type") == "exact"]
                if exact_fields:
                    interest_field = exact_fields[0]['field']
                    interest_level = exact_fields[0]['interest_level']
                    field_text += (
                        f"あなたが興味を持つ「{self._get_field_name(interest_field)}」と"
                        f"この研究室の専門分野が完全に一致しています。\n"
                        f"興味度 {interest_level}/10 で設定されており、分野適合度は {field_score*100:.0f}% です。"
                    )
            elif primary_match == "category":
                field_text += (
                    f"この研究室の専門分野「{lab_field_name}」は、\n"
                    f"あなたの興味分野と同じカテゴリに属しています。\n"
                    f"関連性があるため、分野適合度は {field_score*100:.0f}% となっています。"
                )
            else:
                field_text += (
                    f"あなたの興味分野とこの研究室の専門分野「{lab_field_name}」は\n"
                    f"直接的な一致はありませんが、他の評価項目での適合性を重視しています。"
                )
            detailed_parts.append(field_text)
        
        # 3. 高優先度項目の詳細分析
        if high_priority:
            hp_text = f"\n◆ あなたが重視する項目の分析"
            
            for item in high_priority[:4]:  # 上位4項目まで
                criterion = item["criterion"]
                name = item["name"]
                student_display = item["student_display"]
                lab_display = item["lab_display"]
                similarity = item["similarity"]
                priority = item["priority"]
                
                # 値の解釈
                student_desc = self._get_value_description(criterion, item["student_val"])
                lab_desc = self._get_value_description(criterion, item["lab_val"])
                
                hp_text += f"\n\n【{name}】優先度 {priority:.0f}/10"
                hp_text += f"\n  あなたの希望: {student_display}/10（{student_desc}）"
                hp_text += f"\n  この研究室: {lab_display}/10（{lab_desc}）"
                
                diff = abs(student_display - lab_display)
                if similarity >= 0.8:
                    hp_text += f"\n  → 類似度 {similarity*100:.0f}%：希望とよく一致しています ✓"
                elif similarity >= 0.6:
                    hp_text += f"\n  → 類似度 {similarity*100:.0f}%：概ね希望に沿っています"
                elif similarity >= 0.4:
                    hp_text += f"\n  → 類似度 {similarity*100:.0f}%：やや差異があります（{diff}ポイント差）"
                else:
                    hp_text += f"\n  → 類似度 {similarity*100:.0f}%：希望と異なる点があります（{diff}ポイント差）"
            
            detailed_parts.append(hp_text)
        
        # 4. 強みと注意点のまとめ
        strengths = [p for p in priority_info if p["similarity"] >= 0.8]
        concerns = [p for p in priority_info if p["similarity"] < 0.5 and p["priority"] >= 5]
        
        summary_text = "\n◆ まとめ"
        
        if strengths:
            strength_names = [p["name"] for p in strengths[:4]]
            summary_text += f"\n【強み】{', '.join(strength_names)}で高い適合性があります。"
        
        if concerns:
            concern_items = []
            for p in concerns[:2]:
                diff = abs(p["student_display"] - p["lab_display"])
                concern_items.append(f"{p['name']}（{diff}ポイント差）")
            summary_text += f"\n【確認推奨】{', '.join(concern_items)}については、実際の雰囲気を確認されることをお勧めします。"
        
        if not concerns:
            summary_text += "\n【注意点】特に大きな懸念点はありません。"
        
        detailed_parts.append(summary_text)
        
        # 5. スコア内訳
        score_text = (
            f"\n◆ スコア内訳\n"
            f"  基本項目適合度: {basic_score*100:.1f}%\n"
            f"  研究分野適合度: {field_score*100:.1f}%\n"
            f"  分野重視度: {lambda_weight*10:.0f}/10\n"
            f"  → 最終適合度 = {(1-lambda_weight)*100:.0f}% × 基本 + {lambda_weight*100:.0f}% × 分野 = {total*100:.1f}%"
        )
        detailed_parts.append(score_text)
        
        detailed = "\n".join(detailed_parts)
        
        # ========================================
        # 短縮版の生成（カード表示用）
        # ========================================
        short_parts = []
        
        # 分野一致
        if field_detail:
            primary_match = field_detail.get("primary_match_type", "unknown")
            if primary_match == "exact":
                field_details = field_detail.get("field_details", [])
                exact_fields = [d for d in field_details if d.get("match_type") == "exact"]
                if exact_fields:
                    field_name = self._get_field_name(exact_fields[0]['field'])
                    short_parts.append(f"興味分野「{field_name}」と完全一致")
        
        # 高適合の高優先度項目
        high_match_high_priority = [
            p for p in high_priority if p["similarity"] >= 0.8
        ][:2]
        if high_match_high_priority:
            items = [p["name"] for p in high_match_high_priority]
            short_parts.append(f"重視する{', '.join(items)}が希望と一致")
        
        # 強み（優先度関係なく高適合）
        if not high_match_high_priority and strengths:
            items = [p["name"] for p in strengths[:2]]
            short_parts.append(f"{', '.join(items)}で高い適合性")
        
        # 注意点（優先度が高く、低適合の項目）
        high_priority_concerns = [p for p in concerns if p["priority"] >= 7]
        if high_priority_concerns:
            item = high_priority_concerns[0]
            diff = abs(item["student_display"] - item["lab_display"])
            short_parts.append(f"※{item['name']}に{diff}ポイント差あり")
        
        short = "。".join(short_parts) if short_parts else legacy
        
        return legacy, detailed, short
    
    def _get_recommendation(self, score: float) -> str:
        if score >= 0.80:
            return "強く推薦"
        elif score >= 0.65:
            return "推薦"
        elif score >= 0.50:
            return "検討推奨"
        elif score >= 0.35:
            return "要検討"
        else:
            return "慎重に検討"
    
    def precompute_lab_memberships(self, lab: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        研究室のメンバーシップ値を事前計算
        
        データベースに保存して再利用可能
        """
        memberships = {}
        for criterion in self.CRITERIA:
            raw_value = lab.get(criterion, 5.0)
            normalized = self._normalize_value(raw_value)
            memberships[criterion] = {
                "raw_value": raw_value,
                "normalized": normalized,
                "3level": MembershipFunctions.fuzzify_3level(normalized),
                "2level": MembershipFunctions.fuzzify_2level(normalized)
            }
        return memberships


# ========================================
# テスト
# ========================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FuzzyMultiPathMatcher v2.2 テスト")
    print("（ボーナス削除、分野重視度計算変更）")
    print("=" * 80)
    
    matcher = FuzzyMultiPathMatcher()
    
    # ========================================
    # テスト1: 具体例（研究強度・チームワーク）
    # ========================================
    print("\n" + "=" * 80)
    print("【テスト1】具体例: 研究強度=8(優先度9), チームワーク=5(優先度6)")
    print("          研究室: 研究強度=6, チームワーク=8, 分野=ai_ml")
    print("          学生興味分野: ai_ml(10), image_processing(7)")
    print("          分野重視度=8")
    print("=" * 80)
    
    # 研究強度とチームワークのみ決定木に含める
    # 他の項目は優先度4以下にして決定木から除外
    student_test1 = {
        "research_intensity": 8.0,
        "team_work": 5.0,
        "research_intensity_priority": 9.0,  # 3分岐
        "team_work_priority": 6.0,           # 2分岐
        # 他の項目は優先度4以下（決定木から除外）
        "advisor_style": 5.0, "advisor_style_priority": 4.0,
        "workload": 5.0, "workload_priority": 4.0,
        "theory_practice": 5.0, "theory_practice_priority": 4.0,
        "skill_development": 5.0, "skill_development_priority": 4.0,
        "lab_atmosphere": 5.0, "lab_atmosphere_priority": 4.0,
        "flexibility": 5.0, "flexibility_priority": 4.0,
        "publication_opportunity": 5.0, "publication_opportunity_priority": 4.0,
        "interdisciplinary": 5.0, "interdisciplinary_priority": 4.0,
        "communication_style": 5.0, "communication_style_priority": 4.0,
        "field_interests": {"ai_ml": 10, "image_processing": 7},
        "field_priority": 8.0  # 分野重視度
    }
    
    lab_test1 = {
        "name": "テスト研究室1",
        "research_intensity": 6.0,
        "team_work": 8.0,
        # 他の項目はデフォルト
        "advisor_style": 5.0,
        "workload": 5.0,
        "theory_practice": 5.0,
        "skill_development": 5.0,
        "lab_atmosphere": 5.0,
        "flexibility": 5.0,
        "publication_opportunity": 5.0,
        "interdisciplinary": 5.0,
        "communication_style": 5.0,
        "field_id": "ai_ml"
    }
    
    result1 = matcher.calculate_compatibility(student_test1, lab_test1)
    
    print(f"\n{'='*60}")
    print(f"【テスト1 結果サマリー】")
    print(f"{'='*60}")
    print(f"S_fuzzy: {result1.fuzzy_score:.4f} ({result1.fuzzy_score*100:.2f}%)")
    print(f"S_gaussian: {result1.gaussian_score:.4f} ({result1.gaussian_score*100:.2f}%)")
    print(f"S_basic: {result1.basic_score:.4f} ({result1.basic_score*100:.2f}%)")
    print(f"S_field: {result1.field_score:.4f} ({result1.field_score*100:.2f}%)")
    print(f"λ (分野比重): {result1.field_weight_lambda:.2f}")
    print(f"1-λ (基本比重): {result1.basic_weight:.2f}")
    print(f"最終適合度: {result1.total_compatibility:.4f} ({result1.total_compatibility*100:.2f}%)")
    print(f"有効パス数: {len(result1.fuzzy_paths)}")
    print(f"推薦: {result1.recommendation}")
    
    # ========================================
    # テスト2: 分野重視度を変えた比較
    # ========================================
    print("\n" + "=" * 80)
    print("【テスト2】分野重視度を変えた比較")
    print("=" * 80)
    
    print(f"\n  S_basic = {result1.basic_score:.4f}, S_field = {result1.field_score:.4f} の場合")
    print(f"\n  {'分野重視度':>10} | {'λ':>6} | {'1-λ':>6} | {'最終適合度':>12}")
    print(f"  {'-'*50}")
    
    for fp in [1, 3, 5, 7, 8, 10]:
        lam = fp / 10.0
        total = (1 - lam) * result1.basic_score + lam * result1.field_score
        print(f"  {fp:>10} | {lam:>6.2f} | {1-lam:>6.2f} | {total:>10.4f} ({total*100:.2f}%)")
    
    # ========================================
    # テスト3: フル機能テスト
    # ========================================
    print("\n" + "=" * 80)
    print("【テスト3】フル機能テスト")
    print("=" * 80)
    
    student_test3 = {
        "research_intensity": 8,
        "advisor_style": 7,
        "team_work": 5,
        "workload": 8,
        "theory_practice": 6,
        "skill_development": 7,
        "lab_atmosphere": 6,
        "flexibility": 5,
        "publication_opportunity": 9,
        "interdisciplinary": 4,
        "communication_style": 6,
        
        "research_intensity_priority": 9,
        "publication_opportunity_priority": 9,
        "workload_priority": 7,
        "skill_development_priority": 6,
        "advisor_style_priority": 5,
        "team_work_priority": 6,
        "theory_practice_priority": 4,
        "lab_atmosphere_priority": 4,
        "flexibility_priority": 4,
        "interdisciplinary_priority": 3,
        "communication_style_priority": 3,
        
        "field_interests": {
            "ai_ml": 10,
            "image_processing": 7,
            "web_design": 3
        },
        "field_priority": 8  # 分野重視度
    }
    
    lab_test3 = {
        "name": "AI研究室",
        "research_intensity": 7,
        "advisor_style": 7,
        "team_work": 6,
        "workload": 8,
        "theory_practice": 6,
        "skill_development": 8,
        "lab_atmosphere": 7,
        "flexibility": 6,
        "publication_opportunity": 9,
        "interdisciplinary": 5,
        "communication_style": 7,
        "field_id": "ai_ml"
    }
    
    result3 = matcher.calculate_compatibility(student_test3, lab_test3)
    
    print(f"\n{'='*60}")
    print(f"【テスト3 結果サマリー】")
    print(f"{'='*60}")
    print(f"S_fuzzy: {result3.fuzzy_score:.4f} ({result3.fuzzy_score*100:.2f}%)")
    print(f"S_gaussian: {result3.gaussian_score:.4f} ({result3.gaussian_score*100:.2f}%)")
    print(f"S_basic: {result3.basic_score:.4f} ({result3.basic_score*100:.2f}%)")
    print(f"S_field: {result3.field_score:.4f} ({result3.field_score*100:.2f}%)")
    print(f"λ (分野比重): {result3.field_weight_lambda:.2f}")
    print(f"1-λ (基本比重): {result3.basic_weight:.2f}")
    print(f"最終適合度: {result3.total_compatibility:.4f} ({result3.total_compatibility*100:.2f}%)")
    print(f"有効パス数: {len(result3.fuzzy_paths)}")
    print(f"推薦: {result3.recommendation}")
    
    # ========================================
    # 事前計算のデモ
    # ========================================
    print(f"\n{'='*60}")
    print(f"【研究室メンバーシップ値の事前計算（DB保存用）】")
    print(f"{'='*60}")
    precomputed = matcher.precompute_lab_memberships(lab_test3)
    for criterion in ["research_intensity", "team_work", "publication_opportunity"]:
        data = precomputed[criterion]
        print(f"\n{criterion}:")
        print(f"  生値: {data['raw_value']}, 正規化: {data['normalized']:.2f}")
        print(f"  3分岐: 低={data['3level']['low']:.4f}, 中={data['3level']['medium']:.4f}, 高={data['3level']['high']:.4f}")
        print(f"  2分岐: 低={data['2level']['low']:.4f}, 高={data['2level']['high']:.4f}")