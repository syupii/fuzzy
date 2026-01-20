# backend/core/matching/fuzzy_multipath_matcher.py
"""
ファジィ決定木マッチャー - 統合版（重み付き平均分野スコア対応）

【基本スコアの計算方法】
====================================
S_basic = γ × S_fuzzy + (1 - γ) × S_gaussian

- S_fuzzy: ファジィ決定木スコア（カテゴリ一致度）
- S_gaussian: ガウス類似度スコア（数値の近さ）
- γ: 統合係数（デフォルト 0.5）

【分野スコアの計算方法】
====================================
S_field = Σ(match_score_i × interest_i) / Σ(interest_i)

- 学生が選択した全ての興味分野について
- 研究室とのマッチ度を計算し
- 興味度で重み付き平均を取る
====================================
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import math

# 説明生成モジュールをインポート
try:
    from .explanation_generator import generate_detailed_explanation, generate_short_explanation
except ImportError:
    try:
        from explanation_generator import generate_detailed_explanation, generate_short_explanation
    except ImportError:
        def generate_detailed_explanation(*args, **kwargs):
            return ""
        def generate_short_explanation(*args, **kwargs):
            return ""


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
    basic_score: float      # 統合基本スコア（S_fuzzy + S_gaussian）
    fuzzy_score: float      # ファジィ決定木スコア（S_fuzzy）
    gaussian_score: float   # ガウス類似度スコア（S_gaussian）
    field_score: float
    field_weight_alpha: float
    basic_weight_beta: float
    
    # 詳細情報
    criteria_scores: Dict[str, float]  # ガウス類似度の項目別スコア
    fuzzy_paths: List[FuzzyPath]
    field_detail: Dict[str, Any]
    
    # 説明
    explanation: str
    explanation_detailed: Optional[str] = ""
    explanation_short: Optional[str] = ""
    
    recommendation: str = ""


class MembershipFunctions:
    """
    メンバーシップ関数
    
    【重要な性質】
    ∀x ∈ [0,1]: μ_low(x) + μ_medium(x) + μ_high(x) = 1
    """
    
    @staticmethod
    def low(x: float) -> float:
        """「低い」への所属度"""
        if x <= 0.3:
            return 1.0
        elif x < 0.5:
            return (0.5 - x) / 0.2
        else:
            return 0.0
    
    @staticmethod
    def medium(x: float) -> float:
        """「中」への所属度"""
        if x <= 0.3 or x >= 0.9:
            return 0.0
        elif x < 0.5:
            return (x - 0.3) / 0.2
        elif x <= 0.7:
            return 1.0
        else:
            return (0.9 - x) / 0.2
    
    @staticmethod
    def high(x: float) -> float:
        """「高い」への所属度"""
        if x <= 0.7:
            return 0.0
        elif x < 0.9:
            return (x - 0.7) / 0.2
        else:
            return 1.0
    
    @staticmethod
    def fuzzify(x: float) -> Dict[str, float]:
        """値をファジィ化（3段階: 低・中・高）"""
        return {
            "low": MembershipFunctions.low(x),
            "medium": MembershipFunctions.medium(x),
            "high": MembershipFunctions.high(x)
        }
    
    @staticmethod
    def fuzzify_2level(x: float) -> Dict[str, float]:
        """値をファジィ化（2段階: 低・高）"""
        if x <= 0.5:
            low_val = 1.0 - x / 0.5
            high_val = x / 0.5
        else:
            low_val = 0.0
            high_val = 1.0
        return {"low": low_val, "high": high_val}


class FuzzyMultiPathMatcher:
    """
    ファジィ決定木マッチャー（統合版 + 重み付き平均分野スコア）
    
    【基本スコアの計算】
    S_basic = γ × S_fuzzy + (1 - γ) × S_gaussian
    
    【分野スコアの計算】
    S_field = Σ(match_score_i × interest_i) / Σ(interest_i)
    """
    
    # 評価基準（12項目）
    CRITERIA = [
        "research_intensity", "advisor_style", "team_work",
        "workload", "theory_practice", "skill_development",
        "lab_atmosphere", "flexibility", "publication_opportunity",
        "interdisciplinary", "communication_style", "research_field_match"
    ]
    
    # ============================================================
    # パラメータ設定
    # ============================================================
    
    # 類似度計算パラメータ
    SIMILARITY_SIGMA = 0.3
    
    # 枝刈り閾値
    PRUNING_THRESHOLD = 0.01
    
    # 優先度閾値
    HIGH_PRIORITY_THRESHOLD = 8.0
    MID_PRIORITY_THRESHOLD = 5.0
    
    # ★★★ 統合係数 ★★★
    FUZZY_GAUSSIAN_GAMMA = 0.5  # γ: S_fuzzyの比重
    
    # 分野マッチングパラメータ
    FIELD_EXACT_MATCH_SCORE = 1.0     # 完全一致時のスコア
    FIELD_CATEGORY_DECAY = 0.7        # カテゴリ一致時の減衰係数
    FIELD_NO_MATCH_SCORE = 0.3        # 不一致時のスコア
    FIELD_EXACT_BONUS = 0.15          # 完全一致ボーナス係数
    FIELD_MISMATCH_PENALTY = 0.15     # 不一致ペナルティ係数
    
    # ============================================================
    
    def __init__(self):
        print("✅ FuzzyMultiPathMatcher 初期化完了（統合版 + 重み付き平均分野スコア）")
        print(f"   - σ = {self.SIMILARITY_SIGMA}")
        print(f"   - γ (S_fuzzy比重) = {self.FUZZY_GAUSSIAN_GAMMA}")
        print(f"   - 高優先度閾値 = {self.HIGH_PRIORITY_THRESHOLD}")
        print(f"   - 中優先度閾値 = {self.MID_PRIORITY_THRESHOLD}")
        print(f"   - 分野完全一致スコア = {self.FIELD_EXACT_MATCH_SCORE}")
        print(f"   - 分野カテゴリ減衰 = {self.FIELD_CATEGORY_DECAY}")
        print(f"   - 分野不一致スコア = {self.FIELD_NO_MATCH_SCORE}")
    
    def calculate_compatibility(
        self,
        student: Dict[str, Any],
        lab: Dict[str, Any]
    ) -> CompatibilityResult:
        """適合度計算（統合版）"""
        
        print(f"\n{'#'*70}")
        print(f"### 適合度計算開始 ###")
        print(f"{'#'*70}")
        
        # ========================================
        # 第1段階: 優先度による決定木の構築
        # ========================================
        print(f"\n{'='*60}")
        print(f"【第1段階】優先度による決定木の構築")
        print(f"{'='*60}")
        
        priorities = self._get_sorted_priorities(student)
        tree_layers = self._build_fuzzy_tree(priorities)
        
        print(f"\n決定木構造:")
        for i, layer in enumerate(tree_layers, 1):
            print(f"  Layer {i}: {layer['criterion']} ({layer['branches']}分岐)")
        
        # ========================================
        # 第2段階: 研究室からパスの所属度を計算
        # ========================================
        print(f"\n{'='*60}")
        print(f"【第2段階】研究室からパスの所属度を計算")
        print(f"{'='*60}")
        
        lab_fuzzified = self._fuzzify_lab(lab, tree_layers)
        fuzzy_paths = self._generate_paths_from_lab(tree_layers, lab_fuzzified)
        fuzzy_paths = self._prune_and_normalize_paths(fuzzy_paths)
        
        print(f"\n有効パス数: {len(fuzzy_paths)}")
        for path in fuzzy_paths:
            path_labels = "-".join([layer[1][0].upper() for layer in path.layers])
            print(f"  パス「{path_labels}」: 所属度 = {path.total_membership:.4f}")
        
        # ========================================
        # 第3段階: 学生からリーフ値を計算
        # ========================================
        print(f"\n{'='*60}")
        print(f"【第3段階】学生からリーフ値を計算")
        print(f"{'='*60}")
        
        student_fuzzified = self._fuzzify_student(student, tree_layers)
        fuzzy_paths = self._calculate_leaf_values(
            fuzzy_paths, student_fuzzified, tree_layers, student
        )
        
        print(f"\nリーフ値計算結果:")
        for path in fuzzy_paths:
            path_labels = "-".join([layer[1][0].upper() for layer in path.layers])
            print(f"  パス「{path_labels}」: 所属度={path.total_membership:.4f}, リーフ値={path.leaf_value:.4f}")
        
        # ========================================
        # 第4段階: S_fuzzy の計算
        # ========================================
        print(f"\n{'='*60}")
        print(f"【第4段階】S_fuzzy の計算")
        print(f"{'='*60}")
        
        fuzzy_score = self._calculate_fuzzy_score(fuzzy_paths)
        print(f"\n★ S_fuzzy = {fuzzy_score:.4f}")
        
        # ========================================
        # ガウス類似度スコア（S_gaussian）の計算
        # ========================================
        print(f"\n{'='*60}")
        print(f"【ガウス類似度スコア】S_gaussian の計算")
        print(f"{'='*60}")
        
        gaussian_score, criteria_scores = self._calculate_gaussian_score(student, lab)
        print(f"\n★ S_gaussian = {gaussian_score:.4f}")
        
        # ========================================
        # ★★★ 基本スコアの統合 ★★★
        # ========================================
        print(f"\n{'='*60}")
        print(f"【基本スコアの統合】S_basic = γ × S_fuzzy + (1-γ) × S_gaussian")
        print(f"{'='*60}")
        
        gamma = self.FUZZY_GAUSSIAN_GAMMA
        basic_score = gamma * fuzzy_score + (1 - gamma) * gaussian_score
        
        print(f"\n  γ = {gamma}")
        print(f"  S_fuzzy = {fuzzy_score:.4f}")
        print(f"  S_gaussian = {gaussian_score:.4f}")
        print(f"  S_basic = {gamma} × {fuzzy_score:.4f} + {1-gamma} × {gaussian_score:.4f}")
        print(f"         = {gamma * fuzzy_score:.4f} + {(1-gamma) * gaussian_score:.4f}")
        print(f"         = {basic_score:.4f}")
        print(f"\n★ S_basic = {basic_score:.4f}")
        
        # ========================================
        # ★★★ 分野マッチング（重み付き平均） ★★★
        # ========================================
        print(f"\n{'='*60}")
        print(f"【分野マッチング】重み付き平均方式")
        print(f"{'='*60}")
        
        field_interests = student.get("field_interests", {})
        lab_field = lab.get("field_id", "")
        field_score, field_detail = self._calculate_field_match_weighted_average(
            field_interests, lab_field
        )
        print(f"\n★ 分野スコア = {field_score:.4f}")
        
        # ========================================
        # 最終スコア統合
        # ========================================
        print(f"\n{'='*60}")
        print(f"【最終スコア統合】")
        print(f"{'='*60}")
        
        rfm = student.get("research_field_match", 5.0)
        
        if rfm >= 5.0:
            alpha = 0.5 + (rfm - 5.0) / 5.0 * 0.4
        else:
            alpha = 0.2 + rfm / 5.0 * 0.3
        beta = 1.0 - alpha
        
        print(f"\n  alpha (分野比重) = {alpha:.4f}")
        print(f"  beta (基本比重) = {beta:.4f}")
        print(f"\n  total = beta × S_basic + alpha × field_score")
        print(f"        = {beta:.4f} × {basic_score:.4f} + {alpha:.4f} × {field_score:.4f}")
        
        total = beta * basic_score + alpha * field_score
        print(f"        = {total:.4f}")
        
        # ボーナス/ペナルティ
        match_type = field_detail.get("primary_match_type", "unknown")
        if match_type == "exact":
            bonus = self.FIELD_EXACT_BONUS * alpha
            total += bonus
            print(f"\n  完全一致ボーナス: +{bonus:.4f}")
            field_detail["bonus_applied"] = True
            field_detail["bonus_value"] = bonus
        elif match_type == "no_match":
            penalty = self.FIELD_MISMATCH_PENALTY * alpha
            total -= penalty
            print(f"\n  不一致ペナルティ: -{penalty:.4f}")
            field_detail["penalty_applied"] = True
            field_detail["penalty_value"] = penalty
        
        total = max(0.0, min(1.0, total))
        print(f"\n★★★ 最終適合度 = {total:.4f} ★★★")
        
        # 説明生成
        explanation = self._generate_explanation(
            total, basic_score, field_score, alpha, beta,
            field_detail, len(fuzzy_paths),
            student=student, lab=lab, criteria_scores=criteria_scores
        )
        
        explanation_detailed = generate_detailed_explanation(
            lab=lab, student=student, criteria_scores=criteria_scores,
            field_score=field_score, field_detail=field_detail,
            final_score=total, alpha=alpha, beta=beta
        )
        explanation_short = generate_short_explanation(
            criteria_scores=criteria_scores, field_detail=field_detail,
            final_score=total, student=student
        )
        
        recommendation = self._get_recommendation(total)
        
        return CompatibilityResult(
            total_compatibility=total,
            basic_score=basic_score,
            fuzzy_score=fuzzy_score,
            gaussian_score=gaussian_score,
            field_score=field_score,
            field_weight_alpha=alpha,
            basic_weight_beta=beta,
            criteria_scores=criteria_scores,
            fuzzy_paths=fuzzy_paths,
            field_detail=field_detail,
            explanation=explanation,
            explanation_detailed=explanation_detailed,
            explanation_short=explanation_short,
            recommendation=recommendation
        )
    
    # ========================================
    # 第1段階: 優先度による決定木の構築
    # ========================================
    
    def _get_sorted_priorities(
        self, 
        student: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """優先度ソート"""
        priorities = []
        
        for criterion in self.CRITERIA:
            priority_key = f"{criterion}_priority"
            priority = student.get(priority_key, 5.0)
            priorities.append({
                "criterion": criterion,
                "priority": priority
            })
        
        priorities.sort(key=lambda x: x["priority"], reverse=True)
        return priorities
    
    def _build_fuzzy_tree(
        self, 
        priorities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        適応的決定木構築
        
        優先度 ≥ 8: 3分岐（低・中・高）
        優先度 5〜7: 2分岐（低・高）
        優先度 < 5: 木に含めない
        """
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
    # 第2段階: 研究室からパスの所属度を計算
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
            labels = layer["labels"]
            lab_value = self._normalize_value(lab.get(criterion, 5.0))
            
            if len(labels) == 3:
                memberships = MembershipFunctions.fuzzify(lab_value)
            else:
                memberships = MembershipFunctions.fuzzify_2level(lab_value)
            
            lab_fuzzified[criterion] = memberships
        
        return lab_fuzzified
    
    def _generate_paths_from_lab(
        self,
        tree_layers: List[Dict[str, Any]],
        lab_fuzzified: Dict[str, Dict[str, float]]
    ) -> List[FuzzyPath]:
        """研究室のファジィ化結果からパスを生成"""
        if not tree_layers:
            return [FuzzyPath(
                path_id=0,
                layers=[],
                total_membership=1.0,
                leaf_value=0.0
            )]
        
        all_paths = []
        self._generate_paths_recursive(
            tree_layers, lab_fuzzified, 0, [], 1.0, all_paths
        )
        
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
        """パスを再帰的に生成"""
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
    
    def _prune_and_normalize_paths(
        self, 
        paths: List[FuzzyPath]
    ) -> List[FuzzyPath]:
        """パスの枝刈りと正規化"""
        paths = [p for p in paths if p.total_membership >= self.PRUNING_THRESHOLD]
        
        if not paths:
            return paths
        
        total_membership = sum(p.total_membership for p in paths)
        
        if total_membership > 0:
            for path in paths:
                path.total_membership = path.total_membership / total_membership
        
        return paths
    
    # ========================================
    # 第3段階: 学生からリーフ値を計算
    # ========================================
    
    def _fuzzify_student(
        self,
        student: Dict[str, Any],
        tree_layers: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """学生の各項目をファジィ化"""
        student_fuzzified = {}
        
        for layer in tree_layers:
            criterion = layer["criterion"]
            labels = layer["labels"]
            student_value = self._normalize_value(student.get(criterion, 5.0))
            
            if len(labels) == 3:
                memberships = MembershipFunctions.fuzzify(student_value)
            else:
                memberships = MembershipFunctions.fuzzify_2level(student_value)
            
            student_fuzzified[criterion] = memberships
        
        return student_fuzzified
    
    def _calculate_leaf_values(
        self,
        paths: List[FuzzyPath],
        student_fuzzified: Dict[str, Dict[str, float]],
        tree_layers: List[Dict[str, Any]],
        student: Dict[str, Any]
    ) -> List[FuzzyPath]:
        """
        各パスのリーフ値を計算
        
        リーフ値 = Σ(学生のμ_label × 重み) / Σ(重み)
        """
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
            
            if total_weight > 0:
                path.leaf_value = weighted_sum / total_weight
            else:
                path.leaf_value = 0.5
        
        return paths
    
    # ========================================
    # 第4段階: S_fuzzy の計算
    # ========================================
    
    def _calculate_fuzzy_score(
        self,
        paths: List[FuzzyPath]
    ) -> float:
        """
        S_fuzzy = Σ（所属度 × リーフ値）
        """
        if not paths:
            return 0.5
        
        return sum(path.total_membership * path.leaf_value for path in paths)
    
    # ========================================
    # ガウス類似度スコア
    # ========================================
    
    def _calculate_gaussian_score(
        self,
        student: Dict[str, Any],
        lab: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """
        ガウス類似度による基本スコア
        
        S_gaussian = Σ(similarity_i × weight_i) / Σ(weight_i)
        """
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
        
        gaussian_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        return gaussian_score, criteria_scores
    
    def _calculate_gaussian_similarity(
        self,
        student_val: float,
        lab_val: float
    ) -> float:
        """
        ガウス類似度
        
        Similarity = exp(-(d²)/(2σ²))
        """
        d = abs(student_val - lab_val)
        sigma = self.SIMILARITY_SIGMA
        return math.exp(-(d ** 2) / (2 * sigma ** 2))
    
    def _calculate_priority_weight(
        self,
        priority: float
    ) -> float:
        """
        優先度から重みを計算
        
        weight = (priority / 10)^1.5
        """
        normalized = priority / 10.0
        return normalized ** 1.5
    
    # ========================================
    # ★★★ 分野マッチング（重み付き平均方式） ★★★
    # ========================================
    
    def _calculate_field_match_weighted_average(
        self,
        field_interests: Dict[str, float],
        lab_field: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        分野マッチング（重み付き平均方式）
        
        S_field = Σ(match_score_i × interest_i) / Σ(interest_i)
        
        各興味分野について研究室とのマッチ度を計算し、
        興味度で重み付き平均を取る
        """
        
        print(f"\n  【分野マッチング詳細】")
        print(f"  研究室の分野: {lab_field}")
        print(f"  学生の興味分野: {field_interests}")
        
        # 興味分野がない場合
        if not field_interests:
            print(f"  → 興味分野が指定されていないため、デフォルトスコア 0.5")
            return 0.5, {
                "match_type": "unknown",
                "primary_match_type": "unknown",
                "lab_field": lab_field,
                "message": "興味分野が指定されていません",
                "field_details": []
            }
        
        # 研究室の分野がない場合
        if not lab_field:
            print(f"  → 研究室の分野が指定されていないため、デフォルトスコア 0.5")
            return 0.5, {
                "match_type": "unknown",
                "primary_match_type": "unknown",
                "lab_field": lab_field,
                "message": "研究室の分野が指定されていません",
                "field_details": []
            }
        
        # 各興味分野についてマッチスコアを計算
        weighted_sum = 0.0
        total_weight = 0.0
        field_details = []
        primary_match_type = "no_match"
        has_exact_match = False
        
        print(f"\n  ┌{'─'*70}┐")
        print(f"  │ {'興味分野':<20} │ {'興味度':>8} │ {'マッチタイプ':<15} │ {'スコア':>8} │")
        print(f"  ├{'─'*70}┤")
        
        for interest_field, interest_level in field_interests.items():
            # マッチタイプとスコアを決定
            if interest_field == lab_field:
                # 完全一致
                match_type = "exact"
                match_score = self.FIELD_EXACT_MATCH_SCORE
                has_exact_match = True
            elif self._is_same_category(interest_field, lab_field):
                # カテゴリ一致
                match_type = "category"
                match_score = self.FIELD_CATEGORY_DECAY
            else:
                # 不一致
                match_type = "no_match"
                match_score = self.FIELD_NO_MATCH_SCORE
            
            # 重み付き加算
            weighted_sum += match_score * interest_level
            total_weight += interest_level
            
            # 詳細を記録
            field_details.append({
                "field": interest_field,
                "interest_level": interest_level,
                "match_type": match_type,
                "match_score": match_score,
                "contribution": match_score * interest_level
            })
            
            print(f"  │ {interest_field:<20} │ {interest_level:>8.1f} │ {match_type:<15} │ {match_score:>8.4f} │")
        
        print(f"  └{'─'*70}┘")
        
        # 重み付き平均を計算
        if total_weight > 0:
            field_score = weighted_sum / total_weight
        else:
            field_score = 0.5
        
        # 主要なマッチタイプを決定（ボーナス/ペナルティ判定用）
        if has_exact_match:
            primary_match_type = "exact"
        elif any(d["match_type"] == "category" for d in field_details):
            primary_match_type = "category"
        else:
            primary_match_type = "no_match"
        
        print(f"\n  【計算過程】")
        print(f"  S_field = Σ(match_score × interest) / Σ(interest)")
        print(f"         = {weighted_sum:.4f} / {total_weight:.4f}")
        print(f"         = {field_score:.4f}")
        print(f"\n  主要マッチタイプ: {primary_match_type}")
        
        # 詳細情報を構築
        detail = {
            "match_type": "weighted_average",
            "primary_match_type": primary_match_type,
            "lab_field": lab_field,
            "weighted_sum": weighted_sum,
            "total_weight": total_weight,
            "field_details": field_details,
            "message": self._generate_field_match_message(primary_match_type, field_details)
        }
        
        return field_score, detail
    
    def _generate_field_match_message(
        self,
        primary_match_type: str,
        field_details: List[Dict[str, Any]]
    ) -> str:
        """分野マッチングの説明メッセージを生成"""
        exact_fields = [d for d in field_details if d["match_type"] == "exact"]
        category_fields = [d for d in field_details if d["match_type"] == "category"]
        
        if exact_fields:
            field_name = exact_fields[0]["field"]
            interest = exact_fields[0]["interest_level"]
            return f"興味分野「{field_name}」と完全一致（興味度: {interest}/10）"
        elif category_fields:
            field_name = category_fields[0]["field"]
            return f"興味分野「{field_name}」と同じカテゴリ"
        else:
            return "興味分野との一致なし"
    
    def _is_same_category(
        self,
        field1: str,
        field2: str
    ) -> bool:
        """2つの分野が同じカテゴリか判定"""
        # カテゴリ定義
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
        
        # 各分野のカテゴリを検索
        category1 = None
        category2 = None
        
        for category, fields in FIELD_CATEGORIES.items():
            if field1 in fields:
                category1 = category
            if field2 in fields:
                category2 = category
        
        # 同じカテゴリかどうか
        if category1 and category2 and category1 == category2:
            return True
        
        return False
    
    # ========================================
    # ユーティリティ
    # ========================================
    
    def _normalize_value(
        self,
        value: float
    ) -> float:
        """値を0-1に正規化"""
        if value > 1.0:
            return (value - 1.0) / 9.0
        return value
    
    def _generate_explanation(
        self,
        total: float,
        basic_score: float,
        field_score: float,
        alpha: float,
        beta: float,
        field_detail: Dict,
        num_paths: int,
        student: Dict[str, Any] = None,
        lab: Dict[str, Any] = None,
        criteria_scores: Dict[str, float] = None
    ) -> str:
        """説明文を生成"""
        if student is None or lab is None or criteria_scores is None:
            if total >= 0.8:
                return "非常に高い適合性があります。"
            elif total >= 0.6:
                return "条件によく合致しています。"
            else:
                return "検討の余地があります。"

        priorities = self._get_sorted_priorities(student)
        top_priorities = [p for p in priorities if p['priority'] >= 8]
        
        matched_points = []
        compromise_points = []
        
        criteria_labels = {
            "research_intensity": "研究強度",
            "advisor_style": "指導方針",
            "team_work": "チームワーク",
            "workload": "活動量",
            "theory_practice": "理論・実践",
            "skill_development": "スキル習得",
            "lab_atmosphere": "雰囲気",
            "flexibility": "柔軟性",
            "publication_opportunity": "発表機会",
            "interdisciplinary": "学際性",
            "communication_style": "交流頻度"
        }

        for p in top_priorities:
            key = p['criterion']
            if key == "research_field_match":
                continue
            score = criteria_scores.get(key, 0.0)
            label = criteria_labels.get(key, key)
            if score >= 0.75:
                matched_points.append(label)
            elif score <= 0.45:
                compromise_points.append(label)

        match_type = field_detail.get("primary_match_type", "unknown")
        is_field_prioritized = student.get("research_field_match", 5) >= 7

        sentences = []

        if is_field_prioritized and match_type == "exact":
            sentences.append("あなたが重視する研究分野が完全一致しており、専門性を高めるのに最適な環境です。")
        elif matched_points:
            points_str = "』『".join(matched_points[:2])
            sentences.append(f"あなたが最優先した『{points_str}』の方針が、この研究室の実態と強く合致しています。")
        elif total >= 0.85:
            sentences.append("全体的なバランスが極めて良く、あなたの希望条件を高い水準で満たしています。")

        if match_type == "exact" and not is_field_prioritized:
            sentences.append("研究分野も一致しているため、スムーズに研究に着手できるでしょう。")
        elif match_type != "exact" and basic_score >= 0.75:
            sentences.append("分野は異なりますが、ゼミの雰囲気や指導方針といった「活動スタイル」の相性が抜群です。")
        
        if compromise_points:
            point = compromise_points[0]
            sentences.append(f"『{point}』に関しては希望と少し差がありますが、総合的な適合度は十分に高いです。")

        if not sentences:
            if total >= 0.7:
                sentences.append("大きな欠点がなく、安定して研究に取り組める環境と言えます。")
            else:
                sentences.append("一部の条件で妥協が必要ですが、検討候補として有力です。")

        return "".join(sentences)
    
    def _get_recommendation(
        self,
        score: float
    ) -> str:
        """推薦レベル"""
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


# ========================================
# テスト
# ========================================
if __name__ == "__main__":
    print("=" * 80)
    print("FuzzyMultiPathMatcher テスト（重み付き平均分野スコア対応版）")
    print("=" * 80)
    
    matcher = FuzzyMultiPathMatcher()
    
    # テスト用学生プロファイル（複数の興味分野）
    student = {
        "research_intensity": 8,
        "advisor_style": 7,
        "team_work": 5,
        "workload": 8,
        "theory_practice": 6,
        "research_field_match": 8,
        "skill_development": 7,
        "lab_atmosphere": 6,
        "flexibility": 5,
        "publication_opportunity": 9,
        "interdisciplinary": 4,
        "communication_style": 6,
        
        # 優先度
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
        "research_field_match_priority": 8,
        
        # ★★★ 複数の興味分野 ★★★
        "field_interests": {
            "ai_ml": 10,
            "image_processing": 7,
            "web_design": 3
        }
    }
    
    # テスト用研究室プロファイル
    lab = {
        "name": "AI研究室",
        "research_area": "人工知能・機械学習",
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
    
    result = matcher.calculate_compatibility(student, lab)
    
    print(f"\n{'='*80}")
    print(f"【計算結果サマリー】")
    print(f"{'='*80}")
    print(f"総合適合度: {result.total_compatibility:.4f}")
    print(f"基本スコア（統合）: {result.basic_score:.4f}")
    print(f"  - S_fuzzy: {result.fuzzy_score:.4f}")
    print(f"  - S_gaussian: {result.gaussian_score:.4f}")
    print(f"分野スコア（重み付き平均）: {result.field_score:.4f}")
    print(f"分野比重α: {result.field_weight_alpha:.4f}")
    print(f"基本比重β: {result.basic_weight_beta:.4f}")
    print(f"評価パス数: {len(result.fuzzy_paths)}")
    print(f"推薦: {result.recommendation}")
    print(f"\n【分野マッチング詳細】")
    for detail in result.field_detail.get("field_details", []):
        print(f"  {detail['field']}: 興味度={detail['interest_level']}, マッチ={detail['match_type']}, スコア={detail['match_score']:.2f}")
    print(f"\n【説明】")
    print(result.explanation)
    
    print(f"\n{'='*80}")
    print(f"【分野スコアの計算過程（再確認）】")
    print(f"{'='*80}")
    print(f"""
  学生の興味分野:
    ai_ml: 興味度 10 → 完全一致 → スコア 1.0
    image_processing: 興味度 7 → カテゴリ一致 → スコア 0.7
    web_design: 興味度 3 → カテゴリ一致 → スコア 0.7

  重み付き平均:
    S_field = (1.0×10 + 0.7×7 + 0.7×3) / (10 + 7 + 3)
            = (10.0 + 4.9 + 2.1) / 20
            = 17.0 / 20
            = 0.85
""")