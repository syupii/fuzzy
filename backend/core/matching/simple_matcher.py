# backend/core/matching/simple_matcher.py - 改善版

import numpy as np
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# default_params からのインポートを試行
try:
    from config.default_params import (
        DEFAULT_PARAMS, BASIC_CRITERIA, FIELD_CATEGORIES,
        get_field_category, get_field_name, is_same_category
    )
    PARAMS_AVAILABLE = True
except ImportError:
    PARAMS_AVAILABLE = False
    # フォールバック定義
    BASIC_CRITERIA = [
        "research_intensity", "advisor_style", "team_work", 
        "workload", "theory_practice", "skill_development",
        "lab_atmosphere", "flexibility", "publication_opportunity",
        "interdisciplinary", "communication_style", "research_field_match"
    ]
    
    class FallbackParams:
        default_weights = {c: 1.0 for c in BASIC_CRITERIA}
        similarity_sigma = 0.25  # ★ 0.2 → 0.25 に変更（より緩やか）
        field_exact_match_bonus = 1.0
        field_category_match_ratio = 0.6
        field_no_match_penalty = 0.3
    
    DEFAULT_PARAMS = FallbackParams()
    
    def get_field_category(field_id): return "unknown"
    def get_field_name(field_id): return field_id
    def is_same_category(f1, f2): return False


@dataclass
class CompatibilityResult:
    """適合度計算結果"""
    total_compatibility: float
    basic_score: float
    field_score: float
    field_weight_alpha: float
    basic_weight_beta: float
    criteria_scores: Dict[str, float]
    field_detail: Dict[str, Any]
    tree_path: str
    fuzzy_paths: List[Dict[str, Any]]
    tree_layers: List[str]
    leaf_criteria: List[str]
    explanation: str
    recommendation: str


class SimpleMatcher:
    """
    シンプルマッチャー（12項目対応）- 改善版
    
    主な改善点：
    1. σ値を0.25に調整（より緩やかな減衰）
    2. 完全一致・近似一致にボーナス追加
    3. 優先度の非線形重み付け強化
    4. スコアレンジの拡大
    """
    
    def __init__(self):
        self.params = DEFAULT_PARAMS
        self.criteria = BASIC_CRITERIA
        
        # ★ 改善: σ値を調整可能に
        self.similarity_sigma = 0.25  # デフォルトより緩やか
        
        print("✅ SimpleMatcher 初期化完了（改善版）")
        print(f"   - 評価項目: {len(self.criteria)}項目")
        print(f"   - 類似度σ: {self.similarity_sigma}")
    
    def calculate_compatibility(
        self,
        student: Dict[str, Any],
        lab: Dict[str, Any]
    ) -> CompatibilityResult:
        """適合度計算（改善版）"""
        
        # 優先度取得
        priorities = self._get_sorted_priorities(student)
        
        # ★ 改善: 基本12項目スコア計算（優先度重み付け強化版）
        basic_score, criteria_scores = self._calculate_basic_match_improved(
            student, lab, priorities
        )
        
        # 分野マッチング
        field_interests = student.get("field_interests", {})
        lab_field = lab.get("field_id", "")
        field_score, field_detail = self._calculate_field_match(
            field_interests, lab_field
        )
        
        # research_field_match による統合
        rfm = student.get("research_field_match", 5.0)
        alpha = rfm / 10.0  # 分野の比重
        beta = 1.0 - alpha  # 基本項目の比重
        
        # ★ 改善: スコア統合に高優先度項目のボーナスを追加
        priority_bonus = self._calculate_priority_bonus(
            priorities, criteria_scores, threshold=7.0
        )
        
        # 基本統合
        total = beta * basic_score + alpha * field_score
        
        # 優先度ボーナスを加算（最大+0.15）
        total = min(1.0, total + priority_bonus)
        
        # 決定木情報生成
        tree_path, fuzzy_paths, tree_layers, leaf_criteria = self._generate_tree_info(
            priorities, student, lab
        )
        
        # 説明文生成
        explanation = self._generate_explanation_improved(
            total, basic_score, field_score, alpha, beta, 
            field_detail, priority_bonus, priorities
        )
        
        # 推薦レベル
        recommendation = self._get_recommendation(total)
        
        return CompatibilityResult(
            total_compatibility=total,
            basic_score=basic_score,
            field_score=field_score,
            field_weight_alpha=alpha,
            basic_weight_beta=beta,
            criteria_scores=criteria_scores,
            field_detail=field_detail,
            tree_path=tree_path,
            fuzzy_paths=fuzzy_paths,
            tree_layers=tree_layers,
            leaf_criteria=leaf_criteria,
            explanation=explanation,
            recommendation=recommendation
        )
    
    def _get_sorted_priorities(self, student: Dict[str, Any]) -> List[Dict[str, Any]]:
        """優先度付きリスト作成"""
        priorities = []
        
        for criterion in self.criteria:
            priority_key = f"{criterion}_priority"
            priority = student.get(priority_key, 5.0)
            priorities.append({
                "criterion": criterion,
                "priority": priority
            })
        
        # 優先度順にソート
        priorities.sort(key=lambda x: x["priority"], reverse=True)
        return priorities
    
    def _calculate_basic_match_improved(
        self, 
        student: Dict[str, Any], 
        lab: Dict[str, Any],
        priorities: List[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, float]]:
        """
        基本12項目のマッチング計算（改善版）
        
        改善点:
        1. ガウシアン関数のσを大きく（0.25）
        2. 完全一致・近似一致にボーナス
        3. 優先度の非線形重み付け
        """
        
        weighted_sum = 0.0
        total_weight = 0.0
        criteria_scores = {}
        
        for item in priorities:
            criterion = item["criterion"]
            priority = item["priority"]
            
            # 値の取得と正規化
            student_val = self._normalize_value(student.get(criterion, 5.0))
            lab_val = self._normalize_value(lab.get(criterion, 5.0))
            
            # ★ 改善: 類似度計算（σ=0.25、ボーナス付き）
            similarity = self._calculate_similarity_improved(student_val, lab_val)
            
            # ★ 改善: 優先度から重みを計算（非線形変換）
            weight = self._calculate_priority_weight(priority)
            
            criteria_scores[criterion] = similarity
            weighted_sum += similarity * weight
            total_weight += weight
        
        # 基本スコア
        basic_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        return basic_score, criteria_scores
    
    def _normalize_value(self, value: Any) -> float:
        """値を0-1に正規化"""
        try:
            val = float(value)
            if val > 1.0:
                # 1-10 → 0-1
                return (val - 1.0) / 9.0
            return val
        except (ValueError, TypeError):
            return 0.5
    
    def _calculate_similarity_improved(
        self, 
        student_val: float, 
        lab_val: float
    ) -> float:
        """
        改善版類似度計算
        
        改善点:
        - σ = 0.25（より緩やか）
        - 完全一致（diff < 0.05）で +15% ボーナス
        - 近似一致（diff < 0.15）で +8% ボーナス
        """
        
        difference = abs(student_val - lab_val)
        
        # ガウシアン類似度（改善版：σ=0.25）
        sigma = self.similarity_sigma
        base_similarity = np.exp(-(difference ** 2) / (2 * sigma ** 2))
        
        # ★ ボーナス追加
        if difference < 0.05:
            # 完全一致: +15%
            similarity = min(1.0, base_similarity * 1.15)
        elif difference < 0.15:
            # 近似一致: +8%
            similarity = min(1.0, base_similarity * 1.08)
        else:
            similarity = base_similarity
        
        return float(similarity)
    
    def _calculate_priority_weight(self, priority: float) -> float:
        """
        優先度から重みを計算（非線形変換）
        
        改善点:
        - 指数関数的な重み付け（priority^1.8）
        - 高優先度をより強調
        
        priority 10 → weight 1.00
        priority  8 → weight 0.72
        priority  5 → weight 0.32
        priority  3 → weight 0.11
        priority  1 → weight 0.01
        """
        
        normalized = priority / 10.0
        weight = normalized ** 1.8  # ★ 指数を1.5→1.8に強化
        return weight
    
    def _calculate_priority_bonus(
        self,
        priorities: List[Dict[str, Any]],
        criteria_scores: Dict[str, float],
        threshold: float = 7.0
    ) -> float:
        """
        高優先度項目での適合度に基づくボーナス計算
        
        高優先度項目（priority >= 7）での適合度が高い場合にボーナス
        最大 +0.15（15%）
        """
        
        high_priority_items = [
            item for item in priorities 
            if item["priority"] >= threshold
        ]
        
        if not high_priority_items:
            return 0.0
        
        # 高優先度項目の平均適合度
        high_priority_scores = [
            criteria_scores.get(item["criterion"], 0.5)
            for item in high_priority_items
        ]
        
        avg_high_score = np.mean(high_priority_scores)
        
        # ボーナス計算
        if avg_high_score >= 0.85:
            return 0.15  # 非常に高い適合
        elif avg_high_score >= 0.75:
            return 0.10  # 高い適合
        elif avg_high_score >= 0.65:
            return 0.05  # やや高い適合
        else:
            return 0.0
    
    def _calculate_field_match(
        self,
        field_interests: Dict[str, float],
        lab_field: str
    ) -> Tuple[float, Dict[str, Any]]:
        """分野マッチングスコア計算"""
        
        if not field_interests or not lab_field:
            return 0.5, {
                "match_type": "unknown",
                "lab_field": lab_field,
                "message": "分野情報が不足しています"
            }
        
        # 完全一致チェック
        if lab_field in field_interests:
            interest_level = field_interests[lab_field]
            score = interest_level / 10.0
            
            return score, {
                "match_type": "exact",
                "lab_field": lab_field,
                "lab_field_name": get_field_name(lab_field),
                "interest_level": interest_level,
                "message": f"{get_field_name(lab_field)}と完全一致（興味度: {interest_level}/10）"
            }
        
        # カテゴリ一致チェック
        lab_category = get_field_category(lab_field)
        
        for field_id, interest_level in field_interests.items():
            if is_same_category(lab_field, field_id):
                score = (interest_level / 10.0) * 0.6  # カテゴリ一致は60%
                
                return score, {
                    "match_type": "category",
                    "lab_field": lab_field,
                    "lab_field_name": get_field_name(lab_field),
                    "interest_level": interest_level,
                    "message": f"関連分野に興味あり（{get_field_name(field_id)}）"
                }
        
        # 一致なし
        return 0.3, {
            "match_type": "none",
            "lab_field": lab_field,
            "lab_field_name": get_field_name(lab_field),
            "message": "興味分野と一致しません"
        }
    
    def _generate_tree_info(
        self,
        priorities: List[Dict[str, Any]],
        student: Dict[str, Any],
        lab: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]], List[str], List[str]]:
        """決定木情報生成"""
        
        # 簡易的な決定木パス生成
        top_criteria = [p["criterion"] for p in priorities[:3]]
        
        tree_path = " -> ".join([
            f"{c}={student.get(c, 5):.1f}" 
            for c in top_criteria
        ])
        
        fuzzy_paths = [
            {
                "criterion": p["criterion"],
                "student_value": student.get(p["criterion"], 5),
                "priority": p["priority"]
            }
            for p in priorities[:5]
        ]
        
        tree_layers = [f"Layer{i+1}: {p['criterion']}" for i, p in enumerate(priorities[:3])]
        leaf_criteria = top_criteria
        
        return tree_path, fuzzy_paths, tree_layers, leaf_criteria
    
    def _generate_explanation_improved(
        self,
        total: float,
        basic: float,
        field: float,
        alpha: float,
        beta: float,
        field_detail: Dict[str, Any],
        priority_bonus: float,
        priorities: List[Dict[str, Any]]
    ) -> str:
        """改善版説明文生成"""
        
        parts = []
        
        # スコアレベル
        if total >= 0.8:
            parts.append("非常に高い適合性")
        elif total >= 0.65:
            parts.append("高い適合性")
        elif total >= 0.5:
            parts.append("中程度の適合性")
        elif total >= 0.35:
            parts.append("やや低い適合性")
        else:
            parts.append("低い適合性")
        
        # 分野情報
        match_type = field_detail.get("match_type", "unknown")
        if match_type == "exact":
            parts.append("興味分野と完全一致")
        elif match_type == "category":
            parts.append("関連分野に興味")
        
        # 基本項目
        if basic >= 0.75:
            parts.append("研究スタイルが合致")
        
        # 優先度ボーナス
        if priority_bonus > 0.05:
            high_priority_items = [p for p in priorities if p["priority"] >= 7.0]
            parts.append(f"重視項目で高適合（{len(high_priority_items)}項目）")
        
        return " / ".join(parts)
    
    def _get_recommendation(self, score: float) -> str:
        """推薦レベル（改善版：閾値調整）"""
        if score >= 0.80:
            return "強く推薦"
        elif score >= 0.65:
            return "推薦"
        elif score >= 0.45:
            return "検討推奨"
        elif score >= 0.30:
            return "要検討"
        else:
            return "慎重に検討"


# テスト用
if __name__ == "__main__":
    print("=" * 60)
    print("SimpleMatcher テスト（改善版）")
    print("=" * 60)
    
    matcher = SimpleMatcher()
    
    # 高適合ケース
    student_high = {
        "research_intensity": 8,
        "advisor_style": 7,
        "team_work": 7,
        "workload": 7,
        "theory_practice": 6,
        "research_field_match": 8,  # 分野やや重視
        "skill_development": 8,
        "lab_atmosphere": 7,
        "flexibility": 6,
        "publication_opportunity": 7,
        "interdisciplinary": 6,
        "communication_style": 7,
        
        # 優先度設定
        "research_intensity_priority": 9,
        "advisor_style_priority": 8,
        "skill_development_priority": 9,
        
        "field_interests": {"ai_ml": 9}
    }
    
    lab_high = {
        "research_intensity": 8,
        "advisor_style": 7,
        "team_work": 7,
        "workload": 6,
        "theory_practice": 7,
        "skill_development": 8,
        "lab_atmosphere": 7,
        "flexibility": 7,
        "publication_opportunity": 8,
        "interdisciplinary": 6,
        "communication_style": 7,
        "field_id": "ai_ml"
    }
    
    # 中適合ケース
    lab_medium = {
        "research_intensity": 6,
        "advisor_style": 5,
        "team_work": 6,
        "workload": 7,
        "theory_practice": 5,
        "skill_development": 6,
        "lab_atmosphere": 6,
        "flexibility": 6,
        "publication_opportunity": 5,
        "interdisciplinary": 5,
        "communication_style": 6,
        "field_id": "network_security"
    }
    
    # 低適合ケース
    lab_low = {
        "research_intensity": 3,
        "advisor_style": 3,
        "team_work": 4,
        "workload": 8,
        "theory_practice": 3,
        "skill_development": 4,
        "lab_atmosphere": 4,
        "flexibility": 3,
        "publication_opportunity": 3,
        "interdisciplinary": 3,
        "communication_style": 4,
        "field_id": "web_design"
    }
    
    print("\n【高適合ケース】")
    result = matcher.calculate_compatibility(student_high, lab_high)
    print(f"  総合スコア: {result.total_compatibility:.3f}")
    print(f"  基本スコア: {result.basic_score:.3f}")
    print(f"  分野スコア: {result.field_score:.3f}")
    print(f"  推薦: {result.recommendation}")
    print(f"  説明: {result.explanation}")
    
    print("\n【中適合ケース】")
    result = matcher.calculate_compatibility(student_high, lab_medium)
    print(f"  総合スコア: {result.total_compatibility:.3f}")
    print(f"  基本スコア: {result.basic_score:.3f}")
    print(f"  分野スコア: {result.field_score:.3f}")
    print(f"  推薦: {result.recommendation}")
    print(f"  説明: {result.explanation}")
    
    print("\n【低適合ケース】")
    result = matcher.calculate_compatibility(student_high, lab_low)
    print(f"  総合スコア: {result.total_compatibility:.3f}")
    print(f"  基本スコア: {result.basic_score:.3f}")
    print(f"  分野スコア: {result.field_score:.3f}")
    print(f"  推薦: {result.recommendation}")
    print(f"  説明: {result.explanation}")
    
    print("\n" + "=" * 60)
    print("✅ テスト完了")