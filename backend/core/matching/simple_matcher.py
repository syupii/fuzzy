# core/matching/simple_matcher.py
"""
パターンA: シンプルマッチャー
デフォルトパラメータ + 動的決定木 + 分野マッチング
遺伝的アルゴリズムなし
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

from config.default_params import (
    DEFAULT_PARAMS, BASIC_CRITERIA, FIELD_CATEGORIES,
    get_field_category, get_field_name, is_same_category
)


@dataclass
class CompatibilityResult:
    """適合度計算結果"""
    total_compatibility: float          # 総合適合度 (0-1)
    basic_score: float                  # 基本12項目スコア (0-1)
    field_score: float                  # 分野スコア (0-1)
    field_weight_alpha: float           # 分野の比重 (0-1)
    basic_weight_beta: float            # 基本項目の比重 (0-1)
    criteria_scores: Dict[str, float]   # 項目別スコア
    field_detail: Dict[str, Any]        # 分野マッチ詳細
    tree_layers: List[str]              # 決定木レイヤー
    explanation: str                    # 説明文
    recommendation: str                 # 推薦レベル


class SimpleMatcher:
    """
    パターンA: シンプルマッチャー
    
    特徴:
    - 遺伝的アルゴリズムなし
    - デフォルトパラメータ使用
    - 優先度ベースの動的決定木
    - 分野マッチング完全対応
    - research_field_matchによる動的重み付け
    """
    
    def __init__(self):
        """初期化"""
        self.params = DEFAULT_PARAMS
        self.criteria = BASIC_CRITERIA
        print("✅ シンプルマッチャー初期化完了")
        print(f"   - 評価項目: {len(self.criteria)}項目")
        print(f"   - 対応分野: {len(FIELD_CATEGORIES)}分野")
        print(f"   - デフォルトパラメータ使用")
    
    def calculate_compatibility(
        self,
        student: Dict[str, Any],
        lab: Dict[str, Any]
    ) -> CompatibilityResult:
        """
        適合度計算（分野考慮版）
        
        Args:
            student: 学生プロファイル
                - 12項目の評価値 (1-10)
                - 12項目の優先度 (1-10)
                - research_field_match (1-10): 分野重視度
                - field_interests: {field_id: interest_level (1-10)}
            
            lab: 研究室プロファイル
                - 12項目の特性値 (1-10 or 0-1)
                - field_id: 研究分野ID
        
        Returns:
            CompatibilityResult
        """
        
        # ===== ステップ1: 優先度ソート =====
        priorities = self._get_sorted_priorities(student)
        
        # ===== ステップ2: 動的決定木構築 =====
        tree_layers = self._build_dynamic_tree(priorities)
        
        # ===== ステップ3: 基本12項目の適合度計算 =====
        basic_score, criteria_scores = self._calculate_basic_match(
            student, lab, priorities
        )
        
        # ===== ステップ4: 分野マッチングスコア計算 =====
        field_score, field_detail = self._calculate_field_match(
            student.get("field_interests", {}),
            lab.get("field_id", "unknown")
        )
        
        # ===== ステップ5: research_field_matchによる重み決定 =====
        field_match_pref = student.get("research_field_match", 5.0)
        alpha = field_match_pref / 10.0  # 分野の比重 (0.1 ~ 1.0)
        beta = 1.0 - alpha  # 基本項目の比重
        
        # ===== ステップ6: 最終スコア統合 =====
        total_score = beta * basic_score + alpha * field_score
        
        # 正規化（0-1の範囲）
        total_score = np.clip(total_score, 0, 1)
        
        # ===== ステップ7: 説明文生成 =====
        explanation = self._generate_explanation(
            total_score, basic_score, field_score,
            alpha, beta, field_detail
        )
        
        return CompatibilityResult(
            total_compatibility=float(total_score),
            basic_score=float(basic_score),
            field_score=float(field_score),
            field_weight_alpha=float(alpha),
            basic_weight_beta=float(beta),
            criteria_scores=criteria_scores,
            field_detail=field_detail,
            tree_layers=tree_layers,
            explanation=explanation,
            recommendation=self._get_recommendation(total_score)
        )
    
    def _get_sorted_priorities(
        self,
        student: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        優先度でソート
        
        Returns:
            [{criterion, priority, value}, ...]（優先度降順）
        """
        priorities = []
        
        for criterion in self.criteria:
            value = student.get(criterion, 5.0)
            priority = student.get(f"{criterion}_priority", 5.0)
            
            priorities.append({
                "criterion": criterion,
                "priority": float(priority),
                "value": float(value)
            })
        
        # 優先度で降順ソート
        priorities.sort(key=lambda x: x["priority"], reverse=True)
        
        return priorities
    
    def _build_dynamic_tree(
        self,
        priorities: List[Dict[str, Any]]
    ) -> List[str]:
        """
        動的決定木構築（優先度ベース）
        
        上位5項目を決定木のレイヤーとして使用
        
        Returns:
            ツリー構造（レイヤー順）
        """
        tree_layers = []
        
        for i, item in enumerate(priorities[:5]):
            layer_name = f"Layer{i+1}: {item['criterion']} (優先度: {item['priority']:.1f})"
            tree_layers.append(layer_name)
        
        return tree_layers
    
    def _calculate_basic_match(
        self,
        student: Dict[str, Any],
        lab: Dict[str, Any],
        priorities: List[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, float]]:
        """
        基本12項目の適合度計算
        
        優先度を重みとして使用し、ガウス類似度で評価
        
        Returns:
            (重み付き総合スコア, 項目別スコア)
        """
        criteria_scores = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for item in priorities:
            criterion = item["criterion"]
            priority = item["priority"]
            student_val = item["value"]
            lab_val = lab.get(criterion, 5.0)
            
            # 正規化（1-10 → 0-1）
            student_norm = self._normalize_value(student_val)
            lab_norm = self._normalize_value(lab_val)
            
            # ガウス類似度計算
            similarity = self._gaussian_similarity(
                student_norm, 
                lab_norm, 
                self.params.similarity_sigma
            )
            
            # デフォルト重みを適用
            default_weight = self.params.default_weights.get(criterion, 1.0)
            
            criteria_scores[criterion] = float(similarity)
            
            # 優先度 × デフォルト重み
            combined_weight = (priority / 10.0) * default_weight
            weighted_sum += similarity * combined_weight
            total_weight += combined_weight
        
        # 重み付き平均
        basic_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        return basic_score, criteria_scores
    
    def _normalize_value(self, value: float) -> float:
        """
        値を0-1に正規化
        
        Args:
            value: 1-10 または 0-1 の値
        
        Returns:
            0-1の正規化された値
        """
        if value > 1:
            # 1-10の場合
            return (value - 1) / 9
        else:
            # 既に0-1の場合
            return value
    
    def _gaussian_similarity(
        self,
        val1: float,
        val2: float,
        sigma: float
    ) -> float:
        """
        ガウス関数による類似度計算
        
        Args:
            val1, val2: 比較する値（0-1）
            sigma: 広がりパラメータ（デフォルト: 0.2）
        
        Returns:
            類似度（0-1）
        """
        diff = abs(val1 - val2)
        similarity = np.exp(-0.5 * (diff / sigma) ** 2)
        return float(similarity)
    
    def _calculate_field_match(
        self,
        field_interests: Dict[str, float],
        lab_field_id: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        分野マッチングスコア計算
        
        Args:
            field_interests: {field_id: interest_level (1-10)}
            lab_field_id: 研究室の分野ID
        
        Returns:
            (スコア, 詳細情報)
        """
        if not field_interests or not lab_field_id:
            return 0.5, {
                "match_type": "no_data",
                "message": "分野情報なし"
            }
        
        # ===== パターン1: 完全一致 =====
        if lab_field_id in field_interests:
            interest_level = field_interests[lab_field_id]
            score = (interest_level / 10.0) * self.params.field_exact_match_bonus
            
            return score, {
                "match_type": "exact",
                "lab_field": lab_field_id,
                "lab_field_name": get_field_name(lab_field_id),
                "interest_level": interest_level,
                "message": f"興味分野と完全一致（興味度: {interest_level}/10）"
            }
        
        # ===== パターン2: カテゴリ一致（部分一致） =====
        lab_category = get_field_category(lab_field_id)
        
        related_scores = []
        related_fields = []
        
        for field_id, interest in field_interests.items():
            if is_same_category(field_id, lab_field_id):
                # 同じカテゴリ内の分野
                score = (interest / 10.0) * self.params.field_category_match_ratio
                related_scores.append(score)
                related_fields.append({
                    "field_id": field_id,
                    "field_name": get_field_name(field_id),
                    "interest": interest
                })
        
        if related_scores:
            avg_score = sum(related_scores) / len(related_scores)
            return avg_score, {
                "match_type": "category",
                "lab_field": lab_field_id,
                "lab_field_name": get_field_name(lab_field_id),
                "lab_category": lab_category,
                "related_fields": related_fields,
                "related_count": len(related_fields),
                "message": f"関連分野と一致（{len(related_fields)}分野）"
            }
        
        # ===== パターン3: 不一致 =====
        return self.params.field_no_match_penalty, {
            "match_type": "none",
            "lab_field": lab_field_id,
            "lab_field_name": get_field_name(lab_field_id),
            "message": "興味分野と異なる"
        }
    
    def _generate_explanation(
        self,
        total: float,
        basic: float,
        field: float,
        alpha: float,
        beta: float,
        field_detail: Dict[str, Any]
    ) -> str:
        """説明文生成"""
        parts = []
        
        # 総合評価
        if total >= 0.85:
            parts.append("✅ 非常に高い適合度")
        elif total >= 0.7:
            parts.append("⭐ 高い適合度")
        elif total >= 0.5:
            parts.append("📊 中程度の適合度")
        else:
            parts.append("⚠️ 低い適合度")
        
        # 比重による説明
        if alpha > 0.7:
            # 分野重視（70%以上）
            match_type = field_detail.get("match_type", "unknown")
            if match_type == "exact":
                parts.append("興味分野と完全一致")
            elif match_type == "category":
                parts.append(f"関連分野と一致")
            else:
                parts.append("興味分野と異なる")
        elif beta > 0.7:
            # 基本項目重視（70%以上）
            if basic >= 0.8:
                parts.append("研究スタイルが非常に合う")
            elif basic >= 0.6:
                parts.append("研究スタイルが概ね合う")
            else:
                parts.append("研究スタイルに違いあり")
        else:
            # バランス型
            parts.append(f"分野{int(alpha*100)}%・項目{int(beta*100)}%で総合評価")
            
            # 詳細を追加
            if field_detail.get("match_type") == "exact":
                parts.append("興味分野一致")
            if basic >= 0.7:
                parts.append("スタイル適合")
        
        return " / ".join(parts)
    
    def _get_recommendation(self, score: float) -> str:
        """推薦レベル"""
        if score >= 0.85:
            return "強く推薦"
        elif score >= 0.7:
            return "推薦"
        elif score >= 0.5:
            return "検討推奨"
        else:
            return "慎重に検討"
    
    def batch_calculate(
        self,
        student: Dict[str, Any],
        labs: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], CompatibilityResult]]:
        """
        複数研究室との適合度を一括計算
        
        Args:
            student: 学生プロファイル
            labs: 研究室リスト
        
        Returns:
            [(lab, result), ...] のリスト（適合度降順）
        """
        results = []
        
        for lab in labs:
            result = self.calculate_compatibility(student, lab)
            results.append((lab, result))
        
        # 適合度でソート（降順）
        results.sort(key=lambda x: x[1].total_compatibility, reverse=True)
        
        return results


# 使用例
if __name__ == "__main__":
    print("🧪 シンプルマッチャー テスト\n")
    
    # テスト用学生
    student = {
        # 基本12項目
        "research_intensity": 9,
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
        "innovation_focus": 8,
        
        # 優先度
        "research_intensity_priority": 10,
        "publication_opportunity_priority": 10,
        "workload_priority": 6,
        
        # 分野重視度
        "research_field_match": 7,  # やや分野重視
        
        # 分野興味
        "field_interests": {
            "ai_ml": 10,
            "image_processing": 7
        }
    }
    
    # テスト用研究室
    lab = {
        "id": "ai_lab",
        "name": "人工知能研究室",
        "field_id": "ai_ml",
        "research_intensity": 9,
        "advisor_style": 7,
        "team_work": 8,
        "workload": 8,
        "theory_practice": 6,
        "skill_development": 8,
        "lab_atmosphere": 7,
        "flexibility": 6,
        "publication_opportunity": 9,
        "interdisciplinary": 5,
        "communication_style": 7,
        "innovation_focus": 9
    }
    
    # マッチング
    matcher = SimpleMatcher()
    result = matcher.calculate_compatibility(student, lab)
    
    print("="*60)
    print(f"研究室: {lab['name']}")
    print("="*60)
    print(f"総合適合度: {result.total_compatibility:.1%}")
    print(f"基本項目: {result.basic_score:.1%}")
    print(f"分野: {result.field_score:.1%}")
    print(f"比重: 分野{result.field_weight_alpha:.1%} / 項目{result.basic_weight_beta:.1%}")
    print(f"推薦: {result.recommendation}")
    print(f"説明: {result.explanation}")
    print(f"\n分野詳細: {result.field_detail['message']}")