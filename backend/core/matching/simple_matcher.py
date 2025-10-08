# backend/core/matching/simple_matcher.py - 修正版（fuzzy_paths対応）

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
        similarity_sigma = 0.2
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
    tree_path: str                      # ★追加
    fuzzy_paths: List[Dict[str, Any]]   # ★追加
    tree_layers: List[str]
    leaf_criteria: List[str]            # ★追加
    explanation: str
    recommendation: str


class SimpleMatcher:
    """シンプルマッチャー（12項目対応）"""
    
    def __init__(self):
        self.params = DEFAULT_PARAMS
        self.criteria = BASIC_CRITERIA
        print("✅ SimpleMatcher 初期化完了")
        print(f"   - 評価項目: {len(self.criteria)}項目")
        print(f"   - デフォルトパラメータ使用")
    
    def calculate_compatibility(
        self,
        student: Dict[str, Any],
        lab: Dict[str, Any]
    ) -> CompatibilityResult:
        """適合度計算"""
        
        # 優先度取得
        priorities = self._get_sorted_priorities(student)
        
        # 基本12項目スコア計算
        basic_score, criteria_scores = self._calculate_basic_match(
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
        
        total = beta * basic_score + alpha * field_score
        
        # 決定木情報生成（簡易版）
        tree_path, fuzzy_paths, tree_layers, leaf_criteria = self._generate_tree_info(
            priorities, student, lab
        )
        
        # 説明文生成
        explanation = self._generate_explanation(
            total, basic_score, field_score, alpha, beta, field_detail
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
            tree_path=tree_path,           # ★追加
            fuzzy_paths=fuzzy_paths,       # ★追加
            tree_layers=tree_layers,
            leaf_criteria=leaf_criteria,   # ★追加
            explanation=explanation,
            recommendation=recommendation
        )
    
    def _get_sorted_priorities(self, student: Dict[str, Any]) -> List[Dict[str, Any]]:
        """優先度付きリスト作成"""
        priorities = []
        
        for criterion in self.criteria:
            if criterion == "research_field_match":
                continue  # 特殊項目なのでスキップ
            
            value = student.get(criterion, 5.0)
            priority = student.get(f"{criterion}_priority", 5.0)
            
            priorities.append({
                "criterion": criterion,
                "value": value,
                "priority": priority
            })
        
        # 優先度降順ソート
        priorities.sort(key=lambda x: x["priority"], reverse=True)
        
        return priorities
    
    def _calculate_basic_match(
        self,
        student: Dict[str, Any],
        lab: Dict[str, Any],
        priorities: List[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, float]]:
        """基本12項目適合度計算"""
        
        criteria_scores = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for item in priorities:
            criterion = item["criterion"]
            priority = item["priority"]
            student_val = item["value"]
            lab_val = lab.get(criterion, 5.0)
            
            # 正規化
            student_norm = self._normalize_value(student_val)
            lab_norm = self._normalize_value(lab_val)
            
            # ガウス類似度
            similarity = self._gaussian_similarity(
                student_norm, lab_norm, 
                self.params.similarity_sigma
            )
            
            # デフォルト重み
            default_weight = self.params.default_weights.get(criterion, 1.0)
            
            criteria_scores[criterion] = float(similarity)
            
            # 優先度 × デフォルト重み
            combined_weight = (priority / 10.0) * default_weight
            weighted_sum += similarity * combined_weight
            total_weight += combined_weight
        
        basic_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        return basic_score, criteria_scores
    
    def _normalize_value(self, value: float) -> float:
        """0-1正規化"""
        if value > 1:
            return (value - 1) / 9
        return value
    
    def _gaussian_similarity(self, val1: float, val2: float, sigma: float) -> float:
        """ガウス類似度"""
        diff = abs(val1 - val2)
        return float(np.exp(-0.5 * (diff / sigma) ** 2))
    
    def _calculate_field_match(
        self,
        field_interests: Dict[str, float],
        lab_field_id: str
    ) -> Tuple[float, Dict[str, Any]]:
        """分野マッチング"""
        
        if not field_interests or not lab_field_id:
            return 0.5, {"match_type": "no_data"}
        
        # 完全一致
        if lab_field_id in field_interests:
            interest_level = field_interests[lab_field_id]
            score = (interest_level / 10.0) * self.params.field_exact_match_bonus
            return score, {
                "match_type": "exact",
                "lab_field": lab_field_id,
                "interest_level": interest_level
            }
        
        # カテゴリ一致
        if PARAMS_AVAILABLE:
            lab_category = get_field_category(lab_field_id)
            related_scores = []
            
            for field_id, interest in field_interests.items():
                if is_same_category(field_id, lab_field_id):
                    score = (interest / 10.0) * self.params.field_category_match_ratio
                    related_scores.append(score)
            
            if related_scores:
                avg_score = sum(related_scores) / len(related_scores)
                return avg_score, {
                    "match_type": "category",
                    "lab_field": lab_field_id,
                    "related_count": len(related_scores)
                }
        
        # 不一致
        return self.params.field_no_match_penalty, {
            "match_type": "none",
            "lab_field": lab_field_id
        }
    
    def _generate_tree_info(
        self,
        priorities: List[Dict[str, Any]],
        student: Dict[str, Any],
        lab: Dict[str, Any]
    ) -> Tuple[str, List[Dict[str, Any]], List[str], List[str]]:
        """
        決定木情報生成（簡易版）
        
        Returns:
            (tree_path, fuzzy_paths, tree_layers, leaf_criteria)
        """
        tree_layers = []
        leaf_criteria = []
        path_parts = []
        
        # 上位3項目で決定木パス生成
        for i, item in enumerate(priorities[:3]):
            criterion = item["criterion"]
            priority = item["priority"]
            student_val = item["value"]
            lab_val = lab.get(criterion, 5.0)
            
            tree_layers.append(f"{criterion}(優先度:{priority:.1f})")
            
            # パス判定（簡易）
            student_norm = self._normalize_value(student_val)
            lab_norm = self._normalize_value(lab_val)
            
            if lab_norm < 0.33:
                path_parts.append("低")
            elif lab_norm < 0.67:
                path_parts.append("中")
            else:
                path_parts.append("高")
        
        # 残りはリーフ
        for item in priorities[3:]:
            leaf_criteria.append(item["criterion"])
        
        tree_path = "-".join(path_parts) if path_parts else "未決定"
        
        # ファジィパス情報（簡易版）
        fuzzy_paths = [
            {
                "path": tree_path,
                "membership": 1.0,
                "layer_count": len(tree_layers)
            }
        ]
        
        return tree_path, fuzzy_paths, tree_layers, leaf_criteria
    
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
        
        if total >= 0.85:
            parts.append("✅ 非常に高い適合度")
        elif total >= 0.7:
            parts.append("⭐ 高い適合度")
        elif total >= 0.5:
            parts.append("📊 中程度の適合度")
        else:
            parts.append("⚠️ 低い適合度")
        
        if alpha > 0.7:
            match_type = field_detail.get("match_type", "unknown")
            if match_type == "exact":
                parts.append("興味分野と完全一致")
            elif match_type == "category":
                parts.append("関連分野と一致")
        elif beta > 0.7:
            if basic >= 0.8:
                parts.append("研究スタイルが非常に合う")
        
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


# テスト用
if __name__ == "__main__":
    print("=" * 60)
    print("SimpleMatcher テスト")
    print("=" * 60)
    
    matcher = SimpleMatcher()
    
    student = {
        "research_intensity": 8,
        "advisor_style": 7,
        "team_work": 7,
        "workload": 7,
        "theory_practice": 6,
        "research_field_match": 5,
        "skill_development": 8,
        "lab_atmosphere": 7,
        "flexibility": 6,
        "publication_opportunity": 7,
        "interdisciplinary": 6,
        "communication_style": 7,
        "field_interests": {"ai_ml": 9}
    }
    
    lab = {
        "research_intensity": 7,
        "advisor_style": 7,
        "team_work": 8,
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
    
    result = matcher.calculate_compatibility(student, lab)
    
    print(f"\n結果:")
    print(f"  総合スコア: {result.total_compatibility:.3f}")
    print(f"  基本スコア: {result.basic_score:.3f}")
    print(f"  分野スコア: {result.field_score:.3f}")
    print(f"  決定木パス: {result.tree_path}")
    print(f"  説明: {result.explanation}")
    print(f"  推薦: {result.recommendation}")
    
    print("\n" + "=" * 60)
    print("✅ テスト完了")