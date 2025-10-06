# core/matching/simple_matcher.py
"""
パターンB: 適応的ファジィ決定木マッチャー v3.0
- 12項目評価基準対応
- 20研究分野対応
- 優先度に基づく適応的決定木
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

from config.default_params import (
    DEFAULT_PARAMS, BASIC_CRITERIA, FIELD_CATEGORIES,
    get_field_category, get_field_name, is_same_category,
    HIGH_PRIORITY_THRESHOLD, MID_PRIORITY_THRESHOLD,
    BRANCH_CONFIG
)


@dataclass
class TreeLayer:
    """決定木レイヤー情報"""
    criterion: str
    priority: float
    branches: int                # 2 or 3
    split_points: List[float]    # [0.3, 0.7] or [0.5]
    labels: List[str]            # ['低','中','高'] or ['低','高']


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
    tree_path: str                      # 決定木パス ("高-高-低-高")
    tree_layers: List[str]              # 決定木レイヤー情報
    leaf_criteria: List[str]            # リーフノード項目
    explanation: str                    # 説明文
    recommendation: str                 # 推薦レベル


class SimpleMatcher:
    """
    パターンB: 適応的ファジィ決定木マッチャー
    
    特徴:
    - 12項目評価基準
    - 20研究分野対応
    - 優先度ベースの適応的決定木
    - 優先度 ≥8: 3分岐（低・中・高）
    - 優先度 5-7: 2分岐（低・高）
    - 優先度 <5: リーフノード（重みのみ）
    """
    
    def __init__(self):
        """初期化"""
        self.params = DEFAULT_PARAMS
        self.criteria = BASIC_CRITERIA
        print("✅ パターンB: 適応的マッチャー初期化完了")
        print(f"   - 評価項目: {len(self.criteria)}項目")
        print(f"   - 対応分野: {sum(len(f) for f in FIELD_CATEGORIES.values())}分野")
        print(f"   - 高優先度閾値: {HIGH_PRIORITY_THRESHOLD} (3分岐)")
        print(f"   - 中優先度閾値: {MID_PRIORITY_THRESHOLD} (2分岐)")
    
    def calculate_compatibility(
        self,
        student: Dict[str, Any],
        lab: Dict[str, Any]
    ) -> CompatibilityResult:
        """
        適合度計算（パターンB - 適応的決定木）
        
        Args:
            student: 学生プロファイル
                - 12項目の評価値 (1-10)
                - 12項目の優先度 (1-10)
                - research_field_match (1-10): 分野重視度
                  * 1 = 基本項目重視（分野10%・基本90%）
                  * 10 = 分野重視（分野100%・基本0%）
                - field_interests: {field_id: interest_level (1-10)}
            
            lab: 研究室プロファイル
                - 12項目の特性値 (1-10 or 0-1)
                - field_id: 研究分野ID
        
        Returns:
            CompatibilityResult
        """
        
        # ===== ステップ1: 優先度ソート =====
        priorities = self._get_sorted_priorities(student)
        
        # ===== ステップ2: 適応的決定木構築 =====
        tree_layers, leaf_criteria = self._build_adaptive_tree(priorities)
        
        # ===== ステップ3: 決定木トラバース =====
        tree_path = self._traverse_adaptive_tree(lab, tree_layers)
        
        # ===== ステップ4: 基本13項目の適合度計算 =====
        basic_score, criteria_scores = self._calculate_basic_match(
            student, lab, priorities
        )
        
        # ===== ステップ5: 分野マッチングスコア計算 =====
        field_score, field_detail = self._calculate_field_match(
            student.get("field_interests", {}),
            lab.get("field_id", "unknown")
        )
        
        # ===== ステップ6: research_field_matchによる重み決定 =====
        field_match_pref = student.get("research_field_match", 5.0)
        alpha = field_match_pref / 10.0  # 分野の比重 (0.1 ~ 1.0)
        beta = 1.0 - alpha  # 基本項目の比重
        
        # ===== ステップ7: 最終スコア統合 =====
        total_score = beta * basic_score + alpha * field_score
        total_score = np.clip(total_score, 0, 1)
        
        # ===== ステップ8: 説明文生成 =====
        explanation = self._generate_explanation(
            total_score, basic_score, field_score,
            alpha, beta, field_detail, tree_path,
            len(tree_layers), len(leaf_criteria)
        )
        
        # レイヤー情報の文字列化
        tree_layer_strs = [
            f"Layer{i+1}: {layer.criterion} (優先度: {layer.priority:.1f}, {layer.branches}分岐)"
            for i, layer in enumerate(tree_layers)
        ]
        
        return CompatibilityResult(
            total_compatibility=float(total_score),
            basic_score=float(basic_score),
            field_score=float(field_score),
            field_weight_alpha=float(alpha),
            basic_weight_beta=float(beta),
            criteria_scores=criteria_scores,
            field_detail=field_detail,
            tree_path=tree_path,
            tree_layers=tree_layer_strs,
            leaf_criteria=leaf_criteria,
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
    
    def _build_adaptive_tree(
        self,
        priorities: List[Dict[str, Any]]
    ) -> Tuple[List[TreeLayer], List[str]]:
        """
        適応的決定木構築（パターンB）
        
        優先度に応じて分岐数を決定:
        - 優先度 ≥8: 3分岐（低・中・高）
        - 優先度 5-7: 2分岐（低・高）
        - 優先度 <5: リーフノード
        
        Returns:
            (tree_layers, leaf_criteria)
        """
        tree_layers = []
        leaf_criteria = []
        
        for item in priorities:
            criterion = item["criterion"]
            priority = item["priority"]
            
            if priority >= HIGH_PRIORITY_THRESHOLD:
                # 高優先度: 3分岐
                config = BRANCH_CONFIG["high_priority"]
                tree_layers.append(TreeLayer(
                    criterion=criterion,
                    priority=priority,
                    branches=config["branches"],
                    split_points=config["split_points"],
                    labels=config["labels"]
                ))
            elif priority >= MID_PRIORITY_THRESHOLD:
                # 中優先度: 2分岐
                config = BRANCH_CONFIG["mid_priority"]
                tree_layers.append(TreeLayer(
                    criterion=criterion,
                    priority=priority,
                    branches=config["branches"],
                    split_points=config["split_points"],
                    labels=config["labels"]
                ))
            else:
                # 低優先度: リーフノード
                leaf_criteria.append(criterion)
        
        return tree_layers, leaf_criteria
    
    def _traverse_adaptive_tree(
        self,
        lab: Dict[str, Any],
        tree_layers: List[TreeLayer]
    ) -> str:
        """
        適応的決定木トラバース
        
        研究室の値で各レイヤーを辿り、パスを生成
        
        Returns:
            決定木パス（例: "高-高-低-高-中"）
        """
        path = []
        
        for layer in tree_layers:
            lab_value = lab.get(layer.criterion, 5.0)
            lab_norm = self._normalize_value(lab_value)
            
            if layer.branches == 3:
                # 3分岐
                if lab_norm < layer.split_points[0]:
                    branch = layer.labels[0]  # "低"
                elif lab_norm < layer.split_points[1]:
                    branch = layer.labels[1]  # "中"
                else:
                    branch = layer.labels[2]  # "高"
            else:
                # 2分岐
                if lab_norm < layer.split_points[0]:
                    branch = layer.labels[0]  # "低"
                else:
                    branch = layer.labels[1]  # "高"
            
            path.append(branch)
        
        return "-".join(path) if path else "なし"
    
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
            (分野スコア, 詳細情報)
        """
        if not field_interests:
            return 0.5, {
                "match_type": "no_interest",
                "lab_field": lab_field_id,
                "lab_field_name": get_field_name(lab_field_id),
                "interest_level": 0,
                "message": "分野興味が未設定です"
            }
        
        # 完全一致チェック
        if lab_field_id in field_interests:
            interest = field_interests[lab_field_id]
            score = (interest / 10.0) * self.params.exact_match_weight
            return score, {
                "match_type": "exact",
                "lab_field": lab_field_id,
                "lab_field_name": get_field_name(lab_field_id),
                "interest_level": interest,
                "message": f"興味分野と完全一致！興味度{interest}/10"
            }
        
        # カテゴリ一致チェック
        lab_category = get_field_category(lab_field_id)
        for field_id, interest in field_interests.items():
            if is_same_category(field_id, lab_field_id):
                score = (interest / 10.0) * self.params.category_match_weight
                return score, {
                    "match_type": "category",
                    "lab_field": lab_field_id,
                    "lab_field_name": get_field_name(lab_field_id),
                    "interest_field": field_id,
                    "interest_field_name": get_field_name(field_id),
                    "interest_level": interest,
                    "category": lab_category,
                    "message": f"同カテゴリ（{lab_category}）で一致"
                }
        
        # 不一致
        return self.params.no_match_weight, {
            "match_type": "none",
            "lab_field": lab_field_id,
            "lab_field_name": get_field_name(lab_field_id),
            "message": "興味分野と異なります"
        }
    
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
            val1: 値1（0-1）
            val2: 値2（0-1）
            sigma: 標準偏差
        
        Returns:
            類似度（0-1）
        """
        return np.exp(-0.5 * ((val1 - val2) ** 2) / (sigma ** 2))
    
    def _generate_explanation(
        self,
        total_score: float,
        basic_score: float,
        field_score: float,
        alpha: float,
        beta: float,
        field_detail: Dict[str, Any],
        tree_path: str,
        tree_depth: int,
        leaf_count: int
    ) -> str:
        """説明文生成"""
        
        # スコアレベル
        if total_score >= 0.9:
            level = "✅ 非常に高い適合度"
        elif total_score >= 0.75:
            level = "⭐ 高い適合度"
        elif total_score >= 0.6:
            level = "🔵 中程度の適合度"
        else:
            level = "⚠️ 低めの適合度"
        
        # 分野マッチング
        field_msg = field_detail.get("message", "")
        
        # 決定木情報
        tree_msg = f"優先度に基づき{tree_depth}層の決定木で分類しました（パス: {tree_path}）。"
        if leaf_count > 0:
            tree_msg += f" 残り{leaf_count}項目はリーフノードで評価しました。"
        
        # 比重情報
        weight_msg = f"最終スコアは基本項目{beta*100:.0f}%・分野{alpha*100:.0f}%の比重で統合しました。"
        
        return f"{level}です。{field_msg} {tree_msg} {weight_msg}"
    
    def _get_recommendation(self, score: float) -> str:
        """推薦レベル取得"""
        if score >= 0.9:
            return "強く推薦"
        elif score >= 0.75:
            return "推薦"
        elif score >= 0.6:
            return "検討推奨"
        elif score >= 0.5:
            return "慎重に検討"
        else:
            return "他の選択肢を検討"
    
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


# 使用例・テスト
if __name__ == "__main__":
    print("🧪 パターンB 適応的マッチャー テスト\n")
    
    # テスト用学生（12項目）
    student = {
        # 基本5項目
        "research_intensity": 9,
        "advisor_style": 7,
        "team_work": 5,
        "workload": 8,
        "theory_practice": 6,
        
        # 拡張5項目
        "research_field_match": 7,  # 分野重視度（やや分野重視）
        "skill_development": 7,
        "lab_atmosphere": 6,
        "flexibility": 5,
        "publication_opportunity": 9,
        
        # 特殊2項目
        "interdisciplinary": 4,
        "communication_style": 6,
        
        # 優先度（一部設定）
        "research_intensity_priority": 10,
        "publication_opportunity_priority": 10,
        "innovation_focus_priority": 9,
        "workload_priority": 7,
        
        # 分野興味
        "field_interests": {
            "ai_ml": 10,
            "image_processing": 7
        }
    }
    
    # テスト用研究室（13項目）
    lab = {
        "id": "ai_lab",
        "name": "人工知能研究室",
        "field_id": "ai_ml",
        
        # 基本5項目
        "research_intensity": 9,
        "advisor_style": 7,
        "team_work": 8,
        "workload": 8,
        "theory_practice": 6,
        
        # 拡張5項目
        "research_field_match": 8,
        "skill_development": 8,
        "lab_atmosphere": 7,
        "flexibility": 6,
        "publication_opportunity": 9,
        
        # 特殊3項目
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
    print(f"\n決定木パス: {result.tree_path}")
    print(f"決定木レイヤー数: {len(result.tree_layers)}")
    print(f"リーフノード項目数: {len(result.leaf_criteria)}")
    print(f"\n説明: {result.explanation}")
    print(f"\n分野詳細: {result.field_detail['message']}")
    print("="*60)