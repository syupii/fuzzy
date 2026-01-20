# backend/core/matching/fuzzy_multipath_matcher.py
"""
ファジィ決定木マッチャー - 技術資料完全準拠版（分野重視改善版）
★★★ 説明生成機能追加版 ★★★

【パラメータ調整ガイド】
====================================
このファイル内で調整可能なパラメータ:

1. 層数調整（重要）
   HIGH_PRIORITY_THRESHOLD = 8.0  ← 【ここを変更】
   MID_PRIORITY_THRESHOLD = 5.0   ← 【ここを変更】
   
   推奨設定:
   - バランス型: HIGH=7.0, MID=4.0 (層数10-12、精度90-92%)
   - 高精度型: HIGH=6.0, MID=3.0 (層数12-13、精度92-94%)
   - 軽量型: HIGH=8.0, MID=5.0 (現在の設定、層数8-9、精度88%)

2. 類似度計算
   SIMILARITY_SIGMA = 0.2  ← 【ここを変更】
   
   効果:
   - 小さい値(0.1-0.15): より厳しい評価、高い適合のみ高スコア
   - 標準値(0.2): バランスが良い
   - 大きい値(0.25-0.3): より緩い評価、中程度の適合も高スコア

3. 枝刈り閾値
   PRUNING_THRESHOLD = 0.01  ← 【ここを変更】
   
   効果:
   - 小さい値(0.005): より多くのパスを評価（高精度、遅い）
   - 標準値(0.01): バランスが良い
   - 大きい値(0.02-0.05): 少ないパスで評価（高速、やや精度低下）

4. 分野マッチング（★改善版パラメータ★）
   FIELD_EXACT_BONUS = 0.15        ← 完全一致ボーナス係数
   FIELD_MISMATCH_PENALTY = 0.15   ← 不一致ペナルティ係数
   CATEGORY_DECAY = 0.7            ← カテゴリ一致時の減衰係数
   NO_MATCH_PENALTY = 0.3          ← 不一致時のベーススコア
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
        # フォールバック: インポートできない場合は空の説明を返す
        def generate_detailed_explanation(*args, **kwargs):
            return ""
        def generate_short_explanation(*args, **kwargs):
            return ""


@dataclass
class FuzzyPath:
    """ファジィパス"""
    path_id: int
    layers: List[Tuple[str, str, float]]  # [(criterion, label, membership), ...]
    total_membership: float  # パス全体の所属度
    score: float  # このパスでの適合度スコア


@dataclass
class CompatibilityResult:
    """適合度計算結果"""
    total_compatibility: float
    basic_score: float
    field_score: float
    field_weight_alpha: float
    basic_weight_beta: float
    
    # 詳細情報
    criteria_scores: Dict[str, float]
    fuzzy_paths: List[FuzzyPath]
    field_detail: Dict[str, Any]
    
    # 説明（3種類）
    explanation: str                          # 従来版（ナラティブエンジン）
    explanation_detailed: Optional[str] = ""  # ★追加: 詳細版（セクション分け）
    explanation_short: Optional[str] = ""     # ★追加: 短縮版（カード表示用）
    
    recommendation: str = ""


class MembershipFunctions:
    """メンバーシップ関数（技術資料 3.4.1 の厳密な定義）"""
    
    @staticmethod
    def low(x: float) -> float:
        """「低い」への所属度
        
        μ_low(x) = {
            1                (x ≤ 0.3)
            (0.5-x)/0.2     (0.3 < x < 0.5)
            0                (x ≥ 0.5)
        }
        """
        if x <= 0.3:
            return 1.0
        elif x < 0.5:
            return (0.5 - x) / 0.2
        else:
            return 0.0
    
    @staticmethod
    def medium(x: float) -> float:
        """「中」への所属度
        
        μ_medium(x) = {
            0                           (x ≤ 0.3 or x ≥ 0.9)
            (x-0.3)/0.2                (0.3 < x < 0.5)
            1                           (0.5 ≤ x ≤ 0.7)
            (0.9-x)/0.2                (0.7 < x < 0.9)
        }
        """
        if x <= 0.3 or x >= 0.9:
            return 0.0
        elif x < 0.5:
            return (x - 0.3) / 0.2
        elif x <= 0.7:
            return 1.0
        else:  # 0.7 < x < 0.9
            return (0.9 - x) / 0.2
    
    @staticmethod
    def high(x: float) -> float:
        """「高い」への所属度
        
        μ_high(x) = {
            0                (x ≤ 0.7)
            (x-0.7)/0.2     (0.7 < x < 0.9)
            1                (x ≥ 0.9)
        }
        """
        if x <= 0.7:
            return 0.0
        elif x < 0.9:
            return (x - 0.7) / 0.2
        else:
            return 1.0
    
    @staticmethod
    def fuzzify(x: float) -> Dict[str, float]:
        """値をファジィ化（3段階）"""
        return {
            "low": MembershipFunctions.low(x),
            "medium": MembershipFunctions.medium(x),
            "high": MembershipFunctions.high(x)
        }
    
    @staticmethod
    def fuzzify_2level(x: float) -> Dict[str, float]:
        """値をファジィ化（2段階: 低・高）"""
        if x < 0.5:
            return {
                "low": (0.5 - x) / 0.5,
                "high": x / 0.5
            }
        else:
            return {
                "low": 0.0,
                "high": 1.0
            }


class FuzzyMultiPathMatcher:
    """
    ファジィ決定木マッチャー（分野重視改善版）
    
    主な機能:
    1. 優先度に基づく適応的決定木構築
    2. 複数パスの探索とメンバーシップ度計算
    3. 所属度による重み付け統合
    4. 改善版の分野マッチング（ボーナス/ペナルティ付き）
    """
    
    # 評価基準（12項目 + 分野重視度）
    CRITERIA = [
        "research_intensity", "advisor_style", "team_work",
        "workload", "theory_practice", "skill_development",
        "lab_atmosphere", "flexibility", "publication_opportunity",
        "interdisciplinary", "communication_style", "research_field_match"
    ]
    
    # ============================================================
    # 【重要】パラメータ設定エリア - ここを調整してください
    # ============================================================
    
    # 1. 類似度計算パラメータ（技術資料 3.5.2節）
    SIMILARITY_SIGMA = 0.3  # ガウス類似度のσ
    
    # 2. 枝刈り閾値
    PRUNING_THRESHOLD = 0.01  # 枝刈り閾値
    
    # 3. 優先度閾値（層数に影響）★★★ 最も重要 ★★★
    HIGH_PRIORITY_THRESHOLD = 8.0  # 高優先度閾値
    MID_PRIORITY_THRESHOLD = 5.0   # 中優先度閾値
    
    # 4. 分野マッチングパラメータ（★改善版★）
    FIELD_EXACT_BONUS = 0.15        # 完全一致時のボーナス係数
    FIELD_MISMATCH_PENALTY = 0.15   # 不一致時のペナルティ係数
    CATEGORY_DECAY = 0.7            # カテゴリ一致時の減衰係数
    NO_MATCH_PENALTY = 0.3          # 不一致時のベーススコア
    
    # ============================================================
    
    def __init__(self):
        print("✅ FuzzyMultiPathMatcher 初期化完了（分野重視改善版 + 説明生成機能）")
        print(f"   - σ = {self.SIMILARITY_SIGMA}")
        print(f"   - 高優先度閾値 = {self.HIGH_PRIORITY_THRESHOLD}")
        print(f"   - 中優先度閾値 = {self.MID_PRIORITY_THRESHOLD}")
        print(f"   - 枝刈り閾値 = {self.PRUNING_THRESHOLD}")
        print(f"   - 分野減衰係数 = {self.CATEGORY_DECAY}")
        print(f"   - 完全一致ボーナス = {self.FIELD_EXACT_BONUS}")
        print(f"   - 不一致ペナルティ = {self.FIELD_MISMATCH_PENALTY}")
    
    def calculate_compatibility(
        self,
        student: Dict[str, Any],
        lab: Dict[str, Any]
    ) -> CompatibilityResult:
        """
        適合度計算（改善版）
        
        Step 1 & 2: 優先度ソート + 決定木構築
        Step 3: 複数パスの導出
        Step 4: 複数パスの統合（基本項目スコア）
        Step 5: 分野マッチング
        Step 6: 改善版の最終スコア統合（非線形alpha + ボーナス/ペナルティ）
        """
        
        print(f"\n{'#'*70}")
        print(f"### 適合度計算開始 ###")
        print(f"{'#'*70}")
        
        # Step 1 & 2: 優先度ソート + 決定木構築
        print(f"\n📝 Step 1 & 2: 優先度ソート + 決定木構築")
        priorities = self._get_sorted_priorities(student)
        tree_layers = self._build_fuzzy_tree(priorities)
        print(f"   高優先度項目数: {sum(1 for p in priorities if p['priority'] >= self.HIGH_PRIORITY_THRESHOLD)}")
        print(f"   中優先度項目数: {sum(1 for p in priorities if self.MID_PRIORITY_THRESHOLD <= p['priority'] < self.HIGH_PRIORITY_THRESHOLD)}")
        print(f"   決定木レイヤー数: {len(tree_layers)}")
        
        # Step 3: 複数パスの導出
        print(f"\n🌳 Step 3: 複数パスの導出")
        fuzzy_paths = self._explore_fuzzy_paths(tree_layers, student, lab)
        print(f"   生成パス数（枝刈り前）: {len(fuzzy_paths)}")
        
        # 枝刈り後の正規化
        fuzzy_paths = self._normalize_path_memberships(fuzzy_paths)
        print(f"   有効パス数（枝刈り後）: {len(fuzzy_paths)}")
        if fuzzy_paths:
            print(f"   最大所属度: {max(p.total_membership for p in fuzzy_paths):.4f}")
            print(f"   最小所属度: {min(p.total_membership for p in fuzzy_paths):.4f}")
        
        # Step 4: 複数パスの統合
        print(f"\n🔀 Step 4: 複数パスの統合")
        basic_score, criteria_scores = self._integrate_fuzzy_paths(
            fuzzy_paths, student, lab
        )
        print(f"   基本スコア: {basic_score:.4f}")
        print(f"   項目別スコア:")
        for i, (criterion, score) in enumerate(sorted(criteria_scores.items(), key=lambda x: x[1], reverse=True)[:5], 1):
            print(f"     {i}. {criterion}: {score:.4f}")
        
        # Step 5: 分野マッチング
        print(f"\n🎯 Step 5: 分野マッチング")
        field_interests = student.get("field_interests", {})
        lab_field = lab.get("field_id", "")
        field_score, field_detail = self._calculate_field_match(
            field_interests, lab_field
        )
        print(f"   研究室分野: {lab_field}")
        print(f"   学生の興味: {list(field_interests.keys()) if field_interests else 'なし'}")
        print(f"   マッチタイプ: {field_detail.get('match_type', 'unknown')}")
        print(f"   分野スコア: {field_score:.4f}")
        
        # ========================================
        # ★★★ Step 6: 修正版の最終スコア統合 ★★★
        # ========================================
        
        rfm = student.get("research_field_match", 5.0)
        
        print(f"\n{'='*70}")
        print(f"🔍 【Step 6: 最終スコア統合 - 修正版】")
        print(f"{'='*70}")
        
        # 【修正】rfm=5で alpha=beta=0.5 になる線形補間
        print(f"\n📊 Step 6-1: alpha/beta計算（修正版）")
        print(f"   research_field_match (rfm) = {rfm}")
        
        if rfm >= 5.0:
            # 分野重視モード: rfm 5→10 で alpha 0.5→0.9
            calculated_alpha = 0.5 + (rfm - 5.0) / 5.0 * 0.4
            mode = "分野重視"
            print(f"   モード: {mode}（rfm={rfm} ≥ 5.0）")
            print(f"   計算式: alpha = 0.5 + (rfm - 5.0) / 5.0 × 0.4")
            print(f"   計算式: alpha = 0.5 + ({rfm} - 5.0) / 5.0 × 0.4")
            print(f"   計算式: alpha = 0.5 + {(rfm - 5.0) / 5.0:.4f} × 0.4")
            print(f"   計算式: alpha = 0.5 + {(rfm - 5.0) / 5.0 * 0.4:.4f}")
        else:
            # 基本重視モード: rfm 0→5 で alpha 0.2→0.5
            calculated_alpha = 0.2 + rfm / 5.0 * 0.3
            mode = "基本重視"
            print(f"   モード: {mode}（rfm={rfm} < 5.0）")
            print(f"   計算式: alpha = 0.2 + rfm / 5.0 × 0.3")
            print(f"   計算式: alpha = 0.2 + {rfm} / 5.0 × 0.3")
            print(f"   計算式: alpha = 0.2 + {rfm / 5.0:.4f} × 0.3")
            print(f"   計算式: alpha = 0.2 + {rfm / 5.0 * 0.3:.4f}")
        
        calculated_beta = 1.0 - calculated_alpha
        
        print(f"   計算結果:")
        print(f"   ├─ calculated_alpha = {calculated_alpha:.4f}")
        print(f"   └─ calculated_beta = 1.0 - {calculated_alpha:.4f} = {calculated_beta:.4f}")
        
        alpha = calculated_alpha
        beta = calculated_beta
        
        print(f"\n   【最終確定値】")
        print(f"   ⭐ alpha (分野の比重) = {alpha:.4f} ({alpha*100:.2f}%)")
        print(f"   ⭐ beta (基本の比重) = {beta:.4f} ({beta*100:.2f}%)")
        
        # 期待値チェック
        if rfm >= 5.0:
            expected = "alpha >= beta（分野重視）"
            actual_ok = alpha >= beta
        else:
            expected = "beta >= alpha（基本重視）"
            actual_ok = beta >= alpha
        
        print(f"   期待: {expected}")
        print(f"   実際: {'✅ 正常' if actual_ok else '❌ 異常'}")
        
        # 基本統合
        print(f"\n📈 Step 6-2: 基本統合")
        print(f"   basic_score = {basic_score:.4f}")
        print(f"   field_score = {field_score:.4f}")
        print(f"   計算式: total = beta × basic_score + alpha × field_score")
        print(f"   計算式: total = {beta:.4f} × {basic_score:.4f} + {alpha:.4f} × {field_score:.4f}")
        
        basic_contribution = beta * basic_score
        field_contribution = alpha * field_score
        total = basic_contribution + field_contribution
        
        print(f"   結果:")
        print(f"   ├─ 基本寄与 = {beta:.4f} × {basic_score:.4f} = {basic_contribution:.4f}")
        print(f"   ├─ 分野寄与 = {alpha:.4f} × {field_score:.4f} = {field_contribution:.4f}")
        print(f"   └─ 合計 = {basic_contribution:.4f} + {field_contribution:.4f} = {total:.4f}")
        
        # 【改善】ボーナス/ペナルティ適用
        match_type = field_detail.get("match_type", "unknown")
        
        print(f"\n⭐ Step 6-3: ボーナス/ペナルティ適用")
        print(f"   分野マッチタイプ: {match_type}")
        
        if match_type == "exact":
            # 完全一致ボーナス
            bonus = self.FIELD_EXACT_BONUS * alpha
            print(f"   ✨ 完全一致ボーナス")
            print(f"   計算: {self.FIELD_EXACT_BONUS} × {alpha:.4f} = {bonus:.4f}")
            print(f"   適用前: {total:.4f}")
            total += bonus
            print(f"   適用後: {total:.4f}")
            field_detail["bonus_applied"] = True
            field_detail["bonus_value"] = bonus
            
        elif match_type == "no_match":
            # 不一致ペナルティ
            penalty = self.FIELD_MISMATCH_PENALTY * alpha
            print(f"   ⚠️ 不一致ペナルティ")
            print(f"   計算: {self.FIELD_MISMATCH_PENALTY} × {alpha:.4f} = {penalty:.4f}")
            print(f"   適用前: {total:.4f}")
            total -= penalty
            print(f"   適用後: {total:.4f}")
            field_detail["penalty_applied"] = True
            field_detail["penalty_value"] = penalty
            
        else:
            print(f"   → ボーナス/ペナルティなし")
        
        # 0.0-1.0にクリップ
        print(f"\n🎯 Step 6-4: 最終スコア確定")
        print(f"   クリップ前: {total:.4f}")
        total = max(0.0, min(1.0, total))
        print(f"   クリップ後: {total:.4f} (0.0-1.0の範囲に制限)")
        
        print(f"\n{'='*70}")
        print(f"✅ 【Step 6完了】最終適合度 = {total:.4f}")
        print(f"{'='*70}\n")
        
        # 従来の説明文生成（ナラティブエンジン）
        explanation = self._generate_explanation(
            total, basic_score, field_score, alpha, beta,
            field_detail, len(fuzzy_paths),
            student=student,
            lab=lab,
            criteria_scores=criteria_scores
        )
        
        # ★★★ 追加: 詳細説明・短縮説明を生成 ★★★
        explanation_detailed = generate_detailed_explanation(
            lab, student, criteria_scores, field_score, total
        )
        explanation_short = generate_short_explanation(
            lab, student, criteria_scores, field_score, total
        )
        
        # 推薦レベル
        recommendation = self._get_recommendation(total)
        
        # 最終サマリー
        print(f"\n{'#'*70}")
        print(f"### 計算完了サマリー ###")
        print(f"{'#'*70}")
        print(f"   総合適合度: {total:.4f}")
        print(f"   推薦レベル: {recommendation}")
        print(f"   説明: {explanation}")
        print(f"{'#'*70}\n")
        
        return CompatibilityResult(
            total_compatibility=total,
            basic_score=basic_score,
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
    
    def _get_sorted_priorities(
        self, 
        student: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Step 1: 優先度ソート（技術資料 3.3節）"""
        priorities = []
        
        for criterion in self.CRITERIA:
            priority_key = f"{criterion}_priority"
            priority = student.get(priority_key, 5.0)
            priorities.append({
                "criterion": criterion,
                "priority": priority
            })
        
        # 優先度降順にソート
        priorities.sort(key=lambda x: x["priority"], reverse=True)
        return priorities
    
    def _build_fuzzy_tree(
        self, 
        priorities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Step 2: 適応的決定木構築（技術資料 3.3節）"""
        tree_layers = []
        
        for item in priorities:
            priority = item["priority"]
            criterion = item["criterion"]
            
            if priority >= self.HIGH_PRIORITY_THRESHOLD:
                # 高優先度: 3分岐（低・中・高）
                tree_layers.append({
                    "criterion": criterion,
                    "priority": priority,
                    "branches": 3,
                    "labels": ["low", "medium", "high"]
                })
            elif priority >= self.MID_PRIORITY_THRESHOLD:
                # 中優先度: 2分岐（低・高）
                tree_layers.append({
                    "criterion": criterion,
                    "priority": priority,
                    "branches": 2,
                    "labels": ["low", "high"]
                })
            # 低優先度（< MID_PRIORITY_THRESHOLD）: リーフノード（レイヤーに含めない）
        
        return tree_layers
    
    def _explore_fuzzy_paths(
        self,
        tree_layers: List[Dict[str, Any]],
        student: Dict[str, Any],
        lab: Dict[str, Any]
    ) -> List[FuzzyPath]:
        """
        Step 3: 複数パスの導出（技術資料 3.4節）
        
        全ての可能なパスを探索し、各パスの所属度を計算
        """
        if not tree_layers:
            # レイヤーがない場合は単一パス
            return [FuzzyPath(
                path_id=0,
                layers=[],
                total_membership=1.0,
                score=0.5
            )]
        
        # 再帰的にパスを生成
        all_paths = []
        self._generate_paths_recursive(
            tree_layers, lab, 0, [], 1.0, all_paths
        )
        
        # 枝刈り（所属度が閾値以下のパスを削除）
        pruned_paths = [
            path for path in all_paths
            if path.total_membership >= self.PRUNING_THRESHOLD
        ]
        
        return pruned_paths
    
    def _generate_paths_recursive(
        self,
        tree_layers: List[Dict[str, Any]],
        lab: Dict[str, Any],
        layer_idx: int,
        current_path: List[Tuple[str, str, float]],
        cumulative_membership: float,
        all_paths: List[FuzzyPath]
    ):
        """パスを再帰的に生成"""
        
        # 全レイヤーを処理完了
        if layer_idx >= len(tree_layers):
            all_paths.append(FuzzyPath(
                path_id=len(all_paths),
                layers=current_path.copy(),
                total_membership=cumulative_membership,
                score=0.0  # 後で計算
            ))
            return
        
        # 現在のレイヤー情報
        layer = tree_layers[layer_idx]
        criterion = layer["criterion"]
        labels = layer["labels"]
        
        # 研究室の値を取得・正規化
        lab_value = self._normalize_value(lab.get(criterion, 5.0))
        
        # メンバーシップ度を計算
        if len(labels) == 3:
            # 3分岐
            memberships = MembershipFunctions.fuzzify(lab_value)
        else:
            # 2分岐
            memberships = MembershipFunctions.fuzzify_2level(lab_value)
        
        # 各ブランチでパスを生成
        for label in labels:
            membership = memberships[label]
            
            # 枝刈り: 所属度が閾値以下の場合はスキップ
            new_membership = cumulative_membership * membership
            if new_membership < self.PRUNING_THRESHOLD:
                continue
            
            # 新しいパスを生成
            new_path = current_path + [(criterion, label, membership)]
            self._generate_paths_recursive(
                tree_layers, lab, layer_idx + 1,
                new_path, new_membership, all_paths
            )
    
    def _normalize_path_memberships(
        self, 
        paths: List[FuzzyPath]
    ) -> List[FuzzyPath]:
        """
        パスの所属度を正規化（技術資料 3.4.2節）
        
        wᵢ = Membershipᵢ / Σ Membershipⱼ
        """
        if not paths:
            return paths
        
        total_membership = sum(path.total_membership for path in paths)
        
        if total_membership == 0:
            return paths
        
        # 正規化
        for path in paths:
            path.total_membership = path.total_membership / total_membership
        
        return paths
    
    def _integrate_fuzzy_paths(
        self, 
        fuzzy_paths: List[FuzzyPath],
        student: Dict[str, Any],
        lab: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Step 4: 複数パスの統合（技術資料 3.5節）
        
        各パスのスコアを所属度で重み付け加重平均
        S_basic = Σ(Sᵢ × wᵢ)
        """
        if not fuzzy_paths:
            return 0.5, {}
        
        # 各パスのスコアを計算
        total_score = 0.0
        all_criteria_scores = {}
        
        for path in fuzzy_paths:
            path_score, criteria_scores = self._calculate_path_score(
                path, student, lab
            )
            path.score = path_score
            
            # 所属度で重み付け
            total_score += path.score * path.total_membership
            
            # 項目別スコアを記録
            for criterion, score in criteria_scores.items():
                if criterion not in all_criteria_scores:
                    all_criteria_scores[criterion] = []
                all_criteria_scores[criterion].append(
                    (score, path.total_membership)
                )
        
        # 項目別スコアも重み付け平均
        averaged_criteria_scores = {}
        for criterion, score_weight_pairs in all_criteria_scores.items():
            weighted_sum = sum(s * w for s, w in score_weight_pairs)
            averaged_criteria_scores[criterion] = weighted_sum
        
        return total_score, averaged_criteria_scores
    
    def _calculate_path_score(
        self, 
        path: FuzzyPath,
        student: Dict[str, Any],
        lab: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """
        パスごとのスコア計算（技術資料 3.5.1節）
        """
        criteria_scores = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        # 全ての評価項目について計算
        for criterion in self.CRITERIA:
            # 学生の希望値（正規化済み）
            student_val = self._normalize_value(student.get(criterion, 5.0))
            
            # 研究室の値（正規化済み）
            lab_val = self._normalize_value(lab.get(criterion, 5.0))
            
            # ガウス類似度計算
            similarity = self._calculate_gaussian_similarity(student_val, lab_val)
            
            # 優先度取得
            priority_key = f"{criterion}_priority"
            priority = student.get(priority_key, 5.0)
            
            # 優先度から重みを計算（非線形変換）
            weight = self._calculate_priority_weight(priority)
            
            criteria_scores[criterion] = similarity
            weighted_sum += similarity * weight
            total_weight += weight
        
        # 重み付け平均
        path_score = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        return path_score, criteria_scores
    
    def _calculate_priority_weight(self, priority: float) -> float:
        """
        優先度から重みを計算（非線形変換）
        
        priority 10 → weight 1.00
        priority  8 → weight 0.72
        priority  5 → weight 0.32
        priority  3 → weight 0.11
        priority  1 → weight 0.01
        """
        normalized = priority / 10.0
        weight = normalized ** 1.5
        return weight
    
    def _calculate_gaussian_similarity(
        self,
        student_val: float,
        lab_val: float
    ) -> float:
        """
        ガウス類似度計算（技術資料 3.5.2節）
        
        Similarity = exp(-(d²)/(2σ²))
        """
        d = abs(student_val - lab_val)
        sigma = self.SIMILARITY_SIGMA
        similarity = math.exp(-(d ** 2) / (2 * sigma ** 2))
        return similarity
    
    def _calculate_field_match(
        self,
        field_interests: Dict[str, float],
        lab_field: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Step 5: 分野マッチング（技術資料 3.6節）
        
        完全一致: I/10
        カテゴリ一致: I/10 × 0.7  ★ 減衰係数0.7
        不一致: 0.3  ★ 固定値0.3
        """
        if not field_interests or not lab_field:
            return 0.5, {
                "match_type": "unknown",
                "lab_field": lab_field,
                "message": "分野情報が不足"
            }
        
        # 完全一致チェック
        if lab_field in field_interests:
            interest_level = field_interests[lab_field]
            score = interest_level / 10.0
            
            return score, {
                "match_type": "exact",
                "lab_field": lab_field,
                "interest_level": interest_level,
                "message": f"興味分野と完全一致（興味度: {interest_level}/10）"
            }
        
        # カテゴリ一致チェック
        best_category_score = 0.0
        best_category_field = None
        
        for interest_field, interest_level in field_interests.items():
            if self._is_same_category(lab_field, interest_field):
                # ★ 減衰係数適用（技術資料 3.6節）
                category_score = (interest_level / 10.0) * self.CATEGORY_DECAY
                if category_score > best_category_score:
                    best_category_score = category_score
                    best_category_field = interest_field
        
        if best_category_score > 0:
            return best_category_score, {
                "match_type": "category",
                "lab_field": lab_field,
                "matched_interest": best_category_field,
                "message": "関連分野と一致（カテゴリ一致）"
            }
        
        # 不一致
        return self.NO_MATCH_PENALTY, {
            "match_type": "no_match",
            "lab_field": lab_field,
            "message": "興味分野と異なる"
        }
    
    def _is_same_category(self, field1: str, field2: str) -> bool:
        """2つの分野が同じカテゴリか判定"""
        # config.default_paramsから読み込み
        try:
            from config.default_params import is_same_category
            return is_same_category(field1, field2)
        except:
            return False
    
    def _normalize_value(self, value: float) -> float:
        """値を0-1に正規化"""
        if value > 1.0:
            return value / 10.0
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
        # ★追加: 詳細な判定を行うために引数を増やします
        student: Dict[str, Any] = None,
        lab: Dict[str, Any] = None,
        criteria_scores: Dict[str, float] = None
    ) -> str:
        """
        説得力のある説明文を動的に生成する（ナラティブエンジン）
        """
        # データが不足している場合は簡易メッセージを返す
        if student is None or lab is None or criteria_scores is None:
            if total >= 0.8: return "非常に高い適合性があります。"
            elif total >= 0.6: return "条件によく合致しています。"
            else: return "検討の余地があります。"

        # --- 1. ユーザーの「こだわり（高優先度項目）」を特定 ---
        # 優先度キー（xxx_priority）の値が8以上のものを抽出
        priorities = self._get_sorted_priorities(student)
        top_priorities = [p for p in priorities if p['priority'] >= 8]
        
        # --- 2. 「こだわり」と「実態」の照合 ---
        matched_points = []     # こだわっていて、かつ一致した点 (スコア高)
        compromise_points = []  # こだわっていたが、一致しなかった点 (スコア低)
        
        # 項目名の日本語マッピング
        criteria_labels = {
            "research_intensity": "研究強度", "advisor_style": "指導方針",
            "team_work": "チームワーク", "workload": "活動量",
            "theory_practice": "理論・実践", "skill_development": "スキル習得",
            "lab_atmosphere": "雰囲気", "flexibility": "柔軟性",
            "publication_opportunity": "発表機会", "interdisciplinary": "学際性",
            "communication_style": "交流頻度"
        }

        for p in top_priorities:
            key = p['criterion']
            # 分野は別ロジックで扱うのでスキップ
            if key == "research_field_match":
                continue
                
            score = criteria_scores.get(key, 0.0)
            label = criteria_labels.get(key, key)
            
            if score >= 0.75: # 合致ライン
                matched_points.append(label)
            elif score <= 0.45: # 不一致ライン
                compromise_points.append(label)

        # --- 3. 分野マッチングの状況 ---
        match_type = field_detail.get("match_type", "unknown")
        # 学生が分野を重視しているか (優先度7以上)
        is_field_prioritized = student.get("research_field_match", 5) >= 7

        # --- 4. 文章の組み立て (Narrative Construction) ---
        sentences = []

        # 【A. 決定打（キラーフレーズ）】
        if is_field_prioritized and match_type == "exact":
            sentences.append("あなたが重視する研究分野が完全一致しており、専門性を高めるのに最適な環境です。")
        elif matched_points:
            # マッチしたこだわりポイントを強調（最大2つ）
            points_str = "』『".join(matched_points[:2])
            sentences.append(f"あなたが最優先した『{points_str}』の方針が、この研究室の実態と強く合致しています。")
        elif total >= 0.85:
            sentences.append("全体的なバランスが極めて良く、あなたの希望条件を高い水準で満たしています。")

        # 【B. 展開・補足】
        if match_type == "exact" and not is_field_prioritized:
            sentences.append("研究分野も一致しているため、スムーズに研究に着手できるでしょう。")
        elif match_type != "exact" and basic_score >= 0.75:
            sentences.append("分野は異なりますが、ゼミの雰囲気や指導方針といった「活動スタイル」の相性が抜群です。")
        
        # 【C. 正直な懸念点（信頼性の向上）】
        # 良いことだけでなく「ここは違う」と認めることで説得力が増す
        if compromise_points:
            point = compromise_points[0]
            sentences.append(f"『{point}』に関しては希望と少し差がありますが、総合的な適合度は十分に高いです。")

        # 【D. 結び（条件に当てはまらない場合）】
        if not sentences:
            if total >= 0.7:
                sentences.append("大きな欠点がなく、安定して研究に取り組める環境と言えます。")
            else:
                sentences.append("一部の条件で妥協が必要ですが、検討候補として有力です。")

        return "".join(sentences)
    
    def _get_recommendation(self, score: float) -> str:
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


# テスト用
if __name__ == "__main__":
    print("=" * 60)
    print("FuzzyMultiPathMatcher テスト（分野重視改善版 + 説明生成）")
    print("=" * 60)
    
    matcher = FuzzyMultiPathMatcher()
    
    # テスト用学生プロファイル
    student = {
        "research_intensity": 9,
        "advisor_style": 7,
        "team_work": 5,
        "workload": 8,
        "theory_practice": 6,
        "research_field_match": 9,  # 分野重視
        "skill_development": 7,
        "lab_atmosphere": 6,
        "flexibility": 5,
        "publication_opportunity": 9,
        "interdisciplinary": 4,
        "communication_style": 6,
        
        # 優先度
        "research_intensity_priority": 10,
        "publication_opportunity_priority": 10,
        "workload_priority": 7,
        "skill_development_priority": 6,
        
        # 分野興味
        "field_interests": {"ai_ml": 10, "image_processing": 7}
    }
    
    # テスト用研究室プロファイル
    lab = {
        "name": "AI研究室",
        "research_area": "人工知能・機械学習",
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
        "field_id": "ai_ml"
    }
    
    result = matcher.calculate_compatibility(student, lab)
    
    print(f"\n総合適合度: {result.total_compatibility:.3f}")
    print(f"基本スコア: {result.basic_score:.3f}")
    print(f"分野スコア: {result.field_score:.3f}")
    print(f"分野比重α: {result.field_weight_alpha:.3f}")
    print(f"基本比重β: {result.basic_weight_beta:.3f}")
    print(f"評価パス数: {len(result.fuzzy_paths)}")
    print(f"推薦: {result.recommendation}")
    print(f"\n【従来版説明】")
    print(result.explanation)
    print(f"\n【詳細版説明】")
    print(result.explanation_detailed)
    print(f"\n【短縮版説明】")
    print(result.explanation_short)