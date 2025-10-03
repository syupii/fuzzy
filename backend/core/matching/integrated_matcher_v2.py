# backend/core/matching/integrated_matcher_v2.py
"""
改善版統合マッチングシステム
- 12項目完全対応 (innovation_risk除外)
- research_field_matchを動的重み係数として使用
- field_interestsを適切に考慮
- シンプルで解釈しやすいスコア計算
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CompatibilityResult:
    """適合度計算結果"""
    total_compatibility: float  # 最終適合度 (0-1)
    basic_score: float  # 基本12項目スコア (0-1)
    field_score: float  # 分野マッチングスコア (0-1)
    field_weight: float  # 分野の重み係数
    basic_weight: float  # 基本項目の重み係数
    criteria_scores: Dict[str, float]  # 項目別スコア
    explanation: str  # 説明文
    breakdown: Dict  # 詳細内訳


class ImprovedIntegratedMatcher:
    """
    改善版統合マッチングシステム
    
    特徴:
    - 12項目すべてを公平に評価
    - research_field_matchは分野と基本項目の比重を制御
    - field_interestsで分野マッチングの質を評価
    - 遺伝的重みを基本項目に適用
    """
    
    # 評価項目リスト (12項目、innovation_risk除外)
    EVALUATION_CRITERIA = [
        # 基本5項目
        "research_intensity",
        "advisor_style", 
        "team_work",
        "workload",
        "theory_practice",
        # 拡張7項目 (research_field_match除く)
        "skill_development",
        "lab_atmosphere",
        "flexibility",
        "publication_opportunity",
        "interdisciplinary",
        "communication_style"
    ]
    
    # デフォルト重み (遺伝的アルゴリズムで最適化される)
    DEFAULT_WEIGHTS = np.array([
        1.2,  # research_intensity
        1.1,  # advisor_style
        1.0,  # team_work
        1.0,  # workload
        1.1,  # theory_practice
        0.9,  # skill_development
        0.8,  # lab_atmosphere
        0.8,  # flexibility
        1.0,  # publication_opportunity
        0.7,  # interdisciplinary
        0.8   # communication_style
    ])
    
    def __init__(self, optimized_weights: Optional[np.ndarray] = None):
        """
        Args:
            optimized_weights: 遺伝的アルゴリズムで最適化された重み (11項目)
        """
        if optimized_weights is not None and len(optimized_weights) >= 11:
            self.weights = np.array(optimized_weights[:11])
        else:
            self.weights = self.DEFAULT_WEIGHTS
        
        # 正規化
        self.weights = self.weights / np.sum(self.weights)
    
    def calculate_compatibility(
        self, 
        student: Dict, 
        lab: Dict
    ) -> CompatibilityResult:
        """
        統合適合度を計算
        
        Args:
            student: 学生プロファイル
                - 基本12項目の評価値 (1-10)
                - research_field_match: 分野重視度 (1-10)
                - field_interests: 分野別興味度 {field_id: interest_level}
            
            lab: 研究室プロファイル
                - 基本12項目の特性値 (1-10 or 0-1)
                - field_id: 研究分野ID
        
        Returns:
            CompatibilityResult
        """
        
        # ===== ステップ1: 基本12項目の類似度計算 =====
        criteria_scores, basic_weighted_score = self._calculate_basic_criteria_scores(
            student, lab
        )
        
        # ===== ステップ2: 分野マッチングスコア計算 =====
        field_score = self._calculate_field_matching_score(
            student.get("field_interests", {}),
            lab.get("field_id", "unknown")
        )
        
        # ===== ステップ3: research_field_matchによる比重決定 =====
        field_match_pref = student.get("research_field_match", 5.0)
        alpha = field_match_pref / 10.0  # 分野の比重 (0.1 ~ 1.0)
        beta = 1.0 - alpha  # 基本項目の比重
        
        # ===== ステップ4: 動的重み付け統合 =====
        # 分野スコアに動的ブースト係数を適用
        field_boost_factor = 1.5  # 分野重視の場合のブースト
        effective_field_score = field_score * (1.0 + (alpha - 0.5) * field_boost_factor)
        effective_field_score = np.clip(effective_field_score, 0, 1)
        
        # 重み付き平均による最終スコア
        total_compatibility = (
            beta * basic_weighted_score + 
            alpha * effective_field_score
        )
        
        # 正規化 (0-1の範囲に収める)
        total_compatibility = np.clip(total_compatibility, 0, 1)
        
        # ===== ステップ5: 説明文生成 =====
        explanation = self._generate_explanation(
            total_compatibility,
            basic_weighted_score,
            field_score,
            alpha,
            beta,
            criteria_scores
        )
        
        # ===== 詳細内訳 =====
        breakdown = {
            "basic_weighted_score": float(basic_weighted_score),
            "field_raw_score": float(field_score),
            "field_effective_score": float(effective_field_score),
            "field_boost_factor": field_boost_factor,
            "alpha": float(alpha),
            "beta": float(beta),
            "criteria_count": len(self.EVALUATION_CRITERIA),
            "top_3_criteria": self._get_top_criteria(criteria_scores, 3),
            "bottom_3_criteria": self._get_bottom_criteria(criteria_scores, 3)
        }
        
        return CompatibilityResult(
            total_compatibility=float(total_compatibility),
            basic_score=float(basic_weighted_score),
            field_score=float(field_score),
            field_weight=float(alpha),
            basic_weight=float(beta),
            criteria_scores=criteria_scores,
            explanation=explanation,
            breakdown=breakdown
        )
    
    def _calculate_basic_criteria_scores(
        self,
        student: Dict,
        lab: Dict
    ) -> Tuple[Dict[str, float], float]:
        """
        基本12項目の類似度計算
        
        Returns:
            (項目別スコア辞書, 重み付き総合スコア)
        """
        criteria_scores = {}
        weighted_sum = 0.0
        total_weight = 0.0
        
        for i, criterion in enumerate(self.EVALUATION_CRITERIA):
            student_val = student.get(criterion)
            lab_val = lab.get(criterion)
            
            if student_val is None or lab_val is None:
                criteria_scores[criterion] = 0.5  # データ不足の場合は中間値
                continue
            
            # 正規化 (1-10 → 0-1)
            student_val = float(student_val)
            lab_val = float(lab_val)
            
            if student_val > 1:
                student_val = (student_val - 1) / 9
            if lab_val > 1:
                lab_val = (lab_val - 1) / 9
            
            # 類似度計算 (差分の逆数)
            diff = abs(student_val - lab_val)
            similarity = 1.0 - diff
            
            criteria_scores[criterion] = float(similarity)
            
            # 重み付き合計
            weight = self.weights[i] if i < len(self.weights) else 1.0
            weighted_sum += similarity * weight
            total_weight += weight
        
        # 重み付き平均
        weighted_average = weighted_sum / total_weight if total_weight > 0 else 0.5
        
        return criteria_scores, weighted_average
    
    def _calculate_field_matching_score(
        self,
        field_interests: Dict[str, float],
        lab_field_id: str
    ) -> float:
        """
        分野マッチングスコア計算
        
        Args:
            field_interests: {field_id: interest_level (1-10)}
            lab_field_id: 研究室の分野ID
        
        Returns:
            0.0 ~ 1.0 のスコア
        """
        if not field_interests or not lab_field_id:
            return 0.0
        
        # 完全一致の場合
        if lab_field_id in field_interests:
            interest_level = field_interests[lab_field_id]
            normalized = interest_level / 10.0
            return np.clip(normalized, 0, 1)
        
        # 部分一致の場合 (カテゴリベース)
        # 例: "ai_ml"と"image_processing"は両方"technology"カテゴリ
        lab_category = self._get_field_category(lab_field_id)
        
        max_score = 0.0
        for field_id, interest_level in field_interests.items():
            field_category = self._get_field_category(field_id)
            
            if field_category == lab_category:
                # 同じカテゴリの場合は70%のスコア
                normalized = interest_level / 10.0 * 0.7
                max_score = max(max_score, normalized)
        
        return np.clip(max_score, 0, 1)
    
    def _get_field_category(self, field_id: str) -> str:
        """分野のカテゴリを取得"""
        
        # テクノロジー系
        tech_fields = [
            "ai_ml", "image_processing", "network_security",
            "database_systems", "embedded_iot", "education_linguistics",
            "natural_science_math", "tourism_regional", "business_decision",
            "audio_processing", "system_ethics"
        ]
        
        # クリエイティブ系
        creative_fields = [
            "web_design", "design_visual", "video_animation", "computer_music"
        ]
        
        # エンターテイメント系
        entertainment_fields = ["game_esports", "vr_ar_media"]
        
        # 人文・社会・体育系
        humanities_fields = ["philosophy_humanities", "sports_science"]
        
        if field_id in tech_fields:
            return "technology"
        elif field_id in creative_fields:
            return "creative"
        elif field_id in entertainment_fields:
            return "entertainment"
        elif field_id in humanities_fields:
            return "humanities"
        else:
            return "unknown"
    
    def _generate_explanation(
        self,
        total: float,
        basic: float,
        field: float,
        alpha: float,
        beta: float,
        criteria_scores: Dict[str, float]
    ) -> str:
        """説明文を生成"""
        
        parts = []
        
        # 総合評価
        if total >= 0.85:
            parts.append("✅ 非常に高い適合度")
        elif total >= 0.70:
            parts.append("⭐ 高い適合度")
        elif total >= 0.50:
            parts.append("⚠️ 中程度の適合度")
        else:
            parts.append("❌ 低い適合度")
        
        # 比重の説明
        if alpha > 0.7:
            parts.append(f"分野重視型 (分野{int(alpha*100)}% : 基本{int(beta*100)}%)")
        elif beta > 0.7:
            parts.append(f"基本項目重視型 (基本{int(beta*100)}% : 分野{int(alpha*100)}%)")
        else:
            parts.append(f"バランス型 (分野{int(alpha*100)}% : 基本{int(beta*100)}%)")
        
        # 分野評価
        if alpha > 0.5:  # 分野を重視している場合のみ言及
            if field >= 0.8:
                parts.append("興味分野と完全一致")
            elif field >= 0.5:
                parts.append("興味分野と部分的に一致")
            elif field < 0.3:
                parts.append("興味分野との一致が低い")
        
        # 基本項目評価
        if basic >= 0.8:
            parts.append("研究スタイルが非常に合致")
        elif basic >= 0.6:
            parts.append("研究スタイルが概ね合致")
        elif basic < 0.4:
            parts.append("研究スタイルに課題")
        
        # トップ項目
        top_criteria = self._get_top_criteria(criteria_scores, 2)
        if top_criteria:
            criteria_names = [self._get_criterion_name(c) for c, _ in top_criteria]
            parts.append(f"特に優れた適合: {', '.join(criteria_names)}")
        
        return " / ".join(parts)
    
    def _get_top_criteria(
        self, 
        criteria_scores: Dict[str, float], 
        n: int = 3
    ) -> List[Tuple[str, float]]:
        """上位N項目を取得"""
        sorted_criteria = sorted(
            criteria_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_criteria[:n]
    
    def _get_bottom_criteria(
        self, 
        criteria_scores: Dict[str, float], 
        n: int = 3
    ) -> List[Tuple[str, float]]:
        """下位N項目を取得"""
        sorted_criteria = sorted(
            criteria_scores.items(), 
            key=lambda x: x[1]
        )
        return sorted_criteria[:n]
    
    def _get_criterion_name(self, criterion: str) -> str:
        """項目名の日本語変換"""
        names = {
            "research_intensity": "研究強度",
            "advisor_style": "指導スタイル",
            "team_work": "チームワーク",
            "workload": "ワークロード",
            "theory_practice": "理論・実践",
            "skill_development": "スキル開発",
            "lab_atmosphere": "研究室雰囲気",
            "flexibility": "柔軟性",
            "publication_opportunity": "発表機会",
            "interdisciplinary": "学際性",
            "communication_style": "コミュニケーション"
        }
        return names.get(criterion, criterion)


# ===== 使用例 =====
if __name__ == "__main__":
    print("=" * 70)
    print("改善版統合マッチャー テスト")
    print("=" * 70)
    
    # サンプルデータ
    student_profile = {
        # 基本5項目
        "research_intensity": 8,
        "advisor_style": 7,
        "team_work": 6,
        "workload": 7,
        "theory_practice": 8,
        # 拡張7項目
        "skill_development": 7,
        "lab_atmosphere": 6,
        "flexibility": 8,
        "publication_opportunity": 7,
        "interdisciplinary": 6,
        "communication_style": 7,
        # 分野関連
        "research_field_match": 8,  # 分野を重視 (80%)
        "field_interests": {
            "ai_ml": 9,
            "image_processing": 7,
            "web_design": 5
        }
    }
    
    lab_profile = {
        "id": "lab_001",
        "name": "AI研究室",
        "field_id": "ai_ml",
        # 研究室特性
        "research_intensity": 0.9,
        "advisor_style": 0.6,
        "team_work": 0.8,
        "workload": 0.8,
        "theory_practice": 0.7,
        "skill_development": 0.8,
        "lab_atmosphere": 0.7,
        "flexibility": 0.6,
        "publication_opportunity": 0.8,
        "interdisciplinary": 0.7,
        "communication_style": 0.8
    }
    
    # マッチャー初期化
    matcher = ImprovedIntegratedMatcher()
    
    # 適合度計算
    result = matcher.calculate_compatibility(student_profile, lab_profile)
    
    # 結果表示
    print(f"\n🎯 適合度評価結果")
    print(f"{'=' * 70}")
    print(f"研究室: {lab_profile['name']}")
    print(f"総合適合度: {result.total_compatibility:.3f} ({result.total_compatibility * 100:.1f}%)")
    print(f"\n📊 スコア内訳:")
    print(f"  基本12項目スコア: {result.basic_score:.3f}")
    print(f"  分野マッチングスコア: {result.field_score:.3f}")
    print(f"\n⚖️ 重み配分:")
    print(f"  分野の比重: {result.field_weight:.1%}")
    print(f"  基本項目の比重: {result.basic_weight:.1%}")
    print(f"\n💡 説明:")
    print(f"  {result.explanation}")
    
    print(f"\n📈 項目別適合度 (上位5項目):")
    top_5 = result.breakdown["top_3_criteria"][:5] if "top_3_criteria" in result.breakdown else []
    for criterion, score in top_5:
        name = matcher._get_criterion_name(criterion)
        print(f"  {name}: {score:.3f}")
    
    print(f"\n⚠️ 注意項目 (下位3項目):")
    bottom_3 = result.breakdown["bottom_3_criteria"][:3] if "bottom_3_criteria" in result.breakdown else []
    for criterion, score in bottom_3:
        name = matcher._get_criterion_name(criterion)
        print(f"  {name}: {score:.3f}")
    
    print("\n" + "=" * 70)