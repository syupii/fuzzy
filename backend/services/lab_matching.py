# services/lab_matching.py - 改善版（主要部分の抜粋）

"""
改善点:
1. _calculate_complete_compatibility でσ値を0.25に調整
2. 完全一致・近似一致にボーナス追加
3. 優先度の重み付けを非線形化（^1.8）
4. 高優先度項目での適合にボーナス
"""

def _calculate_complete_compatibility(
    self, 
    student_profile: StudentProfile,
    laboratory: Laboratory
) -> CompatibilityScore:
    """完全な適合性計算（13項目対応）- 改善版"""
    
    # 学生の優先度を取得
    priorities = self._extract_priorities(student_profile)
    
    # 各基準の類似度計算（改善版）
    criteria_scores = {}
    weighted_sum = 0.0
    total_weight = 0.0
    
    for criterion in self.COMPLETE_CRITERIA:
        # 値取得
        student_value = self._get_criterion_value(student_profile, criterion)
        lab_value = self._get_criterion_value(laboratory, criterion)
        
        if student_value is None or lab_value is None:
            continue
        
        # ★ 改善: 類似度計算（σ=0.25、ボーナス付き）
        similarity = self._calculate_similarity_improved(
            student_value, lab_value
        )
        
        # 優先度取得
        priority = priorities.get(criterion, 5.0)
        
        # ★ 改善: 優先度の非線形重み付け
        weight = self._calculate_priority_weight_improved(priority)
        
        criteria_scores[criterion] = similarity
        weighted_sum += similarity * weight
        total_weight += weight
    
    # 基本スコア
    base_score = weighted_sum / total_weight if total_weight > 0 else 0.5
    
    # ★ 改善: 高優先度項目でのボーナス計算
    priority_bonus = self._calculate_priority_bonus(
        priorities, criteria_scores
    )
    
    # 分野適合度
    field_match_score = self._calculate_field_compatibility(
        student_profile, laboratory
    )
    
    # 最終スコア（分野重視度で統合）
    rfm = self._get_criterion_value(student_profile, 'research_field_match')
    if rfm is not None:
        alpha = rfm / 10.0
        beta = 1.0 - alpha
        overall_score = beta * base_score + alpha * field_match_score
    else:
        overall_score = base_score * 0.7 + field_match_score * 0.3
    
    # 優先度ボーナスを加算（最大+15%）
    overall_score = min(1.0, overall_score + priority_bonus)
    
    # 信頼度計算
    confidence = self._calculate_confidence(
        criteria_scores, priorities, overall_score
    )
    
    return CompatibilityScore(
        overall_score=overall_score,
        basic_score=base_score,
        field_match_score=field_match_score,
        criteria_scores=criteria_scores,
        confidence=confidence,
        methodology="improved_fuzzy_with_priority"
    )

def _calculate_similarity_improved(
    self,
    student_value: float,
    lab_value: float
) -> float:
    """
    改善版類似度計算
    
    改善点:
    - σ = 0.25（より緩やか）
    - 完全一致（diff < 0.05）で +15% ボーナス
    - 近似一致（diff < 0.15）で +8% ボーナス
    """
    
    # 正規化
    student_val = self._normalize_value(student_value)
    lab_val = self._normalize_value(lab_value)
    
    difference = abs(student_val - lab_val)
    
    # ガウシアン類似度（σ=0.25）
    sigma = 0.25  # ★ 0.2 → 0.25
    base_similarity = np.exp(-(difference ** 2) / (2 * sigma ** 2))
    
    # ボーナス適用
    if difference < 0.05:
        # 完全一致: +15%
        similarity = min(1.0, base_similarity * 1.15)
    elif difference < 0.15:
        # 近似一致: +8%
        similarity = min(1.0, base_similarity * 1.08)
    else:
        similarity = base_similarity
    
    return float(similarity)

def _calculate_priority_weight_improved(self, priority: float) -> float:
    """
    優先度から重みを計算（非線形変換 - 改善版）
    
    priority 10 → weight 1.00
    priority  8 → weight 0.72
    priority  5 → weight 0.32
    priority  3 → weight 0.11
    priority  1 → weight 0.01
    """
    
    normalized = priority / 10.0
    weight = normalized ** 1.8  # ★ 指数を大きく（1.5 → 1.8）
    return weight

def _calculate_priority_bonus(
    self,
    priorities: Dict[str, float],
    criteria_scores: Dict[str, float]
) -> float:
    """
    高優先度項目での適合度に基づくボーナス計算
    
    高優先度項目（priority >= 7）での適合度が高い場合にボーナス
    最大 +0.15（15%）
    """
    
    # 高優先度項目の抽出
    high_priority_criteria = [
        criterion for criterion, priority in priorities.items()
        if priority >= 7.0
    ]
    
    if not high_priority_criteria:
        return 0.0
    
    # 高優先度項目の平均適合度
    high_priority_scores = [
        criteria_scores.get(criterion, 0.5)
        for criterion in high_priority_criteria
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

def _calculate_confidence(
    self,
    criteria_scores: Dict[str, float],
    priorities: Dict[str, float],
    overall_score: float
) -> float:
    """
    信頼度計算（改善版）
    
    - 高優先度項目の一致度を重視
    - スコアのばらつきも考慮
    """
    
    # 高優先度項目の適合度
    high_priority_criteria = [
        c for c, p in priorities.items() if p >= 7.0
    ]
    
    if high_priority_criteria:
        high_scores = [
            criteria_scores.get(c, 0.5) 
            for c in high_priority_criteria
        ]
        high_avg = np.mean(high_scores)
    else:
        high_avg = overall_score
    
    # 全体のばらつき
    all_scores = list(criteria_scores.values())
    if len(all_scores) > 1:
        std_dev = np.std(all_scores)
        consistency = 1.0 - std_dev  # ばらつきが小さいほど高信頼
    else:
        consistency = 1.0
    
    # 信頼度 = 高優先度適合度 * 0.7 + 一貫性 * 0.3
    confidence = high_avg * 0.7 + consistency * 0.3
    
    return min(1.0, confidence)

def _generate_complete_explanations(
    self, 
    student_profile: StudentProfile,
    laboratory: Laboratory,
    compatibility_score: CompatibilityScore
) -> Dict[str, Any]:
    """完全な説明生成（改善版）"""
    
    reasons = []
    concerns = []
    
    criteria_scores = compatibility_score.criteria_scores
    
    # スコアで項目をソート
    sorted_criteria = sorted(
        criteria_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # 上位3項目（強み）
    high_score_criteria = [
        (c, s) for c, s in sorted_criteria[:5] if s >= 0.75
    ]
    
    if high_score_criteria:
        criteria_names = [self.CRITERIA_NAMES.get(c, c) for c, _ in high_score_criteria[:3]]
        reasons.append(f"{', '.join(criteria_names)}で高い適合性")
    
    # 下位項目（懸念）
    low_score_criteria = [
        (c, s) for c, s in sorted_criteria if s < 0.5
    ]
    
    if len(low_score_criteria) >= 2:
        criteria_names = [self.CRITERIA_NAMES.get(c, c) for c, _ in low_score_criteria[:2]]
        concerns.append(f"{', '.join(criteria_names)}で適合性がやや低い")
    
    # 分野適合
    if compatibility_score.field_match_score > 0.7:
        reasons.append("研究分野が興味と一致")
    elif compatibility_score.field_match_score < 0.3:
        concerns.append("研究分野が興味と異なる")
    
    # 総合評価
    if compatibility_score.overall_score > 0.8:
        reasons.append("総合的に非常に高い適合性")
    elif compatibility_score.overall_score < 0.4:
        concerns.append("総合的な適合度が低め")
    
    # 詳細分析
    detailed_analysis = {
        "top_matches": [
            {
                "criterion": self.CRITERIA_NAMES.get(c, c),
                "score": round(s, 3),
                "level": "excellent" if s >= 0.85 else "good"
            }
            for c, s in high_score_criteria[:5]
        ],
        "areas_of_concern": [
            {
                "criterion": self.CRITERIA_NAMES.get(c, c),
                "score": round(s, 3),
                "level": "low"
            }
            for c, s in low_score_criteria[:3]
        ],
        "field_analysis": {
            "field_match_score": compatibility_score.field_match_score,
            "field_importance": "high" if compatibility_score.field_match_score > 0.7 else "medium"
        }
    }
    
    return {
        "reasons": reasons,
        "concerns": concerns if concerns else ["特に大きな懸念点はありません"],
        "detailed_analysis": detailed_analysis
    }

def _determine_recommendation_level(self, overall_score: float) -> str:
    """推薦レベル決定（改善版：閾値調整）"""
    
    if overall_score >= 0.80:
        return "強く推薦"
    elif overall_score >= 0.65:
        return "推薦"
    elif overall_score >= 0.45:
        return "検討可能"
    elif overall_score >= 0.30:
        return "要慎重検討"
    else:
        return "推奨しない"