# backend/core/matching/explanation_generator.py
"""
自然言語による判断根拠説明生成モジュール

学生にとって分かりやすい、人間が読んで理解しやすい
推薦理由を生成します。
"""

from typing import Dict, List, Any, Tuple


# 評価項目の日本語名と説明
CRITERIA_INFO = {
    "research_intensity": {
        "name": "研究強度",
        "low_desc": "ゆるやかなペース",
        "high_desc": "集中的な研究",
        "student_low": "軽めの研究を希望",
        "student_high": "集中的な研究を希望",
        "lab_low": "ゆるやかなペースで研究を進める",
        "lab_high": "研究に力を入れている",
    },
    "advisor_style": {
        "name": "指導スタイル",
        "low_desc": "きめ細やかな指導",
        "high_desc": "自主性を重視",
        "student_low": "丁寧な指導を希望",
        "student_high": "自由に研究を進めたい",
        "lab_low": "教授が丁寧に指導する",
        "lab_high": "学生の自主性を尊重する",
    },
    "team_work": {
        "name": "チームワーク",
        "low_desc": "個人研究中心",
        "high_desc": "チーム研究中心",
        "student_low": "一人で黙々と研究したい",
        "student_high": "仲間と協力して研究したい",
        "lab_low": "個人での研究が中心",
        "lab_high": "チームでの共同研究が多い",
    },
    "workload": {
        "name": "活動量",
        "low_desc": "余裕のあるペース",
        "high_desc": "忙しい",
        "student_low": "プライベートも大切にしたい",
        "student_high": "忙しくても充実した活動をしたい",
        "lab_low": "比較的余裕のあるスケジュール",
        "lab_high": "活動量が多く忙しい",
    },
    "theory_practice": {
        "name": "理論・実践バランス",
        "low_desc": "理論重視",
        "high_desc": "実践重視",
        "student_low": "理論をしっかり学びたい",
        "student_high": "手を動かして実践したい",
        "lab_low": "理論的な研究が中心",
        "lab_high": "実践的なものづくりが中心",
    },
    "skill_development": {
        "name": "スキル開発",
        "low_desc": "専門特化",
        "high_desc": "幅広いスキル",
        "student_low": "一つの専門を深めたい",
        "student_high": "幅広いスキルを身につけたい",
        "lab_low": "専門分野を深く追求する",
        "lab_high": "多様なスキルが身につく",
    },
    "lab_atmosphere": {
        "name": "研究室の雰囲気",
        "low_desc": "静かで集中できる",
        "high_desc": "活発で賑やか",
        "student_low": "静かな環境で集中したい",
        "student_high": "活発な議論のある環境が良い",
        "lab_low": "落ち着いた静かな雰囲気",
        "lab_high": "活発で賑やかな雰囲気",
    },
    "flexibility": {
        "name": "時間の柔軟性",
        "low_desc": "決まったスケジュール",
        "high_desc": "柔軟なスケジュール",
        "student_low": "決まった時間に活動したい",
        "student_high": "自分のペースで活動したい",
        "lab_low": "コアタイムなど決まった活動時間がある",
        "lab_high": "時間の使い方は自由",
    },
    "publication_opportunity": {
        "name": "論文発表の機会",
        "low_desc": "発表は少なめ",
        "high_desc": "発表機会が豊富",
        "student_low": "論文発表にはこだわらない",
        "student_high": "積極的に発表したい",
        "lab_low": "学会発表は必須ではない",
        "lab_high": "学会発表の機会が多い",
    },
    "interdisciplinary": {
        "name": "学際性",
        "low_desc": "単一分野",
        "high_desc": "分野横断",
        "student_low": "一つの分野を深めたい",
        "student_high": "複数分野にまたがる研究がしたい",
        "lab_low": "専門分野に特化している",
        "lab_high": "他分野との連携が活発",
    },
    "communication_style": {
        "name": "コミュニケーション",
        "low_desc": "少人数で密接",
        "high_desc": "オープンで多人数",
        "student_low": "少人数でじっくり交流したい",
        "student_high": "多くの人と交流したい",
        "lab_low": "少人数での密な関係",
        "lab_high": "オープンで多くの交流がある",
    },
}


def generate_detailed_explanation(
    lab: Dict[str, Any],
    student: Dict[str, Any],
    criteria_scores: Dict[str, float],
    field_score: float,
    field_detail: Dict[str, Any],
    final_score: float,
    alpha: float,
    beta: float
) -> str:
    """
    人間が読みやすい詳細な推薦理由を生成
    
    Args:
        lab: 研究室データ
        student: 学生プロファイル（正規化済み）
        criteria_scores: 項目別類似度スコア
        field_score: 分野スコア
        field_detail: 分野マッチング詳細
        final_score: 最終適合度
        alpha: 分野の比重
        beta: 基本項目の比重
    
    Returns:
        自然言語による説明文
    """
    sections = []
    
    lab_name = lab.get("name", "この研究室")
    
    # ========================================
    # 1. 分野マッチングの説明
    # ========================================
    field_explanation = _generate_field_section(
        lab, student, field_score, field_detail, alpha
    )
    if field_explanation:
        sections.append(field_explanation)
    
    # ========================================
    # 2. 相性の良い点（強み）
    # ========================================
    strengths = _generate_strengths_section(
        lab, student, criteria_scores
    )
    if strengths:
        sections.append(strengths)
    
    # ========================================
    # 3. 事前に知っておくと良い点（注意点）
    # ========================================
    concerns = _generate_concerns_section(
        lab, student, criteria_scores
    )
    if concerns:
        sections.append(concerns)
    
    # ========================================
    # 4. 総合評価
    # ========================================
    summary = _generate_summary_section(final_score, lab_name)
    sections.append(summary)
    
    return "\n\n".join(sections)


def _generate_field_section(
    lab: Dict[str, Any],
    student: Dict[str, Any],
    field_score: float,
    field_detail: Dict[str, Any],
    alpha: float
) -> str:
    """分野マッチングの説明を生成"""
    
    match_type = field_detail.get("match_type", "unknown")
    lab_field = lab.get("research_area", lab.get("field_id", ""))
    
    # 学生が分野を重視しているか
    rfm = student.get("research_field_match", 0.5)
    if isinstance(rfm, float) and rfm <= 1.0:
        rfm_display = int(rfm * 9 + 1)
    else:
        rfm_display = int(rfm)
    
    is_field_prioritized = rfm_display >= 7
    
    if match_type == "exact":
        interest_level = field_detail.get("interest_level", 10)
        if is_field_prioritized:
            return (
                f"【研究分野】\n"
                f"あなたが強く興味を持っている分野と「{lab_field}」が完全に一致しています。"
                f"分野を重視されている（重視度{rfm_display}/10）ことから、"
                f"専門性を高めるのに最適な環境と言えます。"
            )
        else:
            return (
                f"【研究分野】\n"
                f"あなたの興味分野と「{lab_field}」が一致しています。"
                f"研究テーマの面でスムーズに取り組めるでしょう。"
            )
    
    elif match_type == "category":
        matched_interest = field_detail.get("matched_interest", "")
        return (
            f"【研究分野】\n"
            f"「{lab_field}」は、あなたが興味を持っている分野と同じカテゴリに属しています。"
            f"直接一致ではありませんが、関連性のある領域で研究を進められます。"
        )
    
    elif match_type == "no_match":
        if is_field_prioritized:
            return (
                f"【研究分野】\n"
                f"「{lab_field}」はあなたの興味分野とは異なります。"
                f"分野を重視されているため、この点は慎重に検討してください。"
                f"ただし、研究スタイルの相性は良い可能性があります。"
            )
        else:
            return (
                f"【研究分野】\n"
                f"「{lab_field}」はあなたの主な興味分野とは異なりますが、"
                f"研究の進め方やゼミの雰囲気が合えば、新しい分野に挑戦する良い機会かもしれません。"
            )
    
    return ""


def _generate_strengths_section(
    lab: Dict[str, Any],
    student: Dict[str, Any],
    criteria_scores: Dict[str, float]
) -> str:
    """強み（高適合項目）の説明を生成"""
    
    # 高優先度かつ高適合の項目を抽出
    high_match_items = []
    
    for criterion, score in criteria_scores.items():
        if criterion == "research_field_match":
            continue
        
        priority_key = f"{criterion}_priority"
        priority = student.get(priority_key, 5.0)
        
        info = CRITERIA_INFO.get(criterion, {})
        name = info.get("name", criterion)
        
        # 学生の希望値とゼミの値を取得（表示用に1-10スケールに戻す）
        student_val = student.get(criterion, 0.5)
        lab_val = lab.get(criterion, 0.5)
        
        # 正規化されている場合は1-10に変換
        if student_val <= 1.0:
            student_display = int(student_val * 9 + 1)
        else:
            student_display = int(student_val)
        
        if lab_val <= 1.0:
            lab_display = int(lab_val * 9 + 1)
        else:
            lab_display = int(lab_val)
        
        # 高適合（80%以上）かつ優先度が高め（6以上）の項目
        if score >= 0.8 and priority >= 6:
            high_match_items.append({
                "criterion": criterion,
                "name": name,
                "score": score,
                "priority": priority,
                "student_val": student_display,
                "lab_val": lab_display,
            })
    
    if not high_match_items:
        return ""
    
    # 優先度とスコアでソート
    high_match_items.sort(key=lambda x: (x["priority"], x["score"]), reverse=True)
    
    # 上位3つを選択
    top_items = high_match_items[:3]
    
    lines = ["【相性の良い点】"]
    
    for item in top_items:
        info = CRITERIA_INFO.get(item["criterion"], {})
        
        # 学生の希望に応じた説明を生成
        if item["student_val"] >= 7:
            student_pref = info.get("student_high", "")
        elif item["student_val"] <= 4:
            student_pref = info.get("student_low", "")
        else:
            student_pref = f"{info.get('name', '')}は中程度を希望"
        
        if item["score"] >= 0.95:
            match_level = "ほぼ完全に一致して"
        elif item["score"] >= 0.85:
            match_level = "非常によく合って"
        else:
            match_level = "よく合って"
        
        lines.append(
            f"・{item['name']}：あなたの希望（{item['student_val']}/10）と"
            f"ゼミの特徴（{item['lab_val']}/10）が{match_level}います。"
        )
    
    return "\n".join(lines)


def _generate_concerns_section(
    lab: Dict[str, Any],
    student: Dict[str, Any],
    criteria_scores: Dict[str, float]
) -> str:
    """注意点（低適合項目）の説明を生成"""
    
    # 高優先度だが低適合の項目を抽出
    concern_items = []
    
    for criterion, score in criteria_scores.items():
        if criterion == "research_field_match":
            continue
        
        priority_key = f"{criterion}_priority"
        priority = student.get(priority_key, 5.0)
        
        info = CRITERIA_INFO.get(criterion, {})
        name = info.get("name", criterion)
        
        # 学生の希望値とゼミの値を取得
        student_val = student.get(criterion, 0.5)
        lab_val = lab.get(criterion, 0.5)
        
        if student_val <= 1.0:
            student_display = int(student_val * 9 + 1)
        else:
            student_display = int(student_val)
        
        if lab_val <= 1.0:
            lab_display = int(lab_val * 9 + 1)
        else:
            lab_display = int(lab_val)
        
        # 低適合（60%未満）かつ優先度が高め（6以上）の項目
        if score < 0.6 and priority >= 6:
            concern_items.append({
                "criterion": criterion,
                "name": name,
                "score": score,
                "priority": priority,
                "student_val": student_display,
                "lab_val": lab_display,
                "diff": abs(student_display - lab_display),
            })
    
    if not concern_items:
        return ""
    
    # 差が大きい順にソート
    concern_items.sort(key=lambda x: x["diff"], reverse=True)
    
    # 上位2つを選択
    top_concerns = concern_items[:2]
    
    lines = ["【事前に知っておくと良い点】"]
    
    for item in top_concerns:
        info = CRITERIA_INFO.get(item["criterion"], {})
        
        student_val = item["student_val"]
        lab_val = item["lab_val"]
        
        # どちらが高いかで説明を変える
        if student_val > lab_val:
            # 学生の希望が高い
            if item["criterion"] == "team_work":
                lines.append(
                    f"・{item['name']}：あなたはチームでの活動を希望していますが（{student_val}/10）、"
                    f"このゼミは個人研究が中心（{lab_val}/10）です。"
                )
            elif item["criterion"] == "workload":
                lines.append(
                    f"・{item['name']}：あなたは忙しい環境を望んでいますが（{student_val}/10）、"
                    f"このゼミは比較的余裕のあるペース（{lab_val}/10）です。"
                )
            elif item["criterion"] == "publication_opportunity":
                lines.append(
                    f"・{item['name']}：あなたは発表機会を重視していますが（{student_val}/10）、"
                    f"このゼミの発表頻度はやや控えめ（{lab_val}/10）です。"
                )
            else:
                lines.append(
                    f"・{item['name']}：あなたの希望（{student_val}/10）と"
                    f"ゼミの特徴（{lab_val}/10）にやや差があります。"
                )
        else:
            # ゼミの値が高い
            if item["criterion"] == "team_work":
                lines.append(
                    f"・{item['name']}：あなたは個人作業を好みますが（{student_val}/10）、"
                    f"このゼミはチームでの活動が多め（{lab_val}/10）です。"
                )
            elif item["criterion"] == "workload":
                lines.append(
                    f"・{item['name']}：あなたは余裕のあるペースを希望していますが（{student_val}/10）、"
                    f"このゼミは活動量が多め（{lab_val}/10）です。"
                )
            elif item["criterion"] == "research_intensity":
                lines.append(
                    f"・{item['name']}：あなたは軽めの研究を希望していますが（{student_val}/10）、"
                    f"このゼミは研究に力を入れています（{lab_val}/10）。"
                )
            else:
                lines.append(
                    f"・{item['name']}：あなたの希望（{student_val}/10）と"
                    f"ゼミの特徴（{lab_val}/10）にやや差があります。"
                )
    
    lines.append("入室前に見学などで確認することをお勧めします。")
    
    return "\n".join(lines)


def _generate_summary_section(final_score: float, lab_name: str) -> str:
    """総合評価の説明を生成"""
    
    score_percent = int(final_score * 100)
    
    if final_score >= 0.85:
        return (
            f"【総合評価】\n"
            f"適合度{score_percent}%：{lab_name}はあなたにとって非常に相性の良い研究室です。"
            f"興味や研究スタイルの面で、充実した活動が期待できます。"
        )
    elif final_score >= 0.70:
        return (
            f"【総合評価】\n"
            f"適合度{score_percent}%：{lab_name}はあなたに合う可能性が高い研究室です。"
            f"一部希望と異なる点もありますが、全体的には良いマッチングと言えます。"
        )
    elif final_score >= 0.55:
        return (
            f"【総合評価】\n"
            f"適合度{score_percent}%：{lab_name}は検討の余地がある研究室です。"
            f"見学や面談を通じて、実際の雰囲気を確かめることをお勧めします。"
        )
    else:
        return (
            f"【総合評価】\n"
            f"適合度{score_percent}%：{lab_name}はあなたの希望とやや異なる特徴を持っています。"
            f"ただし、数値だけでは測れない魅力もあるかもしれません。"
            f"興味があれば直接確認してみてください。"
        )


def generate_short_explanation(
    criteria_scores: Dict[str, float],
    field_detail: Dict[str, Any],
    final_score: float,
    student: Dict[str, Any]
) -> str:
    """
    短い説明文を生成（カード表示用）
    
    3〜4文程度の簡潔な説明
    """
    sentences = []
    
    # 1. 分野マッチング
    match_type = field_detail.get("match_type", "unknown")
    
    if match_type == "exact":
        sentences.append("研究分野が完全に一致しています。")
    elif match_type == "category":
        sentences.append("関連分野での研究が可能です。")
    
    # 2. 高優先度で高適合の項目を1つ
    best_match = None
    best_score = 0
    
    for criterion, score in criteria_scores.items():
        if criterion == "research_field_match":
            continue
        priority = student.get(f"{criterion}_priority", 5.0)
        if score >= 0.8 and priority >= 7 and score > best_score:
            best_score = score
            best_match = criterion
    
    if best_match:
        info = CRITERIA_INFO.get(best_match, {})
        name = info.get("name", best_match)
        sentences.append(f"あなたが重視する「{name}」の相性が良いです。")
    
    # 3. 注意点があれば1つ
    worst_match = None
    worst_score = 1.0
    
    for criterion, score in criteria_scores.items():
        if criterion == "research_field_match":
            continue
        priority = student.get(f"{criterion}_priority", 5.0)
        if score < 0.5 and priority >= 7 and score < worst_score:
            worst_score = score
            worst_match = criterion
    
    if worst_match:
        info = CRITERIA_INFO.get(worst_match, {})
        name = info.get("name", worst_match)
        sentences.append(f"「{name}」については事前確認をお勧めします。")
    
    # 4. 総合評価
    if final_score >= 0.8:
        sentences.append("総合的に非常に相性の良い研究室です。")
    elif final_score >= 0.65:
        sentences.append("総合的に相性の良い研究室です。")
    elif final_score >= 0.5:
        sentences.append("検討候補として有力な研究室です。")
    
    return "".join(sentences)


# テスト用
if __name__ == "__main__":
    # テストデータ
    student = {
        "research_intensity": 0.9,
        "advisor_style": 0.7,
        "team_work": 0.3,
        "workload": 0.8,
        "theory_practice": 0.6,
        "research_field_match": 0.9,
        "skill_development": 0.7,
        "lab_atmosphere": 0.6,
        "flexibility": 0.5,
        "publication_opportunity": 0.9,
        "interdisciplinary": 0.4,
        "communication_style": 0.6,
        
        "research_intensity_priority": 10,
        "team_work_priority": 8,
        "publication_opportunity_priority": 9,
        "workload_priority": 7,
    }
    
    lab = {
        "name": "河原ゼミ",
        "research_area": "ゲームプログラミング",
        "research_intensity": 0.8,
        "advisor_style": 0.7,
        "team_work": 0.8,  # 学生の希望と差がある
        "workload": 0.7,
        "theory_practice": 0.7,
        "skill_development": 0.8,
        "lab_atmosphere": 0.7,
        "flexibility": 0.6,
        "publication_opportunity": 0.8,
        "interdisciplinary": 0.5,
        "communication_style": 0.7,
    }
    
    criteria_scores = {
        "research_intensity": 0.95,
        "advisor_style": 0.98,
        "team_work": 0.45,  # 低い
        "workload": 0.85,
        "theory_practice": 0.85,
        "skill_development": 0.85,
        "lab_atmosphere": 0.85,
        "flexibility": 0.85,
        "publication_opportunity": 0.85,
        "interdisciplinary": 0.85,
        "communication_style": 0.85,
    }
    
    field_detail = {
        "match_type": "exact",
        "lab_field": "game_dev",
        "interest_level": 10,
    }
    
    print("=" * 60)
    print("詳細説明テスト")
    print("=" * 60)
    
    explanation = generate_detailed_explanation(
        lab=lab,
        student=student,
        criteria_scores=criteria_scores,
        field_score=1.0,
        field_detail=field_detail,
        final_score=0.87,
        alpha=0.7,
        beta=0.3,
    )
    
    print(explanation)
    
    print("\n" + "=" * 60)
    print("短い説明テスト")
    print("=" * 60)
    
    short = generate_short_explanation(
        criteria_scores=criteria_scores,
        field_detail=field_detail,
        final_score=0.87,
        student=student,
    )
    
    print(short)