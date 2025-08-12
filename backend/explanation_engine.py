# backend/explanation_engine.py
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import math
import re


class ExplanationLevel(Enum):
    """説明レベル"""
    BASIC = "basic"           # 基本的な説明
    DETAILED = "detailed"     # 詳細な説明
    EXPERT = "expert"         # 専門的な説明
    USER_FRIENDLY = "user_friendly"  # 一般ユーザー向け


class CertaintyLevel(Enum):
    """確信度レベル"""
    VERY_HIGH = "very_high"   # 90%+
    HIGH = "high"             # 75-90%
    MEDIUM = "medium"         # 50-75%
    LOW = "low"               # 25-50%
    VERY_LOW = "very_low"     # <25%


@dataclass
class ExplanationComponent:
    """説明コンポーネント"""
    type: str                           # 'decision', 'feature', 'confidence', 'comparison'
    content: str                        # 説明文
    confidence: float                   # 信頼度
    importance: float                   # 重要度
    technical_details: Dict = field(default_factory=dict)
    visual_aids: List[str] = field(default_factory=list)


@dataclass
class DecisionExplanation:
    """決定説明"""
    overall_conclusion: str             # 全体結論
    confidence_level: CertaintyLevel   # 確信度レベル
    components: List[ExplanationComponent] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    alternative_scenarios: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class FuzzyExplanationEngine:
    """ファジィ決定木説明エンジン"""

    def __init__(self, language: str = "ja", explanation_level: ExplanationLevel = ExplanationLevel.USER_FRIENDLY):
        self.language = language
        self.explanation_level = explanation_level

        # 説明テンプレート
        self.templates = self._load_explanation_templates()

        # 基準ラベル
        self.criteria_info = {
            'research_intensity': {
                'name_ja': '研究強度',
                'name_en': 'Research Intensity',
                'description_ja': '研究活動の集中度・最先端性',
                'scale_labels_ja': {1: '基礎的', 5: '標準的', 10: '最先端'},
                'importance_keywords': ['革新的', '先端技術', '研究集約', '学術性']
            },
            'advisor_style': {
                'name_ja': '指導スタイル',
                'name_en': 'Advisor Style',
                'description_ja': '教授の指導方針・アプローチ',
                'scale_labels_ja': {1: '厳格', 5: 'バランス', 10: '自由'},
                'importance_keywords': ['メンターシップ', '自主性', '指導密度', '自由度']
            },
            'team_work': {
                'name_ja': 'チームワーク',
                'name_en': 'Team Work',
                'description_ja': '研究での協働・連携の度合い',
                'scale_labels_ja': {1: '個人研究', 5: '混合', 10: 'チーム研究'},
                'importance_keywords': ['協働', '連携', 'コラボレーション', '共同研究']
            },
            'workload': {
                'name_ja': 'ワークロード',
                'name_en': 'Workload',
                'description_ja': '研究の負荷・忙しさの程度',
                'scale_labels_ja': {1: '軽い', 5: '適度', 10: '重い'},
                'importance_keywords': ['負荷', '時間投入', '集中度', '忙しさ']
            },
            'theory_practice': {
                'name_ja': '理論・実践バランス',
                'name_en': 'Theory-Practice Balance',
                'description_ja': '理論研究と実践的研究の比重',
                'scale_labels_ja': {1: '理論重視', 5: 'バランス', 10: '実践重視'},
                'importance_keywords': ['応用性', '実装', '理論的深度', '実用性']
            }
        }

        # 信頼度表現
        self.confidence_expressions = self._get_confidence_expressions()

    def generate_comprehensive_explanation(self,
                                           prediction_result: Dict[str, Any],
                                           lab_info: Dict[str, Any],
                                           user_preferences: Dict[str, float],
                                           decision_steps: List[Dict] = None) -> DecisionExplanation:
        """包括的説明生成"""

        overall_score = prediction_result.get('overall_score', 0.0)
        confidence = prediction_result.get('confidence', 0.0)
        criterion_scores = prediction_result.get('criterion_scores', {})

        # 確信度レベル決定
        confidence_level = self._determine_confidence_level(confidence)

        # 説明コンポーネント生成
        components = []

        # 1. 全体評価説明
        overall_component = self._generate_overall_assessment(
            overall_score, confidence, lab_info['name']
        )
        components.append(overall_component)

        # 2. 基準別詳細説明
        criteria_components = self._generate_criteria_explanations(
            criterion_scores, user_preferences, lab_info
        )
        components.extend(criteria_components)

        # 3. 決定プロセス説明
        if decision_steps:
            process_component = self._generate_process_explanation(
                decision_steps)
            components.append(process_component)

        # 4. 比較・代替案
        comparison_component = self._generate_comparison_insights(
            criterion_scores, user_preferences
        )
        components.append(comparison_component)

        # 推論チェーン構築
        reasoning_chain = self._build_reasoning_chain(
            components, criterion_scores)

        # 代替シナリオ
        alternative_scenarios = self._generate_alternative_scenarios(
            user_preferences, criterion_scores, overall_score
        )

        # 注意点・制約
        caveats = self._generate_caveats(confidence, criterion_scores)

        # 推奨事項
        recommendations = self._generate_recommendations(
            overall_score, criterion_scores, lab_info
        )

        # 全体結論生成
        overall_conclusion = self._generate_overall_conclusion(
            overall_score, confidence_level, lab_info['name']
        )

        return DecisionExplanation(
            overall_conclusion=overall_conclusion,
            confidence_level=confidence_level,
            components=components,
            reasoning_chain=reasoning_chain,
            alternative_scenarios=alternative_scenarios,
            caveats=caveats,
            recommendations=recommendations
        )

    def _generate_overall_assessment(self, score: float, confidence: float, lab_name: str) -> ExplanationComponent:
        """全体評価説明生成"""

        # スコア評価
        if score >= 85:
            score_description = "非常に高い適合度"
            score_emoji = "🎯"
        elif score >= 70:
            score_description = "高い適合度"
            score_emoji = "✅"
        elif score >= 55:
            score_description = "良好な適合度"
            score_emoji = "👍"
        elif score >= 40:
            score_description = "中程度の適合度"
            score_emoji = "⚠️"
        else:
            score_description = "低い適合度"
            score_emoji = "❌"

        # 信頼度表現
        confidence_text = self.confidence_expressions[self._determine_confidence_level(
            confidence)]

        content = f"{score_emoji} **{lab_name}**との適合度は{score:.1f}点で、{score_description}を示しています。\n\n" \
            f"この結果は{confidence_text}で算出されており、" \
            f"ファジィ論理による多面的分析に基づいています。"

        return ExplanationComponent(
            type="overall_assessment",
            content=content,
            confidence=confidence,
            importance=1.0,
            technical_details={
                'score': score,
                'confidence': confidence,
                'assessment_method': 'fuzzy_logic_multi_criteria'
            }
        )

    def _generate_criteria_explanations(self, criterion_scores: Dict[str, Any],
                                        user_preferences: Dict[str, float],
                                        lab_info: Dict[str, Any]) -> List[ExplanationComponent]:
        """基準別説明生成"""

        components = []

        # 基準を重要度順にソート
        sorted_criteria = sorted(
            criterion_scores.items(),
            key=lambda x: x[1].get('weighted_score', 0.0),
            reverse=True
        )

        for criterion, score_data in sorted_criteria:
            if criterion not in self.criteria_info:
                continue

            criterion_info = self.criteria_info[criterion]
            similarity = score_data.get('similarity', 0.0)
            user_val = score_data.get('user_preference', 0.0)
            lab_val = score_data.get('lab_feature', 0.0)
            weight = score_data.get('weight', 0.0)

            # 適合度評価
            if similarity >= 0.8:
                match_level = "非常によく"
                match_emoji = "🎯"
            elif similarity >= 0.6:
                match_level = "よく"
                match_emoji = "✅"
            elif similarity >= 0.4:
                match_level = "ある程度"
                match_emoji = "👍"
            else:
                match_level = "あまり"
                match_emoji = "⚠️"

            # ユーザー・研究室値の表現
            user_label = self._get_scale_label(criterion, user_val)
            lab_label = self._get_scale_label(criterion, lab_val)

            # 差異分析
            difference = abs(user_val - lab_val)
            if difference <= 1.0:
                difference_desc = "ほぼ一致"
            elif difference <= 2.0:
                difference_desc = "近い値"
            elif difference <= 3.0:
                difference_desc = "やや異なる"
            else:
                difference_desc = "大きく異なる"

            # 説明文生成
            content = f"{match_emoji} **{criterion_info['name_ja']}**\n\n" \
                f"あなたの希望: {user_val:.1f} ({user_label}) | " \
                f"研究室の特徴: {lab_val:.1f} ({lab_label})\n\n" \
                f"この基準では{match_level}マッチしています（類似度: {similarity:.2f}）。" \
                f"両者の値は{difference_desc}で、{criterion_info['description_ja']}の観点から" \
                f"{'適合性が高い' if similarity >= 0.6 else '検討が必要'}といえます。"

            # 重要度に応じた追加説明
            if weight >= 0.25:  # 重要な基準
                content += f"\n\n💡 この基準は全体評価において重要な要因（重み: {weight:.1%}）となっています。"

            components.append(ExplanationComponent(
                type="criteria_analysis",
                content=content,
                confidence=similarity,
                importance=weight,
                technical_details={
                    'criterion': criterion,
                    'similarity': similarity,
                    'user_preference': user_val,
                    'lab_feature': lab_val,
                    'weight': weight,
                    'difference': difference
                }
            ))

        return components

    def _generate_process_explanation(self, decision_steps: List[Dict]) -> ExplanationComponent:
        """決定プロセス説明生成"""

        if not decision_steps:
            return ExplanationComponent(
                type="process_explanation",
                content="決定プロセスの詳細情報は利用できません。",
                confidence=0.0,
                importance=0.3
            )

        # 主要な決定ステップ抽出
        key_steps = [step for step in decision_steps if step.get(
            'confidence', 0) > 0.5]

        if not key_steps:
            key_steps = decision_steps[:3]  # 上位3ステップ

        process_description = "🧠 **AI決定プロセス**\n\n"
        process_description += "ファジィ決定木による段階的分析:\n\n"

        for i, step in enumerate(key_steps, 1):
            feature_name = step.get('feature_name', 'unknown')
            feature_value = step.get('feature_value', 0.0)
            chosen_branch = step.get('chosen_branch', 'unknown')
            confidence = step.get('confidence', 0.0)

            if feature_name in self.criteria_info:
                feature_display = self.criteria_info[feature_name]['name_ja']
            else:
                feature_display = feature_name

            process_description += f"{i}. {feature_display} = {feature_value:.1f} → {chosen_branch} " \
                f"(信頼度: {confidence:.2f})\n"

        process_description += f"\n各段階でのファジィメンバーシップ関数により、" \
            f"あいまいさを考慮した柔軟な判断を実現しています。"

        return ExplanationComponent(
            type="process_explanation",
            content=process_description,
            confidence=np.mean([step.get('confidence', 0)
                               for step in key_steps]),
            importance=0.6,
            technical_details={
                'decision_steps': decision_steps,
                'key_steps_count': len(key_steps)
            }
        )

    def _generate_comparison_insights(self, criterion_scores: Dict[str, Any],
                                      user_preferences: Dict[str, float]) -> ExplanationComponent:
        """比較・洞察説明生成"""

        # 強み・弱み分析
        strengths = []
        weaknesses = []

        for criterion, score_data in criterion_scores.items():
            similarity = score_data.get('similarity', 0.0)

            if criterion in self.criteria_info:
                criterion_name = self.criteria_info[criterion]['name_ja']

                if similarity >= 0.7:
                    strengths.append(criterion_name)
                elif similarity < 0.5:
                    weaknesses.append(criterion_name)

        content = "📊 **適合性分析**\n\n"

        if strengths:
            content += f"**✅ 特に適合している領域:** {', '.join(strengths)}\n\n"

        if weaknesses:
            content += f"**⚠️ 注意が必要な領域:** {', '.join(weaknesses)}\n\n"

        # バランス分析
        score_values = [score_data.get('similarity', 0.0)
                        for score_data in criterion_scores.values()]
        score_variance = np.var(score_values)

        if score_variance < 0.05:
            balance_desc = "各基準において非常にバランスの取れた適合性"
        elif score_variance < 0.1:
            balance_desc = "概ねバランスの取れた適合性"
        else:
            balance_desc = "基準間でばらつきのある適合性"

        content += f"**⚖️ 総合バランス:** {balance_desc}を示しています。"

        # 改善提案
        if weaknesses:
            content += f"\n\n💡 {', '.join(weaknesses[:2])}については、" \
                f"研究室見学や教授との面談で詳細を確認することをお勧めします。"

        return ExplanationComponent(
            type="comparison_insights",
            content=content,
            confidence=1.0 - score_variance,  # 分散が小さいほど信頼度高
            importance=0.7,
            technical_details={
                'strengths': strengths,
                'weaknesses': weaknesses,
                'score_variance': score_variance
            }
        )

    def _build_reasoning_chain(self, components: List[ExplanationComponent],
                               criterion_scores: Dict[str, Any]) -> List[str]:
        """推論チェーン構築"""

        chain = []

        # 前提
        chain.append("ユーザーの希望条件と研究室の特徴を多基準で分析")

        # 主要基準の推論
        sorted_criteria = sorted(
            criterion_scores.items(),
            key=lambda x: x[1].get('weighted_score', 0.0),
            reverse=True
        )

        for criterion, score_data in sorted_criteria[:3]:  # 上位3基準
            if criterion in self.criteria_info:
                similarity = score_data.get('similarity', 0.0)
                criterion_name = self.criteria_info[criterion]['name_ja']

                if similarity >= 0.7:
                    chain.append(f"{criterion_name}で高い適合性を確認")
                elif similarity >= 0.5:
                    chain.append(f"{criterion_name}で適度な適合性を確認")
                else:
                    chain.append(f"{criterion_name}で適合性の課題を識別")

        # 統合判断
        chain.append("ファジィ論理による重み付き統合で総合適合度を算出")

        # 結論
        overall_score = np.mean([score_data.get('similarity', 0.0)
                                for score_data in criterion_scores.values()])
        if overall_score >= 0.7:
            chain.append("高い適合度により推奨判定")
        elif overall_score >= 0.5:
            chain.append("適度な適合度により条件付き推奨")
        else:
            chain.append("低い適合度により慎重検討を推奨")

        return chain

    def _generate_alternative_scenarios(self, user_preferences: Dict[str, float],
                                        criterion_scores: Dict[str, Any],
                                        current_score: float) -> List[str]:
        """代替シナリオ生成"""

        scenarios = []

        # 希望調整シナリオ
        for criterion, score_data in criterion_scores.items():
            similarity = score_data.get('similarity', 0.0)

            if similarity < 0.5 and criterion in self.criteria_info:
                criterion_name = self.criteria_info[criterion]['name_ja']
                user_val = score_data.get('user_preference', 0.0)
                lab_val = score_data.get('lab_feature', 0.0)

                if user_val > lab_val:
                    direction = "下げる"
                else:
                    direction = "上げる"

                scenarios.append(
                    f"{criterion_name}の希望を{direction}ことで、"
                    f"適合度が約{(0.7 - similarity) * 100:.0f}点向上する可能性があります"
                )

        # 重み調整シナリオ
        if len(scenarios) < 3:
            scenarios.append(
                f"基準の重要度を調整することで、"
                f"最大{(100 - current_score) * 0.3:.0f}点程度の適合度向上が期待できます"
            )

        # 他の選択肢示唆
        if current_score < 70:
            scenarios.append(
                "複数の研究室を比較検討し、"
                "より適合度の高い選択肢を探すことをお勧めします"
            )

        return scenarios[:3]  # 最大3つまで

    def _generate_caveats(self, confidence: float, criterion_scores: Dict[str, Any]) -> List[str]:
        """注意点・制約生成"""

        caveats = []

        # 信頼度に基づく注意点
        if confidence < 0.6:
            caveats.append(
                "信頼度が中程度のため、追加情報による検証をお勧めします"
            )

        # データ制約
        caveats.append(
            "この分析は現在利用可能なデータに基づいており、"
            "実際の研究環境は変動する可能性があります"
        )

        # 個人差
        caveats.append(
            "研究室適合性は個人の価値観や目標により大きく左右されるため、"
            "最終判断は総合的に行ってください"
        )

        # 時間的制約
        if any(score_data.get('similarity', 0.0) < 0.4 for score_data in criterion_scores.values()):
            caveats.append(
                "一部基準で適合性が低いため、"
                "将来的な適応可能性も考慮して判断してください"
            )

        return caveats

    def _generate_recommendations(self, overall_score: float,
                                  criterion_scores: Dict[str, Any],
                                  lab_info: Dict[str, Any]) -> List[str]:
        """推奨事項生成"""

        recommendations = []

        # スコア別推奨
        if overall_score >= 80:
            recommendations.append(
                "高い適合度を示しているため、積極的に検討することをお勧めします"
            )
            recommendations.append(
                "研究室見学や教授との面談でより詳細な情報を収集しましょう"
            )
        elif overall_score >= 60:
            recommendations.append(
                "良好な適合度ですが、不安な点については事前に確認しましょう"
            )

            # 低スコア基準の特定
            low_criteria = [
                self.criteria_info[criterion]['name_ja']
                for criterion, score_data in criterion_scores.items()
                if score_data.get('similarity', 0.0) < 0.5 and criterion in self.criteria_info
            ]

            if low_criteria:
                recommendations.append(
                    f"{', '.join(low_criteria[:2])}について特に詳しく相談することをお勧めします"
                )
        else:
            recommendations.append(
                "適合度が低めのため、他の選択肢も併せて検討することをお勧めします"
            )
            recommendations.append(
                "現在の希望条件を見直し、柔軟性を持って検討することも大切です"
            )

        # 一般的推奨
        recommendations.append(
            "複数の研究室を比較し、総合的な判断を行うことが重要です"
        )

        return recommendations

    def _generate_overall_conclusion(self, score: float, confidence_level: CertaintyLevel,
                                     lab_name: str) -> str:
        """全体結論生成"""

        confidence_text = self.confidence_expressions[confidence_level]

        if score >= 85:
            conclusion = f"**{lab_name}**は、あなたの希望に非常によく適合しており、" \
                f"{confidence_text}で強く推奨できる選択肢です。"
        elif score >= 70:
            conclusion = f"**{lab_name}**は、あなたの希望によく適合しており、" \
                f"{confidence_text}で推奨できる選択肢です。"
        elif score >= 55:
            conclusion = f"**{lab_name}**は、あなたの希望にある程度適合していますが、" \
                f"いくつかの点で慎重な検討が必要です。"
        elif score >= 40:
            conclusion = f"**{lab_name}**は、あなたの希望との適合性に課題があり、" \
                f"他の選択肢との比較検討をお勧めします。"
        else:
            conclusion = f"**{lab_name}**は、現在の希望条件との適合性が低く、" \
                f"条件の見直しまたは他の選択肢の検討をお勧めします。"

        return conclusion

    def _determine_confidence_level(self, confidence: float) -> CertaintyLevel:
        """確信度レベル決定"""

        if confidence >= 0.9:
            return CertaintyLevel.VERY_HIGH
        elif confidence >= 0.75:
            return CertaintyLevel.HIGH
        elif confidence >= 0.5:
            return CertaintyLevel.MEDIUM
        elif confidence >= 0.25:
            return CertaintyLevel.LOW
        else:
            return CertaintyLevel.VERY_LOW

    def _get_scale_label(self, criterion: str, value: float) -> str:
        """スケールラベル取得"""

        if criterion not in self.criteria_info:
            return f"{value:.1f}"

        scale_labels = self.criteria_info[criterion]['scale_labels_ja']

        # 最も近いラベルを選択
        closest_key = min(scale_labels.keys(), key=lambda x: abs(x - value))

        if abs(closest_key - value) <= 1.5:
            return scale_labels[closest_key]
        else:
            # 中間値の場合
            if value < 3:
                return "低め"
            elif value > 7:
                return "高め"
            else:
                return "中程度"

    def _load_explanation_templates(self) -> Dict[str, str]:
        """説明テンプレート読み込み"""

        return {
            'overall_positive': "🎯 {lab_name}との適合度は{score:.1f}点で、{assessment}を示しています。",
            'overall_negative': "⚠️ {lab_name}との適合度は{score:.1f}点で、{assessment}にとどまっています。",
            'confidence_high': "この結果は高い信頼度（{confidence:.1%}）で算出されています。",
            'confidence_low': "この結果の信頼度は{confidence:.1%}であり、追加検証をお勧めします。",
            'criteria_match': "✅ {criterion}では{match_level}マッチしています（類似度: {similarity:.2f}）。",
            'criteria_mismatch': "⚠️ {criterion}では適合性に課題があります（類似度: {similarity:.2f}）。"
        }

    def _get_confidence_expressions(self) -> Dict[CertaintyLevel, str]:
        """信頼度表現取得"""

        return {
            CertaintyLevel.VERY_HIGH: "非常に高い信頼度",
            CertaintyLevel.HIGH: "高い信頼度",
            CertaintyLevel.MEDIUM: "中程度の信頼度",
            CertaintyLevel.LOW: "低い信頼度",
            CertaintyLevel.VERY_LOW: "非常に低い信頼度"
        }


class NaturalLanguageGenerator:
    """自然言語生成ヘルパー"""

    @staticmethod
    def format_explanation_for_ui(explanation: DecisionExplanation,
                                  format_type: str = "markdown") -> str:
        """UI向け説明文フォーマット"""

        if format_type == "markdown":
            return NaturalLanguageGenerator._format_markdown(explanation)
        elif format_type == "html":
            return NaturalLanguageGenerator._format_html(explanation)
        else:
            return NaturalLanguageGenerator._format_plain_text(explanation)

    @staticmethod
    def _format_markdown(explanation: DecisionExplanation) -> str:
        """Markdown形式フォーマット"""

        output = []

        # 全体結論
        output.append(f"## 🎯 総合判定\n\n{explanation.overall_conclusion}\n")

        # 詳細分析
        output.append("## 📊 詳細分析\n")

        for component in explanation.components:
            if component.importance >= 0.7:  # 重要なコンポーネントのみ
                output.append(f"{component.content}\n")

        # 推論チェーン
        if explanation.reasoning_chain:
            output.append("## 🧠 AI判断プロセス\n")
            for i, step in enumerate(explanation.reasoning_chain, 1):
                output.append(f"{i}. {step}")
            output.append("")

        # 推奨事項
        if explanation.recommendations:
            output.append("## 💡 推奨事項\n")
            for rec in explanation.recommendations:
                output.append(f"- {rec}")
            output.append("")

        # 注意点
        if explanation.caveats:
            output.append("## ⚠️ 注意点\n")
            for caveat in explanation.caveats:
                output.append(f"- {caveat}")

        return "\n".join(output)

    @staticmethod
    def _format_html(explanation: DecisionExplanation) -> str:
        """HTML形式フォーマット"""

        html_parts = []

        # 全体結論
        html_parts.append(f'<div class="overall-conclusion">')
        html_parts.append(f'<h2>🎯 総合判定</h2>')
        html_parts.append(f'<p>{explanation.overall_conclusion}</p>')
        html_parts.append(f'</div>')

        # 詳細分析
        html_parts.append(f'<div class="detailed-analysis">')
        html_parts.append(f'<h2>📊 詳細分析</h2>')

        for component in explanation.components:
            if component.importance >= 0.7:
                html_parts.append(f'<div class="analysis-component">')
                html_parts.append(f'<p>{component.content}</p>')
                html_parts.append(f'</div>')

        html_parts.append(f'</div>')

        # 推奨事項
        if explanation.recommendations:
            html_parts.append(f'<div class="recommendations">')
            html_parts.append(f'<h2>💡 推奨事項</h2>')
            html_parts.append(f'<ul>')
            for rec in explanation.recommendations:
                html_parts.append(f'<li>{rec}</li>')
            html_parts.append(f'</ul>')
            html_parts.append(f'</div>')

        return '\n'.join(html_parts)

    @staticmethod
    def _format_plain_text(explanation: DecisionExplanation) -> str:
        """プレーンテキスト形式フォーマット"""

        output = []

        # 全体結論
        output.append("=== 総合判定 ===")
        output.append(explanation.overall_conclusion)
        output.append("")

        # 重要なコンポーネント
        output.append("=== 詳細分析 ===")
        for component in explanation.components:
            if component.importance >= 0.7:
                # Markdownマークアップを除去
                clean_content = re.sub(
                    r'\*\*([^*]+)\*\*', r'\1', component.content)
                clean_content = re.sub(r'[🎯✅👍⚠️❌🧠📊💡⚖️]', '', clean_content)
                output.append(clean_content.strip())
                output.append("")

        # 推奨事項
        if explanation.recommendations:
            output.append("=== 推奨事項 ===")
            for i, rec in enumerate(explanation.recommendations, 1):
                output.append(f"{i}. {rec}")

        return "\n".join(output)


class ExplanationValidator:
    """説明品質検証"""

    @staticmethod
    def validate_explanation_quality(explanation: DecisionExplanation) -> Dict[str, Any]:
        """説明品質検証"""

        validation_results = {
            'overall_score': 0.0,
            'completeness': 0.0,
            'clarity': 0.0,
            'consistency': 0.0,
            'usefulness': 0.0,
            'issues': []
        }

        # 完全性チェック
        completeness_score = ExplanationValidator._check_completeness(
            explanation)
        validation_results['completeness'] = completeness_score

        # 明確性チェック
        clarity_score = ExplanationValidator._check_clarity(explanation)
        validation_results['clarity'] = clarity_score

        # 一貫性チェック
        consistency_score = ExplanationValidator._check_consistency(
            explanation)
        validation_results['consistency'] = consistency_score

        # 有用性チェック
        usefulness_score = ExplanationValidator._check_usefulness(explanation)
        validation_results['usefulness'] = usefulness_score

        # 総合スコア
        validation_results['overall_score'] = np.mean([
            completeness_score, clarity_score, consistency_score, usefulness_score
        ])

        return validation_results

    @staticmethod
    def _check_completeness(explanation: DecisionExplanation) -> float:
        """完全性チェック"""

        required_elements = [
            explanation.overall_conclusion,
            explanation.components,
            explanation.recommendations
        ]

        present_elements = sum(1 for element in required_elements if element)
        completeness = present_elements / len(required_elements)

        return completeness

    @staticmethod
    def _check_clarity(explanation: DecisionExplanation) -> float:
        """明確性チェック"""

        clarity_score = 1.0

        # 結論の長さチェック
        if len(explanation.overall_conclusion) < 50:
            clarity_score -= 0.2
        elif len(explanation.overall_conclusion) > 500:
            clarity_score -= 0.1

        # コンポーネント数チェック
        if len(explanation.components) > 8:
            clarity_score -= 0.2
        elif len(explanation.components) < 2:
            clarity_score -= 0.3

        return max(0.0, clarity_score)

    @staticmethod
    def _check_consistency(explanation: DecisionExplanation) -> float:
        """一貫性チェック"""

        # 簡易的な一貫性チェック
        consistency_score = 1.0

        # 信頼度レベルと結論の整合性
        if explanation.confidence_level in [CertaintyLevel.HIGH, CertaintyLevel.VERY_HIGH]:
            if "課題" in explanation.overall_conclusion or "低い" in explanation.overall_conclusion:
                consistency_score -= 0.3

        return max(0.0, consistency_score)

    @staticmethod
    def _check_usefulness(explanation: DecisionExplanation) -> float:
        """有用性チェック"""

        usefulness_score = 0.0

        # 推奨事項の存在
        if explanation.recommendations:
            usefulness_score += 0.4

        # 代替シナリオの存在
        if explanation.alternative_scenarios:
            usefulness_score += 0.3

        # 注意点の存在
        if explanation.caveats:
            usefulness_score += 0.3

        return min(1.0, usefulness_score)
