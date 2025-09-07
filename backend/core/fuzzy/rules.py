# core/fuzzy/rules.py - ファジィルール定義

from typing import Dict, List, Tuple, Callable
from enum import Enum
from dataclasses import dataclass
from models.schemas import StudentProfile, Laboratory

class RuleOperator(Enum):
    """ルール演算子"""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"

@dataclass
class FuzzyCondition:
    """ファジィ条件"""
    variable: str           # 変数名 (例: "interest_level", "experience_level")
    linguistic_value: str   # 言語値 (例: "high", "medium", "low")
    weight: float = 1.0     # 条件の重み

@dataclass
class FuzzyRule:
    """ファジィルール"""
    rule_id: str                          # ルールID
    antecedent: List[FuzzyCondition]      # 前件部（条件）
    consequent: FuzzyCondition            # 後件部（結論）
    operator: RuleOperator = RuleOperator.AND  # 前件部の結合演算子
    confidence: float = 1.0               # ルールの信頼度
    description: str = ""                 # ルール説明

class FuzzyRuleBase:
    """ファジィルールベース"""
    
    def __init__(self):
        self.rules: List[FuzzyRule] = []
        self._initialize_rules()
    
    def _initialize_rules(self):
        """研究室マッチング用のルールを初期化"""
        
        # 分野適合性ルール
        self._add_field_matching_rules()
        
        # 経験レベルルール
        self._add_experience_level_rules()
        
        # 研究スタイルルール
        self._add_research_style_rules()
        
        # 指導スタイルルール
        self._add_advisor_style_rules()
        
        # 総合適合性ルール
        self._add_overall_compatibility_rules()
    
    def _add_field_matching_rules(self):
        """分野適合性に関するルール"""
        
        # ルール1: 高い興味 + 高い重要度 → 高い分野適合性
        self.rules.append(FuzzyRule(
            rule_id="FIELD_001",
            antecedent=[
                FuzzyCondition("interest_level", "high", 0.7),
                FuzzyCondition("importance_level", "high", 0.3)
            ],
            consequent=FuzzyCondition("field_compatibility", "very_high"),
            operator=RuleOperator.AND,
            confidence=0.9,
            description="高い興味と重要度を示す分野は高い適合性を持つ"
        ))
        
        # ルール2: 中程度の興味 + 高い経験 → 良い分野適合性
        self.rules.append(FuzzyRule(
            rule_id="FIELD_002",
            antecedent=[
                FuzzyCondition("interest_level", "medium", 0.5),
                FuzzyCondition("experience_level", "high", 0.5)
            ],
            consequent=FuzzyCondition("field_compatibility", "high"),
            operator=RuleOperator.AND,
            confidence=0.8,
            description="経験豊富な分野は興味が中程度でも適合性が高い"
        ))
        
        # ルール3: 低い興味 → 低い分野適合性
        self.rules.append(FuzzyRule(
            rule_id="FIELD_003",
            antecedent=[
                FuzzyCondition("interest_level", "low", 1.0)
            ],
            consequent=FuzzyCondition("field_compatibility", "low"),
            confidence=0.9,
            description="興味の低い分野は適合性も低い"
        ))
    
    def _add_experience_level_rules(self):
        """経験レベルに関するルール"""
        
        # ルール4: 高い経験 + 高い難易度 → 良いマッチング
        self.rules.append(FuzzyRule(
            rule_id="EXP_001",
            antecedent=[
                FuzzyCondition("experience_level", "high", 0.6),
                FuzzyCondition("field_difficulty", "advanced", 0.4)
            ],
            consequent=FuzzyCondition("experience_match", "high"),
            operator=RuleOperator.AND,
            confidence=0.85,
            description="高い経験を持つ学生は高難易度分野に適合"
        ))
        
        # ルール5: 低い経験 + 低い難易度 → 良いマッチング
        self.rules.append(FuzzyRule(
            rule_id="EXP_002",
            antecedent=[
                FuzzyCondition("experience_level", "low", 0.6),
                FuzzyCondition("field_difficulty", "beginner", 0.4)
            ],
            consequent=FuzzyCondition("experience_match", "high"),
            operator=RuleOperator.AND,
            confidence=0.85,
            description="初心者は基礎的な分野に適合"
        ))
        
        # ルール6: 経験と難易度のミスマッチ → 低いマッチング
        self.rules.append(FuzzyRule(
            rule_id="EXP_003",
            antecedent=[
                FuzzyCondition("experience_level", "low", 0.5),
                FuzzyCondition("field_difficulty", "advanced", 0.5)
            ],
            consequent=FuzzyCondition("experience_match", "low"),
            operator=RuleOperator.AND,
            confidence=0.8,
            description="経験不足で高難易度分野は困難"
        ))
    
    def _add_research_style_rules(self):
        """研究スタイルに関するルール"""
        
        # ルール7: 高い研究強度 + 高い研究強度研究室 → 良いマッチング
        self.rules.append(FuzzyRule(
            rule_id="STYLE_001",
            antecedent=[
                FuzzyCondition("student_research_intensity", "high", 0.6),
                FuzzyCondition("lab_research_intensity", "high", 0.4)
            ],
            consequent=FuzzyCondition("research_style_match", "high"),
            operator=RuleOperator.AND,
            confidence=0.9,
            description="研究に集中したい学生は集中的な研究室に適合"
        ))
        
        # ルール8: チーム志向 + チーム重視研究室 → 良いマッチング
        self.rules.append(FuzzyRule(
            rule_id="STYLE_002",
            antecedent=[
                FuzzyCondition("student_team_work", "high", 0.7),
                FuzzyCondition("lab_team_work", "high", 0.3)
            ],
            consequent=FuzzyCondition("collaboration_match", "high"),
            operator=RuleOperator.AND,
            confidence=0.85,
            description="チーム志向の学生は協働的な研究室に適合"
        ))
        
        # ルール9: 理論志向 + 理論重視研究室 → 良いマッチング
        self.rules.append(FuzzyRule(
            rule_id="STYLE_003",
            antecedent=[
                FuzzyCondition("student_theory_practice", "low", 0.6),  # 1が理論重視
                FuzzyCondition("lab_theory_practice", "low", 0.4)
            ],
            consequent=FuzzyCondition("approach_match", "high"),
            operator=RuleOperator.AND,
            confidence=0.8,
            description="理論志向の学生は理論重視の研究室に適合"
        ))
    
    def _add_advisor_style_rules(self):
        """指導スタイルに関するルール"""
        
        # ルール10: 自由な指導を求める + 自由な指導スタイル → 良いマッチング
        self.rules.append(FuzzyRule(
            rule_id="ADVISOR_001",
            antecedent=[
                FuzzyCondition("student_advisor_style", "high", 0.7),  # 10が自由指導
                FuzzyCondition("lab_advisor_style", "high", 0.3)
            ],
            consequent=FuzzyCondition("advisor_match", "high"),
            operator=RuleOperator.AND,
            confidence=0.9,
            description="自由な指導を求める学生は自由な研究室に適合"
        ))
        
        # ルール11: 厳格な指導を求める + 厳格な指導スタイル → 良いマッチング
        self.rules.append(FuzzyRule(
            rule_id="ADVISOR_002",
            antecedent=[
                FuzzyCondition("student_advisor_style", "low", 0.7),   # 1が厳格指導
                FuzzyCondition("lab_advisor_style", "low", 0.3)
            ],
            consequent=FuzzyCondition("advisor_match", "high"),
            operator=RuleOperator.AND,
            confidence=0.9,
            description="厳格な指導を求める学生は厳格な研究室に適合"
        ))
    
    def _add_overall_compatibility_rules(self):
        """総合適合性ルール"""
        
        # ルール12: 高い分野適合性 + 良い経験マッチ → 非常に高い総合適合性
        self.rules.append(FuzzyRule(
            rule_id="OVERALL_001",
            antecedent=[
                FuzzyCondition("field_compatibility", "high", 0.6),
                FuzzyCondition("experience_match", "high", 0.4)
            ],
            consequent=FuzzyCondition("overall_compatibility", "very_high"),
            operator=RuleOperator.AND,
            confidence=0.95,
            description="分野と経験の両方でマッチする場合は総合適合性が非常に高い"
        ))
        
        # ルール13: 良い研究スタイルマッチ + 良い指導スタイルマッチ → 高い総合適合性
        self.rules.append(FuzzyRule(
            rule_id="OVERALL_002",
            antecedent=[
                FuzzyCondition("research_style_match", "high", 0.5),
                FuzzyCondition("advisor_match", "high", 0.5)
            ],
            consequent=FuzzyCondition("overall_compatibility", "high"),
            operator=RuleOperator.AND,
            confidence=0.9,
            description="研究・指導スタイルがマッチする場合は総合適合性が高い"
        ))
        
        # ルール14: 複数の低マッチング → 低い総合適合性
        self.rules.append(FuzzyRule(
            rule_id="OVERALL_003",
            antecedent=[
                FuzzyCondition("field_compatibility", "low", 0.4),
                FuzzyCondition("experience_match", "low", 0.3),
                FuzzyCondition("research_style_match", "low", 0.3)
            ],
            consequent=FuzzyCondition("overall_compatibility", "low"),
            operator=RuleOperator.AND,
            confidence=0.85,
            description="複数の要素でマッチしない場合は総合適合性が低い"
        ))
        
        # ルール15: 高い革新性志向 + 高い革新性研究室 → 良いマッチング
        self.rules.append(FuzzyRule(
            rule_id="INNOVATION_001",
            antecedent=[
                FuzzyCondition("student_innovation_risk", "high", 0.6),
                FuzzyCondition("lab_innovation_risk", "high", 0.4)
            ],
            consequent=FuzzyCondition("innovation_match", "high"),
            operator=RuleOperator.AND,
            confidence=0.8,
            description="革新的な学生は革新的な研究室に適合"
        ))
    
    def add_custom_rule(self, rule: FuzzyRule):
        """カスタムルールを追加"""
        self.rules.append(rule)
    
    def remove_rule(self, rule_id: str):
        """ルールを削除"""
        self.rules = [rule for rule in self.rules if rule.rule_id != rule_id]
    
    def get_rule(self, rule_id: str) -> FuzzyRule:
        """特定のルールを取得"""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        return None
    
    def get_rules_by_consequent(self, consequent_variable: str) -> List[FuzzyRule]:
        """後件部の変数でルールをフィルタ"""
        return [rule for rule in self.rules if rule.consequent.variable == consequent_variable]
    
    def get_applicable_rules(self, context: Dict[str, float]) -> List[FuzzyRule]:
        """現在のコンテキストに適用可能なルールを取得"""
        applicable_rules = []
        
        for rule in self.rules:
            # 前件部の変数がすべてコンテキストに存在するかチェック
            has_all_variables = all(
                condition.variable in context 
                for condition in rule.antecedent
            )
            
            if has_all_variables:
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def validate_rules(self) -> List[str]:
        """ルールの妥当性をチェック"""
        issues = []
        
        rule_ids = [rule.rule_id for rule in self.rules]
        if len(rule_ids) != len(set(rule_ids)):
            issues.append("重複するルールIDが存在します")
        
        for rule in self.rules:
            if rule.confidence < 0 or rule.confidence > 1:
                issues.append(f"ルール {rule.rule_id} の信頼度が無効です: {rule.confidence}")
            
            for condition in rule.antecedent:
                if condition.weight < 0 or condition.weight > 1:
                    issues.append(f"ルール {rule.rule_id} の条件重みが無効です: {condition.weight}")
        
        return issues
    
    def get_rule_statistics(self) -> Dict[str, int]:
        """ルール統計を取得"""
        stats = {
            "total_rules": len(self.rules),
            "field_rules": len([r for r in self.rules if "FIELD" in r.rule_id]),
            "experience_rules": len([r for r in self.rules if "EXP" in r.rule_id]),
            "style_rules": len([r for r in self.rules if "STYLE" in r.rule_id]),
            "advisor_rules": len([r for r in self.rules if "ADVISOR" in r.rule_id]),
            "overall_rules": len([r for r in self.rules if "OVERALL" in r.rule_id]),
            "and_rules": len([r for r in self.rules if r.operator == RuleOperator.AND]),
            "or_rules": len([r for r in self.rules if r.operator == RuleOperator.OR])
        }
        return stats
    
    def print_rules_summary(self):
        """ルール一覧を表示"""
        print("🔧 ファジィルールベース一覧")
        print("=" * 50)
        
        categories = {
            "FIELD": "分野適合性ルール",
            "EXP": "経験レベルルール", 
            "STYLE": "研究スタイルルール",
            "ADVISOR": "指導スタイルルール",
            "OVERALL": "総合適合性ルール",
            "INNOVATION": "革新性ルール"
        }
        
        for category, name in categories.items():
            category_rules = [r for r in self.rules if category in r.rule_id]
            if category_rules:
                print(f"\n📋 {name} ({len(category_rules)}件)")
                for rule in category_rules:
                    print(f"   {rule.rule_id}: {rule.description}")
                    print(f"   　信頼度: {rule.confidence:.2f}")
        
        stats = self.get_rule_statistics()
        print(f"\n📊 統計情報:")
        print(f"   総ルール数: {stats['total_rules']}")
        print(f"   AND演算: {stats['and_rules']}, OR演算: {stats['or_rules']}")