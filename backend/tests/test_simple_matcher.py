# tests/test_simple_matcher.py
"""
シンプルマッチャーのテスト
"""

import pytest
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.matching.simple_matcher import SimpleMatcher, CompatibilityResult


class TestSimpleMatcher:
    """シンプルマッチャーのテストクラス"""
    
    def setup_method(self):
        """各テストの前に実行"""
        self.matcher = SimpleMatcher()
        
        # テスト用学生プロファイル
        self.student_research_focused = {
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
            
            "research_intensity_priority": 10,
            "publication_opportunity_priority": 10,
            "workload_priority": 6,
            
            "research_field_match": 7,  # やや分野重視
            
            "field_interests": {
                "ai_ml": 10,
                "image_processing": 7
            }
        }
        
        # テスト用研究室
        self.lab_ai = {
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
        
        self.lab_web = {
            "id": "web_lab",
            "name": "Webデザイン研究室",
            "field_id": "web_design",
            "research_intensity": 6,
            "advisor_style": 8,
            "team_work": 9,
            "workload": 6,
            "theory_practice": 8,
            "skill_development": 7,
            "lab_atmosphere": 9,
            "flexibility": 8,
            "publication_opportunity": 5,
            "interdisciplinary": 7,
            "communication_style": 9,
            "innovation_focus": 7
        }
    
    def test_matcher_initialization(self):
        """マッチャーの初期化テスト"""
        assert self.matcher is not None
        assert self.matcher.params is not None
        assert len(self.matcher.criteria) == 12
    
    def test_basic_compatibility_calculation(self):
        """基本的な適合度計算のテスト"""
        result = self.matcher.calculate_compatibility(
            self.student_research_focused,
            self.lab_ai
        )
        
        assert isinstance(result, CompatibilityResult)
        assert 0 <= result.total_compatibility <= 1
        assert 0 <= result.basic_score <= 1
        assert 0 <= result.field_score <= 1
    
    def test_exact_field_match(self):
        """完全一致分野マッチのテスト"""
        result = self.matcher.calculate_compatibility(
            self.student_research_focused,
            self.lab_ai
        )
        
        # AI分野で完全一致
        assert result.field_detail["match_type"] == "exact"
        assert result.field_score > 0.8  # 興味度10なので高スコア
    
    def test_no_field_match(self):
        """分野不一致のテスト"""
        result = self.matcher.calculate_compatibility(
            self.student_research_focused,
            self.lab_web
        )
        
        # AI興味 vs Webデザイン
        assert result.field_detail["match_type"] in ["none", "category"]
        assert result.field_score < 0.8  # 不一致なので低スコア
    
    def test_research_field_match_weight(self):
        """research_field_matchによる重み付けのテスト"""
        # 分野重視（research_field_match=10）
        student_field_focused = self.student_research_focused.copy()
        student_field_focused["research_field_match"] = 10
        
        result_field = self.matcher.calculate_compatibility(
            student_field_focused,
            self.lab_ai
        )
        
        # 項目重視（research_field_match=1）
        student_criteria_focused = self.student_research_focused.copy()
        student_criteria_focused["research_field_match"] = 1
        
        result_criteria = self.matcher.calculate_compatibility(
            student_criteria_focused,
            self.lab_ai
        )
        
        # 分野重視の方が分野比重が高い
        assert result_field.field_weight_alpha > result_criteria.field_weight_alpha
        assert result_field.basic_weight_beta < result_criteria.basic_weight_beta
    
    def test_priority_sorting(self):
        """優先度ソートのテスト"""
        priorities = self.matcher._get_sorted_priorities(
            self.student_research_focused
        )
        
        # 優先度降順になっているか
        for i in range(len(priorities) - 1):
            assert priorities[i]["priority"] >= priorities[i+1]["priority"]
    
    def test_dynamic_tree_layers(self):
        """動的決定木レイヤーのテスト"""
        priorities = self.matcher._get_sorted_priorities(
            self.student_research_focused
        )
        tree_layers = self.matcher._build_dynamic_tree(priorities)
        
        # 上位5項目がレイヤーになっているか
        assert len(tree_layers) == 5
        assert all("Layer" in layer for layer in tree_layers)
    
    def test_gaussian_similarity(self):
        """ガウス類似度計算のテスト"""
        # 完全一致
        sim_perfect = self.matcher._gaussian_similarity(0.5, 0.5, 0.2)
        assert sim_perfect == 1.0
        
        # 部分一致
        sim_partial = self.matcher._gaussian_similarity(0.5, 0.7, 0.2)
        assert 0 < sim_partial < 1
        
        # 大きな差
        sim_low = self.matcher._gaussian_similarity(0.1, 0.9, 0.2)
        assert sim_low < 0.5
    
    def test_recommendation_levels(self):
        """推薦レベルのテスト"""
        assert self.matcher._get_recommendation(0.9) == "強く推薦"
        assert self.matcher._get_recommendation(0.75) == "推薦"
        assert self.matcher._get_recommendation(0.6) == "検討推奨"
        assert self.matcher._get_recommendation(0.3) == "慎重に検討"
    
    def test_batch_calculate(self):
        """バッチ計算のテスト"""
        labs = [self.lab_ai, self.lab_web]
        results = self.matcher.batch_calculate(
            self.student_research_focused,
            labs
        )
        
        # 2つの研究室
        assert len(results) == 2
        
        # ソートされているか（降順）
        assert results[0][1].total_compatibility >= results[1][1].total_compatibility
        
        # AI研究室の方が高スコアのはず（分野一致）
        ai_result = next(r for r in results if r[0]["id"] == "ai_lab")
        assert ai_result[1].field_detail["match_type"] == "exact"
    
    def test_criteria_scores(self):
        """項目別スコアのテスト"""
        result = self.matcher.calculate_compatibility(
            self.student_research_focused,
            self.lab_ai
        )
        
        # 全ての基本項目がスコアに含まれているか
        assert len(result.criteria_scores) == 12
        
        # 各スコアが0-1の範囲か
        for score in result.criteria_scores.values():
            assert 0 <= score <= 1
    
    def test_explanation_generation(self):
        """説明文生成のテスト"""
        result = self.matcher.calculate_compatibility(
            self.student_research_focused,
            self.lab_ai
        )
        
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0


class TestFieldMatching:
    """分野マッチングの詳細テスト"""
    
    def setup_method(self):
        """各テストの前に実行"""
        self.matcher = SimpleMatcher()
    
    def test_exact_match_scoring(self):
        """完全一致スコアリングのテスト"""
        field_interests = {"ai_ml": 10}
        lab_field = "ai_ml"
        
        score, detail = self.matcher._calculate_field_match(
            field_interests,
            lab_field
        )
        
        assert detail["match_type"] == "exact"
        assert score == 1.0  # 興味度10 → スコア1.0
    
    def test_category_match_scoring(self):
        """カテゴリ一致スコアリングのテスト"""
        field_interests = {"ai_ml": 10}
        lab_field = "image_processing"  # 同じtechnologyカテゴリ
        
        score, detail = self.matcher._calculate_field_match(
            field_interests,
            lab_field
        )
        
        assert detail["match_type"] == "category"
        assert 0.5 < score < 1.0  # 部分一致
    
    def test_no_match_scoring(self):
        """不一致スコアリングのテスト"""
        field_interests = {"ai_ml": 10}
        lab_field = "web_design"  # 異なるカテゴリ
        
        score, detail = self.matcher._calculate_field_match(
            field_interests,
            lab_field
        )
        
        assert detail["match_type"] == "none"
        assert score == 0.3  # ペナルティスコア


class TestEdgeCases:
    """エッジケースのテスト"""
    
    def setup_method(self):
        """各テストの前に実行"""
        self.matcher = SimpleMatcher()
    
    def test_missing_field_interests(self):
        """分野興味なしのテスト"""
        student = {
            "research_intensity": 5,
            "research_field_match": 5
        }
        lab = {"field_id": "ai_ml", "research_intensity": 5}
        
        result = self.matcher.calculate_compatibility(student, lab)
        
        # エラーにならずに計算できるか
        assert result.total_compatibility is not None
        assert result.field_score == 0.5  # デフォルト値
    
    def test_all_priorities_equal(self):
        """全ての優先度が同じ場合のテスト"""
        student = {
            "research_intensity": 5,
            "advisor_style": 5,
            "research_intensity_priority": 5,
            "advisor_style_priority": 5,
            "research_field_match": 5,
            "field_interests": {"ai_ml": 5}
        }
        lab = {
            "field_id": "ai_ml",
            "research_intensity": 5,
            "advisor_style": 5
        }
        
        result = self.matcher.calculate_compatibility(student, lab)
        
        # 計算が正常に完了するか
        assert result.total_compatibility is not None
    
    def test_extreme_values(self):
        """極端な値のテスト"""
        student = {
            "research_intensity": 1,  # 最小
            "advisor_style": 10,      # 最大
            "research_field_match": 10,
            "field_interests": {"ai_ml": 1}
        }
        lab = {
            "field_id": "ai_ml",
            "research_intensity": 10,
            "advisor_style": 1
        }
        
        result = self.matcher.calculate_compatibility(student, lab)
        
        # スコアが範囲内か
        assert 0 <= result.total_compatibility <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])