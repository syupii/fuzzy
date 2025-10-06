# tests/test_pattern_b.py
"""
パターンB（適応的決定木）のテスト v3.0
- 12項目評価基準対応
- 20研究分野対応
"""

import pytest
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.matching.simple_matcher import SimpleMatcher, CompatibilityResult, TreeLayer
from config.default_params import HIGH_PRIORITY_THRESHOLD, MID_PRIORITY_THRESHOLD


class TestPatternB:
    """パターンB（適応的決定木）のテストクラス"""
    
    def setup_method(self):
        """各テストの前に実行"""
        self.matcher = SimpleMatcher()
        
        # テスト用学生プロファイル（12項目）
        self.student_research_focused = {
            # 基本5項目
            "research_intensity": 9,
            "advisor_style": 7,
            "team_work": 5,
            "workload": 8,
            "theory_practice": 6,
            
            # 拡張5項目
            "research_field_match": 7,
            "skill_development": 7,
            "lab_atmosphere": 6,
            "flexibility": 5,
            "publication_opportunity": 9,
            
            # 特殊2項目
            "interdisciplinary": 4,
            "communication_style": 6,
            
            # 優先度設定（高・中・低の組み合わせ）
            "research_intensity_priority": 10,         # 高
            "publication_opportunity_priority": 10,    # 高
            "workload_priority": 7,                    # 中
            "skill_development_priority": 6,           # 中
            "lab_atmosphere_priority": 5,              # 中
            "team_work_priority": 4,                   # 低
            "advisor_style_priority": 4,               # 低
            
            # 分野興味
            "field_interests": {
                "ai_ml": 10,
                "image_processing": 7
            }
        }
        
        # テスト用研究室（12項目）
        self.lab_ai = {
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
            "research_field_match": 9,
            "skill_development": 8,
            "lab_atmosphere": 7,
            "flexibility": 6,
            "publication_opportunity": 9,
            
            # 特殊2項目
            "interdisciplinary": 5,
            "communication_style": 7,
        }
        
        self.lab_web = {
            "id": "web_lab",
            "name": "Webデザイン研究室",
            "field_id": "web_design",
            
            # 基本5項目
            "research_intensity": 6,
            "advisor_style": 8,
            "team_work": 9,
            "workload": 6,
            "theory_practice": 8,
            
            # 拡張5項目
            "research_field_match": 7,
            "skill_development": 8,
            "lab_atmosphere": 9,
            "flexibility": 8,
            "publication_opportunity": 6,
            
            # 特殊2項目
            "interdisciplinary": 8,
            "communication_style": 9,
        }
    
    def test_adaptive_tree_construction(self):
        """適応的決定木構築のテスト"""
        priorities = self.matcher._get_sorted_priorities(
            self.student_research_focused
        )
        tree_layers, leaf_criteria = self.matcher._build_adaptive_tree(priorities)
        
        # 高優先度（≥8）の項目が3分岐になっているか
        high_priority_layers = [layer for layer in tree_layers if layer.priority >= 8]
        for layer in high_priority_layers:
            assert layer.branches == 3
            assert layer.split_points == [0.3, 0.7]
            assert layer.labels == ["低", "中", "高"]
        
        # 中優先度（5-7）の項目が2分岐になっているか
        mid_priority_layers = [layer for layer in tree_layers if 5 <= layer.priority < 8]
        for layer in mid_priority_layers:
            assert layer.branches == 2
            assert layer.split_points == [0.5]
            assert layer.labels == ["低", "高"]
        
        # 低優先度（<5）の項目がリーフノードになっているか
        assert len(leaf_criteria) > 0
        for criterion in leaf_criteria:
            # 対応する優先度が5未満か確認
            priority_key = f"{criterion}_priority"
            if priority_key in self.student_research_focused:
                priority = self.student_research_focused[priority_key]
                assert priority < 5
    
    def test_tree_path_generation(self):
        """決定木パス生成のテスト"""
        result = self.matcher.calculate_compatibility(
            self.student_research_focused,
            self.lab_ai
        )
        
        # パスが生成されているか
        assert result.tree_path is not None
        assert len(result.tree_path) > 0
        assert result.tree_path != "なし"
        
        # パスが「低」「中」「高」で構成されているか
        path_labels = result.tree_path.split("-")
        for label in path_labels:
            assert label in ["低", "中", "高"]
    
    def test_high_priority_scoring(self):
        """高優先度項目のスコアリングテスト"""
        result = self.matcher.calculate_compatibility(
            self.student_research_focused,
            self.lab_ai
        )
        
        # 高優先度項目のスコアが重視されているか
        research_intensity_score = result.criteria_scores.get("research_intensity", 0)
        publication_score = result.criteria_scores.get("publication_opportunity", 0)
        
        # 完全一致の場合、スコアが高い
        assert research_intensity_score > 0.9
        assert publication_score > 0.9
    
    def test_leaf_criteria_influence(self):
        """リーフノード項目の影響テスト"""
        priorities = self.matcher._get_sorted_priorities(
            self.student_research_focused
        )
        tree_layers, leaf_criteria = self.matcher._build_adaptive_tree(priorities)
        
        # リーフノード項目も評価に含まれているか
        result = self.matcher.calculate_compatibility(
            self.student_research_focused,
            self.lab_ai
        )
        
        for criterion in leaf_criteria:
            assert criterion in result.criteria_scores
    
    def test_priority_thresholds(self):
        """優先度閾値のテスト"""
        # 高優先度閾値
        assert HIGH_PRIORITY_THRESHOLD == 8.0
        
        # 中優先度閾値
        assert MID_PRIORITY_THRESHOLD == 5.0
    
    def test_variable_tree_depth(self):
        """決定木深さの可変性テスト"""
        # 全項目高優先度の学生
        student_all_high = self.student_research_focused.copy()
        for criterion in self.matcher.criteria:
            student_all_high[f"{criterion}_priority"] = 10
        
        priorities_all_high = self.matcher._get_sorted_priorities(student_all_high)
        tree_all_high, leaf_all_high = self.matcher._build_adaptive_tree(priorities_all_high)
        
        # 全項目低優先度の学生
        student_all_low = self.student_research_focused.copy()
        for criterion in self.matcher.criteria:
            student_all_low[f"{criterion}_priority"] = 3
        
        priorities_all_low = self.matcher._get_sorted_priorities(student_all_low)
        tree_all_low, leaf_all_low = self.matcher._build_adaptive_tree(priorities_all_low)
        
        # 高優先度の方が決定木が深い
        assert len(tree_all_high) > len(tree_all_low)
        
        # 低優先度の方がリーフノードが多い
        assert len(leaf_all_high) < len(leaf_all_low)
    
    def test_field_matching_with_pattern_b(self):
        """分野マッチングとパターンBの統合テスト"""
        result = self.matcher.calculate_compatibility(
            self.student_research_focused,
            self.lab_ai
        )
        
        # 完全一致のため、分野スコアが高い
        assert result.field_score > 0.8
        assert result.field_detail["match_type"] == "exact"
    
    def test_explanation_contains_tree_info(self):
        """説明文に決定木情報が含まれているか"""
        result = self.matcher.calculate_compatibility(
            self.student_research_focused,
            self.lab_ai
        )
        
        # 決定木の層数が説明文に含まれているか
        assert "層" in result.explanation or "レイヤー" in result.explanation
        
        # パス情報が含まれているか
        assert "パス" in result.explanation or result.tree_path in result.explanation
    
    def test_batch_calculate_with_pattern_b(self):
        """バッチ計算のテスト"""
        labs = [self.lab_ai, self.lab_web]
        results = self.matcher.batch_calculate(
            self.student_research_focused,
            labs
        )
        
        assert len(results) == 2
        
        # 各結果に決定木情報が含まれているか
        for lab, result in results:
            assert result.tree_path is not None
            assert len(result.tree_layers) > 0
            assert isinstance(result.leaf_criteria, list)
    
    def test_comparison_same_priorities(self):
        """同じ優先度の項目の扱い"""
        student = self.student_research_focused.copy()
        
        # 複数項目に同じ優先度を設定
        student["research_intensity_priority"] = 10
        student["publication_opportunity_priority"] = 10
        student["innovation_focus_priority"] = 10
        
        result = self.matcher.calculate_compatibility(student, self.lab_ai)
        
        # エラーなく処理できるか
        assert result is not None
        assert result.total_compatibility > 0
    
    def test_extreme_priority_distribution(self):
        """極端な優先度分布のテスト"""
        # 全て高優先度
        student_extreme_high = self.student_research_focused.copy()
        for criterion in self.matcher.criteria:
            student_extreme_high[f"{criterion}_priority"] = 10
        
        result_high = self.matcher.calculate_compatibility(
            student_extreme_high,
            self.lab_ai
        )
        
        # 全て低優先度
        student_extreme_low = self.student_research_focused.copy()
        for criterion in self.matcher.criteria:
            student_extreme_low[f"{criterion}_priority"] = 1
        
        result_low = self.matcher.calculate_compatibility(
            student_extreme_low,
            self.lab_ai
        )
        
        # 両方とも有効な結果
        assert 0 <= result_high.total_compatibility <= 1
        assert 0 <= result_low.total_compatibility <= 1
    
    def test_thirteen_criteria_coverage(self):
        """12項目すべてがカバーされているか"""
        result = self.matcher.calculate_compatibility(
            self.student_research_focused,
            self.lab_ai
        )
        
        # 12項目すべてがスコアに含まれているか
        assert len(result.criteria_scores) == 12
        
        # すべての項目が評価されているか
        for criterion in self.matcher.criteria:
            assert criterion in result.criteria_scores


# 実行
if __name__ == "__main__":
    pytest.main([__file__, "-v"])