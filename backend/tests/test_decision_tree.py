"""
ファジィ決定木システムのテストコード
"""

import pytest
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fuzzy.membership import (
    TriangularMF, GaussianMF, TrapezoidalMF,
    FuzzyVariable, create_standard_fuzzy_variable
)
from core.decision_tree.tree import MultiLevelFuzzyClassifier


class TestTriangularMembershipFunction:
    """三角型メンバーシップ関数のテスト"""
    
    def test_triangular_basic(self):
        """基本的な三角型メンバーシップ関数"""
        mf = TriangularMF(0.0, 0.5, 1.0)
        
        # 境界値
        assert mf(0.0) == 0.0
        assert mf(1.0) == 0.0
        
        # ピーク
        assert mf(0.5) == 1.0
        
        # 中間値
        assert abs(mf(0.25) - 0.5) < 0.01
        assert abs(mf(0.75) - 0.5) < 0.01
    
    def test_triangular_range(self):
        """範囲外の値"""
        mf = TriangularMF(0.2, 0.5, 0.8)
        
        assert mf(-0.1) == 0.0
        assert mf(1.5) == 0.0
    
    def test_triangular_invalid_params(self):
        """無効なパラメータ"""
        with pytest.raises(ValueError):
            TriangularMF(0.5, 0.3, 0.8)  # a > b


class TestGaussianMembershipFunction:
    """ガウス型メンバーシップ関数のテスト"""
    
    def test_gaussian_basic(self):
        """基本的なガウス型メンバーシップ関数"""
        mf = GaussianMF(mean=0.5, sigma=0.2)
        
        # 中心でピーク
        assert mf(0.5) == 1.0
        
        # 対称性
        assert abs(mf(0.3) - mf(0.7)) < 0.01
    
    def test_gaussian_sigma_effect(self):
        """sigmaの影響確認"""
        mf_narrow = GaussianMF(0.5, 0.1)
        mf_wide = GaussianMF(0.5, 0.3)
        
        # 狭いsigmaの方が急峻
        assert mf_narrow(0.4) < mf_wide(0.4)
    
    def test_gaussian_invalid_sigma(self):
        """無効なsigma"""
        with pytest.raises(ValueError):
            GaussianMF(0.5, -0.1)


class TestFuzzyVariable:
    """ファジィ変数のテスト"""
    
    def test_create_standard_variable(self):
        """標準的なファジィ変数作成"""
        var = create_standard_fuzzy_variable(
            "test_var",
            universe=(0.0, 1.0),
            n_sets=3
        )
        
        assert var.name == "test_var"
        assert len(var.fuzzy_sets) == 3
        assert "low" in var.fuzzy_sets
        assert "medium" in var.fuzzy_sets
        assert "high" in var.fuzzy_sets
    
    def test_fuzzify(self):
        """ファジィ化テスト"""
        var = create_standard_fuzzy_variable("test", n_sets=3)
        
        # 中間値のファジィ化
        memberships = var.fuzzify(0.5)
        
        assert "low" in memberships
        assert "medium" in memberships
        assert "high" in memberships
        
        # 合計は1を超えない（重複する場合あり）
        assert all(0 <= m <= 1 for m in memberships.values())
    
    def test_defuzzify_centroid(self):
        """重心法による非ファジィ化"""
        var = create_standard_fuzzy_variable("test", n_sets=3)
        
        memberships = {"low": 0.2, "medium": 0.8, "high": 0.3}
        
        result = var.defuzzify(memberships, method="centroid")
        
        # 結果は論理領域内
        assert 0.0 <= result <= 1.0
    
    def test_defuzzify_max(self):
        """最大値法による非ファジィ化"""
        var = create_standard_fuzzy_variable("test", n_sets=3)
        
        memberships = {"low": 0.1, "medium": 0.9, "high": 0.2}
        
        result = var.defuzzify(memberships, method="max")
        
        # mediumが最大なので、中央付近の値
        assert 0.3 <= result <= 0.7


class TestMultiLevelFuzzyClassifier:
    """多階層ファジィ分類器のテスト"""
    
    @pytest.fixture
    def classifier(self):
        """分類器インスタンス"""
        return MultiLevelFuzzyClassifier()
    
    def test_high_intensity_team_oriented(self, classifier):
        """高研究強度・チーム志向の分類"""
        profile = {
            "research_intensity": 0.9,
            "team_work": 0.8,
            "flexibility": 0.6,
            "lab_atmosphere": 0.7
        }
        
        result = classifier.classify(profile)
        
        assert result["level1_branch"] == "high"
        assert result["level2_cluster"] == "team_oriented"
        assert result["primary_cluster"] == "high_team_oriented"
    
    def test_high_intensity_individual_focused(self, classifier):
        """高研究強度・個人志向の分類"""
        profile = {
            "research_intensity": 0.85,
            "team_work": 0.3,  # 低いチームワーク
            "flexibility": 0.6,
            "lab_atmosphere": 0.5
        }
        
        result = classifier.classify(profile)
        
        assert result["level1_branch"] == "high"
        assert result["level2_cluster"] == "individual_focused"
    
    def test_medium_intensity_flexible(self, classifier):
        """中研究強度・柔軟志向の分類"""
        profile = {
            "research_intensity": 0.55,
            "team_work": 0.6,
            "flexibility": 0.85,  # 高い柔軟性
            "lab_atmosphere": 0.6
        }
        
        result = classifier.classify(profile)
        
        assert result["level1_branch"] == "medium"
        assert result["level2_cluster"] == "flexible_style"
    
    def test_medium_intensity_structured(self, classifier):
        """中研究強度・構造志向の分類"""
        profile = {
            "research_intensity": 0.5,
            "team_work": 0.6,
            "flexibility": 0.3,  # 低い柔軟性
            "lab_atmosphere": 0.5
        }
        
        result = classifier.classify(profile)
        
        assert result["level1_branch"] == "medium"
        assert result["level2_cluster"] == "structured_style"
    
    def test_low_intensity_active(self, classifier):
        """低研究強度・活発志向の分類"""
        profile = {
            "research_intensity": 0.3,
            "team_work": 0.5,
            "flexibility": 0.5,
            "lab_atmosphere": 0.8  # 活発な雰囲気
        }
        
        result = classifier.classify(profile)
        
        assert result["level1_branch"] == "low"
        assert result["level2_cluster"] == "active_atmosphere"
    
    def test_low_intensity_quiet(self, classifier):
        """低研究強度・静寂志向の分類"""
        profile = {
            "research_intensity": 0.25,
            "team_work": 0.4,
            "flexibility": 0.5,
            "lab_atmosphere": 0.2  # 静かな雰囲気
        }
        
        result = classifier.classify(profile)
        
        assert result["level1_branch"] == "low"
        assert result["level2_cluster"] == "quiet_atmosphere"
    
    def test_membership_degrees(self, classifier):
        """メンバーシップ度の確認"""
        profile = {
            "research_intensity": 0.65,  # 境界付近
            "team_work": 0.6,
            "flexibility": 0.6,
            "lab_atmosphere": 0.6
        }
        
        result = classifier.classify(profile)
        
        # Level1のメンバーシップ度確認
        l1_memberships = result["level1_memberships"]
        
        assert "low" in l1_memberships
        assert "medium" in l1_memberships
        assert "high" in l1_memberships
        
        # すべて[0, 1]の範囲内
        assert all(0 <= m <= 1 for m in l1_memberships.values())
        
        # Level2のメンバーシップ度確認
        l2_memberships = result["level2_memberships"]
        assert all(0 <= m <= 1 for m in l2_memberships.values())
    
    def test_confidence_scores(self, classifier):
        """信頼度スコアの確認"""
        profile = {
            "research_intensity": 0.9,
            "team_work": 0.85,
            "flexibility": 0.7,
            "lab_atmosphere": 0.7
        }
        
        result = classifier.classify(profile)
        
        # 信頼度が存在
        assert "confidence" in result
        
        confidence = result["confidence"]
        assert "level1" in confidence
        assert "level2" in confidence
        assert "overall" in confidence
        
        # 全体信頼度はlevel1とlevel2の平均
        expected_overall = (confidence["level1"] + confidence["level2"]) / 2
        assert abs(confidence["overall"] - expected_overall) < 0.01
    
    def test_classification_path(self, classifier):
        """分類パスの確認"""
        profile = {
            "research_intensity": 0.8,
            "team_work": 0.7,
            "flexibility": 0.6,
            "lab_atmosphere": 0.6
        }
        
        result = classifier.classify(profile)
        
        # パスが存在
        assert "classification_path" in result
        
        path = result["classification_path"]
        
        # 2レベルのパスが存在
        assert len(path) == 2
        
        # Level1の情報が含まれる
        assert "Level 1:" in path[0]
        assert "Level 2:" in path[1]
        
        # メンバーシップ度が含まれる
        assert "μ=" in path[0]
        assert "μ=" in path[1]
    
    def test_boundary_cases(self, classifier):
        """境界ケースのテスト"""
        # 最小値
        result_min = classifier.classify({
            "research_intensity": 0.0,
            "team_work": 0.0,
            "flexibility": 0.0,
            "lab_atmosphere": 0.0
        })
        assert result_min["level1_branch"] == "low"
        
        # 最大値
        result_max = classifier.classify({
            "research_intensity": 1.0,
            "team_work": 1.0,
            "flexibility": 1.0,
            "lab_atmosphere": 1.0
        })
        assert result_max["level1_branch"] == "high"
        
        # 中央値
        result_mid = classifier.classify({
            "research_intensity": 0.5,
            "team_work": 0.5,
            "flexibility": 0.5,
            "lab_atmosphere": 0.5
        })
        assert result_mid["level1_branch"] == "medium"


class TestIntegration:
    """統合テスト"""
    
    def test_complete_evaluation_flow(self):
        """完全な評価フローのテスト"""
        classifier = MultiLevelFuzzyClassifier()
        
        # 複数の学生プロファイルをテスト
        test_profiles = [
            {
                "name": "研究熱心・チーム型",
                "research_intensity": 0.9,
                "team_work": 0.85,
                "flexibility": 0.6,
                "lab_atmosphere": 0.7
            },
            {
                "name": "バランス型・柔軟",
                "research_intensity": 0.55,
                "team_work": 0.6,
                "flexibility": 0.8,
                "lab_atmosphere": 0.65
            },
            {
                "name": "軽負荷・静寂型",
                "research_intensity": 0.3,
                "team_work": 0.4,
                "flexibility": 0.5,
                "lab_atmosphere": 0.25
            }
        ]
        
        results = []
        for profile in test_profiles:
            result = classifier.classify(profile)
            results.append(result)
            
            # 基本的な整合性チェック
            assert "primary_cluster" in result
            assert "level1_branch" in result
            assert "level2_cluster" in result
            assert "confidence" in result
            assert result["confidence"]["overall"] > 0
        
        # 全て異なるクラスタに分類されることを期待
        clusters = [r["primary_cluster"] for r in results]
        assert len(set(clusters)) >= 2  # 少なくとも2種類のクラスタ


# テスト実行時の情報表示
if __name__ == "__main__":
    print("=" * 70)
    print("ファジィ決定木システム テスト実行")
    print("=" * 70)
    
    pytest.main([__file__, "-v", "--tb=short"])
    
    print("\n" + "=" * 70)
    print("テスト完了")
    print("=" * 70)