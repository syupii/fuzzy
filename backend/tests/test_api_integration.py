# tests/test_api_integration.py
"""
API統合テスト（パターンA版）
FastAPIエンドポイントの動作確認
"""

import pytest
import sys
import os
from fastapi.testclient import TestClient

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# テストクライアント作成
client = TestClient(app)


class TestHealthEndpoint:
    """ヘルスチェックエンドポイントのテスト"""
    
    def test_health_check_status(self):
        """ヘルスチェックのステータス確認"""
        response = client.get("/health")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "unhealthy"]
    
    def test_health_check_version(self):
        """バージョン情報の確認"""
        response = client.get("/health")
        data = response.json()
        
        assert "version" in data
        assert "PatternA" in data["version"]
    
    def test_health_check_pattern(self):
        """パターンA情報の確認"""
        response = client.get("/health")
        data = response.json()
        
        assert "pattern" in data
        assert data["pattern"] == "A (遺伝的アルゴリズムなし)"
        
        # GAが無効化されているか
        assert "features" in data
        assert data["features"]["genetic_optimization"] == False
        assert data["features"]["default_params"] == True


class TestRootEndpoint:
    """ルートエンドポイントのテスト"""
    
    def test_root_message(self):
        """ルートメッセージの確認"""
        response = client.get("/")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_root_features(self):
        """機能情報の確認"""
        response = client.get("/")
        data = response.json()
        
        assert "features" in data
        features = data["features"]
        
        # パターンAの特徴
        assert features["genetic_algorithm"] == False
        assert features["default_parameters"] == True
        assert features["dynamic_decision_tree"] == True
        assert features["field_matching"] == True


class TestCriteriaEndpoint:
    """評価基準エンドポイントのテスト"""
    
    def test_criteria_list(self):
        """評価基準一覧の取得"""
        response = client.get("/api/criteria")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "criteria" in data
        assert "total_count" in data
        
        # 13項目対応
        assert data["total_count"] == 13
        assert data["basic_count"] == 12
        assert data["has_field_match"] == True
    
    def test_criteria_details(self):
        """評価基準詳細の確認"""
        response = client.get("/api/criteria")
        data = response.json()
        
        criteria = data["criteria"]
        
        # 各基準に必要な情報があるか
        for criterion in criteria:
            assert "id" in criterion
            assert "name" in criterion
            assert "description" in criterion
            assert "range" in criterion


class TestFieldsEndpoint:
    """分野エンドポイントのテスト"""
    
    def test_fields_list(self):
        """分野一覧の取得"""
        response = client.get("/api/fields")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "fields" in data
        assert "total_count" in data
        
        # 20分野対応
        assert data["total_count"] == 20
    
    def test_fields_structure(self):
        """分野構造の確認"""
        response = client.get("/api/fields")
        data = response.json()
        
        fields = data["fields"]
        
        # 各分野に必要な情報があるか
        for field in fields:
            assert "id" in field
            assert "name" in field


class TestLabsEndpoint:
    """研究室エンドポイントのテスト"""
    
    def test_labs_list(self):
        """研究室一覧の取得"""
        response = client.get("/api/labs")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "labs" in data
        assert "total_count" in data
        assert data["total_count"] > 0
    
    def test_lab_detail(self):
        """研究室詳細の取得"""
        # まず一覧を取得
        response = client.get("/api/labs")
        labs = response.json()["labs"]
        
        # 最初の研究室の詳細を取得
        if len(labs) > 0:
            lab_id = labs[0]["id"]
            response = client.get(f"/api/labs/{lab_id}")
            
            assert response.status_code == 200
            
            data = response.json()
            assert data["id"] == lab_id
            assert "name" in data
            assert "field_id" in data
            assert "field_name" in data
    
    def test_lab_not_found(self):
        """存在しない研究室のテスト"""
        response = client.get("/api/labs/nonexistent_lab")
        
        assert response.status_code == 404


class TestEvaluateEndpoint:
    """評価エンドポイントのテスト"""
    
    def test_evaluate_basic(self):
        """基本的な評価のテスト"""
        student_profile = {
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
            
            "research_field_match": 7,
            
            "field_interests": {
                "ai_ml": 10,
                "image_processing": 7
            }
        }
        
        response = client.post("/api/evaluate", json=student_profile)
        
        assert response.status_code == 200
        
        data = response.json()
        assert "evaluation_results" in data
        assert "total_labs_evaluated" in data
        assert data["total_labs_evaluated"] > 0
    
    def test_evaluate_results_structure(self):
        """評価結果の構造テスト"""
        student_profile = {
            "research_intensity": 8,
            "research_field_match": 5,
            "field_interests": {"ai_ml": 8}
        }
        
        response = client.post("/api/evaluate", json=student_profile)
        data = response.json()
        
        results = data["evaluation_results"]
        
        # 結果がソートされているか
        for i in range(len(results) - 1):
            assert results[i]["overall_compatibility"] >= results[i+1]["overall_compatibility"]
        
        # 各結果に必要な情報があるか
        for result in results:
            assert "lab_id" in result
            assert "lab_name" in result
            assert "overall_compatibility" in result
            assert "basic_score" in result
            assert "field_score" in result
            assert "field_weight" in result
            assert "basic_weight" in result
            assert "explanation" in result
            assert "recommendation" in result
    
    def test_evaluate_system_info(self):
        """システム情報の確認"""
        student_profile = {
            "research_intensity": 5,
            "research_field_match": 5,
            "field_interests": {"ai_ml": 5}
        }
        
        response = client.post("/api/evaluate", json=student_profile)
        data = response.json()
        
        system_info = data["system_info"]
        
        # パターンA情報
        assert system_info["pattern"] == "A"
        assert system_info["matcher_type"] == "simple"
        assert system_info["uses_genetic_algorithm"] == False
        assert system_info["uses_default_params"] == True
    
    def test_evaluate_missing_field_match(self):
        """research_field_match欠損のテスト"""
        student_profile = {
            "research_intensity": 5,
            "field_interests": {"ai_ml": 5}
            # research_field_match が欠損
        }
        
        response = client.post("/api/evaluate", json=student_profile)
        
        assert response.status_code == 400
    
    def test_evaluate_missing_field_interests(self):
        """field_interests欠損のテスト"""
        student_profile = {
            "research_intensity": 5,
            "research_field_match": 5
            # field_interests が欠損
        }
        
        response = client.post("/api/evaluate", json=student_profile)
        
        assert response.status_code == 400
    
    def test_evaluate_insufficient_criteria(self):
        """基本項目不足のテスト"""
        student_profile = {
            "research_intensity": 5,
            "research_field_match": 5,
            "field_interests": {"ai_ml": 5}
            # 基本項目が1つだけ（最低5つ必要）
        }
        
        response = client.post("/api/evaluate", json=student_profile)
        
        assert response.status_code == 400


class TestExplainEndpoint:
    """説明エンドポイントのテスト"""
    
    def test_explain_recommendation(self):
        """推薦説明のテスト"""
        student_profile = {
            "research_intensity": 9,
            "advisor_style": 7,
            "team_work": 5,
            "research_field_match": 7,
            "field_interests": {"ai_ml": 10}
        }
        
        # まず研究室一覧を取得
        response = client.get("/api/labs")
        labs = response.json()["labs"]
        
        if len(labs) > 0:
            lab_id = labs[0]["id"]
            
            # 説明を取得
            response = client.post(
                f"/api/explain/{lab_id}",
                json=student_profile
            )
            
            assert response.status_code == 200
            
            data = response.json()
            assert "lab_id" in data
            assert "overall_compatibility" in data
            assert "explanation" in data
            assert "score_breakdown" in data
            assert "strengths" in data
            assert "concerns" in data
            assert "field_analysis" in data
    
    def test_explain_nonexistent_lab(self):
        """存在しない研究室の説明テスト"""
        student_profile = {
            "research_intensity": 5,
            "research_field_match": 5,
            "field_interests": {"ai_ml": 5}
        }
        
        response = client.post(
            "/api/explain/nonexistent_lab",
            json=student_profile
        )
        
        assert response.status_code == 404


class TestFieldMatchingLogic:
    """分野マッチングロジックのテスト"""
    
    def test_exact_match_scoring(self):
        """完全一致スコアのテスト"""
        student_profile = {
            "research_intensity": 5,
            "research_field_match": 10,  # 分野100%重視
            "field_interests": {"ai_ml": 10}  # AI分野に最大興味
        }
        
        response = client.post("/api/evaluate", json=student_profile)
        data = response.json()
        
        # AI研究室を探す
        ai_lab_result = next(
            (r for r in data["evaluation_results"] if r["field_id"] == "ai_ml"),
            None
        )
        
        if ai_lab_result:
            # 完全一致なので高スコア
            assert ai_lab_result["field_score"] > 0.8
            assert ai_lab_result["field_detail"]["match_type"] == "exact"
    
    def test_weight_variation(self):
        """重み付けバリエーションのテスト"""
        # 分野重視
        profile_field = {
            "research_intensity": 5,
            "research_field_match": 10,
            "field_interests": {"ai_ml": 10}
        }
        
        # 項目重視
        profile_criteria = {
            "research_intensity": 5,
            "research_field_match": 1,
            "field_interests": {"ai_ml": 10}
        }
        
        response_field = client.post("/api/evaluate", json=profile_field)
        response_criteria = client.post("/api/evaluate", json=profile_criteria)
        
        data_field = response_field.json()
        data_criteria = response_criteria.json()
        
        # 両方とも成功
        assert response_field.status_code == 200
        assert response_criteria.status_code == 200
        
        # 重みが異なることを確認
        result_field = data_field["evaluation_results"][0]
        result_criteria = data_criteria["evaluation_results"][0]
        
        assert result_field["field_weight"] > result_criteria["field_weight"]
        assert result_field["basic_weight"] < result_criteria["basic_weight"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])