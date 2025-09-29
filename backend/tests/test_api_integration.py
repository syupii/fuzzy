"""
API統合テスト
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
        assert data["version"] == "3.0.0"
    
    def test_health_check_system_info(self):
        """システム情報の確認"""
        response = client.get("/health")
        data = response.json()
        
        assert "system_info" in data
        system_info = data["system_info"]
        
        assert "evaluation_criteria" in system_info
        assert system_info["evaluation_criteria"] == 13  # 13項目対応
        
        assert "features" in system_info
        features = system_info["features"]
        assert features["fuzzy_membership"] == True
        assert features["multi_level_tree"] == True
        assert features["13_criteria"] == True


class TestRootEndpoint:
    """ルートエンドポイントのテスト"""
    
    def test_root_message(self):
        """ルートメッセージの確認"""
        response = client.get("/")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "3.0.0"
    
    def test_root_features(self):
        """機能情報の確認"""
        response = client.get("/")
        data = response.json()
        
        assert "features" in data
        features = data["features"]
        
        assert "evaluation_criteria" in features
        assert "13項目対応" in features["evaluation_criteria"]


class TestCriteriaEndpoint:
    """評価基準エンドポイントのテスト"""
    
    def test_criteria_list(self):
        """評価基準一覧の取得"""
        response = client.get("/api/criteria")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "total_count" in data
        assert data["total_count"] == 13
    
    def test_criteria_categories(self):
        """カテゴリ別評価基準"""
        response = client.get("/api/criteria")
        data = response.json()
        
        assert "categories" in data
        categories = data["categories"]
        
        assert "basic" in categories
        assert len(categories["basic"]) == 5
        
        assert "extended" in categories
        assert len(categories["extended"]) == 5
        
        assert "special" in categories
        assert len(categories["special"]) == 3
    
    def test_criteria_descriptions(self):
        """評価基準の説明"""
        response = client.get("/api/criteria")
        data = response.json()
        
        assert "descriptions" in data
        descriptions = data["descriptions"]
        
        # すべての項目に説明がある
        assert "research_intensity" in descriptions
        assert "innovation_risk" in descriptions


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
    
    def test_labs_categories(self):
        """研究室のカテゴリ情報"""
        response = client.get("/api/labs")
        data = response.json()
        
        assert "categories" in data
        categories = data["categories"]
        
        # 主要カテゴリが存在
        assert "テクノロジー・システム" in categories
        assert "クリエイティブ" in categories
        assert "エンターテイメント" in categories
    
    def test_labs_structure(self):
        """研究室データ構造の確認"""
        response = client.get("/api/labs")
        data = response.json()
        
        labs = data["labs"]
        assert len(labs) > 0
        
        # 最初の研究室の構造確認
        lab = labs[0]
        
        # 基本情報
        assert "id" in lab
        assert "name" in lab
        assert "advisor" in lab
        assert "category" in lab
        assert "field" in lab
        
        # 13項目の評価値
        assert "research_intensity" in lab
        assert "advisor_style" in lab
        assert "team_work" in lab
        assert "workload" in lab
        assert "theory_practice" in lab
        assert "research_field_match" in lab
        assert "skill_development" in lab
        assert "lab_atmosphere" in lab
        assert "flexibility" in lab
        assert "publication_opportunity" in lab
        assert "interdisciplinary" in lab
        assert "communication_style" in lab
        assert "innovation_risk" in lab
    
    def test_lab_detail(self):
        """特定研究室の詳細取得"""
        # まず一覧を取得
        response = client.get("/api/labs")
        labs = response.json()["labs"]
        
        if len(labs) > 0:
            lab_id = labs[0]["id"]
            
            # 詳細を取得
            response = client.get(f"/api/labs/{lab_id}")
            
            assert response.status_code == 200
            
            lab = response.json()
            assert lab["id"] == lab_id
    
    def test_lab_not_found(self):
        """存在しない研究室の取得"""
        response = client.get("/api/labs/nonexistent_lab")
        
        assert response.status_code == 404


class TestEvaluateEndpoint:
    """評価エンドポイントのテスト"""
    
    def get_valid_profile(self):
        """有効な学生プロファイルを作成"""
        return {
            "student_profile": {
                "research_intensity": 0.9,
                "advisor_style": 0.7,
                "team_work": 0.8,
                "workload": 0.85,
                "theory_practice": 0.6,
                "research_field_match": 0.9,
                "skill_development": 0.85,
                "lab_atmosphere": 0.8,
                "flexibility": 0.6,
                "publication_opportunity": 0.9,
                "interdisciplinary": 0.7,
                "communication_style": 0.8,
                "innovation_risk": 0.8
            }
        }
    
    def test_evaluate_success(self):
        """正常な評価リクエスト"""
        profile = self.get_valid_profile()
        
        response = client.post("/api/evaluate", json=profile)
        
        assert response.status_code == 200
        
        data = response.json()
        
        # 基本構造確認
        assert "student_profile" in data
        assert "lab_results" in data
        assert "summary" in data
        assert "metadata" in data
    
    def test_evaluate_results_structure(self):
        """評価結果の構造確認"""
        profile = self.get_valid_profile()
        
        response = client.post("/api/evaluate", json=profile)
        data = response.json()
        
        # 結果が存在
        results = data["lab_results"]
        assert len(results) > 0
        
        # 最初の結果の構造確認
        result = results[0]
        
        assert "lab_id" in result
        assert "lab_name" in result
        assert "overall_compatibility" in result
        assert "feature_scores" in result
        assert "cluster_info" in result
        assert "recommendation_level" in result
        assert "fuzzy_analysis" in result
    
    def test_evaluate_cluster_info(self):
        """クラスタ情報の確認"""
        profile = self.get_valid_profile()
        
        response = client.post("/api/evaluate", json=profile)
        data = response.json()
        
        result = data["lab_results"][0]
        cluster_info = result["cluster_info"]
        
        # クラスタ情報の構造
        assert "primary_cluster" in cluster_info
        assert "level1_branch" in cluster_info
        assert "level2_cluster" in cluster_info
        assert "level1_memberships" in cluster_info
        assert "level2_memberships" in cluster_info
        assert "classification_path" in cluster_info
    
    def test_evaluate_fuzzy_analysis(self):
        """ファジィ分析結果の確認"""
        profile = self.get_valid_profile()
        
        response = client.post("/api/evaluate", json=profile)
        data = response.json()
        
        result = data["lab_results"][0]
        fuzzy_analysis = result["fuzzy_analysis"]
        
        assert "top_matching_features" in fuzzy_analysis
        assert "improvement_areas" in fuzzy_analysis
        assert "cluster_interpretation" in fuzzy_analysis
    
    def test_evaluate_sorting(self):
        """結果がスコア順にソートされているか確認"""
        profile = self.get_valid_profile()
        
        response = client.post("/api/evaluate", json=profile)
        data = response.json()
        
        results = data["lab_results"]
        
        # スコアが降順にソートされている
        for i in range(len(results) - 1):
            assert results[i]["overall_compatibility"] >= results[i+1]["overall_compatibility"]
    
    def test_evaluate_summary(self):
        """サマリー情報の確認"""
        profile = self.get_valid_profile()
        
        response = client.post("/api/evaluate", json=profile)
        data = response.json()
        
        summary = data["summary"]
        
        assert "total_labs" in summary
        assert "top_match" in summary
        assert "excellent_matches" in summary
        assert "good_matches" in summary
        assert "evaluation_method" in summary
        
        # 評価方法に"13項目"が含まれる
        assert "13項目" in summary["evaluation_method"]
    
    def test_evaluate_missing_basic_criteria(self):
        """基本項目が不足している場合"""
        profile = {
            "student_profile": {
                "research_intensity": 0.8,
                # advisor_styleが不足
                "team_work": 0.7
            }
        }
        
        response = client.post("/api/evaluate", json=profile)
        
        # 400エラーが返る
        assert response.status_code == 400
    
    def test_evaluate_with_priorities(self):
        """優先度付き評価（フロントエンドAPI互換性）"""
        profile = {
            "student_profile": {
                # 基本5項目
                "research_intensity": 0.9,
                "advisor_style": 0.7,
                "team_work": 0.8,
                "workload": 0.85,
                "theory_practice": 0.6,
                # 優先度オブジェクト（フロントエンドから送信される）
                "priorities": {
                    "research_intensity": 5,
                    "advisor_style": 3,
                    "team_work": 4,
                    "workload": 2,
                    "theory_practice": 3,
                    "research_field_match": 5,
                    "skill_development": 4,
                    "lab_atmosphere": 3,
                    "flexibility": 2,
                    "publication_opportunity": 4,
                    "interdisciplinary": 3,
                    "communication_style": 3,
                    "innovation_risk": 2
                }
            }
        }
        
        response = client.post("/api/evaluate", json=profile)
        
        # 優先度付きでも正常に動作
        assert response.status_code == 200
    
    def test_evaluate_partial_criteria(self):
        """拡張項目が一部不足している場合"""
        profile = {
            "student_profile": {
                # 基本5項目（必須）
                "research_intensity": 0.8,
                "advisor_style": 0.6,
                "team_work": 0.7,
                "workload": 0.7,
                "theory_practice": 0.6
                # 拡張・特殊項目は省略（デフォルト値0.5が使用される）
            }
        }
        
        response = client.post("/api/evaluate", json=profile)
        
        # 基本項目があれば成功
        assert response.status_code == 200


class TestOptimizeEndpoint:
    """最適化エンドポイントのテスト"""
    
    def test_optimize_not_implemented(self):
        """最適化機能の未実装確認"""
        request_data = {
            "student_profiles": [
                {
                    "research_intensity": 0.8,
                    "advisor_style": 0.6,
                    "team_work": 0.7,
                    "workload": 0.7,
                    "theory_practice": 0.6
                }
            ]
        }
        
        response = client.post("/api/optimize", json=request_data)
        
        # 実装されていないことを示すレスポンス
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "not_implemented"


class TestPerformance:
    """パフォーマンステスト"""
    
    def test_evaluate_response_time(self):
        """評価レスポンス時間"""
        import time
        
        profile = {
            "student_profile": {
                "research_intensity": 0.9,
                "advisor_style": 0.7,
                "team_work": 0.8,
                "workload": 0.85,
                "theory_practice": 0.6,
                "research_field_match": 0.9,
                "skill_development": 0.85,
                "lab_atmosphere": 0.8,
                "flexibility": 0.6,
                "publication_opportunity": 0.9,
                "interdisciplinary": 0.7,
                "communication_style": 0.8,
                "innovation_risk": 0.8
            }
        }
        
        start_time = time.time()
        response = client.post("/api/evaluate", json=profile)
        elapsed_time = time.time() - start_time
        
        assert response.status_code == 200
        
        # 1秒以内にレスポンス
        assert elapsed_time < 1.0
        
        print(f"\n⏱️  評価処理時間: {elapsed_time:.3f}秒")


# テスト実行時の情報表示
if __name__ == "__main__":
    print("=" * 70)
    print("API統合テスト実行")
    print("=" * 70)
    
    pytest.main([__file__, "-v", "--tb=short", "-s"])
    
    print("\n" + "=" * 70)
    print("テスト完了")
    print("=" * 70)