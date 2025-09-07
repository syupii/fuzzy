# test_example.py - システム動作確認テスト

import sys
from pathlib import Path
import asyncio
import json

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.schemas import StudentProfile, EvaluationCriteria, FieldInterest
from services.lab_matching import LabMatchingService
from config.settings import settings

async def test_system():
    """システム全体の動作確認テスト"""
    
    print("🎯 道都大学情報メディア学部 研究室選択支援システム")
    print("=" * 60)
    print("🧪 システム動作確認テスト開始")
    print("=" * 60)
    
    # 1. 設定確認
    print("\n📊 1. システム設定確認")
    print(f"   研究分野数: {len(settings.research_fields)}")
    print(f"   評価基準数: {len(settings.evaluation_criteria)}")
    print(f"   カテゴリ数: {len(settings.field_categories)}")
    
    # 分野カテゴリ表示
    for category, fields in settings.field_categories.items():
        print(f"   - {category}: {len(fields)}分野")
    
    # 2. テスト用学生プロフィール作成
    print("\n👤 2. テスト用学生プロフィール作成")
    
    test_profiles = [
        {
            "name": "AI志向学生",
            "profile": StudentProfile(
                student_id="ai_student",
                evaluation_criteria=EvaluationCriteria(
                    research_intensity=9,
                    advisor_style=6,
                    team_work=7,
                    workload=8,
                    theory_practice=6,
                    research_field_match=10,
                    skill_development=9,
                    lab_atmosphere=7,
                    flexibility=5,
                    publication_opportunity=9,
                    interdisciplinary=7,
                    communication_style=6,
                    innovation_risk=9
                ),
                field_interests=[
                    FieldInterest(field_id="ai_machine_learning", interest_level=10, experience_level=7, importance_level=10),
                    FieldInterest(field_id="image_computer_vision", interest_level=9, experience_level=5, importance_level=8),
                    FieldInterest(field_id="data_analysis_statistics", interest_level=8, experience_level=6, importance_level=7)
                ]
            )
        },
        {
            "name": "クリエイティブ志向学生",
            "profile": StudentProfile(
                student_id="creative_student",
                evaluation_criteria=EvaluationCriteria(
                    research_intensity=6,
                    advisor_style=8,
                    team_work=9,
                    workload=6,
                    theory_practice=8,
                    research_field_match=8,
                    skill_development=8,
                    lab_atmosphere=9,
                    flexibility=9,
                    publication_opportunity=6,
                    interdisciplinary=9,
                    communication_style=9,
                    innovation_risk=8
                ),
                field_interests=[
                    FieldInterest(field_id="web_design_branding", interest_level=9, experience_level=6, importance_level=9),
                    FieldInterest(field_id="ux_ui_design_thinking", interest_level=10, experience_level=5, importance_level=10),
                    FieldInterest(field_id="video_animation", interest_level=7, experience_level=4, importance_level=6)
                ]
            )
        },
        {
            "name": "ゲーム開発志向学生",
            "profile": StudentProfile(
                student_id="game_student",
                evaluation_criteria=EvaluationCriteria(
                    research_intensity=7,
                    advisor_style=7,
                    team_work=8,
                    workload=7,
                    theory_practice=8,
                    research_field_match=9,
                    skill_development=9,
                    lab_atmosphere=8,
                    flexibility=7,
                    publication_opportunity=6,
                    interdisciplinary=6,
                    communication_style=8,
                    innovation_risk=8
                ),
                field_interests=[
                    FieldInterest(field_id="game_programming", interest_level=10, experience_level=8, importance_level=10),
                    FieldInterest(field_id="vr_ar_media_architecture", interest_level=8, experience_level=4, importance_level=7),
                    FieldInterest(field_id="ai_machine_learning", interest_level=7, experience_level=5, importance_level=6)
                ]
            )
        }
    ]
    
    # 3. 各学生プロフィールでマッチングテスト
    matching_service = LabMatchingService()
    
    for i, test_case in enumerate(test_profiles, 1):
        print(f"\n🧪 3.{i} {test_case['name']}のマッチングテスト")
        print(f"   選択分野: {len(test_case['profile'].field_interests)}分野")
        
        # 選択分野表示
        for fi in test_case['profile'].field_interests:
            field_name = settings.research_fields[fi.field_id]["name"]
            print(f"     - {field_name} (興味:{fi.interest_level}, 経験:{fi.experience_level}, 重要:{fi.importance_level})")
        
        try:
            # マッチング実行
            result = matching_service.find_best_matches(test_case['profile'])
            
            print(f"   ✅ マッチング成功!")
            print(f"   📊 評価研究室数: {len(result.results)}")
            print(f"   🏆 平均適合度: {result.summary.avg_compatibility:.2f}")
            print(f"   🥇 最高スコア: {result.summary.best_match_score:.2f}")
            print(f"   🧬 最適化適応度: {result.optimization_info['final_fitness']:.4f}")
            
            # トップ3表示
            print(f"   🏅 トップ3研究室:")
            for j, lab_result in enumerate(result.results[:3], 1):
                print(f"     {j}. {lab_result.lab.name} ({lab_result.lab.professor}教授)")
                print(f"        スコア: {lab_result.compatibility.overall_score:.2f}")
                print(f"        分野適合: {lab_result.compatibility.field_compatibility:.3f}")
                print(f"        基準適合: {lab_result.compatibility.criteria_compatibility:.3f}")
            
        except Exception as e:
            print(f"   ❌ マッチングエラー: {str(e)}")
    
    # 4. 分野情報確認
    print(f"\n📚 4. 分野情報確認（サンプル5分野）")
    sample_fields = list(settings.research_fields.keys())[:5]
    
    for field_id in sample_fields:
        field_info = settings.research_fields[field_id]
        print(f"   - {field_info['name']}")
        print(f"     カテゴリ: {field_info['category']}")
        print(f"     担当教員: {', '.join(field_info['faculty'])}")
        print(f"     難易度: {field_info['difficulty']}")
        print(f"     特徴: 技術{field_info['tech_focus']}/創造{field_info['creativity_focus']}/理論実践{field_info['theory_practice']}")
    
    # 5. パフォーマンス測定
    print(f"\n⏱️  5. パフォーマンス測定")
    
    import time
    
    performance_profile = test_profiles[0]['profile']  # AI志向学生を使用
    
    start_time = time.time()
    result = matching_service.find_best_matches(performance_profile)
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    print(f"   処理時間: {processing_time:.2f}秒")
    print(f"   研究室数: {len(result.results)}")
    print(f"   1研究室あたり: {(processing_time / len(result.results) * 1000):.2f}ms")
    
    # 6. システム統計
    print(f"\n📈 6. システム統計")
    print(f"   総研究分野数: {len(settings.research_fields)}")
    print(f"   総評価基準数: {len(settings.evaluation_criteria)}")
    print(f"   カテゴリ分布:")
    
    for category, fields in settings.field_categories.items():
        print(f"     - {category}: {len(fields)}分野")
    
    print(f"   遺伝的アルゴリズム設定:")
    print(f"     - 集団サイズ: {settings.ga_population_size}")
    print(f"     - 世代数: {settings.ga_generations}")
    print(f"     - 変異率: {settings.ga_mutation_rate}")
    print(f"     - 交叉率: {settings.ga_crossover_rate}")
    
    print("\n✅ システム動作確認テスト完了")
    print("=" * 60)

def print_api_usage_examples():
    """API使用例を表示"""
    
    print("\n🔌 API使用例")
    print("=" * 40)
    
    # サンプルリクエスト
    sample_request = {
        "student_id": "sample_001",
        "evaluation_criteria": {
            "research_intensity": 8,
            "advisor_style": 7,
            "team_work": 6,
            "workload": 7,
            "theory_practice": 8,
            "research_field_match": 9,
            "skill_development": 8,
            "lab_atmosphere": 7,
            "flexibility": 6,
            "publication_opportunity": 8,
            "interdisciplinary": 6,
            "communication_style": 7,
            "innovation_risk": 8
        },
        "field_interests": [
            {
                "field_id": "ai_machine_learning",
                "interest_level": 9,
                "experience_level": 6,
                "importance_level": 10
            },
            {
                "field_id": "game_programming",
                "interest_level": 7,
                "experience_level": 5,
                "importance_level": 7
            }
        ]
    }
    
    print("📝 サンプルリクエスト (POST /api/v1/prediction/evaluate):")
    print(json.dumps(sample_request, indent=2, ensure_ascii=False))
    
    print("\n🌐 主要APIエンドポイント:")
    endpoints = [
        ("GET", "/api/v1/prediction/fields", "利用可能な研究分野一覧"),
        ("GET", "/api/v1/prediction/fields/{field_id}", "特定分野の詳細情報"),
        ("GET", "/api/v1/prediction/fields/category/{category}", "カテゴリ別分野一覧"),
        ("POST", "/api/v1/prediction/evaluate", "研究室マッチング実行"),
        ("POST", "/api/v1/prediction/quick-evaluation", "簡易評価"),
        ("GET", "/api/v1/prediction/status", "システム状態"),
        ("GET", "/api/v1/prediction/categories", "分野カテゴリ一覧"),
        ("GET", "/api/v1/prediction/evaluation-criteria", "評価基準一覧")
    ]
    
    for method, endpoint, description in endpoints:
        print(f"   {method:4} {endpoint:45} - {description}")
    
    print(f"\n📚 API仕様書: http://localhost:8000/docs")
    print(f"📖 ReDoc: http://localhost:8000/redoc")

if __name__ == "__main__":
    # 非同期テスト実行
    asyncio.run(test_system())
    
    # API使用例表示
    print_api_usage_examples()
    
    print("\n🚀 システム起動方法:")
    print("   python app.py")
    print("   または")
    print("   uvicorn app:app --reload --host 0.0.0.0 --port 8000")