# backend/advanced_genetic_fuzzy_tree.py
"""
🧬 Advanced Genetic Fuzzy Decision Tree System
高度な遺伝的ファジィ決定木システム

使用方法:
    python advanced_genetic_fuzzy_tree.py --mode train
    python advanced_genetic_fuzzy_tree.py --mode evaluate --model_id genetic_model_20241201_143022
    python advanced_genetic_fuzzy_tree.py --mode compare
"""

from models import db, Lab, Evaluation, create_app
from explanation_engine import FuzzyExplanationEngine, NaturalLanguageGenerator
from optimization_tracker import OptimizationTracker, OptimizationReporter
from model_persistence import (
    AdvancedModelPersistence, ModelVersionManager, ModelComparisonTool
)
from genetic_fuzzy_tree import (
    GeneticFuzzyTreeOptimizer, GeneticParameters, Individual
)
import argparse
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# プロジェクトパス追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def create_synthetic_training_data(n_samples: int = 1000) -> pd.DataFrame:
    """合成訓練データ生成"""

    print(f"🎲 Generating {n_samples} synthetic training samples...")

    np.random.seed(42)

    # 基準データ生成
    data = []

    for i in range(n_samples):
        # ユーザー希望（1-10スケール）
        user_research_intensity = np.random.uniform(1, 10)
        user_advisor_style = np.random.uniform(1, 10)
        user_team_work = np.random.uniform(1, 10)
        user_workload = np.random.uniform(1, 10)
        user_theory_practice = np.random.uniform(1, 10)

        # 研究室特徴（1-10スケール）
        lab_research_intensity = np.random.uniform(1, 10)
        lab_advisor_style = np.random.uniform(1, 10)
        lab_team_work = np.random.uniform(1, 10)
        lab_workload = np.random.uniform(1, 10)
        lab_theory_practice = np.random.uniform(1, 10)

        # 適合度計算（ガウシアン類似度ベース）
        weights = [0.25, 0.20, 0.20, 0.15, 0.20]

        similarities = []
        criteria_pairs = [
            (user_research_intensity, lab_research_intensity),
            (user_advisor_style, lab_advisor_style),
            (user_team_work, lab_team_work),
            (user_workload, lab_workload),
            (user_theory_practice, lab_theory_practice)
        ]

        for user_val, lab_val in criteria_pairs:
            diff = abs(user_val - lab_val)
            similarity = np.exp(-0.5 * (diff / 2.0) ** 2)  # ガウシアン類似度
            similarities.append(similarity)

        # 重み付き適合度
        compatibility = sum(w * s for w, s in zip(weights, similarities))

        # ノイズ追加
        compatibility += np.random.normal(0, 0.05)
        compatibility = max(0.0, min(1.0, compatibility))

        # 個人差・主観性モデリング
        personality_factor = np.random.normal(1.0, 0.1)
        compatibility *= personality_factor
        compatibility = max(0.0, min(1.0, compatibility))

        sample = {
            # ユーザー希望
            'user_research_intensity': user_research_intensity,
            'user_advisor_style': user_advisor_style,
            'user_team_work': user_team_work,
            'user_workload': user_workload,
            'user_theory_practice': user_theory_practice,

            # 研究室特徴
            'lab_research_intensity': lab_research_intensity,
            'lab_advisor_style': lab_advisor_style,
            'lab_team_work': lab_team_work,
            'lab_workload': lab_workload,
            'lab_theory_practice': lab_theory_practice,

            # 入力特徴（差分ベース）
            'research_intensity': user_research_intensity,
            'advisor_style': user_advisor_style,
            'team_work': user_team_work,
            'workload': user_workload,
            'theory_practice': user_theory_practice,

            # ターゲット
            'compatibility': compatibility
        }

        data.append(sample)

    df = pd.DataFrame(data)

    print(f"✅ Generated {len(df)} samples")
    print(
        f"📊 Compatibility stats: mean={df['compatibility'].mean():.3f}, std={df['compatibility'].std():.3f}")

    return df


def load_real_data_from_database() -> pd.DataFrame:
    """データベースから実データ読み込み"""

    print("🗄️ Loading real data from database...")

    try:
        # Flaskアプリ作成
        app = create_app()

        with app.app_context():
            # 評価データ取得
            evaluations = Evaluation.query.all()

            if not evaluations:
                print("⚠️ No evaluation data found in database")
                return None

            # データ変換
            data = []
            for eval_record in evaluations:
                # 評価結果取得
                results = eval_record.get_results()
                if not results:
                    continue

                # 各研究室との適合度
                for result in results:
                    lab_info = result.get('lab', {})
                    compatibility_info = result.get('compatibility', {})

                    sample = {
                        # ユーザー希望
                        'research_intensity': eval_record.research_intensity,
                        'advisor_style': eval_record.advisor_style,
                        'team_work': eval_record.team_work,
                        'workload': eval_record.workload,
                        'theory_practice': eval_record.theory_practice,

                        # 適合度
                        'compatibility': compatibility_info.get('overall_score', 50.0) / 100.0,

                        # メタデータ
                        'lab_id': lab_info.get('id'),
                        'evaluation_id': eval_record.id
                    }

                    data.append(sample)

            df = pd.DataFrame(data)

            print(f"📊 Loaded {len(df)} real data samples")
            print(
                f"📈 Compatibility distribution: {df['compatibility'].describe()}")

            return df

    except Exception as e:
        print(f"❌ Failed to load real data: {e}")
        return None


def train_genetic_model(args):
    """遺伝的モデル訓練"""

    print("🧬 Starting Genetic Fuzzy Tree Training...")
    print("=" * 60)

    # データ準備
    if args.use_real_data:
        training_data = load_real_data_from_database()
        if training_data is None or len(training_data) < 50:
            print("⚠️ Insufficient real data, using synthetic data")
            training_data = create_synthetic_training_data(
                args.training_samples)
    else:
        training_data = create_synthetic_training_data(args.training_samples)

    # テストデータ分割
    test_size = int(len(training_data) * 0.2)
    test_data = training_data.sample(n=test_size, random_state=42)
    training_data = training_data.drop(test_data.index)

    print(f"📊 Training samples: {len(training_data)}")
    print(f"🧪 Test samples: {len(test_data)}")

    # パラメータ設定
    parameters = GeneticParameters(
        population_size=args.population_size,
        generations=args.generations,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        max_depth=args.max_depth,
        tournament_size=args.tournament_size
    )

    print("\n🔧 Optimization Parameters:")
    print(f"   Population Size: {parameters.population_size}")
    print(f"   Generations: {parameters.generations}")
    print(f"   Mutation Rate: {parameters.mutation_rate:.2f}")
    print(f"   Crossover Rate: {parameters.crossover_rate:.2f}")
    print(f"   Max Depth: {parameters.max_depth}")

    # 最適化器初期化
    optimizer = GeneticFuzzyTreeOptimizer(
        parameters=parameters,
        random_seed=args.random_seed
    )

    # 最適化実行
    run_id = f"genetic_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n🚀 Starting optimization (Run ID: {run_id})...")

    try:
        result = optimizer.optimize(
            training_data=training_data,
            test_data=test_data,
            target_column='compatibility',
            run_id=run_id
        )

        print("\n🎉 Optimization completed successfully!")

        # 結果保存
        persistence = AdvancedModelPersistence()
        model_id = persistence.save_genetic_optimization_result(
            result,
            description=f"Genetic Fuzzy Tree trained on {len(training_data)} samples"
        )

        # 結果レポート
        print("\n📊 Final Results:")
        print(f"   Best Fitness: {result['best_fitness']:.4f}")
        print(f"   Final Diversity: {result['final_diversity']:.3f}")
        print(
            f"   Convergence: {result['convergence_analysis']['convergence_detected']}")
        print(f"   Model ID: {model_id}")

        # 詳細レポート生成
        if args.generate_report:
            print("\n📋 Generating detailed report...")
            reporter = OptimizationReporter()
            html_report = reporter.generate_html_report(optimizer.tracker)
            print(f"📄 HTML Report: {html_report}")

        return model_id

    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        raise


def evaluate_model(args):
    """モデル評価"""

    print(f"🧪 Evaluating model: {args.model_id}")
    print("=" * 50)

    # モデル読み込み
    persistence = AdvancedModelPersistence()

    try:
        individual, optimization_result = persistence.load_genetic_model(
            args.model_id)

        print(f"✅ Model loaded successfully")
        print(
            f"🎯 Model Fitness: {individual.fitness_components.overall if individual.fitness_components else 'N/A'}")
        print(f"🧮 Model Complexity: {individual.complexity_score}")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # テストデータ準備
    if args.use_real_data:
        test_data = load_real_data_from_database()
        if test_data is None:
            test_data = create_synthetic_training_data(200)
    else:
        test_data = create_synthetic_training_data(200)

    print(f"📊 Test samples: {len(test_data)}")

    # 予測実行
    print("\n🔍 Running predictions...")

    predictions = []
    explanations = []

    # 説明エンジン初期化
    explanation_engine = FuzzyExplanationEngine()

    for i, (_, row) in enumerate(test_data.head(10).iterrows()):  # 最初の10サンプルで評価
        # 入力準備
        user_preferences = {
            'research_intensity': row['research_intensity'],
            'advisor_style': row['advisor_style'],
            'team_work': row['team_work'],
            'workload': row['workload'],
            'theory_practice': row['theory_practice']
        }

        # 予測実行
        try:
            if individual.tree:
                prediction = individual.tree.predict(user_preferences)

                # 詳細説明付き予測
                detailed_prediction, explanation_details = individual.tree.predict_with_explanation(
                    user_preferences, list(user_preferences.keys())
                )

                predictions.append({
                    'sample_id': i,
                    'actual': row['compatibility'],
                    'predicted': prediction,
                    'detailed_predicted': detailed_prediction,
                    'error': abs(row['compatibility'] - prediction),
                    'user_preferences': user_preferences
                })

                # 包括的説明生成
                prediction_result = {
                    'overall_score': prediction * 100,
                    'confidence': explanation_details.get('confidence', 0.5) * 100,
                    'criterion_scores': {}
                }

                lab_info = {'name': f'Test Lab {i+1}'}

                comprehensive_explanation = explanation_engine.generate_comprehensive_explanation(
                    prediction_result, lab_info, user_preferences
                )

                explanations.append({
                    'sample_id': i,
                    'explanation': comprehensive_explanation,
                    'formatted_explanation': NaturalLanguageGenerator.format_explanation_for_ui(
                        comprehensive_explanation, 'markdown'
                    )
                })

            else:
                print(f"⚠️ No tree available for sample {i}")

        except Exception as e:
            print(f"⚠️ Prediction failed for sample {i}: {e}")

    # 評価指標計算
    if predictions:
        predictions_df = pd.DataFrame(predictions)

        mae = predictions_df['error'].mean()
        rmse = np.sqrt((predictions_df['error'] ** 2).mean())
        max_error = predictions_df['error'].max()

        print(f"\n📈 Prediction Performance:")
        print(f"   MAE (Mean Absolute Error): {mae:.4f}")
        print(f"   RMSE (Root Mean Squared Error): {rmse:.4f}")
        print(f"   Max Error: {max_error:.4f}")
        print(f"   Samples Evaluated: {len(predictions)}")

        # サンプル予測表示
        print(f"\n🔍 Sample Predictions:")
        for pred in predictions[:5]:
            print(
                f"   Sample {pred['sample_id']}: Actual={pred['actual']:.3f}, Predicted={pred['predicted']:.3f}, Error={pred['error']:.3f}")

    # 説明サンプル表示
    if explanations and args.show_explanations:
        print(f"\n💡 Sample Explanation (Sample 0):")
        print("=" * 50)
        print(explanations[0]['formatted_explanation'])


def compare_models(args):
    """モデル比較"""

    print("🔍 Comparing models...")
    print("=" * 40)

    persistence = AdvancedModelPersistence()

    # モデル一覧取得
    models = persistence.list_models('genetic_fuzzy_tree')

    if len(models) < 2:
        print("⚠️ Need at least 2 models for comparison")
        return

    print(f"📊 Found {len(models)} genetic fuzzy tree models")

    # 比較表作成
    comparison_df = ModelComparisonTool.compare_models(models)

    print("\n📋 Model Comparison:")
    print(comparison_df.to_string())

    # 性能ランキング
    ranking = ModelComparisonTool.generate_performance_ranking(
        models, 'best_fitness')

    print(f"\n🏆 Performance Ranking (by best_fitness):")
    for i, model in enumerate(ranking[:5], 1):
        print(f"   {i}. {model['model_id']}: {model['ranking_value']:.4f}")

    # 最良モデル検索
    criteria = {
        'best_fitness': 0.6,
        'model_complexity': -0.2,  # 複雑度は低い方が良い
        'tree_depth': -0.1
    }

    best_model = ModelComparisonTool.find_best_model_by_criteria(
        models, criteria)

    if best_model:
        print(f"\n🎯 Best Model (composite criteria): {best_model['model_id']}")
        print(f"   Composite Score: {best_model['composite_score']:.4f}")
        print(f"   Fitness: {best_model['best_fitness']:.4f}")
        print(f"   Complexity: {best_model['model_complexity']}")


def demonstrate_prediction(args):
    """予測デモンストレーション"""

    print("🎭 Prediction Demonstration")
    print("=" * 40)

    # モデル読み込み
    persistence = AdvancedModelPersistence()

    if args.model_id:
        model_id = args.model_id
    else:
        # 最良モデル自動選択
        models = persistence.list_models('genetic_fuzzy_tree')
        if not models:
            print("❌ No genetic fuzzy tree models found")
            return

        best_model = max(models, key=lambda x: x.get('best_fitness', 0.0))
        model_id = best_model['model_id']
        print(f"🤖 Using best available model: {model_id}")

    try:
        individual, _ = persistence.load_genetic_model(model_id)
        print(f"✅ Model loaded: {model_id}")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # デモ用サンプル
    demo_samples = [
        {
            'name': '理論研究希望の学生',
            'preferences': {
                'research_intensity': 9.0,
                'advisor_style': 3.0,  # 厳格な指導を希望
                'team_work': 4.0,     # 個人研究を好む
                'workload': 8.0,      # 高負荷OK
                'theory_practice': 2.0  # 理論重視
            }
        },
        {
            'name': '実践重視の学生',
            'preferences': {
                'research_intensity': 7.0,
                'advisor_style': 8.0,  # 自由な指導を希望
                'team_work': 8.5,     # チーム研究を好む
                'workload': 6.0,      # 中程度の負荷
                'theory_practice': 9.0  # 実践重視
            }
        },
        {
            'name': 'バランス重視の学生',
            'preferences': {
                'research_intensity': 6.5,
                'advisor_style': 5.5,
                'team_work': 6.0,
                'workload': 5.5,
                'theory_practice': 5.0
            }
        }
    ]

    # 説明エンジン
    explanation_engine = FuzzyExplanationEngine()

    print(
        f"\n🎭 Demonstrating predictions for {len(demo_samples)} sample profiles:")

    for sample in demo_samples:
        print(f"\n👤 {sample['name']}:")
        print("   Preferences:", {k: f"{v:.1f}" for k,
              v in sample['preferences'].items()})

        try:
            # 予測実行
            prediction = individual.tree.predict(sample['preferences'])

            # 詳細予測
            detailed_prediction, explanation_details = individual.tree.predict_with_explanation(
                sample['preferences'], list(sample['preferences'].keys())
            )

            print(
                f"   🎯 Predicted Compatibility: {prediction:.3f} ({prediction*100:.1f}%)")
            print(f"   🔍 Detailed Prediction: {detailed_prediction:.3f}")
            print(
                f"   📊 Confidence: {explanation_details.get('confidence', 0.5)*100:.1f}%")

            # 簡易説明生成
            if prediction >= 0.8:
                assessment = "非常に適合度が高い"
            elif prediction >= 0.6:
                assessment = "適合度が高い"
            elif prediction >= 0.4:
                assessment = "適合度は中程度"
            else:
                assessment = "適合度は低い"

            print(f"   💡 Assessment: {assessment}")

        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")


def cleanup_models(args):
    """モデル清理"""

    print("🧹 Cleaning up old models...")

    persistence = AdvancedModelPersistence()

    deleted_count = persistence.cleanup_old_models(
        keep_latest=args.keep_latest,
        min_fitness=args.min_fitness
    )

    print(f"✅ Cleaned up {deleted_count} models")

    # 統計表示
    stats = persistence.get_storage_statistics()
    print(f"\n📊 Storage Statistics:")
    print(f"   Total Models: {stats['total_models']}")
    print(f"   Total Size: {stats['total_size_bytes'] / (1024*1024):.1f} MB")
    print(f"   Average Fitness: {stats['average_fitness']:.4f}")


def main():
    """メイン関数"""

    parser = argparse.ArgumentParser(
        description="Advanced Genetic Fuzzy Decision Tree System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 新しいモデルを訓練
  python advanced_genetic_fuzzy_tree.py --mode train --generations 30 --population_size 50
  
  # モデルを評価
  python advanced_genetic_fuzzy_tree.py --mode evaluate --model_id genetic_model_20241201_143022
  
  # モデル比較
  python advanced_genetic_fuzzy_tree.py --mode compare
  
  # 予測デモ
  python advanced_genetic_fuzzy_tree.py --mode demo --model_id genetic_model_20241201_143022
  
  # モデル清理
  python advanced_genetic_fuzzy_tree.py --mode cleanup --keep_latest 5
        """
    )

    parser.add_argument('--mode', required=True,
                        choices=['train', 'evaluate',
                                 'compare', 'demo', 'cleanup'],
                        help='動作モード')

    # 共通パラメータ
    parser.add_argument('--model_id', type=str, help='モデルID')
    parser.add_argument('--use_real_data', action='store_true',
                        help='実データを使用（デフォルト：合成データ）')
    parser.add_argument('--random_seed', type=int, default=42, help='乱数シード')

    # 訓練パラメータ
    parser.add_argument('--training_samples', type=int, default=1000,
                        help='訓練サンプル数（合成データ使用時）')
    parser.add_argument('--population_size', type=int,
                        default=50, help='個体群サイズ')
    parser.add_argument('--generations', type=int, default=30, help='世代数')
    parser.add_argument('--mutation_rate', type=float,
                        default=0.15, help='突然変異率')
    parser.add_argument('--crossover_rate', type=float,
                        default=0.8, help='交叉率')
    parser.add_argument('--max_depth', type=int, default=6, help='最大木深度')
    parser.add_argument('--tournament_size', type=int,
                        default=3, help='トーナメントサイズ')
    parser.add_argument('--generate_report', action='store_true',
                        help='詳細レポート生成')

    # 評価パラメータ
    parser.add_argument('--show_explanations', action='store_true',
                        help='説明文を表示')

    # 清理パラメータ
    parser.add_argument('--keep_latest', type=int, default=10,
                        help='保持する最新モデル数')
    parser.add_argument('--min_fitness', type=float, default=0.5,
                        help='保持する最低適応度')

    args = parser.parse_args()

    # ヘッダー表示
    print("🧬 Advanced Genetic Fuzzy Decision Tree System")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    try:
        if args.mode == 'train':
            model_id = train_genetic_model(args)
            print(f"\n🎉 Training completed! Model ID: {model_id}")

        elif args.mode == 'evaluate':
            if not args.model_id:
                print("❌ Model ID required for evaluation mode")
                return
            evaluate_model(args)

        elif args.mode == 'compare':
            compare_models(args)

        elif args.mode == 'demo':
            demonstrate_prediction(args)

        elif args.mode == 'cleanup':
            cleanup_models(args)

    except KeyboardInterrupt:
        print("\n\n⚠️ Operation interrupted by user")
    except Exception as e:
        print(f"\n❌ Operation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
