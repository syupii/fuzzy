# backend/test_genetic_integration.py
from train_genetic_model import GeneticFuzzyTreeTrainer
from explanation_engine import AdvancedExplanationEngine
from model_persistence import ModelPersistence
from evaluation_metrics import MultiObjectiveEvaluator
from optimization_tracker import OptimizationTracker
from advanced_nodes import (
    AdvancedFuzzyDecisionNode,
    AdvancedMembershipFunction,
    NodeSplitInfo
)
from genetic_fuzzy_tree import (
    GeneticFuzzyTreeOptimizer,
    GeneticParameters,
    GeneticIndividual
)
import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
import os
import sys
from pathlib import Path
import logging

# テスト対象モジュールのインポート
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# テスト用ログ設定
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class TestGeneticFuzzyTreeIntegration(unittest.TestCase):
    """遺伝的ファジィ決定木統合テスト"""

    def setUp(self):
        """テストセットアップ"""

        # 乱数シード固定
        np.random.seed(42)

        # テスト用データ生成
        self.create_test_data()

        # 一時ディレクトリ作成
        self.temp_dir = tempfile.mkdtemp()

        # コンポーネント初期化
        self.genetic_params = GeneticParameters(
            population_size=10,  # テスト用に小さく
            generations=5,       # テスト用に短く
            crossover_rate=0.8,
            mutation_rate=0.2,
            max_tree_depth=3     # テスト用に浅く
        )

        self.optimizer = GeneticFuzzyTreeOptimizer(self.genetic_params)
        self.tracker = OptimizationTracker(
            tracking_db_path=os.path.join(self.temp_dir, "test_tracking.db")
        )
        self.evaluator = MultiObjectiveEvaluator()
        self.persistence = ModelPersistence(
            models_directory=os.path.join(self.temp_dir, "models"),
            metadata_db_path=os.path.join(self.temp_dir, "metadata.db")
        )
        self.explanation_engine = AdvancedExplanationEngine()

    def tearDown(self):
        """テストクリーンアップ"""
        # 一時ディレクトリ削除
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_test_data(self):
        """テストデータ作成"""

        n_samples = 50

        # 5つの特徴量（研究室選択シナリオ）
        self.feature_names = [
            'research_intensity',
            'advisor_style',
            'team_work',
            'workload',
            'theory_practice'
        ]

        # ランダムな特徴量生成（1-10スケール）
        data = {}
        for feature in self.feature_names:
            data[feature] = np.random.uniform(1, 10, n_samples)

        self.training_data = pd.DataFrame(data)

        # 目標変数（複雑な非線形関係）
        self.target_values = (
            0.3 * (data['research_intensity'] / 10) +
            0.2 * (1 - np.abs(data['advisor_style'] - 6) / 5) +
            0.2 * (data['team_work'] / 10) +
            0.15 * (1 - np.abs(data['workload'] - 7) / 4) +
            0.15 * (data['theory_practice'] / 10) +
            np.random.normal(0, 0.05, n_samples)  # ノイズ
        )

        # 0-1範囲に正規化
        self.target_values = np.clip(self.target_values, 0, 1)

        # 検証データ分割
        split_idx = int(n_samples * 0.8)
        self.train_data = self.training_data.iloc[:split_idx]
        self.train_targets = self.target_values[:split_idx]
        self.val_data = self.training_data.iloc[split_idx:]
        self.val_targets = self.target_values[split_idx:]

    def test_membership_function_creation(self):
        """メンバーシップ関数作成テスト"""

        # 三角メンバーシップ関数
        mf = AdvancedMembershipFunction(
            name="medium",
            function_type="triangular",
            parameters=[2.0, 5.0, 8.0],
            weight=1.0
        )

        # 境界値テスト
        self.assertEqual(mf.membership(2.0), 0.0)
        self.assertEqual(mf.membership(5.0), 1.0)
        self.assertEqual(mf.membership(8.0), 0.0)
        self.assertAlmostEqual(mf.membership(3.5), 0.5, places=5)

        # ガウシアンメンバーシップ関数
        mf_gaussian = AdvancedMembershipFunction(
            name="gaussian",
            function_type="gaussian",
            parameters=[5.0, 1.0],  # mu=5, sigma=1
            weight=1.0
        )

        self.assertAlmostEqual(mf_gaussian.membership(5.0), 1.0, places=5)
        self.assertGreater(mf_gaussian.membership(4.0), 0.5)
        self.assertLess(mf_gaussian.membership(3.0), 0.5)

    def test_decision_node_creation(self):
        """決定ノード作成テスト"""

        # メンバーシップ関数作成
        mf_low = AdvancedMembershipFunction("low", "triangular", [1, 3, 5])
        mf_high = AdvancedMembershipFunction("high", "triangular", [5, 7, 9])

        # 分割情報作成
        split_info = NodeSplitInfo(
            feature_index=0,
            feature_name="research_intensity",
            split_type="fuzzy_threshold",
            membership_functions={"low": mf_low, "high": mf_high},
            information_gain=0.5,
            split_quality=0.8
        )

        # リーフノード作成
        leaf_low = AdvancedFuzzyDecisionNode(
            node_id="leaf_low",
            depth=1,
            leaf_value=0.3
        )

        leaf_high = AdvancedFuzzyDecisionNode(
            node_id="leaf_high",
            depth=1,
            leaf_value=0.7
        )

        # ルートノード作成
        root = AdvancedFuzzyDecisionNode(
            node_id="root",
            depth=0,
            split_info=split_info,
            children={"low": leaf_low, "high": leaf_high}
        )

        # 予測テスト
        prediction = root.predict([2.0, 0, 0, 0, 0], self.feature_names)
        self.assertIsInstance(prediction, float)
        self.assertGreaterEqual(prediction, 0.0)
        self.assertLessEqual(prediction, 1.0)

        # 説明付き予測テスト
        prediction_with_explanation = root.predict_with_explanation(
            [7.0, 0, 0, 0, 0], self.feature_names)
        self.assertIsInstance(prediction_with_explanation, tuple)
        self.assertEqual(len(prediction_with_explanation), 2)

    def test_genetic_individual_creation(self):
        """遺伝的個体作成テスト"""

        individual = GeneticIndividual()
        self.assertIsNotNone(individual.individual_id)
        self.assertEqual(individual.generation, 0)

        # 遺伝子初期化
        individual.initialize_random_genome(
            feature_count=len(self.feature_names),
            max_depth=3
        )

        self.assertGreater(len(individual.genome), 0)
        self.assertIsInstance(individual.genome, list)

        # 決定木デコード
        tree = individual.decode_genome_to_tree(
            self.feature_names,
            self.train_data,
            self.train_targets
        )

        self.assertIsInstance(tree, AdvancedFuzzyDecisionNode)

    def test_genetic_optimization(self):
        """遺伝的最適化テスト"""

        # 最適化実行
        best_individual = self.optimizer.optimize(
            training_data=self.train_data,
            target_values=self.train_targets,
            validation_data=self.val_data,
            validation_targets=self.val_targets,
            feature_names=self.feature_names
        )

        # 結果検証
        self.assertIsInstance(best_individual, GeneticIndividual)
        self.assertIsNotNone(best_individual.tree)
        self.assertGreater(best_individual.fitness_components.overall, 0.0)
        self.assertLessEqual(best_individual.fitness_components.overall, 1.0)

        # 予測実行
        sample_input = self.train_data.iloc[0].values.tolist()
        prediction = best_individual.tree.predict(
            sample_input, self.feature_names)

        self.assertIsInstance(prediction, float)
        self.assertGreaterEqual(prediction, 0.0)
        self.assertLessEqual(prediction, 1.0)

    def test_optimization_tracking(self):
        """最適化追跡テスト"""

        self.tracker.start_optimization_tracking()

        # ダミー個体作成
        individuals = []
        for i in range(5):
            individual = GeneticIndividual()
            individual.fitness_components.overall = np.random.uniform(0.3, 0.9)
            individual.fitness_components.accuracy = np.random.uniform(
                0.4, 0.8)
            individual.fitness_components.simplicity = np.random.uniform(
                0.5, 0.9)
            individuals.append(individual)

        # 世代記録
        stats = self.tracker.record_generation(
            generation=0,
            population=individuals
        )

        self.assertIsNotNone(stats)
        self.assertEqual(stats.generation, 0)
        self.assertEqual(stats.population_size, 5)
        self.assertGreater(stats.best_fitness, 0.0)

        self.tracker.finish_optimization_tracking()

        # レポート生成
        report = self.tracker.generate_optimization_report()
        self.assertIn('optimization_summary', report)
        self.assertIn('final_population_stats', report)

    def test_model_persistence(self):
        """モデル永続化テスト"""

        # テスト用個体作成
        individual = GeneticIndividual()
        individual.initialize_random_genome(len(self.feature_names), 3)
        individual.decode_genome_to_tree(
            self.feature_names, self.train_data, self.train_targets)
        individual.fitness_components.overall = 0.75

        # ダミーオプティマイザー
        class DummyOptimizer:
            def __init__(self):
                self.parameters = self.genetic_params
                self.convergence_generation = None
                self.generation_stats = []

            def get_optimization_summary(self):
                return {
                    'total_generations': 5,
                    'best_fitness': individual.fitness_components.overall
                }

        dummy_optimizer = DummyOptimizer()

        # モデル保存
        model_id = self.persistence.save_model(
            genetic_individual=individual,
            optimizer=dummy_optimizer,
            feature_names=self.feature_names,
            training_sample_count=len(self.train_data),
            validation_sample_count=len(self.val_data),
            tags=['test_model'],
            description="Test model for integration testing"
        )

        self.assertIsNotNone(model_id)

        # モデル読み込み
        loaded_individual, model_data = self.persistence.load_model(model_id)

        self.assertIsInstance(loaded_individual, GeneticIndividual)
        self.assertEqual(loaded_individual.individual_id,
                         individual.individual_id)
        self.assertAlmostEqual(
            loaded_individual.fitness_components.overall,
            individual.fitness_components.overall,
            places=6
        )

        # モデル一覧
        models = self.persistence.list_models()
        self.assertGreater(len(models), 0)
        self.assertEqual(models[0]['model_id'], model_id)

        # モデル削除
        success = self.persistence.delete_model(model_id)
        self.assertTrue(success)

    def test_multi_objective_evaluation(self):
        """多目的評価テスト"""

        # テスト用個体作成
        individual = GeneticIndividual()
        individual.initialize_random_genome(len(self.feature_names), 3)
        tree = individual.decode_genome_to_tree(
            self.feature_names, self.train_data, self.train_targets)
        individual.tree = tree

        # 評価実行
        fitness_components = self.evaluator.evaluate_individual(
            individual=individual,
            training_data=self.train_data,
            target_values=self.train_targets,
            validation_data=self.val_data,
            validation_targets=self.val_targets,
            feature_names=self.feature_names
        )

        # 結果検証
        self.assertIsNotNone(fitness_components)
        self.assertGreaterEqual(fitness_components.accuracy, 0.0)
        self.assertLessEqual(fitness_components.accuracy, 1.0)
        self.assertGreaterEqual(fitness_components.simplicity, 0.0)
        self.assertLessEqual(fitness_components.simplicity, 1.0)
        self.assertGreaterEqual(fitness_components.interpretability, 0.0)
        self.assertLessEqual(fitness_components.interpretability, 1.0)
        self.assertGreaterEqual(fitness_components.overall, 0.0)
        self.assertLessEqual(fitness_components.overall, 1.0)

    def test_explanation_engine(self):
        """説明エンジンテスト"""

        # テスト用決定木作成
        individual = GeneticIndividual()
        individual.initialize_random_genome(len(self.feature_names), 3)
        tree = individual.decode_genome_to_tree(
            self.feature_names, self.train_data, self.train_targets)

        # サンプル入力
        sample_input = self.train_data.iloc[0].values.tolist()

        # 説明付き予測
        prediction, explanation = tree.predict_with_explanation(
            sample_input, self.feature_names)

        # 包括的説明生成
        comprehensive_explanation = self.explanation_engine.generate_comprehensive_explanation(
            prediction=prediction,
            explanation=explanation,
            feature_vector=sample_input,
            feature_names=self.feature_names,
            decision_tree=tree
        )

        # 結果検証
        self.assertIsInstance(comprehensive_explanation, dict)
        self.assertIn('prediction_value', comprehensive_explanation)
        self.assertIn('confidence', comprehensive_explanation)
        self.assertIn('feature_importance', comprehensive_explanation)
        self.assertIn('narrative_explanation', comprehensive_explanation)
        self.assertIn('reliability_assessment', comprehensive_explanation)

        # 予測値の妥当性
        self.assertIsInstance(
            comprehensive_explanation['prediction_value'], float)
        self.assertGreaterEqual(
            comprehensive_explanation['prediction_value'], 0.0)
        self.assertLessEqual(
            comprehensive_explanation['prediction_value'], 1.0)

    def test_full_training_pipeline(self):
        """完全学習パイプラインテスト"""

        # トレーナー設定
        config = {
            'genetic_parameters': {
                'population_size': 8,
                'generations': 3,
                'crossover_rate': 0.8,
                'mutation_rate': 0.2,
                'max_tree_depth': 3
            },
            'data_parameters': {
                'validation_split': 0.2,
                'random_state': 42
            },
            'output_parameters': {
                'save_model': True,
                'save_tracking': False,  # テストでは無効化
                'generate_report': False,  # テストでは無効化
                'create_visualizations': False,  # テストでは無効化
                'model_output_dir': os.path.join(self.temp_dir, 'models')
            }
        }

        # トレーナー初期化
        trainer = GeneticFuzzyTreeTrainer(config)

        # データ準備（直接指定）
        trainer.prepare_data(
            training_data=self.training_data,
            target_values=self.target_values,
            feature_names=self.feature_names
        )

        # コンポーネントセットアップ
        trainer.setup_components()

        # 学習実行
        best_individual = trainer.train()

        # 結果検証
        self.assertIsInstance(best_individual, GeneticIndividual)
        self.assertIsNotNone(best_individual.tree)
        self.assertGreater(best_individual.fitness_components.overall, 0.0)

        # 評価実行
        if trainer.validation_data is not None:
            evaluation_results = trainer.evaluate_model()
            self.assertIsInstance(evaluation_results, dict)
            self.assertIn('mae', evaluation_results)
            self.assertIn('rmse', evaluation_results)
            self.assertGreater(evaluation_results['test_samples'], 0)

    def test_serialization_deserialization(self):
        """シリアライゼーション・デシリアライゼーションテスト"""

        from model_persistence import ModelSerializer

        # メンバーシップ関数テスト
        mf_original = AdvancedMembershipFunction(
            name="test_mf",
            function_type="triangular",
            parameters=[1.0, 5.0, 9.0],
            weight=0.8
        )

        mf_serialized = ModelSerializer.serialize_membership_function(
            mf_original)
        mf_deserialized = ModelSerializer.deserialize_membership_function(
            mf_serialized)

        self.assertEqual(mf_original.name, mf_deserialized.name)
        self.assertEqual(mf_original.function_type,
                         mf_deserialized.function_type)
        self.assertEqual(mf_original.parameters, mf_deserialized.parameters)
        self.assertEqual(mf_original.weight, mf_deserialized.weight)

        # 決定ノードテスト
        node_original = AdvancedFuzzyDecisionNode(
            node_id="test_node",
            depth=0,
            leaf_value=0.7
        )

        node_serialized = ModelSerializer.serialize_decision_node(
            node_original)
        node_deserialized = ModelSerializer.deserialize_decision_node(
            node_serialized)

        self.assertEqual(node_original.node_id, node_deserialized.node_id)
        self.assertEqual(node_original.depth, node_deserialized.depth)
        self.assertEqual(node_original.is_leaf, node_deserialized.is_leaf)
        self.assertEqual(node_original.leaf_value,
                         node_deserialized.leaf_value)

    def test_error_handling(self):
        """エラーハンドリングテスト"""

        # 不正なデータでの最適化
        empty_data = pd.DataFrame()
        empty_targets = np.array([])

        with self.assertRaises((ValueError, IndexError)):
            self.optimizer.optimize(
                training_data=empty_data,
                target_values=empty_targets,
                feature_names=[]
            )

        # 存在しないモデルの読み込み
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.persistence.load_model("non_existent_model_id")

        # 不正なメンバーシップ関数パラメータ
        invalid_mf = AdvancedMembershipFunction(
            name="invalid",
            function_type="triangular",
            parameters=[5.0, 3.0, 1.0],  # 不正な順序
            weight=1.0
        )

        # エラーは発生させずに適切に処理されることを確認
        result = invalid_mf.membership(4.0)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)


class TestPerformanceBenchmarks(unittest.TestCase):
    """パフォーマンスベンチマークテスト"""

    def setUp(self):
        """テストセットアップ"""
        np.random.seed(42)

        # 中規模テストデータ生成
        n_samples = 200
        n_features = 5

        self.feature_names = [f'feature_{i}' for i in range(n_features)]

        data = {}
        for i, feature in enumerate(self.feature_names):
            data[feature] = np.random.uniform(1, 10, n_samples)

        self.training_data = pd.DataFrame(data)
        self.target_values = np.random.uniform(0, 1, n_samples)

        # パフォーマンス測定用パラメータ
        self.genetic_params = GeneticParameters(
            population_size=20,
            generations=10,
            crossover_rate=0.8,
            mutation_rate=0.2,
            max_tree_depth=4
        )

    def test_optimization_speed(self):
        """最適化速度テスト"""

        import time

        optimizer = GeneticFuzzyTreeOptimizer(self.genetic_params)

        start_time = time.time()

        best_individual = optimizer.optimize(
            training_data=self.training_data,
            target_values=self.target_values,
            feature_names=self.feature_names
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # 実行時間の妥当性チェック（環境依存）
        self.assertLess(execution_time, 300)  # 5分以内
        self.assertIsNotNone(best_individual)

        print(f"\n最適化実行時間: {execution_time:.2f}秒")
        print(f"最良適応度: {best_individual.fitness_components.overall:.4f}")

    def test_prediction_speed(self):
        """予測速度テスト"""

        import time

        # 簡単な決定木作成
        individual = GeneticIndividual()
        individual.initialize_random_genome(len(self.feature_names), 3)
        tree = individual.decode_genome_to_tree(
            self.feature_names,
            self.training_data,
            self.target_values
        )

        # 大量予測のタイミング測定
        n_predictions = 1000
        test_samples = []

        for _ in range(n_predictions):
            sample = np.random.uniform(1, 10, len(self.feature_names)).tolist()
            test_samples.append(sample)

        start_time = time.time()

        predictions = []
        for sample in test_samples:
            prediction = tree.predict(sample, self.feature_names)
            predictions.append(prediction)

        end_time = time.time()
        prediction_time = end_time - start_time
        avg_time_per_prediction = prediction_time / n_predictions

        # 予測速度の妥当性チェック
        self.assertLess(avg_time_per_prediction, 0.01)  # 10ms以内/予測
        self.assertEqual(len(predictions), n_predictions)

        print(f"\n予測総時間: {prediction_time:.4f}秒")
        print(f"平均予測時間: {avg_time_per_prediction*1000:.2f}ms/予測")

    def test_memory_usage(self):
        """メモリ使用量テスト"""

        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 複数モデルの作成と保存
        models = []
        for i in range(10):
            individual = GeneticIndividual()
            individual.initialize_random_genome(len(self.feature_names), 4)
            tree = individual.decode_genome_to_tree(
                self.feature_names,
                self.training_data,
                self.target_values
            )
            models.append(individual)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # メモリ増加の妥当性チェック
        self.assertLess(memory_increase, 100)  # 100MB以内

        print(f"\n初期メモリ: {initial_memory:.1f}MB")
        print(f"最終メモリ: {final_memory:.1f}MB")
        print(f"メモリ増加: {memory_increase:.1f}MB")


def run_integration_tests():
    """統合テスト実行"""

    print("🧪 遺伝的ファジィ決定木統合テスト開始")

    # テストスイート作成
    test_suite = unittest.TestSuite()

    # 基本統合テスト
    test_suite.addTest(unittest.makeSuite(TestGeneticFuzzyTreeIntegration))

    # パフォーマンステスト
    test_suite.addTest(unittest.makeSuite(TestPerformanceBenchmarks))

    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # 結果サマリー
    print(f"\n📊 テスト結果サマリー:")
    print(f"   実行テスト数: {result.testsRun}")
    print(
        f"   成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"   失敗: {len(result.failures)}")
    print(f"   エラー: {len(result.errors)}")

    if result.failures:
        print(f"\n❌ 失敗したテスト:")
        for test, traceback in result.failures:
            print(f"   - {test}")

    if result.errors:
        print(f"\n💥 エラーが発生したテスト:")
        for test, traceback in result.errors:
            print(f"   - {test}")

    success = len(result.failures) == 0 and len(result.errors) == 0

    if success:
        print(f"\n🎉 全テスト成功！")
    else:
        print(f"\n⚠️ 一部テストが失敗しました。")

    return success


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)
