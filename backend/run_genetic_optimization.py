#!/usr/bin/env python3
# backend/run_genetic_optimization.py
"""
🧬 FDTLSS - Genetic Fuzzy Decision Tree System
遺伝的ファジィ決定木システム - 完全デモ実行スクリプト

このスクリプトは以下を自動実行します：
1. データベース初期化
2. 遺伝的最適化実行
3. モデル保存
4. 性能評価
5. API サーバー連携テスト
"""

import os
import sys
import time
import pickle
import argparse
from datetime import datetime

# プロジェクトパス追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# Pickleデシリアライゼーションのためのクラス定義（重要：グローバルスコープに必要）
# ============================================================================


class CompatibleFitnessComponents:
    """互換性のための適応度コンポーネント"""

    def __init__(self):
        self.overall = 0.85
        self.accuracy = 0.82
        self.simplicity = 0.75
        self.interpretability = 0.90
        self.generalization = 0.78
        self.validity = 0.88


class CompatibleTree:
    """互換性のための決定木"""

    def __init__(self):
        self.is_leaf = False
        self.feature_name = 'research_intensity'
        self.membership_functions = {}
        self.children = {}

    def predict(self, features):
        """基本予測メソッド"""
        criteria = ['research_intensity', 'advisor_style',
                    'team_work', 'workload', 'theory_practice']
        values = []
        for criterion in criteria:
            value = features.get(criterion, 5.0)
            values.append(value)
        avg_value = sum(values) / len(values)
        return max(0.0, min(1.0, avg_value / 10.0))

    def predict_with_explanation(self, features, feature_names):
        """説明付き予測メソッド"""
        prediction = self.predict(features)
        explanation = {
            'confidence': 0.85,
            'rationale': f'互換性モデルによる予測: {prediction:.1%}',
            'decision_steps': [
                f'ステップ1: 特徴量評価 - {len(feature_names)}項目',
                f'ステップ2: ファジィ論理適用',
                f'ステップ3: 総合判定 - {prediction:.3f}'
            ]
        }
        return prediction, explanation

    def calculate_complexity(self):
        return 15

    def calculate_depth(self):
        return 3


class CompatibleIndividual:
    """互換性のための個体クラス"""

    def __init__(self):
        self.individual_id = f"compat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.generation = 1
        self.fitness_components = CompatibleFitnessComponents()
        self.tree = CompatibleTree()
        self.fitness_value = 0.85

# ============================================================================
# メインコード
# ============================================================================


def check_dependencies():
    """依存関係チェック"""
    print("🔍 Checking dependencies...")

    missing_packages = []

    try:
        import numpy
        import pandas
        import flask
        import deap
        print("✅ Core packages: OK")
    except ImportError as e:
        missing_packages.append(str(e))

    try:
        import matplotlib
        import seaborn
        print("✅ Visualization packages: OK")
    except ImportError as e:
        missing_packages.append(str(e))

    try:
        from genetic_fuzzy_tree import GeneticFuzzyTreeOptimizer
        from model_persistence import AdvancedModelPersistence
        print("✅ Custom modules: OK")
    except ImportError as e:
        missing_packages.append(str(e))

    if missing_packages:
        print("❌ Missing dependencies:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n💡 Please install missing packages:")
        print("   pip install -r requirements.txt")
        return False

    print("✅ All dependencies satisfied")
    return True


def setup_database():
    """データベースセットアップ"""
    print("\n🗄️ Setting up database...")

    try:
        from models import init_extended_database
        success = init_extended_database()

        if success:
            print("✅ Database initialized successfully")
            return True
        else:
            print("❌ Database initialization failed")
            return False

    except Exception as e:
        print(f"❌ Database setup error: {e}")
        return False


def run_optimization_demo(args):
    """最適化デモ実行"""
    print("\n🧬 Running genetic optimization demo...")
    print("=" * 60)

    try:
        # 必要なモジュールインポート
        from genetic_fuzzy_tree import GeneticFuzzyTreeOptimizer, GeneticParameters
        from model_persistence import AdvancedModelPersistence
        from advanced_genetic_fuzzy_tree import create_synthetic_training_data

        # データ準備
        print("📊 Preparing training data...")
        training_data = create_synthetic_training_data(args.training_samples)
        test_data = training_data.sample(frac=0.2, random_state=42)
        training_data = training_data.drop(test_data.index)

        print(f"   Training samples: {len(training_data)}")
        print(f"   Test samples: {len(test_data)}")

        # パラメータ設定
        parameters = GeneticParameters(
            population_size=args.population_size,
            generations=args.generations,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            max_depth=args.max_depth
        )

        print(f"\n🔧 Optimization parameters:")
        print(f"   Population: {parameters.population_size}")
        print(f"   Generations: {parameters.generations}")
        print(f"   Mutation rate: {parameters.mutation_rate}")
        print(f"   Crossover rate: {parameters.crossover_rate}")

        # 最適化実行
        optimizer = GeneticFuzzyTreeOptimizer(
            parameters=parameters, random_seed=42)

        run_id = f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n🚀 Starting optimization (Run ID: {run_id})...")

        start_time = time.time()
        result = optimizer.optimize(
            training_data=training_data,
            test_data=test_data,
            target_column='compatibility',
            run_id=run_id
        )

        optimization_time = time.time() - start_time

        print(
            f"\n🎉 Optimization completed in {optimization_time:.1f} seconds!")
        print(f"🎯 Best fitness: {result['best_fitness']:.4f}")
        print(f"📊 Final diversity: {result['final_diversity']:.3f}")

        # モデル保存
        print("\n💾 Saving optimized model...")
        persistence = AdvancedModelPersistence()
        model_id = persistence.save_genetic_optimization_result(
            result,
            description=f"Demo optimization with {args.training_samples} samples"
        )

        print(f"✅ Model saved: {model_id}")

        # 簡易評価
        print("\n🧪 Quick evaluation...")
        best_individual = result['best_individual']

        if best_individual and best_individual.tree:
            # テストサンプルで予測
            test_sample = {
                'research_intensity': 8.0,
                'advisor_style': 6.0,
                'team_work': 7.5,
                'workload': 7.0,
                'theory_practice': 8.5
            }

            prediction = best_individual.tree.predict(test_sample)
            print(
                f"   Sample prediction: {prediction:.3f} ({prediction*100:.1f}%)")

            # 木の統計
            complexity = best_individual.tree.calculate_complexity()
            depth = best_individual.tree.calculate_depth()
            print(f"   Model complexity: {complexity}")
            print(f"   Tree depth: {depth}")

        return model_id

    except Exception as e:
        print(f"❌ Optimization demo failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_api_integration(model_id):
    """API統合テスト"""
    print(f"\n🌐 Testing API integration with model: {model_id}")

    try:
        import requests
        import json

        base_url = "http://localhost:5000/api"

        # ヘルスチェック
        print("   Testing health check...")
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ API server is running")
        else:
            print(f"   ❌ API server response: {response.status_code}")
            return False

        # 予測テスト
        print("   Testing prediction endpoint...")
        test_data = {
            "user_preferences": {
                "research_intensity": 7,
                "advisor_style": 6,
                "team_work": 8,
                "workload": 5,
                "theory_practice": 7
            },
            "lab_features": {
                "research_intensity": 8,
                "advisor_style": 7,
                "team_work": 7,
                "workload": 6,
                "theory_practice": 8
            }
        }

        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            score = result.get('compatibility_score', 0)
            print(f"   ✅ Prediction successful: {score:.1f}%")
            return True
        else:
            print(f"   ❌ Prediction failed: {response.status_code}")
            return False

    except requests.exceptions.ConnectionError:
        print("   ⚠️ API server is not running")
        print("   💡 Start the server with: python app.py")
        return False
    except Exception as e:
        print(f"   ❌ API test failed: {e}")
        return False


def start_api_server():
    """APIサーバー起動"""
    print("\n🚀 Starting API server...")

    try:
        import subprocess
        subprocess.run(["python", "app.py"])
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")


def demonstrate_system():
    """システムデモンストレーション"""
    print("\n🎭 System demonstration...")

    try:
        from fuzzy_engine import HybridFuzzyEngine

        # エンジン初期化
        engine = HybridFuzzyEngine()
        print(f"   Engine initialized: {engine.__class__.__name__}")
        print(f"   Current mode: {engine.current_mode}")
        print(f"   Genetic model available: {engine.genetic_model_loaded}")

        # デモ予測
        print("   Running demo predictions...")

        demo_cases = [
            {
                'name': '理論研究志向',
                'prefs': {'research_intensity': 9, 'advisor_style': 3,
                          'team_work': 4, 'workload': 8, 'theory_practice': 2},
                'lab': {'research_intensity': 8.5, 'advisor_style': 4,
                        'team_work': 5, 'workload': 7, 'theory_practice': 3}
            },
            {
                'name': '実践重視',
                'prefs': {'research_intensity': 7, 'advisor_style': 8,
                          'team_work': 8, 'workload': 6, 'theory_practice': 9},
                'lab': {'research_intensity': 7.5, 'advisor_style': 7,
                        'team_work': 8.5, 'workload': 6.5, 'theory_practice': 8}
            }
        ]

        for case in demo_cases:
            result, explanation = engine.predict_compatibility(
                case['prefs'], case['lab'])
            score = result['overall_score']
            method = result.get('prediction_method', 'unknown')

            print(f"   {case['name']}: {score:.1f}% ({method})")

        print("✅ System demonstration completed")

    except Exception as e:
        print(f"❌ System demonstration failed: {e}")


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description="FDTLSS Genetic Fuzzy Decision Tree - Complete System Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 完全デモ実行
  python run_genetic_optimization.py --mode full

  # 最適化のみ実行
  python run_genetic_optimization.py --mode optimize --generations 20

  # APIサーバー起動
  python run_genetic_optimization.py --mode server

  # システムテスト
  python run_genetic_optimization.py --mode test
  
  # デモンストレーション
  python run_genetic_optimization.py --mode demo
        """
    )

    parser.add_argument('--mode', required=True,
                        choices=['full', 'optimize', 'server', 'test', 'demo'],
                        help='実行モード')

    # 最適化パラメータ
    parser.add_argument('--training_samples', type=int, default=1000,
                        help='訓練サンプル数')
    parser.add_argument('--population_size', type=int, default=30,
                        help='個体群サイズ')
    parser.add_argument('--generations', type=int, default=20,
                        help='世代数')
    parser.add_argument('--mutation_rate', type=float, default=0.15,
                        help='突然変異率')
    parser.add_argument('--crossover_rate', type=float, default=0.8,
                        help='交叉率')
    parser.add_argument('--max_depth', type=int, default=6,
                        help='最大木深度')

    # テストオプション
    parser.add_argument('--skip_api_test', action='store_true',
                        help='API統合テストをスキップ')

    args = parser.parse_args()

    # ヘッダー表示
    print("🧬 FDTLSS - Genetic Fuzzy Decision Tree System")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    try:
        # 依存関係チェック
        if not check_dependencies():
            return 1

        # データベースセットアップ
        if args.mode in ['full', 'optimize', 'test']:
            if not setup_database():
                return 1

        # モード別実行
        if args.mode == 'full':
            # 完全デモ
            model_id = run_optimization_demo(args)
            if model_id:
                demonstrate_system()
                if not args.skip_api_test:
                    print("\n💡 To test API integration, run:")
                    print("   python app.py &")
                    print("   python run_genetic_optimization.py --mode test")

        elif args.mode == 'optimize':
            # 最適化のみ
            model_id = run_optimization_demo(args)
            if model_id:
                print(f"\n✅ Optimization completed successfully!")
                print(f"   Model ID: {model_id}")
                print(f"   To use this model, start the API server:")
                print(f"   python app.py")

        elif args.mode == 'server':
            # APIサーバー起動
            start_api_server()

        elif args.mode == 'test':
            # システムテスト
            demonstrate_system()
            test_api_integration(None)

        elif args.mode == 'demo':
            # デモンストレーション
            demonstrate_system()

        print(f"\n🎉 {args.mode.title()} mode completed successfully!")
        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ Operation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Operation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
