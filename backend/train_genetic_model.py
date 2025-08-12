# backend/train_genetic_model.py
"""
遺伝的ファジィ決定木学習実行スクリプト

使用例:
    python train_genetic_model.py --generations 50 --population_size 30
    python train_genetic_model.py --config config.json
"""

import argparse
import json
import logging
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# 自作モジュールインポート
from genetic_fuzzy_tree import GeneticFuzzyTreeOptimizer, GeneticParameters, GeneticIndividual
from optimization_tracker import OptimizationTracker
from evaluation_metrics import MultiObjectiveEvaluator
from model_persistence import ModelPersistence
from explanation_engine import AdvancedExplanationEngine

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('genetic_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TrainingConfiguration:
    """学習設定クラス"""

    def __init__(self, config_dict: Dict[str, Any] = None):
        # デフォルト設定
        self.genetic_parameters = GeneticParameters(
            population_size=50,
            generations=30,
            crossover_rate=0.8,
            mutation_rate=0.2,
            tournament_size=3,
            elitism_ratio=0.1,
            max_tree_depth=5,
            min_samples_split=5,
            min_samples_leaf=2
        )

        # データ設定
        self.data_settings = {
            'train_test_split_ratio': 0.8,
            'validation_split_ratio': 0.2,
            'random_state': 42,
            'normalize_features': True,
            'feature_selection': True,
            'max_features': 10
        }

        # 出力設定
        self.output_settings = {
            'save_model': True,
            'save_plots': True,
            'save_explanations': True,
            'output_directory': 'genetic_training_output',
            'model_name_prefix': 'genetic_fuzzy_tree'
        }

        # 追跡設定
        self.tracking_settings = {
            'enable_tracking': True,
            'detailed_logging': True,
            'save_genealogy': True,
            'create_visualizations': True
        }

        # 設定辞書で上書き
        if config_dict:
            self._update_from_dict(config_dict)

    def _update_from_dict(self, config_dict: Dict[str, Any]):
        """設定辞書から更新"""

        # 遺伝的パラメータ更新
        if 'genetic_parameters' in config_dict:
            genetic_config = config_dict['genetic_parameters']
            for key, value in genetic_config.items():
                if hasattr(self.genetic_parameters, key):
                    setattr(self.genetic_parameters, key, value)

        # その他の設定更新
        for section_name in ['data_settings', 'output_settings', 'tracking_settings']:
            if section_name in config_dict:
                getattr(self, section_name).update(config_dict[section_name])

    def save_to_file(self, filepath: str):
        """設定をファイルに保存"""
        config_dict = {
            'genetic_parameters': self.genetic_parameters.__dict__,
            'data_settings': self.data_settings,
            'output_settings': self.output_settings,
            'tracking_settings': self.tracking_settings
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'TrainingConfiguration':
        """ファイルから設定を読み込み"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        return cls(config_dict)


class GeneticTrainingPipeline:
    """遺伝的学習パイプライン"""

    def __init__(self, config: TrainingConfiguration):
        self.config = config
        self.output_dir = Path(config.output_settings['output_directory'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # コンポーネント初期化
        self.optimizer = None
        self.tracker = None
        self.evaluator = None
        self.persistence = None
        self.explanation_engine = None

        # 学習データ
        self.training_data = None
        self.validation_data = None
        self.target_values = None
        self.validation_targets = None
        self.feature_names = None

        # 結果
        self.best_individual = None
        self.training_history = None

        logger.info("🧬 遺伝的学習パイプライン初期化完了")

    def setup_components(self):
        """コンポーネントセットアップ"""

        # 最適化器
        self.optimizer = GeneticFuzzyTreeOptimizer(
            self.config.genetic_parameters)

        # 追跡器
        if self.config.tracking_settings['enable_tracking']:
            tracking_db_path = self.output_dir / "optimization_tracking.db"
            self.tracker = OptimizationTracker(
                tracking_db_path=str(tracking_db_path),
                enable_detailed_logging=self.config.tracking_settings['detailed_logging']
            )

        # 評価器
        self.evaluator = MultiObjectiveEvaluator()

        # 永続化
        models_dir = self.output_dir / "models"
        metadata_db_path = self.output_dir / "model_metadata.db"
        self.persistence = ModelPersistence(
            models_directory=str(models_dir),
            metadata_db_path=str(metadata_db_path)
        )

        # 説明エンジン
        self.explanation_engine = AdvancedExplanationEngine()

        logger.info("✅ 全コンポーネント初期化完了")

    def load_research_lab_data(self) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """研究室データ読み込み（プロトタイプ互換）"""

        # プロトタイプのデータベースから読み込み
        try:
            import sqlite3

            # プロトタイプのデータベースに接続
            conn = sqlite3.connect('fdtlss.db')

            # 研究室データ取得
            labs_df = pd.read_sql_query("""
                SELECT 
                    research_intensity,
                    advisor_style,
                    team_work,
                    workload,
                    theory_practice,
                    (research_intensity + advisor_style + team_work + workload + theory_practice) / 5.0 as target_score
                FROM labs 
                WHERE is_active = 1
            """, conn)

            conn.close()

            # 特徴量とターゲット分離
            feature_columns = ['research_intensity', 'advisor_style',
                               'team_work', 'workload', 'theory_practice']
            X = labs_df[feature_columns]
            y = labs_df['target_score'].values

            logger.info(f"📊 研究室データ読み込み完了: {len(X)}件")
            return X, y, feature_columns

        except Exception as e:
            logger.warning(f"プロトタイプDBからの読み込み失敗: {e}")
            return self._generate_synthetic_data()

    def _generate_synthetic_data(self) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """合成データ生成（フォールバック）"""

        logger.info("🔄 合成研究室データを生成中...")

        np.random.seed(42)
        n_samples = 300  # より多くのサンプル生成

        # 特徴量定義
        feature_names = [
            'research_intensity',     # 研究強度
            'advisor_style',         # 指導スタイル
            'team_work',            # チームワーク
            'workload',             # ワークロード
            'theory_practice',      # 理論・実践バランス
            'international_focus',  # 国際性
            'industry_connection',  # 産業連携
            'publication_rate',     # 論文発表率
            'student_satisfaction',  # 学生満足度
            'career_support'        # キャリア支援
        ]

        # 相関を考慮した特徴量生成
        data = {}

        # 基本特徴量（1-10スケール）
        data['research_intensity'] = np.random.normal(7.0, 1.5, n_samples)
        data['advisor_style'] = np.random.normal(6.0, 2.0, n_samples)
        data['team_work'] = np.random.normal(6.5, 1.8, n_samples)
        data['workload'] = np.random.normal(6.8, 1.6, n_samples)
        data['theory_practice'] = np.random.normal(6.2, 2.2, n_samples)

        # 拡張特徴量（相関を持たせる）
        data['international_focus'] = (
            0.3 * data['research_intensity'] +
            0.2 * data['advisor_style'] +
            np.random.normal(0, 1.0, n_samples)
        )

        data['industry_connection'] = (
            0.4 * data['theory_practice'] +
            0.2 * data['workload'] +
            np.random.normal(0, 1.2, n_samples)
        )

        data['publication_rate'] = (
            0.5 * data['research_intensity'] +
            0.3 * data['workload'] +
            np.random.normal(0, 1.0, n_samples)
        )

        data['student_satisfaction'] = (
            -0.2 * data['workload'] +
            0.3 * data['advisor_style'] +
            0.2 * data['team_work'] +
            np.random.normal(0, 1.0, n_samples)
        )

        data['career_support'] = (
            0.2 * data['advisor_style'] +
            0.3 * data['industry_connection'] +
            np.random.normal(0, 1.0, n_samples)
        )

        # 1-10の範囲にクリップ
        for feature in feature_names:
            data[feature] = np.clip(data[feature], 1.0, 10.0)

        # DataFrame作成
        X = pd.DataFrame(data)

        # 複雑な非線形ターゲット関数
        y = (
            0.25 * X['research_intensity'] +
            0.20 * X['advisor_style'] +
            0.15 * X['team_work'] +
            0.10 * X['workload'] +
            0.10 * X['theory_practice'] +
            0.08 * X['international_focus'] +
            0.07 * X['industry_connection'] +
            0.05 * X['publication_rate'] +
            # 非線形項
            0.02 * (X['research_intensity'] * X['theory_practice']) / 10.0 +
            0.02 * np.sin(X['advisor_style'] / 2.0) +
            0.01 * (X['team_work'] ** 1.5) / 10.0 +
            np.random.normal(0, 0.3, n_samples)  # ノイズ
        ) / 10.0  # 正規化

        # 0-1の範囲にクリップ
        y = np.clip(y, 0.0, 1.0)

        logger.info(f"📊 合成データ生成完了: {len(X)}件, {len(feature_names)}特徴量")

        return X, y, feature_names

    def prepare_data(self, X: pd.DataFrame, y: np.ndarray, feature_names: List[str]):
        """データ前処理"""

        logger.info("🔄 データ前処理開始...")

        # 訓練・検証分割
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.config.data_settings['validation_split_ratio'],
            random_state=self.config.data_settings['random_state'],
            stratify=None  # 回帰問題のため
        )

        # 特徴量選択（オプション）
        if self.config.data_settings['feature_selection']:
            selected_features = self._select_features(
                X_train, y_train, feature_names)
            X_train = X_train[selected_features]
            X_val = X_val[selected_features]
            feature_names = selected_features
            logger.info(f"📉 特徴量選択: {len(selected_features)}個選択")

        # 正規化（オプション）
        if self.config.data_settings['normalize_features']:
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
            X_val_scaled = pd.DataFrame(
                scaler.transform(X_val),
                columns=X_val.columns,
                index=X_val.index
            )

            # スケーラー保存
            import pickle
            scaler_path = self.output_dir / "feature_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)

            X_train, X_val = X_train_scaled, X_val_scaled
            logger.info("📏 特徴量正規化完了")

        # インスタンス変数設定
        self.training_data = X_train
        self.validation_data = X_val
        self.target_values = y_train
        self.validation_targets = y_val
        self.feature_names = feature_names

        logger.info(f"✅ データ前処理完了")
        logger.info(f"   訓練データ: {len(X_train)}件")
        logger.info(f"   検証データ: {len(X_val)}件")
        logger.info(f"   特徴量数: {len(feature_names)}")

    def _select_features(self, X: pd.DataFrame, y: np.ndarray, feature_names: List[str]) -> List[str]:
        """特徴量選択"""

        from sklearn.feature_selection import SelectKBest, f_regression

        max_features = min(
            self.config.data_settings['max_features'],
            len(feature_names)
        )

        selector = SelectKBest(score_func=f_regression, k=max_features)
        X_selected = selector.fit_transform(X, y)

        # 選択された特徴量のインデックス取得
        selected_indices = selector.get_support(indices=True)
        selected_features = [feature_names[i] for i in selected_indices]

        return selected_features

    def run_optimization(self) -> GeneticIndividual:
        """遺伝的最適化実行"""

        logger.info("🧬 遺伝的最適化開始...")

        # 追跡開始
        if self.tracker:
            self.tracker.start_optimization_tracking()

        start_time = time.time()

        # 最適化実行
        best_individual = self.optimizer.optimize(
            training_data=self.training_data,
            target_values=self.target_values,
            validation_data=self.validation_data,
            validation_targets=self.validation_targets,
            feature_names=self.feature_names
        )

        end_time = time.time()
        optimization_time = end_time - start_time

        # 追跡終了
        if self.tracker:
            self.tracker.finish_optimization_tracking()

        self.best_individual = best_individual

        logger.info("🎉 遺伝的最適化完了!")
        logger.info(f"⏱️ 実行時間: {optimization_time:.2f}秒")
        logger.info(
            f"🏆 最良適応度: {best_individual.fitness_components.overall:.4f}")
        logger.info(f"📊 適応度詳細:")
        logger.info(
            f"   精度: {best_individual.fitness_components.accuracy:.4f}")
        logger.info(
            f"   単純性: {best_individual.fitness_components.simplicity:.4f}")
        logger.info(
            f"   解釈可能性: {best_individual.fitness_components.interpretability:.4f}")

        return best_individual

    def evaluate_model(self, individual: GeneticIndividual) -> Dict[str, Any]:
        """モデル評価"""

        logger.info("📊 モデル詳細評価開始...")

        evaluation_results = {}

        # 基本性能評価
        train_predictions = []
        val_predictions = []

        # 訓練データでの予測
        for i in range(len(self.training_data)):
            feature_vector = self.training_data.iloc[i].values.tolist()
            prediction = individual.tree.predict(
                feature_vector, self.feature_names)
            train_predictions.append(prediction)

        # 検証データでの予測
        for i in range(len(self.validation_data)):
            feature_vector = self.validation_data.iloc[i].values.tolist()
            prediction = individual.tree.predict(
                feature_vector, self.feature_names)
            val_predictions.append(prediction)

        # 性能指標計算
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        train_mse = mean_squared_error(self.target_values, train_predictions)
        train_mae = mean_absolute_error(self.target_values, train_predictions)
        train_r2 = r2_score(self.target_values, train_predictions)

        val_mse = mean_squared_error(self.validation_targets, val_predictions)
        val_mae = mean_absolute_error(self.validation_targets, val_predictions)
        val_r2 = r2_score(self.validation_targets, val_predictions)

        evaluation_results['performance_metrics'] = {
            'train': {'mse': train_mse, 'mae': train_mae, 'r2': train_r2},
            'validation': {'mse': val_mse, 'mae': val_mae, 'r2': val_r2}
        }

        # 構造分析
        node_count = self._count_nodes(individual.tree)
        tree_depth = self._calculate_depth(individual.tree)
        leaf_count = self._count_leaves(individual.tree)

        evaluation_results['structure_analysis'] = {
            'node_count': node_count,
            'tree_depth': tree_depth,
            'leaf_count': leaf_count,
            'balance_ratio': leaf_count / max(1, node_count - leaf_count)
        }

        # 特徴量重要度分析
        feature_importance = self._analyze_feature_importance(individual.tree)
        evaluation_results['feature_importance'] = feature_importance

        logger.info("✅ モデル評価完了")
        logger.info(f"📈 検証R²スコア: {val_r2:.4f}")
        logger.info(f"🌳 木構造: {node_count}ノード, 深度{tree_depth}")

        return evaluation_results

    def generate_explanations(self, individual: GeneticIndividual) -> Dict[str, Any]:
        """説明生成"""

        logger.info("💡 説明生成開始...")

        explanations = {}

        # サンプルケースでの説明生成
        sample_indices = np.random.choice(
            len(self.validation_data),
            size=min(5, len(self.validation_data)),
            replace=False
        )

        sample_explanations = []

        for idx in sample_indices:
            feature_vector = self.validation_data.iloc[idx].values.tolist()
            actual_value = self.validation_targets[idx]

            # 説明付き予測
            prediction, explanation = individual.tree.predict_with_explanation(
                feature_vector, self.feature_names
            )

            # 包括的説明生成
            comprehensive_explanation = self.explanation_engine.generate_comprehensive_explanation(
                prediction=prediction,
                explanation=explanation,
                feature_vector=feature_vector,
                feature_names=self.feature_names,
                decision_tree=individual.tree,
                context={'actual_value': actual_value, 'sample_index': idx}
            )

            sample_explanations.append({
                'sample_index': idx,
                'prediction': prediction,
                'actual_value': actual_value,
                'error': abs(prediction - actual_value),
                'explanation': comprehensive_explanation
            })

        explanations['sample_explanations'] = sample_explanations

        # 全体的な解釈可能性分析
        explanations['global_interpretability'] = self._analyze_global_interpretability(
            individual.tree)

        logger.info(f"✅ 説明生成完了: {len(sample_explanations)}件のサンプル説明")

        return explanations

    def save_results(self,
                     individual: GeneticIndividual,
                     evaluation_results: Dict[str, Any],
                     explanations: Dict[str, Any]):
        """結果保存"""

        logger.info("💾 結果保存開始...")

        # モデル保存
        if self.config.output_settings['save_model']:
            model_id = self.persistence.save_model(
                genetic_individual=individual,
                optimizer=self.optimizer,
                feature_names=self.feature_names,
                training_sample_count=len(self.training_data),
                validation_sample_count=len(self.validation_data),
                version="1.0",
                tags=['genetic_algorithm',
                      'fuzzy_decision_tree', 'multi_objective'],
                description="遺伝的アルゴリズムで最適化されたファジィ決定木"
            )
            logger.info(f"📁 モデル保存完了: {model_id}")

        # 追跡データ保存
        if self.tracker and self.config.tracking_settings['enable_tracking']:
            export_path = self.output_dir / "optimization_history.json"
            self.tracker.export_tracking_data(str(export_path))
            logger.info(f"📊 追跡データ保存: {export_path}")

        # 評価結果保存
        results_path = self.output_dir / "evaluation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2,
                      ensure_ascii=False, default=str)

        # 説明保存
        if self.config.output_settings['save_explanations']:
            explanations_path = self.output_dir / "model_explanations.json"
            with open(explanations_path, 'w', encoding='utf-8') as f:
                json.dump(explanations, f, indent=2,
                          ensure_ascii=False, default=str)

        # 可視化保存
        if self.config.output_settings['save_plots']:
            self._create_and_save_visualizations(
                individual, evaluation_results)

        logger.info("✅ 結果保存完了")

    def _create_and_save_visualizations(self,
                                        individual: GeneticIndividual,
                                        evaluation_results: Dict[str, Any]):
        """可視化作成・保存"""

        plots_dir = self.output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # 1. 適応度進化プロット
        if self.tracker:
            fitness_plot = self.tracker.create_fitness_evolution_plot()
            if fitness_plot.get('type') == 'plotly':
                import plotly.io as pio
                pio.write_html(
                    fitness_plot['data'],
                    file=str(plots_dir / "fitness_evolution.html")
                )

        # 2. 多目的最適化分析
        if self.tracker:
            multi_obj_plot = self.tracker.create_multi_objective_analysis()
            if multi_obj_plot.get('type') == 'plotly':
                import plotly.io as pio
                pio.write_html(
                    multi_obj_plot['data'],
                    file=str(plots_dir / "multi_objective_analysis.html")
                )

        # 3. 特徴量重要度プロット
        if 'feature_importance' in evaluation_results:
            self._create_feature_importance_plot(
                evaluation_results['feature_importance'],
                plots_dir / "feature_importance.png"
            )

        # 4. 予測 vs 実際値プロット
        self._create_prediction_plot(
            individual, plots_dir / "prediction_vs_actual.png")

        logger.info(f"📈 可視化保存完了: {plots_dir}")

    def _create_feature_importance_plot(self, importance_data: Dict, save_path: Path):
        """特徴量重要度プロット作成"""

        plt.figure(figsize=(10, 6))

        features = list(importance_data.keys())
        importances = list(importance_data.values())

        # 重要度順でソート
        sorted_data = sorted(zip(features, importances),
                             key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_data)

        plt.barh(features, importances)
        plt.xlabel('重要度')
        plt.title('特徴量重要度')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _create_prediction_plot(self, individual: GeneticIndividual, save_path: Path):
        """予測プロット作成"""

        # 検証データで予測
        predictions = []
        for i in range(len(self.validation_data)):
            feature_vector = self.validation_data.iloc[i].values.tolist()
            prediction = individual.tree.predict(
                feature_vector, self.feature_names)
            predictions.append(prediction)

        plt.figure(figsize=(8, 8))
        plt.scatter(self.validation_targets, predictions, alpha=0.6)

        # 対角線（完璧な予測）
        min_val = min(min(self.validation_targets), min(predictions))
        max_val = max(max(self.validation_targets), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        plt.xlabel('実際値')
        plt.ylabel('予測値')
        plt.title('予測 vs 実際値')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def run_complete_pipeline(self) -> Dict[str, Any]:
        """完全パイプライン実行"""

        logger.info("🚀 遺伝的ファジィ決定木学習パイプライン開始")

        # 1. セットアップ
        self.setup_components()

        # 2. データ準備
        X, y, feature_names = self.load_research_lab_data()
        self.prepare_data(X, y, feature_names)

        # 3. 最適化実行
        best_individual = self.run_optimization()

        # 4. 評価
        evaluation_results = self.evaluate_model(best_individual)

        # 5. 説明生成
        explanations = self.generate_explanations(best_individual)

        # 6. 結果保存
        self.save_results(best_individual, evaluation_results, explanations)

        # 最終サマリー
        final_summary = {
            'best_fitness': best_individual.fitness_components.overall,
            'fitness_components': best_individual.fitness_components.to_dict(),
            'performance_metrics': evaluation_results['performance_metrics'],
            'structure_analysis': evaluation_results['structure_analysis'],
            'feature_count': len(self.feature_names),
            'training_samples': len(self.training_data),
            'validation_samples': len(self.validation_data),
            'output_directory': str(self.output_dir)
        }

        logger.info("🎉 パイプライン実行完了!")
        logger.info(f"📊 最終適応度: {final_summary['best_fitness']:.4f}")
        logger.info(f"📁 出力ディレクトリ: {final_summary['output_directory']}")

        return final_summary

    def _count_nodes(self, node) -> int:
        """ノード数カウント"""
        if node is None:
            return 0
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count

    def _calculate_depth(self, node) -> int:
        """木の深度計算"""
        if node is None or node.is_leaf:
            return 1
        max_child_depth = 0
        for child in node.children.values():
            child_depth = self._calculate_depth(child)
            max_child_depth = max(max_child_depth, child_depth)
        return 1 + max_child_depth

    def _count_leaves(self, node) -> int:
        """リーフ数カウント"""
        if node is None:
            return 0
        if node.is_leaf:
            return 1
        leaf_count = 0
        for child in node.children.values():
            leaf_count += self._count_leaves(child)
        return leaf_count

    def _analyze_feature_importance(self, node) -> Dict[str, float]:
        """特徴量重要度分析"""
        feature_usage = {}

        def _count_usage(current_node):
            if current_node is None or current_node.is_leaf or not current_node.split_info:
                return

            feature_name = current_node.split_info.feature_name
            info_gain = current_node.split_info.information_gain

            feature_usage[feature_name] = feature_usage.get(
                feature_name, 0) + info_gain

            for child in current_node.children.values():
                _count_usage(child)

        _count_usage(node)

        # 正規化
        if feature_usage:
            max_importance = max(feature_usage.values())
            if max_importance > 0:
                feature_usage = {k: v / max_importance for k,
                                 v in feature_usage.items()}

        return feature_usage

    def _analyze_global_interpretability(self, node) -> Dict[str, Any]:
        """グローバル解釈可能性分析"""

        # 決定ルール抽出
        rules = []

        def _extract_rules(current_node, path=[]):
            if current_node is None:
                return

            if current_node.is_leaf:
                rules.append({
                    'path': path.copy(),
                    'prediction': current_node.leaf_value,
                    'complexity': len(path)
                })
                return

            if current_node.split_info:
                feature_name = current_node.split_info.feature_name
                for term, child in current_node.children.items():
                    new_path = path + [f"{feature_name} is {term}"]
                    _extract_rules(child, new_path)

        _extract_rules(node)

        # 統計
        rule_complexities = [rule['complexity'] for rule in rules]

        return {
            'total_rules': len(rules),
            'avg_rule_complexity': np.mean(rule_complexities) if rule_complexities else 0,
            'max_rule_complexity': max(rule_complexities) if rule_complexities else 0,
            'min_rule_complexity': min(rule_complexities) if rule_complexities else 0,
            'sample_rules': rules[:5]  # 最初の5つのルール
        }


def main():
    """メイン実行関数"""

    parser = argparse.ArgumentParser(description='遺伝的ファジィ決定木学習')

    parser.add_argument('--config', type=str,
                        help='設定ファイルパス (JSON)')
    parser.add_argument('--generations', type=int, default=30,
                        help='世代数 (デフォルト: 30)')
    parser.add_argument('--population_size', type=int, default=50,
                        help='集団サイズ (デフォルト: 50)')
    parser.add_argument('--crossover_rate', type=float, default=0.8,
                        help='交叉率 (デフォルト: 0.8)')
    parser.add_argument('--mutation_rate', type=float, default=0.2,
                        help='突然変異率 (デフォルト: 0.2)')
    parser.add_argument('--output_dir', type=str, default='genetic_training_output',
                        help='出力ディレクトリ (デフォルト: genetic_training_output)')
    parser.add_argument('--no_tracking', action='store_true',
                        help='追跡を無効化')
    parser.add_argument('--no_plots', action='store_true',
                        help='プロット生成を無効化')

    args = parser.parse_args()

    # 設定読み込み
    if args.config:
        config = TrainingConfiguration.load_from_file(args.config)
        logger.info(f"📋 設定ファイル読み込み: {args.config}")
    else:
        config = TrainingConfiguration()

        # コマンドライン引数で上書き
        config.genetic_parameters.generations = args.generations
        config.genetic_parameters.population_size = args.population_size
        config.genetic_parameters.crossover_rate = args.crossover_rate
        config.genetic_parameters.mutation_rate = args.mutation_rate
        config.output_settings['output_directory'] = args.output_dir
        config.tracking_settings['enable_tracking'] = not args.no_tracking
        config.output_settings['save_plots'] = not args.no_plots

    # 設定を出力ディレクトリに保存
    output_dir = Path(config.output_settings['output_directory'])
    output_dir.mkdir(parents=True, exist_ok=True)
    config.save_to_file(str(output_dir / "training_config.json"))

    try:
        # パイプライン実行
        pipeline = GeneticTrainingPipeline(config)
        summary = pipeline.run_complete_pipeline()

        # 成功メッセージ
        print("\n" + "="*60)
        print("🎉 遺伝的ファジィ決定木学習完了!")
        print("="*60)
        print(f"🏆 最良適応度: {summary['best_fitness']:.4f}")
        print(
            f"📊 検証R²スコア: {summary['performance_metrics']['validation']['r2']:.4f}")
        print(f"🌳 木構造: {summary['structure_analysis']['node_count']}ノード")
        print(f"📁 結果: {summary['output_directory']}")
        print("="*60)

        return 0

    except KeyboardInterrupt:
        logger.info("❌ ユーザーによって中断されました")
        return 1
    except Exception as e:
        logger.error(f"❌ 実行エラー: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
