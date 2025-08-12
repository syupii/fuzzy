# backend/model_persistence.py
import os
import pickle
import json
import gzip
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd

from advanced_nodes import AdvancedFuzzyDecisionNode, FitnessComponents
from genetic_fuzzy_tree import Individual, GeneticParameters


@dataclass
class ModelMetadata:
    """モデルメタデータ"""
    model_id: str
    creation_date: datetime
    model_type: str  # 'genetic_fuzzy_tree', 'simple_fuzzy'
    version: str

    # 訓練情報
    training_samples: int
    test_samples: int
    feature_names: List[str]
    target_column: str

    # 性能指標
    best_fitness: float
    fitness_components: Dict[str, float]

    # 最適化情報
    optimization_config: Dict[str, Any]
    convergence_generation: Optional[int]
    total_generations: int

    # 技術詳細
    model_complexity: int
    tree_depth: int
    feature_importance: Dict[str, float]

    # ファイル情報
    file_size_bytes: int
    checksum: str

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式への変換"""
        data = asdict(self)
        data['creation_date'] = self.creation_date.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """辞書からの復元"""
        data = data.copy()
        data['creation_date'] = datetime.fromisoformat(data['creation_date'])
        return cls(**data)


class AdvancedModelPersistence:
    """高度なモデル永続化システム"""

    def __init__(self, models_dir: str = "models", compression: bool = True):
        self.models_dir = models_dir
        self.compression = compression
        self.metadata_file = os.path.join(models_dir, "models_registry.json")

        # ディレクトリ作成
        os.makedirs(models_dir, exist_ok=True)

        # メタデータ読み込み
        self.models_registry = self._load_registry()

        print(f"📁 ModelPersistence initialized: {models_dir}")

    def save_genetic_optimization_result(self, result: Dict[str, Any],
                                         model_id: str = None,
                                         description: str = "") -> str:
        """遺伝的最適化結果保存"""

        if model_id is None:
            model_id = f"genetic_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"💾 Saving genetic optimization result: {model_id}")

        try:
            # 最良個体抽出
            best_individual = result.get('best_individual')
            if not best_individual:
                raise ValueError("No best individual found in result")

            # モデルファイル保存
            model_filepath = self._save_model_file(best_individual, model_id)

            # 最適化結果保存
            result_filepath = self._save_optimization_data(result, model_id)

            # メタデータ作成
            metadata = self._create_metadata(
                model_id, result, best_individual, model_filepath
            )

            # レジストリ更新
            self.models_registry[model_id] = {
                **metadata.to_dict(),
                'description': description,
                'model_filepath': model_filepath,
                'result_filepath': result_filepath
            }

            self._save_registry()

            print(f"✅ Model saved successfully: {model_id}")
            print(f"📄 Model file: {model_filepath}")
            print(f"📊 Result file: {result_filepath}")

            return model_id

        except Exception as e:
            print(f"❌ Failed to save model {model_id}: {e}")
            raise

    def load_genetic_model(self, model_id: str) -> Tuple[Individual, Dict[str, Any]]:
        """遺伝的モデル読み込み"""

        if model_id not in self.models_registry:
            raise ValueError(f"Model {model_id} not found in registry")

        model_info = self.models_registry[model_id]

        print(f"📂 Loading genetic model: {model_id}")

        try:
            # モデルファイル読み込み
            model_filepath = model_info['model_filepath']
            individual = self._load_model_file(model_filepath)

            # 最適化結果読み込み
            result_filepath = model_info['result_filepath']
            optimization_result = self._load_optimization_data(result_filepath)

            print(f"✅ Model loaded successfully: {model_id}")
            print(f"🎯 Fitness: {model_info['best_fitness']:.4f}")
            print(f"🧮 Complexity: {model_info['model_complexity']}")

            return individual, optimization_result

        except Exception as e:
            print(f"❌ Failed to load model {model_id}: {e}")
            raise

    def save_prediction_cache(self, model_id: str, predictions: Dict[str, Any]):
        """予測結果キャッシュ保存"""

        cache_dir = os.path.join(self.models_dir, "prediction_cache")
        os.makedirs(cache_dir, exist_ok=True)

        cache_filepath = os.path.join(
            cache_dir, f"{model_id}_predictions.json")

        with open(cache_filepath, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2,
                      ensure_ascii=False, default=str)

        print(f"💾 Prediction cache saved: {cache_filepath}")

    def load_prediction_cache(self, model_id: str) -> Optional[Dict[str, Any]]:
        """予測結果キャッシュ読み込み"""

        cache_filepath = os.path.join(
            self.models_dir, "prediction_cache", f"{model_id}_predictions.json")

        if not os.path.exists(cache_filepath):
            return None

        try:
            with open(cache_filepath, 'r', encoding='utf-8') as f:
                predictions = json.load(f)

            print(f"📂 Prediction cache loaded: {cache_filepath}")
            return predictions

        except Exception as e:
            print(f"⚠️ Failed to load prediction cache: {e}")
            return None

    def list_models(self, model_type: str = None) -> List[Dict[str, Any]]:
        """モデル一覧取得"""

        models = []

        for model_id, model_info in self.models_registry.items():
            if model_type is None or model_info.get('model_type') == model_type:
                models.append({
                    'model_id': model_id,
                    **model_info
                })

        # 作成日時でソート
        models.sort(key=lambda x: x.get('creation_date', ''), reverse=True)

        return models

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """モデル情報取得"""

        return self.models_registry.get(model_id)

    def delete_model(self, model_id: str) -> bool:
        """モデル削除"""

        if model_id not in self.models_registry:
            print(f"⚠️ Model {model_id} not found")
            return False

        try:
            model_info = self.models_registry[model_id]

            # ファイル削除
            files_to_delete = [
                model_info.get('model_filepath'),
                model_info.get('result_filepath')
            ]

            for filepath in files_to_delete:
                if filepath and os.path.exists(filepath):
                    os.remove(filepath)
                    print(f"🗑️ Deleted: {filepath}")

            # レジストリから削除
            del self.models_registry[model_id]
            self._save_registry()

            print(f"✅ Model {model_id} deleted successfully")
            return True

        except Exception as e:
            print(f"❌ Failed to delete model {model_id}: {e}")
            return False

    def export_model(self, model_id: str, export_path: str,
                     include_optimization_data: bool = True) -> bool:
        """モデルエクスポート"""

        if model_id not in self.models_registry:
            print(f"⚠️ Model {model_id} not found")
            return False

        try:
            model_info = self.models_registry[model_id]

            # エクスポートデータ準備
            export_data = {
                'metadata': model_info,
                'model_data': None,
                'optimization_data': None
            }

            # モデルデータ読み込み
            model_filepath = model_info['model_filepath']
            with open(model_filepath, 'rb') as f:
                if self.compression and model_filepath.endswith('.gz'):
                    import gzip
                    export_data['model_data'] = gzip.decompress(f.read())
                else:
                    export_data['model_data'] = f.read()

            # 最適化データ読み込み
            if include_optimization_data:
                result_filepath = model_info['result_filepath']
                with open(result_filepath, 'rb') as f:
                    if self.compression and result_filepath.endswith('.gz'):
                        import gzip
                        export_data['optimization_data'] = gzip.decompress(
                            f.read())
                    else:
                        export_data['optimization_data'] = f.read()

            # エクスポート
            os.makedirs(os.path.dirname(export_path), exist_ok=True)

            if self.compression:
                with gzip.open(export_path, 'wb') as f:
                    pickle.dump(export_data, f)
            else:
                with open(export_path, 'wb') as f:
                    pickle.dump(export_data, f)

            print(f"📦 Model exported: {export_path}")
            return True

        except Exception as e:
            print(f"❌ Failed to export model {model_id}: {e}")
            return False

    def import_model(self, import_path: str, new_model_id: str = None) -> Optional[str]:
        """モデルインポート"""

        if not os.path.exists(import_path):
            print(f"⚠️ Import file not found: {import_path}")
            return None

        try:
            # インポートデータ読み込み
            if import_path.endswith('.gz'):
                with gzip.open(import_path, 'rb') as f:
                    import_data = pickle.load(f)
            else:
                with open(import_path, 'rb') as f:
                    import_data = pickle.load(f)

            # モデルID決定
            if new_model_id is None:
                original_id = import_data['metadata']['model_id']
                new_model_id = f"{original_id}_imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # モデルファイル保存
            model_data = import_data['model_data']
            model_filepath = os.path.join(
                self.models_dir, f"{new_model_id}_model.pkl")

            if self.compression:
                model_filepath += '.gz'
                with gzip.open(model_filepath, 'wb') as f:
                    f.write(model_data)
            else:
                with open(model_filepath, 'wb') as f:
                    f.write(model_data)

            # 最適化データ保存
            result_filepath = None
            if import_data.get('optimization_data'):
                optimization_data = import_data['optimization_data']
                result_filepath = os.path.join(
                    self.models_dir, f"{new_model_id}_result.pkl")

                if self.compression:
                    result_filepath += '.gz'
                    with gzip.open(result_filepath, 'wb') as f:
                        f.write(optimization_data)
                else:
                    with open(result_filepath, 'wb') as f:
                        f.write(optimization_data)

            # メタデータ更新
            metadata = import_data['metadata'].copy()
            metadata['model_id'] = new_model_id
            metadata['model_filepath'] = model_filepath
            metadata['result_filepath'] = result_filepath

            # レジストリ更新
            self.models_registry[new_model_id] = metadata
            self._save_registry()

            print(f"📦 Model imported successfully: {new_model_id}")
            return new_model_id

        except Exception as e:
            print(f"❌ Failed to import model: {e}")
            return None

    def create_model_backup(self, backup_dir: str = None) -> str:
        """モデル全体バックアップ"""

        if backup_dir is None:
            backup_dir = os.path.join(self.models_dir, "backups")

        os.makedirs(backup_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filepath = os.path.join(
            backup_dir, f"models_backup_{timestamp}.tar.gz")

        try:
            import tarfile

            with tarfile.open(backup_filepath, 'w:gz') as tar:
                tar.add(self.models_dir, arcname='models')

            print(f"💾 Models backup created: {backup_filepath}")
            return backup_filepath

        except Exception as e:
            print(f"❌ Failed to create backup: {e}")
            raise

    def restore_from_backup(self, backup_filepath: str, restore_dir: str = None) -> bool:
        """バックアップから復元"""

        if restore_dir is None:
            restore_dir = self.models_dir + "_restored"

        try:
            import tarfile

            with tarfile.open(backup_filepath, 'r:gz') as tar:
                tar.extractall(path=restore_dir)

            print(f"📂 Models restored to: {restore_dir}")
            return True

        except Exception as e:
            print(f"❌ Failed to restore from backup: {e}")
            return False

    def _save_model_file(self, individual: Individual, model_id: str) -> str:
        """モデルファイル保存"""

        filepath = os.path.join(self.models_dir, f"{model_id}_model.pkl")

        if self.compression:
            filepath += '.gz'
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(individual, f)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(individual, f)

        return filepath

    def _load_model_file(self, filepath: str) -> Individual:
        """モデルファイル読み込み"""

        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rb') as f:
                individual = pickle.load(f)
        else:
            with open(filepath, 'rb') as f:
                individual = pickle.load(f)

        return individual

    def _save_optimization_data(self, result: Dict[str, Any], model_id: str) -> str:
        """最適化データ保存"""

        filepath = os.path.join(self.models_dir, f"{model_id}_result.pkl")

        if self.compression:
            filepath += '.gz'
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(result, f)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(result, f)

        return filepath

    def _load_optimization_data(self, filepath: str) -> Dict[str, Any]:
        """最適化データ読み込み"""

        if filepath.endswith('.gz'):
            with gzip.open(filepath, 'rb') as f:
                result = pickle.load(f)
        else:
            with open(filepath, 'rb') as f:
                result = pickle.load(f)

        return result

    def _create_metadata(self, model_id: str, result: Dict[str, Any],
                         best_individual: Individual, filepath: str) -> ModelMetadata:
        """メタデータ作成"""

        # ファイルサイズとチェックサム
        file_size = os.path.getsize(filepath)

        with open(filepath, 'rb') as f:
            checksum = hashlib.md5(f.read()).hexdigest()

        # 適応度コンポーネント
        fitness_components = {}
        if best_individual.fitness_components:
            fitness_components = best_individual.fitness_components.__dict__

        # 収束分析
        convergence_analysis = result.get('convergence_analysis', {})

        metadata = ModelMetadata(
            model_id=model_id,
            creation_date=datetime.now(),
            model_type='genetic_fuzzy_tree',
            version='1.0',

            # 訓練情報（デフォルト値）
            training_samples=result.get(
                'optimization_config', {}).get('training_samples', 0),
            test_samples=result.get(
                'optimization_config', {}).get('test_samples', 0),
            feature_names=result.get('feature_names', []),
            target_column=result.get('optimization_config', {}).get(
                'target_column', 'compatibility'),

            # 性能指標
            best_fitness=result.get('best_fitness', 0.0),
            fitness_components=fitness_components,

            # 最適化情報
            optimization_config=result.get('optimization_config', {}),
            convergence_generation=convergence_analysis.get(
                'convergence_detected', None),
            total_generations=convergence_analysis.get('total_generations', 0),

            # 技術詳細
            model_complexity=result.get(
                'best_tree_analysis', {}).get('complexity', 0),
            tree_depth=result.get('best_tree_analysis', {}).get('depth', 0),
            feature_importance=result.get(
                'best_tree_analysis', {}).get('feature_importance', {}),

            # ファイル情報
            file_size_bytes=file_size,
            checksum=checksum
        )

        return metadata

    def _load_registry(self) -> Dict[str, Any]:
        """レジストリ読み込み"""

        if not os.path.exists(self.metadata_file):
            return {}

        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load registry: {e}")
            return {}

    def _save_registry(self):
        """レジストリ保存"""

        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.models_registry, f, indent=2,
                          ensure_ascii=False, default=str)
        except Exception as e:
            print(f"⚠️ Failed to save registry: {e}")

    def get_storage_statistics(self) -> Dict[str, Any]:
        """ストレージ統計"""

        stats = {
            'total_models': len(self.models_registry),
            'total_size_bytes': 0,
            'models_by_type': {},
            'average_fitness': 0.0,
            'latest_model': None,
            'oldest_model': None
        }

        if not self.models_registry:
            return stats

        # 統計計算
        fitness_values = []
        creation_dates = []

        for model_id, model_info in self.models_registry.items():
            # サイズ
            stats['total_size_bytes'] += model_info.get('file_size_bytes', 0)

            # タイプ別カウント
            model_type = model_info.get('model_type', 'unknown')
            stats['models_by_type'][model_type] = stats['models_by_type'].get(
                model_type, 0) + 1

            # 適応度
            fitness = model_info.get('best_fitness', 0.0)
            fitness_values.append(fitness)

            # 作成日時
            creation_date = model_info.get('creation_date')
            if creation_date:
                creation_dates.append(creation_date)

        # 平均適応度
        if fitness_values:
            stats['average_fitness'] = np.mean(fitness_values)

        # 最新・最古
        if creation_dates:
            creation_dates.sort()
            stats['oldest_model'] = creation_dates[0]
            stats['latest_model'] = creation_dates[-1]

        return stats

    def cleanup_old_models(self, keep_latest: int = 10, min_fitness: float = 0.5) -> int:
        """古いモデルの清理"""

        if len(self.models_registry) <= keep_latest:
            return 0

        # モデルを適応度と作成日時でソート
        models_list = []
        for model_id, model_info in self.models_registry.items():
            models_list.append({
                'model_id': model_id,
                'fitness': model_info.get('best_fitness', 0.0),
                'creation_date': model_info.get('creation_date', ''),
                **model_info
            })

        # ソート（適応度降順、作成日時降順）
        models_list.sort(key=lambda x: (
            x['fitness'], x['creation_date']), reverse=True)

        # 削除対象決定
        models_to_delete = []

        # 上位keep_latest以外で、最低適応度を下回るモデル
        for i, model in enumerate(models_list):
            if i >= keep_latest or model['fitness'] < min_fitness:
                models_to_delete.append(model['model_id'])

        # 削除実行
        deleted_count = 0
        for model_id in models_to_delete:
            if self.delete_model(model_id):
                deleted_count += 1

        print(f"🧹 Cleaned up {deleted_count} old models")
        return deleted_count


class ModelVersionManager:
    """モデルバージョン管理"""

    def __init__(self, persistence: AdvancedModelPersistence):
        self.persistence = persistence

    def create_model_version(self, base_model_id: str,
                             optimization_result: Dict[str, Any],
                             version_description: str = "") -> str:
        """モデルバージョン作成"""

        if base_model_id not in self.persistence.models_registry:
            raise ValueError(f"Base model {base_model_id} not found")

        # バージョン番号生成
        version_number = self._get_next_version_number(base_model_id)
        new_model_id = f"{base_model_id}_v{version_number}"

        # 新バージョン保存
        model_id = self.persistence.save_genetic_optimization_result(
            optimization_result,
            new_model_id,
            f"Version {version_number} of {base_model_id}. {version_description}"
        )

        print(f"📈 Created model version: {model_id}")
        return model_id

    def get_model_versions(self, base_model_id: str) -> List[Dict[str, Any]]:
        """モデルバージョン一覧取得"""

        versions = []

        for model_id, model_info in self.persistence.models_registry.items():
            if model_id.startswith(f"{base_model_id}_v"):
                versions.append({
                    'model_id': model_id,
                    'version_number': self._extract_version_number(model_id),
                    **model_info
                })

        # バージョン番号でソート
        versions.sort(key=lambda x: x['version_number'])

        return versions

    def compare_model_versions(self, model_id1: str, model_id2: str) -> Dict[str, Any]:
        """モデルバージョン比較"""

        model1_info = self.persistence.get_model_info(model_id1)
        model2_info = self.persistence.get_model_info(model_id2)

        if not model1_info or not model2_info:
            raise ValueError("One or both models not found")

        comparison = {
            'model1': {'id': model_id1, **model1_info},
            'model2': {'id': model_id2, **model2_info},
            'comparison': {
                'fitness_difference': model2_info['best_fitness'] - model1_info['best_fitness'],
                'complexity_difference': model2_info['model_complexity'] - model1_info['model_complexity'],
                'depth_difference': model2_info['tree_depth'] - model1_info['tree_depth'],
                'better_model': model_id2 if model2_info['best_fitness'] > model1_info['best_fitness'] else model_id1
            }
        }

        return comparison

    def _get_next_version_number(self, base_model_id: str) -> int:
        """次のバージョン番号取得"""

        max_version = 0

        for model_id in self.persistence.models_registry.keys():
            if model_id.startswith(f"{base_model_id}_v"):
                version_number = self._extract_version_number(model_id)
                max_version = max(max_version, version_number)

        return max_version + 1

    def _extract_version_number(self, model_id: str) -> int:
        """バージョン番号抽出"""

        import re
        match = re.search(r'_v(\d+)$', model_id)
        return int(match.group(1)) if match else 0


class ModelComparisonTool:
    """モデル比較ツール"""

    @staticmethod
    def compare_models(models: List[Dict[str, Any]],
                       metrics: List[str] = None) -> pd.DataFrame:
        """モデル比較表作成"""

        if metrics is None:
            metrics = ['best_fitness', 'model_complexity',
                       'tree_depth', 'total_generations']

        comparison_data = []

        for model_info in models:
            row = {'model_id': model_info['model_id']}

            for metric in metrics:
                if metric in model_info:
                    row[metric] = model_info[metric]
                elif metric in model_info.get('fitness_components', {}):
                    row[metric] = model_info['fitness_components'][metric]
                else:
                    row[metric] = None

            comparison_data.append(row)

        df = pd.DataFrame(comparison_data)
        return df.set_index('model_id')

    @staticmethod
    def generate_performance_ranking(models: List[Dict[str, Any]],
                                     ranking_metric: str = 'best_fitness') -> List[Dict[str, Any]]:
        """性能ランキング生成"""

        # ランキング作成
        ranked_models = sorted(
            models,
            key=lambda x: x.get(ranking_metric, 0.0),
            reverse=True
        )

        # ランキング情報追加
        for i, model in enumerate(ranked_models, 1):
            model['rank'] = i
            model['ranking_metric'] = ranking_metric
            model['ranking_value'] = model.get(ranking_metric, 0.0)

        return ranked_models

    @staticmethod
    def find_best_model_by_criteria(models: List[Dict[str, Any]],
                                    criteria: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """複合基準による最良モデル検索"""

        best_model = None
        best_score = -float('inf')

        for model in models:
            score = 0.0

            for criterion, weight in criteria.items():
                value = model.get(criterion, 0.0)
                if isinstance(value, dict):  # fitness_components内の場合
                    value = model.get('fitness_components',
                                      {}).get(criterion, 0.0)

                score += weight * value

            if score > best_score:
                best_score = score
                best_model = model
                best_model['composite_score'] = score

        return best_model
