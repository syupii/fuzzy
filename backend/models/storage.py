# models/storage.py - モデル永続化

import os
import pickle
import json
import joblib
import gzip
from typing import Dict, List, Any, Optional, Union, Type
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
import hashlib
import shutil

from models.schemas import (
    StudentProfile, Laboratory, EvaluationResponse,
    OptimizationResult
)
from core.genetic.individual import Individual, WeightVector
from core.genetic.population import Population
from core.decision_tree.tree import FuzzyDecisionTree
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class ModelMetadata:
    """モデルメタデータ"""
    model_id: str
    model_type: str
    version: str
    created_at: datetime
    updated_at: datetime
    file_path: str
    file_size: int
    checksum: str
    description: str
    tags: List[str]
    performance_metrics: Dict[str, float]
    training_data_info: Dict[str, Any]

@dataclass
class StorageConfig:
    """ストレージ設定"""
    base_directory: str = "./data/models"
    compression_enabled: bool = True
    versioning_enabled: bool = True
    max_versions: int = 10
    backup_enabled: bool = True
    encryption_enabled: bool = False
    
    def __post_init__(self):
        # ディレクトリの作成
        Path(self.base_directory).mkdir(parents=True, exist_ok=True)

class ModelStorage:
    """モデル永続化クラス"""
    
    def __init__(self, config: StorageConfig = None):
        self.config = config or StorageConfig()
        self.models_dir = Path(self.config.base_directory)
        self.metadata_dir = self.models_dir / "metadata"
        self.backups_dir = self.models_dir / "backups"
        
        # 必要なディレクトリを作成
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        if self.config.backup_enabled:
            self.backups_dir.mkdir(parents=True, exist_ok=True)
        
        # メタデータキャッシュ
        self.metadata_cache: Dict[str, ModelMetadata] = {}
        
        # メタデータの読み込み
        self._load_metadata_cache()
    
    def save_model(self, model: Any, model_id: str, model_type: str,
                   description: str = "", tags: List[str] = None,
                   performance_metrics: Dict[str, float] = None,
                   training_data_info: Dict[str, Any] = None) -> str:
        """モデルの保存"""
        
        try:
            tags = tags or []
            performance_metrics = performance_metrics or {}
            training_data_info = training_data_info or {}
            
            # バージョン管理
            if self.config.versioning_enabled:
                version = self._get_next_version(model_id)
                versioned_model_id = f"{model_id}_v{version}"
            else:
                version = "1.0.0"
                versioned_model_id = model_id
            
            # ファイルパスの決定
            file_extension = ".pkl.gz" if self.config.compression_enabled else ".pkl"
            file_path = self.models_dir / f"{versioned_model_id}{file_extension}"
            
            # モデルの保存
            self._save_model_file(model, file_path)
            
            # メタデータの作成
            file_size = file_path.stat().st_size
            checksum = self._calculate_checksum(file_path)
            
            metadata = ModelMetadata(
                model_id=model_id,
                model_type=model_type,
                version=version,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                file_path=str(file_path),
                file_size=file_size,
                checksum=checksum,
                description=description,
                tags=tags,
                performance_metrics=performance_metrics,
                training_data_info=training_data_info
            )
            
            # メタデータの保存
            self._save_metadata(versioned_model_id, metadata)
            
            # 古いバージョンのクリーンアップ
            if self.config.versioning_enabled:
                self._cleanup_old_versions(model_id)
            
            logger.info(f"モデル保存完了: {versioned_model_id}")
            return versioned_model_id
            
        except Exception as e:
            logger.error(f"モデル保存エラー: {e}")
            raise
    
    def load_model(self, model_id: str, version: Optional[str] = None) -> Any:
        """モデルの読み込み"""
        
        try:
            # 最新バージョンの取得
            if version is None:
                version = self._get_latest_version(model_id)
            
            versioned_model_id = f"{model_id}_v{version}" if version != "1.0.0" else model_id
            
            # メタデータの取得
            metadata = self._load_metadata(versioned_model_id)
            if not metadata:
                raise FileNotFoundError(f"モデルが見つかりません: {versioned_model_id}")
            
            # ファイルの整合性チェック
            file_path = Path(metadata.file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"モデルファイルが見つかりません: {file_path}")
            
            current_checksum = self._calculate_checksum(file_path)
            if current_checksum != metadata.checksum:
                logger.warning(f"チェックサム不一致: {versioned_model_id}")
            
            # モデルの読み込み
            model = self._load_model_file(file_path)
            
            logger.info(f"モデル読み込み完了: {versioned_model_id}")
            return model
            
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            raise
    
    def delete_model(self, model_id: str, version: Optional[str] = None) -> bool:
        """モデルの削除"""
        
        try:
            if version is None:
                # 全バージョンを削除
                versions = self._get_all_versions(model_id)
                for v in versions:
                    self._delete_single_version(model_id, v)
            else:
                # 指定バージョンのみ削除
                self._delete_single_version(model_id, version)
            
            logger.info(f"モデル削除完了: {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"モデル削除エラー: {e}")
            return False
    
    def list_models(self, model_type: Optional[str] = None, 
                   tags: Optional[List[str]] = None) -> List[ModelMetadata]:
        """モデル一覧の取得"""
        
        models = []
        
        for metadata in self.metadata_cache.values():
            # 型フィルタ
            if model_type and metadata.model_type != model_type:
                continue
            
            # タグフィルタ
            if tags and not any(tag in metadata.tags for tag in tags):
                continue
            
            models.append(metadata)
        
        # 作成日時でソート
        models.sort(key=lambda x: x.created_at, reverse=True)
        
        return models
    
    def get_model_info(self, model_id: str, version: Optional[str] = None) -> Optional[ModelMetadata]:
        """モデル情報の取得"""
        
        if version is None:
            version = self._get_latest_version(model_id)
        
        versioned_model_id = f"{model_id}_v{version}" if version != "1.0.0" else model_id
        
        return self.metadata_cache.get(versioned_model_id)
    
    def backup_model(self, model_id: str, version: Optional[str] = None) -> str:
        """モデルのバックアップ"""
        
        if not self.config.backup_enabled:
            raise ValueError("バックアップが無効です")
        
        try:
            if version is None:
                version = self._get_latest_version(model_id)
            
            versioned_model_id = f"{model_id}_v{version}" if version != "1.0.0" else model_id
            
            # メタデータとモデルファイルのパス
            metadata = self._load_metadata(versioned_model_id)
            if not metadata:
                raise FileNotFoundError(f"モデルが見つかりません: {versioned_model_id}")
            
            # バックアップディレクトリ
            backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = self.backups_dir / f"{versioned_model_id}_{backup_timestamp}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # ファイルのコピー
            model_file = Path(metadata.file_path)
            metadata_file = self.metadata_dir / f"{versioned_model_id}.json"
            
            shutil.copy2(model_file, backup_dir / model_file.name)
            shutil.copy2(metadata_file, backup_dir / metadata_file.name)
            
            logger.info(f"バックアップ完了: {backup_dir}")
            return str(backup_dir)
            
        except Exception as e:
            logger.error(f"バックアップエラー: {e}")
            raise
    
    def restore_model(self, backup_path: str) -> str:
        """バックアップからの復元"""
        
        try:
            backup_dir = Path(backup_path)
            if not backup_dir.exists():
                raise FileNotFoundError(f"バックアップディレクトリが見つかりません: {backup_path}")
            
            # メタデータファイルの検索
            metadata_files = list(backup_dir.glob("*.json"))
            if not metadata_files:
                raise FileNotFoundError("メタデータファイルが見つかりません")
            
            metadata_file = metadata_files[0]
            
            # メタデータの読み込み
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            
            # モデルファイルの検索
            model_files = list(backup_dir.glob("*.pkl*"))
            if not model_files:
                raise FileNotFoundError("モデルファイルが見つかりません")
            
            model_file = model_files[0]
            
            # 復元先パスの決定
            versioned_model_id = metadata_file.stem
            restored_model_path = self.models_dir / model_file.name
            restored_metadata_path = self.metadata_dir / metadata_file.name
            
            # ファイルの復元
            shutil.copy2(model_file, restored_model_path)
            shutil.copy2(metadata_file, restored_metadata_path)
            
            # メタデータキャッシュの更新
            metadata = ModelMetadata(**metadata_dict)
            metadata.file_path = str(restored_model_path)
            self.metadata_cache[versioned_model_id] = metadata
            
            logger.info(f"復元完了: {versioned_model_id}")
            return versioned_model_id
            
        except Exception as e:
            logger.error(f"復元エラー: {e}")
            raise
    
    def _save_model_file(self, model: Any, file_path: Path):
        """モデルファイルの保存"""
        
        if self.config.compression_enabled:
            with gzip.open(file_path, 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(file_path, 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _load_model_file(self, file_path: Path) -> Any:
        """モデルファイルの読み込み"""
        
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
    
    def _save_metadata(self, model_id: str, metadata: ModelMetadata):
        """メタデータの保存"""
        
        metadata_file = self.metadata_dir / f"{model_id}.json"
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            # dataclassを辞書に変換
            metadata_dict = asdict(metadata)
            # datetimeを文字列に変換
            metadata_dict['created_at'] = metadata.created_at.isoformat()
            metadata_dict['updated_at'] = metadata.updated_at.isoformat()
            
            json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
        
        # キャッシュの更新
        self.metadata_cache[model_id] = metadata
    
    def _load_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """メタデータの読み込み"""
        
        # キャッシュから取得
        if model_id in self.metadata_cache:
            return self.metadata_cache[model_id]
        
        # ファイルから読み込み
        metadata_file = self.metadata_dir / f"{model_id}.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            
            # 日時文字列をdatetimeオブジェクトに変換
            metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
            metadata_dict['updated_at'] = datetime.fromisoformat(metadata_dict['updated_at'])
            
            metadata = ModelMetadata(**metadata_dict)
            
            # キャッシュに追加
            self.metadata_cache[model_id] = metadata
            
            return metadata
            
        except Exception as e:
            logger.error(f"メタデータ読み込みエラー: {e}")
            return None
    
    def _load_metadata_cache(self):
        """メタデータキャッシュの読み込み"""
        
        for metadata_file in self.metadata_dir.glob("*.json"):
            model_id = metadata_file.stem
            metadata = self._load_metadata(model_id)
            if metadata:
                self.metadata_cache[model_id] = metadata
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """ファイルのチェックサム計算"""
        
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def _get_next_version(self, model_id: str) -> str:
        """次のバージョン番号を取得"""
        
        versions = self._get_all_versions(model_id)
        
        if not versions:
            return "1.0.0"
        
        # 最新バージョンから次のバージョンを計算
        latest_version = max(versions, key=lambda v: [int(x) for x in v.split('.')])
        major, minor, patch = [int(x) for x in latest_version.split('.')]
        
        return f"{major}.{minor}.{patch + 1}"
    
    def _get_latest_version(self, model_id: str) -> str:
        """最新バージョンを取得"""
        
        versions = self._get_all_versions(model_id)
        
        if not versions:
            return "1.0.0"
        
        return max(versions, key=lambda v: [int(x) for x in v.split('.')])
    
    def _get_all_versions(self, model_id: str) -> List[str]:
        """全バージョンを取得"""
        
        versions = []
        
        for cached_id in self.metadata_cache.keys():
            if cached_id.startswith(f"{model_id}_v"):
                version = cached_id.split('_v')[1]
                versions.append(version)
            elif cached_id == model_id:
                versions.append("1.0.0")
        
        return versions
    
    def _delete_single_version(self, model_id: str, version: str):
        """単一バージョンの削除"""
        
        versioned_model_id = f"{model_id}_v{version}" if version != "1.0.0" else model_id
        
        # メタデータの取得
        metadata = self._load_metadata(versioned_model_id)
        if not metadata:
            return
        
        # ファイルの削除
        model_file = Path(metadata.file_path)
        metadata_file = self.metadata_dir / f"{versioned_model_id}.json"
        
        if model_file.exists():
            model_file.unlink()
        
        if metadata_file.exists():
            metadata_file.unlink()
        
        # キャッシュから削除
        if versioned_model_id in self.metadata_cache:
            del self.metadata_cache[versioned_model_id]
    
    def _cleanup_old_versions(self, model_id: str):
        """古いバージョンのクリーンアップ"""
        
        if not self.config.versioning_enabled:
            return
        
        versions = self._get_all_versions(model_id)
        
        if len(versions) > self.config.max_versions:
            # バージョンをソート
            sorted_versions = sorted(versions, key=lambda v: [int(x) for x in v.split('.')], reverse=True)
            
            # 古いバージョンを削除
            for old_version in sorted_versions[self.config.max_versions:]:
                self._delete_single_version(model_id, old_version)
                logger.info(f"古いバージョンを削除: {model_id}_v{old_version}")

class SpecializedStorage:
    """特化型ストレージクラス"""
    
    def __init__(self, base_storage: ModelStorage):
        self.base_storage = base_storage
    
    def save_fuzzy_decision_tree(self, tree: FuzzyDecisionTree, 
                                tree_id: str, description: str = "",
                                performance_metrics: Dict[str, float] = None) -> str:
        """ファジィ決定木の保存"""
        
        return self.base_storage.save_model(
            model=tree,
            model_id=tree_id,
            model_type="fuzzy_decision_tree",
            description=description,
            tags=["decision_tree", "fuzzy"],
            performance_metrics=performance_metrics
        )
    
    def load_fuzzy_decision_tree(self, tree_id: str, 
                                version: Optional[str] = None) -> FuzzyDecisionTree:
        """ファジィ決定木の読み込み"""
        
        return self.base_storage.load_model(tree_id, version)
    
    def save_genetic_population(self, population: Population, 
                               population_id: str, generation: int,
                               description: str = "") -> str:
        """遺伝的集団の保存"""
        
        training_info = {
            "generation": generation,
            "population_size": len(population.individuals),
            "best_fitness": population.get_best_individual().get_fitness() if population.get_best_individual() else 0.0
        }
        
        return self.base_storage.save_model(
            model=population,
            model_id=f"{population_id}_gen{generation}",
            model_type="genetic_population",
            description=description,
            tags=["genetic_algorithm", "population"],
            training_data_info=training_info
        )
    
    def save_optimized_weights(self, weights: WeightVector, 
                              weights_id: str, optimization_info: Dict[str, Any],
                              description: str = "") -> str:
        """最適化重みの保存"""
        
        performance_metrics = {
            "fitness_score": weights.get_fitness() or 0.0,
            "generation": optimization_info.get("generation", 0),
            "optimization_time": optimization_info.get("execution_time", 0.0)
        }
        
        return self.base_storage.save_model(
            model=weights,
            model_id=weights_id,
            model_type="optimized_weights",
            description=description,
            tags=["genetic_algorithm", "weights", "optimization"],
            performance_metrics=performance_metrics,
            training_data_info=optimization_info
        )
    
    def save_evaluation_results(self, results: List[EvaluationResponse],
                               results_id: str, description: str = "") -> str:
        """評価結果の保存"""
        
        training_info = {
            "total_evaluations": len(results),
            "average_processing_time": sum(r.processing_time for r in results) / len(results) if results else 0.0,
            "average_confidence": sum(r.recommendation_confidence for r in results) / len(results) if results else 0.0
        }
        
        return self.base_storage.save_model(
            model=results,
            model_id=results_id,
            model_type="evaluation_results",
            description=description,
            tags=["evaluation", "results"],
            training_data_info=training_info
        )

# 使用例とテスト
def test_model_storage():
    """モデルストレージのテスト"""
    
    print("💾 モデルストレージテスト開始")
    
    # ストレージ設定
    config = StorageConfig(
        base_directory="./test_models",
        compression_enabled=True,
        versioning_enabled=True,
        max_versions=3
    )
    
    # ストレージの初期化
    storage = ModelStorage(config)
    specialized = SpecializedStorage(storage)
    
    # テストモデル（辞書）の保存
    test_model = {
        "weights": [0.1, 0.2, 0.3, 0.4, 0.5],
        "bias": 0.1,
        "accuracy": 0.85
    }
    
    # モデルの保存
    model_id = storage.save_model(
        model=test_model,
        model_id="test_model",
        model_type="test",
        description="テスト用モデル",
        tags=["test", "example"],
        performance_metrics={"accuracy": 0.85, "loss": 0.15}
    )
    
    print(f"✅ モデル保存完了: {model_id}")
    
    # モデルの読み込み
    loaded_model = storage.load_model("test_model")
    print(f"✅ モデル読み込み完了: {loaded_model}")
    
    # モデル一覧の取得
    models = storage.list_models()
    print(f"📋 モデル一覧: {len(models)}件")
    
    for model in models:
        print(f"  - {model.model_id} ({model.model_type}) v{model.version}")
    
    # モデル情報の取得
    info = storage.get_model_info("test_model")
    if info:
        print(f"📊 モデル情報: {info.model_id} - {info.description}")
        print(f"    ファイルサイズ: {info.file_size} bytes")
        print(f"    性能指標: {info.performance_metrics}")
    
    # バックアップのテスト
    try:
        backup_path = storage.backup_model("test_model")
        print(f"💾 バックアップ完了: {backup_path}")
    except Exception as e:
        print(f"⚠️ バックアップエラー: {e}")
    
    # クリーンアップ
    try:
        shutil.rmtree("./test_models")
        print("🧹 テストファイル削除完了")
    except Exception as e:
        print(f"⚠️ クリーンアップエラー: {e}")
    
    print("✅ モデルストレージテスト完了")

if __name__ == "__main__":
    test_model_storage()