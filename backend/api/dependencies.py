# api/dependencies.py - 依存性注入

from fastapi import Depends, HTTPException, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any, List
import time
import hashlib
import logging
from datetime import datetime, timedelta
from functools import lru_cache

from models.schemas import StudentProfile, Laboratory
from services.lab_matching import LabMatchingService, MatchingConfig
from services.prediction import PredictionService
from models.storage import ModelStorage, StorageConfig, SpecializedStorage
from utils.logging_config import get_logger, get_performance_logger, get_audit_logger
from config.settings import settings

logger = get_logger(__name__)

# セキュリティ設定
security = HTTPBearer(auto_error=False)

class RateLimiter:
    """レート制限クラス"""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """リクエストが許可されるかチェック"""
        
        current_time = time.time()
        
        # クライアントのリクエスト履歴を取得
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        client_requests = self.requests[client_id]
        
        # 期限切れのリクエストを削除
        cutoff_time = current_time - self.window_seconds
        client_requests[:] = [req_time for req_time in client_requests if req_time > cutoff_time]
        
        # レート制限チェック
        if len(client_requests) >= self.max_requests:
            return False
        
        # 新しいリクエストを記録
        client_requests.append(current_time)
        return True
    
    def get_reset_time(self, client_id: str) -> int:
        """リセット時刻を取得（UNIX timestamp）"""
        
        if client_id not in self.requests or not self.requests[client_id]:
            return int(time.time())
        
        oldest_request = min(self.requests[client_id])
        return int(oldest_request + self.window_seconds)

# レート制限インスタンス
rate_limiter = RateLimiter(max_requests=1000, window_seconds=3600)  # 1時間に1000リクエスト

class RequestContext:
    """リクエストコンテキスト"""
    
    def __init__(self, request: Request):
        self.request = request
        self.start_time = time.time()
        self.client_ip = self._get_client_ip()
        self.user_agent = request.headers.get("user-agent", "unknown")
        self.request_id = self._generate_request_id()
        
        # パフォーマンスロガー
        self.perf_logger = get_performance_logger()
        self.audit_logger = get_audit_logger()
    
    def _get_client_ip(self) -> str:
        """クライアントIPアドレスの取得"""
        
        # プロキシ経由の場合のヘッダーをチェック
        forwarded_for = self.request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = self.request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # 直接接続の場合
        if hasattr(self.request, "client") and self.request.client:
            return self.request.client.host
        
        return "unknown"
    
    def _generate_request_id(self) -> str:
        """リクエストIDの生成"""
        
        timestamp = str(time.time())
        client_info = f"{self.client_ip}_{self.user_agent}"
        hash_input = f"{timestamp}_{client_info}".encode()
        
        return hashlib.md5(hash_input).hexdigest()[:12]
    
    def log_request_start(self, endpoint: str):
        """リクエスト開始ログ"""
        
        if self.audit_logger:
            self.audit_logger.log_user_action(
                user_id=self.client_ip,
                action="api_request",
                resource=endpoint,
                result="started",
                details={
                    "request_id": self.request_id,
                    "user_agent": self.user_agent,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def log_request_end(self, endpoint: str, status: str, response_size: int = 0):
        """リクエスト終了ログ"""
        
        execution_time = time.time() - self.start_time
        
        if self.perf_logger:
            self.perf_logger.log_prediction_performance(
                student_id=self.request_id,
                processing_time=execution_time,
                lab_count=response_size,
                algorithm_version="api_v1",
                confidence=1.0
            )
        
        if self.audit_logger:
            self.audit_logger.log_user_action(
                user_id=self.client_ip,
                action="api_request",
                resource=endpoint,
                result=status,
                details={
                    "request_id": self.request_id,
                    "execution_time": execution_time,
                    "response_size": response_size
                }
            )

# サービスインスタンス管理
_service_instances: Dict[str, Any] = {}

@lru_cache()
def get_matching_config() -> MatchingConfig:
    """マッチング設定の取得"""
    
    return MatchingConfig(
        basic_criteria_weight=0.6,
        extended_criteria_weight=0.3,
        special_criteria_weight=0.1,
        field_match_bonus=0.2,
        min_compatibility_threshold=0.3,
        high_compatibility_threshold=0.7,
        confidence_threshold=0.5,
        max_recommendations=20,
        enable_genetic_optimization=True,
        use_fuzzy_inference=True,
        enable_explanation=True
    )

@lru_cache()
def get_storage_config() -> StorageConfig:
    """ストレージ設定の取得"""
    
    return StorageConfig(
        base_directory=settings.models_dir,
        compression_enabled=True,
        versioning_enabled=True,
        max_versions=10,
        backup_enabled=True
    )

def get_lab_matching_service() -> LabMatchingService:
    """研究室マッチングサービスの取得"""
    
    if "lab_matching_service" not in _service_instances:
        config = get_matching_config()
        service = LabMatchingService(config)
        _service_instances["lab_matching_service"] = service
        logger.info("研究室マッチングサービス初期化完了")
    
    return _service_instances["lab_matching_service"]

def get_prediction_service() -> Optional[PredictionService]:
    """予測サービスの取得"""
    
    if "prediction_service" not in _service_instances:
        try:
            from services.prediction import PredictionService
            service = PredictionService()
            _service_instances["prediction_service"] = service
            logger.info("予測サービス初期化完了")
        except ImportError as e:
            logger.warning(f"予測サービスの初期化に失敗: {e}")
            _service_instances["prediction_service"] = None
    
    return _service_instances["prediction_service"]

def get_model_storage() -> ModelStorage:
    """モデルストレージの取得"""
    
    if "model_storage" not in _service_instances:
        config = get_storage_config()
        storage = ModelStorage(config)
        _service_instances["model_storage"] = storage
        logger.info("モデルストレージ初期化完了")
    
    return _service_instances["model_storage"]

def get_specialized_storage() -> SpecializedStorage:
    """特化型ストレージの取得"""
    
    if "specialized_storage" not in _service_instances:
        base_storage = get_model_storage()
        storage = SpecializedStorage(base_storage)
        _service_instances["specialized_storage"] = storage
    
    return _service_instances["specialized_storage"]

def get_request_context(request: Request) -> RequestContext:
    """リクエストコンテキストの取得"""
    
    return RequestContext(request)

def verify_rate_limit(request: Request) -> RequestContext:
    """レート制限の確認"""
    
    context = get_request_context(request)
    
    # レート制限チェック
    if not rate_limiter.is_allowed(context.client_ip):
        reset_time = rate_limiter.get_reset_time(context.client_ip)
        
        logger.warning(f"レート制限超過: {context.client_ip}")
        
        raise HTTPException(
            status_code=429,
            detail="Too Many Requests",
            headers={
                "X-RateLimit-Limit": str(rate_limiter.max_requests),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_time)
            }
        )
    
    return context

def verify_content_type(content_type: Optional[str] = Header(None)) -> bool:
    """Content-Typeの確認"""
    
    if content_type and "application/json" not in content_type:
        raise HTTPException(
            status_code=415,
            detail="Unsupported Media Type. Expected application/json"
        )
    
    return True

def verify_api_key(authorization: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[str]:
    """APIキーの確認（将来の認証用）"""
    
    # 現在は認証なしで通す
    return "anonymous"

def validate_student_profile(profile: StudentProfile) -> StudentProfile:
    """学生プロフィールのバリデーション"""
    
    # 基本バリデーション（Pydanticで実行済み）
    
    # カスタムバリデーション
    errors = []
    
    # 必須評価基準のチェック
    required_criteria = ["research_intensity", "advisor_style", "team_work", "workload", "theory_practice"]
    criteria_dict = profile.evaluation_criteria.dict()
    
    for criterion in required_criteria:
        value = criteria_dict.get(criterion)
        if value is None:
            errors.append(f"必須評価基準が未設定: {criterion}")
        elif not (1 <= value <= 10):
            errors.append(f"評価基準の値が範囲外: {criterion}={value}")
    
    # 分野興味のチェック
    if not profile.field_interests:
        errors.append("最低1つの研究分野への興味が必要です")
    
    # 優先順位の重複チェック
    priorities = [interest.priority for interest in profile.field_interests]
    if len(priorities) != len(set(priorities)):
        errors.append("研究分野の優先順位に重複があります")
    
    if errors:
        raise HTTPException(
            status_code=400,
            detail=f"学生プロフィールバリデーションエラー: {'; '.join(errors)}"
        )
    
    return profile

def validate_lab_ids(lab_ids: List[str], 
                    matching_service: LabMatchingService = Depends(get_lab_matching_service)) -> List[str]:
    """研究室IDリストのバリデーション"""
    
    if not lab_ids:
        return lab_ids
    
    invalid_ids = []
    
    for lab_id in lab_ids:
        lab = matching_service.get_laboratory(lab_id)
        if not lab:
            invalid_ids.append(lab_id)
    
    if invalid_ids:
        raise HTTPException(
            status_code=404,
            detail=f"無効な研究室ID: {', '.join(invalid_ids)}"
        )
    
    return lab_ids

def validate_pagination(page: int = 1, size: int = 10, max_size: int = 100) -> tuple:
    """ページネーションパラメータのバリデーション"""
    
    if page < 1:
        raise HTTPException(
            status_code=400,
            detail="ページ番号は1以上である必要があります"
        )
    
    if size < 1:
        raise HTTPException(
            status_code=400,
            detail="ページサイズは1以上である必要があります"
        )
    
    if size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"ページサイズは{max_size}以下である必要があります"
        )
    
    offset = (page - 1) * size
    
    return offset, size

class ServiceHealth:
    """サービスヘルスチェック"""
    
    @staticmethod
    def check_lab_matching_service() -> Dict[str, Any]:
        """研究室マッチングサービスの健全性チェック"""
        
        try:
            service = get_lab_matching_service()
            stats = service.get_service_statistics()
            
            return {
                "status": "healthy",
                "available_laboratories": stats["available_laboratories"],
                "total_evaluations": stats["total_evaluations"],
                "success_rate": stats["success_rate"]
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    @staticmethod
    def check_prediction_service() -> Dict[str, Any]:
        """予測サービスの健全性チェック"""
        
        try:
            service = get_prediction_service()
            
            if service is None:
                return {
                    "status": "unavailable",
                    "error": "予測サービスが利用できません"
                }
            
            return {
                "status": "healthy",
                "service_available": True
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    @staticmethod
    def check_model_storage() -> Dict[str, Any]:
        """モデルストレージの健全性チェック"""
        
        try:
            storage = get_model_storage()
            models = storage.list_models()
            
            return {
                "status": "healthy",
                "total_models": len(models),
                "storage_directory": str(storage.models_dir)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    @staticmethod
    def get_system_health() -> Dict[str, Any]:
        """システム全体の健全性チェック"""
        
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "services": {
                "lab_matching": ServiceHealth.check_lab_matching_service(),
                "prediction": ServiceHealth.check_prediction_service(),
                "model_storage": ServiceHealth.check_model_storage()
            }
        }
        
        # 全体ステータスの判定
        service_statuses = [service["status"] for service in health_status["services"].values()]
        
        if "unhealthy" in service_statuses:
            health_status["overall_status"] = "unhealthy"
        elif "unavailable" in service_statuses:
            health_status["overall_status"] = "degraded"
        
        return health_status

def get_system_health() -> Dict[str, Any]:
    """システムヘルスの取得（依存性注入用）"""
    
    return ServiceHealth.get_system_health()

# サービス初期化
def initialize_services():
    """サービスの初期化"""
    
    try:
        # 必須サービスの初期化
        get_lab_matching_service()
        get_model_storage()
        
        # オプショナルサービスの初期化
        get_prediction_service()
        
        logger.info("全サービス初期化完了")
        
    except Exception as e:
        logger.error(f"サービス初期化エラー: {e}")
        raise

# サービスシャットダウン
def shutdown_services():
    """サービスのシャットダウン"""
    
    try:
        # サービスのクリーンアップ
        for service_name, service in _service_instances.items():
            if hasattr(service, 'shutdown'):
                service.shutdown()
            logger.info(f"{service_name} シャットダウン完了")
        
        _service_instances.clear()
        logger.info("全サービスシャットダウン完了")
        
    except Exception as e:
        logger.error(f"サービスシャットダウンエラー: {e}")

# 使用例とテスト
def test_dependencies():
    """依存性注入のテスト"""
    
    print("🔗 依存性注入テスト開始")
    
    # サービス初期化
    initialize_services()
    
    # マッチングサービスのテスト
    matching_service = get_lab_matching_service()
    print(f"✅ マッチングサービス: {type(matching_service).__name__}")
    
    # ストレージのテスト
    storage = get_model_storage()
    print(f"✅ モデルストレージ: {type(storage).__name__}")
    
    # ヘルスチェックのテスト
    health = ServiceHealth.get_system_health()
    print(f"🏥 システムヘルス: {health['overall_status']}")
    
    for service_name, service_health in health['services'].items():
        print(f"  - {service_name}: {service_health['status']}")
    
    # シャットダウン
    shutdown_services()
    
    print("✅ 依存性注入テスト完了")

if __name__ == "__main__":
    test_dependencies()