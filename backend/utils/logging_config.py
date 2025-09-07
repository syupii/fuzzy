# utils/logging_config.py - ログ設定

import logging
import logging.handlers
import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json

from config.settings import settings

class CustomFormatter(logging.Formatter):
    """カスタムログフォーマッター"""
    
    def __init__(self, include_context: bool = True):
        self.include_context = include_context
        
        # カラーコード
        self.colors = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m', # Magenta
            'RESET': '\033[0m'      # Reset
        }
        
        # ベースフォーマット
        if self.include_context:
            fmt = '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s'
        else:
            fmt = '%(asctime)s | %(levelname)s | %(message)s'
        
        super().__init__(fmt, datefmt='%Y-%m-%d %H:%M:%S')
    
    def format(self, record):
        """ログレコードのフォーマット"""
        
        # 色付きレベル名（コンソール出力用）
        if hasattr(record, 'use_colors') and record.use_colors:
            levelname = record.levelname
            if levelname in self.colors:
                record.levelname = f"{self.colors[levelname]}{levelname}{self.colors['RESET']}"
        
        # 追加コンテキスト情報
        if self.include_context:
            # モジュール名の短縮
            if len(record.name) > 30:
                record.name = "..." + record.name[-27:]
            
            # ファイル名の短縮
            if len(record.filename) > 20:
                record.filename = "..." + record.filename[-17:]
        
        return super().format(record)

class JSONFormatter(logging.Formatter):
    """JSON形式のログフォーマッター"""
    
    def format(self, record):
        """JSON形式でログをフォーマット"""
        
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # 例外情報の追加
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # 追加属性の処理
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 'exc_text']:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)

class LoggingConfig:
    """ログ設定管理クラス"""
    
    def __init__(self):
        self.log_dir = Path(settings.logs_dir)
        self.log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
        self.formatters = {}
        self.handlers = {}
        self.loggers = {}
        
        # ログディレクトリの作成
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # フォーマッターの設定
        self._setup_formatters()
        
        # ハンドラーの設定
        self._setup_handlers()
    
    def _setup_formatters(self):
        """フォーマッターの設定"""
        
        # コンソール用フォーマッター
        self.formatters['console'] = CustomFormatter(include_context=False)
        
        # ファイル用フォーマッター
        self.formatters['file'] = CustomFormatter(include_context=True)
        
        # JSON用フォーマッター
        self.formatters['json'] = JSONFormatter()
        
        # エラー専用フォーマッター
        self.formatters['error'] = CustomFormatter(include_context=True)
    
    def _setup_handlers(self):
        """ハンドラーの設定"""
        
        # コンソールハンドラー
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(self.formatters['console'])
        
        # 色付きログのためのフィルター
        console_handler.addFilter(lambda record: setattr(record, 'use_colors', True) or True)
        
        self.handlers['console'] = console_handler
        
        # アプリケーションログファイルハンドラー
        app_log_file = self.log_dir / 'application.log'
        app_file_handler = logging.handlers.RotatingFileHandler(
            app_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        app_file_handler.setLevel(self.log_level)
        app_file_handler.setFormatter(self.formatters['file'])
        self.handlers['app_file'] = app_file_handler
        
        # エラーログファイルハンドラー
        error_log_file = self.log_dir / 'error.log'
        error_file_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=10,
            encoding='utf-8'
        )
        error_file_handler.setLevel(logging.ERROR)
        error_file_handler.setFormatter(self.formatters['error'])
        self.handlers['error_file'] = error_file_handler
        
        # 統計ログハンドラー（JSON形式）
        stats_log_file = self.log_dir / 'statistics.jsonl'
        stats_handler = logging.handlers.RotatingFileHandler(
            stats_log_file,
            maxBytes=20 * 1024 * 1024,  # 20MB
            backupCount=3,
            encoding='utf-8'
        )
        stats_handler.setLevel(logging.INFO)
        stats_handler.setFormatter(self.formatters['json'])
        self.handlers['stats'] = stats_handler
        
        # APIアクセスログハンドラー
        api_log_file = self.log_dir / 'api_access.log'
        api_handler = logging.handlers.TimedRotatingFileHandler(
            api_log_file,
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8'
        )
        api_handler.setLevel(logging.INFO)
        api_handler.setFormatter(self.formatters['file'])
        self.handlers['api'] = api_handler
    
    def setup_logger(self, logger_name: str, 
                    handlers: list = None, 
                    level: Optional[int] = None) -> logging.Logger:
        """指定されたロガーの設定"""
        
        if handlers is None:
            handlers = ['console', 'app_file', 'error_file']
        
        logger = logging.getLogger(logger_name)
        logger.setLevel(level or self.log_level)
        
        # 既存のハンドラーをクリア
        logger.handlers.clear()
        
        # 指定されたハンドラーを追加
        for handler_name in handlers:
            if handler_name in self.handlers:
                logger.addHandler(self.handlers[handler_name])
        
        # 親ロガーへの伝播を防ぐ
        logger.propagate = False
        
        self.loggers[logger_name] = logger
        return logger
    
    def setup_module_loggers(self):
        """モジュール別ロガーの設定"""
        
        # ルートロガーの設定
        root_logger = self.setup_logger('', ['console', 'app_file', 'error_file'])
        
        # アプリケーションロガー
        app_logger = self.setup_logger('app', ['console', 'app_file', 'error_file'])
        
        # APIロガー
        api_logger = self.setup_logger('api', ['console', 'api', 'error_file'])
        
        # 統計ロガー
        stats_logger = self.setup_logger('stats', ['stats'])
        
        # コアモジュールロガー
        self.setup_logger('core.fuzzy', ['console', 'app_file', 'error_file'])
        self.setup_logger('core.genetic', ['console', 'app_file', 'error_file'])
        self.setup_logger('core.decision_tree', ['console', 'app_file', 'error_file'])
        
        # サービスロガー
        self.setup_logger('services', ['console', 'app_file', 'error_file'])
        
        # ユーティリティロガー
        self.setup_logger('utils', ['console', 'app_file', 'error_file'])
        
        # 外部ライブラリのログレベル調整
        logging.getLogger('uvicorn').setLevel(logging.WARNING)
        logging.getLogger('fastapi').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)
    
    def get_logger(self, name: str) -> logging.Logger:
        """ロガーの取得"""
        
        if name in self.loggers:
            return self.loggers[name]
        else:
            return self.setup_logger(name)
    
    def log_system_info(self):
        """システム情報のログ出力"""
        
        logger = self.get_logger('system')
        
        logger.info("="*60)
        logger.info("🧬🌳 遺伝的ファジィ決定木研究室選択支援システム")
        logger.info("="*60)
        logger.info(f"Python バージョン: {sys.version}")
        logger.info(f"ログレベル: {logging.getLevelName(self.log_level)}")
        logger.info(f"ログディレクトリ: {self.log_dir}")
        logger.info(f"設定環境: {settings.environment}")
        logger.info(f"デバッグモード: {settings.debug}")
        logger.info("="*60)

class PerformanceLogger:
    """性能ログ専用クラス"""
    
    def __init__(self, logging_config: LoggingConfig):
        self.logger = logging_config.get_logger('performance')
        self.performance_data = []
    
    def log_prediction_performance(self, 
                                 student_id: str,
                                 processing_time: float,
                                 lab_count: int,
                                 algorithm_version: str,
                                 confidence: float):
        """予測性能のログ"""
        
        perf_data = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'prediction',
            'student_id': student_id,
            'processing_time': processing_time,
            'lab_count': lab_count,
            'algorithm_version': algorithm_version,
            'confidence': confidence
        }
        
        self.logger.info("予測性能", extra=perf_data)
        self.performance_data.append(perf_data)
    
    def log_optimization_performance(self,
                                   optimization_id: str,
                                   algorithm: str,
                                   population_size: int,
                                   generations: int,
                                   best_fitness: float,
                                   execution_time: float):
        """最適化性能のログ"""
        
        opt_data = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'optimization',
            'optimization_id': optimization_id,
            'algorithm': algorithm,
            'population_size': population_size,
            'generations': generations,
            'best_fitness': best_fitness,
            'execution_time': execution_time
        }
        
        self.logger.info("最適化性能", extra=opt_data)
        self.performance_data.append(opt_data)
    
    def log_system_resource_usage(self,
                                 cpu_usage: float,
                                 memory_usage: float,
                                 active_requests: int):
        """システムリソース使用状況のログ"""
        
        resource_data = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'system_resources',
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'active_requests': active_requests
        }
        
        self.logger.info("リソース使用状況", extra=resource_data)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """性能サマリーの取得"""
        
        if not self.performance_data:
            return {}
        
        prediction_events = [d for d in self.performance_data if d['event_type'] == 'prediction']
        optimization_events = [d for d in self.performance_data if d['event_type'] == 'optimization']
        
        summary = {
            'total_events': len(self.performance_data),
            'prediction_events': len(prediction_events),
            'optimization_events': len(optimization_events)
        }
        
        if prediction_events:
            processing_times = [e['processing_time'] for e in prediction_events]
            summary['prediction_performance'] = {
                'average_processing_time': sum(processing_times) / len(processing_times),
                'max_processing_time': max(processing_times),
                'min_processing_time': min(processing_times)
            }
        
        if optimization_events:
            execution_times = [e['execution_time'] for e in optimization_events]
            best_fitnesses = [e['best_fitness'] for e in optimization_events]
            summary['optimization_performance'] = {
                'average_execution_time': sum(execution_times) / len(execution_times),
                'average_best_fitness': sum(best_fitnesses) / len(best_fitnesses)
            }
        
        return summary

class AuditLogger:
    """監査ログクラス"""
    
    def __init__(self, logging_config: LoggingConfig):
        self.logger = logging_config.get_logger('audit')
    
    def log_user_action(self, user_id: str, action: str, 
                       resource: str, result: str, details: Dict[str, Any] = None):
        """ユーザーアクションの監査ログ"""
        
        audit_data = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'action': action,
            'resource': resource,
            'result': result,
            'details': details or {}
        }
        
        self.logger.info(f"ユーザーアクション: {user_id} {action} {resource} -> {result}", 
                        extra=audit_data)
    
    def log_system_change(self, component: str, change_type: str, 
                         old_value: Any, new_value: Any, changed_by: str):
        """システム変更の監査ログ"""
        
        change_data = {
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'change_type': change_type,
            'old_value': str(old_value),
            'new_value': str(new_value),
            'changed_by': changed_by
        }
        
        self.logger.info(f"システム変更: {component} {change_type} by {changed_by}", 
                        extra=change_data)
    
    def log_data_access(self, user_id: str, data_type: str, 
                       access_type: str, record_count: int):
        """データアクセスの監査ログ"""
        
        access_data = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'data_type': data_type,
            'access_type': access_type,
            'record_count': record_count
        }
        
        self.logger.info(f"データアクセス: {user_id} {access_type} {data_type} ({record_count} records)", 
                        extra=access_data)

# グローバル設定インスタンス
logging_config = LoggingConfig()

def setup_logging():
    """ログ設定の初期化"""
    
    try:
        # モジュール別ロガーの設定
        logging_config.setup_module_loggers()
        
        # システム情報のログ出力
        logging_config.log_system_info()
        
        # 性能ログとしての監査ログの初期化
        performance_logger = PerformanceLogger(logging_config)
        audit_logger = AuditLogger(logging_config)
        
        # グローバル変数として保存
        globals()['performance_logger'] = performance_logger
        globals()['audit_logger'] = audit_logger
        
        logger = logging_config.get_logger('setup')
        logger.info("✅ ログ設定完了")
        
        return True
        
    except Exception as e:
        print(f"❌ ログ設定エラー: {e}")
        return False

def get_logger(name: str) -> logging.Logger:
    """ロガーの取得（便利関数）"""
    return logging_config.get_logger(name)

def get_performance_logger() -> PerformanceLogger:
    """性能ロガーの取得"""
    return globals().get('performance_logger')

def get_audit_logger() -> AuditLogger:
    """監査ロガーの取得"""
    return globals().get('audit_logger')

# ログデコレータ
def log_execution_time(logger_name: str = None):
    """実行時間ログデコレータ"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name or func.__module__)
            
            start_time = datetime.now()
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                logger.info(f"関数実行完了: {func.__name__} ({execution_time:.3f}秒)")
                return result
                
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"関数実行エラー: {func.__name__} ({execution_time:.3f}秒) - {e}")
                raise
                
        return wrapper
    return decorator

def log_api_request(func):
    """API リクエストログデコレータ"""
    
    def wrapper(*args, **kwargs):
        logger = get_logger('api')
        
        # リクエスト情報の抽出
        request_info = f"{func.__name__}"
        if args:
            request_info += f" with {len(args)} args"
        if kwargs:
            request_info += f" and {len(kwargs)} kwargs"
        
        start_time = datetime.now()
        logger.info(f"API リクエスト開始: {request_info}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"API リクエスト完了: {request_info} ({execution_time:.3f}秒)")
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"API リクエストエラー: {request_info} ({execution_time:.3f}秒) - {e}")
            raise
            
    return wrapper

# 使用例とテスト
def test_logging():
    """ログ設定のテスト"""
    
    print("📝 ログ設定テスト開始")
    
    # ログ設定の初期化
    success = setup_logging()
    print(f"ログ設定: {'成功' if success else '失敗'}")
    
    # 各レベルのログテスト
    logger = get_logger('test')
    
    logger.debug("これはデバッグメッセージです")
    logger.info("これは情報メッセージです")
    logger.warning("これは警告メッセージです")
    logger.error("これはエラーメッセージです")
    
    # 性能ログのテスト
    perf_logger = get_performance_logger()
    if perf_logger:
        perf_logger.log_prediction_performance(
            student_id="test_001",
            processing_time=0.123,
            lab_count=5,
            algorithm_version="v1.0",
            confidence=0.85
        )
        print("✅ 性能ログテスト完了")
    
    # 監査ログのテスト
    audit_logger = get_audit_logger()
    if audit_logger:
        audit_logger.log_user_action(
            user_id="test_user",
            action="predict",
            resource="lab_compatibility",
            result="success"
        )
        print("✅ 監査ログテスト完了")
    
    print("✅ ログ設定テスト完了")

if __name__ == "__main__":
    test_logging()