# services/prediction.py - 予測サービス

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging
from datetime import datetime

from models.schemas import StudentProfile, Laboratory, LabResult, CompatibilityScore, EvaluationResponse
from core.fuzzy.inference import FuzzyInferenceEngine, InferenceResult
from core.decision_tree.tree import FuzzyDecisionTree, PredictionResult
from core.decision_tree.builder import FuzzyTreeBuilder, BuilderConfig
from services.optimization import OptimizationService, OptimizationConfig
from services.lab_matching import LabMatchingService
from utils.metrics import CompatibilityMetrics, PredictionEvaluator
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class PredictionContext:
    """予測コンテキスト"""
    student_id: str
    timestamp: datetime
    algorithm_version: str
    model_parameters: Dict[str, Any]
    execution_time: float
    confidence_level: float

class PredictionService:
    """統合予測サービス"""
    
    def __init__(self):
        self.fuzzy_engine = FuzzyInferenceEngine()
        self.decision_tree: Optional[FuzzyDecisionTree] = None
        self.optimization_service = OptimizationService()
        self.lab_matching_service = LabMatchingService()
        
        # 予測履歴
        self.prediction_history: List[Dict[str, Any]] = []
        
        # 性能メトリクス
        self.metrics = CompatibilityMetrics()
        self.evaluator = PredictionEvaluator()
        
        # 統計情報
        self.prediction_stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "average_execution_time": 0.0,
            "average_confidence": 0.0
        }
    
    def predict_lab_compatibility(self, student_profile: StudentProfile,
                                 target_lab: Laboratory) -> Dict[str, Any]:
        """単一研究室との適合性予測"""
        
        start_time = datetime.now()
        
        try:
            # ファジィ推論による詳細分析
            fuzzy_results = self.fuzzy_engine.infer_lab_compatibility(student_profile, target_lab)
            
            # 決定木予測（利用可能な場合）
            tree_prediction = None
            if self.decision_tree is not None:
                tree_input = self._prepare_tree_input(student_profile, target_lab)
                tree_prediction = self.decision_tree.predict(tree_input)
            
            # 統合スコア計算
            compatibility_score = self._calculate_integrated_score(
                fuzzy_results, tree_prediction, student_profile, target_lab
            )
            
            # 推奨事項生成
            recommendations = self._generate_detailed_recommendations(
                fuzzy_results, compatibility_score, student_profile, target_lab
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # 結果構築
            result = {
                "lab": target_lab,
                "compatibility_score": compatibility_score,
                "fuzzy_analysis": self._format_fuzzy_results(fuzzy_results),
                "tree_prediction": self._format_tree_prediction(tree_prediction),
                "recommendations": recommendations,
                "context": PredictionContext(
                    student_id=student_profile.student_id,
                    timestamp=start_time,
                    algorithm_version="2.0.0",
                    model_parameters=self._get_model_parameters(),
                    execution_time=execution_time,
                    confidence_level=compatibility_score.overall_score / 10.0
                ),
                "explanation": self._generate_explanation(fuzzy_results, compatibility_score)
            }
            
            # 履歴記録
            self._record_prediction(result)
            
            return result
            
        except Exception as e:
            logger.error(f"単一研究室予測エラー: {str(e)}")
            return self._create_error_result(student_profile, target_lab, str(e))
    
    def predict_multiple_labs(self, student_profile: StudentProfile,
                            labs: List[Laboratory]) -> EvaluationResponse:
        """複数研究室の適合性予測とランキング"""
        
        logger.info(f"複数研究室予測開始: 学生ID={student_profile.student_id}, 研究室数={len(labs)}")
        
        # 最適化ベースの高精度予測
        return self.lab_matching_service.find_best_matches(student_profile)
    
    def predict_with_decision_tree(self, student_profile: StudentProfile,
                                  labs: List[Laboratory]) -> List[PredictionResult]:
        """決定木ベースの予測"""
        
        if self.decision_tree is None:
            raise ValueError("決定木モデルが構築されていません")
        
        results = []
        
        for lab in labs:
            tree_input = self._prepare_tree_input(student_profile, lab)
            prediction = self.decision_tree.predict(tree_input)
            results.append(prediction)
        
        return results
    
    def train_decision_tree(self, training_data: List[Dict[str, Any]]) -> FuzzyDecisionTree:
        """決定木モデルの学習"""
        
        logger.info(f"決定木学習開始: サンプル数={len(training_data)}")
        
        # データ準備
        X, y = self._prepare_training_data(training_data)
        
        # 決定木構築
        config = BuilderConfig(
            max_depth=settings.max_tree_depth,
            min_samples_split=settings.min_samples_split,
            criterion="fuzzy_entropy",
            fuzzy_threshold=0.1
        )
        
        builder = FuzzyTreeBuilder(config)
        self.decision_tree = builder.build_tree(X, y)
        
        # 評価
        self._evaluate_decision_tree(X, y)
        
        logger.info("決定木学習完了")
        
        return self.decision_tree
    
    def predict_career_paths(self, student_profile: StudentProfile,
                           selected_labs: List[Laboratory]) -> Dict[str, Any]:
        """キャリアパス予測"""
        
        career_analysis = {
            "potential_careers": [],
            "skill_development_paths": [],
            "industry_recommendations": [],
            "graduate_school_options": []
        }
        
        # 選択研究室の分野分析
        field_distribution = self._analyze_field_distribution(selected_labs)
        
        # キャリアパス推定
        for field_category, weight in field_distribution.items():
            if weight > 0.3:  # 重要な分野
                careers = self._get_careers_for_field(field_category)
                for career in careers:
                    career["relevance_score"] = weight
                    career_analysis["potential_careers"].append(career)
        
        # スキル開発パス
        skill_paths = self._generate_skill_development_paths(student_profile, selected_labs)
        career_analysis["skill_development_paths"] = skill_paths
        
        return career_analysis
    
    def explain_prediction(self, prediction_result: Dict[str, Any]) -> str:
        """予測結果の詳細説明を生成"""
        
        explanation_parts = []
        
        # 総合スコア説明
        overall_score = prediction_result["compatibility_score"].overall_score
        explanation_parts.append(f"総合適合度スコアは{overall_score:.1f}点です。")
        
        if overall_score >= 8.0:
            explanation_parts.append("これは非常に高い適合性を示しており、強く推奨される研究室です。")
        elif overall_score >= 6.5:
            explanation_parts.append("これは良好な適合性を示しており、検討に値する研究室です。")
        elif overall_score >= 5.0:
            explanation_parts.append("これは中程度の適合性です。他の選択肢と比較検討することをお勧めします。")
        else:
            explanation_parts.append("適合性がやや低い可能性があります。慎重に検討してください。")
        
        # ファジィ分析説明
        if "fuzzy_analysis" in prediction_result:
            fuzzy_results = prediction_result["fuzzy_analysis"]
            
            # 分野適合性
            field_compatibility = fuzzy_results.get("field_compatibility", {})
            if field_compatibility.get("crisp_value", 0) >= 7:
                explanation_parts.append("研究分野の興味と非常によく一致しています。")
            elif field_compatibility.get("crisp_value", 0) >= 5:
                explanation_parts.append("研究分野の興味とある程度一致しています。")
            else:
                explanation_parts.append("研究分野の適合性を再確認することをお勧めします。")
        
        # 推奨事項
        recommendations = prediction_result.get("recommendations", [])
        if recommendations:
            explanation_parts.append("具体的な推奨事項:")
            for i, rec in enumerate(recommendations[:3], 1):
                explanation_parts.append(f"{i}. {rec}")
        
        return "\n".join(explanation_parts)
    
    def _calculate_integrated_score(self, fuzzy_results: Dict[str, InferenceResult],
                                   tree_prediction: Optional[PredictionResult],
                                   student: StudentProfile, lab: Laboratory) -> CompatibilityScore:
        """ファジィ推論と決定木の統合スコア計算"""
        
        # ファジィ推論スコア
        fuzzy_score = 0.0
        if "overall_compatibility" in fuzzy_results:
            fuzzy_score = fuzzy_results["overall_compatibility"].crisp_value
        
        # 決定木スコア
        tree_score = 0.0
        if tree_prediction is not None:
            tree_score = tree_prediction.confidence * 10  # 10点満点に変換
        
        # 統合計算
        if tree_prediction is not None:
            # 両方利用可能
            integrated_score = fuzzy_score * 0.7 + tree_score * 0.3
        else:
            # ファジィ推論のみ
            integrated_score = fuzzy_score
        
        # 詳細スコア
        detailed_scores = {}
        for component, result in fuzzy_results.items():
            detailed_scores[component] = result.crisp_value
        
        return CompatibilityScore(
            overall_score=round(integrated_score, 2),
            field_compatibility=fuzzy_results.get("field_compatibility", InferenceResult("", 0, {}, [], 0)).crisp_value / 10,
            criteria_compatibility=fuzzy_results.get("research_style_match", InferenceResult("", 0, {}, [], 0)).crisp_value / 10,
            detailed_scores=detailed_scores
        )
    
    def _prepare_tree_input(self, student: StudentProfile, lab: Laboratory) -> Dict[str, Any]:
        """決定木用の入力データ準備"""
        
        input_data = {}
        
        # 学生の評価基準
        student_criteria = student.evaluation_criteria.dict()
        for key, value in student_criteria.items():
            input_data[f"student_{key}"] = float(value)
        
        # 研究室特徴
        lab_features = lab.features.dict()
        for key, value in lab_features.items():
            input_data[f"lab_{key}"] = float(value)
        
        # 分野関連特徴
        student_fields = {fi.field_id: fi for fi in student.field_interests}
        
        # 分野マッチング特徴
        field_match_score = 0.0
        matched_fields = 0
        
        for field_id in lab.research_fields:
            if field_id in student_fields:
                field_interest = student_fields[field_id]
                field_score = (
                    field_interest.interest_level * 0.5 +
                    field_interest.experience_level * 0.3 +
                    field_interest.importance_level * 0.2
                )
                field_match_score += field_score
                matched_fields += 1
        
        input_data["field_match_score"] = field_match_score / matched_fields if matched_fields > 0 else 0
        input_data["matched_fields_count"] = matched_fields
        
        return input_data
    
    def _prepare_training_data(self, training_data: List[Dict[str, Any]]) -> Tuple[List[Dict], List[str]]:
        """学習データの準備"""
        
        X = []
        y = []
        
        for sample in training_data:
            # 特徴量
            features = {
                key: value for key, value in sample.items()
                if key not in ["compatibility_label", "overall_score"]
            }
            X.append(features)
            
            # ラベル（適合性レベル）
            if "compatibility_label" in sample:
                y.append(sample["compatibility_label"])
            elif "overall_score" in sample:
                # スコアからラベルを生成
                score = sample["overall_score"]
                if score >= 8:
                    label = "excellent"
                elif score >= 6.5:
                    label = "good"
                elif score >= 5:
                    label = "fair"
                else:
                    label = "poor"
                y.append(label)
            else:
                y.append("unknown")
        
        return X, y
    
    def _evaluate_decision_tree(self, X: List[Dict], y: List[str]) -> None:
        """決定木の評価"""
        
        if self.decision_tree is None:
            return
        
        predictions = self.decision_tree.predict(X)
        predicted_classes = [pred.predicted_class for pred in predictions]
        
        # 精度計算
        correct = sum(1 for true, pred in zip(y, predicted_classes) if true == pred)
        accuracy = correct / len(y) if y else 0.0
        
        logger.info(f"決定木精度: {accuracy:.3f}")
        
        # 木の統計
        summary = self.decision_tree.get_tree_summary()
        logger.info(f"決定木統計: 深度={summary['structure']['depth']}, "
                   f"ノード数={summary['structure']['node_count']}")
    
    def _generate_detailed_recommendations(self, fuzzy_results: Dict[str, InferenceResult],
                                         compatibility: CompatibilityScore,
                                         student: StudentProfile, lab: Laboratory) -> List[str]:
        """詳細な推奨事項生成"""
        
        recommendations = []
        
        # スコアベースの基本推奨
        if compatibility.overall_score >= 8.0:
            recommendations.append("この研究室は非常に高い適合性を示しています。積極的に志望することをお勧めします。")
        elif compatibility.overall_score >= 6.5:
            recommendations.append("この研究室は良好な適合性があります。詳細な情報収集をしてみてください。")
        
        # 分野適合性に基づく推奨
        if "field_compatibility" in fuzzy_results:
            field_result = fuzzy_results["field_compatibility"]
            if field_result.crisp_value >= 8:
                recommendations.append("研究分野の興味と非常によく一致しています。")
            elif field_result.crisp_value < 5:
                recommendations.append("研究分野の適合性について、教授との面談で詳しく確認することをお勧めします。")
        
        # 経験レベルに基づく推奨
        student_fields = {fi.field_id: fi for fi in student.field_interests}
        
        # 経験不足分野のチェック
        inexperienced_fields = []
        for field_id in lab.research_fields:
            if field_id in student_fields:
                if student_fields[field_id].experience_level <= 3 and student_fields[field_id].interest_level >= 7:
                    field_info = settings.research_fields.get(field_id, {})
                    inexperienced_fields.append(field_info.get("name", field_id))
        
        if inexperienced_fields:
            recommendations.append(f"以下の分野について基礎学習を強化することをお勧めします: {', '.join(inexperienced_fields[:2])}")
        
        # 研究スタイルマッチング
        if "research_style_match" in fuzzy_results:
            style_result = fuzzy_results["research_style_match"]
            if style_result.crisp_value >= 7:
                recommendations.append("研究スタイルがよく合っています。")
            elif style_result.crisp_value < 5:
                recommendations.append("研究スタイルについて事前に確認することをお勧めします。")
        
        return recommendations
    
    def _format_fuzzy_results(self, fuzzy_results: Dict[str, InferenceResult]) -> Dict[str, Any]:
        """ファジィ推論結果のフォーマット"""
        
        formatted = {}
        
        for component, result in fuzzy_results.items():
            formatted[component] = {
                "crisp_value": round(result.crisp_value, 3),
                "confidence": round(result.confidence, 3),
                "activated_rules": result.activated_rules,
                "linguistic_evaluation": self.fuzzy_engine.membership_func.linguistic_evaluation(result.crisp_value)
            }
        
        return formatted
    
    def _format_tree_prediction(self, tree_prediction: Optional[PredictionResult]) -> Optional[Dict[str, Any]]:
        """決定木予測結果のフォーマット"""
        
        if tree_prediction is None:
            return None
        
        return {
            "predicted_class": tree_prediction.predicted_class,
            "confidence": round(tree_prediction.confidence, 3),
            "class_probabilities": tree_prediction.class_probabilities,
            "decision_path": tree_prediction.path,
            "path_length": len(tree_prediction.path)
        }
    
    def _analyze_field_distribution(self, labs: List[Laboratory]) -> Dict[str, float]:
        """研究室の分野分布分析"""
        
        field_counts = {}
        total_fields = 0
        
        for lab in labs:
            for field_id in lab.research_fields:
                field_info = settings.research_fields.get(field_id, {})
                category = field_info.get("category", "その他")
                
                field_counts[category] = field_counts.get(category, 0) + 1
                total_fields += 1
        
        # 正規化
        return {
            category: count / total_fields
            for category, count in field_counts.items()
        } if total_fields > 0 else {}
    
    def _get_careers_for_field(self, field_category: str) -> List[Dict[str, Any]]:
        """分野カテゴリに対応するキャリア情報を取得"""
        
        career_mapping = {
            "テクノロジー・システム": [
                {"title": "ソフトウェアエンジニア", "industry": "IT", "growth_potential": "高"},
                {"title": "データサイエンティスト", "industry": "IT・金融", "growth_potential": "高"},
                {"title": "システムアーキテクト", "industry": "IT", "growth_potential": "中"},
                {"title": "研究開発エンジニア", "industry": "製造業", "growth_potential": "中"}
            ],
            "クリエイティブ・デザイン": [
                {"title": "UI/UXデザイナー", "industry": "IT・広告", "growth_potential": "高"},
                {"title": "グラフィックデザイナー", "industry": "広告・出版", "growth_potential": "中"},
                {"title": "Webデザイナー", "industry": "IT・広告", "growth_potential": "中"},
                {"title": "アートディレクター", "industry": "広告・エンタメ", "growth_potential": "中"}
            ],
            "メディア・エンターテイメント": [
                {"title": "ゲームデザイナー", "industry": "ゲーム", "growth_potential": "高"},
                {"title": "映像クリエイター", "industry": "エンタメ・広告", "growth_potential": "中"},
                {"title": "サウンドデザイナー", "industry": "エンタメ・ゲーム", "growth_potential": "中"}
            ]
        }
        
        return career_mapping.get(field_category, [])
    
    def _generate_skill_development_paths(self, student: StudentProfile,
                                        labs: List[Laboratory]) -> List[Dict[str, Any]]:
        """スキル開発パスの生成"""
        
        paths = []
        
        # 選択分野に基づくスキルパス
        selected_fields = [fi.field_id for fi in student.field_interests]
        
        for field_id in selected_fields:
            field_info = settings.research_fields.get(field_id, {})
            
            path = {
                "field": field_info.get("name", field_id),
                "current_level": next((fi.experience_level for fi in student.field_interests if fi.field_id == field_id), 0),
                "target_level": 8,
                "recommended_skills": self._get_skills_for_field(field_id),
                "learning_resources": self._get_learning_resources(field_id)
            }
            paths.append(path)
        
        return paths
    
    def _get_skills_for_field(self, field_id: str) -> List[str]:
        """分野に必要なスキルを取得"""
        
        skill_mapping = {
            "ai_machine_learning": ["Python", "TensorFlow", "データ分析", "統計学"],
            "web_design_branding": ["HTML/CSS", "JavaScript", "デザインツール", "UX原則"],
            "game_programming": ["Unity", "C#", "ゲームデザイン", "3Dモデリング"]
        }
        
        return skill_mapping.get(field_id, ["プログラミング", "論理思考", "問題解決"])
    
    def _get_learning_resources(self, field_id: str) -> List[str]:
        """学習リソースを取得"""
        
        return [
            "オンライン講座",
            "専門書籍",
            "実践プロジェクト",
            "インターンシップ"
        ]
    
    def _get_model_parameters(self) -> Dict[str, Any]:
        """現在のモデルパラメータを取得"""
        
        return {
            "fuzzy_threshold": 0.1,
            "optimization_generations": settings.ga_generations,
            "population_size": settings.ga_population_size,
            "decision_tree_depth": settings.max_tree_depth if self.decision_tree else None
        }
    
    def _record_prediction(self, result: Dict[str, Any]) -> None:
        """予測履歴の記録"""
        
        self.prediction_history.append({
            "timestamp": result["context"].timestamp,
            "student_id": result["context"].student_id,
            "lab_id": result["lab"].id,
            "compatibility_score": result["compatibility_score"].overall_score,
            "execution_time": result["context"].execution_time
        })
        
        # 統計更新
        self.prediction_stats["total_predictions"] += 1
        if result["compatibility_score"].overall_score > 0:
            self.prediction_stats["successful_predictions"] += 1
        
        # 実行時間の移動平均更新
        current_avg = self.prediction_stats["average_execution_time"]
        total_predictions = self.prediction_stats["total_predictions"]
        new_time = result["context"].execution_time
        
        self.prediction_stats["average_execution_time"] = (
            (current_avg * (total_predictions - 1) + new_time) / total_predictions
        )
    
    def _create_error_result(self, student: StudentProfile, lab: Laboratory, error_msg: str) -> Dict[str, Any]:
        """エラー時の結果を作成"""
        
        return {
            "lab": lab,
            "compatibility_score": CompatibilityScore(overall_score=0.0, field_compatibility=0.0, criteria_compatibility=0.0, detailed_scores={}),
            "error": error_msg,
            "recommendations": ["予測中にエラーが発生しました。管理者に連絡してください。"],
            "context": PredictionContext(
                student_id=student.student_id,
                timestamp=datetime.now(),
                algorithm_version="2.0.0",
                model_parameters={},
                execution_time=0.0,
                confidence_level=0.0
            )
        }
    
    def _generate_explanation(self, fuzzy_results: Dict[str, InferenceResult],
                            compatibility: CompatibilityScore) -> str:
        """予測説明の生成"""
        
        explanation_parts = []
        
        # 総合評価
        score = compatibility.overall_score
        if score >= 8:
            explanation_parts.append(f"適合度{score:.1f}点は非常に高いスコアです。")
        elif score >= 6:
            explanation_parts.append(f"適合度{score:.1f}点は良好なスコアです。")
        else:
            explanation_parts.append(f"適合度{score:.1f}点は改善の余地があります。")
        
        # 発火ルール
        if fuzzy_results:
            total_rules = sum(len(result.activated_rules) for result in fuzzy_results.values())
            explanation_parts.append(f"{total_rules}個のファジィルールが適用されました。")
        
        return " ".join(explanation_parts)
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """予測統計情報を取得"""
        
        return {
            "total_predictions": self.prediction_stats["total_predictions"],
            "success_rate": (
                self.prediction_stats["successful_predictions"] / 
                max(1, self.prediction_stats["total_predictions"])
            ),
            "average_execution_time": self.prediction_stats["average_execution_time"],
            "recent_predictions": len([
                p for p in self.prediction_history 
                if (datetime.now() - p["timestamp"]).days <= 7
            ]),
            "model_status": {
                "fuzzy_engine": "active",
                "decision_tree": "active" if self.decision_tree else "inactive",
                "optimization_service": "active"
            }
        }
    
    def clear_history(self) -> None:
        """履歴をクリア"""
        self.prediction_history.clear()
        self.prediction_stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "average_execution_time": 0.0,
            "average_confidence": 0.0
        }