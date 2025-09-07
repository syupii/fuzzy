# services/lab_matching.py - 研究室マッチングサービス

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from models.schemas import (
    StudentProfile, Laboratory, LabResult, CompatibilityScore,
    EvaluationResponse, ResearchFieldEnum
)
from core.fuzzy.inference import FuzzyInferenceEngine, SimpleFuzzyInferenceEngine
from core.genetic.evolution import EvolutionEngine, EvolutionConfig
from core.genetic.individual import WeightVector
from core.decision_tree.tree import FuzzyDecisionTree
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class MatchingConfig:
    """マッチング設定"""
    # 重み設定
    basic_criteria_weight: float = 0.6      # 基本5項目の重み
    extended_criteria_weight: float = 0.3   # 拡張5項目の重み
    special_criteria_weight: float = 0.1    # 特殊3項目の重み
    field_match_bonus: float = 0.2          # 分野適合ボーナス
    
    # 閾値設定
    min_compatibility_threshold: float = 0.3
    high_compatibility_threshold: float = 0.7
    confidence_threshold: float = 0.5
    
    # ランキング設定
    max_recommendations: int = 10
    diversity_factor: float = 0.1
    
    # 最適化設定
    enable_genetic_optimization: bool = True
    optimization_samples: int = 50
    
    # その他
    use_fuzzy_inference: bool = True
    enable_explanation: bool = True

class LabMatchingService:
    """研究室マッチングサービス"""
    
    def __init__(self, config: MatchingConfig = None):
        self.config = config or MatchingConfig()
        
        # 推論エンジンの初期化
        try:
            self.fuzzy_engine = FuzzyInferenceEngine()
            self.fallback_engine = SimpleFuzzyInferenceEngine()
            logger.info("ファジィ推論エンジン初期化完了")
        except Exception as e:
            logger.warning(f"ファジィ推論エンジン初期化失敗: {e}")
            self.fuzzy_engine = None
            self.fallback_engine = SimpleFuzzyInferenceEngine()
        
        # 遺伝的最適化エンジン
        self.genetic_engine: Optional[EvolutionEngine] = None
        self.optimized_weights: Optional[WeightVector] = None
        
        # 研究室データ
        self.laboratories: List[Laboratory] = []
        self.lab_cache: Dict[str, Laboratory] = {}
        
        # マッチング履歴
        self.matching_history: List[Dict[str, Any]] = []
        
        # 統計情報
        self.total_evaluations = 0
        self.successful_matches = 0
        
        # 研究室データの初期化
        self._initialize_laboratory_data()
    
    def _initialize_laboratory_data(self):
        """研究室データの初期化"""
        
        try:
            # 設定から研究室データを読み込み
            self.laboratories = self._create_sample_laboratories()
            
            # キャッシュの構築
            self.lab_cache = {lab.lab_id: lab for lab in self.laboratories}
            
            logger.info(f"研究室データ初期化完了: {len(self.laboratories)}研究室")
            
        except Exception as e:
            logger.error(f"研究室データ初期化エラー: {e}")
            self.laboratories = []
            self.lab_cache = {}
    
    def _create_sample_laboratories(self) -> List[Laboratory]:
        """サンプル研究室データの作成"""
        
        from models.schemas import Faculty, EvaluationCriteria
        
        laboratories = []
        
        # AI・機械学習分野の研究室例
        ai_labs = [
            {
                "lab_id": "ai_itoh_lab",
                "faculty": Faculty(
                    name="伊藤雅彦",
                    name_en="Masahiko ITOH",
                    specialties=["情報可視化", "ユーザインタフェース", "データ工学"]
                ),
                "research_field": ResearchFieldEnum.AI_MACHINE_LEARNING,
                "characteristics": EvaluationCriteria(
                    research_intensity=7.5,
                    advisor_style=7.0,
                    team_work=8.0,
                    workload=7.0,
                    theory_practice=6.5,
                    research_field_match=9.0,
                    skill_development=8.5,
                    lab_atmosphere=8.0,
                    flexibility=7.5,
                    publication_opportunity=7.0,
                    interdisciplinary=6.0,
                    communication_style=8.0,
                    innovation_risk=6.5
                ),
                "description": "情報可視化とユーザインタフェースに特化した研究室"
            },
            {
                "lab_id": "ai_uchiyama_lab",
                "faculty": Faculty(
                    name="内山敏雄",
                    name_en="Toshio UCHIYAMA",
                    specialties=["データ解析", "機械学習", "レコメンド", "テキストマイニング"]
                ),
                "research_field": ResearchFieldEnum.AI_MACHINE_LEARNING,
                "characteristics": EvaluationCriteria(
                    research_intensity=8.5,
                    advisor_style=6.5,
                    team_work=7.0,
                    workload=8.0,
                    theory_practice=8.0,
                    research_field_match=9.5,
                    skill_development=9.0,
                    lab_atmosphere=7.5,
                    flexibility=6.5,
                    publication_opportunity=8.5,
                    interdisciplinary=7.0,
                    communication_style=7.0,
                    innovation_risk=8.0
                ),
                "description": "機械学習とデータマイニングの実践的研究"
            }
        ]
        
        # 画像・映像処理分野の研究室例
        image_labs = [
            {
                "lab_id": "img_mori_lab",
                "faculty": Faculty(
                    name="森圭佑",
                    name_en="Keisuke MORI",
                    specialties=["情報計測", "音声・画像情報処理", "医用情報処理"]
                ),
                "research_field": ResearchFieldEnum.IMAGE_VIDEO_PROCESSING,
                "characteristics": EvaluationCriteria(
                    research_intensity=8.0,
                    advisor_style=7.5,
                    team_work=6.5,
                    workload=7.5,
                    theory_practice=7.5,
                    research_field_match=8.5,
                    skill_development=8.0,
                    lab_atmosphere=7.0,
                    flexibility=7.0,
                    publication_opportunity=7.5,
                    interdisciplinary=8.5,
                    communication_style=7.0,
                    innovation_risk=7.5
                ),
                "description": "医用画像処理と信号処理の研究"
            }
        ]
        
        # 研究室オブジェクトの作成
        for lab_data in ai_labs + image_labs:
            lab = Laboratory(**lab_data)
            laboratories.append(lab)
        
        return laboratories
    
    def evaluate_student_lab_compatibility(self, student_profile: StudentProfile,
                                         target_laboratories: Optional[List[Laboratory]] = None) -> EvaluationResponse:
        """学生と研究室群の適合性評価"""
        
        start_time = datetime.now()
        evaluation_id = f"eval_{int(start_time.timestamp())}"
        
        # 対象研究室の設定
        if target_laboratories is None:
            target_laboratories = self.laboratories
        
        if not target_laboratories:
            raise ValueError("評価対象の研究室がありません")
        
        logger.info(f"適合性評価開始: 学生{student_profile.student_id}, 研究室{len(target_laboratories)}件")
        
        # 各研究室との適合性評価
        lab_results = []
        
        for lab in target_laboratories:
            try:
                compatibility_score = self._calculate_compatibility(student_profile, lab)
                reasons, concerns = self._generate_explanations(student_profile, lab, compatibility_score)
                
                lab_result = LabResult(
                    laboratory=lab,
                    compatibility_score=compatibility_score,
                    ranking=0,  # 後で設定
                    reasons=reasons,
                    concerns=concerns
                )
                
                lab_results.append(lab_result)
                self.total_evaluations += 1
                
            except Exception as e:
                logger.warning(f"研究室{lab.lab_id}の評価でエラー: {e}")
                continue
        
        # ランキングの設定
        lab_results.sort(key=lambda x: x.compatibility_score.overall_score, reverse=True)
        for i, result in enumerate(lab_results):
            result.ranking = i + 1
        
        # 上位のみ保持
        lab_results = lab_results[:self.config.max_recommendations]
        
        # 統計情報の計算
        processing_time = (datetime.now() - start_time).total_seconds()
        scores = [result.compatibility_score.overall_score for result in lab_results]
        
        score_distribution = {
            "mean": float(np.mean(scores)) if scores else 0.0,
            "std": float(np.std(scores)) if scores else 0.0,
            "min": float(np.min(scores)) if scores else 0.0,
            "max": float(np.max(scores)) if scores else 0.0
        }
        
        # 推薦信頼度の計算
        recommendation_confidence = self._calculate_recommendation_confidence(lab_results)
        
        # 評価レスポンスの構築
        response = EvaluationResponse(
            student_profile=student_profile,
            lab_results=lab_results,
            processing_time=processing_time,
            algorithm_version="v1.0.0",
            total_labs_evaluated=len(target_laboratories),
            score_distribution=score_distribution,
            recommendation_confidence=recommendation_confidence,
            evaluation_id=evaluation_id,
            timestamp=start_time
        )
        
        # マッチング履歴に記録
        self._record_matching_history(student_profile, response)
        
        if lab_results:
            self.successful_matches += 1
        
        logger.info(f"適合性評価完了: {len(lab_results)}件の結果, 処理時間{processing_time:.2f}秒")
        
        return response
    
    def _calculate_compatibility(self, student_profile: StudentProfile, 
                               laboratory: Laboratory) -> CompatibilityScore:
        """適合性スコアの計算"""
        
        # ファジィ推論を使用
        if self.config.use_fuzzy_inference and self.fuzzy_engine:
            try:
                inference_result = self.fuzzy_engine.infer_lab_compatibility(
                    student_profile, laboratory
                )
                overall_score = inference_result.output_value
                confidence = inference_result.confidence
            except Exception as e:
                logger.warning(f"ファジィ推論エラー: {e}")
                # フォールバック
                inference_result = self.fallback_engine.infer_lab_compatibility(
                    student_profile, laboratory
                )
                overall_score = inference_result.output_value
                confidence = inference_result.confidence
        else:
            # 簡易計算
            inference_result = self.fallback_engine.infer_lab_compatibility(
                student_profile, laboratory
            )
            overall_score = inference_result.output_value
            confidence = inference_result.confidence
        
        # 各基準の適合性スコア計算
        criteria_scores = self._calculate_criteria_scores(student_profile, laboratory)
        
        # 分野適合性スコア
        field_match_score = self._calculate_field_match_score(student_profile, laboratory)
        
        # 最適化された重みが利用可能な場合は適用
        if self.optimized_weights:
            overall_score = self._apply_optimized_weights(
                criteria_scores, field_match_score, self.optimized_weights
            )
        
        return CompatibilityScore(
            overall_score=min(1.0, max(0.0, overall_score)),
            criteria_scores=criteria_scores,
            field_match_score=field_match_score,
            confidence=confidence
        )
    
    def _calculate_criteria_scores(self, student_profile: StudentProfile,
                                 laboratory: Laboratory) -> Dict[str, float]:
        """各評価基準の適合性スコア計算"""
        
        student_criteria = student_profile.evaluation_criteria.dict()
        lab_criteria = laboratory.characteristics.dict()
        
        criteria_scores = {}
        
        for criterion, student_value in student_criteria.items():
            if student_value is not None:
                lab_value = lab_criteria.get(criterion, 5.0)  # デフォルト値
                
                if lab_value is not None:
                    # 差分ベースの適合性計算
                    diff = abs(student_value - lab_value)
                    max_diff = 9.0  # 最大差分（1-10の範囲）
                    
                    # 近似性スコア（差が小さいほど高スコア）
                    similarity_score = max(0.0, 1.0 - diff / max_diff)
                    
                    # 基準別の重み調整
                    weight = self._get_criterion_weight(criterion)
                    criteria_scores[criterion] = similarity_score * weight
                else:
                    criteria_scores[criterion] = 0.5  # 不明な場合は中間値
            else:
                criteria_scores[criterion] = 0.5
        
        return criteria_scores
    
    def _get_criterion_weight(self, criterion: str) -> float:
        """基準別重みの取得"""
        
        basic_criteria = [
            "research_intensity", "advisor_style", "team_work", 
            "workload", "theory_practice"
        ]
        
        extended_criteria = [
            "research_field_match", "skill_development", "lab_atmosphere",
            "flexibility", "publication_opportunity"
        ]
        
        special_criteria = [
            "interdisciplinary", "communication_style", "innovation_risk"
        ]
        
        if criterion in basic_criteria:
            return self.config.basic_criteria_weight / len(basic_criteria)
        elif criterion in extended_criteria:
            return self.config.extended_criteria_weight / len(extended_criteria)
        elif criterion in special_criteria:
            return self.config.special_criteria_weight / len(special_criteria)
        else:
            return 0.1  # その他
    
    def _calculate_field_match_score(self, student_profile: StudentProfile,
                                   laboratory: Laboratory) -> float:
        """分野適合性スコアの計算"""
        
        lab_field = laboratory.research_field.value
        
        # 学生の分野興味から適合度を計算
        best_match_score = 0.0
        
        for interest in student_profile.field_interests:
            if interest.field.value == lab_field:
                # 興味レベルと優先順位を考慮
                interest_score = interest.interest_level / 10.0  # 0-1に正規化
                priority_bonus = max(0, (len(student_profile.field_interests) - interest.priority + 1) / len(student_profile.field_interests))
                
                field_score = interest_score * (1 + priority_bonus * 0.2)
                best_match_score = max(best_match_score, field_score)
        
        return min(1.0, best_match_score)
    
    def _apply_optimized_weights(self, criteria_scores: Dict[str, float],
                               field_match_score: float, 
                               weights: WeightVector) -> float:
        """最適化された重みの適用"""
        
        weight_genes = weights.get_genes()
        total_score = 0.0
        total_weight = 0.0
        
        # 各基準スコアに重みを適用
        for criterion, score in criteria_scores.items():
            weight = weight_genes.get(criterion, 0.1)
            total_score += score * weight
            total_weight += weight
        
        # 分野適合性ボーナス
        field_weight = weight_genes.get("research_field_match", self.config.field_match_bonus)
        total_score += field_match_score * field_weight
        total_weight += field_weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_explanations(self, student_profile: StudentProfile,
                             laboratory: Laboratory, 
                             compatibility_score: CompatibilityScore) -> Tuple[List[str], List[str]]:
        """推薦理由と懸念点の生成"""
        
        reasons = []
        concerns = []
        
        if not self.config.enable_explanation:
            return reasons, concerns
        
        # 高スコア基準の特定
        high_score_criteria = [
            criterion for criterion, score in compatibility_score.criteria_scores.items()
            if score > 0.7
        ]
        
        # 低スコア基準の特定
        low_score_criteria = [
            criterion for criterion, score in compatibility_score.criteria_scores.items()
            if score < 0.4
        ]
        
        # 推薦理由の生成
        if compatibility_score.field_match_score > 0.7:
            reasons.append("研究分野の興味と非常によく一致しています")
        
        if len(high_score_criteria) >= 3:
            criteria_names = self._translate_criteria_names(high_score_criteria[:3])
            reasons.append(f"{', '.join(criteria_names)}において高い適合性があります")
        
        if compatibility_score.overall_score > self.config.high_compatibility_threshold:
            reasons.append("総合的な適合度が非常に高い研究室です")
        
        # 懸念点の生成
        if len(low_score_criteria) >= 2:
            criteria_names = self._translate_criteria_names(low_score_criteria[:2])
            concerns.append(f"{', '.join(criteria_names)}において適合性が低い可能性があります")
        
        if compatibility_score.field_match_score < 0.3:
            concerns.append("研究分野の適合性が低い可能性があります")
        
        if compatibility_score.confidence < self.config.confidence_threshold:
            concerns.append("適合性の判定信頼度がやや低めです")
        
        return reasons, concerns
    
    def _translate_criteria_names(self, criteria: List[str]) -> List[str]:
        """基準名の日本語変換"""
        
        translation_map = {
            "research_intensity": "研究強度",
            "advisor_style": "指導スタイル",
            "team_work": "チームワーク",
            "workload": "作業負荷",
            "theory_practice": "理論・実践バランス",
            "research_field_match": "研究分野適合性",
            "skill_development": "スキル開発",
            "lab_atmosphere": "研究室雰囲気",
            "flexibility": "柔軟性",
            "publication_opportunity": "論文発表機会",
            "interdisciplinary": "学際性",
            "communication_style": "コミュニケーション",
            "innovation_risk": "革新性・リスク許容度"
        }
        
        return [translation_map.get(criterion, criterion) for criterion in criteria]
    
    def _calculate_recommendation_confidence(self, lab_results: List[LabResult]) -> float:
        """推薦信頼度の計算"""
        
        if not lab_results:
            return 0.0
        
        # トップ結果の信頼度
        top_confidence = lab_results[0].compatibility_score.confidence
        
        # スコア分布の分散（低いほど信頼度高）
        scores = [result.compatibility_score.overall_score for result in lab_results]
        score_std = float(np.std(scores)) if len(scores) > 1 else 0.0
        
        # 分散が小さく、トップの信頼度が高いほど全体信頼度高
        distribution_confidence = max(0.0, 1.0 - score_std * 2)
        
        return (top_confidence + distribution_confidence) / 2
    
    def _record_matching_history(self, student_profile: StudentProfile,
                               response: EvaluationResponse):
        """マッチング履歴の記録"""
        
        history_record = {
            "timestamp": response.timestamp.isoformat(),
            "student_id": student_profile.student_id,
            "evaluation_id": response.evaluation_id,
            "total_labs_evaluated": response.total_labs_evaluated,
            "top_match_score": response.lab_results[0].compatibility_score.overall_score if response.lab_results else 0.0,
            "processing_time": response.processing_time,
            "recommendation_confidence": response.recommendation_confidence
        }
        
        self.matching_history.append(history_record)
        
        # 履歴サイズ制限
        if len(self.matching_history) > 1000:
            self.matching_history = self.matching_history[-500:]
    
    def optimize_weights(self, sample_data: List[Tuple[StudentProfile, Laboratory, float]],
                        evolution_config: Optional[EvolutionConfig] = None) -> WeightVector:
        """遺伝的アルゴリズムによる重み最適化"""
        
        if not self.config.enable_genetic_optimization:
            logger.info("遺伝的最適化は無効化されています")
            return None
        
        if not sample_data:
            logger.warning("最適化用サンプルデータがありません")
            return None
        
        logger.info(f"重み最適化開始: {len(sample_data)}サンプル")
        
        # 進化設定
        if evolution_config is None:
            evolution_config = EvolutionConfig(
                population_size=20,
                max_generations=30,
                crossover_rate=0.8,
                mutation_rate=0.1,
                verbose=True
            )
        
        # 進化エンジンの初期化
        from core.genetic.evolution import EvolutionEngine
        self.genetic_engine = EvolutionEngine(evolution_config, WeightVector)
        
        # 重み名の設定
        weight_names = list(sample_data[0][0].evaluation_criteria.dict().keys())
        weight_names.append("research_field_match")
        
        self.genetic_engine.initialize_population(weight_names=weight_names)
        
        # 適応度関数の定義
        def fitness_function(individual: WeightVector) -> float:
            total_error = 0.0
            
            for student, lab, target_score in sample_data:
                try:
                    # 予測スコアの計算
                    criteria_scores = self._calculate_criteria_scores(student, lab)
                    field_match_score = self._calculate_field_match_score(student, lab)
                    predicted_score = self._apply_optimized_weights(
                        criteria_scores, field_match_score, individual
                    )
                    
                    # 誤差の計算
                    error = abs(predicted_score - target_score)
                    total_error += error
                    
                except Exception as e:
                    logger.warning(f"個体評価エラー: {e}")
                    total_error += 1.0  # ペナルティ
            
            # 適応度（誤差が小さいほど高い）
            average_error = total_error / len(sample_data)
            fitness = max(0.0, 1.0 - average_error)
            
            return fitness
        
        # 進化実行
        try:
            evolution_result = self.genetic_engine.evolve(fitness_function)
            
            if evolution_result.success:
                self.optimized_weights = evolution_result.best_individual
                logger.info(f"重み最適化完了: 適応度{evolution_result.best_fitness:.6f}")
                
                # 最適化された重みをログ出力
                optimized_genes = self.optimized_weights.get_genes()
                logger.info(f"最適化重み: {optimized_genes}")
                
                return self.optimized_weights
            else:
                logger.warning("重み最適化が収束しませんでした")
                return None
                
        except Exception as e:
            logger.error(f"重み最適化エラー: {e}")
            return None
    
    def get_service_statistics(self) -> Dict[str, Any]:
        """サービス統計情報の取得"""
        
        return {
            "total_evaluations": self.total_evaluations,
            "successful_matches": self.successful_matches,
            "success_rate": self.successful_matches / max(self.total_evaluations, 1),
            "available_laboratories": len(self.laboratories),
            "matching_history_size": len(self.matching_history),
            "optimization_enabled": self.config.enable_genetic_optimization,
            "optimized_weights_available": self.optimized_weights is not None,
            "fuzzy_inference_available": self.fuzzy_engine is not None
        }
    
    def add_laboratory(self, laboratory: Laboratory):
        """研究室の追加"""
        self.laboratories.append(laboratory)
        self.lab_cache[laboratory.lab_id] = laboratory
        logger.info(f"研究室追加: {laboratory.lab_id}")
    
    def get_laboratory(self, lab_id: str) -> Optional[Laboratory]:
        """研究室の取得"""
        return self.lab_cache.get(lab_id)
    
    def get_all_laboratories(self) -> List[Laboratory]:
        """全研究室の取得"""
        return self.laboratories.copy()

# 使用例とテスト
def test_lab_matching_service():
    """研究室マッチングサービスのテスト"""
    
    print("🔬 研究室マッチングサービステスト開始")
    
    # サービスの初期化
    config = MatchingConfig(
        enable_genetic_optimization=False,  # テスト用に無効化
        max_recommendations=5
    )
    
    service = LabMatchingService(config)
    
    # テスト用学生プロフィール
    from models.schemas import StudentProfile, EvaluationCriteria, FieldInterest
    
    test_student = StudentProfile(
        student_id="test_student_001",
        evaluation_criteria=EvaluationCriteria(
            research_intensity=8.0,
            advisor_style=7.0,
            team_work=6.0,
            workload=7.0,
            theory_practice=8.0,
            skill_development=8.0,
            lab_atmosphere=7.0
        ),
        field_interests=[
            FieldInterest(
                field=ResearchFieldEnum.AI_MACHINE_LEARNING,
                interest_level=9.0,
                priority=1
            )
        ]
    )
    
    # マッチング実行
    response = service.evaluate_student_lab_compatibility(test_student)
    
    print(f"📊 マッチング結果:")
    print(f"  評価研究室数: {response.total_labs_evaluated}")
    print(f"  推薦研究室数: {len(response.lab_results)}")
    print(f"  処理時間: {response.processing_time:.3f}秒")
    print(f"  推薦信頼度: {response.recommendation_confidence:.3f}")
    
    # トップ3の結果表示
    for i, result in enumerate(response.lab_results[:3]):
        print(f"\n  {i+1}位: {result.laboratory.faculty.name}研究室")
        print(f"    適合度: {result.compatibility_score.overall_score:.3f}")
        print(f"    分野適合: {result.compatibility_score.field_match_score:.3f}")
        print(f"    理由: {', '.join(result.reasons[:2])}")
    
    # 統計情報
    stats = service.get_service_statistics()
    print(f"\n📈 サービス統計:")
    print(f"  総評価数: {stats['total_evaluations']}")
    print(f"  成功率: {stats['success_rate']:.3f}")
    print(f"  利用可能研究室数: {stats['available_laboratories']}")
    
    print("✅ 研究室マッチングサービステスト完了")

if __name__ == "__main__":
    test_lab_matching_service()