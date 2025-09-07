# services/lab_matching.py - 研究室マッチングサービス

from typing import List, Dict, Tuple
import numpy as np

from models.schemas import (
    StudentProfile, Laboratory, LabResult, CompatibilityScore, 
    EvaluationSummary, EvaluationResponse
)
from core.fuzzy.membership import MembershipFunction
from core.genetic.evolution import GeneticAlgorithm, Individual
from config.settings import settings

class LabMatchingService:
    """研究室マッチングメインサービス"""
    
    def __init__(self):
        self.membership_func = MembershipFunction()
        self.ga_config = {
            "population_size": settings.ga_population_size,
            "generations": settings.ga_generations,
            "mutation_rate": settings.ga_mutation_rate,
            "crossover_rate": settings.ga_crossover_rate,
            "elite_size": 5
        }
        self.genetic_algorithm = GeneticAlgorithm(self.ga_config)
    
    def find_best_matches(self, student_profile: StudentProfile) -> EvaluationResponse:
        """最適な研究室マッチングを実行"""
        
        print("🎯 研究室マッチング開始")
        print(f"📊 学生プロフィール: {len(student_profile.field_interests)}分野選択")
        
        # サンプル研究室データを生成（実際のデータが来るまで）
        labs = self._generate_sample_labs()
        print(f"🏫 対象研究室数: {len(labs)}")
        
        # 遺伝的アルゴリズムで最適重みを探索
        print("🧬 遺伝的アルゴリズムによる最適化...")
        best_individual = self.genetic_algorithm.evolve(
            student_profile, labs, settings.research_fields, settings.evaluation_criteria
        )
        
        # 各研究室のマッチングスコア計算
        print("📈 マッチングスコア計算...")
        lab_results = []
        
        for lab in labs:
            compatibility_score = self._calculate_compatibility_score(
                student_profile, lab, best_individual
            )
            
            recommendations = self._generate_recommendations(
                student_profile, lab, compatibility_score
            )
            
            lab_result = LabResult(
                lab=lab,
                compatibility=compatibility_score,
                ranking_position=0,  # 後でソート後に設定
                recommendations=recommendations
            )
            lab_results.append(lab_result)
        
        # スコア順でソート
        lab_results.sort(key=lambda x: x.compatibility.overall_score, reverse=True)
        
        # ランキング位置を設定
        for i, result in enumerate(lab_results):
            result.ranking_position = i + 1
        
        # 評価サマリー生成
        summary = self._generate_summary(student_profile, lab_results)
        
        # 最適化情報
        optimization_info = {
            "final_fitness": best_individual.fitness,
            "generations": settings.ga_generations,
            "population_size": settings.ga_population_size,
            "confidence": self.membership_func.calculate_confidence(student_profile.field_interests)
        }
        
        print("✅ マッチング完了")
        
        return EvaluationResponse(
            results=lab_results,
            summary=summary,
            optimization_info=optimization_info
        )
    
    def _calculate_compatibility_score(self, student: StudentProfile, 
                                     lab: Laboratory, individual: Individual) -> CompatibilityScore:
        """詳細な適合性スコアを計算"""
        
        # 分野適合性計算
        field_compatibility = self._calculate_field_compatibility(student, lab, individual)
        
        # 評価基準適合性計算
        criteria_compatibility = self._calculate_criteria_compatibility(student, lab, individual)
        
        # 詳細スコア計算
        detailed_scores = {}
        
        # 分野別スコア
        student_fields = {fi.field_id: fi for fi in student.field_interests}
        for field_id in lab.research_fields:
            if field_id in student_fields:
                field_info = settings.research_fields.get(field_id, {})
                student_interest = student_fields[field_id]
                
                field_score = self.membership_func.field_compatibility(
                    student_interest, field_info
                )
                detailed_scores[f"field_{field_id}"] = field_score
        
        # 評価基準別スコア
        student_criteria = student.evaluation_criteria.dict()
        lab_features = lab.features.dict()
        
        for criterion in settings.evaluation_criteria:
            if criterion in student_criteria and criterion in lab_features:
                similarity = self.membership_func.criteria_similarity(
                    student_criteria[criterion], lab_features[criterion]
                )
                detailed_scores[f"criteria_{criterion}"] = similarity
        
        # 総合スコア計算（10点満点）
        overall_score = (field_compatibility * 0.6 + criteria_compatibility * 0.4) * 10
        
        return CompatibilityScore(
            overall_score=round(overall_score, 2),
            field_compatibility=round(field_compatibility, 3),
            criteria_compatibility=round(criteria_compatibility, 3),
            detailed_scores=detailed_scores
        )
    
    def _calculate_field_compatibility(self, student: StudentProfile, 
                                     lab: Laboratory, individual: Individual) -> float:
        """分野適合性を計算"""
        
        total_score = 0.0
        total_weight = 0.0
        
        student_fields = {fi.field_id: fi for fi in student.field_interests}
        
        for field_id in lab.research_fields:
            if field_id in student_fields:
                student_interest = student_fields[field_id]
                field_info = settings.research_fields.get(field_id, {})
                
                # ファジィメンバーシップによる適合性
                compatibility = self.membership_func.field_compatibility(
                    student_interest, field_info
                )
                
                # 個体の分野重み
                weight = individual.field_weights.get(field_id, 0.0)
                
                total_score += compatibility * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_criteria_compatibility(self, student: StudentProfile,
                                        lab: Laboratory, individual: Individual) -> float:
        """評価基準適合性を計算"""
        
        total_score = 0.0
        total_weight = 0.0
        
        student_criteria = student.evaluation_criteria.dict()
        lab_features = lab.features.dict()
        
        for criterion in settings.evaluation_criteria:
            if criterion in student_criteria and criterion in lab_features:
                similarity = self.membership_func.criteria_similarity(
                    student_criteria[criterion], lab_features[criterion]
                )
                
                # 個体の基準重み
                weight = individual.criteria_weights.get(criterion, 0.0)
                
                total_score += similarity * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _generate_recommendations(self, student: StudentProfile, lab: Laboratory,
                                compatibility: CompatibilityScore) -> List[str]:
        """推奨事項を生成"""
        
        recommendations = []
        
        # スコアに基づく基本推奨
        if compatibility.overall_score >= 8.0:
            recommendations.append("非常に高い適合性を示しています。積極的に検討することをお勧めします。")
        elif compatibility.overall_score >= 6.5:
            recommendations.append("良好な適合性があります。詳細を確認してみてください。")
        elif compatibility.overall_score >= 5.0:
            recommendations.append("中程度の適合性です。他の選択肢と比較検討してください。")
        else:
            recommendations.append("適合性が低い可能性があります。慎重に検討してください。")
        
        # 分野適合性に基づく推奨
        if compatibility.field_compatibility >= 0.8:
            recommendations.append("研究分野の興味と非常によく一致しています。")
        elif compatibility.field_compatibility < 0.5:
            recommendations.append("研究分野の適合性を再確認することをお勧めします。")
        
        # 評価基準に基づく推奨
        if compatibility.criteria_compatibility >= 0.8:
            recommendations.append("研究環境や指導スタイルがあなたの希望によく合っています。")
        elif compatibility.criteria_compatibility < 0.5:
            recommendations.append("研究環境について詳しく確認することをお勧めします。")
        
        # 具体的なアドバイス
        student_fields = {fi.field_id: fi for fi in student.field_interests}
        
        # 経験レベルが低い分野への対応
        low_experience_fields = [
            fi for fi in student.field_interests 
            if fi.experience_level <= 3 and fi.interest_level >= 7
        ]
        
        if low_experience_fields and any(lef.field_id in lab.research_fields for lef in low_experience_fields):
            recommendations.append("興味はあるが経験の少ない分野があります。基礎から学べる環境かを確認してください。")
        
        return recommendations
    
    def _generate_summary(self, student: StudentProfile, 
                         lab_results: List[LabResult]) -> EvaluationSummary:
        """評価サマリーを生成"""
        
        if not lab_results:
            return EvaluationSummary(
                total_labs=0,
                avg_compatibility=0.0,
                best_match_score=0.0,
                selected_fields_count=0,
                field_distribution={}
            )
        
        # 基本統計
        scores = [result.compatibility.overall_score for result in lab_results]
        avg_compatibility = np.mean(scores)
        best_match_score = max(scores)
        
        # 分野分布
        field_distribution = {}
        selected_field_ids = [fi.field_id for fi in student.field_interests]
        
        for field_id in selected_field_ids:
            field_info = settings.research_fields.get(field_id, {})
            category = field_info.get("category", "その他")
            field_distribution[category] = field_distribution.get(category, 0) + 1
        
        return EvaluationSummary(
            total_labs=len(lab_results),
            avg_compatibility=round(avg_compatibility, 2),
            best_match_score=round(best_match_score, 2),
            selected_fields_count=len(student.field_interests),
            field_distribution=field_distribution
        )
    
    def _generate_sample_labs(self) -> List[Laboratory]:
        """サンプル研究室データを生成（実際のデータが来るまで）"""
        
        sample_labs = []
        
        # 主要分野の代表的な研究室
        lab_templates = [
            {
                "id": "lab_ai_001",
                "name": "人工知能研究室",
                "professor": "伊藤正彦",
                "research_area": "人工知能・機械学習",
                "fields": ["ai_machine_learning", "image_computer_vision"],
                "base_features": {"research_intensity": 8.5, "theory_practice": 6.5, "innovation_risk": 8.0}
            },
            {
                "id": "lab_game_001",
                "name": "ゲーム開発研究室",
                "professor": "森川悟",
                "research_area": "ゲーム開発",
                "fields": ["game_programming"],
                "base_features": {"research_intensity": 7.0, "team_work": 9.0, "creativity_focus": 8.5}
            },
            {
                "id": "lab_web_001",
                "name": "Webデザイン研究室",
                "professor": "杉澤愛美",
                "research_area": "Webデザイン・UI/UX",
                "fields": ["web_design_branding", "ux_ui_design_thinking"],
                "base_features": {"flexibility": 9.0, "lab_atmosphere": 8.5, "theory_practice": 8.0}
            },
            {
                "id": "lab_vr_001",
                "name": "VR/AR研究室",
                "professor": "向田茂",
                "research_area": "VR/AR・メディアアート",
                "fields": ["vr_ar_media_architecture", "image_computer_vision"],
                "base_features": {"innovation_risk": 9.0, "interdisciplinary": 8.0, "tech_focus": 8.5}
            },
            {
                "id": "lab_security_001",
                "name": "ネットワークセキュリティ研究室",
                "professor": "佐々木洋平",
                "research_area": "ネットワーク・セキュリティ",
                "fields": ["network_security"],
                "base_features": {"research_intensity": 7.5, "advisor_style": 6.0, "theory_practice": 7.0}
            }
        ]
        
        for template in lab_templates:
            # 基本特徴量を設定
            features = {}
            for criterion in settings.evaluation_criteria:
                if criterion in template["base_features"]:
                    features[criterion] = template["base_features"][criterion]
                else:
                    # デフォルト値（6.0-7.5の範囲でランダム）
                    features[criterion] = round(np.random.uniform(6.0, 7.5), 1)
            
            from models.schemas import LabFeatures
            lab_features = LabFeatures(**features)
            
            lab = Laboratory(
                id=template["id"],
                name=template["name"],
                professor=template["professor"],
                research_area=template["research_area"],
                specialization=f"{template['research_area']}の研究",
                research_fields=template["fields"],
                description=f"{template['professor']}教授による{template['research_area']}の研究室",
                features=lab_features
            )
            
            sample_labs.append(lab)
        
        return sample_labs