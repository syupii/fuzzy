# services/priority_matching_service.py - 統合優先度対応マッチングサービス
"""
遺伝的アルゴリズム × ファジィ推論 × 決定木 × 優先度対応 統合マッチングサービス
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# コアエンジンのインポート（優先度対応版）
from core.fuzzy.inference import PriorityAwareFuzzyInferenceEngine
from core.genetic.evolution import PriorityAwareGeneticEvolutionEngine  
from core.decision_tree.tree import PriorityAwareFuzzyDecisionTree

logger = logging.getLogger(__name__)

@dataclass
class PriorityMatchingResult:
    """優先度マッチング結果"""
    lab_id: str
    lab_name: str
    overall_score: float
    fuzzy_score: float
    genetic_score: float
    decision_tree_score: float
    priority_weighted_score: float
    confidence: float
    explanation: str
    priority_analysis: Dict[str, Any]

class IntegratedPriorityMatchingService:
    """統合優先度対応マッチングサービス"""
    
    def __init__(self):
        # 各エンジンの初期化
        self.fuzzy_engine = PriorityAwareFuzzyInferenceEngine()
        self.genetic_engine = PriorityAwareGeneticEvolutionEngine(
            population_size=30,
            generations=20
        )
        self.decision_tree = PriorityAwareFuzzyDecisionTree(
            max_depth=8,
            min_samples_leaf=3
        )
        
        # システム統計
        self.stats = {
            'total_evaluations': 0,
            'priority_evaluations': 0,
            'average_processing_time': 0.0,
            'engine_usage': {
                'fuzzy': 0,
                'genetic': 0, 
                'decision_tree': 0,
                'integrated': 0
            }
        }
        
        # 評価基準
        self.criteria = [
            'research_intensity', 'advisor_style', 'team_work', 'workload',
            'theory_practice', 'research_field_match', 'skill_development',
            'lab_atmosphere', 'flexibility', 'publication_opportunity', 
            'interdisciplinary', 'communication_style'
        ]
        
        logger.info("統合優先度対応マッチングサービス初期化完了")
    
    def evaluate_with_priorities(
        self,
        student_profile: Dict[str, Any],
        lab_database: List[Dict[str, Any]],
        priorities: Dict[str, float],
        use_all_engines: bool = True
    ) -> List[PriorityMatchingResult]:
        """優先度を考慮した包括的マッチング評価"""
        
        start_time = time.time()
        
        try:
            # 入力検証
            self._validate_inputs(student_profile, priorities)
            
            # 各研究室との適合度評価
            results = []
            
            for lab in lab_database:
                result = self._evaluate_single_lab_with_priorities(
                    student_profile, lab, priorities, use_all_engines
                )
                results.append(result)
            
            # 結果ソート（優先度加重スコア順）
            results.sort(key=lambda x: x.priority_weighted_score, reverse=True)
            
            # 統計更新
            processing_time = time.time() - start_time
            self._update_stats(processing_time, use_priorities=True)
            
            logger.info(f"優先度マッチング評価完了: {len(results)}件、処理時間: {processing_time:.3f}秒")
            
            return results
            
        except Exception as e:
            logger.error(f"優先度マッチング評価エラー: {e}")
            raise
    
    def _evaluate_single_lab_with_priorities(
        self,
        student_profile: Dict[str, Any],
        lab: Dict[str, Any],
        priorities: Dict[str, float],
        use_all_engines: bool
    ) -> PriorityMatchingResult:
        """単一研究室の優先度評価"""
        
        scores = {}
        explanations = {}
        
        # 1. ファジィ推論による評価
        try:
            fuzzy_score, fuzzy_explanation = self.fuzzy_engine.predict_with_priorities(
                student_profile, lab, priorities
            )
            scores['fuzzy'] = fuzzy_score
            explanations['fuzzy'] = fuzzy_explanation
            self.stats['engine_usage']['fuzzy'] += 1
        except Exception as e:
            logger.warning(f"ファジィ推論エラー: {e}")
            scores['fuzzy'] = 0.5
            explanations['fuzzy'] = "ファジィ推論実行エラー"
        
        # 2. 遺伝的アルゴリズムによる評価
        genetic_score = 0.5
        genetic_explanation = "遺伝的アルゴリズム未実行"
        
        if use_all_engines:
            try:
                # 単一研究室用の簡易遺伝的評価
                genetic_score = self._evaluate_genetic_single_lab(
                    student_profile, lab, priorities
                )
                genetic_explanation = f"遺伝的アルゴリズム評価: {genetic_score:.3f}"
                self.stats['engine_usage']['genetic'] += 1
            except Exception as e:
                logger.warning(f"遺伝的アルゴリズムエラー: {e}")
        
        scores['genetic'] = genetic_score
        explanations['genetic'] = genetic_explanation
        
        # 3. 決定木による評価  
        decision_tree_score = 0.5
        decision_tree_explanation = "決定木未学習"
        
        if use_all_engines:
            try:
                # 決定木は事前学習が必要だが、簡易評価を実行
                decision_tree_score = self._evaluate_decision_tree_single_lab(
                    student_profile, lab, priorities
                )
                decision_tree_explanation = f"決定木評価: {decision_tree_score:.3f}"
                self.stats['engine_usage']['decision_tree'] += 1
            except Exception as e:
                logger.warning(f"決定木エラー: {e}")
        
        scores['decision_tree'] = decision_tree_score
        explanations['decision_tree'] = decision_tree_explanation
        
        # 4. 統合スコア計算（優先度による重み付け）
        overall_score, priority_weighted_score, confidence = self._calculate_integrated_scores(
            scores, priorities
        )
        
        # 5. 優先度分析
        priority_analysis = self._analyze_priorities(
            student_profile, lab, priorities, scores
        )
        
        # 6. 統合説明文生成
        integrated_explanation = self._generate_integrated_explanation(
            explanations, scores, priorities
        )
        
        self.stats['engine_usage']['integrated'] += 1
        
        return PriorityMatchingResult(
            lab_id=lab.get('id', 'unknown'),
            lab_name=lab.get('name', 'Unknown Lab'),
            overall_score=overall_score,
            fuzzy_score=scores['fuzzy'],
            genetic_score=scores['genetic'],
            decision_tree_score=scores['decision_tree'],
            priority_weighted_score=priority_weighted_score,
            confidence=confidence,
            explanation=integrated_explanation,
            priority_analysis=priority_analysis
        )
    
    def _evaluate_genetic_single_lab(
        self,
        student_profile: Dict[str, Any],
        lab: Dict[str, Any],
        priorities: Dict[str, float]
    ) -> float:
        """単一研究室に対する遺伝的アルゴリズム評価"""
        
        # 簡易版遺伝的評価（完全な進化は計算コストが高いため）
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for criterion in self.criteria:
            if criterion in student_profile and criterion in lab:
                student_val = float(student_profile[criterion])
                lab_val = float(lab[criterion])
                priority = priorities.get(criterion, 5.0)
                
                # 正規化
                if student_val > 1.0:
                    student_val /= 10.0
                if lab_val > 1.0:
                    lab_val /= 10.0
                
                # 遺伝的アルゴリズム風の評価
                base_match = 1.0 - abs(student_val - lab_val)
                
                # 優先度による進化的重み付け
                evolution_weight = (priority / 10.0) ** 1.5  # 非線形重み
                weighted_match = base_match * evolution_weight
                
                total_weighted_score += weighted_match
                total_weight += evolution_weight
        
        return total_weighted_score / total_weight if total_weight > 0 else 0.5
    
    def _evaluate_decision_tree_single_lab(
        self,
        student_profile: Dict[str, Any], 
        lab: Dict[str, Any],
        priorities: Dict[str, float]
    ) -> float:
        """単一研究室に対する決定木評価"""
        
        # 決定木風の段階的評価
        score = 0.5  # デフォルトスコア
        
        # 最優先項目での分岐
        top_priority_criterion = max(priorities.items(), key=lambda x: x[1])
        criterion_name, priority_value = top_priority_criterion
        
        if criterion_name in student_profile and criterion_name in lab:
            student_val = float(student_profile[criterion_name])
            lab_val = float(lab[criterion_name])
            
            # 正規化
            if student_val > 1.0:
                student_val /= 10.0
            if lab_val > 1.0:
                lab_val /= 10.0
            
            # 決定木風の閾値判定
            diff = abs(student_val - lab_val)
            
            if diff < 0.2:  # 高適合
                score = 0.8
            elif diff < 0.4:  # 中適合
                score = 0.6
            else:  # 低適合
                score = 0.3
            
            # 優先度による調整
            priority_adjustment = (priority_value / 10.0 - 0.5) * 0.2
            score = max(0.0, min(1.0, score + priority_adjustment))
        
        return score
    
    def _calculate_integrated_scores(
        self,
        scores: Dict[str, float],
        priorities: Dict[str, float]
    ) -> Tuple[float, float, float]:
        """統合スコア計算"""
        
        # 各エンジンスコアの重み付き平均
        engine_weights = {
            'fuzzy': 0.4,      # ファジィ推論 40%
            'genetic': 0.35,   # 遺伝的アルゴリズム 35%  
            'decision_tree': 0.25  # 決定木 25%
        }
        
        overall_score = sum(
            scores[engine] * weight 
            for engine, weight in engine_weights.items()
        )
        
        # 優先度による調整
        priority_factor = sum(priorities.values()) / len(priorities) / 10.0
        priority_weighted_score = overall_score * (0.7 + 0.3 * priority_factor)
        
        # 信頼度計算
        score_variance = sum(
            (scores[engine] - overall_score) ** 2 
            for engine in engine_weights.keys()
        ) / len(engine_weights)
        
        confidence = max(0.3, 1.0 - score_variance)
        
        return overall_score, priority_weighted_score, confidence
    
    def _analyze_priorities(
        self,
        student_profile: Dict[str, Any],
        lab: Dict[str, Any], 
        priorities: Dict[str, float],
        scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """優先度分析"""
        
        # 優先度上位項目の抽出
        top_priorities = sorted(
            priorities.items(), key=lambda x: x[1], reverse=True
        )[:5]
        
        # 各優先項目での適合状況
        priority_matches = {}
        for criterion, priority in top_priorities:
            if criterion in student_profile and criterion in lab:
                student_val = float(student_profile[criterion])
                lab_val = float(lab[criterion])
                
                if student_val > 1.0:
                    student_val /= 10.0
                if lab_val > 1.0:
                    lab_val /= 10.0
                
                match_score = 1.0 - abs(student_val - lab_val)
                
                priority_matches[criterion] = {
                    'priority_level': priority,
                    'match_score': match_score,
                    'student_value': student_val,
                    'lab_value': lab_val,
                    'impact': match_score * (priority / 10.0)
                }
        
        # 総合優先度効果
        total_priority_impact = sum(
            data['impact'] for data in priority_matches.values()
        )
        
        return {
            'top_priority_matches': priority_matches,
            'total_priority_impact': total_priority_impact,
            'priority_alignment': total_priority_impact / len(top_priorities) if top_priorities else 0,
            'high_priority_conflicts': [
                criterion for criterion, data in priority_matches.items()
                if data['priority_level'] >= 8 and data['match_score'] < 0.4
            ]
        }
    
    def _generate_integrated_explanation(
        self,
        explanations: Dict[str, str],
        scores: Dict[str, float],
        priorities: Dict[str, float]
    ) -> str:
        """統合説明文生成"""
        
        # スコア要約
        score_summary = f"統合スコア: ファジィ{scores['fuzzy']:.2f}, " \
                       f"遺伝的{scores['genetic']:.2f}, 決定木{scores['decision_tree']:.2f}"
        
        # 優先度要約
        top_priority = max(priorities.items(), key=lambda x: x[1])
        priority_summary = f"最重要項目: {top_priority[0]}({top_priority[1]}/10)"
        
        return f"{score_summary}. {priority_summary}による重み付け評価実施。"
    
    def _validate_inputs(
        self,
        student_profile: Dict[str, Any],
        priorities: Dict[str, float]
    ):
        """入力検証"""
        
        # 必須項目チェック
        missing_criteria = [
            criterion for criterion in self.criteria
            if criterion not in student_profile
        ]
        
        if missing_criteria:
            raise ValueError(f"必須評価基準が不足: {missing_criteria}")
        
        # 優先度範囲チェック
        invalid_priorities = [
            criterion for criterion, priority in priorities.items()
            if not (1 <= priority <= 10)
        ]
        
        if invalid_priorities:
            raise ValueError(f"優先度が範囲外(1-10): {invalid_priorities}")
    
    def _update_stats(self, processing_time: float, use_priorities: bool = False):
        """統計更新"""
        
        self.stats['total_evaluations'] += 1
        
        if use_priorities:
            self.stats['priority_evaluations'] += 1
        
        # 移動平均で処理時間更新
        alpha = 0.1
        self.stats['average_processing_time'] = (
            alpha * processing_time + 
            (1 - alpha) * self.stats['average_processing_time']
        )
    
    def get_service_stats(self) -> Dict[str, Any]:
        """サービス統計取得"""
        
        return {
            'service_info': {
                'name': 'IntegratedPriorityMatchingService',
                'version': '1.0.0',
                'criteria_count': len(self.criteria),
                'engines': ['fuzzy', 'genetic', 'decision_tree']
            },
            'performance_stats': self.stats,
            'priority_support': {
                'enabled': True,
                'criteria_supported': self.criteria,
                'priority_range': '1-10',
                'weighting_algorithm': 'adaptive_priority_weighting'
            }
        }
    
    def optimize_lab_assignments(
        self,
        student_profiles: List[Dict[str, Any]],
        priorities_list: List[Dict[str, float]],
        lab_database: List[Dict[str, Any]],
        max_students_per_lab: int = 10
    ) -> Dict[str, Any]:
        """研究室配属最適化（優先度考慮）"""
        
        start_time = time.time()
        
        try:
            # 遺伝的アルゴリズムによる配属最適化
            optimization_result = self.genetic_engine.evolve_with_priorities(
                student_profiles, lab_database, priorities_list
            )
            
            # 配属結果の後処理
            assignments = self._process_assignment_results(
                optimization_result, max_students_per_lab
            )
            
            processing_time = time.time() - start_time
            
            return {
                'optimization_completed': True,
                'processing_time': processing_time,
                'assignments': assignments,
                'optimization_details': optimization_result,
                'constraints': {
                    'max_students_per_lab': max_students_per_lab,
                    'priority_weighting': True
                }
            }
            
        except Exception as e:
            logger.error(f"配属最適化エラー: {e}")
            raise
    
    def _process_assignment_results(
        self,
        optimization_result: Dict[str, Any],
        max_students_per_lab: int
    ) -> List[Dict[str, Any]]:
        """配属結果の後処理"""
        
        assignments = []
        lab_capacities = {}
        
        for student_result in optimization_result.get('evolution_results', []):
            student_id = student_result['student_id']
            optimized_matches = student_result.get('optimized_matches', [])
            
            # 配属可能な研究室を探索
            assigned = False
            for match in optimized_matches:
                lab_id = match['lab_id']
                current_count = lab_capacities.get(lab_id, 0)
                
                if current_count < max_students_per_lab:
                    assignments.append({
                        'student_id': student_id,
                        'assigned_lab_id': lab_id,
                        'assignment_score': match['optimized_score'],
                        'ranking_position': len([m for m in optimized_matches if m['optimized_score'] > match['optimized_score']]) + 1
                    })
                    
                    lab_capacities[lab_id] = current_count + 1
                    assigned = True
                    break
            
            if not assigned:
                # 配属できない場合
                assignments.append({
                    'student_id': student_id,
                    'assigned_lab_id': None,
                    'assignment_score': 0.0,
                    'status': 'unassigned'
                })
        
        return assignments


# 使用例とテスト用のヘルパー関数
def create_sample_priorities() -> Dict[str, float]:
    """サンプル優先度データ作成"""
    
    return {
        'research_intensity': 9,        # 研究集中度を最重視
        'research_field_match': 10,     # 分野適合を最重視
        'publication_opportunity': 8,   # 論文機会を重視
        'advisor_style': 6,            # 指導スタイルは中程度
        'team_work': 7,                # チームワークをやや重視
        'workload': 5,                 # 負荷は標準
        'theory_practice': 6,          # バランスは中程度
        'skill_development': 7,        # スキル開発をやや重視
        'lab_atmosphere': 5,           # 雰囲気は標準
        'flexibility': 6,              # 柔軟性は中程度
        'interdisciplinary': 4,        # 学際性は低優先
        'communication_style': 5       # コミュニケーションは標準
    }

def demonstrate_priority_matching():
    """優先度マッチングのデモンストレーション"""
    
    # サービス初期化
    service = IntegratedPriorityMatchingService()
    
    # サンプルデータ
    student_profile = {
        'research_intensity': 8,
        'advisor_style': 6,
        'team_work': 7,
        'workload': 6,
        'theory_practice': 7,
        'research_field_match': 9,
        'skill_development': 8,
        'lab_atmosphere': 7,
        'flexibility': 8,
        'publication_opportunity': 9,
        'interdisciplinary': 6,
        'communication_style': 7
    }
    
    priorities = create_sample_priorities()
    
    # サンプル研究室データ
    lab_database = [
        {
            'id': 'lab_ai_01',
            'name': 'AI研究室',
            'research_intensity': 9,
            'advisor_style': 7,
            'team_work': 8,
            'workload': 8,
            'theory_practice': 6,
            'research_field_match': 9,
            'skill_development': 8,
            'lab_atmosphere': 8,
            'flexibility': 6,
            'publication_opportunity': 9,
            'interdisciplinary': 7,
            'communication_style': 7
        },
        {
            'id': 'lab_web_01', 
            'name': 'Web開発研究室',
            'research_intensity': 6,
            'advisor_style': 8,
            'team_work': 9,
            'workload': 7,
            'theory_practice': 8,
            'research_field_match': 6,
            'skill_development': 9,
            'lab_atmosphere': 9,
            'flexibility': 8,
            'publication_opportunity': 6,
            'interdisciplinary': 8,
            'communication_style': 9
        }
    ]
    
    # 評価実行
    results = service.evaluate_with_priorities(
        student_profile, lab_database, priorities
    )
    
    # 結果出力
    print("=== 優先度対応マッチング結果 ===")
    for i, result in enumerate(results, 1):
        print(f"\n{i}位: {result.lab_name}")
        print(f"  統合スコア: {result.overall_score:.3f}")
        print(f"  優先度加重スコア: {result.priority_weighted_score:.3f}")
        print(f"  信頼度: {result.confidence:.3f}")
        print(f"  説明: {result.explanation}")
        print(f"  優先度分析: {result.priority_analysis['priority_alignment']:.3f}")
    
    # サービス統計
    stats = service.get_service_stats()
    print(f"\n=== サービス統計 ===")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    demonstrate_priority_matching()