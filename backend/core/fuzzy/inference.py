# core/fuzzy/inference.py - 優先度対応版
"""
優先度対応ファジィ推論エンジン
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

class PriorityAwareFuzzyInferenceEngine:
    """優先度を考慮したファジィ推論エンジン"""
    
    def __init__(self):
        self.membership_functions = {}
        self.rules = []
        self.criteria = [
            'research_intensity', 'advisor_style', 'team_work', 'workload', 
            'theory_practice', 'research_field_match', 'skill_development',
            'lab_atmosphere', 'flexibility', 'publication_opportunity',
            'interdisciplinary', 'communication_style'
        ]
        self._initialize_membership_functions()
        self._initialize_rules()
    
    def _initialize_membership_functions(self):
        """メンバーシップ関数の初期化"""
        
        for criterion in self.criteria:
            self.membership_functions[criterion] = {
                'low': self._triangular_membership_function(0, 0, 4),
                'medium': self._triangular_membership_function(2, 5, 8),
                'high': self._triangular_membership_function(6, 10, 10)
            }
    
    def _triangular_membership_function(self, a: float, b: float, c: float):
        """三角形メンバーシップ関数"""
        def membership(x: float) -> float:
            if x <= a or x >= c:
                return 0.0
            elif a < x <= b:
                return (x - a) / (b - a)
            else:  # b < x < c
                return (c - x) / (c - b)
        return membership
    
    def _initialize_rules(self):
        """ファジィルールの初期化"""
        
        # 基本的なファジィルール
        self.rules = [
            # 高い適合性のルール
            {
                'conditions': ['high', 'high', 'high', 'medium', 'high'],
                'conclusion': 'very_high',
                'weight': 0.9
            },
            # 中程度の適合性のルール  
            {
                'conditions': ['medium', 'medium', 'medium', 'medium', 'medium'],
                'conclusion': 'medium',
                'weight': 0.7
            },
            # 低い適合性のルール
            {
                'conditions': ['low', 'low', 'low', 'high', 'low'],
                'conclusion': 'low',
                'weight': 0.3
            }
        ]
    
    def predict_with_priorities(
        self, 
        student_profile: Dict[str, Any],
        lab_profile: Dict[str, Any],
        priorities: Dict[str, float]
    ) -> Tuple[float, str]:
        """優先度を考慮したファジィ推論予測"""
        
        # 各基準の適合度計算
        criterion_matches = {}
        total_weighted_score = 0.0
        total_priority_weight = 0.0
        
        for criterion in self.criteria:
            if criterion in student_profile and criterion in lab_profile:
                student_val = float(student_profile[criterion])
                lab_val = float(lab_profile[criterion])
                priority = priorities.get(criterion, 5.0)
                
                # 正規化
                if student_val > 1.0:
                    student_val /= 10.0
                if lab_val > 1.0:
                    lab_val /= 10.0
                
                # ファジィマッチング計算
                fuzzy_match = self._calculate_fuzzy_matching(
                    student_val, lab_val, criterion
                )
                
                # 優先度による重み付け
                priority_weight = priority / 10.0  # 0.1 to 1.0
                weighted_match = fuzzy_match * priority_weight
                
                criterion_matches[criterion] = {
                    'match': fuzzy_match,
                    'priority': priority,
                    'weighted_match': weighted_match
                }
                
                total_weighted_score += weighted_match
                total_priority_weight += priority_weight
        
        # 最終スコア計算
        if total_priority_weight > 0:
            final_score = total_weighted_score / total_priority_weight
        else:
            final_score = 0.0
        
        # ファジィルールベース推論の適用
        rule_adjusted_score = self._apply_fuzzy_rules_with_priorities(
            criterion_matches, final_score, priorities
        )
        
        # 説明文生成
        explanation = self._generate_fuzzy_explanation(
            criterion_matches, rule_adjusted_score, priorities
        )
        
        return rule_adjusted_score, explanation
    
    def _calculate_fuzzy_matching(
        self, 
        student_val: float, 
        lab_val: float, 
        criterion: str
    ) -> float:
        """ファジィマッチング計算"""
        
        # 差分計算
        difference = abs(student_val - lab_val)
        
        # ファジィ類似度計算（ガウシアン関数ベース）
        sigma = 0.3  # 標準偏差
        fuzzy_similarity = math.exp(-(difference ** 2) / (2 * sigma ** 2))
        
        return fuzzy_similarity
    
    def _apply_fuzzy_rules_with_priorities(
        self, 
        criterion_matches: Dict, 
        base_score: float, 
        priorities: Dict[str, float]
    ) -> float:
        """優先度を考慮したファジィルールの適用"""
        
        # 高優先度項目の影響を強化
        high_priority_criteria = [
            k for k, v in priorities.items() if v >= 8.0
        ]
        
        adjustment = 0.0
        if high_priority_criteria:
            high_priority_matches = [
                criterion_matches.get(criterion, {}).get('match', 0.5)
                for criterion in high_priority_criteria
                if criterion in criterion_matches
            ]
            
            if high_priority_matches:
                avg_high_priority_match = sum(high_priority_matches) / len(high_priority_matches)
                # 高優先度項目が高適合の場合はボーナス
                if avg_high_priority_match > 0.7:
                    adjustment += 0.1 * (avg_high_priority_match - 0.7)
                # 高優先度項目が低適合の場合はペナルティ
                elif avg_high_priority_match < 0.3:
                    adjustment -= 0.1 * (0.3 - avg_high_priority_match)
        
        final_score = min(1.0, max(0.0, base_score + adjustment))
        return final_score
    
    def _generate_fuzzy_explanation(
        self, 
        criterion_matches: Dict, 
        final_score: float, 
        priorities: Dict[str, float]
    ) -> str:
        """ファジィ推論の説明文生成"""
        
        # 上位優先度項目を特定
        top_priorities = sorted(
            priorities.items(), key=lambda x: x[1], reverse=True
        )[:3]
        
        explanations = []
        for criterion, priority in top_priorities:
            if criterion in criterion_matches:
                match_data = criterion_matches[criterion]
                match_score = match_data['match']
                
                if match_score > 0.8:
                    explanations.append(f"重要な{criterion}で高適合")
                elif match_score > 0.6:
                    explanations.append(f"{criterion}で適度な適合")
                else:
                    explanations.append(f"重要な{criterion}で適合に課題")
        
        base_text = f"ファジィ推論スコア: {final_score:.2f}"
        detail_text = "、".join(explanations) if explanations else "標準評価"
        
        return f"{base_text}（{detail_text}）"


# core/genetic/evolution.py - 優先度対応版
"""
優先度対応遺伝的アルゴリズム進化エンジン
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Any

class PriorityAwareGeneticEvolutionEngine:
    """優先度を考慮した遺伝的アルゴリズム進化エンジン"""
    
    def __init__(
        self, 
        population_size: int = 50,
        generations: int = 30,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.criteria = [
            'research_intensity', 'advisor_style', 'team_work', 'workload',
            'theory_practice', 'research_field_match', 'skill_development',
            'lab_atmosphere', 'flexibility', 'publication_opportunity',
            'interdisciplinary', 'communication_style'
        ]
    
    def evolve_with_priorities(
        self,
        student_profiles: List[Dict[str, Any]],
        lab_database: List[Dict[str, Any]], 
        priorities_list: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """優先度を考慮した遺伝的アルゴリズム進化"""
        
        evolution_results = []
        
        for i, (student_profile, priorities) in enumerate(zip(student_profiles, priorities_list)):
            # 個体初期化（重みベクトル）
            population = self._initialize_priority_aware_population(priorities)
            
            best_fitness_history = []
            best_individual = None
            best_fitness = 0.0
            
            # 世代進化
            for generation in range(self.generations):
                # 適応度計算（優先度考慮）
                fitness_scores = []
                for individual in population:
                    fitness = self._calculate_priority_aware_fitness(
                        individual, student_profile, lab_database, priorities
                    )
                    fitness_scores.append(fitness)
                    
                    if fitness > best_fitness:
                        best_fitness = fitness
                        best_individual = individual.copy()
                
                best_fitness_history.append(best_fitness)
                
                # 選択・交叉・突然変異
                new_population = self._evolve_generation_with_priorities(
                    population, fitness_scores, priorities
                )
                population = new_population
            
            # 最適化結果
            optimized_matches = self._generate_optimized_matches(
                best_individual, student_profile, lab_database, priorities
            )
            
            evolution_results.append({
                'student_id': i,
                'best_individual': best_individual,
                'best_fitness': best_fitness,
                'fitness_history': best_fitness_history,
                'optimized_matches': optimized_matches,
                'priorities_applied': priorities
            })
        
        return {
            'evolution_completed': True,
            'students_processed': len(student_profiles),
            'evolution_results': evolution_results,
            'algorithm_parameters': {
                'population_size': self.population_size,
                'generations': self.generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate
            }
        }
    
    def _initialize_priority_aware_population(
        self, 
        priorities: Dict[str, float]
    ) -> List[List[float]]:
        """優先度を考慮した初期集団生成"""
        
        population = []
        
        for _ in range(self.population_size):
            individual = []
            for criterion in self.criteria:
                priority = priorities.get(criterion, 5.0)
                # 優先度に基づいた重み初期化（優先度が高い項目ほど大きな重み）
                base_weight = priority / 10.0
                noise = random.uniform(-0.2, 0.2)
                weight = max(0.1, min(1.0, base_weight + noise))
                individual.append(weight)
            
            population.append(individual)
        
        return population
    
    def _calculate_priority_aware_fitness(
        self,
        individual: List[float],
        student_profile: Dict[str, Any],
        lab_database: List[Dict[str, Any]],
        priorities: Dict[str, float]
    ) -> float:
        """優先度を考慮した適応度計算"""
        
        total_fitness = 0.0
        
        for lab in lab_database:
            lab_fitness = 0.0
            total_weight = 0.0
            
            for i, criterion in enumerate(self.criteria):
                if criterion in student_profile and criterion in lab:
                    student_val = float(student_profile[criterion])
                    lab_val = float(lab[criterion])
                    priority = priorities.get(criterion, 5.0)
                    
                    # 正規化
                    if student_val > 1.0:
                        student_val /= 10.0
                    if lab_val > 1.0:
                        lab_val /= 10.0
                    
                    # 適合度計算
                    match = 1.0 - abs(student_val - lab_val)
                    
                    # 個体の重みと優先度を組み合わせ
                    combined_weight = individual[i] * (priority / 10.0)
                    weighted_match = match * combined_weight
                    
                    lab_fitness += weighted_match
                    total_weight += combined_weight
            
            # 正規化された研究室適合度
            if total_weight > 0:
                normalized_lab_fitness = lab_fitness / total_weight
            else:
                normalized_lab_fitness = 0.0
            
            total_fitness += normalized_lab_fitness
        
        # 平均適応度
        return total_fitness / len(lab_database) if lab_database else 0.0
    
    def _evolve_generation_with_priorities(
        self,
        population: List[List[float]],
        fitness_scores: List[float],
        priorities: Dict[str, float]
    ) -> List[List[float]]:
        """優先度を考慮した世代進化"""
        
        new_population = []
        
        # エリート選択（上位20%を保持）
        elite_count = max(1, int(0.2 * self.population_size))
        sorted_indices = sorted(
            range(len(fitness_scores)), 
            key=lambda i: fitness_scores[i], 
            reverse=True
        )
        
        for i in range(elite_count):
            new_population.append(population[sorted_indices[i]].copy())
        
        # 残りを交叉・突然変異で生成
        while len(new_population) < self.population_size:
            # トーナメント選択
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # 交叉
            if random.random() < self.crossover_rate:
                child1, child2 = self._priority_aware_crossover(
                    parent1, parent2, priorities
                )
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # 突然変異
            if random.random() < self.mutation_rate:
                child1 = self._priority_aware_mutation(child1, priorities)
            if random.random() < self.mutation_rate:
                child2 = self._priority_aware_mutation(child2, priorities)
            
            new_population.extend([child1, child2])
        
        return new_population[:self.population_size]
    
    def _tournament_selection(
        self, 
        population: List[List[float]], 
        fitness_scores: List[float]
    ) -> List[float]:
        """トーナメント選択"""
        
        tournament_size = 3
        tournament_indices = random.sample(
            range(len(population)), 
            min(tournament_size, len(population))
        )
        
        best_index = max(
            tournament_indices, 
            key=lambda i: fitness_scores[i]
        )
        
        return population[best_index].copy()
    
    def _priority_aware_crossover(
        self,
        parent1: List[float],
        parent2: List[float], 
        priorities: Dict[str, float]
    ) -> Tuple[List[float], List[float]]:
        """優先度を考慮した交叉"""
        
        child1, child2 = [], []
        
        for i, criterion in enumerate(self.criteria):
            priority = priorities.get(criterion, 5.0)
            priority_influence = priority / 10.0
            
            # 優先度が高い項目では親の特徴をより強く継承
            if random.random() < 0.5 + 0.3 * priority_influence:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        
        return child1, child2
    
    def _priority_aware_mutation(
        self, 
        individual: List[float], 
        priorities: Dict[str, float]
    ) -> List[float]:
        """優先度を考慮した突然変異"""
        
        mutated = individual.copy()
        
        for i, criterion in enumerate(self.criteria):
            priority = priorities.get(criterion, 5.0)
            
            # 優先度が低い項目ほど大きく変異
            mutation_strength = 0.3 * (1.0 - priority / 10.0)
            
            if random.random() < 0.1:  # 10%の確率で突然変異
                noise = random.uniform(-mutation_strength, mutation_strength)
                mutated[i] = max(0.1, min(1.0, mutated[i] + noise))
        
        return mutated
    
    def _generate_optimized_matches(
        self,
        best_individual: List[float],
        student_profile: Dict[str, Any],
        lab_database: List[Dict[str, Any]],
        priorities: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """最適化された適合結果生成"""
        
        matches = []
        
        for lab in lab_database:
            match_score = 0.0
            total_weight = 0.0
            
            for i, criterion in enumerate(self.criteria):
                if criterion in student_profile and criterion in lab:
                    student_val = float(student_profile[criterion])
                    lab_val = float(lab[criterion])
                    priority = priorities.get(criterion, 5.0)
                    
                    # 正規化
                    if student_val > 1.0:
                        student_val /= 10.0
                    if lab_val > 1.0:
                        lab_val /= 10.0
                    
                    # 適合度計算
                    match = 1.0 - abs(student_val - lab_val)
                    
                    # 最適化された重みを適用
                    optimized_weight = best_individual[i] * (priority / 10.0)
                    weighted_match = match * optimized_weight
                    
                    match_score += weighted_match
                    total_weight += optimized_weight
            
            # 正規化
            if total_weight > 0:
                final_score = match_score / total_weight
            else:
                final_score = 0.0
            
            matches.append({
                'lab_id': lab['id'],
                'lab_name': lab['name'],
                'optimized_score': final_score,
                'lab': lab
            })
        
        # スコア順でソート
        matches.sort(key=lambda x: x['optimized_score'], reverse=True)
        
        return matches


# core/decision_tree/tree.py - 優先度対応版  
"""
優先度対応ファジィ決定木
"""

import math
import random
from typing import Dict, List, Any, Optional, Tuple

class PriorityAwareFuzzyDecisionTree:
    """優先度を考慮したファジィ決定木"""
    
    def __init__(self, max_depth: int = 6, min_samples_leaf: int = 5):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = None
        self.criteria = [
            'research_intensity', 'advisor_style', 'team_work', 'workload',
            'theory_practice', 'research_field_match', 'skill_development', 
            'lab_atmosphere', 'flexibility', 'publication_opportunity',
            'interdisciplinary', 'communication_style'
        ]
        self.feature_importances = {}
    
    def fit_with_priorities(
        self,
        training_data: List[Dict[str, Any]],
        priorities: Dict[str, float]
    ):
        """優先度を考慮した決定木学習"""
        
        # 優先度に基づく特徴重要度計算
        self.feature_importances = self._calculate_priority_based_importance(priorities)
        
        # 決定木構築
        self.root = self._build_tree_with_priorities(
            training_data, priorities, depth=0
        )
    
    def predict_with_priorities(
        self,
        student_profile: Dict[str, Any],
        priorities: Dict[str, float]
    ) -> Tuple[float, str]:
        """優先度を考慮した予測"""
        
        if self.root is None:
            return 0.5, "決定木が学習されていません"
        
        # 決定木を使った予測
        prediction, path = self._traverse_tree_with_priorities(
            self.root, student_profile, priorities
        )
        
        # 説明文生成
        explanation = self._generate_decision_explanation(path, priorities)
        
        return prediction, explanation
    
    def _calculate_priority_based_importance(
        self, 
        priorities: Dict[str, float]
    ) -> Dict[str, float]:
        """優先度に基づく特徴重要度計算"""
        
        importances = {}
        total_priority = sum(priorities.values())
        
        for criterion in self.criteria:
            priority = priorities.get(criterion, 5.0)
            # 優先度を正規化して重要度とする
            importance = priority / total_priority if total_priority > 0 else 1.0/len(self.criteria)
            importances[criterion] = importance
        
        return importances
    
    def _build_tree_with_priorities(
        self,
        data: List[Dict[str, Any]], 
        priorities: Dict[str, float],
        depth: int
    ) -> Optional['DecisionNode']:
        """優先度を考慮した決定木構築"""
        
        if depth >= self.max_depth or len(data) < self.min_samples_leaf:
            # 葉ノード作成
            return self._create_leaf_node(data)
        
        # 最適な分割特徴選択（優先度考慮）
        best_feature, best_threshold, best_gain = self._find_best_split_with_priorities(
            data, priorities
        )
        
        if best_feature is None or best_gain <= 0:
            return self._create_leaf_node(data)
        
        # データ分割
        left_data, right_data = self._split_data(data, best_feature, best_threshold)
        
        if len(left_data) < self.min_samples_leaf or len(right_data) < self.min_samples_leaf:
            return self._create_leaf_node(data)
        
        # 子ノード作成
        left_child = self._build_tree_with_priorities(left_data, priorities, depth + 1)
        right_child = self._build_tree_with_priorities(right_data, priorities, depth + 1)
        
        return DecisionNode(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            priority_weight=priorities.get(best_feature, 5.0)
        )
    
    def _find_best_split_with_priorities(
        self,
        data: List[Dict[str, Any]],
        priorities: Dict[str, float]
    ) -> Tuple[Optional[str], Optional[float], float]:
        """優先度を考慮した最適分割点探索"""
        
        best_feature = None
        best_threshold = None 
        best_gain = 0.0
        
        for feature in self.criteria:
            if feature not in data[0]:
                continue
            
            # 特徴値の範囲取得
            feature_values = [float(sample[feature]) for sample in data if feature in sample]
            if len(set(feature_values)) < 2:
                continue
            
            # 複数の閾値を試行
            unique_values = sorted(set(feature_values))
            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                
                # 情報利得計算（優先度による重み付き）
                gain = self._calculate_weighted_information_gain(
                    data, feature, threshold, priorities
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _calculate_weighted_information_gain(
        self,
        data: List[Dict[str, Any]],
        feature: str,
        threshold: float, 
        priorities: Dict[str, float]
    ) -> float:
        """優先度による重み付き情報利得計算"""
        
        # データ分割
        left_data, right_data = self._split_data(data, feature, threshold)
        
        if len(left_data) == 0 or len(right_data) == 0:
            return 0.0
        
        # 親ノードのエントロピー
        parent_entropy = self._calculate_entropy(data)
        
        # 子ノードの重み付きエントロピー
        total_samples = len(data)
        left_weight = len(left_data) / total_samples
        right_weight = len(right_data) / total_samples
        
        left_entropy = self._calculate_entropy(left_data)
        right_entropy = self._calculate_entropy(right_data)
        
        weighted_child_entropy = (
            left_weight * left_entropy + 
            right_weight * right_entropy
        )
        
        # 基本情報利得
        information_gain = parent_entropy - weighted_child_entropy
        
        # 優先度による重み付け
        priority_weight = priorities.get(feature, 5.0) / 10.0
        weighted_gain = information_gain * priority_weight
        
        return weighted_gain
    
    def _calculate_entropy(self, data: List[Dict[str, Any]]) -> float:
        """エントロピー計算（簡易版）"""
        
        if len(data) == 0:
            return 0.0
        
        # 適合度の分布に基づくエントロピー計算
        compatibility_scores = []
        for sample in data:
            # サンプルデータから適合度を推定
            score = self._estimate_compatibility(sample)
            compatibility_scores.append(score)
        
        # 適合度を3つのカテゴリに分割（高・中・低）
        high_count = sum(1 for score in compatibility_scores if score > 0.7)
        medium_count = sum(1 for score in compatibility_scores if 0.3 <= score <= 0.7)
        low_count = len(compatibility_scores) - high_count - medium_count
        
        total_count = len(compatibility_scores)
        entropy = 0.0
        
        for count in [high_count, medium_count, low_count]:
            if count > 0:
                probability = count / total_count
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _estimate_compatibility(self, sample: Dict[str, Any]) -> float:
        """サンプルから適合度推定（簡易版）"""
        
        # 基本的な適合度推定ロジック
        total_score = 0.0
        count = 0
        
        for criterion in self.criteria:
            if criterion in sample:
                value = float(sample[criterion])
                # 正規化された値として扱う
                if value > 1.0:
                    value /= 10.0
                
                # 中間値（0.5）に近いほど高スコア
                score = 1.0 - abs(value - 0.5) * 2
                total_score += score
                count += 1
        
        return total_score / count if count > 0 else 0.5
    
    def _split_data(
        self, 
        data: List[Dict[str, Any]], 
        feature: str, 
        threshold: float
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """データ分割"""
        
        left_data = []
        right_data = []
        
        for sample in data:
            if feature in sample:
                value = float(sample[feature])
                if value <= threshold:
                    left_data.append(sample)
                else:
                    right_data.append(sample)
        
        return left_data, right_data
    
    def _create_leaf_node(self, data: List[Dict[str, Any]]) -> 'LeafNode':
        """葉ノード作成"""
        
        # 葉ノードの予測値を計算
        if len(data) == 0:
            prediction = 0.5
        else:
            predictions = [self._estimate_compatibility(sample) for sample in data]
            prediction = sum(predictions) / len(predictions)
        
        return LeafNode(prediction, len(data))
    
    def _traverse_tree_with_priorities(
        self,
        node,
        student_profile: Dict[str, Any],
        priorities: Dict[str, float]
    ) -> Tuple[float, List[str]]:
        """優先度を考慮した決定木探索"""
        
        path = []
        current_node = node
        
        while isinstance(current_node, DecisionNode):
            feature = current_node.feature
            threshold = current_node.threshold
            priority = priorities.get(feature, 5.0)
            
            if feature in student_profile:
                value = float(student_profile[feature])
                
                if value <= threshold:
                    path.append(f"{feature} <= {threshold:.2f} (優先度: {priority})")
                    current_node = current_node.left
                else:
                    path.append(f"{feature} > {threshold:.2f} (優先度: {priority})")
                    current_node = current_node.right
            else:
                # 特徴が欠損している場合は右に進む
                path.append(f"{feature} (欠損) > {threshold:.2f}")
                current_node = current_node.right
        
        # 葉ノードに到達
        if isinstance(current_node, LeafNode):
            prediction = current_node.prediction
        else:
            prediction = 0.5
        
        return prediction, path
    
    def _generate_decision_explanation(
        self, 
        path: List[str], 
        priorities: Dict[str, float]
    ) -> str:
        """決定経路の説明文生成"""
        
        if not path:
            return "決定経路を特定できませんでした"
        
        path_summary = " → ".join(path[:3])  # 最初の3ステップ
        
        return f"決定木による判定: {path_summary}"


class DecisionNode:
    """決定ノード"""
    
    def __init__(self, feature: str, threshold: float, left, right, priority_weight: float = 1.0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.priority_weight = priority_weight


class LeafNode:
    """葉ノード"""
    
    def __init__(self, prediction: float, sample_count: int):
        self.prediction = prediction
        self.sample_count = sample_count