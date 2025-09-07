# core/genetic/evolution.py - 遺伝的アルゴリズム

import random
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from models.schemas import StudentProfile, Laboratory

@dataclass
class Individual:
    """遺伝的アルゴリズムの個体"""
    field_weights: Dict[str, float]     # 分野重み
    criteria_weights: Dict[str, float]  # 評価基準重み
    fitness: float = 0.0               # 適応度

class GeneticAlgorithm:
    """遺伝的アルゴリズムクラス"""
    
    def __init__(self, config: Dict):
        self.population_size = config.get("population_size", 30)
        self.generations = config.get("generations", 50)
        self.mutation_rate = config.get("mutation_rate", 0.1)
        self.crossover_rate = config.get("crossover_rate", 0.8)
        self.elite_size = config.get("elite_size", 5)
        
        self.population: List[Individual] = []
        self.best_individual: Individual = None
        
    def initialize_population(self, research_fields: Dict[str, Dict], 
                            evaluation_criteria: List[str]) -> None:
        """集団を初期化"""
        
        self.population = []
        
        for _ in range(self.population_size):
            # 分野重みをランダム初期化
            field_weights = {
                field_id: random.uniform(0.1, 1.0) 
                for field_id in research_fields.keys()
            }
            
            # 評価基準重みをランダム初期化
            criteria_weights = {
                criterion: random.uniform(0.1, 1.0)
                for criterion in evaluation_criteria
            }
            
            # 重みを正規化
            field_sum = sum(field_weights.values())
            if field_sum > 0:
                field_weights = {k: v/field_sum for k, v in field_weights.items()}
            
            criteria_sum = sum(criteria_weights.values())
            if criteria_sum > 0:
                criteria_weights = {k: v/criteria_sum for k, v in criteria_weights.items()}
            
            individual = Individual(
                field_weights=field_weights,
                criteria_weights=criteria_weights
            )
            self.population.append(individual)
    
    def evaluate_fitness(self, individual: Individual, 
                        student_profile: StudentProfile,
                        labs: List[Laboratory]) -> float:
        """個体の適応度を評価"""
        
        total_score = 0.0
        total_weight = 0.0
        
        # 各研究室に対してスコアを計算
        for lab in labs:
            lab_score = self._calculate_lab_score(individual, student_profile, lab)
            
            # 学生の分野興味に基づく重み
            lab_weight = self._calculate_lab_weight(student_profile, lab)
            
            total_score += lab_score * lab_weight
            total_weight += lab_weight
        
        # 平均適合度
        avg_score = total_score / total_weight if total_weight > 0 else 0.0
        
        # 多様性ボーナス
        diversity_bonus = self._calculate_diversity_bonus(individual)
        
        # 一貫性ボーナス
        consistency_bonus = self._calculate_consistency_bonus(individual, student_profile)
        
        # 最終適応度
        fitness = avg_score + diversity_bonus * 0.1 + consistency_bonus * 0.1
        
        return min(1.0, max(0.0, fitness))
    
    def _calculate_lab_score(self, individual: Individual, 
                           student: StudentProfile, lab: Laboratory) -> float:
        """研究室スコアを計算"""
        
        # 分野適合性
        field_score = 0.0
        field_count = 0
        
        student_fields = {fi.field_id: fi for fi in student.field_interests}
        
        for field_id in lab.research_fields:
            if field_id in student_fields:
                student_interest = student_fields[field_id]
                field_weight = individual.field_weights.get(field_id, 0)
                
                # 分野スコア計算
                interest_normalized = student_interest.interest_level / 10.0
                experience_normalized = student_interest.experience_level / 10.0
                importance_normalized = student_interest.importance_level / 10.0
                
                field_compatibility = (
                    interest_normalized * 0.5 +
                    experience_normalized * 0.3 +
                    importance_normalized * 0.2
                )
                
                field_score += field_compatibility * field_weight
                field_count += 1
        
        if field_count > 0:
            field_score /= field_count
        
        # 評価基準適合性
        criteria_score = 0.0
        criteria_count = 0
        
        student_criteria = student.evaluation_criteria.dict()
        lab_features = lab.features.dict()
        
        for criterion, weight in individual.criteria_weights.items():
            if criterion in student_criteria and criterion in lab_features:
                student_val = student_criteria[criterion]
                lab_val = lab_features[criterion]
                
                # 類似度計算（ガウシアン）
                distance = abs(student_val - lab_val)
                similarity = np.exp(-(distance ** 2) / (2 * 2.0 ** 2))
                
                criteria_score += similarity * weight
                criteria_count += 1
        
        if criteria_count > 0:
            criteria_score /= criteria_count
        
        # 総合スコア
        total_score = field_score * 0.6 + criteria_score * 0.4
        return total_score
    
    def _calculate_lab_weight(self, student: StudentProfile, lab: Laboratory) -> float:
        """研究室の重み計算"""
        
        weight = 0.0
        
        student_fields = {fi.field_id: fi for fi in student.field_interests}
        
        for field_id in lab.research_fields:
            if field_id in student_fields:
                importance = student_fields[field_id].importance_level / 10.0
                weight += importance
        
        return max(0.1, weight)  # 最小重み0.1
    
    def _calculate_diversity_bonus(self, individual: Individual) -> float:
        """多様性ボーナス計算"""
        
        # 分野重みの分散（エントロピー）
        field_weights = list(individual.field_weights.values())
        if not field_weights:
            return 0.0
        
        # 正規化
        total = sum(field_weights)
        if total == 0:
            return 0.0
        
        normalized = [w/total for w in field_weights]
        
        # エントロピー計算
        entropy = -sum(p * np.log(p + 1e-10) for p in normalized if p > 0)
        max_entropy = np.log(len(field_weights))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _calculate_consistency_bonus(self, individual: Individual, 
                                   student: StudentProfile) -> float:
        """一貫性ボーナス計算"""
        
        # 学生の分野興味と個体の重みの一貫性
        student_fields = {fi.field_id: fi for fi in student.field_interests}
        
        consistency = 0.0
        count = 0
        
        for field_id, weight in individual.field_weights.items():
            if field_id in student_fields:
                student_importance = student_fields[field_id].importance_level / 10.0
                
                # 重みと重要度の相関
                correlation = 1.0 - abs(weight - student_importance)
                consistency += correlation
                count += 1
        
        return consistency / count if count > 0 else 0.0
    
    def selection(self) -> List[Individual]:
        """選択操作（トーナメント選択）"""
        
        selected = []
        tournament_size = 3
        
        for _ in range(self.population_size):
            tournament = random.sample(self.population, min(tournament_size, len(self.population)))
            winner = max(tournament, key=lambda x: x.fitness)
            selected.append(winner)
        
        return selected
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """交叉操作"""
        
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        # 分野重み交叉（一様交叉）
        child1_field_weights = {}
        child2_field_weights = {}
        
        for field_id in parent1.field_weights.keys():
            if random.random() < 0.5:
                child1_field_weights[field_id] = parent1.field_weights[field_id]
                child2_field_weights[field_id] = parent2.field_weights[field_id]
            else:
                child1_field_weights[field_id] = parent2.field_weights[field_id]
                child2_field_weights[field_id] = parent1.field_weights[field_id]
        
        # 評価基準重み交叉
        child1_criteria_weights = {}
        child2_criteria_weights = {}
        
        for criterion in parent1.criteria_weights.keys():
            if random.random() < 0.5:
                child1_criteria_weights[criterion] = parent1.criteria_weights[criterion]
                child2_criteria_weights[criterion] = parent2.criteria_weights[criterion]
            else:
                child1_criteria_weights[criterion] = parent2.criteria_weights[criterion]
                child2_criteria_weights[criterion] = parent1.criteria_weights[criterion]
        
        # 重み正規化
        self._normalize_weights(child1_field_weights, child1_criteria_weights)
        self._normalize_weights(child2_field_weights, child2_criteria_weights)
        
        child1 = Individual(child1_field_weights, child1_criteria_weights)
        child2 = Individual(child2_field_weights, child2_criteria_weights)
        
        return child1, child2
    
    def mutation(self, individual: Individual) -> Individual:
        """変異操作"""
        
        if random.random() > self.mutation_rate:
            return individual
        
        # 分野重み変異
        for field_id in individual.field_weights.keys():
            if random.random() < 0.2:  # 20%の確率で変異
                noise = random.uniform(-0.1, 0.1)
                individual.field_weights[field_id] += noise
                individual.field_weights[field_id] = max(0.01, min(1.0, individual.field_weights[field_id]))
        
        # 評価基準重み変異
        for criterion in individual.criteria_weights.keys():
            if random.random() < 0.2:
                noise = random.uniform(-0.1, 0.1)
                individual.criteria_weights[criterion] += noise
                individual.criteria_weights[criterion] = max(0.01, min(1.0, individual.criteria_weights[criterion]))
        
        # 重み正規化
        self._normalize_weights(individual.field_weights, individual.criteria_weights)
        
        return individual
    
    def _normalize_weights(self, field_weights: Dict[str, float], 
                          criteria_weights: Dict[str, float]) -> None:
        """重みを正規化"""
        
        # 分野重み正規化
        field_sum = sum(field_weights.values())
        if field_sum > 0:
            for key in field_weights:
                field_weights[key] /= field_sum
        
        # 評価基準重み正規化
        criteria_sum = sum(criteria_weights.values())
        if criteria_sum > 0:
            for key in criteria_weights:
                criteria_weights[key] /= criteria_sum
    
    def evolve(self, student_profile: StudentProfile, labs: List[Laboratory],
              research_fields: Dict[str, Dict], evaluation_criteria: List[str]) -> Individual:
        """進化アルゴリズムを実行"""
        
        # 初期化
        self.initialize_population(research_fields, evaluation_criteria)
        
        print(f"🧬 遺伝的アルゴリズム開始: 集団サイズ{self.population_size}, 世代数{self.generations}")
        
        for generation in range(self.generations):
            
            # 適応度評価
            for individual in self.population:
                individual.fitness = self.evaluate_fitness(individual, student_profile, labs)
            
            # エリート保存
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            elite = self.population[:self.elite_size]
            
            # 進捗表示
            if generation % 10 == 0:
                best_fitness = elite[0].fitness
                avg_fitness = np.mean([ind.fitness for ind in self.population])
                print(f"世代 {generation:2d}: 最高={best_fitness:.4f}, 平均={avg_fitness:.4f}")
            
            # 新しい集団生成
            new_population = elite[:]
            
            # 選択
            selected = self.selection()
            
            # 交叉・変異
            while len(new_population) < self.population_size:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                new_population.extend([child1, child2])
            
            # 集団更新
            self.population = new_population[:self.population_size]
        
        # 最良個体
        final_evaluation = [(ind, self.evaluate_fitness(ind, student_profile, labs)) for ind in self.population]
        final_evaluation.sort(key=lambda x: x[1], reverse=True)
        
        self.best_individual = final_evaluation[0][0]
        self.best_individual.fitness = final_evaluation[0][1]
        
        print(f"✅ 最適化完了: 最終適応度 = {self.best_individual.fitness:.4f}")
        
        return self.best_individual