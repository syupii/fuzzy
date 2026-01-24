#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遺伝的アルゴリズムによる逆引き分析 v2.2
研究室ごとの理想的な学生プロファイルを探索

【v2.2 変更点】
- research_field_match を基本項目から削除（11項目）
- field_priority（分野重視度）として別途管理
- 最終スコア計算: S = (1 - λ) × S_basic + λ × S_field
- ボーナス/ペナルティ削除
- S_basic = γ × S_fuzzy + (1 - γ) × S_gaussian

使用方法:
    python genetic_optimizer_reverse_lookup.py --lab_id lab_001
    python genetic_optimizer_reverse_lookup.py --all --output results/
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import math

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class GAConfig:
    """遺伝的アルゴリズムの設定"""
    population_size: int = 100  # 個体数
    generations: int = 50  # 世代数
    crossover_rate: float = 0.8  # 交叉率
    mutation_rate: float = 0.1  # 突然変異率
    elite_size: int = 5  # エリート保存数
    tournament_size: int = 5  # トーナメント選択サイズ


# 研究分野定義（20分野）
RESEARCH_FIELDS = {
    # テクノロジー・システム（12分野）
    "ai_ml": "人工知能・機械学習",
    "image_processing": "画像・映像処理",
    "network_security": "ネットワーク・セキュリティ",
    "database_systems": "データベース・情報システム",
    "embedded_iot": "組込み・IoT",
    "education_linguistics": "教育・言語学",
    "natural_science_math": "自然科学・数理",
    "tourism_regional": "観光情報・地域システム",
    "business_decision": "経営情報・意思決定支援",
    "audio_processing": "音声・音響情報処理",
    "system_ethics": "システム運用・情報倫理",
    "medical_healthcare": "医療情報・ヘルスケア",
    
    # クリエイティブ（4分野）
    "web_design": "Webデザイン・UI/UX",
    "design_visual": "デザイン・視覚表現",
    "video_animation": "映像・アニメーション",
    "computer_music": "コンピュータ音楽・サウンドアート",
    
    # エンターテイメント（2分野）
    "game_esports": "ゲーム開発・eスポーツ",
    "vr_ar_media": "VR/AR・メディアアート",
    
    # 人文・社会・体育（2分野）
    "philosophy_humanities": "哲学・人文・環境行動学",
    "sports_science": "スポーツ・体育科学"
}

# 分野カテゴリマッピング
FIELD_CATEGORIES = {
    "テクノロジー・システム": [
        "ai_ml", "image_processing", "network_security", "database_systems",
        "embedded_iot", "education_linguistics", "natural_science_math",
        "tourism_regional", "business_decision", "audio_processing",
        "system_ethics", "medical_healthcare"
    ],
    "クリエイティブ": [
        "web_design", "design_visual", "video_animation", "computer_music"
    ],
    "エンターテイメント": [
        "game_esports", "vr_ar_media"
    ],
    "人文・社会・体育": [
        "philosophy_humanities", "sports_science"
    ]
}

# 基本11項目（research_field_matchを削除）
BASIC_CRITERIA = [
    "research_intensity",
    "advisor_style",
    "team_work",
    "workload",
    "theory_practice",
    "skill_development",
    "lab_atmosphere",
    "flexibility",
    "publication_opportunity",
    "interdisciplinary",
    "communication_style"
]

# 分野マッチングスコア
FIELD_EXACT_MATCH_SCORE = 1.0
FIELD_CATEGORY_DECAY = 0.7
FIELD_NO_MATCH_SCORE = 0.3


class MembershipFunctions:
    """
    メンバーシップ関数（v2.1準拠）
    
    【3分岐（優先度 ≥ 8）】
    - 低（三角）: ピーク 0.1、終了 0.5
    - 中（台形）: 開始 0.2、上辺 0.4-0.7、終了 0.9
    - 高（三角）: 開始 0.6、ピーク 1.0
    
    【2分岐（優先度 5〜7）】
    - 低（三角）: ピーク 0.1、終了 0.6
    - 高（三角）: 開始 0.4、ピーク 1.0
    """
    
    @staticmethod
    def low_3level(x: float) -> float:
        """低（三角形）- 3分岐用"""
        if x <= 0.1:
            return 1.0
        elif x < 0.5:
            return (0.5 - x) / 0.4
        else:
            return 0.0
    
    @staticmethod
    def medium_3level(x: float) -> float:
        """中（台形）- 3分岐用"""
        if x <= 0.2:
            return 0.0
        elif x < 0.4:
            return (x - 0.2) / 0.2
        elif x <= 0.7:
            return 1.0
        elif x < 0.9:
            return (0.9 - x) / 0.2
        else:
            return 0.0
    
    @staticmethod
    def high_3level(x: float) -> float:
        """高（三角形）- 3分岐用"""
        if x <= 0.6:
            return 0.0
        elif x < 1.0:
            return (x - 0.6) / 0.4
        else:
            return 1.0
    
    @staticmethod
    def low_2level(x: float) -> float:
        """低（三角形）- 2分岐用"""
        if x <= 0.1:
            return 1.0
        elif x < 0.6:
            return (0.6 - x) / 0.5
        else:
            return 0.0
    
    @staticmethod
    def high_2level(x: float) -> float:
        """高（三角形）- 2分岐用"""
        if x <= 0.4:
            return 0.0
        elif x < 1.0:
            return (x - 0.4) / 0.6
        else:
            return 1.0
    
    @staticmethod
    def fuzzify_3level(x: float) -> Dict[str, float]:
        """3段階ファジィ化（正規化済み）"""
        raw_low = MembershipFunctions.low_3level(x)
        raw_medium = MembershipFunctions.medium_3level(x)
        raw_high = MembershipFunctions.high_3level(x)
        
        total = raw_low + raw_medium + raw_high
        
        if total > 0:
            return {
                "low": raw_low / total,
                "medium": raw_medium / total,
                "high": raw_high / total
            }
        else:
            return {"low": 0.0, "medium": 1.0, "high": 0.0}
    
    @staticmethod
    def fuzzify_2level(x: float) -> Dict[str, float]:
        """2段階ファジィ化（正規化済み）"""
        raw_low = MembershipFunctions.low_2level(x)
        raw_high = MembershipFunctions.high_2level(x)
        
        total = raw_low + raw_high
        
        if total > 0:
            return {
                "low": raw_low / total,
                "high": raw_high / total
            }
        else:
            return {"low": 0.5, "high": 0.5}


@dataclass
class FuzzyPath:
    """ファジィパス"""
    path_id: int
    layers: List[Tuple[str, str, float]]
    total_membership: float
    leaf_value: float


class StudentProfile:
    """学生プロファイル（染色体）"""
    
    def __init__(
        self,
        criteria_values: np.ndarray = None,
        criteria_priorities: np.ndarray = None,
        field_interests: Dict[str, float] = None,
        field_priority: float = None
    ):
        """
        Args:
            criteria_values: 11項目の評価値 [0, 1]の範囲
            criteria_priorities: 11項目の優先度 [0, 1]の範囲
            field_interests: 分野興味度 {field_id: [0, 1]}
            field_priority: 分野重視度 [0, 1]の範囲
        """
        if criteria_values is None:
            self.criteria_values = np.random.uniform(0, 1, len(BASIC_CRITERIA))
        else:
            self.criteria_values = criteria_values.copy()
        
        if criteria_priorities is None:
            self.criteria_priorities = np.random.uniform(0, 1, len(BASIC_CRITERIA))
        else:
            self.criteria_priorities = criteria_priorities.copy()
        
        if field_interests is None:
            num_interests = np.random.randint(2, 4)
            selected_fields = np.random.choice(list(RESEARCH_FIELDS.keys()), num_interests, replace=False)
            self.field_interests = {
                field: np.random.uniform(0.5, 1.0) 
                for field in selected_fields
            }
        else:
            self.field_interests = field_interests.copy()
        
        if field_priority is None:
            self.field_priority = np.random.uniform(0, 1)
        else:
            self.field_priority = field_priority
        
        self.fitness = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換（1-10スケールに戻す）"""
        result = {}
        
        for i, criterion in enumerate(BASIC_CRITERIA):
            result[criterion] = float(self.criteria_values[i] * 9 + 1)
            result[f"{criterion}_priority"] = float(self.criteria_priorities[i] * 9 + 1)
        
        result["field_interests"] = {
            field: float(val * 9 + 1) 
            for field, val in self.field_interests.items()
        }
        result["field_priority"] = float(self.field_priority * 9 + 1)
        
        return result
    
    def copy(self):
        """コピーを作成"""
        new_profile = StudentProfile(
            self.criteria_values.copy(),
            self.criteria_priorities.copy(),
            self.field_interests.copy(),
            self.field_priority
        )
        new_profile.fitness = self.fitness
        return new_profile


class LabMatcher:
    """
    研究室マッチング評価器（v2.2準拠）
    
    【計算式】
    S_basic = γ × S_fuzzy + (1 - γ) × S_gaussian
    λ = field_priority / 10
    S = (1 - λ) × S_basic + λ × S_field
    """
    
    # パラメータ
    SIMILARITY_SIGMA = 0.3
    PRUNING_THRESHOLD = 0.01
    HIGH_PRIORITY_THRESHOLD = 0.8  # [0,1]スケールで8/10
    MID_PRIORITY_THRESHOLD = 0.5   # [0,1]スケールで5/10
    FUZZY_GAUSSIAN_GAMMA = 0.5
    
    def __init__(self, lab_profile: Dict[str, Any]):
        """
        Args:
            lab_profile: 研究室プロファイル
        """
        self.lab_profile = lab_profile
        self.lab_criteria = self._normalize_criteria(lab_profile)
        self.lab_field = lab_profile.get("field_id", "")
        self.lab_fields = lab_profile.get("research_fields", [])
        
        if not self.lab_field and self.lab_fields:
            self.lab_field = self.lab_fields[0]
    
    def _normalize_criteria(self, profile: Dict[str, Any]) -> np.ndarray:
        """評価値を正規化 [1, 10] -> [0, 1]"""
        values = []
        for criterion in BASIC_CRITERIA:
            val = profile.get(criterion, 5.5)
            if val > 1:
                normalized = val / 10.0
            else:
                normalized = val
            values.append(normalized)
        return np.array(values)
    
    def _normalize_value(self, value: float) -> float:
        """値を0-1に正規化"""
        if value >= 1.0 and value <= 10.0:
            return value / 10.0
        elif value > 1.0:
            return min(value / 10.0, 1.0)
        return value
    
    def _calculate_priority_weight(self, priority: float) -> float:
        """優先度から重みを計算"""
        return priority ** 1.5
    
    def _gaussian_similarity(self, val1: float, val2: float) -> float:
        """ガウス類似度計算"""
        d = abs(val1 - val2)
        return math.exp(-(d ** 2) / (2 * self.SIMILARITY_SIGMA ** 2))
    
    def _get_sorted_priorities(self, student: StudentProfile) -> List[Dict[str, Any]]:
        """優先度でソートされた項目リストを取得"""
        priorities = []
        for i, criterion in enumerate(BASIC_CRITERIA):
            priority = student.criteria_priorities[i]
            priorities.append({"criterion": criterion, "priority": priority, "index": i})
        priorities.sort(key=lambda x: x["priority"], reverse=True)
        return priorities
    
    def _build_fuzzy_tree(self, priorities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """ファジィ決定木を構築"""
        tree_layers = []
        for item in priorities:
            priority = item["priority"]
            criterion = item["criterion"]
            index = item["index"]
            
            if priority >= self.HIGH_PRIORITY_THRESHOLD:
                tree_layers.append({
                    "criterion": criterion,
                    "priority": priority,
                    "index": index,
                    "branches": 3,
                    "labels": ["low", "medium", "high"]
                })
            elif priority >= self.MID_PRIORITY_THRESHOLD:
                tree_layers.append({
                    "criterion": criterion,
                    "priority": priority,
                    "index": index,
                    "branches": 2,
                    "labels": ["low", "high"]
                })
        return tree_layers
    
    def _fuzzify_lab(self, tree_layers: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """研究室の各項目をファジィ化"""
        lab_fuzzified = {}
        for layer in tree_layers:
            criterion = layer["criterion"]
            branches = layer["branches"]
            index = layer["index"]
            normalized = self.lab_criteria[index]
            
            if branches == 3:
                memberships = MembershipFunctions.fuzzify_3level(normalized)
            else:
                memberships = MembershipFunctions.fuzzify_2level(normalized)
            
            lab_fuzzified[criterion] = memberships
        return lab_fuzzified
    
    def _generate_paths_recursive(
        self,
        tree_layers: List[Dict[str, Any]],
        lab_fuzzified: Dict[str, Dict[str, float]],
        layer_idx: int,
        current_path: List[Tuple[str, str, float]],
        cumulative_membership: float,
        all_paths: List[FuzzyPath]
    ):
        """パスを再帰的に生成"""
        if layer_idx >= len(tree_layers):
            all_paths.append(FuzzyPath(
                path_id=len(all_paths),
                layers=current_path.copy(),
                total_membership=cumulative_membership,
                leaf_value=0.0
            ))
            return
        
        layer = tree_layers[layer_idx]
        criterion = layer["criterion"]
        labels = layer["labels"]
        memberships = lab_fuzzified[criterion]
        
        for label in labels:
            membership = memberships[label]
            new_membership = cumulative_membership * membership
            
            if new_membership < self.PRUNING_THRESHOLD:
                continue
            
            new_path = current_path + [(criterion, label, membership)]
            self._generate_paths_recursive(
                tree_layers, lab_fuzzified, layer_idx + 1,
                new_path, new_membership, all_paths
            )
    
    def _generate_paths_from_lab(
        self,
        tree_layers: List[Dict[str, Any]],
        lab_fuzzified: Dict[str, Dict[str, float]]
    ) -> List[FuzzyPath]:
        """研究室からパスを生成"""
        if not tree_layers:
            return [FuzzyPath(path_id=0, layers=[], total_membership=1.0, leaf_value=0.0)]
        
        all_paths = []
        self._generate_paths_recursive(tree_layers, lab_fuzzified, 0, [], 1.0, all_paths)
        return all_paths
    
    def _prune_and_normalize_paths(self, paths: List[FuzzyPath]) -> List[FuzzyPath]:
        """パスを枝刈りして正規化"""
        paths = [p for p in paths if p.total_membership >= self.PRUNING_THRESHOLD]
        if not paths:
            return paths
        total = sum(p.total_membership for p in paths)
        if total > 0:
            for path in paths:
                path.total_membership = path.total_membership / total
        return paths
    
    def _fuzzify_student(
        self,
        student: StudentProfile,
        tree_layers: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """学生の各項目をファジィ化"""
        student_fuzzified = {}
        for layer in tree_layers:
            criterion = layer["criterion"]
            branches = layer["branches"]
            index = layer["index"]
            normalized = student.criteria_values[index]
            
            if branches == 3:
                memberships = MembershipFunctions.fuzzify_3level(normalized)
            else:
                memberships = MembershipFunctions.fuzzify_2level(normalized)
            
            student_fuzzified[criterion] = memberships
        return student_fuzzified
    
    def _calculate_leaf_values(
        self,
        paths: List[FuzzyPath],
        student_fuzzified: Dict[str, Dict[str, float]],
        tree_layers: List[Dict[str, Any]],
        student: StudentProfile
    ) -> List[FuzzyPath]:
        """各パスのリーフ値を計算"""
        for path in paths:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for criterion, label, _ in path.layers:
                student_membership = student_fuzzified[criterion][label]
                layer = next((l for l in tree_layers if l["criterion"] == criterion), None)
                if layer:
                    priority = layer["priority"]
                    weight = self._calculate_priority_weight(priority)
                    weighted_sum += student_membership * weight
                    total_weight += weight
            
            path.leaf_value = weighted_sum / total_weight if total_weight > 0 else 0.5
        return paths
    
    def _calculate_fuzzy_score(self, paths: List[FuzzyPath]) -> float:
        """S_fuzzy を計算"""
        if not paths:
            return 0.5
        return sum(path.total_membership * path.leaf_value for path in paths)
    
    def _calculate_gaussian_score(self, student: StudentProfile) -> float:
        """S_gaussian を計算"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for i, criterion in enumerate(BASIC_CRITERIA):
            student_val = student.criteria_values[i]
            lab_val = self.lab_criteria[i]
            similarity = self._gaussian_similarity(student_val, lab_val)
            priority = student.criteria_priorities[i]
            weight = self._calculate_priority_weight(priority)
            
            weighted_sum += similarity * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def _calculate_basic_score(self, student: StudentProfile) -> Tuple[float, float, float]:
        """
        基本スコアを計算
        
        Returns:
            (S_basic, S_fuzzy, S_gaussian)
        """
        # 決定木構築
        priorities = self._get_sorted_priorities(student)
        tree_layers = self._build_fuzzy_tree(priorities)
        
        if not tree_layers:
            # 決定木が構築できない場合はガウス類似度のみ
            gaussian_score = self._calculate_gaussian_score(student)
            return gaussian_score, 0.5, gaussian_score
        
        # 研究室のファジィ化
        lab_fuzzified = self._fuzzify_lab(tree_layers)
        
        # パス生成
        fuzzy_paths = self._generate_paths_from_lab(tree_layers, lab_fuzzified)
        fuzzy_paths = self._prune_and_normalize_paths(fuzzy_paths)
        
        if not fuzzy_paths:
            gaussian_score = self._calculate_gaussian_score(student)
            return gaussian_score, 0.5, gaussian_score
        
        # 学生のファジィ化
        student_fuzzified = self._fuzzify_student(student, tree_layers)
        
        # リーフ値計算
        fuzzy_paths = self._calculate_leaf_values(
            fuzzy_paths, student_fuzzified, tree_layers, student
        )
        
        # S_fuzzy
        fuzzy_score = self._calculate_fuzzy_score(fuzzy_paths)
        
        # S_gaussian
        gaussian_score = self._calculate_gaussian_score(student)
        
        # S_basic = γ × S_fuzzy + (1 - γ) × S_gaussian
        gamma = self.FUZZY_GAUSSIAN_GAMMA
        basic_score = gamma * fuzzy_score + (1 - gamma) * gaussian_score
        
        return basic_score, fuzzy_score, gaussian_score
    
    def _get_field_category(self, field_id: str) -> Optional[str]:
        """分野のカテゴリを取得"""
        for category, fields in FIELD_CATEGORIES.items():
            if field_id in fields:
                return category
        return None
    
    def _calculate_field_score(self, student: StudentProfile) -> float:
        """
        分野スコアを計算（重み付き平均）
        
        S_field = Σ(match_score × interest) / Σ(interest)
        """
        if not student.field_interests:
            return 0.5
        
        if not self.lab_field:
            return 0.5
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        lab_category = self._get_field_category(self.lab_field)
        
        for interest_field, interest_level in student.field_interests.items():
            if interest_field == self.lab_field:
                # 完全一致
                match_score = FIELD_EXACT_MATCH_SCORE
            elif lab_category:
                student_category = self._get_field_category(interest_field)
                if student_category and student_category == lab_category:
                    # カテゴリ一致
                    match_score = FIELD_CATEGORY_DECAY
                else:
                    # 不一致
                    match_score = FIELD_NO_MATCH_SCORE
            else:
                match_score = FIELD_NO_MATCH_SCORE
            
            weighted_sum += match_score * interest_level
            total_weight += interest_level
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def evaluate(self, student: StudentProfile) -> float:
        """
        総合評価（v2.2準拠）
        
        λ = field_priority / 10（ここでは field_priority は [0,1]）
        S = (1 - λ) × S_basic + λ × S_field
        """
        # 基本スコア
        basic_score, fuzzy_score, gaussian_score = self._calculate_basic_score(student)
        
        # 分野スコア
        field_score = self._calculate_field_score(student)
        
        # 最終スコア
        # λ = field_priority（[0,1]スケール、つまり field_priority/10 に相当）
        lambda_weight = student.field_priority
        
        final_score = (1 - lambda_weight) * basic_score + lambda_weight * field_score
        
        return final_score
    
    def evaluate_detailed(self, student: StudentProfile) -> Dict[str, Any]:
        """詳細な評価結果を返す"""
        basic_score, fuzzy_score, gaussian_score = self._calculate_basic_score(student)
        field_score = self._calculate_field_score(student)
        lambda_weight = student.field_priority
        final_score = (1 - lambda_weight) * basic_score + lambda_weight * field_score
        
        return {
            "final_score": final_score,
            "basic_score": basic_score,
            "fuzzy_score": fuzzy_score,
            "gaussian_score": gaussian_score,
            "field_score": field_score,
            "lambda_weight": lambda_weight,
            "basic_weight": 1 - lambda_weight
        }


class GeneticOptimizer:
    """遺伝的アルゴリズムによる最適化"""
    
    def __init__(self, config: GAConfig = None):
        self.config = config or GAConfig()
        self.population: List[StudentProfile] = []
        self.best_individual: StudentProfile = None
        self.best_fitness: float = 0.0
        self.fitness_history: List[float] = []
    
    def initialize_population(self) -> List[StudentProfile]:
        """初期個体群の生成"""
        return [StudentProfile() for _ in range(self.config.population_size)]
    
    def evaluate_population(self, population: List[StudentProfile], matcher: LabMatcher):
        """個体群の評価"""
        for individual in population:
            individual.fitness = matcher.evaluate(individual)
    
    def tournament_selection(self, population: List[StudentProfile]) -> StudentProfile:
        """トーナメント選択"""
        tournament = np.random.choice(population, self.config.tournament_size, replace=False)
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(
        self,
        parent1: StudentProfile,
        parent2: StudentProfile
    ) -> Tuple[StudentProfile, StudentProfile]:
        """交叉（一様交叉）"""
        if np.random.rand() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # 基本項目値の交叉
        mask_values = np.random.rand(len(BASIC_CRITERIA)) < 0.5
        child1_values = np.where(mask_values, parent1.criteria_values, parent2.criteria_values)
        child2_values = np.where(mask_values, parent2.criteria_values, parent1.criteria_values)
        
        # 優先度の交叉
        mask_priorities = np.random.rand(len(BASIC_CRITERIA)) < 0.5
        child1_priorities = np.where(mask_priorities, parent1.criteria_priorities, parent2.criteria_priorities)
        child2_priorities = np.where(mask_priorities, parent2.criteria_priorities, parent1.criteria_priorities)
        
        # 分野興味の交叉
        all_fields = set(parent1.field_interests.keys()) | set(parent2.field_interests.keys())
        
        child1_fields = {}
        child2_fields = {}
        
        for field in all_fields:
            if np.random.rand() < 0.5:
                if field in parent1.field_interests:
                    child1_fields[field] = parent1.field_interests[field]
                if field in parent2.field_interests:
                    child2_fields[field] = parent2.field_interests[field]
            else:
                if field in parent2.field_interests:
                    child1_fields[field] = parent2.field_interests[field]
                if field in parent1.field_interests:
                    child2_fields[field] = parent1.field_interests[field]
        
        # 分野重視度の交叉
        if np.random.rand() < 0.5:
            child1_field_priority = parent1.field_priority
            child2_field_priority = parent2.field_priority
        else:
            child1_field_priority = parent2.field_priority
            child2_field_priority = parent1.field_priority
        
        child1 = StudentProfile(child1_values, child1_priorities, child1_fields, child1_field_priority)
        child2 = StudentProfile(child2_values, child2_priorities, child2_fields, child2_field_priority)
        
        return child1, child2
    
    def mutate(self, individual: StudentProfile):
        """突然変異"""
        # 基本項目値の突然変異
        for i in range(len(BASIC_CRITERIA)):
            if np.random.rand() < self.config.mutation_rate:
                individual.criteria_values[i] += np.random.normal(0, 0.1)
                individual.criteria_values[i] = np.clip(individual.criteria_values[i], 0, 1)
        
        # 優先度の突然変異
        for i in range(len(BASIC_CRITERIA)):
            if np.random.rand() < self.config.mutation_rate:
                individual.criteria_priorities[i] += np.random.normal(0, 0.1)
                individual.criteria_priorities[i] = np.clip(individual.criteria_priorities[i], 0, 1)
        
        # 分野興味の突然変異
        if np.random.rand() < self.config.mutation_rate:
            if np.random.rand() < 0.5 and len(individual.field_interests) < 5:
                # 追加
                available_fields = [f for f in RESEARCH_FIELDS.keys() if f not in individual.field_interests]
                if available_fields:
                    new_field = np.random.choice(available_fields)
                    individual.field_interests[new_field] = np.random.uniform(0.5, 1.0)
            elif len(individual.field_interests) > 1:
                # 削除
                field_to_remove = np.random.choice(list(individual.field_interests.keys()))
                del individual.field_interests[field_to_remove]
        
        # 既存分野の興味度を変異
        for field in list(individual.field_interests.keys()):
            if np.random.rand() < self.config.mutation_rate:
                individual.field_interests[field] += np.random.normal(0, 0.1)
                individual.field_interests[field] = np.clip(individual.field_interests[field], 0, 1)
        
        # 分野重視度の突然変異
        if np.random.rand() < self.config.mutation_rate:
            individual.field_priority += np.random.normal(0, 0.1)
            individual.field_priority = np.clip(individual.field_priority, 0, 1)
    
    def optimize(self, lab_profile: Dict[str, Any]) -> StudentProfile:
        """最適化実行"""
        matcher = LabMatcher(lab_profile)
        
        # 初期化
        self.population = self.initialize_population()
        self.evaluate_population(self.population, matcher)
        
        # 最良個体を記録
        self.best_individual = max(self.population, key=lambda x: x.fitness).copy()
        self.best_fitness = self.best_individual.fitness
        self.fitness_history = [self.best_fitness]
        
        logger.info(f"初期世代: 最良適合度 = {self.best_fitness:.4f}")
        
        # 世代ループ
        for generation in range(self.config.generations):
            # エリート保存
            elites = sorted(self.population, key=lambda x: x.fitness, reverse=True)[:self.config.elite_size]
            
            # 新世代生成
            new_population = [elite.copy() for elite in elites]
            
            while len(new_population) < self.config.population_size:
                # 選択
                parent1 = self.tournament_selection(self.population)
                parent2 = self.tournament_selection(self.population)
                
                # 交叉
                child1, child2 = self.crossover(parent1, parent2)
                
                # 突然変異
                self.mutate(child1)
                self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # 個体数を調整
            self.population = new_population[:self.config.population_size]
            
            # 評価
            self.evaluate_population(self.population, matcher)
            
            # 最良個体更新
            current_best = max(self.population, key=lambda x: x.fitness)
            if current_best.fitness > self.best_fitness:
                self.best_individual = current_best.copy()
                self.best_fitness = current_best.fitness
            
            self.fitness_history.append(self.best_fitness)
            
            if (generation + 1) % 10 == 0:
                avg_fitness = np.mean([ind.fitness for ind in self.population])
                logger.info(f"世代 {generation + 1}/{self.config.generations}: "
                          f"最良={self.best_fitness:.4f}, 平均={avg_fitness:.4f}")
        
        logger.info(f"最適化完了: 最終適合度 = {self.best_fitness:.4f}")
        
        return self.best_individual


def flatten_lab_data(labs_data: List[Dict]) -> List[Dict]:
    """研究室データをフラット化"""
    flattened = []
    
    for lab in labs_data:
        flat_lab = lab.copy()
        
        # features をフラット化
        if "features" in lab:
            features = lab["features"]
            for key, value in features.items():
                flat_lab[key] = value
            del flat_lab["features"]
        
        # キー名の正規化
        if "lab_id" not in flat_lab:
            lab_id = flat_lab.get("id") or flat_lab.get("laboratory_id") or flat_lab.get("labId")
            if lab_id:
                flat_lab["lab_id"] = lab_id
            else:
                logger.warning(f"研究室IDが見つかりません: {flat_lab}")
                continue
        
        if "lab_name" not in flat_lab:
            lab_name = flat_lab.get("name") or flat_lab.get("laboratory_name") or flat_lab.get("labName")
            if lab_name:
                flat_lab["lab_name"] = lab_name
        
        if "field_id" not in flat_lab:
            research_fields = flat_lab.get("research_fields", [])
            if research_fields:
                flat_lab["field_id"] = research_fields[0].lower().replace("・", "_").replace(" ", "_")
        
        flattened.append(flat_lab)
    
    return flattened


def load_lab_database(use_lab_database_class: bool = True) -> List[Dict]:
    """研究室データベースを読み込み"""
    if use_lab_database_class:
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            
            from data.models.labs_database import LabDatabase
            
            db = LabDatabase()
            raw_labs_data = db.get_all_labs()
            labs_data = flatten_lab_data(raw_labs_data)
            
            logger.info(f"{len(labs_data)}件の研究室データを読み込みました（LabDatabaseクラス使用）")
            return labs_data
            
        except ImportError as e:
            logger.warning(f"LabDatabaseクラスのインポートに失敗: {e}")
            logger.info("フォールバック: 直接JSON読み込みを試みます")
            use_lab_database_class = False
    
    if not use_lab_database_class:
        import os
        
        possible_paths = [
            "data/labs_database.json",
            "backend/data/labs_database.json",
            "../data/labs_database.json",
            os.path.join(os.path.dirname(__file__), "data", "labs_database.json"),
        ]
        
        actual_path = None
        for path in possible_paths:
            if os.path.exists(path):
                actual_path = path
                logger.info(f"研究室データベースを読み込み: {path}")
                break
        
        if actual_path is None:
            error_msg = "研究室データベースが見つかりません"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        with open(actual_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            labs = data.get("labs") or data.get("laboratories") or data.get("lab_list")
        elif isinstance(data, list):
            labs = data
        else:
            raise ValueError(f"不明なデータ構造: {type(data)}")
        
        if not labs:
            raise ValueError("研究室データが空です")
        
        labs_data = flatten_lab_data(labs)
        logger.info(f"{len(labs_data)}件の研究室データを読み込みました（直接JSON読み込み）")
        return labs_data


def analyze_single_lab(lab_id: str, config: GAConfig) -> Dict[str, Any]:
    """単一研究室の最適学生プロファイルを探索"""
    logger.info(f"\n{'='*60}")
    logger.info(f"研究室 {lab_id} の最適学生プロファイル探索を開始")
    logger.info(f"{'='*60}")
    
    # 研究室データ読み込み
    labs = load_lab_database()
    lab = next((l for l in labs if l["lab_id"] == lab_id), None)
    
    if not lab:
        logger.error(f"研究室 {lab_id} が見つかりません")
        return None
    
    logger.info(f"研究室名: {lab.get('lab_name', 'N/A')}")
    logger.info(f"専門分野: {', '.join([RESEARCH_FIELDS.get(f, f) for f in lab.get('research_fields', [])])}")
    
    # 遺伝的アルゴリズム実行
    optimizer = GeneticOptimizer(config)
    best_student = optimizer.optimize(lab)
    
    # 詳細評価
    matcher = LabMatcher(lab)
    detailed_eval = matcher.evaluate_detailed(best_student)
    
    # 結果をまとめる
    result = {
        "lab_id": lab_id,
        "lab_name": lab.get("lab_name", "N/A"),
        "research_fields": lab.get("research_fields", []),
        "optimization_config": {
            "population_size": config.population_size,
            "generations": config.generations,
            "crossover_rate": config.crossover_rate,
            "mutation_rate": config.mutation_rate
        },
        "optimal_student_profile": best_student.to_dict(),
        "evaluation_details": {
            "final_score": float(detailed_eval["final_score"]),
            "basic_score": float(detailed_eval["basic_score"]),
            "fuzzy_score": float(detailed_eval["fuzzy_score"]),
            "gaussian_score": float(detailed_eval["gaussian_score"]),
            "field_score": float(detailed_eval["field_score"]),
            "lambda_weight": float(detailed_eval["lambda_weight"]),
            "basic_weight": float(detailed_eval["basic_weight"])
        },
        "final_compatibility_score": float(best_student.fitness),
        "fitness_history": [float(f) for f in optimizer.fitness_history],
        "analysis_timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"\n最適学生プロファイル:")
    logger.info(f"  最終適合度: {best_student.fitness:.4f} ({best_student.fitness*100:.2f}%)")
    logger.info(f"  スコア内訳:")
    logger.info(f"    S_fuzzy: {detailed_eval['fuzzy_score']:.4f}")
    logger.info(f"    S_gaussian: {detailed_eval['gaussian_score']:.4f}")
    logger.info(f"    S_basic: {detailed_eval['basic_score']:.4f}")
    logger.info(f"    S_field: {detailed_eval['field_score']:.4f}")
    logger.info(f"    λ (分野比重): {detailed_eval['lambda_weight']:.4f}")
    logger.info(f"  基本項目:")
    for i, criterion in enumerate(BASIC_CRITERIA):
        value = best_student.criteria_values[i] * 9 + 1
        priority = best_student.criteria_priorities[i] * 9 + 1
        logger.info(f"    {criterion}: 値={value:.2f}, 優先度={priority:.2f}")
    logger.info(f"  分野重視度: {best_student.field_priority * 9 + 1:.2f}")
    logger.info(f"  分野興味:")
    for field, interest in best_student.field_interests.items():
        denorm_interest = interest * 9 + 1
        logger.info(f"    {RESEARCH_FIELDS.get(field, field)}: {denorm_interest:.2f}")
    
    return result


def analyze_all_labs(config: GAConfig, output_dir: str = "results/genetic_optimization"):
    """全研究室の最適学生プロファイルを探索"""
    labs = load_lab_database()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for i, lab in enumerate(labs, 1):
        lab_id = lab["lab_id"]
        logger.info(f"\n進捗: {i}/{len(labs)}")
        
        result = analyze_single_lab(lab_id, config)
        if result:
            all_results.append(result)
            
            # 個別ファイルに保存
            lab_output_dir = output_path / lab_id
            lab_output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(lab_output_dir / f"{lab_id}_optimal_student.json", 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 全体サマリーを保存
    summary = {
        "total_labs": len(all_results),
        "analysis_date": datetime.now().isoformat(),
        "algorithm_version": "v2.2",
        "config": {
            "population_size": config.population_size,
            "generations": config.generations,
            "crossover_rate": config.crossover_rate,
            "mutation_rate": config.mutation_rate
        },
        "formula": {
            "S_basic": "γ × S_fuzzy + (1 - γ) × S_gaussian (γ=0.5)",
            "S_final": "(1 - λ) × S_basic + λ × S_field",
            "lambda": "field_priority / 10"
        },
        "results": all_results
    }
    
    with open(output_path / "all_labs_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"全{len(all_results)}研究室の分析が完了しました")
    logger.info(f"結果は {output_path} に保存されました")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="遺伝的アルゴリズムによる逆引き分析 v2.2")
    parser.add_argument("--lab_id", type=str, help="分析する研究室ID（例: lab_001）")
    parser.add_argument("--all", action="store_true", help="全研究室を分析")
    parser.add_argument("--output", type=str, default="results/genetic_optimization", help="出力ディレクトリ")
    parser.add_argument("--population", type=int, default=100, help="個体数")
    parser.add_argument("--generations", type=int, default=50, help="世代数")
    parser.add_argument("--crossover_rate", type=float, default=0.8, help="交叉率")
    parser.add_argument("--mutation_rate", type=float, default=0.1, help="突然変異率")
    
    args = parser.parse_args()
    
    # 設定
    config = GAConfig(
        population_size=args.population,
        generations=args.generations,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate
    )
    
    if args.all:
        analyze_all_labs(config, args.output)
    elif args.lab_id:
        result = analyze_single_lab(args.lab_id, config)
        if result:
            output_path = Path(args.output) / args.lab_id
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / f"{args.lab_id}_optimal_student.json", 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"結果を {output_path} に保存しました")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()