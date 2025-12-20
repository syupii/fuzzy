"""
改善版 遺伝的アルゴリズムによる逆引き分析

【主な改善点】
1. research_field_matchを固定値として扱うモード追加
2. 分野興味の初期化を改善（対象研究室の分野に高い興味を持つように）
3. 分野一致率を測定・報告
4. 複数のresearch_field_match値での比較分析

使用方法:
    # 固定rfmモード（推奨）
    python improved_genetic_optimizer.py --all --fixed-rfm 5
    
    # 複数rfm値で比較
    python improved_genetic_optimizer.py --all --compare-rfm
    
    # 単一研究室
    python improved_genetic_optimizer.py --lab_id lab_001 --fixed-rfm 5
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime
from collections import defaultdict
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
    population_size: int = 100
    generations: int = 50
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elite_size: int = 5
    tournament_size: int = 5
    
    # 改善版の設定
    fixed_rfm: float = None  # Noneの場合は最適化、値を指定すると固定
    rfm_min: float = 1.0     # rfm最適化時の最小値
    rfm_max: float = 10.0    # rfm最適化時の最大値
    prioritize_target_field: bool = True  # 対象分野を優先して初期化


# 定数定義
FIELD_NAMES = [
    'ai_ml', 'image_processing', 'network_security', 'database_systems',
    'embedded_iot', 'education_linguistics', 'natural_science_math',
    'tourism_regional', 'business_decision', 'audio_processing',
    'system_ethics', 'medical_healthcare', 'web_design', 'design_visual',
    'video_animation', 'computer_music', 'game_esports', 'vr_ar_media',
    'philosophy_humanities', 'sports_science'
]

FIELD_LABELS = {
    'ai_ml': '人工知能・機械学習',
    'image_processing': '画像・映像処理',
    'network_security': 'ネットワーク・セキュリティ',
    'database_systems': 'データベース・情報システム',
    'embedded_iot': '組込み・IoT',
    'education_linguistics': '教育・言語学',
    'natural_science_math': '自然科学・数理',
    'tourism_regional': '観光情報・地域システム',
    'business_decision': '経営情報・意思決定支援',
    'audio_processing': '音声・音響情報処理',
    'system_ethics': 'システム運用・情報倫理',
    'medical_healthcare': '医療情報・ヘルスケア',
    'web_design': 'Webデザイン・UI/UX',
    'design_visual': 'デザイン・視覚表現',
    'video_animation': '映像・アニメーション',
    'computer_music': 'コンピュータ音楽・サウンドアート',
    'game_esports': 'ゲーム開発・eスポーツ',
    'vr_ar_media': 'VR/AR・メディアアート',
    'philosophy_humanities': '哲学・人文・環境行動学',
    'sports_science': 'スポーツ・体育科学'
}

FIELD_CATEGORIES = {
    "テクノロジー・システム": [
        "ai_ml", "image_processing", "network_security", "database_systems",
        "embedded_iot", "education_linguistics", "natural_science_math",
        "tourism_regional", "business_decision", "audio_processing",
        "system_ethics", "medical_healthcare"
    ],
    "クリエイティブ": ["web_design", "design_visual", "video_animation", "computer_music"],
    "エンターテイメント": ["game_esports", "vr_ar_media"],
    "人文・社会・体育": ["philosophy_humanities", "sports_science"]
}

BASIC_CRITERIA = [
    "research_intensity", "advisor_style", "team_work", "workload",
    "theory_practice", "skill_development", "lab_atmosphere",
    "flexibility", "publication_opportunity", "interdisciplinary",
    "communication_style"
]  # research_field_matchは除外（特別扱い）

DEFAULT_WEIGHTS = {
    "research_intensity": 1.2, "advisor_style": 1.2, "team_work": 1.2,
    "workload": 1.2, "theory_practice": 1.2, "skill_development": 1.0,
    "lab_atmosphere": 1.0, "flexibility": 1.0, "publication_opportunity": 1.0,
    "interdisciplinary": 0.8, "communication_style": 0.8
}


def get_field_category(field_id: str) -> str:
    """分野IDからカテゴリを取得"""
    for category, fields in FIELD_CATEGORIES.items():
        if field_id in fields:
            return category
    return None


class StudentProfile:
    """学生プロファイル（染色体）- 改善版"""
    
    def __init__(self, 
                 criteria_values: np.ndarray = None, 
                 field_interests: Dict[str, float] = None,
                 research_field_match: float = None,
                 target_field_id: str = None,
                 config: GAConfig = None):
        """
        Args:
            criteria_values: 11項目の評価値 [0, 1]の範囲（research_field_matchは除く）
            field_interests: 分野興味度 {field_id: [0, 1]}
            research_field_match: 分野重視度（Noneの場合はランダム）
            target_field_id: 対象研究室の分野ID（初期化に使用）
            config: GA設定
        """
        self.config = config or GAConfig()
        self.target_field_id = target_field_id
        
        if criteria_values is None:
            # ランダム初期化
            self.criteria_values = np.random.uniform(0, 1, len(BASIC_CRITERIA))
        else:
            self.criteria_values = criteria_values.copy()
        
        # research_field_match の設定
        if self.config.fixed_rfm is not None:
            # 固定値モード
            self.research_field_match = (self.config.fixed_rfm - 1) / 9  # 正規化
        elif research_field_match is not None:
            self.research_field_match = research_field_match
        else:
            # ランダム初期化（範囲制限付き）
            rfm_min_norm = (self.config.rfm_min - 1) / 9
            rfm_max_norm = (self.config.rfm_max - 1) / 9
            self.research_field_match = np.random.uniform(rfm_min_norm, rfm_max_norm)
        
        # field_interests の初期化
        if field_interests is None:
            self.field_interests = self._initialize_field_interests(target_field_id)
        else:
            self.field_interests = field_interests.copy()
        
        self.fitness = 0.0
    
    def _initialize_field_interests(self, target_field_id: str = None) -> Dict[str, float]:
        """分野興味の初期化（改善版）"""
        field_interests = {}
        
        if self.config.prioritize_target_field and target_field_id:
            # 対象研究室の分野に高い興味を設定
            field_interests[target_field_id] = np.random.uniform(0.7, 1.0)
            
            # 同じカテゴリの分野にもある程度の興味
            target_category = get_field_category(target_field_id)
            if target_category:
                for field in FIELD_CATEGORIES.get(target_category, []):
                    if field != target_field_id and np.random.rand() < 0.5:
                        field_interests[field] = np.random.uniform(0.4, 0.8)
            
            # 他カテゴリは低めの興味（またはなし）
            for field in FIELD_NAMES:
                if field not in field_interests and np.random.rand() < 0.2:
                    field_interests[field] = np.random.uniform(0.1, 0.4)
        else:
            # ランダムに2-4分野を選択
            num_interests = np.random.randint(2, 5)
            selected_fields = np.random.choice(FIELD_NAMES, num_interests, replace=False)
            field_interests = {
                field: np.random.uniform(0.3, 1.0) 
                for field in selected_fields
            }
        
        return field_interests
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換（1-10スケール）"""
        result = {}
        
        # 基本項目
        for criterion, val in zip(BASIC_CRITERIA, self.criteria_values):
            result[criterion] = float(val * 9 + 1)
        
        # research_field_match
        result['research_field_match'] = float(self.research_field_match * 9 + 1)
        
        # field_interests
        result['field_interests'] = {
            field: float(val * 9 + 1) 
            for field, val in self.field_interests.items()
        }
        
        return result
    
    def copy(self):
        """コピーを作成"""
        new_profile = StudentProfile(
            self.criteria_values.copy(),
            self.field_interests.copy(),
            self.research_field_match,
            self.target_field_id,
            self.config
        )
        new_profile.fitness = self.fitness
        return new_profile


class LabMatcher:
    """研究室マッチング評価器（改善版）"""
    
    def __init__(self, lab_profile: Dict[str, Any], all_labs: List[Dict] = None):
        """
        Args:
            lab_profile: 対象研究室のプロファイル
            all_labs: 全研究室リスト（順位計算用）
        """
        self.lab_profile = lab_profile
        self.all_labs = all_labs or []
        self.lab_criteria = self._get_lab_criteria(lab_profile)
        self.lab_field_id = self._get_lab_field_id(lab_profile)
        self.sigma = 0.2
    
    def _get_lab_criteria(self, profile: Dict) -> Dict[str, float]:
        """研究室の評価値を取得"""
        values = {}
        for criterion in BASIC_CRITERIA:
            val = profile.get(criterion, 5.5)
            values[criterion] = (val - 1) / 9  # 正規化
        return values
    
    def _get_lab_field_id(self, lab: Dict) -> str:
        """研究室の専門分野IDを取得"""
        if 'field_id' in lab:
            return lab['field_id']
        
        # research_fieldsから推定
        field_mapping = {
            '人工知能': 'ai_ml', '機械学習': 'ai_ml', 'AI': 'ai_ml',
            '画像処理': 'image_processing', '3DCG': 'image_processing',
            'セキュリティ': 'network_security', 'ネットワーク': 'network_security',
            'データベース': 'database_systems', 'データ工学': 'database_systems',
            'IoT': 'embedded_iot', '組み込み': 'embedded_iot',
            '教育': 'education_linguistics', '言語': 'education_linguistics',
            '数学': 'natural_science_math', '統計': 'natural_science_math',
            '観光': 'tourism_regional', '地域': 'tourism_regional',
            '音声': 'audio_processing', '音響': 'audio_processing',
            'システム': 'system_ethics',
            '医療': 'medical_healthcare',
            'Web': 'web_design', 'UI': 'web_design', 'UX': 'web_design',
            'デザイン': 'design_visual', 'イラスト': 'design_visual',
            '映像': 'video_animation', 'アニメ': 'video_animation',
            '音楽': 'computer_music', 'メディアアート': 'computer_music',
            'ゲーム': 'game_esports', 'eスポーツ': 'game_esports',
            'VR': 'vr_ar_media', 'AR': 'vr_ar_media',
            '哲学': 'philosophy_humanities', '芸術学': 'philosophy_humanities',
            'スポーツ': 'sports_science', 'トレーニング': 'sports_science'
        }
        
        for field in lab.get('research_fields', []):
            for keyword, fid in field_mapping.items():
                if keyword in field:
                    return fid
        
        return 'system_ethics'
    
    def _gaussian_similarity(self, val1: float, val2: float) -> float:
        """ガウス類似度計算"""
        diff = abs(val1 - val2)
        return math.exp(-(diff ** 2) / (2 * self.sigma ** 2))
    
    def _calculate_basic_score(self, student: StudentProfile) -> float:
        """基本項目スコア計算"""
        total_score = 0
        total_weight = 0
        
        for i, criterion in enumerate(BASIC_CRITERIA):
            student_val = student.criteria_values[i]
            lab_val = self.lab_criteria.get(criterion, 0.5)
            weight = DEFAULT_WEIGHTS.get(criterion, 1.0)
            
            sim = self._gaussian_similarity(student_val, lab_val)
            total_score += sim * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def _calculate_field_score(self, student: StudentProfile, lab_field_id: str = None) -> float:
        """分野スコア計算"""
        if lab_field_id is None:
            lab_field_id = self.lab_field_id
        
        if not lab_field_id:
            return 0.3
        
        lab_category = get_field_category(lab_field_id)
        best_score = 0.3
        
        for field_id, interest in student.field_interests.items():
            if field_id == lab_field_id:
                # 完全一致
                score = interest
            elif get_field_category(field_id) == lab_category:
                # カテゴリ一致
                score = interest * 0.7
            else:
                # 不一致
                score = 0.3
            
            best_score = max(best_score, score)
        
        return best_score
    
    def evaluate(self, student: StudentProfile) -> float:
        """単一研究室との適合度評価"""
        basic_score = self._calculate_basic_score(student)
        field_score = self._calculate_field_score(student)
        
        alpha = student.research_field_match
        beta = 1 - alpha
        
        return beta * basic_score + alpha * field_score
    
    def evaluate_with_ranking(self, student: StudentProfile) -> Tuple[float, int]:
        """
        全研究室との比較を含む評価
        
        Returns:
            (対象研究室のスコア, 対象研究室の順位)
        """
        if not self.all_labs:
            return self.evaluate(student), 1
        
        target_lab_id = self.lab_profile.get('id') or self.lab_profile.get('lab_id')
        
        scores = []
        for lab in self.all_labs:
            lab_id = lab.get('id') or lab.get('lab_id')
            lab_field_id = self._get_lab_field_id(lab)
            lab_criteria = self._get_lab_criteria(lab)
            
            # 基本スコア
            basic_score = 0
            total_weight = 0
            for i, criterion in enumerate(BASIC_CRITERIA):
                student_val = student.criteria_values[i]
                lab_val = lab_criteria.get(criterion, 0.5)
                weight = DEFAULT_WEIGHTS.get(criterion, 1.0)
                sim = self._gaussian_similarity(student_val, lab_val)
                basic_score += sim * weight
                total_weight += weight
            basic_score /= total_weight if total_weight > 0 else 1
            
            # 分野スコア
            field_score = self._calculate_field_score(student, lab_field_id)
            
            # 最終スコア
            alpha = student.research_field_match
            beta = 1 - alpha
            final_score = beta * basic_score + alpha * field_score
            
            scores.append((lab_id, final_score))
        
        # ソートして順位を計算
        scores.sort(key=lambda x: -x[1])
        
        target_score = 0
        target_rank = len(scores)
        
        for rank, (lab_id, score) in enumerate(scores, 1):
            if lab_id == target_lab_id:
                target_score = score
                target_rank = rank
                break
        
        return target_score, target_rank


class ImprovedGeneticOptimizer:
    """改善版遺伝的アルゴリズム"""
    
    def __init__(self, config: GAConfig = None):
        self.config = config or GAConfig()
        self.population: List[StudentProfile] = []
        self.best_individual: StudentProfile = None
        self.best_fitness: float = 0.0
        self.best_rank: int = 999
        self.fitness_history: List[float] = []
        self.rank_history: List[int] = []
    
    def initialize_population(self, target_field_id: str = None) -> List[StudentProfile]:
        """初期個体群の生成"""
        return [
            StudentProfile(
                target_field_id=target_field_id,
                config=self.config
            )
            for _ in range(self.config.population_size)
        ]
    
    def evaluate_population(self, population: List[StudentProfile], matcher: LabMatcher):
        """個体群の評価（順位も考慮）"""
        for individual in population:
            score, rank = matcher.evaluate_with_ranking(individual)
            # フィットネス = スコア + 順位ボーナス
            # 1位なら大きなボーナス
            rank_bonus = 0.5 if rank == 1 else 0.2 / rank
            individual.fitness = score + rank_bonus
            individual.rank = rank
    
    def tournament_selection(self, population: List[StudentProfile]) -> StudentProfile:
        """トーナメント選択"""
        tournament = np.random.choice(population, self.config.tournament_size, replace=False)
        # 順位が良い個体を優先
        return min(tournament, key=lambda x: (x.rank, -x.fitness))
    
    def crossover(self, parent1: StudentProfile, parent2: StudentProfile) -> Tuple[StudentProfile, StudentProfile]:
        """交叉"""
        if np.random.rand() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # 基本項目の交叉
        mask = np.random.rand(len(BASIC_CRITERIA)) < 0.5
        child1_criteria = np.where(mask, parent1.criteria_values, parent2.criteria_values)
        child2_criteria = np.where(mask, parent2.criteria_values, parent1.criteria_values)
        
        # research_field_matchの交叉（固定モードでない場合）
        if self.config.fixed_rfm is None:
            if np.random.rand() < 0.5:
                child1_rfm = parent1.research_field_match
                child2_rfm = parent2.research_field_match
            else:
                child1_rfm = parent2.research_field_match
                child2_rfm = parent1.research_field_match
        else:
            child1_rfm = parent1.research_field_match
            child2_rfm = parent2.research_field_match
        
        # 分野興味の交叉
        all_fields = set(parent1.field_interests.keys()) | set(parent2.field_interests.keys())
        
        child1_fields = {}
        child2_fields = {}
        
        for field in all_fields:
            p1_val = parent1.field_interests.get(field, 0)
            p2_val = parent2.field_interests.get(field, 0)
            
            if np.random.rand() < 0.5:
                if p1_val > 0:
                    child1_fields[field] = p1_val
                if p2_val > 0:
                    child2_fields[field] = p2_val
            else:
                if p2_val > 0:
                    child1_fields[field] = p2_val
                if p1_val > 0:
                    child2_fields[field] = p1_val
        
        child1 = StudentProfile(
            child1_criteria, child1_fields, child1_rfm,
            parent1.target_field_id, self.config
        )
        child2 = StudentProfile(
            child2_criteria, child2_fields, child2_rfm,
            parent2.target_field_id, self.config
        )
        
        return child1, child2
    
    def mutate(self, individual: StudentProfile):
        """突然変異"""
        # 基本項目の突然変異
        for i in range(len(BASIC_CRITERIA)):
            if np.random.rand() < self.config.mutation_rate:
                individual.criteria_values[i] += np.random.normal(0, 0.1)
                individual.criteria_values[i] = np.clip(individual.criteria_values[i], 0, 1)
        
        # research_field_matchの突然変異（固定モードでない場合）
        if self.config.fixed_rfm is None:
            if np.random.rand() < self.config.mutation_rate:
                individual.research_field_match += np.random.normal(0, 0.1)
                rfm_min_norm = (self.config.rfm_min - 1) / 9
                rfm_max_norm = (self.config.rfm_max - 1) / 9
                individual.research_field_match = np.clip(
                    individual.research_field_match, rfm_min_norm, rfm_max_norm
                )
        
        # 分野興味の突然変異
        target_field = individual.target_field_id
        
        # 対象分野の興味を維持・強化
        if target_field and np.random.rand() < self.config.mutation_rate * 2:
            if target_field in individual.field_interests:
                individual.field_interests[target_field] = np.clip(
                    individual.field_interests[target_field] + np.random.normal(0.1, 0.05),
                    0.5, 1.0
                )
            else:
                individual.field_interests[target_field] = np.random.uniform(0.7, 1.0)
        
        # 他分野の変異
        for field in list(individual.field_interests.keys()):
            if field != target_field and np.random.rand() < self.config.mutation_rate:
                individual.field_interests[field] += np.random.normal(0, 0.15)
                individual.field_interests[field] = np.clip(
                    individual.field_interests[field], 0, 1
                )
                # 興味が低すぎたら削除
                if individual.field_interests[field] < 0.1:
                    del individual.field_interests[field]
        
        # 新しい分野を追加
        if np.random.rand() < self.config.mutation_rate * 0.5:
            available = [f for f in FIELD_NAMES if f not in individual.field_interests]
            if available:
                new_field = np.random.choice(available)
                individual.field_interests[new_field] = np.random.uniform(0.3, 0.7)
    
    def optimize(self, lab_profile: Dict[str, Any], all_labs: List[Dict] = None) -> StudentProfile:
        """最適化実行"""
        matcher = LabMatcher(lab_profile, all_labs)
        target_field_id = matcher.lab_field_id
        
        # 初期化
        self.population = self.initialize_population(target_field_id)
        self.evaluate_population(self.population, matcher)
        
        # 最良個体を記録
        best = min(self.population, key=lambda x: (x.rank, -x.fitness))
        self.best_individual = best.copy()
        self.best_fitness = best.fitness
        self.best_rank = best.rank
        self.fitness_history = [self.best_fitness]
        self.rank_history = [self.best_rank]
        
        logger.info(f"初期世代: 最良順位={self.best_rank}, 適合度={self.best_fitness:.4f}")
        
        # 世代ループ
        for generation in range(self.config.generations):
            # エリート保存
            elites = sorted(self.population, key=lambda x: (x.rank, -x.fitness))[:self.config.elite_size]
            
            # 新世代生成
            new_population = [e.copy() for e in elites]
            
            while len(new_population) < self.config.population_size:
                parent1 = self.tournament_selection(self.population)
                parent2 = self.tournament_selection(self.population)
                
                child1, child2 = self.crossover(parent1, parent2)
                
                self.mutate(child1)
                self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            self.population = new_population[:self.config.population_size]
            self.evaluate_population(self.population, matcher)
            
            # 最良個体更新
            current_best = min(self.population, key=lambda x: (x.rank, -x.fitness))
            if current_best.rank < self.best_rank or \
               (current_best.rank == self.best_rank and current_best.fitness > self.best_fitness):
                self.best_individual = current_best.copy()
                self.best_fitness = current_best.fitness
                self.best_rank = current_best.rank
            
            self.fitness_history.append(self.best_fitness)
            self.rank_history.append(self.best_rank)
            
            if (generation + 1) % 10 == 0:
                avg_rank = np.mean([ind.rank for ind in self.population])
                logger.info(f"世代 {generation + 1}/{self.config.generations}: "
                          f"最良順位={self.best_rank}, 平均順位={avg_rank:.1f}")
        
        logger.info(f"最適化完了: 最終順位={self.best_rank}, 適合度={self.best_fitness:.4f}")
        
        return self.best_individual


def load_lab_database() -> List[Dict]:
    """研究室データベースを読み込み"""
    import os
    
    possible_paths = [
        "data/labs_database.json",
        "backend/data/labs_database.json",
        "../data/labs_database.json",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"研究室データを読み込み: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, dict):
                labs = data.get("labs", [])
            else:
                labs = data
            
            # フラット化
            flattened = []
            for lab in labs:
                flat_lab = lab.copy()
                if "features" in lab:
                    for key, value in lab["features"].items():
                        flat_lab[key] = value
                flattened.append(flat_lab)
            
            return flattened
    
    raise FileNotFoundError("研究室データベースが見つかりません")


def analyze_single_lab(lab_id: str, config: GAConfig, all_labs: List[Dict]) -> Dict[str, Any]:
    """単一研究室の最適学生プロファイルを探索"""
    lab = next((l for l in all_labs if (l.get('id') or l.get('lab_id')) == lab_id), None)
    
    if not lab:
        logger.error(f"研究室 {lab_id} が見つかりません")
        return None
    
    lab_name = lab.get('name') or lab.get('lab_name')
    logger.info(f"\n{'='*60}")
    logger.info(f"研究室: {lab_name} ({lab_id})")
    logger.info(f"{'='*60}")
    
    # 分野ID取得
    matcher = LabMatcher(lab)
    lab_field_id = matcher.lab_field_id
    logger.info(f"専門分野: {FIELD_LABELS.get(lab_field_id, lab_field_id)}")
    
    if config.fixed_rfm is not None:
        logger.info(f"research_field_match: {config.fixed_rfm}（固定）")
    
    # 遺伝的アルゴリズム実行
    optimizer = ImprovedGeneticOptimizer(config)
    best_student = optimizer.optimize(lab, all_labs)
    
    # 分野一致を確認
    top_field = max(best_student.field_interests.items(), key=lambda x: x[1]) if best_student.field_interests else (None, 0)
    field_match = top_field[0] == lab_field_id if top_field[0] else False
    
    result = {
        "lab_id": lab_id,
        "lab_name": lab_name,
        "lab_field_id": lab_field_id,
        "optimization_config": {
            "population_size": config.population_size,
            "generations": config.generations,
            "crossover_rate": config.crossover_rate,
            "mutation_rate": config.mutation_rate,
            "fixed_rfm": config.fixed_rfm
        },
        "optimal_student_profile": best_student.to_dict(),
        "final_rank": optimizer.best_rank,
        "final_fitness": float(optimizer.best_fitness),
        "fitness_history": [float(f) for f in optimizer.fitness_history],
        "rank_history": optimizer.rank_history,
        "field_analysis": {
            "top_field_interest": {
                "field_id": top_field[0],
                "field_label": FIELD_LABELS.get(top_field[0], top_field[0]) if top_field[0] else None,
                "value": float(top_field[1] * 9 + 1) if top_field[1] else 0
            },
            "field_match": field_match,
            "match_status": "✓ 一致" if field_match else "✗ 不一致"
        },
        "analysis_timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"\n最適プロファイル:")
    logger.info(f"  最終順位: {optimizer.best_rank}")
    logger.info(f"  分野一致: {result['field_analysis']['match_status']}")
    logger.info(f"  research_field_match: {best_student.to_dict()['research_field_match']:.2f}")
    
    return result


def analyze_all_labs_with_fixed_rfm(config: GAConfig, output_dir: str = "results/improved_ga"):
    """固定rfmで全研究室を分析"""
    all_labs = load_lab_database()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    field_match_count = 0
    
    for i, lab in enumerate(all_labs, 1):
        lab_id = lab.get('id') or lab.get('lab_id')
        logger.info(f"\n進捗: {i}/{len(all_labs)}")
        
        result = analyze_single_lab(lab_id, config, all_labs)
        if result:
            all_results.append(result)
            if result['field_analysis']['field_match']:
                field_match_count += 1
    
    # サマリー
    summary = {
        "total_labs": len(all_results),
        "analysis_date": datetime.now().isoformat(),
        "config": {
            "population_size": config.population_size,
            "generations": config.generations,
            "fixed_rfm": config.fixed_rfm
        },
        "field_match_summary": {
            "match_count": field_match_count,
            "total_count": len(all_results),
            "match_rate": field_match_count / len(all_results) if all_results else 0
        },
        "results": all_results
    }
    
    # 保存
    rfm_str = f"rfm{config.fixed_rfm}" if config.fixed_rfm else "rfm_optimized"
    with open(output_path / f"all_labs_{rfm_str}.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"分析完了！")
    logger.info(f"研究室数: {len(all_results)}")
    logger.info(f"分野一致率: {field_match_count}/{len(all_results)} ({summary['field_match_summary']['match_rate']*100:.1f}%)")
    logger.info(f"結果: {output_path}")
    logger.info(f"{'='*60}")
    
    return summary


def compare_rfm_values(output_dir: str = "results/improved_ga"):
    """複数のresearch_field_match値で比較分析"""
    rfm_values = [1, 3, 5, 7, 10]
    all_labs = load_lab_database()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    comparison_results = {}
    
    for rfm in rfm_values:
        logger.info(f"\n{'#'*70}")
        logger.info(f"# research_field_match = {rfm} で分析")
        logger.info(f"{'#'*70}")
        
        config = GAConfig(
            population_size=50,  # 比較用に軽量化
            generations=30,
            fixed_rfm=rfm
        )
        
        summary = analyze_all_labs_with_fixed_rfm(config, output_dir)
        comparison_results[rfm] = {
            "match_rate": summary["field_match_summary"]["match_rate"],
            "match_count": summary["field_match_summary"]["match_count"],
            "results": [
                {
                    "lab_name": r["lab_name"],
                    "field_match": r["field_analysis"]["field_match"],
                    "final_rank": r["final_rank"]
                }
                for r in summary["results"]
            ]
        }
    
    # 比較サマリー
    comparison_summary = {
        "analysis_date": datetime.now().isoformat(),
        "rfm_values_tested": rfm_values,
        "comparison": comparison_results
    }
    
    with open(output_path / "rfm_comparison.json", 'w', encoding='utf-8') as f:
        json.dump(comparison_summary, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n{'='*70}")
    logger.info("比較結果:")
    for rfm, data in comparison_results.items():
        logger.info(f"  rfm={rfm}: 分野一致率 {data['match_rate']*100:.1f}%")
    logger.info(f"{'='*70}")
    
    return comparison_summary


def main():
    parser = argparse.ArgumentParser(description="改善版遺伝的アルゴリズム逆引き分析")
    parser.add_argument("--lab_id", type=str, help="分析する研究室ID")
    parser.add_argument("--all", action="store_true", help="全研究室を分析")
    parser.add_argument("--compare-rfm", action="store_true", help="複数rfm値で比較")
    parser.add_argument("--fixed-rfm", type=float, default=5, help="固定するresearch_field_match値")
    parser.add_argument("--output", type=str, default="results/improved_ga", help="出力ディレクトリ")
    parser.add_argument("--population", type=int, default=100, help="個体数")
    parser.add_argument("--generations", type=int, default=50, help="世代数")
    
    args = parser.parse_args()
    
    config = GAConfig(
        population_size=args.population,
        generations=args.generations,
        fixed_rfm=args.fixed_rfm
    )
    
    if args.compare_rfm:
        compare_rfm_values(args.output)
    elif args.all:
        analyze_all_labs_with_fixed_rfm(config, args.output)
    elif args.lab_id:
        all_labs = load_lab_database()
        result = analyze_single_lab(args.lab_id, config, all_labs)
        if result:
            output_path = Path(args.output) / args.lab_id
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / f"{args.lab_id}_optimal.json", 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()