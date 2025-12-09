#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遺伝的アルゴリズムによる逆引き分析
研究室ごとの理想的な学生プロファイルを探索

使用方法:
    python genetic_optimizer_reverse_lookup.py --lab_id lab_001 --target_rank 1
    python genetic_optimizer_reverse_lookup.py --all --output results/
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import logging
from datetime import datetime

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

# 基本12項目
BASIC_CRITERIA = [
    "research_intensity",
    "advisor_style",
    "team_work",
    "workload",
    "theory_practice",
    "research_field_match",
    "skill_development",
    "lab_atmosphere",
    "flexibility",
    "publication_opportunity",
    "interdisciplinary",
    "communication_style"
]


class StudentProfile:
    """学生プロファイル（染色体）"""
    
    def __init__(self, criteria_values: np.ndarray = None, field_interests: Dict[str, float] = None):
        """
        Args:
            criteria_values: 12項目の評価値 [0, 1]の範囲
            field_interests: 分野興味度 {field_id: [0, 1]}
        """
        if criteria_values is None:
            # ランダム初期化
            self.criteria_values = np.random.uniform(0, 1, len(BASIC_CRITERIA))
        else:
            self.criteria_values = criteria_values.copy()
        
        if field_interests is None:
            # ランダムに2-3分野を選択して興味を持つ
            num_interests = np.random.randint(2, 4)
            selected_fields = np.random.choice(list(RESEARCH_FIELDS.keys()), num_interests, replace=False)
            self.field_interests = {
                field: np.random.uniform(0.5, 1.0) 
                for field in selected_fields
            }
        else:
            self.field_interests = field_interests.copy()
        
        self.fitness = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換（1-10スケールに戻す）"""
        return {
            **{criterion: float(val * 9 + 1) for criterion, val in zip(BASIC_CRITERIA, self.criteria_values)},
            "field_interests": {field: float(val * 9 + 1) for field, val in self.field_interests.items()}
        }
    
    def copy(self):
        """コピーを作成"""
        return StudentProfile(self.criteria_values.copy(), self.field_interests.copy())


class LabMatcher:
    """研究室マッチング評価器"""
    
    def __init__(self, lab_profile: Dict[str, Any]):
        """
        Args:
            lab_profile: 研究室プロファイル
        """
        self.lab_profile = lab_profile
        self.lab_criteria = self._normalize_criteria(lab_profile)
        self.lab_fields = lab_profile.get("research_fields", [])
        
        # ガウス類似度のパラメータ
        self.sigma = 0.2
    
    def _normalize_criteria(self, profile: Dict[str, Any]) -> np.ndarray:
        """評価値を正規化 [1, 10] -> [0, 1]"""
        values = []
        for criterion in BASIC_CRITERIA:
            val = profile.get(criterion, 5.5)  # デフォルト中央値
            normalized = (val - 1) / 9
            values.append(normalized)
        return np.array(values)
    
    def _gaussian_similarity(self, val1: float, val2: float) -> float:
        """ガウス類似度計算"""
        diff = abs(val1 - val2)
        return np.exp(-(diff ** 2) / (2 * self.sigma ** 2))
    
    def _calculate_basic_score(self, student: StudentProfile) -> float:
        """基本項目スコア計算"""
        similarities = np.array([
            self._gaussian_similarity(s, l)
            for s, l in zip(student.criteria_values, self.lab_criteria)
        ])
        
        # 優先度は遺伝的アルゴリズムでは一律1.0として扱う
        # （最適化するのは学生の評価値のみ）
        weights = np.ones(len(BASIC_CRITERIA))
        
        weighted_sum = np.sum(similarities * weights)
        total_weight = np.sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _get_field_category(self, field_id: str) -> str:
        """分野のカテゴリを取得"""
        for category, fields in FIELD_CATEGORIES.items():
            if field_id in fields:
                return category
        return None
    
    def _calculate_field_score(self, student: StudentProfile) -> float:
        """分野スコア計算"""
        if not self.lab_fields:
            return 0.3  # デフォルト
        
        max_score = 0.0
        
        for lab_field in self.lab_fields:
            if lab_field in student.field_interests:
                # 完全一致
                interest = student.field_interests[lab_field]
                score = interest
                max_score = max(max_score, score)
            else:
                # カテゴリ一致をチェック
                lab_category = self._get_field_category(lab_field)
                for student_field, interest in student.field_interests.items():
                    student_category = self._get_field_category(student_field)
                    if lab_category and student_category and lab_category == student_category:
                        # カテゴリ一致
                        score = interest * 0.7
                        max_score = max(max_score, score)
        
        # 全く一致しない場合
        if max_score == 0.0:
            max_score = 0.3
        
        return max_score
    
    def evaluate(self, student: StudentProfile) -> float:
        """総合評価"""
        basic_score = self._calculate_basic_score(student)
        field_score = self._calculate_field_score(student)
        
        # research_field_matchを取得（分野重視度）
        field_match_idx = BASIC_CRITERIA.index("research_field_match")
        alpha = student.criteria_values[field_match_idx]  # [0, 1]
        beta = 1 - alpha
        
        final_score = beta * basic_score + alpha * field_score
        
        return final_score


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
    
    def crossover(self, parent1: StudentProfile, parent2: StudentProfile) -> Tuple[StudentProfile, StudentProfile]:
        """交叉（一様交叉）"""
        if np.random.rand() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # 基本項目の交叉
        mask = np.random.rand(len(BASIC_CRITERIA)) < 0.5
        child1_criteria = np.where(mask, parent1.criteria_values, parent2.criteria_values)
        child2_criteria = np.where(mask, parent2.criteria_values, parent1.criteria_values)
        
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
        
        child1 = StudentProfile(child1_criteria, child1_fields)
        child2 = StudentProfile(child2_criteria, child2_fields)
        
        return child1, child2
    
    def mutate(self, individual: StudentProfile):
        """突然変異"""
        # 基本項目の突然変異
        for i in range(len(BASIC_CRITERIA)):
            if np.random.rand() < self.config.mutation_rate:
                # ガウス変異
                individual.criteria_values[i] += np.random.normal(0, 0.1)
                individual.criteria_values[i] = np.clip(individual.criteria_values[i], 0, 1)
        
        # 分野興味の突然変異
        if np.random.rand() < self.config.mutation_rate:
            # 新しい分野を追加 or 既存分野を削除
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
            new_population = elites.copy()
            
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


def load_lab_database(filepath: str = "backend/data/labs_database.json") -> List[Dict]:
    """研究室データベースを読み込み"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("labs", [])


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
        "final_compatibility_score": float(best_student.fitness),
        "fitness_history": [float(f) for f in optimizer.fitness_history],
        "analysis_timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"\n最適学生プロファイル:")
    logger.info(f"  最終適合度: {best_student.fitness:.4f} ({best_student.fitness*100:.2f}%)")
    logger.info(f"  基本項目:")
    for criterion, value in zip(BASIC_CRITERIA, best_student.criteria_values):
        denorm_value = value * 9 + 1
        logger.info(f"    {criterion}: {denorm_value:.2f}")
    logger.info(f"  分野興味:")
    for field, interest in best_student.field_interests.items():
        denorm_interest = interest * 9 + 1
        logger.info(f"    {RESEARCH_FIELDS[field]}: {denorm_interest:.2f}")
    
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
        "config": {
            "population_size": config.population_size,
            "generations": config.generations
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
    parser = argparse.ArgumentParser(description="遺伝的アルゴリズムによる逆引き分析")
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
        # 全研究室分析
        analyze_all_labs(config, args.output)
    elif args.lab_id:
        # 単一研究室分析
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