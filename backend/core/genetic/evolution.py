"""
遺伝的アルゴリズム - 進化エンジン
決定木の最適化を実行
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Callable
from dataclasses import dataclass
import time

from .individual import Individual, FuzzyTreeGene
from .population import Population, PopulationConfig


@dataclass
class EvolutionConfig:
    """進化設定"""
    generations: int = 100
    population_size: int = 50
    elite_size: int = 5
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    tournament_size: int = 5
    selection_method: str = "tournament"
    convergence_threshold: float = 0.001
    convergence_patience: int = 15
    verbose: bool = True


class FuzzyTreeEvaluator:
    """ファジィ決定木の評価器"""
    
    def __init__(self, training_data: List[Dict[str, Any]]):
        """
        Args:
            training_data: 訓練データのリスト
                          各要素は {"profile": {...}, "label": "...", "score": ...}
        """
        self.training_data = training_data
    
    def triangular_membership(self, x: float, a: float, b: float, c: float) -> float:
        """三角型メンバーシップ関数"""
        if x <= a or x >= c:
            return 0.0
        elif x == b:
            return 1.0
        elif x < b:
            return (x - a) / (b - a) if (b - a) > 0 else 0.0
        else:
            return (c - x) / (c - b) if (c - b) > 0 else 0.0
    
    def classify_with_gene(self, profile: Dict[str, float], gene: FuzzyTreeGene) -> str:
        """遺伝子を使ってプロファイルを分類
        
        Args:
            profile: 学生プロファイル
            gene: ファジィ決定木の遺伝子
            
        Returns:
            分類されたクラスタ
        """
        # Level 1: 特徴値を取得
        level1_value = profile.get(gene.level1_feature, 0.5)
        
        # Level 1: メンバーシップ度を計算
        low_params = gene.membership_params["low"]
        medium_params = gene.membership_params["medium"]
        high_params = gene.membership_params["high"]
        
        low_degree = self.triangular_membership(level1_value, *low_params)
        medium_degree = self.triangular_membership(level1_value, *medium_params)
        high_degree = self.triangular_membership(level1_value, *high_params)
        
        # 最大メンバーシップのブランチを選択
        memberships = {"low": low_degree, "medium": medium_degree, "high": high_degree}
        level1_branch = max(memberships, key=memberships.get)
        
        # Level 2: 特徴を取得
        level2_feature = gene.level2_features[level1_branch]
        level2_value = profile.get(level2_feature, 0.5)
        level2_threshold = gene.level2_thresholds[level1_branch]
        
        # Level 2: 分類
        if level2_value >= level2_threshold:
            level2_cluster = "high"
        else:
            level2_cluster = "low"
        
        # 最終クラスタ
        final_cluster = f"{level1_branch}_{level2_cluster}"
        
        return final_cluster
    
    def calculate_compatibility_with_gene(self, 
                                         student: Dict[str, float],
                                         lab: Dict[str, float],
                                         gene: FuzzyTreeGene) -> float:
        """遺伝子を使って適合度を計算
        
        Args:
            student: 学生プロファイル
            lab: 研究室プロファイル
            gene: ファジィ決定木の遺伝子
            
        Returns:
            適合度スコア（0〜1）
        """
        weighted_sum = 0.0
        
        # 13項目の適合度を計算
        for feature, weight in gene.importance_weights.items():
            student_val = student.get(feature, 0.5)
            lab_val = lab.get(feature, 0.5)
            
            # 差分
            diff = abs(student_val - lab_val)
            
            # ガウス型類似度
            similarity = np.exp(-0.5 * (diff / 0.2) ** 2)
            
            weighted_sum += similarity * weight
        
        return weighted_sum
    
    def evaluate_individual(self, individual: Individual) -> float:
        """個体の適合度を評価
        
        Args:
            individual: 評価する個体
            
        Returns:
            適合度スコア（0〜1）
        """
        gene = individual.gene
        total_score = 0.0
        
        for data in self.training_data:
            profile = data["profile"]
            true_label = data.get("label")
            true_score = data.get("score")
            
            # 分類精度の評価
            if true_label:
                predicted_label = self.classify_with_gene(profile, gene)
                
                # ラベルが一致すれば加点
                if predicted_label == true_label:
                    total_score += 1.0
                else:
                    # 部分的に一致（Level1だけ一致など）
                    if predicted_label.split("_")[0] == true_label.split("_")[0]:
                        total_score += 0.5
            
            # スコア予測精度の評価（研究室適合度の場合）
            if true_score is not None and "lab_profile" in data:
                lab_profile = data["lab_profile"]
                predicted_score = self.calculate_compatibility_with_gene(
                    profile, lab_profile, gene
                )
                
                # 誤差が小さいほど高得点
                error = abs(predicted_score - true_score)
                score_accuracy = max(0, 1.0 - error)
                total_score += score_accuracy
        
        # 平均スコア
        if len(self.training_data) > 0:
            avg_score = total_score / len(self.training_data)
        else:
            avg_score = 0.0
        
        # ペナルティ: 複雑さ
        # 使用している特徴の数が少ないほど良い（シンプルな木を優遇）
        used_features = {gene.level1_feature}
        used_features.update(gene.level2_features.values())
        complexity_penalty = len(used_features) / len(Individual.AVAILABLE_FEATURES) * 0.1
        
        final_score = max(0.0, avg_score - complexity_penalty)
        
        return final_score


class EvolutionEngine:
    """進化エンジン"""
    
    def __init__(self, config: EvolutionConfig = None):
        """
        Args:
            config: 進化設定
        """
        self.config = config or EvolutionConfig()
        self.population = None
        self.evaluator = None
        self.evolution_history = []
    
    def optimize(self, training_data: List[Dict[str, Any]]) -> Tuple[Individual, List[Dict]]:
        """最適化を実行
        
        Args:
            training_data: 訓練データ
            
        Returns:
            (最良個体, 進化履歴)
        """
        print("=" * 70)
        print("🧬 遺伝的アルゴリズムによる決定木最適化開始")
        print("=" * 70)
        
        start_time = time.time()
        
        # 評価器を初期化
        self.evaluator = FuzzyTreeEvaluator(training_data)
        
        # 集団を初期化
        pop_config = PopulationConfig(
            population_size=self.config.population_size,
            elite_size=self.config.elite_size,
            crossover_rate=self.config.crossover_rate,
            mutation_rate=self.config.mutation_rate,
            tournament_size=self.config.tournament_size
        )
        self.population = Population(pop_config)
        self.population.initialize()
        
        # 初期評価
        self.population.evaluate(self.evaluator.evaluate_individual)
        
        if self.config.verbose:
            stats = self.population.get_statistics()
            print(f"\n📊 初期世代:")
            print(f"   最良適合度: {stats['best_fitness']:.4f}")
            print(f"   平均適合度: {stats['avg_fitness']:.4f}")
            print(f"   多様性: {stats['diversity']:.4f}")
        
        # 進化ループ
        for generation in range(self.config.generations):
            # 次世代生成
            self.population.evolve(selection_method=self.config.selection_method)
            
            # 評価
            self.population.evaluate(self.evaluator.evaluate_individual)
            
            # 統計情報を記録
            stats = self.population.get_statistics()
            self.evolution_history.append(stats)
            
            # 進捗表示
            if self.config.verbose and (generation + 1) % 10 == 0:
                print(f"\n📈 世代 {generation + 1}:")
                print(f"   最良適合度: {stats['best_fitness']:.4f}")
                print(f"   平均適合度: {stats['avg_fitness']:.4f}")
                print(f"   多様性: {stats['diversity']:.4f}")
                
                best_gene = stats['best_individual'].gene
                print(f"   最良個体のLevel1特徴: {best_gene.level1_feature}")
            
            # 収束判定
            if self.population.has_converged(
                threshold=self.config.convergence_threshold,
                patience=self.config.convergence_patience
            ):
                if self.config.verbose:
                    print(f"\n✅ 収束しました (世代 {generation + 1})")
                break
        
        elapsed_time = time.time() - start_time
        
        # 最終結果
        best_individual = self.population.best_individual
        final_stats = self.population.get_statistics()
        
        print("\n" + "=" * 70)
        print("🎉 最適化完了")
        print("=" * 70)
        print(f"⏱️  経過時間: {elapsed_time:.2f}秒")
        print(f"🏆 最良適合度: {final_stats['best_fitness']:.4f}")
        print(f"📊 最終世代: {final_stats['generation']}")
        print(f"🌳 最適決定木構造:")
        print(f"   Level1特徴: {best_individual.gene.level1_feature}")
        print(f"   Level2特徴: {best_individual.gene.level2_features}")
        print(f"   重要度トップ3:")
        
        sorted_weights = sorted(
            best_individual.gene.importance_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for i, (feature, weight) in enumerate(sorted_weights[:3], 1):
            print(f"      {i}. {feature}: {weight:.4f}")
        
        return best_individual, self.evolution_history
    
    def get_optimized_decision_tree(self) -> Dict[str, Any]:
        """最適化された決定木の設定を取得
        
        Returns:
            決定木設定の辞書
        """
        if self.population is None or self.population.best_individual is None:
            raise ValueError("最適化が実行されていません")
        
        best_gene = self.population.best_individual.gene
        
        return {
            "level1_feature": best_gene.level1_feature,
            "level1_thresholds": {
                "low": best_gene.level1_threshold_low,
                "high": best_gene.level1_threshold_high
            },
            "level2_features": best_gene.level2_features.copy(),
            "level2_thresholds": best_gene.level2_thresholds.copy(),
            "membership_params": best_gene.membership_params.copy(),
            "importance_weights": best_gene.importance_weights.copy(),
            "fitness": self.population.best_individual.fitness
        }


# 使用例とテスト
if __name__ == "__main__":
    print("=" * 70)
    print("遺伝的アルゴリズム - 進化エンジンテスト")
    print("=" * 70)
    
    # サンプル訓練データ生成
    print("\n📚 訓練データ生成中...")
    training_data = []
    
    for i in range(50):
        profile = {
            "research_intensity": np.random.uniform(0.0, 1.0),
            "advisor_style": np.random.uniform(0.0, 1.0),
            "team_work": np.random.uniform(0.0, 1.0),
            "workload": np.random.uniform(0.0, 1.0),
            "theory_practice": np.random.uniform(0.0, 1.0),
            "research_field_match": np.random.uniform(0.0, 1.0),
            "skill_development": np.random.uniform(0.0, 1.0),
            "lab_atmosphere": np.random.uniform(0.0, 1.0),
            "flexibility": np.random.uniform(0.0, 1.0),
            "publication_opportunity": np.random.uniform(0.0, 1.0),
            "interdisciplinary": np.random.uniform(0.0, 1.0),
            "communication_style": np.random.uniform(0.0, 1.0),
            "innovation_risk": np.random.uniform(0.0, 1.0)
        }
        
        # ラベル生成（ダミー）
        if profile["research_intensity"] > 0.7:
            if profile["team_work"] > 0.7:
                label = "high_high"
            else:
                label = "high_low"
        elif profile["research_intensity"] > 0.4:
            if profile["flexibility"] > 0.6:
                label = "medium_high"
            else:
                label = "medium_low"
        else:
            if profile["lab_atmosphere"] > 0.6:
                label = "low_high"
            else:
                label = "low_low"
        
        training_data.append({
            "profile": profile,
            "label": label
        })
    
    print(f"✅ 訓練データ: {len(training_data)}サンプル")
    
    # 進化エンジン実行
    config = EvolutionConfig(
        generations=30,
        population_size=20,
        elite_size=2,
        verbose=True
    )
    
    engine = EvolutionEngine(config)
    best_individual, history = engine.optimize(training_data)
    
    # 最適化された決定木を取得
    optimal_tree = engine.get_optimized_decision_tree()
    print(f"\n📋 最適決定木設定:")
    print(f"   適合度: {optimal_tree['fitness']:.4f}")
    
    print("\n" + "=" * 70)