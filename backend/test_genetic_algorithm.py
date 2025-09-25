#!/usr/bin/env python3
"""
遺伝的アルゴリズムの動作確認テストスクリプト
本物の遺伝的アルゴリズムが実装されているかを検証
"""

import sys
import os
import random
import numpy as np
import time

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from core.genetic.evolution import EvolutionEngine, EvolutionConfig, Individual

def test_basic_genetic_algorithm():
    """基本的な遺伝的アルゴリズムのテスト"""
    
    print("\n" + "=" * 70)
    print("🧬 テスト1: 基本的な遺伝的アルゴリズムの動作確認")
    print("=" * 70)
    
    # テスト用適応度関数：全ての重みを1.0に近づける
    def simple_fitness(chromosome):
        target = 1.0
        error = sum((x - target) ** 2 for x in chromosome)
        fitness = 1.0 / (error + 1e-6)
        return fitness
    
    # 設定
    config = EvolutionConfig(
        population_size=20,
        generations=30,
        mutation_rate=0.15,
        crossover_rate=0.8,
        elitism_rate=0.1,
        max_stagnation=10
    )
    
    print(f"\n📋 設定:")
    print(f"  集団サイズ: {config.population_size}")
    print(f"  世代数: {config.generations}")
    print(f"  変異率: {config.mutation_rate}")
    print(f"  交叉率: {config.crossover_rate}")
    
    # エンジン作成
    engine = EvolutionEngine(config)
    
    # 初期集団の状態確認
    engine.initialize_population()
    initial_individual = engine.population[0]
    initial_fitness = simple_fitness(initial_individual.chromosome)
    
    print(f"\n🌱 初期状態:")
    print(f"  初期個体の染色体（先頭5要素）: {[f'{x:.3f}' for x in initial_individual.chromosome[:5]]}")
    print(f"  初期適応度: {initial_fitness:.6f}")
    
    # 進化実行
    result = engine.evolve(simple_fitness, verbose=True)
    
    # 結果検証
    print(f"\n✅ 最終結果:")
    print(f"  最良適応度: {result.best_fitness:.6f}")
    print(f"  世代数: {result.generation}")
    print(f"  改善度: {result.best_fitness / initial_fitness:.2f}x")
    print(f"  最良個体（先頭5要素）: {[f'{x:.3f}' for x in result.best_individual.chromosome[:5]]}")
    
    # 改善が見られたか確認
    improvement = result.best_fitness > initial_fitness
    print(f"\n🎯 改善判定: {'✅ PASS - 適応度が改善されました' if improvement else '❌ FAIL - 改善が見られません'}")
    
    return improvement

def test_lab_matching_optimization():
    """研究室マッチング最適化のテスト"""
    
    print("\n" + "=" * 70)
    print("🧬 テスト2: 研究室マッチング最適化シミュレーション")
    print("=" * 70)
    
    # サンプル学生プロファイル
    student_profiles = [
        {
            "research_intensity": 0.8,
            "advisor_style": 0.7,
            "team_work": 0.6,
            "workload": 0.75,
            "theory_practice": 0.7
        },
        {
            "research_intensity": 0.6,
            "advisor_style": 0.8,
            "team_work": 0.85,
            "workload": 0.65,
            "theory_practice": 0.8
        }
    ]
    
    # サンプル研究室
    labs = [
        {
            "id": "lab_001",
            "name": "AI研究室",
            "research_intensity": 0.85,
            "advisor_style": 0.65,
            "team_work": 0.75,
            "workload": 0.80,
            "theory_practice": 0.70
        },
        {
            "id": "lab_002",
            "name": "Web開発研究室",
            "research_intensity": 0.65,
            "advisor_style": 0.80,
            "team_work": 0.85,
            "workload": 0.70,
            "theory_practice": 0.85
        },
        {
            "id": "lab_003",
            "name": "理論研究室",
            "research_intensity": 0.90,
            "advisor_style": 0.60,
            "team_work": 0.60,
            "workload": 0.85,
            "theory_practice": 0.50
        }
    ]
    
    features = ["research_intensity", "advisor_style", "team_work", "workload", "theory_practice"]
    
    # 適応度関数：重み付きマッチング誤差の最小化
    def matching_fitness(weights):
        total_fitness = 0.0
        
        for student in student_profiles:
            for lab in labs:
                weighted_distance = 0.0
                for i, feature in enumerate(features):
                    student_val = student[feature]
                    lab_val = lab[feature]
                    diff = abs(student_val - lab_val)
                    weighted_distance += weights[i] * diff
                
                # 誤差を適応度に変換（誤差が小さいほど高い適応度）
                match_score = 1.0 / (weighted_distance / sum(weights) + 1e-6)
                total_fitness += match_score
        
        return total_fitness / (len(student_profiles) * len(labs))
    
    print(f"\n📊 テストデータ:")
    print(f"  学生プロファイル数: {len(student_profiles)}")
    print(f"  研究室数: {len(labs)}")
    print(f"  評価特徴: {features}")
    
    # デフォルト重みでの評価
    default_weights = [1.0] * len(features)
    default_fitness = matching_fitness(default_weights)
    print(f"\n🔧 デフォルト重み（均等）の適応度: {default_fitness:.6f}")
    
    # 遺伝的アルゴリズムで最適化
    config = EvolutionConfig(
        population_size=30,
        generations=50,
        mutation_rate=0.12,
        crossover_rate=0.85,
        elitism_rate=0.15,
        max_stagnation=15
    )
    
    engine = EvolutionEngine(config)
    
    print(f"\n🚀 遺伝的アルゴリズム最適化実行中...")
    result = engine.evolve(matching_fitness, verbose=False)
    
    # 最適化された重みでの評価
    optimized_weights = result.best_individual.chromosome[:len(features)]
    optimized_fitness = matching_fitness(optimized_weights)
    
    print(f"\n✅ 最適化結果:")
    print(f"  最良適応度: {result.best_fitness:.6f}")
    print(f"  処理時間: {result.processing_time:.2f}秒")
    print(f"  実行世代数: {result.generation}")
    print(f"  改善率: {(result.best_fitness / default_fitness - 1) * 100:.2f}%")
    
    print(f"\n📊 最適重み:")
    for i, (feature, weight) in enumerate(zip(features, optimized_weights)):
        default_w = default_weights[i]
        change = ((weight / default_w - 1) * 100) if default_w > 0 else 0
        print(f"  {feature:20s}: {weight:.3f} (変化: {change:+.1f}%)")
    
    # マッチング例の表示
    print(f"\n🎯 最適化重み使用時のマッチング例:")
    student = student_profiles[0]
    print(f"  学生プロファイル: research_intensity={student['research_intensity']}, "
          f"team_work={student['team_work']}")
    
    # 各研究室との適合度を計算
    matches = []
    for lab in labs:
        weighted_distance = sum(
            optimized_weights[i] * abs(student[feature] - lab[feature])
            for i, feature in enumerate(features)
        )
        compatibility = 1.0 / (weighted_distance / sum(optimized_weights) + 1e-6)
        matches.append((lab["name"], compatibility))
    
    matches.sort(key=lambda x: x[1], reverse=True)
    for rank, (lab_name, compat) in enumerate(matches, 1):
        print(f"  {rank}. {lab_name}: 適合度 {compat:.4f}")
    
    # 改善判定
    improvement = result.best_fitness > default_fitness
    print(f"\n🎯 最適化判定: {'✅ PASS - 適応度が向上しました' if improvement else '❌ FAIL - 改善が見られません'}")
    
    return improvement

def test_convergence_behavior():
    """収束挙動のテスト"""
    
    print("\n" + "=" * 70)
    print("🧬 テスト3: 収束挙動の確認")
    print("=" * 70)
    
    # 簡単な目的関数：特定の値に収束させる
    target_values = [1.0, 0.5, 1.5, 0.8, 1.2]
    
    def convergence_fitness(chromosome):
        error = sum((chromosome[i] - target_values[i % len(target_values)]) ** 2 
                   for i in range(min(len(chromosome), len(target_values))))
        fitness = 1.0 / (error + 1e-8)
        return fitness
    
    # 厳しい収束条件
    config = EvolutionConfig(
        population_size=25,
        generations=100,
        mutation_rate=0.1,
        crossover_rate=0.85,
        elitism_rate=0.2,
        convergence_threshold=1e-6,
        max_stagnation=10
    )
    
    engine = EvolutionEngine(config)
    result = engine.evolve(convergence_fitness, verbose=False)
    
    print(f"\n📊 収束結果:")
    print(f"  収束判定: {'✅ 達成' if result.convergence_achieved else '❌ 未達成'}")
    print(f"  実行世代数: {result.generation} / {config.generations}")
    print(f"  最終適応度: {result.best_fitness:.8f}")
    
    # 最適解との比較
    best_chromosome = result.best_individual.chromosome[:len(target_values)]
    print(f"\n🎯 目標値との比較:")
    for i, (target, actual) in enumerate(zip(target_values, best_chromosome)):
        error = abs(target - actual)
        print(f"  要素{i}: 目標={target:.3f}, 実測={actual:.3f}, 誤差={error:.6f}")
    
    # 適応度の推移を表示
    print(f"\n📈 適応度推移:")
    history_points = 10
    step = max(1, len(result.fitness_history) // history_points)
    for i in range(0, len(result.fitness_history), step):
        fitness = result.fitness_history[i]
        bar_length = int(fitness / result.best_fitness * 40)
        print(f"  世代{i:3d}: {'█' * bar_length} {fitness:.6f}")
    
    return result.convergence_achieved or result.generation < config.generations

def test_genetic_operators():
    """遺伝的操作のテスト"""
    
    print("\n" + "=" * 70)
    print("🧬 テスト4: 遺伝的操作（選択・交叉・変異）の確認")
    print("=" * 70)
    
    config = EvolutionConfig(
        population_size=10,
        mutation_rate=0.2,
        crossover_rate=0.8,
        tournament_size=3
    )
    
    engine = EvolutionEngine(config)
    engine.initialize_population()
    
    # 適応度をランダムに設定
    for ind in engine.population:
        ind.fitness = random.uniform(0, 10)
    
    print(f"\n🎯 選択操作テスト:")
    selected = engine.tournament_selection()
    print(f"  トーナメント選択で選ばれた個体の適応度: {selected.fitness:.4f}")
    print(f"  集団の平均適応度: {sum(ind.fitness for ind in engine.population) / len(engine.population):.4f}")
    
    print(f"\n🔀 交叉操作テスト:")
    parent1 = engine.population[0]
    parent2 = engine.population[1]
    print(f"  親1の染色体（先頭3要素）: {[f'{x:.3f}' for x in parent1.chromosome[:3]]}")
    print(f"  親2の染色体（先頭3要素）: {[f'{x:.3f}' for x in parent2.chromosome[:3]]}")
    
    child1, child2 = engine.crossover(parent1, parent2)
    print(f"  子1の染色体（先頭3要素）: {[f'{x:.3f}' for x in child1.chromosome[:3]]}")
    print(f"  子2の染色体（先頭3要素）: {[f'{x:.3f}' for x in child2.chromosome[:3]]}")
    
    # 交叉が実際に行われたか確認
    crossover_happened = (child1.chromosome != parent1.chromosome or 
                         child2.chromosome != parent2.chromosome)
    print(f"  交叉実行: {'✅' if crossover_happened else '❌'}")
    
    print(f"\n🔄 変異操作テスト:")
    original = Individual(chromosome=[1.0] * 5)
    print(f"  変異前: {[f'{x:.3f}' for x in original.chromosome]}")
    
    mutated = engine.mutate(original)
    print(f"  変異後: {[f'{x:.3f}' for x in mutated.chromosome]}")
    
    # 変異が実際に行われたか確認
    mutation_happened = mutated.chromosome != original.chromosome
    print(f"  変異実行: {'✅' if mutation_happened else '❌'}")
    
    return crossover_happened and mutation_happened

def run_all_tests():
    """全テストを実行"""
    
    print("\n" + "=" * 70)
    print("🧬 遺伝的アルゴリズム 包括テスト")
    print("=" * 70)
    
    start_time = time.time()
    
    results = {
        "基本動作": test_basic_genetic_algorithm(),
        "研究室マッチング最適化": test_lab_matching_optimization(),
        "収束挙動": test_convergence_behavior(),
        "遺伝的操作": test_genetic_operators()
    }
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("📊 テスト結果サマリー")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\n総合結果: {'✅ 全テスト合格' if all_passed else '❌ 一部テスト失敗'}")
    print(f"実行時間: {total_time:.2f}秒")
    
    if all_passed:
        print("\n🎉 遺伝的アルゴリズムは正しく実装されています！")
    else:
        print("\n⚠️  一部の機能に問題があります。コードを確認してください。")
    
    return all_passed

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)