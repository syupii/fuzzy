#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ファジィ決定木の推測過程可視化プログラム

このプログラムは以下の機能を提供します：
1. ファジィ決定木の構造表示
2. 推測過程の詳細な計算過程表示
3. メンバーシップ関数の可視化
4. 決定経路のトレース
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import json
from dataclasses import dataclass
from collections import defaultdict
import seaborn as sns

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style("whitegrid")

@dataclass
class PredictionStep:
    """推測ステップの詳細情報"""
    node_id: int
    feature_name: str
    feature_value: float
    membership_values: Dict[str, float]
    chosen_branch: str
    confidence: float
    explanation: str

@dataclass
class MembershipFunction:
    """メンバーシップ関数"""
    name: str
    function_type: str  # 'triangular', 'trapezoidal', 'gaussian'
    parameters: List[float]
    
    def calculate(self, x: float) -> float:
        """メンバーシップ値計算"""
        if self.function_type == 'triangular':
            a, b, c = self.parameters
            if x <= a or x >= c:
                return 0.0
            elif a < x <= b:
                return (x - a) / (b - a)
            else:  # b < x < c
                return (c - x) / (c - b)
        elif self.function_type == 'trapezoidal':
            a, b, c, d = self.parameters
            if x <= a or x >= d:
                return 0.0
            elif a < x <= b:
                return (x - a) / (b - a)
            elif b < x <= c:
                return 1.0
            else:  # c < x < d
                return (d - x) / (d - c)
        elif self.function_type == 'gaussian':
            center, sigma = self.parameters
            return np.exp(-0.5 * ((x - center) / sigma) ** 2)
        else:
            return 0.0

class FuzzyTreeNode:
    """ファジィ決定木ノード"""
    
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.is_leaf = False
        self.feature_name: Optional[str] = None
        self.feature_index: Optional[int] = None
        self.membership_functions: Dict[str, MembershipFunction] = {}
        self.children: Dict[str, 'FuzzyTreeNode'] = {}
        self.leaf_value: Optional[float] = None
        self.samples_count: int = 0
        self.depth: int = 0
        
    def predict_with_trace(self, feature_vector: List[float], feature_names: List[str]) -> Tuple[float, List[PredictionStep]]:
        """推測過程をトレースしながら予測"""
        trace = []
        return self._predict_recursive(feature_vector, feature_names, trace)
    
    def _predict_recursive(self, feature_vector: List[float], feature_names: List[str], trace: List[PredictionStep]) -> Tuple[float, List[PredictionStep]]:
        """再帰的予測（トレース付き）"""
        if self.is_leaf:
            return self.leaf_value, trace
        
        # 特徴値取得
        feature_value = feature_vector[self.feature_index]
        
        # 各言語変数に対するメンバーシップ値計算
        membership_values = {}
        for label, mf in self.membership_functions.items():
            membership_values[label] = mf.calculate(feature_value)
        
        # 最大メンバーシップ値を持つ分岐選択
        max_membership = max(membership_values.values())
        chosen_branch = max(membership_values, key=membership_values.get)
        
        # 信頼度計算（メンバーシップ値の最大値）
        confidence = max_membership
        
        # ステップ記録
        explanation = f"特徴量 '{self.feature_name}' の値 {feature_value:.2f} に対して、"
        explanation += f"言語変数 '{chosen_branch}' のメンバーシップ値が最大 ({max_membership:.3f})"
        
        step = PredictionStep(
            node_id=self.node_id,
            feature_name=self.feature_name,
            feature_value=feature_value,
            membership_values=membership_values.copy(),
            chosen_branch=chosen_branch,
            confidence=confidence,
            explanation=explanation
        )
        trace.append(step)
        
        # 子ノードに移動
        if chosen_branch in self.children:
            return self.children[chosen_branch]._predict_recursive(feature_vector, feature_names, trace)
        else:
            # 子ノードが存在しない場合のデフォルト値
            return 0.5, trace

class FuzzyTreeVisualizer:
    """ファジィ決定木可視化クラス"""
    
    def __init__(self):
        self.fig_size = (15, 10)
        self.node_colors = {
            'internal': '#E3F2FD',
            'leaf': '#C8E6C9',
            'selected': '#FFCDD2'
        }
        
    def visualize_tree_structure(self, root: FuzzyTreeNode, save_path: str = None):
        """決定木構造の可視化"""
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # グラフ構築
        G = nx.DiGraph()
        pos = {}
        labels = {}
        node_colors = []
        
        self._build_graph(root, G, pos, labels, node_colors, 0, 0, 8)
        
        # 描画
        nx.draw(G, pos, ax=ax, node_color=node_colors, node_size=3000,
                with_labels=False, arrows=True, edge_color='gray', 
                arrowsize=20, arrowstyle='->')
        
        # ラベル描画
        for node, (x, y) in pos.items():
            ax.text(x, y, labels[node], ha='center', va='center', 
                   fontsize=9, wrap=True, bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor='white', alpha=0.8))
        
        ax.set_title('ファジィ決定木構造', fontsize=16, fontweight='bold')
        ax.set_aspect('equal')
        
        if save_path:
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _build_graph(self, node: FuzzyTreeNode, G: nx.DiGraph, pos: Dict, 
                    labels: Dict, node_colors: List, x: float, y: float, 
                    width: float, depth: int = 0):
        """グラフ構築（再帰）"""
        node_id = f"node_{node.node_id}"
        G.add_node(node_id)
        pos[node_id] = (x, y)
        
        if node.is_leaf:
            labels[node_id] = f"葉ノード\n予測値: {node.leaf_value:.3f}\nサンプル数: {node.samples_count}"
            node_colors.append(self.node_colors['leaf'])
        else:
            labels[node_id] = f"{node.feature_name}\n深度: {depth}\nサンプル数: {node.samples_count}"
            node_colors.append(self.node_colors['internal'])
            
            # 子ノード配置
            num_children = len(node.children)
            if num_children > 0:
                child_width = width / num_children
                start_x = x - width/2 + child_width/2
                
                for i, (branch_name, child) in enumerate(node.children.items()):
                    child_x = start_x + i * child_width
                    child_y = y - 2
                    
                    child_id = f"node_{child.node_id}"
                    G.add_edge(node_id, child_id)
                    
                    # エッジラベル
                    edge_x = (x + child_x) / 2
                    edge_y = (y + child_y) / 2
                    plt.text(edge_x, edge_y, branch_name, ha='center', 
                            fontsize=8, bbox=dict(boxstyle="round,pad=0.2", 
                            facecolor='yellow', alpha=0.7))
                    
                    self._build_graph(child, G, pos, labels, node_colors, 
                                    child_x, child_y, child_width * 0.8, depth + 1)

    def visualize_prediction_process(self, feature_vector: List[float], 
                                   feature_names: List[str], 
                                   prediction_steps: List[PredictionStep],
                                   save_path: str = None):
        """推測過程の詳細可視化"""
        num_steps = len(prediction_steps)
        fig, axes = plt.subplots(num_steps, 2, figsize=(16, 4 * num_steps))
        
        if num_steps == 1:
            axes = axes.reshape(1, -1)
        
        for i, step in enumerate(prediction_steps):
            # 左側: メンバーシップ関数
            ax_left = axes[i, 0]
            self._plot_membership_functions(step, ax_left)
            
            # 右側: 決定情報
            ax_right = axes[i, 1]
            self._plot_decision_info(step, ax_right)
        
        plt.suptitle('ファジィ決定木推測過程', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_membership_functions(self, step: PredictionStep, ax):
        """メンバーシップ関数プロット"""
        # サンプルデータでメンバーシップ関数を描画
        x_range = np.linspace(0, 10, 1000)
        
        for label, membership_value in step.membership_values.items():
            # 簡易的な三角形メンバーシップ関数を仮定
            if label == 'low':
                y = np.maximum(0, 1 - x_range / 3)
            elif label == 'medium':
                y = np.maximum(0, np.minimum((x_range - 2) / 3, (8 - x_range) / 3))
            else:  # high
                y = np.maximum(0, (x_range - 7) / 3)
            
            color = 'red' if label == step.chosen_branch else 'blue'
            alpha = 0.8 if label == step.chosen_branch else 0.3
            ax.plot(x_range, y, label=f'{label} (μ={membership_value:.3f})', 
                   color=color, alpha=alpha, linewidth=2)
        
        # 現在の特徴値を示す線
        ax.axvline(x=step.feature_value, color='green', linestyle='--', 
                  linewidth=2, label=f'入力値: {step.feature_value:.2f}')
        
        ax.set_xlabel('特徴値')
        ax.set_ylabel('メンバーシップ値')
        ax.set_title(f'ステップ {step.node_id + 1}: {step.feature_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
    
    def _plot_decision_info(self, step: PredictionStep, ax):
        """決定情報プロット"""
        # メンバーシップ値の棒グラフ
        labels = list(step.membership_values.keys())
        values = list(step.membership_values.values())
        colors = ['red' if label == step.chosen_branch else 'lightblue' 
                 for label in labels]
        
        bars = ax.bar(labels, values, color=colors, alpha=0.7)
        
        # 値をバーの上に表示
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('メンバーシップ値')
        ax.set_title(f'決定: {step.chosen_branch} (信頼度: {step.confidence:.3f})')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        # 説明テキストを下部に追加
        ax.text(0.5, -0.15, step.explanation, transform=ax.transAxes,
               ha='center', va='top', wrap=True, fontsize=10,
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow'))

    def create_detailed_report(self, feature_vector: List[float], 
                             feature_names: List[str],
                             prediction_steps: List[PredictionStep],
                             final_prediction: float) -> str:
        """詳細レポート生成"""
        report = "=" * 60 + "\n"
        report += "ファジィ決定木 推測過程詳細レポート\n"
        report += "=" * 60 + "\n\n"
        
        # 入力情報
        report += "📊 入力データ:\n"
        for i, (name, value) in enumerate(zip(feature_names, feature_vector)):
            report += f"  {name}: {value:.3f}\n"
        report += "\n"
        
        # 推測過程
        report += "🌳 決定木推測過程:\n"
        for i, step in enumerate(prediction_steps, 1):
            report += f"\n--- ステップ {i} (ノード ID: {step.node_id}) ---\n"
            report += f"特徴量: {step.feature_name}\n"
            report += f"入力値: {step.feature_value:.3f}\n"
            report += f"メンバーシップ値:\n"
            
            for label, value in step.membership_values.items():
                marker = "👉" if label == step.chosen_branch else "  "
                report += f"  {marker} {label}: {value:.3f}\n"
            
            report += f"選択された分岐: {step.chosen_branch}\n"
            report += f"信頼度: {step.confidence:.3f}\n"
            report += f"説明: {step.explanation}\n"
        
        # 最終結果
        report += "\n" + "=" * 40 + "\n"
        report += f"🎯 最終予測値: {final_prediction:.3f}\n"
        report += f"📈 全体信頼度: {np.mean([s.confidence for s in prediction_steps]):.3f}\n"
        report += "=" * 40 + "\n"
        
        return report

def create_sample_fuzzy_tree():
    """サンプルファジィ決定木作成"""
    # ルートノード（研究分野の興味度）
    root = FuzzyTreeNode(0)
    root.feature_name = "研究分野興味度"
    root.feature_index = 0
    root.samples_count = 100
    root.depth = 0
    
    # メンバーシップ関数定義
    root.membership_functions = {
        'low': MembershipFunction('low', 'triangular', [0, 0, 5]),
        'medium': MembershipFunction('medium', 'triangular', [2, 5, 8]),
        'high': MembershipFunction('high', 'triangular', [5, 10, 10])
    }
    
    # 子ノード1（指導教員との相性）
    child1 = FuzzyTreeNode(1)
    child1.feature_name = "指導教員相性"
    child1.feature_index = 1
    child1.samples_count = 40
    child1.depth = 1
    child1.membership_functions = {
        'low': MembershipFunction('low', 'triangular', [0, 0, 4]),
        'high': MembershipFunction('high', 'triangular', [4, 10, 10])
    }
    
    # 子ノード2（設備充実度）
    child2 = FuzzyTreeNode(2)
    child2.feature_name = "設備充実度"
    child2.feature_index = 2
    child2.samples_count = 60
    child2.depth = 1
    child2.membership_functions = {
        'low': MembershipFunction('low', 'triangular', [0, 0, 5]),
        'high': MembershipFunction('high', 'triangular', [3, 10, 10])
    }
    
    # 葉ノード
    leaf1 = FuzzyTreeNode(3)
    leaf1.is_leaf = True
    leaf1.leaf_value = 0.2
    leaf1.samples_count = 15
    leaf1.depth = 2
    
    leaf2 = FuzzyTreeNode(4)
    leaf2.is_leaf = True
    leaf2.leaf_value = 0.8
    leaf2.samples_count = 25
    leaf2.depth = 2
    
    leaf3 = FuzzyTreeNode(5)
    leaf3.is_leaf = True
    leaf3.leaf_value = 0.6
    leaf3.samples_count = 30
    leaf3.depth = 2
    
    leaf4 = FuzzyTreeNode(6)
    leaf4.is_leaf = True
    leaf4.leaf_value = 0.9
    leaf4.samples_count = 30
    leaf4.depth = 2
    
    # 構造構築
    root.children = {'low': child1, 'medium': child2, 'high': child2}
    child1.children = {'low': leaf1, 'high': leaf2}
    child2.children = {'low': leaf3, 'high': leaf4}
    
    return root

# 保存先設定
SAVE_DIR = "experiments/results/visualizations"
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    """メイン実行関数"""
    print("🌳 ファジィ決定木推測過程可視化デモ")
    print("=" * 50)
    print(f"📁 保存先: {SAVE_DIR}")
    
    # サンプル木作成
    tree = create_sample_fuzzy_tree()
    visualizer = FuzzyTreeVisualizer()
    
    # テストデータ
    feature_vector = [7.5, 6.0, 8.0]  # [研究分野興味度, 指導教員相性, 設備充実度]
    feature_names = ["研究分野興味度", "指導教員相性", "設備充実度"]
    
    print("📊 入力データ:")
    for name, value in zip(feature_names, feature_vector):
        print(f"  {name}: {value}")
    print()
    
    # 1. 木構造可視化
    print("1. 決定木構造を可視化中...")
    structure_path = os.path.join(SAVE_DIR, "fuzzy_tree_structure.png")
    visualizer.visualize_tree_structure(tree, structure_path)
    
    # 2. 推測実行
    print("2. 推測過程を実行中...")
    prediction, steps = tree.predict_with_trace(feature_vector, feature_names)
    
    # 3. 推測過程可視化
    print("3. 推測過程を可視化中...")
    process_path = os.path.join(SAVE_DIR, "prediction_process.png")
    visualizer.visualize_prediction_process(
        feature_vector, feature_names, steps, process_path
    )
    
    # 4. 詳細レポート生成
    print("4. 詳細レポートを生成中...")
    report = visualizer.create_detailed_report(
        feature_vector, feature_names, steps, prediction
    )
    print(report)
    
    # レポートをファイルに保存
    report_path = os.path.join(SAVE_DIR, "prediction_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print("✅ 処理完了!")
    print("生成されたファイル:")
    print(f"  - {structure_path} (木構造)")
    print(f"  - {process_path} (推測過程)")
    print(f"  - {report_path} (詳細レポート)")

if __name__ == "__main__":
    main()