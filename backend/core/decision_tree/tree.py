"""
多階層ファジィ決定木モジュール
3レベル階層決定木による学生分類
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum


class NodeType(Enum):
    """ノードタイプ"""
    ROOT = "root"
    INTERNAL = "internal"
    LEAF = "leaf"


@dataclass
class TreeNode:
    """決定木ノード"""
    node_id: str
    node_type: NodeType
    feature: Optional[str] = None  # 分岐に使用する特徴
    threshold: Optional[float] = None  # 分岐閾値
    cluster_label: Optional[str] = None  # リーフノードのクラスタラベル
    left_child: Optional['TreeNode'] = None
    right_child: Optional['TreeNode'] = None
    depth: int = 0
    samples_count: int = 0
    
    def is_leaf(self) -> bool:
        """リーフノードかどうか"""
        return self.node_type == NodeType.LEAF
    
    def __repr__(self) -> str:
        if self.is_leaf():
            return f"LeafNode(cluster={self.cluster_label})"
        else:
            return f"Node(feature={self.feature}, threshold={self.threshold:.2f})"


class FuzzyDecisionTreeBuilder:
    """ファジィ決定木構築クラス"""
    
    def __init__(self, max_depth: int = 3, min_samples_split: int = 2):
        """
        Args:
            max_depth: 最大深さ
            min_samples_split: 分岐に必要な最小サンプル数
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.node_counter = 0
    
    def _generate_node_id(self) -> str:
        """ノードIDを生成"""
        node_id = f"node_{self.node_counter}"
        self.node_counter += 1
        return node_id
    
    def _calculate_fuzzy_entropy(self, memberships: List[Dict[str, float]]) -> float:
        """ファジィエントロピーを計算
        
        Args:
            memberships: 各サンプルのメンバーシップ度のリスト
            
        Returns:
            エントロピー値
        """
        if not memberships:
            return 0.0
        
        # 各クラスのファジィカーディナリティを計算
        class_counts: Dict[str, float] = {}
        
        for membership in memberships:
            for label, degree in membership.items():
                class_counts[label] = class_counts.get(label, 0.0) + degree
        
        total = sum(class_counts.values())
        
        if total == 0:
            return 0.0
        
        # エントロピー計算
        entropy = 0.0
        for count in class_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _split_samples(self, samples: List[Dict[str, float]], 
                      feature: str, threshold: float) -> Tuple[List, List]:
        """サンプルを分割
        
        Args:
            samples: サンプルリスト
            feature: 分岐特徴
            threshold: 閾値
            
        Returns:
            (left_samples, right_samples)
        """
        left = []
        right = []
        
        for sample in samples:
            if sample.get(feature, 0.5) <= threshold:
                left.append(sample)
            else:
                right.append(sample)
        
        return left, right
    
    def _find_best_split(self, samples: List[Dict[str, float]], 
                        features: List[str]) -> Tuple[str, float, float]:
        """最適な分岐を見つける
        
        Args:
            samples: サンプルリスト
            features: 候補特徴リスト
            
        Returns:
            (best_feature, best_threshold, best_gain)
        """
        best_gain = -float('inf')
        best_feature = None
        best_threshold = None
        
        # 現在のエントロピー
        current_entropy = self._calculate_fuzzy_entropy(
            [s.get("memberships", {}) for s in samples]
        )
        
        for feature in features:
            # 特徴値の範囲を取得
            values = [s.get(feature, 0.5) for s in samples]
            
            # 候補閾値（四分位点を使用）
            thresholds = [
                np.percentile(values, 25),
                np.percentile(values, 50),
                np.percentile(values, 75)
            ]
            
            for threshold in thresholds:
                # 分割
                left, right = self._split_samples(samples, feature, threshold)
                
                if len(left) < self.min_samples_split or len(right) < self.min_samples_split:
                    continue
                
                # 分割後のエントロピー
                left_entropy = self._calculate_fuzzy_entropy(
                    [s.get("memberships", {}) for s in left]
                )
                right_entropy = self._calculate_fuzzy_entropy(
                    [s.get("memberships", {}) for s in right]
                )
                
                # 重み付きエントロピー
                n = len(samples)
                weighted_entropy = (len(left) / n) * left_entropy + (len(right) / n) * right_entropy
                
                # 情報利得
                gain = current_entropy - weighted_entropy
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _create_leaf(self, samples: List[Dict[str, float]], depth: int) -> TreeNode:
        """リーフノードを作成
        
        Args:
            samples: サンプルリスト
            depth: 現在の深さ
            
        Returns:
            リーフノード
        """
        # 最も多いクラスラベルを選択
        memberships = [s.get("memberships", {}) for s in samples]
        class_counts: Dict[str, float] = {}
        
        for membership in memberships:
            for label, degree in membership.items():
                class_counts[label] = class_counts.get(label, 0.0) + degree
        
        if class_counts:
            dominant_class = max(class_counts.items(), key=lambda x: x[1])[0]
        else:
            dominant_class = "unknown"
        
        return TreeNode(
            node_id=self._generate_node_id(),
            node_type=NodeType.LEAF,
            cluster_label=dominant_class,
            depth=depth,
            samples_count=len(samples)
        )
    
    def _build_tree_recursive(self, samples: List[Dict[str, float]], 
                             features: List[str], depth: int) -> TreeNode:
        """再帰的に決定木を構築
        
        Args:
            samples: サンプルリスト
            features: 使用可能な特徴リスト
            depth: 現在の深さ
            
        Returns:
            構築されたノード
        """
        # 終了条件チェック
        if depth >= self.max_depth or len(samples) < self.min_samples_split:
            return self._create_leaf(samples, depth)
        
        # 最適な分岐を見つける
        best_feature, best_threshold, best_gain = self._find_best_split(samples, features)
        
        if best_feature is None or best_gain <= 0:
            return self._create_leaf(samples, depth)
        
        # サンプルを分割
        left_samples, right_samples = self._split_samples(samples, best_feature, best_threshold)
        
        if len(left_samples) == 0 or len(right_samples) == 0:
            return self._create_leaf(samples, depth)
        
        # 内部ノードを作成
        node = TreeNode(
            node_id=self._generate_node_id(),
            node_type=NodeType.INTERNAL if depth > 0 else NodeType.ROOT,
            feature=best_feature,
            threshold=best_threshold,
            depth=depth,
            samples_count=len(samples)
        )
        
        # 再帰的に子ノードを構築
        node.left_child = self._build_tree_recursive(left_samples, features, depth + 1)
        node.right_child = self._build_tree_recursive(right_samples, features, depth + 1)
        
        return node
    
    def fit(self, samples: List[Dict[str, float]], features: List[str]):
        """決定木を学習
        
        Args:
            samples: 訓練サンプル
            features: 特徴リスト
        """
        self.node_counter = 0
        self.root = self._build_tree_recursive(samples, features, depth=0)
    
    def predict(self, sample: Dict[str, float]) -> str:
        """サンプルを分類
        
        Args:
            sample: 入力サンプル
            
        Returns:
            クラスタラベル
        """
        if self.root is None:
            raise ValueError("Tree not fitted yet")
        
        node = self.root
        
        while not node.is_leaf():
            feature_value = sample.get(node.feature, 0.5)
            
            if feature_value <= node.threshold:
                node = node.left_child
            else:
                node = node.right_child
        
        return node.cluster_label
    
    def get_path(self, sample: Dict[str, float]) -> List[str]:
        """サンプルの分類パスを取得
        
        Args:
            sample: 入力サンプル
            
        Returns:
            パス記述のリスト
        """
        if self.root is None:
            raise ValueError("Tree not fitted yet")
        
        path = []
        node = self.root
        
        while not node.is_leaf():
            feature_value = sample.get(node.feature, 0.5)
            
            if feature_value <= node.threshold:
                path.append(f"{node.feature} <= {node.threshold:.2f}")
                node = node.left_child
            else:
                path.append(f"{node.feature} > {node.threshold:.2f}")
                node = node.right_child
        
        path.append(f"→ Cluster: {node.cluster_label}")
        
        return path
    
    def print_tree(self, node: Optional[TreeNode] = None, prefix: str = "", is_left: bool = True):
        """決定木を視覚的に表示
        
        Args:
            node: 表示するノード
            prefix: 表示プレフィックス
            is_left: 左の子かどうか
        """
        if node is None:
            node = self.root
        
        if node is None:
            print("Tree not fitted yet")
            return
        
        # ノード情報を表示
        connector = "├── " if is_left else "└── "
        
        if node.is_leaf():
            print(f"{prefix}{connector}🍃 Leaf: {node.cluster_label} (samples={node.samples_count})")
        else:
            print(f"{prefix}{connector}📊 {node.feature} <= {node.threshold:.2f} (samples={node.samples_count})")
        
        # 子ノードを再帰的に表示
        if not node.is_leaf():
            new_prefix = prefix + ("│   " if is_left else "    ")
            
            if node.left_child:
                self.print_tree(node.left_child, new_prefix, True)
            
            if node.right_child:
                self.print_tree(node.right_child, new_prefix, False)


class MultiLevelFuzzyClassifier:
    """多階層ファジィ分類器
    
    3レベルの階層的分類を実装
    - Level 1: research_intensity による大分類
    - Level 2: 各ブランチでさらに細分化
    """
    
    def __init__(self):
        self.structure = self._define_structure()
    
    def _define_structure(self) -> Dict:
        """階層構造を定義"""
        return {
            "level1": {
                "feature": "research_intensity",
                "branches": {
                    "high": {
                        "range": (0.7, 1.0),
                        "level2": {
                            "feature": "team_work",
                            "clusters": {
                                "team_oriented": {"range": (0.7, 1.0)},
                                "individual_focused": {"range": (0.0, 0.7)}
                            }
                        }
                    },
                    "medium": {
                        "range": (0.4, 0.7),
                        "level2": {
                            "feature": "flexibility",
                            "clusters": {
                                "flexible_style": {"range": (0.6, 1.0)},
                                "structured_style": {"range": (0.0, 0.6)}
                            }
                        }
                    },
                    "low": {
                        "range": (0.0, 0.4),
                        "level2": {
                            "feature": "lab_atmosphere",
                            "clusters": {
                                "active_atmosphere": {"range": (0.6, 1.0)},
                                "quiet_atmosphere": {"range": (0.0, 0.6)}
                            }
                        }
                    }
                }
            }
        }
    
    def _triangular_membership(self, x: float, a: float, b: float, c: float) -> float:
        """三角型メンバーシップ関数"""
        if x <= a or x >= c:
            return 0.0
        elif x == b:
            return 1.0
        elif x < b:
            return (x - a) / (b - a)
        else:
            return (c - x) / (c - b)
    
    def classify(self, profile: Dict[str, float]) -> Dict[str, Any]:
        """プロファイルを分類
        
        Args:
            profile: 学生プロファイル
            
        Returns:
            分類結果（クラスタ、メンバーシップ度、パスなど）
        """
        research_intensity = profile.get("research_intensity", 0.5)
        
        # Level 1: 大分類
        level1_memberships = {
            "low": self._triangular_membership(research_intensity, 0.0, 0.0, 0.5),
            "medium": self._triangular_membership(research_intensity, 0.3, 0.5, 0.7),
            "high": self._triangular_membership(research_intensity, 0.5, 1.0, 1.0)
        }
        
        # 最大メンバーシップのブランチを選択
        primary_branch = max(level1_memberships.items(), key=lambda x: x[1])[0]
        
        # Level 2: 細分化
        branch_config = self.structure["level1"]["branches"][primary_branch]
        level2_feature = branch_config["level2"]["feature"]
        level2_value = profile.get(level2_feature, 0.5)
        
        level2_memberships = {}
        for cluster_name, cluster_config in branch_config["level2"]["clusters"].items():
            range_min, range_max = cluster_config["range"]
            center = (range_min + range_max) / 2
            
            membership = self._triangular_membership(
                level2_value,
                range_min - 0.2,
                center,
                range_max + 0.2
            )
            level2_memberships[cluster_name] = membership
        
        # 最終クラスタ
        final_cluster = max(level2_memberships.items(), key=lambda x: x[1])[0]
        
        return {
            "primary_cluster": f"{primary_branch}_{final_cluster}",
            "level1_branch": primary_branch,
            "level2_cluster": final_cluster,
            "level1_memberships": level1_memberships,
            "level2_memberships": level2_memberships,
            "classification_path": [
                f"Level 1: {primary_branch} (μ={level1_memberships[primary_branch]:.3f})",
                f"Level 2: {final_cluster} (μ={level2_memberships[final_cluster]:.3f})"
            ],
            "confidence": {
                "level1": level1_memberships[primary_branch],
                "level2": level2_memberships[final_cluster],
                "overall": (level1_memberships[primary_branch] + level2_memberships[final_cluster]) / 2
            }
        }


# 使用例とテスト
if __name__ == "__main__":
    print("=" * 70)
    print("多階層ファジィ決定木テスト")
    print("=" * 70)
    
    # テストデータ作成
    test_profiles = [
        {
            "name": "研究熱心・チーム志向",
            "research_intensity": 0.9,
            "team_work": 0.8,
            "flexibility": 0.6,
            "lab_atmosphere": 0.7
        },
        {
            "name": "バランス型・柔軟志向",
            "research_intensity": 0.5,
            "team_work": 0.6,
            "flexibility": 0.8,
            "lab_atmosphere": 0.6
        },
        {
            "name": "軽負荷・静寂志向",
            "research_intensity": 0.3,
            "team_work": 0.4,
            "flexibility": 0.5,
            "lab_atmosphere": 0.3
        }
    ]
    
    # 分類器の作成
    classifier = MultiLevelFuzzyClassifier()
    
    print("\n🔍 学生プロファイル分類結果\n")
    
    for profile in test_profiles:
        print(f"【{profile['name']}】")
        result = classifier.classify(profile)
        
        print(f"  最終クラスタ: {result['primary_cluster']}")
        print(f"  分類パス:")
        for path in result['classification_path']:
            print(f"    {path}")
        print(f"  信頼度: {result['confidence']['overall']:.3f}")
        print()
    
    print("=" * 70)