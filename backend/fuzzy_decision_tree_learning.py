# fuzzy_decision_tree_learning.py
# -*- coding: utf-8 -*-
"""
ファジィ決定木学習システム
本格的なファジィ決定木の構築と学習
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random
import time
from datetime import datetime

# Windows文字エンコーディング設定
if sys.platform.startswith('win'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

class MembershipType(Enum):
    """メンバーシップ関数タイプ"""
    TRIANGULAR = "triangular"
    GAUSSIAN = "gaussian"
    TRAPEZOIDAL = "trapezoidal"

class MembershipFunction:
    """メンバーシップ関数"""
    
    def __init__(self, name: str, mf_type: MembershipType, parameters: List[float]):
        self.name = name
        self.mf_type = mf_type
        self.parameters = parameters
    
    def membership(self, value: float) -> float:
        """メンバーシップ度計算"""
        try:
            if self.mf_type == MembershipType.TRIANGULAR:
                return self._triangular_membership(value)
            elif self.mf_type == MembershipType.GAUSSIAN:
                return self._gaussian_membership(value)
            elif self.mf_type == MembershipType.TRAPEZOIDAL:
                return self._trapezoidal_membership(value)
            else:
                return 0.0
        except:
            return 0.0
    
    def _triangular_membership(self, value: float) -> float:
        """三角メンバーシップ"""
        if len(self.parameters) < 3:
            return 0.0
        
        a, b, c = self.parameters[:3]
        
        if value <= a or value >= c:
            return 0.0
        elif value == b:
            return 1.0
        elif value < b:
            return (value - a) / (b - a) if b != a else 0.0
        else:
            return (c - value) / (c - b) if c != b else 0.0
    
    def _gaussian_membership(self, value: float) -> float:
        """ガウシアンメンバーシップ"""
        if len(self.parameters) < 2:
            return 0.0
        
        center, sigma = self.parameters[:2]
        if sigma <= 0:
            sigma = 0.1
        return np.exp(-0.5 * ((value - center) / sigma) ** 2)
    
    def _trapezoidal_membership(self, value: float) -> float:
        """台形メンバーシップ"""
        if len(self.parameters) < 4:
            return 0.0
        
        a, b, c, d = self.parameters[:4]
        
        if value <= a or value >= d:
            return 0.0
        elif b <= value <= c:
            return 1.0
        elif a < value < b:
            return (value - a) / (b - a) if b != a else 0.0
        else:  # c < value < d
            return (d - value) / (d - c) if d != c else 0.0

class FuzzyDecisionNode:
    """ファジィ決定ノード"""
    
    def __init__(self, node_id: str = None, is_leaf: bool = False, 
                 feature_name: str = None, leaf_value: float = None):
        self.node_id = node_id or f"node_{int(time.time())}_{random.randint(1000, 9999)}"
        self.is_leaf = is_leaf
        self.feature_name = feature_name
        self.leaf_value = leaf_value
        self.membership_functions: Dict[str, MembershipFunction] = {}
        self.children: Dict[str, 'FuzzyDecisionNode'] = {}
        self.depth = 0
        self.sample_count = 0
        self.weighted_target_sum = 0.0
        
        # 学習用統計
        self.training_samples = []
        self.training_targets = []
    
    def add_membership_function(self, label: str, mf: MembershipFunction):
        """メンバーシップ関数追加"""
        self.membership_functions[label] = mf
    
    def add_child(self, label: str, child: 'FuzzyDecisionNode'):
        """子ノード追加"""
        self.children[label] = child
        if child:
            child.depth = self.depth + 1
    
    def predict(self, features: Dict[str, float]) -> float:
        """予測実行"""
        if self.is_leaf:
            return self.leaf_value if self.leaf_value is not None else 0.5
        
        if not self.feature_name or self.feature_name not in features:
            return 0.5
        
        feature_value = features[self.feature_name]
        
        # ファジィ推論
        weighted_sum = 0.0
        total_weight = 0.0
        
        for label, mf in self.membership_functions.items():
            membership_degree = mf.membership(feature_value)
            
            if membership_degree > 0 and label in self.children:
                child_prediction = self.children[label].predict(features)
                weighted_sum += membership_degree * child_prediction
                total_weight += membership_degree
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def predict_with_explanation(self, features: Dict[str, float], 
                               feature_names: List[str]) -> Tuple[float, Dict[str, Any]]:
        """説明付き予測"""
        prediction = self.predict(features)
        
        explanation = {
            'confidence': self._calculate_confidence(features),
            'rationale': f'ファジィ決定木による予測: {prediction:.3f}',
            'decision_steps': self._generate_decision_path(features),
            'feature_importance': self._calculate_feature_importance(features, feature_names),
            'node_info': {
                'node_id': self.node_id,
                'depth': self.depth,
                'is_leaf': self.is_leaf,
                'sample_count': self.sample_count
            }
        }
        
        return prediction, explanation
    
    def _calculate_confidence(self, features: Dict[str, float]) -> float:
        """信頼度計算"""
        if self.is_leaf:
            # サンプル数ベースの信頼度
            return min(0.95, 0.5 + (self.sample_count / 100.0))
        
        if not self.feature_name or self.feature_name not in features:
            return 0.3
        
        feature_value = features[self.feature_name]
        max_membership = 0.0
        
        for mf in self.membership_functions.values():
            membership = mf.membership(feature_value)
            max_membership = max(max_membership, membership)
        
        # メンバーシップ度ベースの信頼度
        return 0.4 + (max_membership * 0.5)
    
    def _generate_decision_path(self, features: Dict[str, float]) -> List[str]:
        """決定パス生成"""
        path = []
        
        if self.is_leaf:
            path.append(f"リーフノード: 値={self.leaf_value:.3f}, サンプル数={self.sample_count}")
        else:
            if self.feature_name and self.feature_name in features:
                feature_value = features[self.feature_name]
                path.append(f"特徴量 {self.feature_name} = {feature_value:.2f}")
                
                # メンバーシップ度計算
                memberships = []
                for label, mf in self.membership_functions.items():
                    membership = mf.membership(feature_value)
                    memberships.append((label, membership))
                
                # 最も高いメンバーシップ度を持つラベル
                best_label, best_membership = max(memberships, key=lambda x: x[1])
                path.append(f"→ {best_label} (メンバーシップ度: {best_membership:.3f})")
                
                # 全メンバーシップ度表示
                path.append("全メンバーシップ度: " + 
                          ", ".join([f"{label}={mem:.3f}" for label, mem in memberships]))
                
                if best_label in self.children:
                    child_path = self.children[best_label]._generate_decision_path(features)
                    path.extend([f"  {step}" for step in child_path])
        
        return path
    
    def _calculate_feature_importance(self, features: Dict[str, float], 
                                    feature_names: List[str]) -> Dict[str, float]:
        """特徴量重要度計算"""
        importance = {}
        
        if self.is_leaf:
            return importance
        
        if self.feature_name and self.feature_name in features:
            # 現在のノードの重要度
            importance[self.feature_name] = 1.0
            
            # 子ノードの重要度を再帰的に計算
            for child in self.children.values():
                child_importance = child._calculate_feature_importance(features, feature_names)
                for feature, imp in child_importance.items():
                    importance[feature] = importance.get(feature, 0) + imp * 0.5
        
        return importance
    
    def calculate_complexity(self) -> int:
        """複雑度計算"""
        if self.is_leaf:
            return 1
        
        complexity = 1 + len(self.membership_functions)
        for child in self.children.values():
            complexity += child.calculate_complexity()
        
        return complexity
    
    def calculate_depth(self) -> int:
        """深度計算"""
        if self.is_leaf:
            return 1
        
        max_child_depth = 0
        for child in self.children.values():
            child_depth = child.calculate_depth()
            max_child_depth = max(max_child_depth, child_depth)
        
        return 1 + max_child_depth

class FuzzyDecisionTreeLearner:
    """ファジィ決定木学習器"""
    
    def __init__(self, max_depth: int = 6, min_samples_split: int = 10, 
                 min_samples_leaf: int = 5, membership_functions_per_feature: int = 3):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.membership_functions_per_feature = membership_functions_per_feature
        
        self.root: Optional[FuzzyDecisionNode] = None
        self.feature_names: List[str] = []
        self.feature_ranges: Dict[str, Tuple[float, float]] = {}
        
        print(f"[LEARNER] ファジィ決定木学習器初期化")
        print(f"   最大深度: {max_depth}")
        print(f"   分割最小サンプル数: {min_samples_split}")
        print(f"   リーフ最小サンプル数: {min_samples_leaf}")
        print(f"   特徴量あたりメンバーシップ関数数: {membership_functions_per_feature}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FuzzyDecisionTreeLearner':
        """ファジィ決定木学習"""
        
        print(f"\n[LEARNING] ファジィ決定木学習開始")
        print(f"   訓練サンプル数: {len(X)}")
        print(f"   特徴量数: {len(X.columns)}")
        
        self.feature_names = list(X.columns)
        
        # 特徴量範囲計算
        for feature in self.feature_names:
            min_val = X[feature].min()
            max_val = X[feature].max()
            self.feature_ranges[feature] = (min_val, max_val)
            print(f"   {feature}: [{min_val:.2f}, {max_val:.2f}]")
        
        # ルートノード構築
        self.root = self._build_tree(X, y, depth=0)
        
        print(f"[LEARNING] 学習完了")
        print(f"   木の深度: {self.root.calculate_depth()}")
        print(f"   木の複雑度: {self.root.calculate_complexity()}")
        
        return self
    
    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int = 0) -> FuzzyDecisionNode:
        """木構築（再帰的）"""
        
        # 停止条件チェック
        if (depth >= self.max_depth or 
            len(X) < self.min_samples_split or
            self._is_pure(y)):
            
            return self._create_leaf_node(X, y, depth)
        
        # 最良分割探索
        best_feature, best_score = self._find_best_split(X, y)
        
        if best_feature is None:
            return self._create_leaf_node(X, y, depth)
        
        # 内部ノード作成
        node = FuzzyDecisionNode(feature_name=best_feature)
        node.depth = depth
        node.sample_count = len(X)
        node.weighted_target_sum = y.sum()
        
        # メンバーシップ関数生成
        membership_functions = self._generate_membership_functions(best_feature, X[best_feature])
        
        for label, mf in membership_functions.items():
            node.add_membership_function(label, mf)
        
        # 子ノード生成
        for label, mf in membership_functions.items():
            # ファジィ分割
            child_X, child_y = self._fuzzy_split(X, y, best_feature, mf)
            
            if len(child_X) >= self.min_samples_leaf:
                child = self._build_tree(child_X, child_y, depth + 1)
                node.add_child(label, child)
            else:
                # サンプル数が少ない場合はリーフ作成
                leaf = self._create_leaf_node(child_X, child_y, depth + 1)
                node.add_child(label, leaf)
        
        return node
    
    def _is_pure(self, y: pd.Series, threshold: float = 0.01) -> bool:
        """純度チェック"""
        if len(y) <= 1:
            return True
        
        variance = y.var()
        return variance < threshold
    
    def _create_leaf_node(self, X: pd.DataFrame, y: pd.Series, depth: int) -> FuzzyDecisionNode:
        """リーフノード作成"""
        leaf_value = y.mean() if len(y) > 0 else 0.5
        
        leaf = FuzzyDecisionNode(is_leaf=True, leaf_value=leaf_value)
        leaf.depth = depth
        leaf.sample_count = len(X)
        leaf.weighted_target_sum = y.sum()
        
        return leaf
    
    def _find_best_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Optional[str], float]:
        """最良分割探索"""
        best_feature = None
        best_score = -np.inf
        
        for feature in self.feature_names:
            score = self._calculate_split_score(X, y, feature)
            
            if score > best_score:
                best_score = score
                best_feature = feature
        
        return best_feature, best_score
    
    def _calculate_split_score(self, X: pd.DataFrame, y: pd.Series, feature: str) -> float:
        """分割スコア計算（ファジィ情報ゲイン）"""
        
        # メンバーシップ関数生成
        membership_functions = self._generate_membership_functions(feature, X[feature])
        
        # 親ノードのエントロピー
        parent_entropy = self._calculate_entropy(y)
        
        # 子ノードの重み付きエントロピー
        weighted_entropy = 0.0
        total_weight = 0.0
        
        for label, mf in membership_functions.items():
            child_X, child_y = self._fuzzy_split(X, y, feature, mf)
            
            if len(child_y) > 0:
                weight = len(child_y) / len(y)
                entropy = self._calculate_entropy(child_y)
                weighted_entropy += weight * entropy
                total_weight += weight
        
        # 情報ゲイン計算
        if total_weight > 0:
            information_gain = parent_entropy - (weighted_entropy / total_weight)
        else:
            information_gain = 0.0
        
        return information_gain
    
    def _calculate_entropy(self, y: pd.Series) -> float:
        """エントロピー計算（回帰用）"""
        if len(y) <= 1:
            return 0.0
        
        # 分散ベースのエントロピー
        variance = y.var()
        return np.log(1 + variance)
    
    def _generate_membership_functions(self, feature: str, values: pd.Series) -> Dict[str, MembershipFunction]:
        """メンバーシップ関数生成"""
        
        min_val, max_val = self.feature_ranges[feature]
        range_size = max_val - min_val
        
        if range_size <= 0:
            range_size = 1.0
        
        membership_functions = {}
        num_mf = self.membership_functions_per_feature
        
        labels = ['low', 'medium', 'high', 'very_high'][:num_mf]
        
        for i, label in enumerate(labels):
            # 三角メンバーシップ関数のパラメータ
            center = min_val + (i + 0.5) * range_size / num_mf
            spread = range_size / num_mf * 0.8
            
            a = max(min_val, center - spread)
            b = center
            c = min(max_val, center + spread)
            
            mf = MembershipFunction(
                name=label,
                mf_type=MembershipType.TRIANGULAR,
                parameters=[a, b, c]
            )
            
            membership_functions[label] = mf
        
        return membership_functions
    
    def _fuzzy_split(self, X: pd.DataFrame, y: pd.Series, 
                    feature: str, mf: MembershipFunction) -> Tuple[pd.DataFrame, pd.Series]:
        """ファジィ分割"""
        
        # メンバーシップ度計算
        memberships = X[feature].apply(mf.membership)
        
        # しきい値以上のサンプルを選択
        threshold = 0.1
        mask = memberships >= threshold
        
        return X[mask], y[mask]
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測"""
        if self.root is None:
            raise ValueError("モデルが学習されていません。先にfit()を呼び出してください。")
        
        predictions = []
        for idx, row in X.iterrows():
            features = row.to_dict()
            prediction = self.root.predict(features)
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def predict_with_explanation(self, sample: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
        """説明付き予測"""
        if self.root is None:
            raise ValueError("モデルが学習されていません。")
        
        return self.root.predict_with_explanation(sample, self.feature_names)

def create_research_lab_dataset(n_samples: int = 1000) -> Tuple[pd.DataFrame, pd.Series]:
    """研究室データセット作成（学習用）"""
    
    print(f"[DATA] 研究室データセット作成中 ({n_samples}サンプル)")
    
    np.random.seed(42)
    
    data = []
    targets = []
    
    for i in range(n_samples):
        # より現実的な特徴量生成
        sample = {}
        
        # 基本特徴量（1-10スケール）
        sample['research_intensity'] = max(1, min(10, np.random.normal(6.5, 2.0)))
        sample['advisor_style'] = max(1, min(10, np.random.normal(6.0, 2.2)))
        sample['team_work'] = max(1, min(10, np.random.normal(6.8, 2.1)))
        sample['workload'] = max(1, min(10, np.random.normal(6.2, 2.3)))
        sample['theory_practice'] = max(1, min(10, np.random.normal(6.4, 2.0)))
        
        # 追加特徴量
        sample['publication_focus'] = max(1, min(10, np.random.normal(7.0, 1.8)))
        sample['industry_connection'] = max(1, min(10, np.random.normal(5.5, 2.5)))
        sample['international_collaboration'] = max(1, min(10, np.random.normal(6.0, 2.2)))
        
        # 複雑な適合度計算
        weights = {
            'research_intensity': 0.20,
            'advisor_style': 0.15,
            'team_work': 0.15,
            'workload': 0.15,
            'theory_practice': 0.15,
            'publication_focus': 0.10,
            'industry_connection': 0.05,
            'international_collaboration': 0.05
        }
        
        # 基本適合度
        compatibility = sum(weights[key] * value for key, value in sample.items())
        compatibility /= 10.0  # 0-1スケールに正規化
        
        # 非線形効果追加
        # 研究強度と理論実践のバランス
        balance_bonus = 1 - abs(sample['research_intensity'] - sample['theory_practice']) / 10.0
        compatibility += 0.05 * balance_bonus
        
        # 指導スタイルとチームワークの相性
        if abs(sample['advisor_style'] - sample['team_work']) < 2:
            compatibility += 0.03
        
        # ワークロードの適正範囲
        if 5 <= sample['workload'] <= 7:
            compatibility += 0.02
        
        # ノイズ追加
        noise = np.random.normal(0, 0.06)
        compatibility += noise
        
        # 範囲制約
        compatibility = max(0.0, min(1.0, compatibility))
        
        data.append(sample)
        targets.append(compatibility)
    
    # DataFrameとSeriesに変換
    X = pd.DataFrame(data)
    y = pd.Series(targets)
    
    print(f"[DATA] データセット作成完了")
    print(f"   特徴量: {list(X.columns)}")
    print(f"   適合度範囲: [{y.min():.3f}, {y.max():.3f}]")
    print(f"   適合度平均: {y.mean():.3f}")
    
    return X, y

def train_fuzzy_decision_tree():
    """ファジィ決定木訓練実行"""
    
    print("=" * 60)
    print("[TRAINING] ファジィ決定木学習システム")
    print("=" * 60)
    
    # データセット生成
    X_train, y_train = create_research_lab_dataset(800)
    X_test, y_test = create_research_lab_dataset(200)
    
    # 学習器作成
    learner = FuzzyDecisionTreeLearner(
        max_depth=5,
        min_samples_split=15,
        min_samples_leaf=8,
        membership_functions_per_feature=3
    )
    
    # 学習実行
    start_time = time.time()
    learner.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"[TRAINING] 学習時間: {training_time:.2f}秒")
    
    # 予測実行
    print(f"\n[EVALUATION] モデル評価")
    
    # 訓練データ評価
    train_predictions = learner.predict(X_train)
    train_mse = np.mean((train_predictions - y_train) ** 2)
    train_mae = np.mean(np.abs(train_predictions - y_train))
    
    # テストデータ評価
    test_predictions = learner.predict(X_test)
    test_mse = np.mean((test_predictions - y_test) ** 2)
    test_mae = np.mean(np.abs(test_predictions - y_test))
    
    print(f"   訓練データ - MSE: {train_mse:.4f}, MAE: {train_mae:.4f}")
    print(f"   テストデータ - MSE: {test_mse:.4f}, MAE: {test_mae:.4f}")
    
    # 予測例
    print(f"\n[EXAMPLES] 予測例と説明")
    
    for i in range(3):
        sample = X_test.iloc[i].to_dict()
        target = y_test.iloc[i]
        
        prediction, explanation = learner.predict_with_explanation(sample)
        
        print(f"\n--- 例 {i+1} ---")
        print(f"実際値: {target:.3f}, 予測値: {prediction:.3f}, 誤差: {abs(target-prediction):.3f}")
        print(f"信頼度: {explanation['confidence']:.1%}")
        print(f"説明: {explanation['rationale']}")
        
        print("決定パス:")
        for step in explanation['decision_steps'][:5]:  # 最初の5ステップ
            print(f"  {step}")
        
        print("特徴量重要度:")
        importance = explanation['feature_importance']
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for feature, imp in sorted_importance[:3]:  # 上位3特徴量
            print(f"  {feature}: {imp:.3f}")
    
    # モデル保存
    print(f"\n[SAVING] モデル保存")
    
    model_data = {
        'learner': learner,
        'training_info': {
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'training_time': training_time,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'test_mae': test_mae
        },
        'tree_info': {
            'depth': learner.root.calculate_depth(),
            'complexity': learner.root.calculate_complexity()
        },
        'created_at': datetime.now().isoformat()
    }
    
    try:
        os.makedirs('models', exist_ok=True)
        
        model_path = 'models/fuzzy_decision_tree_learned.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"[SAVED] モデル保存完了: {model_path}")
        
        # サイズ確認
        size = os.path.getsize(model_path)
        print(f"[INFO] ファイルサイズ: {size} bytes")
        
        return model_data
        
    except Exception as e:
        print(f"[ERROR] モデル保存失敗: {e}")
        return None

def main():
    """メイン実行"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='ファジィ決定木学習システム')
    parser.add_argument('--mode', choices=['train', 'test', 'demo'], 
                       default='train', help='実行モード')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        print("[MODE] 学習モード")
        model_data = train_fuzzy_decision_tree()
        
        if model_data:
            print(f"\n[SUCCESS] ファジィ決定木学習完了!")
            print(f"[NEXT] 学習済みモデルがmodels/fuzzy_decision_tree_learned.pklに保存されました")
        else:
            print(f"\n[ERROR] 学習に失敗しました")
            
    elif args.mode == 'test':
        print("[MODE] テストモード")
        
        try:
            model_path = 'models/fuzzy_decision_tree_learned.pkl'
            
            if not os.path.exists(model_path):
                print(f"[ERROR] 学習済みモデルが見つかりません: {model_path}")
                print(f"[ACTION] 先に --mode train を実行してください")
                return 1
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            learner = model_data['learner']
            print(f"[LOADED] 学習済みモデル読み込み完了")
            print(f"   木の深度: {model_data['tree_info']['depth']}")
            print(f"   木の複雑度: {model_data['tree_info']['complexity']}")
            print(f"   テストMSE: {model_data['training_info']['test_mse']:.4f}")
            
            # テストサンプルで予測
            test_sample = {
                'research_intensity': 8.0,
                'advisor_style': 6.5,
                'team_work': 7.0,
                'workload': 6.0,
                'theory_practice': 8.5,
                'publication_focus': 7.5,
                'industry_connection': 5.0,
                'international_collaboration': 6.0
            }
            
            prediction, explanation = learner.predict_with_explanation(test_sample)
            
            print(f"\n[TEST] テスト予測")
            print(f"   予測値: {prediction:.3f}")
            print(f"   信頼度: {explanation['confidence']:.1%}")
            print(f"   説明: {explanation['rationale']}")
            
            print(f"\n決定パス:")
            for i, step in enumerate(explanation['decision_steps']):
                print(f"   {i+1}. {step}")
            
            print(f"\n特徴量重要度:")
            importance = explanation['feature_importance']
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for feature, imp in sorted_importance:
                print(f"   {feature}: {imp:.3f}")
            
        except Exception as e:
            print(f"[ERROR] テスト失敗: {e}")
            import traceback
            traceback.print_exc()
            return 1
            
    elif args.mode == 'demo':
        print("[MODE] デモモード")
        
        # 簡単なデモ実行
        print("[DEMO] 小規模データでのファジィ決定木学習デモ")
        
        # 小規模データセット
        X_demo, y_demo = create_research_lab_dataset(100)
        
        # 学習器作成（小規模用設定）
        learner = FuzzyDecisionTreeLearner(
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=3,
            membership_functions_per_feature=2
        )
        
        # 学習実行
        learner.fit(X_demo, y_demo)
        
        # デモ予測
        demo_sample = {
            'research_intensity': 7.0,
            'advisor_style': 8.0,
            'team_work': 6.0,
            'workload': 7.5,
            'theory_practice': 6.5,
            'publication_focus': 8.0,
            'industry_connection': 4.0,
            'international_collaboration': 7.0
        }
        
        prediction, explanation = learner.predict_with_explanation(demo_sample)
        
        print(f"\n[DEMO] デモ予測結果")
        print(f"   予測値: {prediction:.3f}")
        print(f"   信頼度: {explanation['confidence']:.1%}")
        
        print(f"\n決定パス:")
        for step in explanation['decision_steps']:
            print(f"   {step}")
        
        print(f"\n[DEMO] デモ完了")
    
    return 0

if __name__ == '__main__':
    exit_code = main()
    
    print(f"\n終了コード: {exit_code}")
    if exit_code == 0:
        print("[SUCCESS] 処理が正常に完了しました")
    else:
        print("[ERROR] 処理中にエラーが発生しました")
    
    sys.exit(exit_code)