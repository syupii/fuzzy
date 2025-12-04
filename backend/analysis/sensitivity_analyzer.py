import sys
import os
import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 日本語フォント設定（Windows環境の場合）
plt.rcParams['font.family'] = 'Meiryo'

# ------------------------------------------------------------------
# 1. パス設定とモジュールのインポート
# ------------------------------------------------------------------
# backendディレクトリをPythonの検索パスに追加して、coreやservicesを読み込めるようにする
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_root = os.path.dirname(current_dir) # backendフォルダ
sys.path.append(backend_root)

# あなたのシステムから必要なクラスをインポート
# ※ services.lab_matching がメインのロジックだと仮定しています
from services.lab_matching import LabMatchingService
from config.default_params import DEFAULT_PARAMS # もし定数定義があれば使う

# ------------------------------------------------------------------
# 2. 仮想データ生成クラス
# ------------------------------------------------------------------
class SimulationDataGenerator:
    def __init__(self):
        # 13項目の基本パラメータ名（仮定）
        self.basic_params = [
            'GPA', 'Math', 'Programming', 'English', 'Physics', 
            'Communication', 'Leadership', 'Creativity', 'Logic', 
            'Diligence', 'Teamwork', 'Autonomy', 'StressTolerance'
        ]
        # 27分野の興味スコア（仮定: Field_01 ~ Field_27）
        self.interest_fields = [f'Field_{i:02d}' for i in range(1, 28)]

    def generate_random_student(self):
        """
        システムの入力形式に合わせた辞書データを生成する
        """
        # 基本項目 (0.0 ~ 5.0 の範囲と仮定)
        data = {k: np.random.uniform(1.0, 5.0) for k in self.basic_params}
        
        # 興味分野 (0 ~ 100 の範囲と仮定)
        interests = {k: np.random.randint(0, 101) for k in self.interest_fields}
        
        # システムが期待する入力形式に統合 (schemas.pyの定義に合わせる)
        # ※ ここはあなたのAPIが受け取るJSON構造に合わせて調整してください
        input_payload = {
            "basic_stats": data,
            "interests": interests,
            "preferences": {"rank_preference": np.random.randint(1, 4)} # 希望順位など
        }
        return input_payload

# ------------------------------------------------------------------
# 3. 分析メイン処理
# ------------------------------------------------------------------
async def run_analysis():
    print(">>> システムを初期化中...")
    
    # サービス層のインスタンス化（DB読み込み等がここで行われる想定）
    # ※ もし初期化に引数が必要なら追加してください
    matcher_service = LabMatchingService() 
    generator = SimulationDataGenerator()
    
    n_samples = 3000  # 試行回数
    results = []

    print(f">>> {n_samples}件のシミュレーションを実行中...")

    for i in range(n_samples):
        # 1. 仮想データの生成
        mock_input = generator.generate_random_student()
        
        # 2. システムによる推論実行
        # FastAPIのサービスは async def で定義されていることが多いため await を使用
        # ※ メソッド名は実際のコードに合わせて変更してください (例: match, predict_top_lab 等)
        prediction = await matcher_service.match_student(mock_input) 
        
        # 3. 結果の抽出
        # predictionの結果構造に合わせて調整が必要です
        # 例: prediction = {"best_lab": "LabA", "score": 95.5, "all_scores": {...}}
        winner_lab = prediction.get('best_lab_id') or prediction.get('name')
        winner_score = prediction.get('score')
        
        # 2位とのスコア差（感度）を計算
        all_scores = prediction.get('all_candidates', [])
        margin = 0
        if len(all_scores) >= 2:
            # スコア順に並んでいると仮定。辞書なら sorted(values) する
            scores = sorted([x['score'] for x in all_scores], reverse=True)
            margin = scores[0] - scores[1]

        # 4. 分析用データとしてフラット化して保存
        # 入力パラメータを展開
        record = {}
        record.update(mock_input['basic_stats'])
        record.update(mock_input['interests'])
        record['Winning_Lab'] = winner_lab
        record['Margin'] = margin
        results.append(record)

    # DataFrame化
    df = pd.DataFrame(results)
    
    print(">>> 分析完了。可視化を実行します。")
    visualize_results(df)

# ------------------------------------------------------------------
# 4. 可視化関数
# ------------------------------------------------------------------
def visualize_results(df):
    # ① ヒートマップ（各ゼミの「勝ちパターン」偏差）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Marginは入力パラメータではないので除外
    if 'Margin' in numeric_cols: numeric_cols.remove('Margin')
    
    global_mean = df[numeric_cols].mean()
    lab_means = df.groupby('Winning_Lab')[numeric_cols].mean()
    importance = lab_means - global_mean
    
    plt.figure(figsize=(16, 10))
    # 項目が多いので、偏差が大きい上位20項目などに絞ると見やすいかも
    sns.heatmap(importance.iloc[:, :20], cmap='coolwarm', center=0, annot=False)
    plt.title('ゼミ別：推薦決定要因のヒートマップ（赤＝この値が高いと選ばれやすい）')
    plt.tight_layout()
    plt.savefig('backend/analysis/factor_heatmap.png') # 保存
    print("画像保存: backend/analysis/factor_heatmap.png")

    # ② 感度分析（箱ひげ図）
    plt.figure(figsize=(14, 6))
    order = sorted(df['Winning_Lab'].unique())
    sns.boxplot(x='Winning_Lab', y='Margin', data=df, order=order)
    plt.title('感度分析：1位と2位のスコア差（高いほど「揺るぎない」推薦）')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('backend/analysis/sensitivity_boxplot.png')
    print("画像保存: backend/analysis/sensitivity_boxplot.png")

# ------------------------------------------------------------------
# 実行エントリポイント
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 非同期関数を実行するための定型句
    asyncio.run(run_analysis())