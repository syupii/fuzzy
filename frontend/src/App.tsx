import React, { useState, useEffect } from 'react';
import {
  Container,
  AppBar,
  Toolbar,
  Typography,
  Box,
  CssBaseline,
  Paper,
  Alert,
  Chip,
  Button,
  Grid,
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { Science, Refresh, Info, TrendingUp, Psychology } from '@mui/icons-material';
import EvaluationForm from './components/EvaluationForm';
import ResultsList from './components/ResultsList';
import { apiService, EvaluationResponse } from './services/api';

// 型定義を追加
interface HealthStatus {
  status: string;
  message: string;
  version?: string;
  database?: {
    status: string;
    lab_count: number;
    evaluation_count: number;
    table_counts?: { [key: string]: number };
    size_info?: { [key: string]: any };
  };
  lab_count?: number; // 直接のプロパティの場合
  [key: string]: any;
}

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    success: {
      main: '#2e7d32',
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          textTransform: 'none',
          fontWeight: 600,
        },
      },
    },
  },
});

function App() {
  const [results, setResults] = useState<EvaluationResponse | null>(null);
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const health = await apiService.healthCheck();
      console.log('🔍 Health Status Response:', health); // デバッグ用
      setHealthStatus(health);
      setError(null);
    } catch (err: any) {
      setError('バックエンドサーバーに接続できません。サーバーが起動しているか確認してください。');
      console.error('Health check failed:', err);
    }
  };

  const handleResults = (newResults: EvaluationResponse) => {
    setResults(newResults);
    // 結果セクションまでスクロール
    setTimeout(() => {
      const resultsElement = document.getElementById('results-section');
      if (resultsElement) {
        resultsElement.scrollIntoView({ behavior: 'smooth' });
      }
    }, 100);
  };

  const clearResults = () => {
    setResults(null);
  };

  // 安全にlab_countを取得する関数
  const getLabCount = (health: HealthStatus | null): number => {
    if (!health) return 0;
    
    // database.lab_countを優先、なければ直接のlab_countプロパティ
    return health.database?.lab_count || health.lab_count || 0;
  };

  // データベース状態を取得
  const getDatabaseStatus = (health: HealthStatus | null): string => {
    if (!health) return '不明';
    return health.database?.status || 'OK';
  };

  // バージョン情報を取得
  const getVersion = (health: HealthStatus | null): string => {
    if (!health) return '不明';
    return health.version || 'v2.0';
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      
      {/* ヘッダー */}
      <AppBar position="static" elevation={0}>
        <Toolbar>
          <Science sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            FDTLSS v2.0 - 拡張版研究室マッチングシステム
          </Typography>
          <Chip 
            label="20項目評価" 
            color="secondary" 
            variant="filled"
            size="small"
          />
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ py: 4, minHeight: '100vh' }}>
        {/* エラー表示 */}
        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
            <Button onClick={checkHealth} size="small" sx={{ ml: 2 }}>
              再接続
            </Button>
          </Alert>
        )}

        {/* システム状態表示 */}
        {healthStatus && !error && (
          <Alert severity="success" sx={{ mb: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
              <Info />
              システム正常動作中 | 
              研究室データ: {getLabCount(healthStatus)}件 |
              データベース: {getDatabaseStatus(healthStatus)} | 
              バージョン: {getVersion(healthStatus)}
              {healthStatus.database?.evaluation_count !== undefined && (
                <> | 評価履歴: {healthStatus.database.evaluation_count}件</>
              )}
            </Box>
          </Alert>
        )}

        {/* デバッグ情報（開発時のみ表示） */}
        {process.env.NODE_ENV === 'development' && healthStatus && (
          <Alert severity="info" sx={{ mb: 3 }}>
            <details>
              <summary style={{ cursor: 'pointer' }}>🔧 Debug Info</summary>
              <pre style={{ fontSize: '12px', marginTop: '10px', overflow: 'auto' }}>
                {JSON.stringify(healthStatus, null, 2)}
              </pre>
            </details>
          </Alert>
        )}

        {/* メインヘッダー */}
        <Paper 
          sx={{ 
            p: 4, 
            mb: 4, 
            textAlign: 'center',
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            color: 'white'
          }}
        >
          <Typography variant="h3" gutterBottom fontWeight="bold">
            拡張版研究室マッチングシステム
          </Typography>
          <Typography variant="h6" sx={{ mb: 2, opacity: 0.9 }}>
            学生調査に基づく19項目による高精度適合度評価
          </Typography>
          
          {/* 特徴アイコン */}
          <Grid container spacing={2} justifyContent="center" sx={{ mt: 2 }}>
            <Grid item>
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <Psychology sx={{ fontSize: 32, mb: 1 }} />
                <Typography variant="caption">ファジィ論理</Typography>
              </Box>
            </Grid>
            <Grid item>
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <TrendingUp sx={{ fontSize: 32, mb: 1 }} />
                <Typography variant="caption">調査基盤</Typography>
              </Box>
            </Grid>
            <Grid item>
              <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                <Science sx={{ fontSize: 32, mb: 1 }} />
                <Typography variant="caption">20項目評価</Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>

        {/* 評価フォーム */}
        <Box sx={{ mb: 4 }}>
          <EvaluationForm onResults={handleResults} />
        </Box>

        {/* 結果表示 */}
        {results && (
          <Box id="results-section">
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h4" component="h2">
                📊 評価結果
              </Typography>
              <Button
                variant="outlined"
                onClick={clearResults}
                startIcon={<Refresh />}
              >
                新しい評価
              </Button>
            </Box>
            <ResultsList data={results} />
          </Box>
        )}

        {/* 説明セクション */}
        {!results && (
          <Paper sx={{ p: 4, mt: 4 }}>
            <Typography variant="h5" gutterBottom>
              🔬 システムについて
            </Typography>
            <Box sx={{ mt: 2 }}>
              <Typography variant="body1" paragraph>
                このシステムは、<strong>実際の学生調査結果</strong>と<strong>ファジィ論理</strong>を組み合わせた
                高度なアルゴリズムにより、あなたの希望と各研究室の特徴を19の観点から多角的に分析し、
                最適なマッチングを提供します。
              </Typography>
              
              <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
                📋 評価基準（19項目）
              </Typography>
              
              <Grid container spacing={3} sx={{ mt: 1 }}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                    基本的な研究環境（5項目）
                  </Typography>
                  <Box component="ul" sx={{ pl: 3, mb: 2 }}>
                    <li><strong>研究強度</strong>: 研究活動の集中度・最先端性</li>
                    <li><strong>指導スタイル</strong>: 教授の指導方針（厳格 ↔ 自由）</li>
                    <li><strong>チームワーク</strong>: 研究での協働度（個人 ↔ チーム）</li>
                    <li><strong>ワークロード</strong>: 研究の負荷・忙しさ</li>
                    <li><strong>理論・実践バランス</strong>: 理論研究と実践的研究の比重</li>
                  </Box>

                  <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                    学習・成長（3項目）
                  </Typography>
                  <Box component="ul" sx={{ pl: 3, mb: 2 }}>
                    <li><strong>スキル開発</strong>: 専門特化 ↔ 幅広いスキル</li>
                    <li><strong>学習ペース</strong>: じっくり型 ↔ 高速習得型</li>
                    <li><strong>難易度志向</strong>: 安定した課題 ↔ 挑戦的課題</li>
                  </Box>

                  <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                    コミュニケーション・環境（3項目）
                  </Typography>
                  <Box component="ul" sx={{ pl: 3, mb: 2 }}>
                    <li><strong>コミュニケーション</strong>: 少人数密接 ↔ オープン交流</li>
                    <li><strong>ミーティング頻度</strong>: 必要最小限 ↔ 頻繁な相談</li>
                    <li><strong>研究室雰囲気</strong>: 静寂集中 ↔ 活発議論</li>
                  </Box>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                    研究アプローチ（3項目）
                  </Typography>
                  <Box component="ul" sx={{ pl: 3, mb: 2 }}>
                    <li><strong>革新性リスク</strong>: 確実な成果 ↔ 革新的挑戦</li>
                    <li><strong>手法志向</strong>: 伝統的手法 ↔ 新しい手法</li>
                    <li><strong>学際性</strong>: 専門特化 ↔ 分野横断</li>
                  </Box>

                  <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                    時間・ライフスタイル（2項目）
                  </Typography>
                  <Box component="ul" sx={{ pl: 3, mb: 2 }}>
                    <li><strong>時間の柔軟性</strong>: 規則正しい ↔ 自由なスケジュール</li>
                    <li><strong>時間外研究</strong>: 平日のみ ↔ 夜間・休日も</li>
                  </Box>

                  <Typography variant="subtitle1" fontWeight="bold" gutterBottom color="error">
                    🔥 重要成功要因（4項目）
                  </Typography>
                  <Box component="ul" sx={{ pl: 3, mb: 2 }}>
                    <li><strong>論文執筆機会</strong>: 在学中の論文著者可能性</li>
                    <li><strong>経済的支援</strong>: 研究費用・経済面サポート</li>
                    <li><strong>研究室上下関係</strong>: 厳格な階層 ↔ フラット</li>
                    <li><strong>コアタイム柔軟性</strong>: 必須滞在時間の自由度</li>
                  </Box>
                </Grid>
              </Grid>

              <Alert severity="info" sx={{ mt: 3 }}>
                <Typography variant="body2">
                  <strong>📊 調査データに基づく設計</strong><br />
                  重要成功要因の4項目は、理系大学生79人への調査と複数の学術文献レビューから
                  特定された、研究室選択で最も重視される要因です。これらの項目は評価重みが
                  高く設定されており、より精度の高いマッチングを実現します。
                </Typography>
              </Alert>
            </Box>
          </Paper>
        )}
      </Container>

      {/* フッター */}
      <Box
        component="footer"
        sx={{
          py: 3,
          px: 2,
          mt: 'auto',
          backgroundColor: 'grey.100',
        }}
      >
        <Container maxWidth="lg">
          <Typography variant="body2" color="text.secondary" align="center">
            © 2025 FDTLSS v2.0 - Enhanced Fuzzy Decision Tree Lab Selection System | 
            20項目評価版 | 
            分離アーキテクチャ（React + Flask + SQLite）| 
            調査データベース設計
          </Typography>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;