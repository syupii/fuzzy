// src/App.tsx - 修正版
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

  // 修正：EvaluationResponseを受け取るように変更
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
      <AppBar position="static" elevation={0}>
        <Toolbar>
          <Science sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            ファジィ決定木研究室選択支援システム（拡張版）
          </Typography>
          <Button color="inherit" onClick={checkHealth} startIcon={<Refresh />}>
            再接続
          </Button>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* システム状態表示 */}
        {healthStatus && (
          <Alert
            severity={healthStatus.status === 'healthy' ? 'success' : 'warning'}
            sx={{ mb: 3 }}
            action={
              <Box sx={{ display: 'flex', gap: 1 }}>
                <Chip
                  label={`研究室: ${getLabCount(healthStatus)}`}
                  size="small"
                  color={healthStatus.status === 'healthy' ? 'success' : 'warning'}
                />
                <Chip
                  label={`DB: ${getDatabaseStatus(healthStatus)}`}
                  size="small"
                  color={getDatabaseStatus(healthStatus) === 'OK' ? 'success' : 'warning'}
                />
                <Chip
                  label={getVersion(healthStatus)}
                  size="small"
                  variant="outlined"
                />
              </Box>
            }
          >
            <Box>
              <Typography variant="body1" fontWeight="bold">
                {healthStatus.message}
              </Typography>
            </Box>
          </Alert>
        )}

        {/* エラー表示 */}
        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {/* ヒーローセクション */}
        <Paper
          sx={{
            p: 4,
            mb: 4,
            background: 'linear-gradient(45deg, #1976d2 30%, #42a5f5 90%)',
            color: 'white',
            textAlign: 'center',
          }}
        >
          <Typography variant="h4" component="h1" gutterBottom fontWeight="bold">
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
                このシステムは、実際の学生調査に基づいて設計された20項目の評価基準を使用して、
                あなたに最適な研究室を見つけるお手伝いをします。
              </Typography>
              <Typography variant="body1" paragraph>
                ファジィ決定木と遺伝的アルゴリズムを組み合わせることで、
                従来の単純な点数マッチングでは捉えられない複雑な適合性を評価します。
              </Typography>
              <Typography variant="body1" paragraph>
                研究分野の興味、技術スタックの経験、研究環境の好みなど、
                多角的な視点から総合的な適合度を算出します。
              </Typography>
            </Box>

            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mt: 3 }}>
              <Chip icon={<Psychology />} label="ファジィ論理ベース" color="primary" />
              <Chip icon={<TrendingUp />} label="学生調査基盤" color="secondary" />
              <Chip icon={<Science />} label="20項目評価" color="success" />
            </Box>
          </Paper>
        )}
      </Container>
    </ThemeProvider>
  );
}

export default App;