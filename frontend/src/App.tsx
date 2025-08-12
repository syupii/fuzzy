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
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { Science, Refresh, Info } from '@mui/icons-material';
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

  // 安全にデータベース状態を取得する関数
  const getDatabaseStatus = (health: HealthStatus | null): string => {
    if (!health) return 'unknown';
    
    // database.statusがあればそれを、なければstatusを返す
    return health.database?.status || health.status || 'unknown';
  };

  // 安全にバージョンを取得する関数
  const getVersion = (health: HealthStatus | null): string => {
    if (!health) return 'unknown';
    
    return health.version || 'v1.0';
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      
      {/* ヘッダー */}
      <AppBar position="static" elevation={2}>
        <Toolbar>
          <Science sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            ファジィ決定木研究室選択支援システム
          </Typography>
          
          {healthStatus && (
            <Chip 
              label={`研究室数: ${getLabCount(healthStatus)}`}
              color="success"
              size="small"
              sx={{ mr: 2 }}
            />
          )}
          
          <Chip 
            label="Prototype v1.0"
            color="secondary"
            size="small"
          />
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        {/* システム状態表示 */}
        {error && (
          <Alert 
            severity="error" 
            sx={{ mb: 3 }}
            action={
              <Button color="inherit" size="small" onClick={checkHealth}>
                <Refresh />
              </Button>
            }
          >
            {error}
          </Alert>
        )}

        {healthStatus && (
          <Alert severity="success" sx={{ mb: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Info />
              システム正常動作中 | 
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
            研究室マッチングシステム
          </Typography>
          
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
                 評価結果
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
               システムについて
            </Typography>
            <Box sx={{ mt: 2 }}>
              <Typography variant="body1" paragraph>
                このシステムは、<strong>ファジィ論理</strong>を用いた高度なアルゴリズムにより、
                あなたの希望と各研究室の特徴を多角的に分析し、最適なマッチングを提供します。
              </Typography>
              
              <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
                 評価基準
              </Typography>
              <Box component="ul" sx={{ pl: 3 }}>
                <li><strong> 研究強度</strong>: 研究活動の集中度・最先端性</li>
                <li><strong> 指導スタイル</strong>: 教授の指導方針（厳格 ↔ 自由）</li>
                <li><strong> チームワーク</strong>: 研究での協働度（個人 ↔ チーム）</li>
                <li><strong> ワークロード</strong>: 研究の負荷・忙しさ</li>
                <li><strong> 理論・実践バランス</strong>: 理論研究と実践的研究の比重</li>
              </Box>
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
            © 2025 FDTLSS - Fuzzy Decision Tree Lab Selection System | 
            Prototype Version | 
            分離アーキテクチャ（React + Flask + SQLite）
          </Typography>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;