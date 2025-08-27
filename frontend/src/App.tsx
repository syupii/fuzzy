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
  Stepper,
  Step,
  StepLabel,
  StepContent,
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { Science, Refresh, Info, TrendingUp, Psychology, Assessment, School, CheckCircle } from '@mui/icons-material';
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

// システムの使用ステップ定義
const systemSteps = [
  {
    label: 'システム確認',
    description: 'バックエンドシステムとデータベースの状態を確認',
    icon: <Assessment />,
    completed: false,
  },
  {
    label: '評価項目設定',
    description: '20項目の評価基準を1-10スケールで設定',
    icon: <School />,
    completed: false,
  },
  {
    label: '適合度評価',
    description: 'ファジィ決定木アルゴリズムで研究室との適合度を計算',
    icon: <Psychology />,
    completed: false,
  },
  {
    label: '結果確認',
    description: '評価結果と推薦研究室を確認',
    icon: <CheckCircle />,
    completed: false,
  },
];

function App() {
  const [results, setResults] = useState<EvaluationResponse | null>(null);
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeStep, setActiveStep] = useState(0);
  const [steps, setSteps] = useState(systemSteps);

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const health = await apiService.healthCheck();
      console.log('🔍 Health Status Response:', health); // デバッグ用
      setHealthStatus(health);
      setError(null);
      
      // ステップ1完了
      updateStepCompletion(0, true);
      if (activeStep === 0) {
        setActiveStep(1);
      }
    } catch (err: any) {
      setError('バックエンドサーバーに接続できません。サーバーが起動しているか確認してください。');
      console.error('Health check failed:', err);
      updateStepCompletion(0, false);
    }
  };

  const updateStepCompletion = (stepIndex: number, completed: boolean) => {
    setSteps(prevSteps => 
      prevSteps.map((step, index) => 
        index === stepIndex ? { ...step, completed } : step
      )
    );
  };

  const handleStepClick = (stepIndex: number) => {
    setActiveStep(stepIndex);
  };

  const handleResults = (newResults: EvaluationResponse) => {
    setResults(newResults);
    
    // ステップ2, 3完了
    updateStepCompletion(1, true);
    updateStepCompletion(2, true);
    setActiveStep(3);
    
    // 結果セクションまでスクロール
    setTimeout(() => {
      const resultsElement = document.getElementById('results-section');
      if (resultsElement) {
        resultsElement.scrollIntoView({ behavior: 'smooth' });
        updateStepCompletion(3, true);
      }
    }, 100);
  };

  const clearResults = () => {
    setResults(null);
    // ステップをリセット
    updateStepCompletion(2, false);
    updateStepCompletion(3, false);
    setActiveStep(1);
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

        {/* プログレスステッパー */}
        <Paper sx={{ p: 3, mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            📊 システム使用手順
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', justifyContent: 'center' }}>
            {steps.map((step, index) => (
              <Box
                key={index}
                sx={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  gap: 1,
                  minWidth: 120,
                  cursor: 'pointer',
                  p: 2,
                  borderRadius: 2,
                  bgcolor: activeStep === index ? 'primary.50' : 'transparent',
                  '&:hover': { bgcolor: 'action.hover' },
                  border: activeStep === index ? '2px solid' : '1px solid',
                  borderColor: activeStep === index ? 'primary.main' : 'divider',
                }}
                onClick={() => handleStepClick(index)}
              >
                <Box
                  sx={{
                    width: 40,
                    height: 40,
                    borderRadius: '50%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    bgcolor: step.completed ? 'success.main' : 'background.paper',
                    color: step.completed ? 'white' : 'text.primary',
                    cursor: 'pointer',
                    fontSize: '20px',
                    '&:hover': { bgcolor: step.completed ? 'success.dark' : 'action.hover' },
                    '& > *': { fontSize: 'inherit' }
                  }}
                  onClick={() => handleStepClick(index)}
                >
                  {step.icon}
                </Box>
                <Typography 
                  variant="caption" 
                  textAlign="center"
                  color={activeStep === index ? 'primary.main' : 'text.secondary'}
                  fontWeight={activeStep === index ? 'bold' : 'normal'}
                >
                  {step.label}
                </Typography>
                {step.completed && (
                  <Chip label="完了" size="small" color="success" />
                )}
              </Box>
            ))}
          </Box>
        </Paper>

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
            学生調査に基づく20項目による高精度適合度評価
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
                このシステムは、遺伝的アルゴリズムによって最適化されたファジィ決定木を使用して、
                学生の希望と研究室の特徴を20項目の詳細な基準で比較し、最適なマッチングを提供します。
              </Typography>
              
              <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
                🎯 主な特徴
              </Typography>
              
              <Grid container spacing={2} sx={{ mt: 1 }}>
                <Grid item xs={12} md={6}>
                  <Box sx={{ p: 2, bgcolor: 'primary.50', borderRadius: 1 }}>
                    <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                      📊 20項目評価
                    </Typography>
                    <Typography variant="body2">
                      研究強度、指導スタイル、チームワーク、ワークロード、理論・実践バランスなど、
                      学生調査に基づく重要な評価基準を包括的に分析
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Box sx={{ p: 2, bgcolor: 'success.50', borderRadius: 1 }}>
                    <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                      🧠 AI最適化
                    </Typography>
                    <Typography variant="body2">
                      遺伝的アルゴリズムで最適化されたファジィ決定木により、
                      複雑な判断基準を統合した高精度な適合度予測を実現
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Box sx={{ p: 2, bgcolor: 'warning.50', borderRadius: 1 }}>
                    <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                      📚 学生調査基盤
                    </Typography>
                    <Typography variant="body2">
                      実際の学生へのアンケート調査結果に基づいて重要項目を特定し、
                      現実的で実用性の高い評価システムを構築
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Box sx={{ p: 2, bgcolor: 'info.50', borderRadius: 1 }}>
                    <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                      🔍 詳細分析
                    </Typography>
                    <Typography variant="body2">
                      各項目の類似度、重み、信頼度を詳細に分析し、
                      なぜその研究室が推薦されるのかを透明性高く説明
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
              
              <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>
                🚀 使い方
              </Typography>
              
              <Box component="ol" sx={{ pl: 3 }}>
                <li>
                  <Typography variant="body1" paragraph>
                    <strong>評価項目設定:</strong> 20項目について、あなたの希望を1-10のスケールで設定
                  </Typography>
                </li>
                <li>
                  <Typography variant="body1" paragraph>
                    <strong>適合度評価実行:</strong> AIアルゴリズムが全研究室との適合度を計算
                  </Typography>
                </li>
                <li>
                  <Typography variant="body1" paragraph>
                    <strong>結果確認:</strong> 適合度順にランキング表示、詳細な分析結果も確認可能
                  </Typography>
                </li>
                <li>
                  <Typography variant="body1" paragraph>
                    <strong>研究室選択:</strong> 推薦結果を参考に、研究室見学や面談を実施
                  </Typography>
                </li>
              </Box>
            </Box>
          </Paper>
        )}

        {/* フッター */}
        <Box sx={{ textAlign: 'center', mt: 6, py: 3, borderTop: 1, borderColor: 'divider' }}>
          <Typography variant="body2" color="text.secondary">
            FDTLSS v2.0 - Fuzzy Decision Tree Laboratory Selection Support System
          </Typography>
          <Typography variant="caption" color="text.secondary">
            遺伝的アルゴリズムとファジィ論理による研究室選択支援システム
          </Typography>
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App;