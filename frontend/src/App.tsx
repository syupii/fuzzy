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
  Switch,
  FormControlLabel,
  Tooltip,
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { 
  Science, 
  Refresh, 
  Info, 
  Settings, 
  TrendingUp, 
  AutoAwesome 
} from '@mui/icons-material';
import EvaluationForm from './components/EvaluationForm';
import EnhancedEvaluationForm from './components/EnhancedEvaluationForm';
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
  const [useEnhancedForm, setUseEnhancedForm] = useState<boolean>(true); // 拡張モードをデフォルトに

  useEffect(() => {
    checkHealth();
    // ローカルストレージから設定を復元
    const savedFormMode = localStorage.getItem('fdtlss_enhanced_mode');
    if (savedFormMode !== null) {
      setUseEnhancedForm(JSON.parse(savedFormMode));
    }
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

  const handleFormModeChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newMode = event.target.checked;
    setUseEnhancedForm(newMode);
    localStorage.setItem('fdtlss_enhanced_mode', JSON.stringify(newMode));
    
    // 結果をクリア（フォームが変わったため）
    if (results) {
      setResults(null);
    }
  };

  // 安全にlab_countを取得する関数
  const getLabCount = (health: HealthStatus | null): number => {
    if (!health) return 0;
    
    // database.lab_countを優先、なければ直接のlab_countプロパティ
    return health.database?.lab_count || health.lab_count || 0;
  };

  // データベースステータスを取得
  const getDatabaseStatus = (health: HealthStatus | null): string => {
    if (!health) return '不明';
    
    const labCount = getLabCount(health);
    return health.database?.status === 'connected' || health.status === 'healthy' 
      ? `正常 (${labCount}研究室)` 
      : '接続エラー';
  };

  // バージョン情報を取得
  const getVersion = (health: HealthStatus | null): string => {
    return health?.version || 'dev';
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      
      {/* アプリバー */}
      <AppBar position="static" elevation={0}>
        <Toolbar>
          <Science sx={{ mr: 2 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            FDTLSS - ファジィ決定木研究室選択支援システム
          </Typography>
          
          {/* フォームモード切り替え */}
          <Tooltip title={useEnhancedForm ? "シンプルモードに切り替え" : "拡張モードに切り替え"}>
            <FormControlLabel
              control={
                <Switch
                  checked={useEnhancedForm}
                  onChange={handleFormModeChange}
                  color="secondary"
                />
              }
              label={
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  {useEnhancedForm ? <AutoAwesome /> : <Settings />}
                  <Typography variant="body2">
                    {useEnhancedForm ? '拡張モード' : 'シンプルモード'}
                  </Typography>
                </Box>
              }
              sx={{ color: 'white' }}
            />
          </Tooltip>
        </Toolbar>
      </AppBar>

      <Container maxWidth="lg" sx={{ py: 4, minHeight: 'calc(100vh - 200px)' }}>
        {/* エラー表示 */}
        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Info />
              {error}
            </Box>
            <Button 
              size="small" 
              onClick={checkHealth} 
              sx={{ mt: 1 }}
              startIcon={<Refresh />}
            >
              再接続を試行
            </Button>
          </Alert>
        )}

        {/* システム状態表示 */}
        {healthStatus && !error && (
          <Alert severity="success" sx={{ mb: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
              <Chip 
                icon={<Science />} 
                label="システム正常" 
                color="success" 
                size="small" 
              />
              <Chip 
                label={`DB: ${getDatabaseStatus(healthStatus)}`} 
                color="primary" 
                size="small" 
              />
              <Chip 
                label={`v${getVersion(healthStatus)}`} 
                color="default" 
                size="small" 
              />
              {healthStatus.database?.evaluation_count !== undefined && (
                <Chip 
                  label={`評価履歴: ${healthStatus.database.evaluation_count}件`} 
                  color="info" 
                  size="small" 
                />
              )}
              {useEnhancedForm && (
                <Chip 
                  icon={<AutoAwesome />}
                  label="拡張機能有効" 
                  color="secondary" 
                  size="small" 
                />
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
            background: useEnhancedForm 
              ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
              : 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
            color: 'white'
          }}
        >
          <Typography variant="h3" gutterBottom fontWeight="bold">
            🎯 研究室マッチングシステム
          </Typography>
          <Typography variant="h6" sx={{ opacity: 0.9 }}>
            {useEnhancedForm 
              ? '🔬 北海道情報大学 研究分野特化型マッチング'
              : '📊 基本設定による研究室選択支援'
            }
          </Typography>
          
          {/* モード説明 */}
          <Box sx={{ mt: 3 }}>
            <Typography variant="body2" sx={{ opacity: 0.8 }}>
              {useEnhancedForm 
                ? '基本設定 + 研究分野興味度による高精度マッチング。北海道情報大学の実際の研究分野に基づいた詳細評価が可能です。'
                : 'シンプルな5項目評価による基本的なマッチング。手軽に研究室の適合度を確認できます。'
              }
            </Typography>
          </Box>
        </Paper>

        {/* 評価フォーム */}
        <Box sx={{ mb: 4 }}>
          {useEnhancedForm ? (
            <EnhancedEvaluationForm onResults={handleResults} />
          ) : (
            <EvaluationForm onResults={handleResults} />
          )}
        </Box>

        {/* 結果表示 */}
        {results && (
          <Box id="results-section">
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
              <Typography variant="h4" component="h2">
                📊 評価結果
              </Typography>
              <Box sx={{ display: 'flex', gap: 2 }}>
                {/* 結果に関する追加情報 */}
                {results.summary.field_analysis && (
                  <Chip
                    icon={<TrendingUp />}
                    label={`${results.summary.field_analysis.selected_fields_count}分野解析済み`}
                    color="primary"
                    variant="outlined"
                  />
                )}
                <Button
                  variant="outlined"
                  onClick={clearResults}
                  startIcon={<Refresh />}
                >
                  新しい評価
                </Button>
              </Box>
            </Box>
            <ResultsList data={results} />
          </Box>
        )}

        {/* 機能説明セクション */}
        {!results && (
          <Paper sx={{ p: 4, mt: 4 }}>
            <Typography variant="h5" gutterBottom>
              💡 システム機能
            </Typography>
            
            <Box sx={{ mt: 3 }}>
              <Typography variant="h6" gutterBottom color="primary">
                🔄 フォームモード
              </Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1fr' }, gap: 3, mb: 3 }}>
                <Paper sx={{ p: 3, border: useEnhancedForm ? '2px solid #1976d2' : '1px solid #e0e0e0' }}>
                  <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                    🎨 拡張モード（推奨）
                  </Typography>
                  <Typography variant="body2" paragraph>
                    北海道情報大学の16の実際の研究分野から興味領域を選択し、
                    基本設定と組み合わせた高精度マッチングを実現。
                  </Typography>
                  <Box component="ul" sx={{ pl: 3, mb: 0 }}>
                    <li>分野別興味度設定</li>
                    <li>AIによる分野推薦</li>
                    <li>カテゴリー別分析</li>
                    <li>より精密な適合度計算</li>
                  </Box>
                </Paper>
                
                <Paper sx={{ p: 3, border: !useEnhancedForm ? '2px solid #1976d2' : '1px solid #e0e0e0' }}>
                  <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                    ⚡ シンプルモード
                  </Typography>
                  <Typography variant="body2" paragraph>
                    5つの基本項目による手軽な研究室マッチング。
                    初回利用や概要把握に最適。
                  </Typography>
                  <Box component="ul" sx={{ pl: 3, mb: 0 }}>
                    <li>研究強度・指導スタイル等</li>
                    <li>高速評価</li>
                    <li>分かりやすい結果</li>
                    <li>デモデータ対応</li>
                  </Box>
                </Paper>
              </Box>

              <Typography variant="h6" gutterBottom color="primary">
                🤖 アルゴリズム特徴
              </Typography>
              <Typography variant="body1" paragraph>
                このシステムは、<strong>適応型ファジィ決定木（AFDT）</strong>を用いた
                最先端のアルゴリズムにより、あいまいな要求も含めて柔軟に処理し、
                各ユーザーに最適化された推薦を提供します。
              </Typography>
              
              <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr 1fr' }, gap: 2 }}>
                <Paper sx={{ p: 2, textAlign: 'center', backgroundColor: '#f8f9fa' }}>
                  <Typography variant="h6" color="primary">🧠</Typography>
                  <Typography variant="subtitle2">ファジィ論理</Typography>
                  <Typography variant="body2">あいまいな評価も適切に処理</Typography>
                </Paper>
                <Paper sx={{ p: 2, textAlign: 'center', backgroundColor: '#f8f9fa' }}>
                  <Typography variant="h6" color="primary">🌳</Typography>
                  <Typography variant="subtitle2">決定木</Typography>
                  <Typography variant="body2">透明性の高い判断プロセス</Typography>
                </Paper>
                <Paper sx={{ p: 2, textAlign: 'center', backgroundColor: '#f8f9fa' }}>
                  <Typography variant="h6" color="primary">🎯</Typography>
                  <Typography variant="subtitle2">適応学習</Typography>
                  <Typography variant="body2">利用により精度が向上</Typography>
                </Paper>
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
            {useEnhancedForm ? ' 拡張版（研究分野特化）' : ' 標準版'} | 
            Prototype Version | 
            React + Flask + SQLite
          </Typography>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;