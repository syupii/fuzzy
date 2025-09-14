// frontend/src/App.tsx - 修正版
import React, { useState, useEffect } from 'react';
import {
  ThemeProvider,
  createTheme,
  CssBaseline,
  AppBar,
  Toolbar,
  Typography,
  Container,
  Box,
  Tabs,
  Tab,
  Alert,
  CircularProgress,
  Backdrop
} from '@mui/material';
import EvaluationForm from './components/EvaluationForm';
import ResultsList from './components/ResultsList'; // ★ default importに修正
import { EvaluationResponse, testApiConnection } from './services/api';

// Material-UI テーマ設定
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: [
      'Noto Sans JP',
      'Roboto',
      'Arial',
      'sans-serif'
    ].join(','),
  },
});

// タブパネルコンポーネント
interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ py: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function App() {
  const [currentTab, setCurrentTab] = useState(0);
  const [evaluationResponse, setEvaluationResponse] = useState<EvaluationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');

  // 初期接続チェック
  useEffect(() => {
    const checkConnection = async () => {
      try {
        const isConnected = await testApiConnection();
        setConnectionStatus(isConnected ? 'connected' : 'disconnected');
        if (!isConnected) {
          setError('バックエンドサーバーに接続できません。サーバーが起動しているか確認してください。');
        }
      } catch (err) {
        setConnectionStatus('disconnected');
        setError('接続チェック中にエラーが発生しました。');
      }
    };

    checkConnection();
  }, []);

  // タブ変更ハンドラ
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setCurrentTab(newValue);
  };

  // 評価結果ハンドラ
  const handleEvaluationResults = (response: EvaluationResponse) => {
    console.log('📊 評価結果受信:', response);
    setEvaluationResponse(response);
    setError(null);
    setCurrentTab(1); // 結果タブに自動切り替え
  };

  // エラーハンドラ
  const handleEvaluationError = (errorMessage: string) => {
    console.error('❌ 評価エラー:', errorMessage);
    setError(errorMessage);
    setEvaluationResponse(null);
  };

  // 接続状態表示
  const renderConnectionStatus = () => {
    if (connectionStatus === 'checking') {
      return (
        <Alert severity="info" sx={{ mb: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <CircularProgress size={16} />
            <Typography variant="body2">
              バックエンドサーバー接続確認中...
            </Typography>
          </Box>
        </Alert>
      );
    }

    if (connectionStatus === 'disconnected') {
      return (
        <Alert severity="error" sx={{ mb: 2 }}>
          <Typography variant="body2">
            バックエンドサーバーに接続できません。サーバーが起動しているか確認してください。
          </Typography>
        </Alert>
      );
    }

    return (
      <Alert severity="success" sx={{ mb: 2 }}>
        <Typography variant="body2">
          ✅ サーバー接続完了 - システム利用可能
        </Typography>
      </Alert>
    );
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ minHeight: '100vh', bgcolor: 'background.default' }}>
        {/* ヘッダー */}
        <AppBar position="static" elevation={0}>
          <Toolbar>
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              🧬 遺伝的アルゴリズム × ファジィ決定木 研究室マッチングシステム
            </Typography>
          </Toolbar>
        </AppBar>

        <Container maxWidth="lg" sx={{ py: 4 }}>
          {/* 接続状態 */}
          {renderConnectionStatus()}

          {/* エラー表示 */}
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {/* メインタブ */}
          <Box sx={{ bgcolor: 'white', borderRadius: 2, overflow: 'hidden', boxShadow: 1 }}>
            <Tabs
              value={currentTab}
              onChange={handleTabChange}
              sx={{ borderBottom: 1, borderColor: 'divider', bgcolor: 'grey.50' }}
            >
              <Tab label="評価・設定" />
              <Tab
                label={`結果 ${evaluationResponse ? `(${evaluationResponse.lab_results?.length || 0}件)` : ''}`}
                disabled={!evaluationResponse}
              />
            </Tabs>

            <TabPanel value={currentTab} index={0}>
              <EvaluationForm
                onResults={handleEvaluationResults}
                onError={handleEvaluationError}
              />
            </TabPanel>

            <TabPanel value={currentTab} index={1}>
              {evaluationResponse ? (
                <ResultsList evaluationResponse={evaluationResponse} />
              ) : (
                <Alert severity="info">
                  まず評価を実行してください。
                </Alert>
              )}
            </TabPanel>
          </Box>

          {/* ローディングオーバーレイ */}
          <Backdrop
            sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}
            open={isLoading}
          >
            <Box textAlign="center">
              <CircularProgress color="inherit" />
              <Typography variant="h6" sx={{ mt: 2 }}>
                AIによる研究室適合性評価中...
              </Typography>
            </Box>
          </Backdrop>
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;