// frontend/src/App.tsx - 構文エラー修正版
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
import { EvaluationForm } from './components/EvaluationForm';
import { ResultsList } from './components/ResultsList';
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
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <CircularProgress size={20} sx={{ mr: 2 }} />
            バックエンドサーバーとの接続を確認中...
          </Box>
        </Alert>
      );
    }

    if (connectionStatus === 'disconnected') {
      return (
        <Alert severity="error" sx={{ mb: 2 }}>
          バックエンドサーバーに接続できません。
          <br />
          ターミナルで `cd backend && python app.py` を実行してサーバーを起動してください。
        </Alert>
      );
    }

    return (
      <Alert severity="success" sx={{ mb: 2 }}>
        ✅ バックエンドサーバーに接続済み
      </Alert>
    );
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />

      {/* ローディングバックドロップ */}
      <Backdrop
        sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}
        open={isLoading}
      >
        <Box sx={{ textAlign: 'center' }}>
          <CircularProgress color="inherit" size={60} />
          <Typography variant="h6" sx={{ mt: 2 }}>
            研究室を評価中...
          </Typography>
        </Box>
      </Backdrop>

      {/* アプリバー */}
      <AppBar position="static" elevation={1}>
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            🧬🌳 研究室選択支援システム v2.0
          </Typography>
          <Typography variant="body2">
            遺伝的アルゴリズム + ファジィ決定木
          </Typography>
        </Toolbar>
      </AppBar>

      {/* メインコンテンツ */}
      <Container maxWidth="lg" sx={{ mt: 2, mb: 4 }}>

        {/* 接続状態表示 */}
        {renderConnectionStatus()}

        {/* エラー表示 */}
        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* タブナビゲーション */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 1 }}>
          <Tabs
            value={currentTab}
            onChange={handleTabChange}
            aria-label="研究室評価システムタブ"
          >
            <Tab
              label="研究室評価"
              id="tab-0"
              aria-controls="tabpanel-0"
            />
            <Tab
              label={`評価結果 ${evaluationResponse ? `(${evaluationResponse.summary.total_labs}件)` : ''}`}
              id="tab-1"
              aria-controls="tabpanel-1"
              disabled={!evaluationResponse}
            />
          </Tabs>
        </Box>

        {/* タブコンテンツ */}
        <Box>
          {/* 評価フォームタブ */}
          <TabPanel value={currentTab} index={0}>
            {/* プロパティ名修正: onResults使用 */}
            <EvaluationForm
              onResults={handleEvaluationResults}
              onError={handleEvaluationError}
            />
          </TabPanel>

          {/* 結果表示タブ */}
          <TabPanel value={currentTab} index={1}>
            {evaluationResponse ? (
              <ResultsList evaluationResponse={evaluationResponse} />
            ) : (
              <Alert severity="info">
                評価結果を表示するには、まず「研究室評価」タブで評価を実行してください。
              </Alert>
            )}
          </TabPanel>
        </Box>

        {/* フッター */}
        <Box sx={{ mt: 4, pt: 2, borderTop: 1, borderColor: 'divider' }}>
          <Typography variant="body2" color="text.secondary" align="center">
            研究室選択支援システム v2.0 - 遺伝的アルゴリズムとファジィ決定木による最適マッチング
          </Typography>
          <Typography variant="caption" color="text.secondary" align="center" display="block">
            13項目の評価基準と11の研究分野を用いた高精度な適合性評価
          </Typography>
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App;