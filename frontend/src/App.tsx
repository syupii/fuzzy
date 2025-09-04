// src/App.tsx - 完全修正版
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
  Card,
  CardContent,
  LinearProgress,
  IconButton,
  Menu,
  MenuItem,
  Snackbar,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Fab,
  Tooltip,
} from '@mui/material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import {
  Science,
  Refresh,
  Info,
  TrendingUp,
  Psychology,
  Share,
  History,
  Settings,
  Help,
  GetApp,
  Close,
  CloudDownload,
  Assessment,
} from '@mui/icons-material';
import EvaluationForm from './components/EvaluationForm';
import ResultsList from './components/ResultsList';
import { apiService, EvaluationResponse } from './services/api';

// 型定義
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
  lab_count?: number;
}

interface SystemStats {
  total_evaluations: number;
  avg_score: number;
  popular_criteria: string[];
  lab_rankings: Array<{
    lab_name: string;
    avg_score: number;
    evaluation_count: number;
  }>;
}

interface AppState {
  results: EvaluationResponse | null;
  healthStatus: HealthStatus | null;
  systemStats: SystemStats | null;
  loading: boolean;
  error: string | null;
  showInfo: boolean;
  evaluationHistory: EvaluationResponse[];
}

interface SnackbarState {
  open: boolean;
  message: string;
  severity: 'success' | 'error' | 'warning' | 'info';
}

// テーマ設定
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
    h3: {
      fontWeight: 600,
    },
    h4: {
      fontWeight: 500,
    },
  },
  shape: {
    borderRadius: 8,
  },
});

const App: React.FC = () => {
  const [state, setState] = useState<AppState>({
    results: null,
    healthStatus: null,
    systemStats: null,
    loading: false,
    error: null,
    showInfo: false,
    evaluationHistory: [],
  });

  const [snackbar, setSnackbar] = useState<SnackbarState>({
    open: false,
    message: '',
    severity: 'info',
  });

  const [historyMenuAnchor, setHistoryMenuAnchor] = useState<null | HTMLElement>(null);
  const [settingsDialogOpen, setSettingsDialogOpen] = useState(false);

  // 初期化処理
  useEffect(() => {
    initializeApp();
    loadHistory();
  }, []);

  const initializeApp = async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      // ヘルス状態取得
      const health = await apiService.getHealthStatus();
      setState(prev => ({ ...prev, healthStatus: health }));

      // システム統計取得
      try {
        const stats = await apiService.getSystemStats();
        setState(prev => ({ ...prev, systemStats: stats }));
      } catch (statsError) {
        console.warn('システム統計の取得に失敗しました:', statsError);
      }

    } catch (error) {
      console.error('初期化エラー:', error);
      setState(prev => ({
        ...prev,
        error: 'サーバーに接続できません。バックエンドが起動しているか確認してください。'
      }));
    } finally {
      setState(prev => ({ ...prev, loading: false }));
    }
  };

  const loadHistory = () => {
    try {
      const savedHistory = localStorage.getItem('evaluationHistory');
      if (savedHistory) {
        const history = JSON.parse(savedHistory) as EvaluationResponse[];
        setState(prev => ({ ...prev, evaluationHistory: history.slice(0, 10) })); // 最新10件のみ
      }
    } catch (error) {
      console.warn('履歴の読み込みに失敗しました:', error);
    }
  };

  const saveToHistory = (result: EvaluationResponse) => {
    try {
      const newHistory = [result, ...state.evaluationHistory].slice(0, 10);
      setState(prev => ({ ...prev, evaluationHistory: newHistory }));
      localStorage.setItem('evaluationHistory', JSON.stringify(newHistory));
    } catch (error) {
      console.warn('履歴の保存に失敗しました:', error);
    }
  };

  const showSnackbar = (message: string, severity: SnackbarState['severity'] = 'info') => {
    setSnackbar({ open: true, message, severity });
  };

  const handleResults = (results: EvaluationResponse) => {
    setState(prev => ({ ...prev, results }));
    saveToHistory(results);
    showSnackbar('評価が完了しました！', 'success');
  };

  const handleRefresh = () => {
    setState(prev => ({ ...prev, results: null }));
    initializeApp();
    showSnackbar('アプリケーションを更新しました', 'info');
  };

  const handleHistorySelect = (result: EvaluationResponse) => {
    setState(prev => ({ ...prev, results: result }));
    setHistoryMenuAnchor(null);
    showSnackbar('履歴から結果を復元しました', 'info');
  };

  const shareResults = async () => {
    if (!state.results) return;

    try {
      const bestLab = state.results.results.length > 0 ? state.results.results[0].lab.name : '未評価';
      const bestScore = state.results.results.length > 0 ? state.results.results[0].compatibility.overall_score : 0;

      if (navigator.share) {
        await navigator.share({
          title: '研究室マッチング結果',
          text: `最適研究室: ${bestLab} (適合度: ${bestScore.toFixed(1)}%)`,
          url: window.location.href,
        });
        showSnackbar('結果を共有しました', 'success');
      } else {
        // フォールバック: クリップボードにコピー
        const shareText = `研究室マッチング結果\n最適研究室: ${bestLab}\n適合度: ${bestScore.toFixed(1)}%`;
        await navigator.clipboard.writeText(shareText);
        showSnackbar('結果をクリップボードにコピーしました', 'info');
      }
    } catch (error) {
      showSnackbar('共有に失敗しました', 'error');
    }
  };

  const exportResults = () => {
    if (!state.results) return;

    try {
      const dataStr = JSON.stringify(state.results, null, 2);
      const dataBlob = new Blob([dataStr], { type: 'application/json' });
      const url = URL.createObjectURL(dataBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `research_lab_evaluation_${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      showSnackbar('結果をエクスポートしました', 'success');
    } catch (error) {
      showSnackbar('エクスポートに失敗しました', 'error');
    }
  };

  const clearHistory = () => {
    setState(prev => ({ ...prev, evaluationHistory: [] }));
    localStorage.removeItem('evaluationHistory');
    showSnackbar('履歴をクリアしました', 'info');
    setSettingsDialogOpen(false);
  };

  const getServerStatusColor = () => {
    if (!state.healthStatus) return 'error';
    return state.healthStatus.status === 'healthy' ? 'success' : 'warning';
  };

  const getServerStatusText = () => {
    if (!state.healthStatus) return 'サーバー未接続';
    return state.healthStatus.status === 'healthy' ? 'サーバー正常' : 'サーバー警告';
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1, minHeight: '100vh', bgcolor: 'background.default' }}>
        {/* アプリバー */}
        <AppBar position="static" elevation={0}>
          <Toolbar>
            <Science sx={{ mr: 2 }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              ファジィ決定木研究室選択支援システム
            </Typography>

            {/* ヘルス状態表示 */}
            <Chip
              label={getServerStatusText()}
              color={getServerStatusColor()}
              size="small"
              sx={{ mr: 2 }}
            />

            {/* アクションボタン */}
            <Tooltip title="履歴">
              <IconButton
                color="inherit"
                onClick={(e) => setHistoryMenuAnchor(e.currentTarget)}
                disabled={state.evaluationHistory.length === 0}
              >
                <History />
              </IconButton>
            </Tooltip>

            <Tooltip title="設定">
              <IconButton color="inherit" onClick={() => setSettingsDialogOpen(true)}>
                <Settings />
              </IconButton>
            </Tooltip>

            <Tooltip title="更新">
              <IconButton color="inherit" onClick={handleRefresh}>
                <Refresh />
              </IconButton>
            </Tooltip>

            <Tooltip title="情報">
              <IconButton
                color="inherit"
                onClick={() => setState(prev => ({ ...prev, showInfo: !prev.showInfo }))}
              >
                <Info />
              </IconButton>
            </Tooltip>
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl" sx={{ mt: 3, mb: 3 }}>
          {/* エラー表示 */}
          {state.error && (
            <Alert severity="error" sx={{ mb: 3 }} onClose={() => setState(prev => ({ ...prev, error: null }))}>
              {state.error}
            </Alert>
          )}

          {/* ローディング表示 */}
          {state.loading && (
            <Box sx={{ mb: 3 }}>
              <LinearProgress />
            </Box>
          )}

          {/* システム情報表示 */}
          {state.showInfo && (
            <Paper elevation={2} sx={{ p: 3, mb: 3, bgcolor: 'primary.main', color: 'white' }}>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>
                    システム状態
                  </Typography>
                  {state.healthStatus && (
                    <Box>
                      <Typography variant="body2">
                        ステータス: {state.healthStatus.status}
                      </Typography>
                      <Typography variant="body2">
                        バージョン: {state.healthStatus.version || 'Unknown'}
                      </Typography>
                      {state.healthStatus.database && (
                        <Typography variant="body2">
                          データベース: {state.healthStatus.database.lab_count} 研究室
                        </Typography>
                      )}
                    </Box>
                  )}
                </Grid>

                <Grid item xs={12} md={6}>
                  <Typography variant="h6" gutterBottom>
                    システム統計
                  </Typography>
                  {state.systemStats ? (
                    <Box>
                      <Typography variant="body2">
                        総評価数: {state.systemStats.total_evaluations}
                      </Typography>
                      <Typography variant="body2">
                        平均スコア: {state.systemStats.avg_score.toFixed(1)}
                      </Typography>
                      <Typography variant="body2">
                        履歴: {state.evaluationHistory.length} 件
                      </Typography>
                    </Box>
                  ) : (
                    <Typography variant="body2">統計データを取得中...</Typography>
                  )}
                </Grid>
              </Grid>
            </Paper>
          )}

          {/* メインコンテンツ */}
          <Grid container spacing={3}>
            <Grid item xs={12}>
              {!state.results ? (
                <>
                  {/* ヘッダーセクション */}
                  <Paper elevation={3} sx={{ p: 4, mb: 4, textAlign: 'center', bgcolor: 'primary.main', color: 'white' }}>
                    <Psychology sx={{ fontSize: 60, mb: 2 }} />
                    <Typography variant="h3" gutterBottom>
                      研究室適合度評価システム
                    </Typography>
                    <Typography variant="h6" sx={{ opacity: 0.9 }}>
                      遺伝的アルゴリズムとファジィ決定木を用いたインテリジェントマッチング
                    </Typography>
                  </Paper>

                  {/* 評価フォーム */}
                  <EvaluationForm onResults={handleResults} />
                </>
              ) : (
                <>
                  {/* 結果表示 */}
                  <ResultsList data={state.results} />

                  {/* アクションボタン */}
                  <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', mt: 4 }}>
                    <Button
                      variant="outlined"
                      onClick={() => setState(prev => ({ ...prev, results: null }))}
                      startIcon={<Refresh />}
                    >
                      新しい評価
                    </Button>

                    <Button
                      variant="outlined"
                      onClick={shareResults}
                      startIcon={<Share />}
                    >
                      結果を共有
                    </Button>

                    <Button
                      variant="outlined"
                      onClick={exportResults}
                      startIcon={<GetApp />}
                    >
                      エクスポート
                    </Button>
                  </Box>
                </>
              )}
            </Grid>
          </Grid>
        </Container>

        {/* フローティングアクションボタン */}
        {state.results && (
          <Fab
            color="primary"
            sx={{ position: 'fixed', bottom: 16, right: 16 }}
            onClick={() => setState(prev => ({ ...prev, results: null }))}
          >
            <Assessment />
          </Fab>
        )}

        {/* 履歴メニュー */}
        <Menu
          anchorEl={historyMenuAnchor}
          open={Boolean(historyMenuAnchor)}
          onClose={() => setHistoryMenuAnchor(null)}
          PaperProps={{ sx: { maxHeight: 400, width: 300 } }}
        >
          {state.evaluationHistory.length === 0 ? (
            <MenuItem disabled>履歴がありません</MenuItem>
          ) : (
            state.evaluationHistory.map((result, index) => {
              const bestLab = result.results.length > 0 ? result.results[0].lab.name : '未評価';
              const bestScore = result.results.length > 0 ? result.results[0].compatibility.overall_score : 0;

              return (
                <MenuItem key={index} onClick={() => handleHistorySelect(result)}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
                    <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
                      <Typography variant="body2">
                        {bestLab}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        適合度: {bestScore.toFixed(1)}%
                      </Typography>
                    </Box>
                  </Box>
                </MenuItem>
              );
            })
          )}
        </Menu>

        {/* 設定ダイアログ */}
        <Dialog open={settingsDialogOpen} onClose={() => setSettingsDialogOpen(false)}>
          <DialogTitle>設定</DialogTitle>
          <DialogContent>
            <Box sx={{ pt: 2 }}>
              <Typography variant="h6" gutterBottom>
                データ管理
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                評価履歴: {state.evaluationHistory.length} 件
              </Typography>
              <Button
                variant="outlined"
                color="error"
                onClick={clearHistory}
                disabled={state.evaluationHistory.length === 0}
                startIcon={<Close />}
              >
                履歴をクリア
              </Button>
            </Box>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setSettingsDialogOpen(false)}>閉じる</Button>
          </DialogActions>
        </Dialog>

        {/* スナックバー */}
        <Snackbar
          open={snackbar.open}
          autoHideDuration={4000}
          onClose={() => setSnackbar(prev => ({ ...prev, open: false }))}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
        >
          <Alert
            onClose={() => setSnackbar(prev => ({ ...prev, open: false }))}
            severity={snackbar.severity}
            variant="filled"
          >
            {snackbar.message}
          </Alert>
        </Snackbar>
      </Box>
    </ThemeProvider>
  );
};

export default App;