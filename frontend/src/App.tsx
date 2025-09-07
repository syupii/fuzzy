// src/App.tsx - SystemStats型定義完全修正版
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

// システム統計情報の型定義
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

// ヘルスステータスの型定義
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

// アプリケーション状態の型定義
interface AppState {
  results: EvaluationResponse | null;
  healthStatus: HealthStatus | null;
  systemStats: SystemStats | null;
  loading: boolean;
  error: string | null;
  showInfo: boolean;
  evaluationHistory: EvaluationResponse[];
}

// スナックバー状態の型定義
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

  // アンカー状態管理
  const [anchorEls, setAnchorEls] = useState<{
    history?: HTMLElement | null;
    export?: HTMLElement | null;
  }>({});

  const [dialogOpen, setDialogOpen] = useState(false);

  useEffect(() => {
    initializeSystem();
  }, []);

  const initializeSystem = async () => {
    setState(prev => ({ ...prev, loading: true, error: null }));

    try {
      // ヘルスチェック
      try {
        const health = await apiService.getHealthStatus();
        setState(prev => ({ ...prev, healthStatus: health }));
      } catch (healthError) {
        console.warn('ヘルスチェックに失敗しました:', healthError);
      }

      // システム統計取得
      try {
        const stats = await apiService.getSystemStats();
        setState(prev => ({ ...prev, systemStats: stats }));
      } catch (statsError) {
        console.warn('システム統計の取得に失敗しました:', statsError);
      }

      // 履歴をローカルストレージから復元
      try {
        const savedHistory = localStorage.getItem('evaluation_history');
        if (savedHistory) {
          const history = JSON.parse(savedHistory);
          setState(prev => ({ ...prev, evaluationHistory: history }));
        }
      } catch (historyError) {
        console.warn('履歴の復元に失敗しました:', historyError);
      }

      setState(prev => ({ ...prev, loading: false }));
      showSnackbar('システムの初期化が完了しました', 'success');
    } catch (error) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: error instanceof Error ? error.message : 'システムの初期化に失敗しました'
      }));
      showSnackbar('システムの初期化に失敗しました', 'error');
    }
  };

  const handleResults = (results: EvaluationResponse) => {
    setState(prev => ({
      ...prev,
      results,
      evaluationHistory: [results, ...prev.evaluationHistory.slice(0, 9)] // 最新10件を保持
    }));

    // 履歴をローカルストレージに保存
    try {
      const newHistory = [results, ...state.evaluationHistory.slice(0, 9)];
      localStorage.setItem('evaluation_history', JSON.stringify(newHistory));
    } catch (error) {
      console.warn('履歴の保存に失敗しました:', error);
    }

    showSnackbar('研究室の評価が完了しました！', 'success');
  };

  const showSnackbar = (message: string, severity: SnackbarState['severity']) => {
    setSnackbar({ open: true, message, severity });
  };

  const handleSnackbarClose = () => {
    setSnackbar(prev => ({ ...prev, open: false }));
  };

  const handleMenuOpen = (type: keyof typeof anchorEls, event: React.MouseEvent<HTMLElement>) => {
    setAnchorEls(prev => ({ ...prev, [type]: event.currentTarget }));
  };

  const handleMenuClose = (type: keyof typeof anchorEls) => {
    setAnchorEls(prev => ({ ...prev, [type]: null }));
  };

  const handleHistorySelect = (result: EvaluationResponse) => {
    setState(prev => ({ ...prev, results: result }));
    handleMenuClose('history');
    showSnackbar('履歴から結果を復元しました', 'info');
  };

  const handleShare = async () => {
    if (!state.results) return;

    try {
      const bestLab = state.results.results.length > 0 ? state.results.results[0].lab_name : '未評価';
      const bestScore = state.results.results.length > 0 ? state.results.results[0].overall_score : 0;

      if (navigator.share) {
        await navigator.share({
          title: '研究室選択支援システム結果',
          text: `最適な研究室: ${bestLab} (適合度: ${bestScore.toFixed(1)}点)`,
          url: window.location.href,
        });
      } else {
        // フォールバック: クリップボードにコピー
        const shareText = `研究室選択結果\n最適な研究室: ${bestLab}\n適合度: ${bestScore.toFixed(1)}点\n${window.location.href}`;
        await navigator.clipboard.writeText(shareText);
        showSnackbar('結果をクリップボードにコピーしました', 'success');
      }
    } catch (error) {
      showSnackbar('共有に失敗しました', 'error');
    }
  };

  const handleExport = (format: 'json' | 'csv') => {
    if (!state.results) return;

    try {
      let content: string;
      let fileName: string;
      let mimeType: string;

      if (format === 'json') {
        content = JSON.stringify(state.results, null, 2);
        fileName = `研究室評価結果_${new Date().toISOString().split('T')[0]}.json`;
        mimeType = 'application/json';
      } else {
        // CSV形式
        const headers = ['順位', '研究室名', '指導教員', '総合スコア', '分野適合性'];
        const rows = state.results.results.map((lab, index) => [
          index + 1,
          lab.lab_name,
          lab.advisor,
          lab.overall_score.toFixed(1),
          lab.field_compatibility.toFixed(1)
        ]);

        content = [headers, ...rows].map(row => row.join(',')).join('\n');
        fileName = `研究室評価結果_${new Date().toISOString().split('T')[0]}.csv`;
        mimeType = 'text/csv';
      }

      const blob = new Blob([content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = fileName;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      showSnackbar(`${format.toUpperCase()}ファイルをダウンロードしました`, 'success');
    } catch (error) {
      showSnackbar('エクスポートに失敗しました', 'error');
    }

    handleMenuClose('export');
  };

  const renderHeader = () => (
    <AppBar position="static" elevation={0}>
      <Toolbar>
        <Science sx={{ mr: 2 }} />
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          研究室選択支援システム
        </Typography>

        {/* ステータス表示 */}
        <Box display="flex" alignItems="center" gap={1} mr={2}>
          {state.healthStatus && (
            <Chip
              label={state.healthStatus.status === 'healthy' ? '正常稼働' : '要確認'}
              color={state.healthStatus.status === 'healthy' ? 'success' : 'warning'}
              size="small"
            />
          )}
          {state.systemStats && (
            <Chip
              label={`評価数: ${state.systemStats.total_evaluations}`}
              color="secondary"
              size="small"
            />
          )}
        </Box>

        {/* 操作ボタン */}
        <IconButton color="inherit" onClick={() => setDialogOpen(true)}>
          <Info />
        </IconButton>

        <IconButton
          color="inherit"
          onClick={initializeSystem}
          disabled={state.loading}
        >
          <Refresh />
        </IconButton>

        {/* 履歴メニュー */}
        <IconButton
          color="inherit"
          onClick={(e) => handleMenuOpen('history', e)}
          disabled={state.evaluationHistory.length === 0}
        >
          <History />
        </IconButton>
        <Menu
          anchorEl={anchorEls.history}
          open={Boolean(anchorEls.history)}
          onClose={() => handleMenuClose('history')}
        >
          {state.evaluationHistory.length === 0 ? (
            <MenuItem disabled>
              <Typography variant="body2">履歴がありません</Typography>
            </MenuItem>
          ) : (
            state.evaluationHistory.map((result, index) => {
              const bestLab = result.results.length > 0 ? result.results[0].lab_name : '未評価';
              const bestScore = result.results.length > 0 ? result.results[0].overall_score : 0;

              return (
                <MenuItem key={index} onClick={() => handleHistorySelect(result)}>
                  <Box>
                    <Typography variant="body2" fontWeight="bold">
                      {bestLab}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      スコア: {bestScore.toFixed(1)}点
                    </Typography>
                  </Box>
                </MenuItem>
              );
            })
          )}
        </Menu>

        {/* エクスポートメニュー */}
        {state.results && (
          <>
            <IconButton
              color="inherit"
              onClick={(e) => handleMenuOpen('export', e)}
            >
              <GetApp />
            </IconButton>
            <Menu
              anchorEl={anchorEls.export}
              open={Boolean(anchorEls.export)}
              onClose={() => handleMenuClose('export')}
            >
              <MenuItem onClick={() => handleExport('json')}>
                <CloudDownload sx={{ mr: 1 }} />
                JSON形式でエクスポート
              </MenuItem>
              <MenuItem onClick={() => handleExport('csv')}>
                <Assessment sx={{ mr: 1 }} />
                CSV形式でエクスポート
              </MenuItem>
            </Menu>

            <IconButton color="inherit" onClick={handleShare}>
              <Share />
            </IconButton>
          </>
        )}
      </Toolbar>
    </AppBar>
  );

  const renderInfoDialog = () => (
    <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box display="flex" alignItems="center" gap={1}>
          <Science color="primary" />
          システム情報
        </Box>
      </DialogTitle>
      <DialogContent>
        <Grid container spacing={3}>
          {/* システム概要 */}
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              🧠 システム概要
            </Typography>
            <Typography variant="body2" paragraph>
              このシステムは、遺伝的アルゴリズムを用いたファジィ決定木により、
              あなたの価値観と研究興味に最適な研究室を見つけます。
            </Typography>
          </Grid>

          {/* 技術スタック */}
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              ⚙️ 技術スタック
            </Typography>
            <Box display="flex" flexDirection="column" gap={1}>
              <Chip icon={<Psychology />} label="ファジィ推論エンジン" variant="outlined" />
              <Chip icon={<TrendingUp />} label="遺伝的アルゴリズム" variant="outlined" />
              <Chip icon={<Science />} label="決定木アルゴリズム" variant="outlined" />
            </Box>
          </Grid>

          {/* システム状態 */}
          <Grid item xs={12} md={6}>
            <Typography variant="h6" gutterBottom>
              📊 システム状態
            </Typography>
            {state.healthStatus && (
              <Box mb={2}>
                <Typography variant="body2">
                  ステータス: {state.healthStatus.status}
                </Typography>
                <Typography variant="body2">
                  データベース: {state.healthStatus.database?.lab_count || 0}研究室
                </Typography>
                {state.systemStats && (
                  <Typography variant="body2">
                    総評価数: {state.systemStats.total_evaluations}
                  </Typography>
                )}
              </Box>
            )}
          </Grid>

          {/* 評価基準 */}
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              📋 評価基準（13項目）
            </Typography>
            <Grid container spacing={1}>
              <Grid item xs={12} sm={4}>
                <Typography variant="subtitle2" color="primary">基本項目（5項目）</Typography>
                <Typography variant="body2">研究強度、指導スタイル、チームワーク、ワークロード、理論・実践バランス</Typography>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Typography variant="subtitle2" color="secondary">拡張項目（5項目）</Typography>
                <Typography variant="body2">研究分野適合性、スキル開発、研究室雰囲気、柔軟性、論文発表機会</Typography>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Typography variant="subtitle2" color="warning.main">特殊項目（3項目）</Typography>
                <Typography variant="body2">学際性、コミュニケーション、革新性・リスク許容度</Typography>
              </Grid>
            </Grid>
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setDialogOpen(false)}>
          閉じる
        </Button>
      </DialogActions>
    </Dialog>
  );

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1 }}>
        {renderHeader()}

        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
          {state.loading && (
            <Paper sx={{ p: 3, mb: 3, textAlign: 'center' }}>
              <LinearProgress sx={{ mb: 2 }} />
              <Typography>システムを初期化中...</Typography>
            </Paper>
          )}

          {state.error && (
            <Alert severity="error" sx={{ mb: 3 }}>
              {state.error}
            </Alert>
          )}

          <Grid container spacing={4}>
            <Grid item xs={12} lg={state.results ? 6 : 12}>
              <EvaluationForm onResults={handleResults} />
            </Grid>

            {state.results && (
              <Grid item xs={12} lg={6}>
                <ResultsList data={state.results} />
              </Grid>
            )}
          </Grid>
        </Container>

        {/* フローティングアクションボタン */}
        {state.results && (
          <Fab
            color="primary"
            sx={{ position: 'fixed', bottom: 16, right: 16 }}
            onClick={() => setState(prev => ({ ...prev, results: null }))}
          >
            <Tooltip title="新しい評価を開始">
              <Refresh />
            </Tooltip>
          </Fab>
        )}

        {/* スナックバー */}
        <Snackbar
          open={snackbar.open}
          autoHideDuration={6000}
          onClose={handleSnackbarClose}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
        >
          <Alert onClose={handleSnackbarClose} severity={snackbar.severity}>
            {snackbar.message}
          </Alert>
        </Snackbar>

        {/* 情報ダイアログ */}
        {renderInfoDialog()}
      </Box>
    </ThemeProvider>
  );
};

export default App;