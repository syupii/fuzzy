// frontend/src/App.tsx - エラー修正版
import React, { useState } from 'react';
import {
  Container,
  Box,
  Typography,
  Stepper,
  Step,
  StepLabel,
  Button,
  Alert,
  Paper,
  Grid,
  AppBar,
  Toolbar,
  IconButton,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  Badge,
  Fab,
  Snackbar
} from '@mui/material';
import {
  Menu as MenuIcon,
  Science,
  Star,
  Assessment,
  Settings,
  Help,
  Refresh,
  GetApp,
  Share
} from '@mui/icons-material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

// コンポーネントのインポート
import EvaluationForm from './components/EvaluationForm';
import ResultsList from './components/ResultsList'; // PriorityAwareResultsの代わりにResultsListを使用

// 型定義
interface EvaluationResponse {
  lab_results: any[];
  summary: any;
  metadata?: any;
}

// テーマ設定
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#dc004e',
      light: '#ff5983',
      dark: '#9a0036',
    },
    success: {
      main: '#2e7d32',
    },
    warning: {
      main: '#f57c00',
    },
    info: {
      main: '#0288d1',
    },
  },
  typography: {
    fontFamily: [
      '-apple-system',
      'BlinkMacSystemFont',
      '"Segoe UI"',
      'Roboto',
      '"Helvetica Neue"',
      'Arial',
      'sans-serif',
    ].join(','),
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 600,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 2px 12px rgba(0,0,0,0.08)',
        },
      },
    },
  },
});

// メインアプリケーションコンポーネント
const App: React.FC = () => {
  // ステート管理
  const [activeStep, setActiveStep] = useState(0);
  const [evaluationResults, setEvaluationResults] = useState<EvaluationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  // ステップ定義
  const steps = [
    {
      label: '評価基準設定',
      description: '12項目の評価基準と優先度を設定',
      icon: <Star />
    },
    {
      label: '研究分野選択',
      description: '18分野から興味のある分野を選択',
      icon: <Science />
    },
    {
      label: '結果確認',
      description: 'AI統合評価による適合度結果',
      icon: <Assessment />
    }
  ];

  // 評価結果ハンドラー
  const handleEvaluationResults = (response: EvaluationResponse) => {
    setEvaluationResults(response);
    setActiveStep(2);
    setSnackbarMessage('評価が完了しました！');
    setSnackbarOpen(true);
  };

  // エラーハンドラー
  const handleError = (errorMessage: string) => {
    setError(errorMessage);
    setSnackbarMessage(`エラー: ${errorMessage}`);
    setSnackbarOpen(true);
  };

  // 再評価
  const handleReEvaluate = () => {
    setEvaluationResults(null);
    setError('');
    setActiveStep(0);
    setSnackbarMessage('新しい評価を開始してください');
    setSnackbarOpen(true);
  };

  // 結果エクスポート
  const handleExportResults = () => {
    if (!evaluationResults) return;

    const dataStr = JSON.stringify(evaluationResults, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);

    const exportFileDefaultName = `research_lab_evaluation_${new Date().toISOString().slice(0, 10)}.json`;

    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', exportFileDefaultName);
    linkElement.click();

    setSnackbarMessage('結果をエクスポートしました');
    setSnackbarOpen(true);
  };

  // 結果共有
  const handleShareResults = async () => {
    if (!evaluationResults) return;

    if (navigator.share) {
      try {
        await navigator.share({
          title: '研究室適合度評価結果',
          text: `優先度対応AI評価により ${evaluationResults.lab_results.length}件の研究室を評価しました。`,
          url: window.location.href,
        });
        setSnackbarMessage('結果を共有しました');
        setSnackbarOpen(true);
      } catch (error) {
        console.error('共有エラー:', error);
      }
    } else {
      // Web Share APIが利用できない場合はクリップボードにコピー
      const shareText = `研究室適合度評価結果\n評価対象: ${evaluationResults.lab_results.length}件\n最高適合度: ${(evaluationResults.summary.max_score * 100).toFixed(1)}%\n優先度対応AI統合評価システム使用`;

      try {
        await navigator.clipboard.writeText(shareText);
        setSnackbarMessage('結果をクリップボードにコピーしました');
        setSnackbarOpen(true);
      } catch (error) {
        console.error('クリップボードエラー:', error);
      }
    }
  };

  // サイドドロワー
  const renderDrawer = () => (
    <Drawer
      anchor="left"
      open={drawerOpen}
      onClose={() => setDrawerOpen(false)}
    >
      <Box sx={{ width: 250 }} role="presentation">
        <Box sx={{ p: 2 }}>
          <Typography variant="h6" color="primary">
            🧬 研究室選択支援システム
          </Typography>
          <Typography variant="caption" color="text.secondary">
            v5.0.0 - 優先度対応版
          </Typography>
        </Box>
        <Divider />

        <List>
          <ListItem button onClick={() => setActiveStep(0)}>
            <ListItemIcon><Star /></ListItemIcon>
            <ListItemText
              primary="評価基準設定"
              secondary="12項目 + 優先度"
            />
          </ListItem>

          <ListItem button onClick={() => setActiveStep(1)}>
            <ListItemIcon><Science /></ListItemIcon>
            <ListItemText
              primary="研究分野選択"
              secondary="18分野対応"
            />
          </ListItem>

          <ListItem button onClick={() => setActiveStep(2)} disabled={!evaluationResults}>
            <ListItemIcon>
              <Badge badgeContent={evaluationResults?.lab_results?.length || 0} color="primary">
                <Assessment />
              </Badge>
            </ListItemIcon>
            <ListItemText
              primary="評価結果"
              secondary="AI統合評価"
            />
          </ListItem>
        </List>

        <Divider />

        <List>
          <ListItem button onClick={handleReEvaluate}>
            <ListItemIcon><Refresh /></ListItemIcon>
            <ListItemText primary="新しい評価" />
          </ListItem>

          {evaluationResults && (
            <>
              <ListItem button onClick={handleExportResults}>
                <ListItemIcon><GetApp /></ListItemIcon>
                <ListItemText primary="結果をエクスポート" />
              </ListItem>

              <ListItem button onClick={handleShareResults}>
                <ListItemIcon><Share /></ListItemIcon>
                <ListItemText primary="結果を共有" />
              </ListItem>
            </>
          )}
        </List>
      </Box>
    </Drawer>
  );

  // メインコンテンツ
  const renderMainContent = () => {
    switch (activeStep) {
      case 0:
      case 1:
        return (
          <EvaluationForm
            onResults={handleEvaluationResults}
            onError={handleError}
          />
        );
      case 2:
        if (!evaluationResults) {
          return (
            <Box textAlign="center" sx={{ py: 8 }}>
              <Assessment sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h5" gutterBottom>
                評価結果がありません
              </Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                まず評価基準を設定して評価を実行してください
              </Typography>
              <Button
                variant="contained"
                onClick={() => setActiveStep(0)}
                startIcon={<Star />}
              >
                評価を開始する
              </Button>
            </Box>
          );
        }


        return (
          <Box>
            {/* 評価結果サマリー */}
            <Paper sx={{ p: 3, mb: 3, backgroundColor: 'primary.main', color: 'white' }}>
              <Typography variant="h6" gutterBottom>
                <Assessment sx={{ mr: 1, verticalAlign: 'middle' }} />
                評価結果サマリー
              </Typography>

              <Grid container spacing={3}>
                <Grid item xs={3}>
                  <Box textAlign="center">
                    <Typography variant="h4">{evaluationResults.lab_results.length}</Typography>
                    <Typography variant="body2">評価対象研究室</Typography>
                  </Box>
                </Grid>
                <Grid item xs={3}>
                  <Box textAlign="center">
                    <Typography variant="h4">{evaluationResults.summary?.high_compatibility_count || 0}</Typography>
                    <Typography variant="body2">高適合研究室</Typography>
                  </Box>
                </Grid>
                <Grid item xs={3}>
                  <Box textAlign="center">
                    <Typography variant="h4">{((evaluationResults.summary?.max_score || 0) * 100).toFixed(1)}%</Typography>
                    <Typography variant="body2">最高適合度</Typography>
                  </Box>
                </Grid>
                <Grid item xs={3}>
                  <Box textAlign="center">
                    <Typography variant="h4">{((evaluationResults.summary?.avg_score || 0) * 100).toFixed(1)}%</Typography>
                    <Typography variant="body2">平均適合度</Typography>
                  </Box>
                </Grid>
              </Grid>

              {evaluationResults.summary?.priority_weighting_applied && (
                <Alert severity="info" sx={{ mt: 2 }}>
                  <Typography variant="body2">
                    優先度による重み付け評価を実行しました。
                    設定された優先度に基づいてスコアが調整されています。
                  </Typography>
                </Alert>
              )}
            </Paper>

            {/* 研究室結果一覧 */}
            <ResultsList
              results={evaluationResults.lab_results}
              metadata={evaluationResults.metadata}
              onLabSelected={(lab) => console.log('Selected lab:', lab)}
            />
          </Box>
        );
      default:
        return <div>未知のステップです</div>;
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1 }}>
        {/* アプリバー */}
        <AppBar position="static" elevation={1}>
          <Toolbar>
            <IconButton
              size="large"
              edge="start"
              color="inherit"
              aria-label="menu"
              onClick={() => setDrawerOpen(true)}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>

            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              研究室選択支援システム - 優先度対応AI統合版
            </Typography>

            {evaluationResults && (
              <Badge
                badgeContent={evaluationResults.summary?.high_compatibility_count || 0}
                color="success"
                sx={{ mr: 2 }}
              >
                <Assessment />
              </Badge>
            )}

            <IconButton color="inherit" onClick={() => setActiveStep(0)}>
              <Settings />
            </IconButton>
          </Toolbar>
        </AppBar>

        {/* サイドドロワー */}
        {renderDrawer()}

        {/* メインコンテンツ */}
        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
          {/* システム紹介 */}
          <Paper sx={{ p: 3, mb: 4, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
            <Typography variant="h4" gutterBottom>
              🧬🌳⭐🤖 AI統合研究室マッチングシステム
            </Typography>
            <Typography variant="h6" sx={{ opacity: 0.9, mb: 2 }}>
              遺伝的アルゴリズム × ファジィ推論 × 決定木 × 優先度対応
            </Typography>
            <Typography variant="body1" sx={{ opacity: 0.8 }}>
              12項目の評価基準に優先度を設定し、18分野の研究分野から最適な研究室を見つけます。
              最新のAI技術により、あなたの希望に最も適合する研究室を高精度で推薦します。
            </Typography>
          </Paper>

          {/* エラー表示 */}
          {error && (
            <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError('')}>
              {error}
            </Alert>
          )}

          {/* ステッパー */}
          <Paper sx={{ p: 2, mb: 4 }}>
            <Stepper activeStep={activeStep} alternativeLabel>
              {steps.map((step, index) => (
                <Step key={step.label}>
                  <StepLabel
                    StepIconComponent={() => (
                      <Box
                        sx={{
                          width: 40,
                          height: 40,
                          borderRadius: '50%',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          backgroundColor: index <= activeStep ? 'primary.main' : 'grey.300',
                          color: 'white',
                          transition: 'all 0.3s ease'
                        }}
                      >
                        {step.icon}
                      </Box>
                    )}
                  >
                    <Typography variant="subtitle1">{step.label}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      {step.description}
                    </Typography>
                  </StepLabel>
                </Step>
              ))}
            </Stepper>
          </Paper>

          {/* メインコンテンツエリア */}
          <Paper sx={{ p: 3, minHeight: '60vh' }}>
            {renderMainContent()}
          </Paper>

          {/* 機能紹介 */}
          {activeStep === 0 && (
            <Paper sx={{ p: 3, mt: 4, backgroundColor: 'grey.50' }}>
              <Typography variant="h6" gutterBottom color="primary">
                🎯 新機能：優先度設定
              </Typography>
              <Typography variant="body2" paragraph>
                各評価基準（12項目）について、1-10の段階で重要度（優先度）を設定できます。
                設定した優先度により、マッチングスコアが重み付けされ、より精度の高い評価が可能です。
              </Typography>

              <Typography variant="h6" gutterBottom color="primary" sx={{ mt: 2 }}>
                🤖 AI統合評価エンジン
              </Typography>
              <Typography variant="body2">
                • <strong>ファジィ推論</strong>：曖昧な評価基準を自然言語的に処理<br />
                • <strong>遺伝的アルゴリズム</strong>：最適解を進化的に探索<br />
                • <strong>決定木</strong>：論理的な判定ルールを適用<br />
                • <strong>優先度統合</strong>：個人の重視項目を反映した総合評価
              </Typography>
            </Paper>
          )}
        </Container>

        {/* フローティングアクションボタン */}
        {evaluationResults && (
          <>
            <Fab
              color="primary"
              aria-label="export"
              onClick={handleExportResults}
              sx={{ position: 'fixed', bottom: 16, right: 80 }}
            >
              <GetApp />
            </Fab>

            <Fab
              color="secondary"
              aria-label="share"
              onClick={handleShareResults}
              sx={{ position: 'fixed', bottom: 16, right: 16 }}
            >
              <Share />
            </Fab>
          </>
        )}

        {/* スナックバー */}
        <Snackbar
          open={snackbarOpen}
          autoHideDuration={4000}
          onClose={() => setSnackbarOpen(false)}
          message={snackbarMessage}
        />
      </Box>
    </ThemeProvider>
  );
};

export default App;