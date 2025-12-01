// frontend/src/App.tsx
// studentProfileをResultsListに渡す版

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
  CircularProgress,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Science,
  Star,
  Assessment,
  Settings,
  Help,
  Refresh,
} from '@mui/icons-material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

// コンポーネントのインポート
import EvaluationForm from './components/EvaluationForm';
import ResultsList from './components/ResultsList';

// 型定義
interface EvaluationResponse {
  evaluation_results?: any[];
  lab_results?: any[];
  results?: any[];
  summary?: any;
  metadata?: any;
  system_info?: any;
  total_labs_evaluated?: number;
  student_profile?: any; // ★★★ 学生プロファイルを追加 ★★★
}

// テーマ設定
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
});

const App: React.FC = () => {
  const [activeStep, setActiveStep] = useState(0);
  const [evaluationResults, setEvaluationResults] = useState<EvaluationResponse | null>(null);
  const [studentProfile, setStudentProfile] = useState<any>(null); // ★★★ 学生プロファイルを保持 ★★★
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');
  const [drawerOpen, setDrawerOpen] = useState(false);

  const steps = [
    { label: '評価基準設定', icon: <Star /> },
    { label: '研究分野選択', icon: <Science /> },
    { label: '結果確認', icon: <Assessment /> }
  ];

  const handleEvaluationResults = (response: EvaluationResponse) => {
    const normalizedResponse = {
      ...response,
      evaluation_results: response.evaluation_results || response.lab_results || response.results || [],
      summary: response.summary || {
        total_labs: response.total_labs_evaluated || 0,
        avg_score: 0,
        high_compatibility_count: 0,
      }
    };
    setEvaluationResults(normalizedResponse);

    // ★★★ 学生プロファイルを保存 ★★★
    if (response.student_profile) {
      setStudentProfile(response.student_profile);
    }

    setActiveStep(2);
    setError('');
  };

  const handleError = (errorMessage: string) => {
    setError(errorMessage);
    setLoading(false);
  };

  const handleReset = () => {
    setActiveStep(0);
    setEvaluationResults(null);
    setStudentProfile(null); // ★★★ リセット時にクリア ★★★
    setError('');
  };

  const getResults = () => {
    if (!evaluationResults) return [];
    return evaluationResults.evaluation_results || [];
  };

  const renderStepContent = (step: number) => {
    if (loading) {
      return (
        <Box textAlign="center" py={8}>
          <CircularProgress size={60} />
          <Typography variant="h6" sx={{ mt: 2 }}>評価処理中...</Typography>
        </Box>
      );
    }

    if (error) {
      return <Alert severity="error" sx={{ mb: 3 }}>{error}</Alert>;
    }

    switch (step) {
      case 0:
      case 1:
        return (
          <EvaluationForm
            onResults={handleEvaluationResults}
            onError={handleError}
          />
        );

      case 2:
        const results = getResults();

        if (!evaluationResults || results.length === 0) {
          return (
            <Box textAlign="center" py={8}>
              <Assessment sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h5" gutterBottom>評価結果がありません</Typography>
              <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
                まず評価基準を設定して評価を実行してください
              </Typography>
              <Button variant="contained" onClick={() => setActiveStep(0)} startIcon={<Star />}>
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
                <Grid item xs={12} sm={3}>
                  <Box textAlign="center">
                    <Typography variant="h4">{results.length}</Typography>
                    <Typography variant="body2">評価対象研究室</Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={3}>
                  <Box textAlign="center">
                    <Typography variant="h4">
                      {results.filter((r: any) => (r.overall_compatibility || 0) >= 0.7).length}
                    </Typography>
                    <Typography variant="body2">高適合研究室</Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={3}>
                  <Box textAlign="center">
                    <Typography variant="h4">
                      {results.length > 0
                        ? (results.reduce((sum: number, r: any) => sum + (r.overall_compatibility || 0), 0) / results.length * 100).toFixed(1) + '%'
                        : '0.0%'}
                    </Typography>
                    <Typography variant="body2">平均適合度</Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={3}>
                  <Box textAlign="center">
                    <Typography variant="h4">
                      {evaluationResults?.system_info?.processing_time?.toFixed(2) || '0.00'}s
                    </Typography>
                    <Typography variant="body2">処理時間</Typography>
                  </Box>
                </Grid>
              </Grid>
            </Paper>

            {/* ★★★ ResultsListにstudentProfileを渡す ★★★ */}
            <ResultsList
              results={results}
              metadata={evaluationResults.system_info || evaluationResults.metadata}
              studentProfile={studentProfile}
            />

            {/* アクションボタン */}
            <Box textAlign="center" mt={4}>
              <Button variant="outlined" onClick={handleReset} startIcon={<Refresh />} size="large">
                新しい評価を開始
              </Button>
            </Box>
          </Box>
        );

      default:
        return null;
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1 }}>
        <AppBar position="static" elevation={0}>
          <Toolbar>
            <IconButton edge="start" color="inherit" aria-label="menu" onClick={() => setDrawerOpen(true)} sx={{ mr: 2 }}>
              <MenuIcon />
            </IconButton>
            <Science sx={{ mr: 2 }} />
            <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
              研究室マッチングシステム
            </Typography>
          </Toolbar>
        </AppBar>

        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
          <Paper sx={{ p: 3, mb: 4 }}>
            <Stepper activeStep={activeStep} alternativeLabel>
              {steps.map((step, index) => (
                <Step key={index}>
                  <StepLabel icon={step.icon}>{step.label}</StepLabel>
                </Step>
              ))}
            </Stepper>
          </Paper>
          {renderStepContent(activeStep)}
        </Container>

        <Drawer anchor="left" open={drawerOpen} onClose={() => setDrawerOpen(false)}>
          <Box sx={{ width: 250 }} role="presentation">
            <List>
              <ListItem button onClick={() => { setActiveStep(0); setDrawerOpen(false); }}>
                <ListItemIcon><Star /></ListItemIcon>
                <ListItemText primary="評価開始" />
              </ListItem>
              <ListItem button onClick={() => { setActiveStep(2); setDrawerOpen(false); }}>
                <ListItemIcon><Assessment /></ListItemIcon>
                <ListItemText primary="結果表示" />
              </ListItem>
              <Divider />
              <ListItem button><ListItemIcon><Settings /></ListItemIcon><ListItemText primary="設定" /></ListItem>
              <ListItem button><ListItemIcon><Help /></ListItemIcon><ListItemText primary="ヘルプ" /></ListItem>
            </List>
          </Box>
        </Drawer>
      </Box>
    </ThemeProvider>
  );
};

export default App;