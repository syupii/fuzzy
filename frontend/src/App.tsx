import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Box,
  Stepper,
  Step,
  StepLabel,
  Button,
  Paper,
  CircularProgress,
  Alert,
  Backdrop,
  StepContent,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import {
  Science,
  Code,
  Assessment,
  CheckCircle,
} from '@mui/icons-material';
import EvaluationForm from './components/EvaluationForm';
import ResultsList from './components/ResultsList';
import FieldSelectionForm from './components/FieldSelectionForm';
import TechStackSelectionForm from './components/TechStackSelectionForm';
import { 
  apiService, 
  EvaluationResponse, 
  EvaluationPreferences,
  FieldInterest,
  TechStackPreference 
} from './services/api';

// 型定義を追加
interface HealthStatus {
  status: string;
  message: string;
}

interface StepData {
  label: string;
  icon: React.ReactElement;
  description: string;
}

const App: React.FC = () => {
  // ステップ管理
  const [activeStep, setActiveStep] = useState(0);
  const [completedSteps, setCompletedSteps] = useState<Set<number>>(new Set());

  // データ状態
  const [fieldSelections, setFieldSelections] = useState<{ [fieldId: string]: FieldInterest }>({});
  const [techStackPreferences, setTechStackPreferences] = useState<TechStackPreference>({
    languagePreferences: [],
    frameworkExperience: [],
    learningWillingness: 5,
    careerGoals: [],
  });
  const [evaluationPreferences, setEvaluationPreferences] = useState<EvaluationPreferences | null>(null);
  const [results, setResults] = useState<EvaluationResponse | null>(null);

  // UI状態
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);

  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  // ステップ定義
  const steps: StepData[] = [
    {
      label: '研究分野選択',
      icon: <Science />,
      description: '興味のある研究分野を選択してください',
    },
    {
      label: '技術スタック選択',
      icon: <Code />,
      description: 'プログラミング言語や技術、キャリア目標を選択してください',
    },
    {
      label: '詳細設定',
      icon: <Assessment />,
      description: '研究室に対する詳細な希望を設定してください',
    },
    {
      label: '結果表示',
      icon: <CheckCircle />,
      description: 'あなたに最適な研究室の推薦結果をご確認ください',
    },
  ];

  // 初期化（ヘルスチェック）
  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const status = await apiService.getHealthStatus();
      setHealthStatus(status);
    } catch (error) {
      console.error('Health check failed:', error);
      setHealthStatus({ status: 'error', message: 'サーバーに接続できません。' });
    }
  };

  // ステップ進行
  const handleNext = () => {
    if (activeStep < steps.length - 1) {
      setCompletedSteps(prev => new Set(prev).add(activeStep));
      setActiveStep(prev => prev + 1);
    }
  };

  const handleBack = () => {
    if (activeStep > 0) {
      setActiveStep(prev => prev - 1);
    }
  };

  const handleStepClick = (step: number) => {
    if (completedSteps.has(step) || step <= Math.max(...Array.from(completedSteps)) + 1) {
      setActiveStep(step);
    }
  };

  // 研究分野選択の処理
  const handleFieldSelectionComplete = () => {
    const selectedCount = Object.values(fieldSelections).filter(f => f.isSelected).length;
    if (selectedCount > 0) {
      handleNext();
    } else {
      setError('最低1つの研究分野を選択してください。');
    }
  };

  const handleFieldChange = (fieldId: string, interest: FieldInterest) => {
    setFieldSelections(prev => ({
      ...prev,
      [fieldId]: interest,
    }));
    setError(null);
  };

  // 技術スタック選択の処理
  const handleTechStackComplete = () => {
    const hasLanguages = techStackPreferences.languagePreferences.length > 0;
    const hasCareerGoals = techStackPreferences.careerGoals.length > 0;
    
    if (hasLanguages || hasCareerGoals) {
      handleNext();
    } else {
      setError('プログラミング言語またはキャリア目標を最低1つ選択してください。');
    }
  };

  const handleTechStackChange = (preferences: TechStackPreference) => {
    setTechStackPreferences(preferences);
    setError(null);
  };

  // 詳細評価の処理
  const handleEvaluationComplete = async (preferences: EvaluationPreferences) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // 拡張された評価を実行
      const response = await apiService.evaluateWithTechStack(
        preferences,
        fieldSelections,
        techStackPreferences
      );
      
      setResults(response);
      setEvaluationPreferences(preferences);
      handleNext();
    } catch (error) {
      console.error('Evaluation failed:', error);
      setError('評価処理中にエラーが発生しました。もう一度お試しください。');
    } finally {
      setIsLoading(false);
    }
  };

  // やり直し処理
  const handleRestart = () => {
    setActiveStep(0);
    setCompletedSteps(new Set());
    setFieldSelections({});
    setTechStackPreferences({
      languagePreferences: [],
      frameworkExperience: [],
      learningWillingness: 5,
      careerGoals: [],
    });
    setEvaluationPreferences(null);
    setResults(null);
    setError(null);
  };

  // ステップコンテンツの描画
  const renderStepContent = (step: number) => {
    switch (step) {
      case 0:
        return (
          <FieldSelectionForm
            selectedFields={fieldSelections}
            onFieldChange={handleFieldChange}
            onSubmit={handleFieldSelectionComplete}
          />
        );
      case 1:
        return (
          <TechStackSelectionForm
            preferences={techStackPreferences}
            onPreferencesChange={handleTechStackChange}
            onSubmit={handleTechStackComplete}
          />
        );
      case 2:
        return (
          <EvaluationForm
            onResults={handleEvaluationComplete}
            fieldSelections={fieldSelections}
            techStackPreferences={techStackPreferences}
          />
        );
      case 3:
        return results ? (
          <ResultsList data={results} />
        ) : (
          <Box sx={{ textAlign: 'center', py: 4 }}>
            <Typography variant="h6" gutterBottom>
              結果がありません
            </Typography>
            <Button variant="contained" onClick={handleRestart}>
              最初からやり直す
            </Button>
          </Box>
        );
      default:
        return null;
    }
  };

  // ローディング画面
  if (isLoading) {
    return (
      <Backdrop open sx={{ color: '#fff', zIndex: (theme) => theme.zIndex.drawer + 1 }}>
        <Box sx={{ textAlign: 'center' }}>
          <CircularProgress color="inherit" size={60} />
          <Typography variant="h6" sx={{ mt: 2 }}>
            研究室を分析中...
          </Typography>
          <Typography variant="body2" sx={{ mt: 1, opacity: 0.8 }}>
            あなたの選択内容に基づいて最適な研究室を探しています
          </Typography>
        </Box>
      </Backdrop>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* ヘッダー */}
      <Paper sx={{ p: 4, mb: 4, textAlign: 'center', bgcolor: 'primary.main', color: 'white' }}>
        <Typography variant="h3" component="h1" gutterBottom>
          🎓 研究室選択支援システム
        </Typography>
        <Typography variant="h6" sx={{ opacity: 0.9 }}>
          北海道情報大学 - AI駆動型マッチングシステム
        </Typography>
      </Paper>

      {/* ヘルスステータス */}
      {healthStatus && healthStatus.status === 'error' && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          {healthStatus.message} 一部機能が制限される可能性があります。
        </Alert>
      )}

      {/* エラー表示 */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* ステッパー */}
      <Paper sx={{ p: 3, mb: 4 }}>
        <Stepper 
          activeStep={activeStep} 
          orientation={isMobile ? 'vertical' : 'horizontal'}
          sx={{ mb: 3 }}
        >
          {steps.map((step, index) => (
            <Step key={step.label} completed={completedSteps.has(index)}>
              <StepLabel 
                StepIconComponent={({ completed, active }) => (
                  <Box
                    sx={{
                      width: 40,
                      height: 40,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      borderRadius: '50%',
                      bgcolor: completed ? 'success.main' : active ? 'primary.main' : 'grey.300',
                      color: 'white',
                      cursor: completedSteps.has(index) || index <= Math.max(...Array.from(completedSteps)) + 1 ? 'pointer' : 'default',
                    }}
                    onClick={() => handleStepClick(index)}
                  >
                    {React.cloneElement(step.icon, { fontSize: 'small' })}
                  </Box>
                )}
                onClick={() => handleStepClick(index)}
                sx={{ cursor: 'pointer' }}
              >
                <Typography variant="subtitle1" fontWeight={activeStep === index ? 'bold' : 'normal'}>
                  {step.label}
                </Typography>
                {!isMobile && (
                  <Typography variant="body2" color="text.secondary">
                    {step.description}
                  </Typography>
                )}
              </StepLabel>
              {isMobile && activeStep === index && (
                <StepContent>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    {step.description}
                  </Typography>
                </StepContent>
              )}
            </Step>
          ))}
        </Stepper>

        {/* 進行状況インジケーター */}
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <Typography variant="body2" color="text.secondary">
            進行状況: {Math.round((completedSteps.size / steps.length) * 100)}%
          </Typography>
          <Box sx={{ flexGrow: 1, height: 4, bgcolor: 'grey.200', borderRadius: 2 }}>
            <Box
              sx={{
                width: `${(completedSteps.size / steps.length) * 100}%`,
                height: '100%',
                bgcolor: 'primary.main',
                borderRadius: 2,
                transition: 'width 0.3s ease',
              }}
            />
          </Box>
        </Box>

        {/* ナビゲーションボタン */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
          <Button
            onClick={handleBack}
            disabled={activeStep === 0}
            variant="outlined"
          >
            戻る
          </Button>
          
          <Box sx={{ display: 'flex', gap: 1 }}>
            {activeStep === steps.length - 1 && (
              <Button
                onClick={handleRestart}
                variant="outlined"
                color="secondary"
              >
                やり直し
              </Button>
            )}
            
            {activeStep < steps.length - 1 && completedSteps.has(activeStep) && (
              <Button
                onClick={handleNext}
                variant="contained"
              >
                次へ
              </Button>
            )}
          </Box>
        </Box>
      </Paper>

      {/* メインコンテンツ */}
      <Paper sx={{ minHeight: 600 }}>
        {renderStepContent(activeStep)}
      </Paper>

      {/* 選択サマリー（デバッグ用 - 開発環境でのみ表示） */}
      {process.env.NODE_ENV === 'development' && (
        <Paper sx={{ p: 2, mt: 4, bgcolor: 'grey.50' }}>
          <Typography variant="h6" gutterBottom>デバッグ情報</Typography>
          <Typography variant="body2" component="div">
            <strong>選択された研究分野:</strong>{' '}
            {Object.entries(fieldSelections)
              .filter(([, interest]) => interest.isSelected)
              .map(([fieldId]) => fieldId)
              .join(', ') || 'なし'}
          </Typography>
          <Typography variant="body2" component="div">
            <strong>選択されたプログラミング言語:</strong>{' '}
            {techStackPreferences.languagePreferences.join(', ') || 'なし'}
          </Typography>
          <Typography variant="body2" component="div">
            <strong>経験のある技術:</strong>{' '}
            {techStackPreferences.frameworkExperience.join(', ') || 'なし'}
          </Typography>
          <Typography variant="body2" component="div">
            <strong>学習意欲:</strong> {techStackPreferences.learningWillingness}/10
          </Typography>
          <Typography variant="body2" component="div">
            <strong>キャリア目標:</strong>{' '}
            {techStackPreferences.careerGoals.join(', ') || 'なし'}
          </Typography>
        </Paper>
      )}

      {/* フッター */}
      <Box sx={{ textAlign: 'center', mt: 6, py: 3, borderTop: '1px solid #e0e0e0' }}>
        <Typography variant="body2" color="text.secondary">
          © 2025 北海道情報大学 研究室選択支援システム
        </Typography>
        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
          遺伝的アルゴリズム × ファジィ論理 × AI による高精度マッチング
        </Typography>
      </Box>
    </Container>
  );
};

export default App;