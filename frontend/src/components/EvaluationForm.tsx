import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Slider,
  Button,
  Grid,
  Alert,
  CircularProgress,
  Chip,
  Paper,
} from '@mui/material';
import { Science, Psychology, TrendingUp } from '@mui/icons-material';
import { apiService, EvaluationPreferences, EvaluationResponse } from '../services/api';

interface EvaluationFormProps {
  onResults: (results: EvaluationResponse) => void;
}

const EvaluationForm: React.FC<EvaluationFormProps> = ({ onResults }) => {
  const [preferences, setPreferences] = useState<EvaluationPreferences>({
    research_intensity: 7.0,
    advisor_style: 6.0,
    team_work: 7.0,
    workload: 6.0,
    theory_practice: 7.0,
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const criteriaInfo = {
    research_intensity: {
      label: '研究強度',
      description: '研究にどれだけ集中したいか',
      min: '基礎的',
      max: '最先端',
    },
    advisor_style: {
      label: '指導スタイル',
      description: '希望する指導の方法',
      min: '厳格',
      max: '自由',
    },
    team_work: {
      label: 'チームワーク',
      description: '研究での協働の度合い',
      min: '個人研究',
      max: 'チーム研究',
    },
    workload: {
      label: 'ワークロード',
      description: '研究の負荷・忙しさ',
      min: '軽め',
      max: '重め',
    },
    theory_practice: {
      label: '理論・実践バランス',
      description: '理論と実践のどちらを重視するか',
      min: '理論重視',
      max: '実践重視',
    }
  };

  const handleSliderChange = (criterion: keyof EvaluationPreferences) => (
    event: Event,
    newValue: number | number[]
  ) => {
    setPreferences({
      ...preferences,
      [criterion]: newValue as number,
    });
  };

  const handleEvaluate = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiService.evaluateCompatibility(preferences);
      onResults(response);
      
      // セッションIDを保存
      if (response.summary.session_id) {
        localStorage.setItem('fdtlss_session_id', response.summary.session_id);
      }
    } catch (err: any) {
      setError(err.response?.data?.error || '評価に失敗しました。サーバーが起動しているか確認してください。');
    } finally {
      setLoading(false);
    }
  };

  const loadDemoData = async () => {
    try {
      const response = await apiService.getDemoData();
      setPreferences(response.demo_preferences);
    } catch (err) {
      console.error('Demo data load failed:', err);
    }
  };

  const getScoreColor = (value: number): 'success' | 'warning' | 'info' | 'error' => {
    if (value >= 8) return 'success';
    if (value >= 6) return 'warning';
    if (value >= 4) return 'info';
    return 'error';
  };

  return (
    <Card elevation={3} sx={{ mb: 4 }}>
      <CardContent>
        <Box sx={{ textAlign: 'center', mb: 4 }}>
          <Science color="primary" sx={{ fontSize: 48, mb: 2 }} />
          <Typography variant="h4" component="h2" gutterBottom color="primary">
            研究室適合度評価
          </Typography>
          <Typography variant="body1" color="text.secondary">
            あなたの希望や好みを入力してください（1-10スケール）
          </Typography>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        <Grid container spacing={3}>
          {Object.entries(criteriaInfo).map(([key, info]) => (
            <Grid item xs={12} md={6} key={key}> {/* ここに 'item' を追加 */}
              <Paper elevation={1} sx={{ p: 3, height: '100%' }}>
                <Box sx={{ mb: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <Typography variant="h6" sx={{ flexGrow: 1 }}>
                      {info.label}
                    </Typography>
                    <Chip 
                      label={preferences[key as keyof EvaluationPreferences].toFixed(1)} 
                      color={getScoreColor(preferences[key as keyof EvaluationPreferences])}
                      size="small"
                    />
                  </Box>
                  
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    {info.description}
                  </Typography>
                  
                  <Slider
                    value={preferences[key as keyof EvaluationPreferences]}
                    onChange={handleSliderChange(key as keyof EvaluationPreferences)}
                    min={1}
                    max={10}
                    step={0.5}
                    marks={[
                      { value: 1, label: info.min },
                      { value: 5, label: '中間' },
                      { value: 10, label: info.max },
                    ]}
                    valueLabelDisplay="auto"
                    sx={{ mt: 2 }}
                  />
                </Box>
              </Paper>
            </Grid>
          ))}
        </Grid>

        <Box sx={{ mt: 4, display: 'flex', gap: 2, justifyContent: 'center' }}>
          <Button
            variant="contained"
            size="large"
            onClick={handleEvaluate}
            disabled={loading}
            startIcon={loading ? <CircularProgress size={20} /> : <Psychology />}
            sx={{ 
              minWidth: 200,
              py: 1.5,
              fontSize: '1.1rem'
            }}
          >
            {loading ? ' 計算中...' : ' 適合度を計算する'}
          </Button>
          
          <Button
            variant="outlined"
            size="large"
            onClick={loadDemoData}
            startIcon={<TrendingUp />}
            sx={{ minWidth: 150 }}
          >
            📊 デモデータ
          </Button>
        </Box>

        
      </CardContent>
    </Card>
  );
};

export default EvaluationForm;