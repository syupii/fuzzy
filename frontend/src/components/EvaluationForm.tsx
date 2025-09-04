// src/components/EvaluationForm.tsx - 修正版
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
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
} from '@mui/material';
import {
  Science,
  Psychology,
  TrendingUp,
  ExpandMore,
  School,
  Groups,
  Explore,
  Schedule,
  Article,
  AttachMoney,
  AccountTree,
  AccessTime
} from '@mui/icons-material';
import {
  apiService,
  EvaluationPreferences,
  EvaluationResponse
} from '../services/api';

interface EvaluationFormProps {
  onResults: (results: EvaluationResponse) => void;
}

const EvaluationForm: React.FC<EvaluationFormProps> = ({ onResults }) => {
  // シンプルな初期設定（エラーを避けるため基本項目のみ）
  const [preferences, setPreferences] = useState<EvaluationPreferences>({
    research_intensity: 7.0,
    advisor_style: 6.0,
    team_work: 7.0,
    workload: 6.0,
    theory_practice: 7.0,
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 基本的な評価基準情報
  const criteriaCategories = {
    basic: {
      title: '基本的な研究環境',
      icon: <Science color="primary" />,
      criteria: {
        research_intensity: {
          label: '研究強度',
          description: '研究に集中できる環境か',
          icon: <Psychology />,
        },
        advisor_style: {
          label: '指導スタイル',
          description: '教授の指導方針との適合性',
          icon: <School />,
        },
        team_work: {
          label: 'チームワーク',
          description: '研究室内の協力体制',
          icon: <Groups />,
        },
        workload: {
          label: '作業負荷',
          description: '研究以外の負荷の程度',
          icon: <Schedule />,
        },
        theory_practice: {
          label: '理論・実践バランス',
          description: '理論研究か実践的研究か',
          icon: <TrendingUp />,
        },
      },
    },
  };

  const handleSliderChange = (field: keyof EvaluationPreferences) => (event: Event, newValue: number | number[]) => {
    setPreferences(prev => ({
      ...prev,
      [field]: Array.isArray(newValue) ? newValue[0] : newValue
    }));
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiService.evaluateCompatibility(preferences);
      console.log('Evaluation response:', response);
      onResults(response);
    } catch (err: any) {
      setError(err.response?.data?.error || '評価に失敗しました。サーバーが起動しているか確認してください。');
      console.error('Evaluation failed:', err);
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

  const resetToDefaults = () => {
    setPreferences({
      research_intensity: 7.0,
      advisor_style: 6.0,
      team_work: 7.0,
      workload: 6.0,
      theory_practice: 7.0,
    });
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
            5項目の評価基準であなたの希望を分析します（1-10スケール）
          </Typography>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {/* コントロールボタン */}
        <Box sx={{ display: 'flex', gap: 2, mb: 3, justifyContent: 'center' }}>
          <Button variant="outlined" onClick={loadDemoData} size="small">
            デモデータ読込
          </Button>
          <Button variant="outlined" onClick={resetToDefaults} size="small">
            初期値に戻す
          </Button>
        </Box>

        {/* カテゴリ別評価項目 */}
        {Object.entries(criteriaCategories).map(([categoryKey, category]) => (
          <Accordion key={categoryKey} defaultExpanded>
            <AccordionSummary
              expandIcon={<ExpandMore />}
              aria-controls={`${categoryKey}-content`}
              id={`${categoryKey}-header`}
            >
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                {category.icon}
                <Typography variant="h6">{category.title}</Typography>
                <Chip
                  label={`${Object.keys(category.criteria).length}項目`}
                  size="small"
                  color="primary"
                />
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={3}>
                {Object.entries(category.criteria).map(([key, criterion]) => (
                  <Grid item xs={12} md={6} key={key}>
                    <Paper
                      elevation={1}
                      sx={{
                        p: 3,
                        height: '100%',
                        border: '1px solid',
                        borderColor: 'divider',
                        '&:hover': {
                          borderColor: 'primary.main',
                          boxShadow: 2
                        }
                      }}
                    >
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        {criterion.icon}
                        <Typography variant="h6" sx={{ ml: 1 }}>
                          {criterion.label}
                        </Typography>
                      </Box>

                      <Typography
                        variant="body2"
                        color="text.secondary"
                        sx={{ mb: 2 }}
                      >
                        {criterion.description}
                      </Typography>

                      <Box sx={{ mt: 2 }}>
                        <Typography variant="body1" sx={{ mb: 1 }}>
                          現在の値: <strong>{preferences[key as keyof EvaluationPreferences].toFixed(1)}</strong>
                        </Typography>
                        <Slider
                          value={preferences[key as keyof EvaluationPreferences]}
                          onChange={handleSliderChange(key as keyof EvaluationPreferences)}
                          min={1}
                          max={10}
                          step={0.1}
                          valueLabelDisplay="auto"
                          sx={{
                            '& .MuiSlider-thumb': {
                              width: 20,
                              height: 20,
                            },
                            '& .MuiSlider-track': {
                              height: 6,
                            },
                            '& .MuiSlider-rail': {
                              height: 6,
                            }
                          }}
                        />
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: 'text.secondary' }}>
                          <span>1 (低い)</span>
                          <span>10 (高い)</span>
                        </Box>
                      </Box>
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            </AccordionDetails>
          </Accordion>
        ))}

        {/* 提出ボタン */}
        <Box sx={{ textAlign: 'center', mt: 4 }}>
          <Button
            variant="contained"
            size="large"
            onClick={handleSubmit}
            disabled={loading}
            startIcon={loading ? <CircularProgress size={20} /> : <Psychology />}
            sx={{ px: 6, py: 2, fontSize: '1.1rem' }}
          >
            {loading ? '評価中...' : '適合度を評価'}
          </Button>

          <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
            ファジィ論理アルゴリズムによる高精度マッチング
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default EvaluationForm;