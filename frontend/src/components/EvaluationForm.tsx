// frontend/src/components/EvaluationForm.tsx - シンプル版
import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Grid,
  Slider,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
  LinearProgress,
  Tabs,
  Tab
} from '@mui/material';
import {
  ExpandMore,
  Science,
  Palette,
  SportsEsports,
  Psychology,
  School,
  Timeline
} from '@mui/icons-material';
import {
  EvaluationPreferences,
  EvaluationResponse,
  ResearchFieldInterests,
  StudentProfile,
  RESEARCH_FIELDS,
  FIELD_CATEGORIES,
  CRITERIA_INFO,
  fieldUtils,
  apiService
} from '../services/api';

interface EvaluationFormProps {
  onResults: (response: EvaluationResponse) => void;
  onError: (error: string) => void;
}

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
      {...other}
    >
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const EvaluationForm: React.FC<EvaluationFormProps> = ({ onResults, onError }) => {
  const [tabValue, setTabValue] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>('');

  // 評価基準の状態
  const [preferences, setPreferences] = useState<EvaluationPreferences>({
    research_intensity: 5,
    advisor_style: 5,
    team_work: 5,
    workload: 5,
    theory_practice: 5,
    research_field_match: 5,
    skill_development: 5,
    lab_atmosphere: 5,
    flexibility: 5,
    publication_opportunity: 5,
    interdisciplinary: 5,
    communication_style: 5,
    innovation_risk: 5,
  });

  // 研究分野の興味の状態
  const [fieldInterests, setFieldInterests] = useState<ResearchFieldInterests>({
    'ai_machine_learning': 5,
    'image_video_processing': 5,
    'network_security': 5,
    'database_systems': 5,
    'embedded_iot': 5,
    'web_ui_ux': 5,
    'design_visual': 5,
    'video_animation': 5,
    'computer_music': 5,
    'game_esports': 5,
    'vr_ar_media_art': 5,
  });

  // 評価基準変更ハンドラー
  const handlePreferenceChange = (criterion: keyof EvaluationPreferences, value: number) => {
    setPreferences(prev => ({
      ...prev,
      [criterion]: value
    }));
  };

  // 研究分野興味変更ハンドラー
  const handleFieldInterestChange = (field: string, value: number) => {
    setFieldInterests(prev => ({
      ...prev,
      [field]: value
    }));
  };

  // 評価実行
  const handleEvaluate = async () => {
    try {
      setIsLoading(true);
      setError('');

      const response = await apiService.evaluateLabs(preferences);
      onResults(response);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '評価処理でエラーが発生しました';
      setError(errorMessage);
      onError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  // デモデータ読み込み
  const handleLoadDemo = async () => {
    try {
      const demoProfile = await apiService.getDemoProfile();
      setPreferences(demoProfile.evaluation_criteria);

      // デモの研究分野興味を設定
      const demoFieldInterests: ResearchFieldInterests = {};
      Object.keys(demoProfile.field_interests).forEach(field => {
        demoFieldInterests[field] = demoProfile.field_interests[field];
      });
      setFieldInterests(prev => ({ ...prev, ...demoFieldInterests }));

      setTabValue(0);
    } catch (err) {
      setError('デモデータの読み込みに失敗しました');
    }
  };

  // 基本評価基準コンポーネント
  const renderBasicCriteria = () => (
    <Box>
      <Typography variant="h6" gutterBottom>
        基本評価基準
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        これらの項目はマッチング精度に大きく影響します
      </Typography>

      <Grid container spacing={3}>
        {Object.entries(CRITERIA_INFO).map(([key, info]) => (
          <Grid item xs={12} md={6} key={key}>
            <Card variant="outlined" sx={{ p: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                {info.name}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                {info.description}
              </Typography>
              <Box sx={{ px: 2 }}>
                <Slider
                  value={preferences[key as keyof EvaluationPreferences]}
                  onChange={(_, value) => handlePreferenceChange(key as keyof EvaluationPreferences, value as number)}
                  min={1}
                  max={10}
                  step={1}
                  marks
                  valueLabelDisplay="on"
                />
              </Box>
              <Typography variant="caption" color="text.secondary">
                {info.range}
              </Typography>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );

  // 研究分野興味コンポーネント
  const renderFieldInterests = () => (
    <Box>
      <Typography variant="h6" gutterBottom>
        研究分野への興味度設定
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        各研究分野への興味レベルを1-10で設定してください
      </Typography>

      {FIELD_CATEGORIES.map((category) => {
        const fields = fieldUtils.getFieldsByCategory(category);
        return (
          <Accordion key={category} defaultExpanded={category === 'テクノロジー・システム'}>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6">
                {category === 'テクノロジー・システム' && <Science sx={{ mr: 1 }} />}
                {category === 'クリエイティブ' && <Palette sx={{ mr: 1 }} />}
                {category === 'エンターテイメント' && <SportsEsports sx={{ mr: 1 }} />}
                {category} ({fields.length}分野)
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                {fields.map((field) => (
                  <Grid item xs={12} md={6} key={field.id}>
                    <Box sx={{ p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
                      <Typography variant="subtitle1" gutterBottom>
                        {field.name}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                        {field.description}
                      </Typography>
                      <Chip
                        label={`教員数: ${field.faculty_count}名`}
                        size="small"
                        sx={{ mb: 2 }}
                      />
                      <Box sx={{ px: 2 }}>
                        <Slider
                          value={fieldInterests[field.id] || 5}
                          onChange={(_, value) => handleFieldInterestChange(field.id, value as number)}
                          min={1}
                          max={10}
                          step={1}
                          marks
                          valueLabelDisplay="on"
                        />
                      </Box>
                      <Typography variant="caption" color="text.secondary">
                        1(興味なし) ～ 10(非常に興味あり)
                      </Typography>
                    </Box>
                  </Grid>
                ))}
              </Grid>
            </AccordionDetails>
          </Accordion>
        );
      })}
    </Box>
  );

  // 評価実行コンポーネント
  const renderEvaluationExecute = () => (
    <Box>
      <Typography variant="h6" gutterBottom>
        研究室マッチング実行
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        設定した条件に基づいて研究室マッチングを実行します
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Card sx={{ p: 3, mb: 3 }}>
        <Typography variant="subtitle1" gutterBottom>
          設定サマリー
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Typography variant="body2">
              評価基準: {Object.keys(CRITERIA_INFO).length}項目設定済み
            </Typography>
          </Grid>
          <Grid item xs={6}>
            <Typography variant="body2">
              研究分野: {RESEARCH_FIELDS.length}分野設定済み
            </Typography>
          </Grid>
        </Grid>
      </Card>

      <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
        <Button
          variant="outlined"
          onClick={handleLoadDemo}
          startIcon={<School />}
        >
          デモデータ読み込み
        </Button>
        <Button
          variant="contained"
          onClick={handleEvaluate}
          disabled={isLoading}
          startIcon={<Timeline />}
          size="large"
        >
          {isLoading ? '評価中...' : '研究室マッチング実行'}
        </Button>
      </Box>

      {isLoading && (
        <Box sx={{ mt: 2 }}>
          <LinearProgress />
          <Typography variant="body2" textAlign="center" sx={{ mt: 1 }}>
            AIによる研究室適合性を評価中...
          </Typography>
        </Box>
      )}
    </Box>
  );

  return (
    <Box sx={{ width: '100%' }}>
      <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)}>
        <Tab label="基本設定" />
        <Tab label="研究分野" />
        <Tab label="評価実行" />
      </Tabs>

      <TabPanel value={tabValue} index={0}>
        {renderBasicCriteria()}
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        {renderFieldInterests()}
      </TabPanel>

      <TabPanel value={tabValue} index={2}>
        {renderEvaluationExecute()}
      </TabPanel>
    </Box>
  );
};

export default EvaluationForm;