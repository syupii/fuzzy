// frontend/src/components/EvaluationForm.tsx - 完全修正版
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
  Tab,
  FormControlLabel,
  Checkbox,
  FormGroup
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
  StudentProfile,
  RESEARCH_FIELDS,
  FIELD_CATEGORIES,
  CRITERIA_INFO,
  fieldUtils,
  apiService
} from '../services/api';

// ローカル型定義
interface ResearchFieldInterests {
  [key: string]: number;
}

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

  // 研究分野選択の状態
  const [selectedFields, setSelectedFields] = useState<Set<string>>(new Set());
  const [fieldInterests, setFieldInterests] = useState<ResearchFieldInterests>({});

  // 評価基準変更ハンドラー
  const handlePreferenceChange = (criterion: keyof EvaluationPreferences, value: number) => {
    setPreferences(prev => ({
      ...prev,
      [criterion]: value
    }));
  };

  // 研究分野選択ハンドラー
  const handleFieldToggle = (fieldId: string) => {
    setSelectedFields(prev => {
      const newSet = new Set(prev);
      if (newSet.has(fieldId)) {
        newSet.delete(fieldId);
        // 選択解除時は興味度も削除
        setFieldInterests(prevInterests => {
          const newInterests = { ...prevInterests };
          delete newInterests[fieldId];
          return newInterests;
        });
      } else {
        newSet.add(fieldId);
        // 選択時はデフォルト興味度5を設定
        setFieldInterests(prevInterests => ({
          ...prevInterests,
          [fieldId]: 5
        }));
      }
      return newSet;
    });
  };

  // 研究分野興味度変更ハンドラー
  const handleFieldInterestChange = (fieldId: string, value: number) => {
    setFieldInterests(prev => ({
      ...prev,
      [fieldId]: value
    }));
  };

  // デモデータ読み込み
  const handleLoadDemo = async () => {
    try {
      // デモ用の設定を読み込み
      const demoPreferences: EvaluationPreferences = {
        research_intensity: 7,
        advisor_style: 6,
        team_work: 8,
        workload: 6,
        theory_practice: 7,
        research_field_match: 9,
        skill_development: 8,
        lab_atmosphere: 7,
        flexibility: 6,
        publication_opportunity: 8,
        interdisciplinary: 7,
        communication_style: 6,
        innovation_risk: 7,
      };

      setPreferences(demoPreferences);

      // デモ用研究分野選択
      const demoFields = ['ai_ml', 'image_processing', 'web_design'];
      const demoSelectedFields = new Set(demoFields);
      const demoFieldInterests: ResearchFieldInterests = {
        'ai_ml': 9,
        'image_processing': 7,
        'web_design': 6
      };

      setSelectedFields(demoSelectedFields);
      setFieldInterests(demoFieldInterests);

      setTabValue(0);
      setError('');
    } catch (err) {
      setError('デモデータの読み込みに失敗しました');
    }
  };

  // 評価実行
  const handleEvaluate = async () => {
    try {
      setIsLoading(true);
      setError('');

      // 選択された分野の興味度と選択されていない分野（0）を統合
      const allFieldInterests: ResearchFieldInterests = {};
      RESEARCH_FIELDS.forEach(field => {
        allFieldInterests[field.id] = selectedFields.has(field.id)
          ? (fieldInterests[field.id] || 5)
          : 0;
      });

      // バックエンドが期待する形式に合わせて統合プロファイルを作成
      const studentProfile = {
        ...preferences,
        field_interests: allFieldInterests
      };

      // バックエンドが期待する形式でデータを送信
      const evaluationData = {
        student_profile: studentProfile
      };

      console.log('🚀 送信データ:', evaluationData);

      const response = await apiService.evaluateLabs(evaluationData);
      onResults(response);
      setTabValue(2); // 結果確認タブに移動
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '評価処理でエラーが発生しました';
      setError(errorMessage);
      onError(errorMessage);
    } finally {
      setIsLoading(false);
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
              <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                {info.range}
              </Typography>
              <Slider
                value={preferences[key as keyof EvaluationPreferences]}
                onChange={(_, value) => handlePreferenceChange(key as keyof EvaluationPreferences, value as number)}
                min={1}
                max={10}
                step={1}
                marks
                valueLabelDisplay="on"
                sx={{ mt: 1 }}
              />
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );

  // 研究分野選択コンポーネント
  const renderFieldInterests = () => {
    // カテゴリアイコンマップ
    const categoryIcons: { [key: string]: React.ReactNode } = {
      'テクノロジー・システム': <Science color="primary" />,
      'クリエイティブ': <Palette color="secondary" />,
      'エンターテイメント': <SportsEsports color="error" />,
      '人文・社会・体育': <Psychology color="warning" />
    };

    return (
      <Box>
        <Typography variant="h6" gutterBottom>
          研究分野選択
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          興味のある分野を選択し、興味の度合いを設定してください（複数選択可能）
        </Typography>

        {FIELD_CATEGORIES.map(category => {
          const fieldsInCategory = fieldUtils.getFieldsByCategory(category);

          return (
            <Accordion key={category} defaultExpanded>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  {categoryIcons[category]}
                  <Typography variant="subtitle1">
                    {category} ({fieldsInCategory.length}分野)
                  </Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  {fieldsInCategory.map(field => (
                    <Grid item xs={12} key={field.id}>
                      <Card
                        variant="outlined"
                        sx={{
                          p: 2,
                          bgcolor: selectedFields.has(field.id) ? 'primary.50' : 'transparent',
                          border: selectedFields.has(field.id) ? 2 : 1,
                          borderColor: selectedFields.has(field.id) ? 'primary.main' : 'divider'
                        }}
                      >
                        <Box sx={{ mb: 2 }}>
                          <FormControlLabel
                            control={
                              <Checkbox
                                checked={selectedFields.has(field.id)}
                                onChange={() => handleFieldToggle(field.id)}
                                color="primary"
                              />
                            }
                            label={
                              <Box>
                                <Typography variant="subtitle2">
                                  {field.name}
                                </Typography>
                                <Typography variant="body2" color="text.secondary">
                                  {field.description}
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                  教員数: {field.faculty_count}名
                                </Typography>
                              </Box>
                            }
                          />
                        </Box>

                        {selectedFields.has(field.id) && (
                          <Box sx={{ mt: 2, pl: 4 }}>
                            <Typography variant="body2" gutterBottom>
                              興味度: {fieldInterests[field.id] || 5}/10
                            </Typography>
                            <Slider
                              value={fieldInterests[field.id] || 5}
                              onChange={(_, value) => handleFieldInterestChange(field.id, value as number)}
                              min={1}
                              max={10}
                              step={1}
                              marks
                              valueLabelDisplay="on"
                              size="small"
                            />
                          </Box>
                        )}
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </AccordionDetails>
            </Accordion>
          );
        })}

        {selectedFields.size > 0 && (
          <Box sx={{ mt: 3, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
            <Typography variant="subtitle2" gutterBottom>
              選択中の分野 ({selectedFields.size}分野):
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {Array.from(selectedFields).map(fieldId => {
                const field = RESEARCH_FIELDS.find(f => f.id === fieldId);
                const interest = fieldInterests[fieldId] || 5;
                return field ? (
                  <Chip
                    key={fieldId}
                    label={`${field.name} (${interest}/10)`}
                    size="small"
                    color="primary"
                    variant="outlined"
                  />
                ) : null;
              })}
            </Box>
          </Box>
        )}
      </Box>
    );
  };

  // 評価実行コンポーネント
  const renderEvaluationExecute = () => {
    const selectedFieldsCount = selectedFields.size;

    return (
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
                研究分野: {selectedFieldsCount}分野選択中
              </Typography>
            </Grid>
          </Grid>

          {selectedFieldsCount > 0 && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" gutterBottom>
                <strong>選択した分野と興味度:</strong>
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {Array.from(selectedFields).map(fieldId => {
                  const field = RESEARCH_FIELDS.find(f => f.id === fieldId);
                  const interest = fieldInterests[fieldId] || 5;
                  return field ? (
                    <Chip
                      key={fieldId}
                      label={`${field.name} (${interest}/10)`}
                      size="small"
                      color="primary"
                      variant="outlined"
                    />
                  ) : null;
                })}
              </Box>
            </Box>
          )}
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
            disabled={isLoading || selectedFieldsCount === 0}
            startIcon={<Timeline />}
            size="large"
          >
            {isLoading ? '評価中...' : '研究室マッチング実行'}
          </Button>
        </Box>

        {selectedFieldsCount === 0 && (
          <Alert severity="warning" sx={{ mt: 2 }}>
            研究分野を最低1つ以上選択してください
          </Alert>
        )}

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
  };

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