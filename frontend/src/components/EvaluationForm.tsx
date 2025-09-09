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

  // 評価基準の状態（innovation_riskを含む）
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

  // ★ 重要: この2つのstateを正しく定義
  const [selectedFields, setSelectedFields] = useState<Set<string>>(new Set());
  const [fieldInterests, setFieldInterests] = useState<ResearchFieldInterests>({});

  // 評価基準変更ハンドラー
  const handlePreferenceChange = (criterion: keyof EvaluationPreferences, value: number) => {
    setPreferences(prev => ({
      ...prev,
      [criterion]: value
    }));
  };

  // 研究分野選択ハンドラー（チェックボックス用）
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

      // 選択された分野の興味度と選択されていない分野（0）を統合
      const allFieldInterests: ResearchFieldInterests = {};
      RESEARCH_FIELDS.forEach(field => {
        allFieldInterests[field.id] = selectedFields.has(field.id)
          ? (fieldInterests[field.id] || 5)
          : 0;
      });

      // APIには統合された興味度データを送信
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

      // デモの研究分野選択と興味度を設定
      const demoSelectedFields = new Set<string>();
      const demoFieldInterests: ResearchFieldInterests = {};

      Object.entries(demoProfile.field_interests).forEach(([field, interest]) => {
        if (interest > 0) {
          demoSelectedFields.add(field);
          demoFieldInterests[field] = interest;
        }
      });

      setSelectedFields(demoSelectedFields);
      setFieldInterests(demoFieldInterests);

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

  // 研究分野興味コンポーネント（2段階入力版）
  const renderFieldInterests = () => {
    const selectedCount = selectedFields.size;

    return (
      <Box>
        <Typography variant="h6" gutterBottom>
          研究分野選択・興味度設定
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          まず興味のある分野をチェックし、次に各分野への興味度を詳細設定してください
        </Typography>

        <Alert severity="info" sx={{ mb: 3 }}>
          選択済み分野: {selectedCount}件
          {selectedCount === 0 && " - 最低1つ以上の分野を選択してください"}
        </Alert>

        {/* Step 1: 分野選択 */}
        <Typography variant="h6" sx={{ mb: 2 }}>
          ステップ1: 興味のある分野を選択
        </Typography>

        {FIELD_CATEGORIES.map((category) => {
          const fields = fieldUtils.getFieldsByCategory(category);
          const categorySelectedCount = fields.filter(field => selectedFields.has(field.id)).length;

          return (
            <Accordion key={category} defaultExpanded={category === 'テクノロジー・システム'}>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Typography variant="h6">
                  {category === 'テクノロジー・システム' && <Science sx={{ mr: 1 }} />}
                  {category === 'クリエイティブ' && <Palette sx={{ mr: 1 }} />}
                  {category === 'エンターテイメント' && <SportsEsports sx={{ mr: 1 }} />}
                  {category} ({fields.length}分野)
                  {categorySelectedCount > 0 && (
                    <Chip
                      label={`${categorySelectedCount}選択中`}
                      size="small"
                      color="primary"
                      sx={{ ml: 2 }}
                    />
                  )}
                </Typography>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={2}>
                  {fields.map((field) => {
                    const isSelected = selectedFields.has(field.id);

                    return (
                      <Grid item xs={12} md={6} key={field.id}>
                        <Card
                          variant="outlined"
                          sx={{
                            p: 2,
                            cursor: 'pointer',
                            backgroundColor: isSelected ? 'primary.50' : 'background.paper',
                            border: isSelected ? 2 : 1,
                            borderColor: isSelected ? 'primary.main' : 'divider',
                            '&:hover': {
                              backgroundColor: isSelected ? 'primary.100' : 'grey.50'
                            }
                          }}
                          onClick={() => handleFieldToggle(field.id)}
                        >
                          <FormControlLabel
                            control={
                              <Checkbox
                                checked={isSelected}
                                onChange={() => handleFieldToggle(field.id)}
                                color="primary"
                              />
                            }
                            label={
                              <Box>
                                <Typography variant="subtitle1" gutterBottom>
                                  {field.name}
                                </Typography>
                                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                                  {field.description}
                                </Typography>
                                <Chip
                                  label={`教員数: ${field.faculty_count}名`}
                                  size="small"
                                  variant="outlined"
                                />
                              </Box>
                            }
                            sx={{ alignItems: 'flex-start', width: '100%' }}
                          />
                        </Card>
                      </Grid>
                    );
                  })}
                </Grid>
              </AccordionDetails>
            </Accordion>
          );
        })}

        {/* Step 2: 興味度詳細設定 */}
        {selectedCount > 0 && (
          <Box sx={{ mt: 4 }}>
            <Typography variant="h6" sx={{ mb: 2 }}>
              ステップ2: 選択した分野の興味度を詳細設定
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              選択した各分野への興味度を1-10で設定してください
            </Typography>

            <Grid container spacing={3}>
              {Array.from(selectedFields).map((fieldId) => {
                const field = RESEARCH_FIELDS.find(f => f.id === fieldId);
                if (!field) return null;

                return (
                  <Grid item xs={12} md={6} key={fieldId}>
                    <Card variant="outlined" sx={{ p: 3 }}>
                      <Typography variant="subtitle1" gutterBottom>
                        {field.name}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                        {field.description}
                      </Typography>
                      <Chip
                        label={`教員数: ${field.faculty_count}名`}
                        size="small"
                        sx={{ mb: 2 }}
                      />
                      <Box sx={{ px: 2 }}>
                        <Slider
                          value={fieldInterests[fieldId] || 5}
                          onChange={(_, value) => handleFieldInterestChange(fieldId, value as number)}
                          min={1}
                          max={10}
                          step={1}
                          marks
                          valueLabelDisplay="on"
                        />
                      </Box>
                      <Typography variant="caption" color="text.secondary">
                        1(少し興味あり) ～ 10(非常に興味あり)
                      </Typography>
                    </Card>
                  </Grid>
                );
              })}
            </Grid>
          </Box>
        )}
      </Box>
    );
  };

  // 評価実行コンポーネント
  const renderEvaluationExecute = () => {
    // ★ selectedFields は正しく定義されているので使用可能
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
                {Array.from(selectedFields).map((fieldId) => {
                  // ★ 型を明示的にstring指定
                  const fieldIdStr = fieldId as string;
                  const field = RESEARCH_FIELDS.find(f => f.id === fieldIdStr);
                  const interest = fieldInterests[fieldIdStr] || 5;
                  return field ? (
                    <Chip
                      key={fieldIdStr}
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