// frontend/src/components/EvaluationForm.tsx - 分野重視度特別対応版
import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
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
  Chip,
  Divider,
  Paper,
} from '@mui/material';
import {
  ExpandMore,
  Timeline,
} from '@mui/icons-material';

import { apiService, StudentProfile, RESEARCH_FIELDS, FIELD_CATEGORIES, ResearchField } from '../services/api';

// --- 型定義 (変更なし) ---
interface EvaluationPreferencesWithPriority {
  research_intensity: number;
  advisor_style: number;
  team_work: number;
  workload: number;
  theory_practice: number;
  research_field_match: number;
  skill_development: number;
  lab_atmosphere: number;
  flexibility: number;
  publication_opportunity: number;
  interdisciplinary: number;
  communication_style: number;
  research_intensity_priority: number;
  advisor_style_priority: number;
  team_work_priority: number;
  workload_priority: number;
  theory_practice_priority: number;
  research_field_match_priority: number;
  skill_development_priority: number;
  lab_atmosphere_priority: number;
  flexibility_priority: number;
  publication_opportunity_priority: number;
  interdisciplinary_priority: number;
  communication_style_priority: number;
}

interface ResearchFieldInterests {
  [key: string]: number;
}

interface EvaluationFormProps {
  onResults: (response: any, inputValues?: any) => void;
  onError: (error: string) => void;
}

interface TabPanelProps {
  children?: React.ReactNode;
  value: number;
  index: number;
}

interface CriteriaInfo {
  name: string;
  description: string;
  range: string;
}

const CRITERIA_INFO: Record<string, CriteriaInfo> = {
  research_intensity: {
    name: '研究強度',
    description: '研究にどれだけ集中的に取り組みたいか',
    range: '1 (軽い) ～ 10 (集中)'
  },
  advisor_style: {
    name: '指導スタイル',
    description: '教授からの指導の受け方の好み',
    range: '1 (厳格) ～ 10 (自由)'
  },
  team_work: {
    name: 'チームワーク',
    description: '研究での他者との協働の程度',
    range: '1 (個人) ～ 10 (チーム)'
  },
  workload: {
    name: 'ワークロード',
    description: '研究活動の忙しさに対する許容度',
    range: '1 (軽い) ～ 10 (重い)'
  },
  theory_practice: {
    name: '理論・実践バランス',
    description: '理論研究と実践的研究のバランス',
    range: '1 (理論) ～ 10 (実践)'
  },
  research_field_match: { // 特別扱いするためリスト表示からは除外されます
    name: '分野重視度',
    description: '分野マッチングと基本項目のバランス',
    range: '基本項目重視 <-> 分野一致重視'
  },
  skill_development: {
    name: 'スキル開発',
    description: '専門性と汎用性のバランス',
    range: '1 (専門) ～ 10 (汎用)'
  },
  lab_atmosphere: {
    name: '研究室雰囲気',
    description: '研究室の全体的な雰囲気',
    range: '1 (静寂) ～ 10 (活発)'
  },
  flexibility: {
    name: '柔軟性',
    description: '研究時間の自由度',
    range: '1 (固定) ～ 10 (柔軟)'
  },
  publication_opportunity: {
    name: '論文発表機会',
    description: '研究成果の論文化機会',
    range: '1 (少) ～ 10 (多)'
  },
  interdisciplinary: {
    name: '学際性',
    description: '他分野との連携の程度',
    range: '1 (単一) ～ 10 (学際)'
  },
  communication_style: {
    name: 'コミュニケーション',
    description: '研究室での交流スタイル',
    range: '1 (密接) ～ 10 (オープン)'
  }
};

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
  <div hidden={value !== index}>
    {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
  </div>
);

const EvaluationForm: React.FC<EvaluationFormProps> = ({ onResults, onError }) => {
  const [tabValue, setTabValue] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>('');

  const [preferences, setPreferences] = useState<EvaluationPreferencesWithPriority>({
    research_intensity: 5, research_intensity_priority: 5,
    advisor_style: 5, advisor_style_priority: 5,
    team_work: 5, team_work_priority: 5,
    workload: 5, workload_priority: 5,
    theory_practice: 5, theory_practice_priority: 5,
    research_field_match: 5, research_field_match_priority: 5,
    skill_development: 5, skill_development_priority: 5,
    lab_atmosphere: 5, lab_atmosphere_priority: 5,
    flexibility: 5, flexibility_priority: 5,
    publication_opportunity: 5, publication_opportunity_priority: 5,
    interdisciplinary: 5, interdisciplinary_priority: 5,
    communication_style: 5, communication_style_priority: 5
  });

  const [selectedFields, setSelectedFields] = useState<Set<string>>(new Set());
  const [fieldInterests, setFieldInterests] = useState<ResearchFieldInterests>({});

  const handlePreferenceChange = (key: keyof EvaluationPreferencesWithPriority, value: number) => {
    setPreferences(prev => ({ ...prev, [key]: value }));
  };

  const handleFieldToggle = (fieldId: string) => {
    setSelectedFields(prev => {
      const newSet = new Set(prev);
      if (newSet.has(fieldId)) {
        newSet.delete(fieldId);
        const newInterests = { ...fieldInterests };
        delete newInterests[fieldId];
        setFieldInterests(newInterests);
      } else {
        newSet.add(fieldId);
        setFieldInterests(prev => ({ ...prev, [fieldId]: 5 }));
      }
      return newSet;
    });
  };

  const handleFieldInterestChange = (fieldId: string, value: number) => {
    setFieldInterests(prev => ({ ...prev, [fieldId]: value }));
  };

  const handleEvaluate = async () => {
    setIsLoading(true);
    setError('');
    try {
      // APIコールのためのデータ構築（変更なし）
      const studentProfile: StudentProfile = {
        ...preferences,
        field_interests: Object.fromEntries(
          Array.from(selectedFields).map(fieldId => [
            fieldId,
            fieldInterests[fieldId] || 5
          ])
        )
      };
      const result = await apiService.evaluate(studentProfile);

      onResults(result, {
        basicCriteria: preferences,
        fieldInterests: fieldInterests
      });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '評価処理でエラーが発生しました';
      setError(errorMessage);
      onError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  // ✅ 修正版 renderBasicCriteria
  // 分野重視度を特別扱いし、残りの11項目をグリッド表示
  const renderBasicCriteria = () => {
    // 特別項目(分野重視度)と、その他の通常項目に分離
    const specialKey = 'research_field_match';
    const specialInfo = CRITERIA_INFO[specialKey];
    const normalCriteria = Object.entries(CRITERIA_INFO).filter(([key]) => key !== specialKey);

    return (
      <Box>
        <Typography variant="h6" gutterBottom>
          評価基準設定
        </Typography>

        {/* 1. 特別項目：分野重視度（重要度スライダーなし、特別デザイン） */}
        <Paper
          elevation={0}
          sx={{
            p: 2.5,
            mb: 4,
            bgcolor: 'aliceblue', // 特別感を出す薄い青背景
            border: '1px solid',
            borderColor: 'primary.200',
            borderRadius: 2
          }}
        >
          <Grid container spacing={3} alignItems="center">
            <Grid item xs={12} md={4}>
              <Typography variant="subtitle1" color="primary" sx={{ fontWeight: 'bold' }}>
                {specialInfo.name}（全体バランス）
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                {specialInfo.description}
              </Typography>
            </Grid>
            <Grid item xs={12} md={8}>
              <Box sx={{ px: 1 }}>
                <Slider
                  value={preferences[specialKey as keyof EvaluationPreferencesWithPriority]}
                  onChange={(_, value) => handlePreferenceChange(specialKey as keyof EvaluationPreferencesWithPriority, value as number)}
                  min={1}
                  max={10}
                  step={1}
                  marks
                  valueLabelDisplay="auto"
                  sx={{ height: 8 }} // 少し太くして存在感を出す
                />
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                  <Typography variant="caption" sx={{ fontWeight: 'bold', color: 'text.secondary' }}>
                    基本項目（雰囲気・環境）重視
                  </Typography>
                  <Typography variant="caption" sx={{ fontWeight: 'bold', color: 'primary.main' }}>
                    分野一致（研究内容）重視
                  </Typography>
                </Box>
              </Box>
            </Grid>
          </Grid>
        </Paper>

        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
          以下の項目について、あなたの希望値と重要度を設定してください
        </Typography>

        {/* 2. 通常項目：残り11項目を3列グリッドで表示 */}
        <Grid container spacing={2}>
          {normalCriteria.map(([criterionKey, criterionInfo]) => (
            <Grid item xs={12} sm={6} md={4} key={criterionKey}>
              <Card
                variant="outlined"
                sx={{
                  p: 1.5,
                  height: '100%',
                  display: 'flex',
                  flexDirection: 'column',
                  transition: 'all 0.3s',
                  '&:hover': {
                    boxShadow: 2,
                    borderColor: 'primary.main'
                  }
                }}
              >
                {/* ヘッダー部分 */}
                <Box sx={{ mb: 1 }}>
                  <Typography variant="subtitle2" sx={{ fontWeight: 'bold', fontSize: '0.95rem' }}>
                    {criterionInfo.name}
                  </Typography>
                  <Typography variant="caption" color="text.secondary" sx={{ display: 'block', lineHeight: 1.2 }}>
                    {criterionInfo.description}
                  </Typography>
                  <Typography variant="caption" color="text.disabled" sx={{ fontSize: '0.7rem' }}>
                    {criterionInfo.range}
                  </Typography>
                </Box>

                <Divider sx={{ my: 0.5, opacity: 0.5 }} />

                {/* 入力エリア */}
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                  {/* 希望値 */}
                  <Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="caption" sx={{ fontWeight: 500 }}>希望値</Typography>
                      <Typography variant="caption" color="primary" sx={{ fontWeight: 'bold' }}>
                        {preferences[criterionKey as keyof EvaluationPreferencesWithPriority]}
                      </Typography>
                    </Box>
                    <Slider
                      size="small"
                      value={preferences[criterionKey as keyof EvaluationPreferencesWithPriority]}
                      onChange={(_, value) => handlePreferenceChange(criterionKey as keyof EvaluationPreferencesWithPriority, value as number)}
                      min={1} max={10} step={1} marks
                      sx={{ py: 0.5 }}
                    />
                  </Box>

                  {/* 重要度 */}
                  <Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Typography variant="caption" sx={{ fontWeight: 500 }}>優先度</Typography>
                      <Typography variant="caption" color="warning.main" sx={{ fontWeight: 'bold' }}>
                        {preferences[`${criterionKey}_priority` as keyof EvaluationPreferencesWithPriority]}
                      </Typography>
                    </Box>
                    <Slider
                      size="small"
                      value={preferences[`${criterionKey}_priority` as keyof EvaluationPreferencesWithPriority]}
                      onChange={(_, value) => handlePreferenceChange(`${criterionKey}_priority` as keyof EvaluationPreferencesWithPriority, value as number)}
                      min={1} max={10} step={1} marks
                      color="warning"
                      sx={{ py: 0.5 }}
                    />
                  </Box>
                </Box>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Box>
    );
  };

  const renderFieldInterests = () => {
    const categories = Object.keys(FIELD_CATEGORIES) as Array<keyof typeof FIELD_CATEGORIES>;
    return (
      <Box>
        <Typography variant="h6" gutterBottom>研究分野選択（27分野体系）</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          興味のある研究分野を選択し、興味度を設定してください
        </Typography>
        {categories.map(category => (
          <Accordion key={category} defaultExpanded>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6">{category}</Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                {FIELD_CATEGORIES[category].map((field: ResearchField) => (
                  <Grid item xs={12} sm={6} md={4} key={field.id}>
                    <Card
                      variant="outlined"
                      sx={{
                        p: 2,
                        borderWidth: selectedFields.has(field.id) ? 2 : 1,
                        borderColor: selectedFields.has(field.id) ? 'primary.main' : 'grey.300'
                      }}
                    >
                      <FormControlLabel
                        control={
                          <Checkbox
                            checked={selectedFields.has(field.id)}
                            onChange={() => handleFieldToggle(field.id)}
                          />
                        }
                        label={field.name}
                      />
                      {selectedFields.has(field.id) && (
                        <Box sx={{ mt: 2 }}>
                          <Typography variant="body2" gutterBottom>
                            興味度: {fieldInterests[field.id] || 5}
                          </Typography>
                          <Slider
                            value={fieldInterests[field.id] || 5}
                            onChange={(_, value) => handleFieldInterestChange(field.id, value as number)}
                            min={1} max={10} step={1} marks valueLabelDisplay="on"
                          />
                        </Box>
                      )}
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </AccordionDetails>
          </Accordion>
        ))}
      </Box>
    );
  };

  const renderEvaluationExecute = () => {
    const selectedFieldsCount = selectedFields.size;
    return (
      <Box>
        <Typography variant="h6" gutterBottom>研究室マッチング実行</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          設定した条件に基づいて研究室マッチングを実行します
        </Typography>
        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
        <Card sx={{ p: 3, mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom>設定サマリー</Typography>
          <Grid container spacing={2}>
            <Grid item xs={6}>
              <Typography variant="body2">評価基準: 12項目設定済み</Typography>
            </Grid>
            <Grid item xs={6}>
              <Typography variant="body2">研究分野: {selectedFieldsCount}分野選択中</Typography>
            </Grid>
          </Grid>
          {selectedFieldsCount > 0 && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="body2" gutterBottom><strong>選択した分野と興味度:</strong></Typography>
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
        <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center', alignItems: 'center' }}>
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
              優先度を考慮した研究室適合性を評価中...
            </Typography>
          </Box>
        )}
      </Box>
    );
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Tabs value={tabValue} onChange={(_, newValue) => setTabValue(newValue)}>
        <Tab label="評価基準設定" />
        <Tab label="研究分野選択（27分野）" />
        <Tab label="評価実行" />
      </Tabs>
      <TabPanel value={tabValue} index={0}>{renderBasicCriteria()}</TabPanel>
      <TabPanel value={tabValue} index={1}>{renderFieldInterests()}</TabPanel>
      <TabPanel value={tabValue} index={2}>{renderEvaluationExecute()}</TabPanel>
    </Box>
  );
};

export default EvaluationForm;