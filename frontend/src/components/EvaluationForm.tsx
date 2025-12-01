// frontend/src/components/EvaluationForm.tsx - 完全修正版
import React, { useState, useEffect } from 'react';
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
  Menu,
  MenuItem,
  CircularProgress,
  Divider,
} from '@mui/material';
import {
  ExpandMore,
  School,
  Timeline,
  ArrowDropDown,
} from '@mui/icons-material';

import { apiService, StudentProfile, RESEARCH_FIELDS, FIELD_CATEGORIES, ResearchField } from '../services/api';

// --- 型定義 ---
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
    range: '1 (軽い研究) ～ 10 (集中研究)'
  },
  advisor_style: {
    name: '指導スタイル',
    description: '教授からの指導の受け方の好み',
    range: '1 (厳格指導) ～ 10 (自由指導)'
  },
  team_work: {
    name: 'チームワーク',
    description: '研究での他者との協働の程度',
    range: '1 (個人研究) ～ 10 (チーム研究)'
  },
  workload: {
    name: 'ワークロード',
    description: '研究活動の忙しさに対する許容度',
    range: '1 (軽い負荷) ～ 10 (重い負荷)'
  },
  theory_practice: {
    name: '理論・実践バランス',
    description: '理論研究と実践的研究のバランス',
    range: '1 (理論重視) ～ 10 (実践重視)'
  },
  research_field_match: {
    name: '分野重視度',
    description: '分野マッチングと基本項目のバランス',
    range: '1 (基本項目重視) ～ 10 (分野重視)'
  },
  skill_development: {
    name: 'スキル開発',
    description: '専門性と汎用性のバランス',
    range: '1 (専門特化) ～ 10 (幅広いスキル)'
  },
  lab_atmosphere: {
    name: '研究室雰囲気',
    description: '研究室の全体的な雰囲気',
    range: '1 (静寂集中) ～ 10 (活発議論)'
  },
  flexibility: {
    name: '柔軟性',
    description: '研究時間の自由度',
    range: '1 (固定スケジュール) ～ 10 (柔軟スケジュール)'
  },
  publication_opportunity: {
    name: '論文発表機会',
    description: '研究成果の論文化機会',
    range: '1 (少ない機会) ～ 10 (豊富な機会)'
  },
  interdisciplinary: {
    name: '学際性',
    description: '他分野との連携の程度',
    range: '1 (単一分野) ～ 10 (学際連携)'
  },
  communication_style: {
    name: 'コミュニケーション',
    description: '研究室での交流スタイル',
    range: '1 (少人数密接) ～ 10 (オープン交流)'
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
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);

  const [demoProfileNames, setDemoProfileNames] = useState<string[]>([]);
  const [loadingDemoProfiles, setLoadingDemoProfiles] = useState(false);

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

  useEffect(() => {
    const fetchDemoProfileNames = async () => {
      try {
        const names = await apiService.getDemoProfileNames();
        setDemoProfileNames(names);
      } catch (err) {
        console.error('デモプロファイル名の取得に失敗:', err);
      }
    };
    fetchDemoProfileNames();
  }, []);

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

  const handleDemoMenuOpen = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };

  const handleDemoMenuClose = () => {
    setAnchorEl(null);
  };

  const handleLoadDemo = async (profileName: string) => {
    setLoadingDemoProfiles(true);
    try {
      const demoData = await apiService.getDemoProfileSimple(profileName);

      const newPreferences: EvaluationPreferencesWithPriority = {
        research_intensity: demoData.research_intensity,
        advisor_style: demoData.advisor_style,
        team_work: demoData.team_work,
        workload: demoData.workload,
        theory_practice: demoData.theory_practice,
        research_field_match: demoData.research_field_match,
        skill_development: demoData.skill_development,
        lab_atmosphere: demoData.lab_atmosphere,
        flexibility: demoData.flexibility,
        publication_opportunity: demoData.publication_opportunity,
        interdisciplinary: demoData.interdisciplinary,
        communication_style: demoData.communication_style,
        research_intensity_priority: demoData.research_intensity_priority || 5,
        advisor_style_priority: demoData.advisor_style_priority || 5,
        team_work_priority: demoData.team_work_priority || 5,
        workload_priority: demoData.workload_priority || 5,
        theory_practice_priority: demoData.theory_practice_priority || 5,
        research_field_match_priority: demoData.research_field_match_priority || 5,
        skill_development_priority: demoData.skill_development_priority || 5,
        lab_atmosphere_priority: demoData.lab_atmosphere_priority || 5,
        flexibility_priority: demoData.flexibility_priority || 5,
        publication_opportunity_priority: demoData.publication_opportunity_priority || 5,
        interdisciplinary_priority: demoData.interdisciplinary_priority || 5,
        communication_style_priority: demoData.communication_style_priority || 5
      };

      setPreferences(newPreferences);

      const newSelectedFields = new Set<string>();
      const newFieldInterests: ResearchFieldInterests = {};

      if (demoData.field_interests && typeof demoData.field_interests === 'object') {
        Object.entries(demoData.field_interests).forEach(([fieldId, interestLevel]) => {
          newSelectedFields.add(fieldId);
          newFieldInterests[fieldId] = Number(interestLevel);
        });
      }

      setSelectedFields(newSelectedFields);
      setFieldInterests(newFieldInterests);
      handleDemoMenuClose();
      setError('');
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'デモプロファイルの読み込みに失敗しました';
      setError(errorMessage);
    } finally {
      setLoadingDemoProfiles(false);
    }
  };

  const handleEvaluate = async () => {
    setIsLoading(true);
    setError('');
    try {
      const studentProfile: StudentProfile = {
        research_intensity: preferences.research_intensity,
        advisor_style: preferences.advisor_style,
        team_work: preferences.team_work,
        workload: preferences.workload,
        theory_practice: preferences.theory_practice,
        research_field_match: preferences.research_field_match,
        skill_development: preferences.skill_development,
        lab_atmosphere: preferences.lab_atmosphere,
        flexibility: preferences.flexibility,
        publication_opportunity: preferences.publication_opportunity,
        interdisciplinary: preferences.interdisciplinary,
        communication_style: preferences.communication_style,
        research_intensity_priority: preferences.research_intensity_priority,
        advisor_style_priority: preferences.advisor_style_priority,
        team_work_priority: preferences.team_work_priority,
        workload_priority: preferences.workload_priority,
        theory_practice_priority: preferences.theory_practice_priority,
        research_field_match_priority: preferences.research_field_match_priority,
        skill_development_priority: preferences.skill_development_priority,
        lab_atmosphere_priority: preferences.lab_atmosphere_priority,
        flexibility_priority: preferences.flexibility_priority,
        publication_opportunity_priority: preferences.publication_opportunity_priority,
        interdisciplinary_priority: preferences.interdisciplinary_priority,
        communication_style_priority: preferences.communication_style_priority,
        field_interests: Object.fromEntries(
          Array.from(selectedFields).map(fieldId => [
            fieldId,
            fieldInterests[fieldId] || 5
          ])
        )
      };
      const result = await apiService.evaluate(studentProfile);
      const inputValues = {
        basicCriteria: {
          research_intensity: preferences.research_intensity,
          advisor_style: preferences.advisor_style,
          team_work: preferences.team_work,
          workload: preferences.workload,
          theory_practice: preferences.theory_practice,
          research_field_match: preferences.research_field_match,
          skill_development: preferences.skill_development,
          lab_atmosphere: preferences.lab_atmosphere,
          flexibility: preferences.flexibility,
          publication_opportunity: preferences.publication_opportunity,
          interdisciplinary: preferences.interdisciplinary,
          communication_style: preferences.communication_style,
        },
        fieldInterests: fieldInterests
      };

      onResults(result, inputValues);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '評価処理でエラーが発生しました';
      setError(errorMessage);
      onError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  // ✅ 正しい renderBasicCriteria
  const renderBasicCriteria = () => (
    <Box>
      <Typography variant="h6" gutterBottom>
        評価基準設定
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        各項目について「希望値」と「重要度（優先度）」を設定してください
      </Typography>
      <Grid container spacing={3}>
        {Object.entries(CRITERIA_INFO).map(([criterionKey, criterionInfo]) => (
          <Grid item xs={12} sm={6} key={criterionKey}>
            <Card
              variant="outlined"
              sx={{
                p: 2.5,
                height: '100%',
                transition: 'all 0.3s',
                '&:hover': {
                  boxShadow: 3,
                  borderColor: 'primary.main'
                }
              }}
            >
              {/* タイトル */}
              <Typography variant="h6" color="primary" gutterBottom sx={{ fontWeight: 'bold' }}>
                {criterionInfo.name}
              </Typography>

              {/* 説明文（タイトル直下） */}
              <Typography variant="body2" color="text.secondary" gutterBottom>
                {criterionInfo.description}
              </Typography>

              {/* 範囲情報 */}
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 2 }}>
                {criterionInfo.range}
              </Typography>

              <Divider sx={{ mb: 2 }} />

              {/* スライダー部分 */}
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <Typography variant="body2" gutterBottom sx={{ fontWeight: 'medium' }}>
                    希望値: {preferences[criterionKey as keyof EvaluationPreferencesWithPriority]}
                  </Typography>
                  <Slider
                    value={preferences[criterionKey as keyof EvaluationPreferencesWithPriority]}
                    onChange={(_, value) => handlePreferenceChange(criterionKey as keyof EvaluationPreferencesWithPriority, value as number)}
                    min={1}
                    max={10}
                    step={1}
                    marks
                    valueLabelDisplay="auto"
                  />
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="body2" gutterBottom sx={{ fontWeight: 'medium' }}>
                    重要度: {preferences[`${criterionKey}_priority` as keyof EvaluationPreferencesWithPriority]}
                  </Typography>
                  <Slider
                    value={preferences[`${criterionKey}_priority` as keyof EvaluationPreferencesWithPriority]}
                    onChange={(_, value) => handlePreferenceChange(`${criterionKey}_priority` as keyof EvaluationPreferencesWithPriority, value as number)}
                    min={1}
                    max={10}
                    step={1}
                    marks
                    valueLabelDisplay="auto"
                    color="warning"
                  />
                </Grid>
              </Grid>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );

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
              <Typography variant="body2">評価基準: {Object.keys(CRITERIA_INFO).length}項目設定済み</Typography>
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
            variant="outlined"
            onClick={handleDemoMenuOpen}
            startIcon={loadingDemoProfiles ? <CircularProgress size={20} /> : <School />}
            endIcon={<ArrowDropDown />}
            disabled={loadingDemoProfiles}
          >
            {loadingDemoProfiles ? '読み込み中...' : 'デモデータ選択'}
          </Button>
          <Menu anchorEl={anchorEl} open={Boolean(anchorEl)} onClose={handleDemoMenuClose}>
            {demoProfileNames.length > 0 ? (
              demoProfileNames.map(profileName => (
                <MenuItem key={profileName} onClick={() => handleLoadDemo(profileName)}>
                  {profileName}
                </MenuItem>
              ))
            ) : (
              <MenuItem disabled>読み込み中...</MenuItem>
            )}
          </Menu>
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