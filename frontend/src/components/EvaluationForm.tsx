// frontend/src/components/EvaluationForm.tsx - チェックボックス式改善版

import React, { useState } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  Checkbox,
  Chip,
  FormControlLabel,
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
  Paper,
  Stepper,
  Step,
  StepLabel,
  List,
  ListItem,
  ListItemText,
  Divider
} from '@mui/material';
import {
  ExpandMore,
  Science,
  Palette,
  SportsEsports,
  School,
  Timeline,
  CheckBox,
  TuneRounded,
  PersonOutline
} from '@mui/icons-material';
import {
  EvaluationPreferences,
  EvaluationResponse,
  ResearchField,
  FieldSelectionState,
  SelectedFieldInterest,
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

  // 評価基準の状態（13項目）
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
    innovation_risk: 5
  });

  // 分野選択状態（チェックボックス）
  const [fieldSelection, setFieldSelection] = useState<FieldSelectionState>({});

  // 選択された分野の詳細評価
  const [selectedFieldInterests, setSelectedFieldInterests] = useState<SelectedFieldInterest[]>([]);

  // 選択された分野リストを取得
  const getSelectedFields = (): ResearchField[] => {
    return RESEARCH_FIELDS.filter(field => fieldSelection[field.id]);
  };

  // 分野選択の切り替え
  const handleFieldToggle = (fieldId: string) => {
    setFieldSelection(prev => {
      const newSelection = { ...prev, [fieldId]: !prev[fieldId] };

      // 選択解除の場合、詳細評価からも削除
      if (!newSelection[fieldId]) {
        setSelectedFieldInterests(prev =>
          prev.filter(interest => interest.fieldId !== fieldId)
        );
      } else {
        // 新規選択の場合、詳細評価に追加
        const exists = selectedFieldInterests.some(interest => interest.fieldId === fieldId);
        if (!exists) {
          const newInterest: SelectedFieldInterest = {
            fieldId,
            interestLevel: 5,
            experienceLevel: 5,
            priority: selectedFieldInterests.length + 1
          };
          setSelectedFieldInterests(prev => [...prev, newInterest]);
        }
      }

      return newSelection;
    });
  };

  // 詳細評価の更新
  const handleDetailedInterestChange = (fieldId: string, type: 'interest' | 'experience' | 'priority', value: number) => {
    setSelectedFieldInterests(prev =>
      prev.map(interest =>
        interest.fieldId === fieldId
          ? {
            ...interest,
            [type === 'interest' ? 'interestLevel' : type === 'experience' ? 'experienceLevel' : 'priority']: value
          }
          : interest
      )
    );
  };

  // 評価基準変更ハンドラー
  const handlePreferenceChange = (criterion: keyof EvaluationPreferences, value: number) => {
    setPreferences(prev => ({
      ...prev,
      [criterion]: value
    }));
  };

  // 評価実行
  const handleEvaluate = async () => {
    try {
      setIsLoading(true);
      setError('');

      if (selectedFieldInterests.length === 0) {
        throw new Error('少なくとも1つの研究分野を選択してください');
      }

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

      // デモで確実に動作する分野を選択（既存の11分野から）
      const validDemoFields = ['ai_machine_learning', 'web_design_ui_ux', 'game_development_esports'];
      const demoSelection: FieldSelectionState = {};
      const demoInterests: SelectedFieldInterest[] = [];

      validDemoFields.forEach((fieldId, index) => {
        demoSelection[fieldId] = true;
        demoInterests.push({
          fieldId,
          interestLevel: 8 - index,  // 8, 7, 6
          experienceLevel: 6,
          priority: index + 1
        });
      });

      setFieldSelection(demoSelection);
      setSelectedFieldInterests(demoInterests);
      setTabValue(0);

      console.log('✅ デモデータ設定完了:', {
        preferences: demoProfile.evaluation_criteria,
        selectedFields: demoInterests
      });
    } catch (err) {
      setError('デモデータの読み込みに失敗しました');
    }
  };

  // カテゴリ別のアイコン取得
  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'テクノロジー・システム': return <Science />;
      case 'クリエイティブ': return <Palette />;
      case 'エンターテイメント': return <SportsEsports />;
      case '人文・社会・体育': return <PersonOutline />;
      default: return <Science />;
    }
  };

  // 基本評価基準コンポーネント
  const renderBasicCriteria = () => (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <TuneRounded sx={{ mr: 1, color: 'primary.main' }} />
        <Typography variant="h5" gutterBottom sx={{ mb: 0 }}>
          基本評価基準（13項目）
        </Typography>
      </Box>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        これらの項目は研究室マッチング精度に大きく影響します。すべての項目を設定してください。
      </Typography>

      <Grid container spacing={3}>
        {Object.entries(CRITERIA_INFO).map(([key, info]) => (
          <Grid item xs={12} md={6} lg={4} key={key}>
            <Card variant="outlined" sx={{ p: 2, height: '100%' }}>
              <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
                {info.name}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2, minHeight: '2.5em' }}>
                {info.description}
              </Typography>
              <Box sx={{ px: 2 }}>
                <Slider
                  value={preferences[key as keyof EvaluationPreferences] || 5}
                  onChange={(_, value) => handlePreferenceChange(key as keyof EvaluationPreferences, value as number)}
                  min={1}
                  max={10}
                  step={1}
                  marks={[
                    { value: 1, label: '1' },
                    { value: 5, label: '5' },
                    { value: 10, label: '10' }
                  ]}
                  valueLabelDisplay="on"
                  sx={{ mb: 1 }}
                />
              </Box>
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
                {info.range}
              </Typography>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );

  // 分野選択コンポーネント（Step 1）
  const renderFieldSelection = () => (
    <Box>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
        <CheckBox sx={{ mr: 1, color: 'primary.main' }} />
        <Typography variant="h5" gutterBottom sx={{ mb: 0 }}>
          興味のある研究分野を選択
        </Typography>
      </Box>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        興味のある分野をチェックしてください（複数選択可）。選択した分野のみ詳細評価を行います。
      </Typography>

      {/* 選択状況サマリー */}
      <Paper sx={{ p: 2, mb: 3, backgroundColor: 'primary.50' }}>
        <Typography variant="subtitle1" gutterBottom>
          選択状況: {getSelectedFields().length} / {RESEARCH_FIELDS.length} 分野
        </Typography>
        {getSelectedFields().length > 0 && (
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mt: 1 }}>
            {getSelectedFields().map(field => (
              <Chip
                key={field.id}
                label={field.name}
                size="small"
                color="primary"
                variant="outlined"
                onDelete={() => handleFieldToggle(field.id)}
              />
            ))}
          </Box>
        )}
      </Paper>

      {FIELD_CATEGORIES.map((category) => {
        const fields = fieldUtils.getFieldsByCategory(category);
        const selectedInCategory = fields.filter(field => fieldSelection[field.id]).length;

        return (
          <Accordion key={category} defaultExpanded={category === 'テクノロジー・システム'}>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                {getCategoryIcon(category)}
                <Typography variant="h6">
                  {category} ({selectedInCategory}/{fields.length}分野選択済み)
                </Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                {fields.map((field) => (
                  <Grid item xs={12} md={6} key={field.id}>
                    <Card
                      variant="outlined"
                      sx={{
                        p: 2,
                        cursor: 'pointer',
                        backgroundColor: fieldSelection[field.id] ? 'primary.50' : 'background.paper',
                        borderColor: fieldSelection[field.id] ? 'primary.main' : 'divider',
                        '&:hover': { backgroundColor: fieldSelection[field.id] ? 'primary.100' : 'grey.50' }
                      }}
                      onClick={() => handleFieldToggle(field.id)}
                    >
                      <FormControlLabel
                        control={
                          <Checkbox
                            checked={fieldSelection[field.id] || false}
                            onChange={() => handleFieldToggle(field.id)}
                            color="primary"
                          />
                        }
                        label=""
                        sx={{ m: 0, position: 'absolute', top: 8, left: 8 }}
                      />
                      <Box sx={{ ml: 4 }}>
                        <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
                          {field.name}
                        </Typography>
                        <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                          {field.description}
                        </Typography>
                        <Chip
                          label={`教員数: ${field.faculty_count}名`}
                          size="small"
                          sx={{ mb: 1 }}
                        />
                        <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                          担当: {fieldUtils.getFacultyNames(field.id)}
                        </Typography>
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 1 }}>
                          {field.keywords.slice(0, 3).map(keyword => (
                            <Chip key={keyword} label={keyword} size="small" variant="outlined" />
                          ))}
                        </Box>
                      </Box>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </AccordionDetails>
          </Accordion>
        );
      })}
    </Box>
  );

  // 詳細評価コンポーネント（Step 2）
  const renderDetailedEvaluation = () => {
    const selectedFields = getSelectedFields();

    if (selectedFields.length === 0) {
      return (
        <Box sx={{ textAlign: 'center', py: 4 }}>
          <CheckBox sx={{ fontSize: 64, color: 'grey.400', mb: 2 }} />
          <Typography variant="h6" color="text.secondary">
            研究分野を選択してください
          </Typography>
          <Typography variant="body2" color="text.secondary">
            前のタブで興味のある分野を選択すると、ここで詳細評価を行えます
          </Typography>
        </Box>
      );
    }

    return (
      <Box>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
          <TuneRounded sx={{ mr: 1, color: 'primary.main' }} />
          <Typography variant="h5" gutterBottom sx={{ mb: 0 }}>
            選択分野の詳細評価（{selectedFields.length}分野）
          </Typography>
        </Box>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          選択した各分野について、興味レベル・経験レベル・優先順位を設定してください
        </Typography>

        <Grid container spacing={3}>
          {selectedFields.map((field, index) => {
            const interest = selectedFieldInterests.find(i => i.fieldId === field.id);

            return (
              <Grid item xs={12} lg={6} key={field.id}>
                <Card sx={{ p: 3, height: '100%' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                    {getCategoryIcon(field.category)}
                    <Typography variant="h6" sx={{ ml: 1, flexGrow: 1 }}>
                      {field.name}
                    </Typography>
                    <Chip
                      label={`優先度 ${interest?.priority || index + 1}位`}
                      color="primary"
                      size="small"
                    />
                  </Box>

                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    担当教員: {fieldUtils.getFacultyNames(field.id)}
                  </Typography>

                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      興味レベル: {interest?.interestLevel || 5}
                    </Typography>
                    <Slider
                      value={interest?.interestLevel || 5}
                      onChange={(_, value) => handleDetailedInterestChange(field.id, 'interest', value as number)}
                      min={1}
                      max={10}
                      step={1}
                      marks={[
                        { value: 1, label: '興味なし' },
                        { value: 5, label: '普通' },
                        { value: 10, label: '非常に興味' }
                      ]}
                      valueLabelDisplay="on"
                      sx={{ mb: 2 }}
                    />
                  </Box>

                  <Box sx={{ mb: 3 }}>
                    <Typography variant="subtitle2" gutterBottom>
                      経験レベル: {interest?.experienceLevel || 5}
                    </Typography>
                    <Slider
                      value={interest?.experienceLevel || 5}
                      onChange={(_, value) => handleDetailedInterestChange(field.id, 'experience', value as number)}
                      min={1}
                      max={10}
                      step={1}
                      marks={[
                        { value: 1, label: '未経験' },
                        { value: 5, label: '基礎レベル' },
                        { value: 10, label: '上級レベル' }
                      ]}
                      valueLabelDisplay="on"
                      sx={{ mb: 2 }}
                    />
                  </Box>

                  <Box>
                    <Typography variant="subtitle2" gutterBottom>
                      優先順位: {interest?.priority || index + 1}位
                    </Typography>
                    <Slider
                      value={interest?.priority || index + 1}
                      onChange={(_, value) => handleDetailedInterestChange(field.id, 'priority', value as number)}
                      min={1}
                      max={selectedFields.length}
                      step={1}
                      marks={selectedFields.map((_, i) => ({ value: i + 1, label: `${i + 1}位` }))}
                      valueLabelDisplay="on"
                    />
                  </Box>
                </Card>
              </Grid>
            );
          })}
        </Grid>
      </Box>
    );
  };

  // 評価実行コンポーネント
  const renderEvaluationExecute = () => {
    const selectedFields = getSelectedFields();

    return (
      <Box>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
          <Timeline sx={{ mr: 1, color: 'primary.main' }} />
          <Typography variant="h5" gutterBottom sx={{ mb: 0 }}>
            研究室マッチング実行
          </Typography>
        </Box>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          設定した条件に基づいて研究室マッチングを実行します
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {/* 設定サマリー */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} md={6}>
            <Card sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                評価基準設定
              </Typography>
              <Typography variant="body2" color="text.secondary">
                13項目すべて設定完了
              </Typography>
              <List dense>
                <ListItem>
                  <ListItemText
                    primary="基本項目"
                    secondary="5項目設定済み"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="拡張項目"
                    secondary="5項目設定済み"
                  />
                </ListItem>
                <ListItem>
                  <ListItemText
                    primary="特殊項目"
                    secondary="3項目設定済み"
                  />
                </ListItem>
              </List>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                研究分野選択
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                {selectedFields.length}分野選択済み / 全{RESEARCH_FIELDS.length}分野
              </Typography>
              {selectedFields.length > 0 ? (
                <List dense>
                  {selectedFields.slice(0, 3).map(field => (
                    <ListItem key={field.id}>
                      <ListItemText
                        primary={field.name}
                        secondary={`興味レベル: ${selectedFieldInterests.find(i => i.fieldId === field.id)?.interestLevel || 5}/10`}
                      />
                    </ListItem>
                  ))}
                  {selectedFields.length > 3 && (
                    <ListItem>
                      <ListItemText
                        primary={`他 ${selectedFields.length - 3}分野`}
                        secondary="選択済み"
                      />
                    </ListItem>
                  )}
                </List>
              ) : (
                <Alert severity="warning">
                  研究分野が選択されていません
                </Alert>
              )}
            </Card>
          </Grid>
        </Grid>

        {/* 実行ボタン */}
        <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
          <Button
            variant="outlined"
            onClick={handleLoadDemo}
            startIcon={<School />}
            size="large"
          >
            デモデータ読み込み
          </Button>
          <Button
            variant="contained"
            onClick={handleEvaluate}
            disabled={isLoading || selectedFields.length === 0}
            startIcon={<Timeline />}
            size="large"
            sx={{ minWidth: 200 }}
          >
            {isLoading ? '評価中...' : '研究室マッチング実行'}
          </Button>
        </Box>

        {isLoading && (
          <Box sx={{ mt: 3 }}>
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
      <Tabs
        value={tabValue}
        onChange={(_, newValue) => setTabValue(newValue)}
        variant="fullWidth"
        sx={{ borderBottom: 1, borderColor: 'divider' }}
      >
        <Tab label="基本設定" icon={<TuneRounded />} />
        <Tab label="分野選択" icon={<CheckBox />} />
        <Tab label="詳細評価" icon={<School />} />
        <Tab label="実行" icon={<Timeline />} />
      </Tabs>

      <TabPanel value={tabValue} index={0}>
        {renderBasicCriteria()}
      </TabPanel>

      <TabPanel value={tabValue} index={1}>
        {renderFieldSelection()}
      </TabPanel>

      <TabPanel value={tabValue} index={2}>
        {renderDetailedEvaluation()}
      </TabPanel>

      <TabPanel value={tabValue} index={3}>
        {renderEvaluationExecute()}
      </TabPanel>
    </Box>
  );
};

export default EvaluationForm;