// src/components/EvaluationForm.tsx - バックエンド13項目仕様に合わせた修正版
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
  AccessTime,
  Settings,
  EmojiEvents,
  ConnectWithoutContact,
  Lightbulb
} from '@mui/icons-material';
import {
  apiService,
  EvaluationPreferences,
  EvaluationResponse,
  ResearchFieldInterests,
  StudentProfile,
  RESEARCH_FIELDS,
  FIELD_CATEGORIES,
  CRITERIA_INFO,
  fieldUtils
} from '../services/api';

interface EvaluationFormProps {
  onResults: (results: EvaluationResponse) => void;
}

const EvaluationForm: React.FC<EvaluationFormProps> = ({ onResults }) => {
  // 13項目評価基準の初期値（バックエンド仕様）
  const [preferences, setPreferences] = useState<EvaluationPreferences>({
    // 基本項目（5項目）
    research_intensity: 7.0,
    advisor_style: 6.0,
    team_work: 7.0,
    workload: 6.0,
    theory_practice: 7.0,

    // 拡張項目（5項目）
    research_field_match: 8.0,
    skill_development: 7.0,
    lab_atmosphere: 7.0,
    flexibility: 7.0,
    publication_opportunity: 8.0,

    // 特殊項目（3項目）
    interdisciplinary: 6.0,
    communication_style: 6.0,
    innovation_risk: 6.0,
  });

  // 研究分野の興味度
  const [fieldInterests, setFieldInterests] = useState<ResearchFieldInterests>({
    "人工知能・機械学習": 7.0,
    "画像・映像処理": 5.0,
    "コンピュータネットワーク・セキュリティ": 4.0,
    "データベース・情報システム": 5.0,
    "組込み・IoT": 4.0,
    "Webデザイン・UI/UX": 6.0,
    "デザイン・視覚表現": 5.0,
    "映像・アニメーション": 4.0,
    "コンピュータ音楽・サウンドアート": 3.0,
    "ゲーム開発・eスポーツ": 6.0,
    "VR/AR・メディアアート": 5.0
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeStep, setActiveStep] = useState(0);

  // カテゴリ別の評価基準情報
  const criteriaCategories = {
    basic: {
      title: '基本的な研究環境',
      icon: <School />,
      description: '研究活動の基本的な条件・環境に関する設定',
      items: ['research_intensity', 'advisor_style', 'team_work', 'workload', 'theory_practice']
    },
    extended: {
      title: '研究の詳細条件',
      icon: <TrendingUp />,
      description: '研究の専門性や成果に関する詳細な条件',
      items: ['research_field_match', 'skill_development', 'lab_atmosphere', 'flexibility', 'publication_opportunity']
    },
    special: {
      title: '特殊な研究アプローチ',
      icon: <Lightbulb />,
      description: '革新性や学際性など特殊な研究アプローチに関する設定',
      items: ['interdisciplinary', 'communication_style', 'innovation_risk']
    }
  };

  const getCriteriaIcon = (criteriaKey: string) => {
    const iconMap: { [key: string]: React.ReactElement } = {
      research_intensity: <Science />,
      advisor_style: <School />,
      team_work: <Groups />,
      workload: <Schedule />,
      theory_practice: <Psychology />,
      research_field_match: <Explore />,
      skill_development: <TrendingUp />,
      lab_atmosphere: <Groups />,
      flexibility: <AccessTime />,
      publication_opportunity: <Article />,
      interdisciplinary: <AccountTree />,
      communication_style: <ConnectWithoutContact />,
      innovation_risk: <Lightbulb />
    };
    return iconMap[criteriaKey] || <Settings />;
  };

  const handlePreferenceChange = (criteria: keyof EvaluationPreferences, value: number) => {
    setPreferences(prev => ({
      ...prev,
      [criteria]: value
    }));
  };

  const handleFieldInterestChange = (field: keyof ResearchFieldInterests, value: number) => {
    setFieldInterests(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);

    try {
      const studentProfile: StudentProfile = {
        preferences,
        field_interests: fieldInterests,
        metadata: {
          timestamp: Date.now(),
          session_id: `session_${Math.random().toString(36).substr(2, 9)}`
        }
      };

      const response = await apiService.evaluateLabs(studentProfile);
      onResults(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : '評価処理でエラーが発生しました');
    } finally {
      setLoading(false);
    }
  };

  const renderCriteriaSlider = (criteriaKey: keyof EvaluationPreferences) => {
    const info = CRITERIA_INFO[criteriaKey];
    const value = preferences[criteriaKey];

    return (
      <Grid item xs={12} md={6} key={criteriaKey}>
        <Card sx={{ p: 2, height: '100%' }}>
          <Box display="flex" alignItems="center" gap={1} mb={1}>
            {getCriteriaIcon(criteriaKey)}
            <Typography variant="h6" component="h3">
              {info.label}
            </Typography>
          </Box>

          <Typography variant="body2" color="text.secondary" mb={2}>
            {info.description}
          </Typography>

          <Box sx={{ px: 1 }}>
            <Slider
              value={value}
              onChange={(_, newValue) => handlePreferenceChange(criteriaKey, newValue as number)}
              min={1}
              max={10}
              step={0.5}
              marks={[
                { value: 1, label: '1' },
                { value: 5, label: '5' },
                { value: 10, label: '10' }
              ]}
              valueLabelDisplay="on"
              sx={{ mb: 1 }}
            />
          </Box>

          <Typography variant="caption" color="text.secondary">
            {info.range}
          </Typography>

          <Box mt={1}>
            <Chip
              label={`現在値: ${value}`}
              size="small"
              color={value >= 7 ? 'primary' : value >= 4 ? 'default' : 'secondary'}
            />
          </Box>
        </Card>
      </Grid>
    );
  };

  const renderCriteriaCategory = (categoryKey: keyof typeof criteriaCategories) => {
    const category = criteriaCategories[categoryKey];

    return (
      <Accordion key={categoryKey} defaultExpanded>
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Box display="flex" alignItems="center" gap={1}>
            {category.icon}
            <Box>
              <Typography variant="h6">{category.title}</Typography>
              <Typography variant="body2" color="text.secondary">
                {category.description}
              </Typography>
            </Box>
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Grid container spacing={3}>
            {category.items.map(criteriaKey =>
              renderCriteriaSlider(criteriaKey as keyof EvaluationPreferences)
            )}
          </Grid>
        </AccordionDetails>
      </Accordion>
    );
  };

  const renderFieldInterests = () => {
    return (
      <Card sx={{ p: 3 }}>
        <Box display="flex" alignItems="center" gap={1} mb={2}>
          <Explore />
          <Typography variant="h5">研究分野への興味度</Typography>
        </Box>

        <Typography variant="body2" color="text.secondary" mb={3}>
          各研究分野への興味の強さを設定してください（1: 興味なし 〜 10: 非常に興味あり）
        </Typography>

        {Object.entries(FIELD_CATEGORIES).map(([categoryName, fields]) => (
          <Box key={categoryName} mb={4}>
            <Typography variant="h6" gutterBottom color="primary">
              {categoryName}分野
            </Typography>

            <Grid container spacing={2}>
              {fields.map((field) => (
                <Grid item xs={12} md={6} key={field}>
                  <Box sx={{ p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
                    <Typography variant="subtitle1" gutterBottom>
                      {field}
                    </Typography>

                    <Box sx={{ px: 1 }}>
                      <Slider
                        value={fieldInterests[field as keyof ResearchFieldInterests]}
                        onChange={(_, newValue) =>
                          handleFieldInterestChange(field as keyof ResearchFieldInterests, newValue as number)
                        }
                        min={1}
                        max={10}
                        step={0.5}
                        marks={[
                          { value: 1, label: '1' },
                          { value: 5, label: '5' },
                          { value: 10, label: '10' }
                        ]}
                        valueLabelDisplay="on"
                      />
                    </Box>

                    <Chip
                      label={`興味度: ${fieldInterests[field as keyof ResearchFieldInterests]}`}
                      size="small"
                      color={fieldInterests[field as keyof ResearchFieldInterests] >= 7 ? 'primary' : 'default'}
                    />
                  </Box>
                </Grid>
              ))}
            </Grid>
          </Box>
        ))}
      </Card>
    );
  };

  return (
    <Box>
      <Card sx={{ p: 3, mb: 3 }}>
        <Box display="flex" alignItems="center" gap={2} mb={2}>
          <Science color="primary" />
          <Typography variant="h4" component="h1">
            研究室選択支援システム
          </Typography>
        </Box>

        <Typography variant="body1" color="text.secondary" mb={2}>
          あなたの研究に対する価値観と興味を入力して、最適な研究室を見つけましょう。
          13の評価基準と11の研究分野から総合的に判定します。
        </Typography>

        <Box display="flex" gap={1} flexWrap="wrap">
          <Chip icon={<School />} label="13項目評価" color="primary" />
          <Chip icon={<Explore />} label="11研究分野" color="secondary" />
          <Chip icon={<Psychology />} label="ファジィ推論" />
          <Chip icon={<TrendingUp />} label="遺伝的アルゴリズム" />
        </Box>
      </Card>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Box mb={3}>
        <Typography variant="h5" gutterBottom>
          📊 評価基準の設定（13項目）
        </Typography>
        <Typography variant="body2" color="text.secondary" mb={2}>
          研究室選択で重視する項目について、あなたの価値観を設定してください
        </Typography>

        {Object.keys(criteriaCategories).map(categoryKey =>
          renderCriteriaCategory(categoryKey as keyof typeof criteriaCategories)
        )}
      </Box>

      <Box mb={3}>
        {renderFieldInterests()}
      </Box>

      <Paper sx={{ p: 3, textAlign: 'center' }}>
        <Button
          variant="contained"
          size="large"
          onClick={handleSubmit}
          disabled={loading}
          startIcon={loading ? <CircularProgress size={20} /> : <Science />}
          sx={{ minWidth: 200 }}
        >
          {loading ? '評価中...' : '研究室を評価する'}
        </Button>

        {loading && (
          <Typography variant="body2" color="text.secondary" mt={2}>
            ファジィ決定木と遺伝的アルゴリズムを使用して最適な研究室を分析中...
          </Typography>
        )}
      </Paper>
    </Box>
  );
};

export default EvaluationForm;