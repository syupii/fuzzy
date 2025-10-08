// frontend/src/components/EvaluationForm.tsx - 優先度対応版
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
  FormGroup,
  Divider
} from '@mui/material';
import {
  ExpandMore,
  Science,
  Palette,
  SportsEsports,
  Psychology,
  School,
  Timeline,
  Star,
  TrendingUp
} from '@mui/icons-material';

// 型定義の拡張
interface EvaluationPreferencesWithPriority {
  // 評価値（1-10）
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

  // 優先度（1-10）
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
  onResults: (response: any) => void;
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

// 評価基準の詳細情報
const CRITERIA_INFO = {
  research_intensity: {
    name: '研究強度',
    description: '研究にどれだけ集中的に取り組みたいか',
    range: '1（軽い研究）〜 10（集中研究）'
  },
  advisor_style: {
    name: '指導スタイル',
    description: '教授からの指導の受け方の好み',
    range: '1（厳格指導）〜 10（自由指導）'
  },
  team_work: {
    name: 'チームワーク',
    description: '研究での他者との協働の程度',
    range: '1（個人研究）〜 10（チーム研究）'
  },
  workload: {
    name: 'ワークロード',
    description: '研究活動の忙しさに対する許容度',
    range: '1（軽い負荷）〜 10（重い負荷）'
  },
  theory_practice: {
    name: '理論・実践バランス',
    description: '理論研究と実践的研究のバランス',
    range: '1（理論重視）〜 10（実践重視）'
  },
  research_field_match: {
    name: '研究分野適合性',
    description: '自分の興味と研究室の分野の一致度',
    range: '1（広い分野）〜 10（専門特化）'
  },
  skill_development: {
    name: 'スキル開発',
    description: '専門性と汎用性のバランス',
    range: '1（専門特化）〜 10（幅広いスキル）'
  },
  lab_atmosphere: {
    name: '研究室雰囲気',
    description: '研究室の全体的な雰囲気',
    range: '1（静寂集中）〜 10（活発議論）'
  },
  flexibility: {
    name: '柔軟性',
    description: '研究時間の自由度',
    range: '1（固定スケジュール）〜 10（柔軟スケジュール）'
  },
  publication_opportunity: {
    name: '論文発表機会',
    description: '研究成果の論文化機会',
    range: '1（少ない機会）〜 10（豊富な機会）'
  },
  interdisciplinary: {
    name: '学際性',
    description: '他分野との連携の程度',
    range: '1（単一分野）〜 10（学際連携）'
  },
  communication_style: {
    name: 'コミュニケーション',
    description: '研究室での交流スタイル',
    range: '1（少人数密接）〜 10（オープン交流）'
  }
};

// 研究分野データ（簡略版）
const RESEARCH_FIELDS = [
  { id: 'ai_ml', name: '人工知能・機械学習', category: 'テクノロジー・システム' },
  { id: 'image_processing', name: '画像・映像処理', category: 'テクノロジー・システム' },
  { id: 'network_security', name: 'ネットワーク・セキュリティ', category: 'テクノロジー・システム' },
  { id: 'database_systems', name: 'データベース・情報システム', category: 'テクノロジー・システム' },
  { id: 'embedded_iot', name: '組込み・IoT', category: 'テクノロジー・システム' },
  { id: 'education_linguistics', name: '教育・言語学', category: 'テクノロジー・システム' },
  { id: 'natural_science_math', name: '自然科学・数理', category: 'テクノロジー・システム' },
  { id: 'medical_healthcare', name: '医療情報・ヘルスケア', category: 'テクノロジー・システム' },
  { id: 'tourism_regional', name: '観光情報・地域システム', category: 'テクノロジー・システム' },
  { id: 'business_decision', name: '経営情報・意思決定支援', category: 'テクノロジー・システム' },
  { id: 'audio_processing', name: '音声・音響情報処理', category: 'テクノロジー・システム' },
  { id: 'system_ethics', name: 'システム運用・情報倫理', category: 'テクノロジー・システム' },
  { id: 'web_design', name: 'Webデザイン・UI/UX', category: 'クリエイティブ' },
  { id: 'design_visual', name: 'デザイン・視覚表現', category: 'クリエイティブ' },
  { id: 'video_animation', name: '映像・アニメーション', category: 'クリエイティブ' },
  { id: 'computer_music', name: 'コンピュータ音楽・サウンドアート', category: 'クリエイティブ' },
  { id: 'game_esports', name: 'ゲーム開発・eスポーツ', category: 'エンターテイメント' },
  { id: 'vr_ar_media', name: 'VR/AR・メディアアート', category: 'エンターテイメント' },
  { id: 'philosophy_humanities', name: '哲学・人文・環境行動学', category: '人文・社会・体育' },
  { id: 'sports_science', name: 'スポーツ・体育科学', category: '人文・社会・体育' }
];

const EvaluationForm: React.FC<EvaluationFormProps> = ({ onResults, onError }) => {
  const [tabValue, setTabValue] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string>('');

  // 評価基準の状態（値 + 優先度）
  const [preferences, setPreferences] = useState<EvaluationPreferencesWithPriority>({
    // 評価値（デフォルト: 5）
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

    // 優先度（デフォルト: 5）
    research_intensity_priority: 5,
    advisor_style_priority: 5,
    team_work_priority: 5,
    workload_priority: 5,
    theory_practice_priority: 5,
    research_field_match_priority: 5,
    skill_development_priority: 5,
    lab_atmosphere_priority: 5,
    flexibility_priority: 5,
    publication_opportunity_priority: 5,
    interdisciplinary_priority: 5,
    communication_style_priority: 5
  });

  // 研究分野選択の状態
  const [selectedFields, setSelectedFields] = useState<Set<string>>(new Set());
  const [fieldInterests, setFieldInterests] = useState<ResearchFieldInterests>({});

  // 評価基準変更ハンドラー
  const handlePreferenceChange = (key: keyof EvaluationPreferencesWithPriority, value: number) => {
    setPreferences(prev => ({
      ...prev,
      [key]: value
    }));
  };

  // 研究分野選択ハンドラー
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
        setFieldInterests(prev => ({
          ...prev,
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
  const handleLoadDemo = () => {
    setPreferences({
      research_intensity: 8, research_intensity_priority: 9,
      advisor_style: 6, advisor_style_priority: 7,
      team_work: 7, team_work_priority: 8,
      workload: 6, workload_priority: 5,
      theory_practice: 7, theory_practice_priority: 6,
      research_field_match: 9, research_field_match_priority: 10,
      skill_development: 8, skill_development_priority: 7,
      lab_atmosphere: 7, lab_atmosphere_priority: 6,
      flexibility: 8, flexibility_priority: 8,
      publication_opportunity: 9, publication_opportunity_priority: 9,
      interdisciplinary: 6, interdisciplinary_priority: 5,
      communication_style: 7, communication_style_priority: 6
    });
    setSelectedFields(new Set(['ai_ml', 'image_processing']));
    setFieldInterests({ 'ai_ml': 9, 'image_processing': 7 });
  };

  // 評価実行
  const handleEvaluate = async () => {
    setIsLoading(true);
    setError('');

    try {
      // バックエンド用データ構築
      const studentProfile = {
        // 基本の評価値
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

        // 優先度（新規追加）
        priorities: {
          research_intensity: preferences.research_intensity_priority,
          advisor_style: preferences.advisor_style_priority,
          team_work: preferences.team_work_priority,
          workload: preferences.workload_priority,
          theory_practice: preferences.theory_practice_priority,
          research_field_match: preferences.research_field_match_priority,
          skill_development: preferences.skill_development_priority,
          lab_atmosphere: preferences.lab_atmosphere_priority,
          flexibility: preferences.flexibility_priority,
          publication_opportunity: preferences.publication_opportunity_priority,
          interdisciplinary: preferences.interdisciplinary_priority,
          communication_style: preferences.communication_style_priority
        },

        // 研究分野興味
        field_interests: fieldInterests
      };

      const evaluationData = {
        student_profile: studentProfile
      };

      console.log('送信データ（優先度対応）:', evaluationData);

      // API呼び出し
      const response = await fetch('/api/evaluate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(evaluationData)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      onResults(result);
      setTabValue(2);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : '評価処理でエラーが発生しました';
      setError(errorMessage);
      onError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  // 基本評価基準コンポーネント（優先度対応）
  const renderBasicCriteria = () => (
    <Box>
      <Typography variant="h6" gutterBottom>
        評価基準設定（優先度対応）
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        各項目について「希望値」と「重要度（優先度）」を設定してください
      </Typography>

      <Grid container spacing={3}>
        {Object.entries(CRITERIA_INFO).map(([key, info]) => (
          <Grid item xs={12} key={key}>
            <Card variant="outlined" sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom color="primary">
                {info.name}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                {info.description}
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ mb: 3, display: 'block' }}>
                {info.range}
              </Typography>

              <Grid container spacing={4}>
                {/* 希望値スライダー */}
                <Grid item xs={12} md={6}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <TrendingUp sx={{ mr: 1, fontSize: 20, color: 'primary.main' }} />
                    <Typography variant="subtitle2">
                      希望値: {preferences[key as keyof EvaluationPreferencesWithPriority]}
                    </Typography>
                  </Box>
                  <Slider
                    value={preferences[key as keyof EvaluationPreferencesWithPriority]}
                    onChange={(_, value) => handlePreferenceChange(key as keyof EvaluationPreferencesWithPriority, value as number)}
                    min={1}
                    max={10}
                    step={1}
                    marks
                    valueLabelDisplay="on"
                    sx={{ mt: 1 }}
                  />
                </Grid>

                {/* 優先度スライダー */}
                <Grid item xs={12} md={6}>
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                    <Star sx={{ mr: 1, fontSize: 20, color: 'warning.main' }} />
                    <Typography variant="subtitle2">
                      重要度: {preferences[`${key}_priority` as keyof EvaluationPreferencesWithPriority]}
                    </Typography>
                  </Box>
                  <Slider
                    value={preferences[`${key}_priority` as keyof EvaluationPreferencesWithPriority]}
                    onChange={(_, value) => handlePreferenceChange(`${key}_priority` as keyof EvaluationPreferencesWithPriority, value as number)}
                    min={1}
                    max={10}
                    step={1}
                    marks
                    valueLabelDisplay="on"
                    sx={{
                      mt: 1,
                      '& .MuiSlider-thumb': {
                        color: 'warning.main',
                      },
                      '& .MuiSlider-track': {
                        color: 'warning.main',
                      },
                      '& .MuiSlider-rail': {
                        color: 'warning.light',
                      }
                    }}
                  />
                </Grid>
              </Grid>

              <Divider sx={{ my: 2 }} />

              {/* 重み付けされた値の表示 */}
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body2" color="text.secondary">
                  重み付きスコア:
                </Typography>
                <Typography variant="body2" color="primary" fontWeight="bold">
                  {(preferences[key as keyof EvaluationPreferencesWithPriority] *
                    preferences[`${key}_priority` as keyof EvaluationPreferencesWithPriority] / 10).toFixed(1)}
                </Typography>
              </Box>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );

  // 研究分野選択コンポーネント
  const renderFieldInterests = () => {
    const categoryIcons: { [key: string]: React.ReactNode } = {
      'テクノロジー・システム': <Science color="primary" />,
      'クリエイティブ': <Palette color="secondary" />,
      'エンターテイメント': <SportsEsports color="error" />,
      '人文・社会・体育': <Psychology color="warning" />
    };

    // Setイテレーションの問題を修正
    const categorySet = new Set(RESEARCH_FIELDS.map(f => f.category));
    const categories = Array.from(categorySet);

    return (
      <Box>
        <Typography variant="h6" gutterBottom>
          研究分野選択
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          興味のある研究分野を選択し、興味度を設定してください
        </Typography>

        {categories.map(category => (
          <Accordion key={category} defaultExpanded>
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Box sx={{ display: 'flex', alignItems: 'center' }}>
                {categoryIcons[category]}
                <Typography variant="h6" sx={{ ml: 1 }}>
                  {category}
                </Typography>
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                {RESEARCH_FIELDS.filter(f => f.category === category).map(field => (
                  <Grid item xs={12} md={6} key={field.id}>
                    <Card
                      variant="outlined"
                      sx={{
                        p: 2,
                        border: selectedFields.has(field.id) ? 2 : 1,
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
                            min={1}
                            max={10}
                            step={1}
                            marks
                            valueLabelDisplay="on"
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

  // 評価実行コンポーネント
  const renderEvaluationExecute = () => {
    const selectedFieldsCount = selectedFields.size;
    const totalWeightedScore = Object.keys(CRITERIA_INFO).reduce((sum, key) => {
      const value = preferences[key as keyof EvaluationPreferencesWithPriority];
      const priority = preferences[`${key}_priority` as keyof EvaluationPreferencesWithPriority];
      return sum + (value * priority / 10);
    }, 0);

    return (
      <Box>
        <Typography variant="h6" gutterBottom>
          研究室マッチング実行
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          設定した条件（優先度対応）に基づいて研究室マッチングを実行します
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        <Card sx={{ p: 3, mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            設定サマリー（優先度対応）
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={4}>
              <Typography variant="body2">
                評価基準: {Object.keys(CRITERIA_INFO).length}項目設定済み
              </Typography>
            </Grid>
            <Grid item xs={4}>
              <Typography variant="body2">
                研究分野: {selectedFieldsCount}分野選択中
              </Typography>
            </Grid>
            <Grid item xs={4}>
              <Typography variant="body2" color="primary" fontWeight="bold">
                総合重み付きスコア: {totalWeightedScore.toFixed(1)}
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
            {isLoading ? '評価中...' : '研究室マッチング実行（優先度対応）'}
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
              優先度を考慮したAI研究室適合性を評価中...
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
        <Tab label="研究分野選択" />
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