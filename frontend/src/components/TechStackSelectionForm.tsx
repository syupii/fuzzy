import React, { useState } from 'react';
import {
  Box,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  FormControlLabel,
  Checkbox,
  Chip,
  Grid,
  Card,
  CardContent,
  Button,
  ButtonGroup,
  Paper,
  Slider,
  TextField,
  Autocomplete,
  Alert,
  LinearProgress,
  Tooltip,
} from '@mui/material';
import {
  ExpandMore,
  Code,
  Web,
  Psychology,
  Computer,
} from '@mui/icons-material';
import {
  PROGRAMMING_LANGUAGES,
  TECH_FRAMEWORKS,
  ProgrammingLanguage,
  TechFramework,
  TechStackPreference,
  languageUtils,
  techUtils,
} from '../services/api';

interface TechStackSelectionFormProps {
  preferences: TechStackPreference;
  onPreferencesChange: (preferences: TechStackPreference) => void;
  onSubmit: () => void;
}

const TechStackSelectionForm: React.FC<TechStackSelectionFormProps> = ({
  preferences,
  onPreferencesChange,
  onSubmit,
}) => {
  const [expandedSections, setExpandedSections] = useState<string[]>(['languages']);
  const [filterDifficulty, setFilterDifficulty] = useState<string[]>([]);
  const [filterDemand, setFilterDemand] = useState<string[]>([]);

  // 難易度の色分け
  const getDifficultyColor = (difficulty: string) => {
    const colorMap: { [key: string]: string } = {
      beginner: '#4caf50',
      intermediate: '#ff9800',
      advanced: '#f44336',
    };
    return colorMap[difficulty] || '#757575';
  };

  // 市場需要の色分け
  const getDemandColor = (demand: string) => {
    const colorMap: { [key: string]: string } = {
      high: '#4caf50',
      medium: '#ff9800',
      low: '#757575',
    };
    return colorMap[demand] || '#757575';
  };

  // セクション展開/折りたたみ
  const handleSectionToggle = (section: string) => {
    setExpandedSections(prev =>
      prev.includes(section)
        ? prev.filter(s => s !== section)
        : [...prev, section]
    );
  };

  // 言語選択の変更
  const handleLanguageToggle = (languageId: string) => {
    const newLanguages = preferences.languagePreferences.includes(languageId)
      ? preferences.languagePreferences.filter(id => id !== languageId)
      : [...preferences.languagePreferences, languageId];
    
    onPreferencesChange({
      ...preferences,
      languagePreferences: newLanguages,
    });
  };

  // フレームワーク経験の変更
  const handleFrameworkToggle = (frameworkId: string) => {
    const newFrameworks = preferences.frameworkExperience.includes(frameworkId)
      ? preferences.frameworkExperience.filter(id => id !== frameworkId)
      : [...preferences.frameworkExperience, frameworkId];
    
    onPreferencesChange({
      ...preferences,
      frameworkExperience: newFrameworks,
    });
  };

  // 学習意欲の変更
  const handleLearningWillingnessChange = (value: number) => {
    onPreferencesChange({
      ...preferences,
      learningWillingness: value,
    });
  };

  // キャリア目標の変更
  const handleCareerGoalsChange = (goals: string[]) => {
    onPreferencesChange({
      ...preferences,
      careerGoals: goals,
    });
  };

  // フィルタリング
  const getFilteredLanguages = () => {
    return PROGRAMMING_LANGUAGES.filter(lang => {
      if (filterDifficulty.length > 0 && !filterDifficulty.includes(lang.difficulty)) {
        return false;
      }
      if (filterDemand.length > 0 && !filterDemand.includes(lang.marketDemand)) {
        return false;
      }
      return true;
    });
  };

  // キャリア目標の選択肢
  const careerGoalOptions = [
    'Webエンジニア',
    'モバイルエンジニア',
    'バックエンドエンジニア',
    'フロントエンドエンジニア',
    'フルスタックエンジニア',
    'データサイエンティスト',
    'AIエンジニア',
    'MLエンジニア',
    'DevOpsエンジニア',
    'インフラエンジニア',
    'セキュリティエンジニア',
    'ゲーム開発者',
    'システムエンジニア',
    '研究者・アカデミック',
    'ITコンサルタント',
    'プロダクトマネージャー',
    '起業家・CTO',
    '技術営業',
  ];

  // 統計情報
  const getStats = () => {
    return {
      selectedLanguages: preferences.languagePreferences.length,
      selectedFrameworks: preferences.frameworkExperience.length,
      learningWillingness: preferences.learningWillingness,
      careerGoals: preferences.careerGoals.length,
    };
  };

  const stats = getStats();

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3, textAlign: 'center' }}>
        💻 技術スタック・キャリア選択
      </Typography>

      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body2">
          習得したい・経験のあるプログラミング言語、技術フレームワーク、
          およびキャリア目標を選択してください。これらの情報により、
          あなたの技術的興味と将来の目標に最も適した研究室を見つけることができます。
        </Typography>
      </Alert>

      {/* 統計情報 */}
      <Paper sx={{ p: 2, mb: 3, bgcolor: '#f5f5f5' }}>
        <Grid container spacing={3}>
          <Grid item xs={6} sm={3}>
            <Box textAlign="center">
              <Typography variant="h6" color="primary">
                {stats.selectedLanguages}
              </Typography>
              <Typography variant="body2">選択言語</Typography>
            </Box>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Box textAlign="center">
              <Typography variant="h6" color="secondary">
                {stats.selectedFrameworks}
              </Typography>
              <Typography variant="body2">経験技術</Typography>
            </Box>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Box textAlign="center">
              <Typography variant="h6" color="error">
                {stats.learningWillingness}/10
              </Typography>
              <Typography variant="body2">学習意欲</Typography>
            </Box>
          </Grid>
          <Grid item xs={6} sm={3}>
            <Box textAlign="center">
              <Typography variant="h6" color="success.main">
                {stats.careerGoals}
              </Typography>
              <Typography variant="body2">キャリア目標</Typography>
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {/* プログラミング言語選択 */}
      <Accordion
        expanded={expandedSections.includes('languages')}
        onChange={() => handleSectionToggle('languages')}
        sx={{ mb: 2 }}
      >
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Code />
            <Typography variant="h6">プログラミング言語</Typography>
            <Chip
              label={`${stats.selectedLanguages}/${PROGRAMMING_LANGUAGES.length}`}
              size="small"
              color="primary"
              variant="outlined"
            />
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          {/* フィルター */}
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle2" gutterBottom>フィルター:</Typography>
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <Box>
                <Typography variant="body2" gutterBottom>難易度:</Typography>
                <ButtonGroup size="small">
                  {['beginner', 'intermediate', 'advanced'].map(difficulty => (
                    <Button
                      key={difficulty}
                      variant={filterDifficulty.includes(difficulty) ? 'contained' : 'outlined'}
                      onClick={() => {
                        setFilterDifficulty(prev =>
                          prev.includes(difficulty)
                            ? prev.filter(d => d !== difficulty)
                            : [...prev, difficulty]
                        );
                      }}
                      sx={{ fontSize: '0.75rem' }}
                    >
                      {difficulty === 'beginner' ? '初級' : 
                       difficulty === 'intermediate' ? '中級' : '上級'}
                    </Button>
                  ))}
                </ButtonGroup>
              </Box>
              <Box>
                <Typography variant="body2" gutterBottom>市場需要:</Typography>
                <ButtonGroup size="small">
                  {['high', 'medium', 'low'].map(demand => (
                    <Button
                      key={demand}
                      variant={filterDemand.includes(demand) ? 'contained' : 'outlined'}
                      onClick={() => {
                        setFilterDemand(prev =>
                          prev.includes(demand)
                            ? prev.filter(d => d !== demand)
                            : [...prev, demand]
                        );
                      }}
                      sx={{ fontSize: '0.75rem' }}
                    >
                      {demand === 'high' ? '高需要' : demand === 'medium' ? '中需要' : '低需要'}
                    </Button>
                  ))}
                </ButtonGroup>
              </Box>
            </Box>
          </Box>

          {/* 言語選択カード */}
          <Grid container spacing={2}>
            {getFilteredLanguages().map(language => {
              const isSelected = preferences.languagePreferences.includes(language.id);
              
              return (
                <Grid item xs={12} sm={6} md={4} key={language.id}>
                  <Card 
                    variant="outlined" 
                    sx={{ 
                      height: '100%',
                      border: isSelected ? '2px solid #1976d2' : '1px solid #e0e0e0',
                      bgcolor: isSelected ? '#f3f7ff' : 'white',
                      cursor: 'pointer',
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        transform: 'translateY(-2px)',
                        boxShadow: 2,
                      }
                    }}
                    onClick={() => handleLanguageToggle(language.id)}
                  >
                    <CardContent sx={{ p: 2 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={isSelected}
                              onChange={() => handleLanguageToggle(language.id)}
                              color="primary"
                            />
                          }
                          label={
                            <Typography variant="subtitle1" fontWeight="bold">
                              {language.name}
                            </Typography>
                          }
                          onClick={(e) => e.stopPropagation()}
                        />
                      </Box>
                      
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                        カテゴリ: {language.category}
                      </Typography>

                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Chip
                          label={language.difficulty === 'beginner' ? '初級' : 
                                language.difficulty === 'intermediate' ? '中級' : '上級'}
                          size="small"
                          sx={{ 
                            bgcolor: getDifficultyColor(language.difficulty),
                            color: 'white',
                            fontSize: '0.7rem'
                          }}
                        />
                        <Chip
                          label={`需要: ${language.marketDemand === 'high' ? '高' : 
                                language.marketDemand === 'medium' ? '中' : '低'}`}
                          size="small"
                          sx={{ 
                            bgcolor: getDemandColor(language.marketDemand),
                            color: 'white',
                            fontSize: '0.7rem'
                          }}
                        />
                      </Box>

                      <Typography variant="body2" sx={{ mb: 1 }}>
                        学習コスト: {language.learningCurve}/10
                      </Typography>
                      <LinearProgress 
                        variant="determinate" 
                        value={language.learningCurve * 10} 
                        color={language.learningCurve <= 3 ? 'success' : 
                               language.learningCurve <= 6 ? 'warning' : 'error'}
                        sx={{ mb: 2 }}
                      />

                      <Typography variant="body2" gutterBottom>主な用途:</Typography>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {language.applications.slice(0, 3).map(app => (
                          <Chip
                            key={app}
                            label={app}
                            size="small"
                            variant="outlined"
                            sx={{ fontSize: '0.65rem' }}
                          />
                        ))}
                        {language.applications.length > 3 && (
                          <Tooltip title={language.applications.slice(3).join(', ')}>
                            <Chip
                              label={`+${language.applications.length - 3}`}
                              size="small"
                              variant="outlined"
                              sx={{ fontSize: '0.65rem' }}
                            />
                          </Tooltip>
                        )}
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              );
            })}
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* 技術フレームワーク・ツール */}
      <Accordion
        expanded={expandedSections.includes('frameworks')}
        onChange={() => handleSectionToggle('frameworks')}
        sx={{ mb: 2 }}
      >
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Web />
            <Typography variant="h6">技術フレームワーク・ツール</Typography>
            <Chip
              label={`${stats.selectedFrameworks}/${TECH_FRAMEWORKS.length}`}
              size="small"
              color="secondary"
              variant="outlined"
            />
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            経験がある、または学びたい技術フレームワークやツールを選択してください。
          </Typography>
          
          <Grid container spacing={2}>
            {TECH_FRAMEWORKS.map(framework => {
              const isSelected = preferences.frameworkExperience.includes(framework.id);
              
              return (
                <Grid item xs={12} sm={6} md={4} key={framework.id}>
                  <Card 
                    variant="outlined" 
                    sx={{ 
                      border: isSelected ? '2px solid #9c27b0' : '1px solid #e0e0e0',
                      bgcolor: isSelected ? '#f8f3ff' : 'white',
                      cursor: 'pointer',
                      transition: 'all 0.3s ease',
                      '&:hover': {
                        transform: 'translateY(-2px)',
                        boxShadow: 2,
                      }
                    }}
                    onClick={() => handleFrameworkToggle(framework.id)}
                  >
                    <CardContent sx={{ p: 2 }}>
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                        <FormControlLabel
                          control={
                            <Checkbox
                              checked={isSelected}
                              onChange={() => handleFrameworkToggle(framework.id)}
                              color="secondary"
                            />
                          }
                          label={
                            <Typography variant="subtitle1" fontWeight="bold">
                              {framework.name}
                            </Typography>
                          }
                          onClick={(e) => e.stopPropagation()}
                        />
                      </Box>
                      
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                        {framework.category}
                        {framework.language && ` • ${framework.language}`}
                      </Typography>

                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                        <Chip
                          label={framework.difficulty === 'beginner' ? '初級' : 
                                framework.difficulty === 'intermediate' ? '中級' : '上級'}
                          size="small"
                          sx={{ 
                            bgcolor: getDifficultyColor(framework.difficulty),
                            color: 'white',
                            fontSize: '0.7rem'
                          }}
                        />
                        <Chip
                          label={`人気度: ${framework.popularity}/10`}
                          size="small"
                          color="info"
                          variant="outlined"
                          sx={{ fontSize: '0.7rem' }}
                        />
                      </Box>

                      <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
                        学習リソース: {
                          framework.learningResources === 'abundant' ? '豊富' :
                          framework.learningResources === 'moderate' ? '普通' : '限定的'
                        }
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              );
            })}
          </Grid>
        </AccordionDetails>
      </Accordion>

      {/* 学習意欲 */}
      <Accordion
        expanded={expandedSections.includes('learning')}
        onChange={() => handleSectionToggle('learning')}
        sx={{ mb: 2 }}
      >
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Computer />
            <Typography variant="h6">学習意欲・技術習得</Typography>
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Typography variant="body1" gutterBottom>
            新しい技術を学ぶ意欲はどの程度ですか？
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            現在の値: {preferences.learningWillingness}/10
          </Typography>
          <Slider
            value={preferences.learningWillingness}
            onChange={(_, value) => handleLearningWillingnessChange(value as number)}
            min={1}
            max={10}
            step={1}
            marks={[
              { value: 1, label: '低い' },
              { value: 5, label: '普通' },
              { value: 10, label: '非常に高い' }
            ]}
            valueLabelDisplay="auto"
            color="error"
          />
        </AccordionDetails>
      </Accordion>

      {/* キャリア目標 */}
      <Accordion
        expanded={expandedSections.includes('career')}
        onChange={() => handleSectionToggle('career')}
        sx={{ mb: 2 }}
      >
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Psychology />
            <Typography variant="h6">キャリア目標</Typography>
            <Chip
              label={`${stats.careerGoals}選択`}
              size="small"
              color="success"
              variant="outlined"
            />
          </Box>
        </AccordionSummary>
        <AccordionDetails>
          <Autocomplete
            multiple
            options={careerGoalOptions}
            value={preferences.careerGoals}
            onChange={(_, value) => handleCareerGoalsChange(value)}
            renderTags={(value, getTagProps) =>
              value.map((option, index) => (
                <Chip variant="outlined" label={option} {...getTagProps({ index })} key={option} />
              ))
            }
            renderInput={(params) => (
              <TextField
                {...params}
                variant="outlined"
                label="将来のキャリア目標"
                placeholder="目標を選択または入力してください"
                helperText="複数選択可能です。あなたの将来の目標に合った研究室を見つけるのに役立ちます。"
              />
            )}
          />
        </AccordionDetails>
      </Accordion>

      {/* 送信ボタン */}
      <Box sx={{ textAlign: 'center', mt: 4 }}>
        <Button
          variant="contained"
          size="large"
          onClick={onSubmit}
          disabled={stats.selectedLanguages === 0 && stats.careerGoals === 0}
          sx={{ minWidth: 200, py: 1.5 }}
        >
          選択完了
        </Button>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          言語またはキャリア目標を最低1つ選択してください
        </Typography>
      </Box>
    </Box>
  );
};

export default TechStackSelectionForm;