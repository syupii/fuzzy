import React, { useState, useEffect } from 'react';
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
  FormControlLabel,
  Checkbox,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Tooltip,
  Badge,
} from '@mui/material';
import {
  Science,
  Psychology,
  TrendingUp,
  ExpandMore,
  AutoAwesome,
  Assessment,
  Category,
} from '@mui/icons-material';
import {
  apiService,
  EnhancedEvaluationPreferences,
  EvaluationResponse,
  RESEARCH_FIELDS,
  fieldUtils,
  ResearchField,
} from '../services/api';

interface EnhancedEvaluationFormProps {
  onResults: (results: EvaluationResponse) => void;
}

const EnhancedEvaluationForm: React.FC<EnhancedEvaluationFormProps> = ({ onResults }) => {
  const [preferences, setPreferences] = useState<EnhancedEvaluationPreferences>({
    research_intensity: 7.0,
    advisor_style: 6.0,
    team_work: 7.0,
    workload: 6.0,
    theory_practice: 7.0,
    research_field_interests: {},
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showFieldSelection, setShowFieldSelection] = useState(false);
  const [fieldRecommendations, setFieldRecommendations] = useState<string[]>([]);

  const criteriaInfo = {
    research_intensity: {
      label: '研究強度',
      description: '研究にどれだけ集中したいか',
      min: '基礎的',
      max: '最先端',
      emoji: '🔬',
    },
    advisor_style: {
      label: '指導スタイル',
      description: '希望する指導の方法',
      min: '厳格',
      max: '自由',
      emoji: '👨‍🏫',
    },
    team_work: {
      label: 'チームワーク',
      description: '研究での協働の度合い',
      min: '個人研究',
      max: 'チーム研究',
      emoji: '🤝',
    },
    workload: {
      label: 'ワークロード',
      description: '研究の負荷・忙しさ',
      min: '軽め',
      max: '重め',
      emoji: '⚡',
    },
    theory_practice: {
      label: '理論・実践バランス',
      description: '理論と実践のどちらを重視するか',
      min: '理論重視',
      max: '実践重視',
      emoji: '⚖️',
    }
  };

  // フィールド統計計算
  const fieldStats = fieldUtils.calculateFieldStats(preferences.research_field_interests || {});

  // 基本設定変更時の推薦フィールド取得
  useEffect(() => {
    const fetchRecommendations = async () => {
      try {
        const basicPrefs = {
          research_intensity: preferences.research_intensity,
          advisor_style: preferences.advisor_style,
          team_work: preferences.team_work,
          workload: preferences.workload,
          theory_practice: preferences.theory_practice,
        };
        const recommendations = await apiService.getFieldRecommendations(basicPrefs);
        setFieldRecommendations(recommendations.recommended_fields || []);
      } catch (error) {
        console.log('フィールド推薦の取得に失敗しました:', error);
        setFieldRecommendations([]);
      }
    };

    // 基本設定が変更されたら推薦を更新（デバウンス）
    const timeoutId = setTimeout(fetchRecommendations, 1000);
    return () => clearTimeout(timeoutId);
  }, [preferences.research_intensity, preferences.advisor_style, preferences.team_work, preferences.workload, preferences.theory_practice]);

  const handleSliderChange = (criterion: keyof EnhancedEvaluationPreferences) => (
    event: Event,
    newValue: number | number[]
  ) => {
    setPreferences({
      ...preferences,
      [criterion]: newValue as number,
    });
  };

  const handleFieldSelection = (fieldId: string, isSelected: boolean) => {
    const currentInterests = preferences.research_field_interests || {};
    const updatedInterests = {
      ...currentInterests,
      [fieldId]: {
        isSelected,
        interestLevel: isSelected ? (currentInterests[fieldId]?.interestLevel || 7.0) : 0
      }
    };

    const newStats = fieldUtils.calculateFieldStats(updatedInterests);
    
    setPreferences({
      ...preferences,
      research_field_interests: updatedInterests,
      selected_fields_count: newStats.selectedCount,
      average_field_interest: newStats.averageInterest,
      primary_category: newStats.primaryCategory,
    });
  };

  const handleInterestLevelChange = (fieldId: string, level: number) => {
    const currentInterests = preferences.research_field_interests || {};
    const updatedInterests = {
      ...currentInterests,
      [fieldId]: {
        ...(currentInterests[fieldId] || { isSelected: false }),
        interestLevel: level
      }
    };

    const newStats = fieldUtils.calculateFieldStats(updatedInterests);

    setPreferences({
      ...preferences,
      research_field_interests: updatedInterests,
      average_field_interest: newStats.averageInterest,
    });
  };

  const handleEvaluate = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await apiService.evaluateCompatibility(preferences);
      onResults(response);
      
      // セッションIDを保存
      if (response.summary.session_id) {
        localStorage.setItem('fdtlss_session_id', response.summary.session_id);
      }
    } catch (err: any) {
      setError(err.response?.data?.error || '評価に失敗しました。サーバーが起動しているか確認してください。');
    } finally {
      setLoading(false);
    }
  };

  const loadDemoData = async () => {
    try {
      const response = await apiService.getDemoData();
      setPreferences(response.demo_preferences);
      if (response.suggested_fields && response.suggested_fields.length > 0) {
        setShowFieldSelection(true);
        setFieldRecommendations(response.suggested_fields);
      }
    } catch (err) {
      console.error('Demo data load failed:', err);
    }
  };

  const getScoreColor = (value: number): 'success' | 'warning' | 'info' | 'error' => {
    if (value >= 8) return 'success';
    if (value >= 6) return 'warning';
    if (value >= 4) return 'info';
    return 'error';
  };

  const groupedFields = fieldUtils.groupFieldsByCategory(RESEARCH_FIELDS);

  return (
    <Box>
      {/* メインヘッダー */}
      <Card elevation={3} sx={{ mb: 4 }}>
        <CardContent>
          <Box sx={{ textAlign: 'center', mb: 4 }}>
            <Science color="primary" sx={{ fontSize: 48, mb: 2 }} />
            <Typography variant="h4" component="h2" gutterBottom color="primary">
              🎯 研究室適合度評価（拡張版）
            </Typography>
            <Typography variant="body1" color="text.secondary">
              基本設定と研究分野興味度を設定して、より精密なマッチングを実現
            </Typography>
          </Box>

          {error && (
            <Alert severity="error" sx={{ mb: 3 }}>
              {error}
            </Alert>
          )}

          {/* 基本設定セクション */}
          <Typography variant="h5" gutterBottom sx={{ mt: 4, mb: 3 }}>
            📊 基本的な研究スタイル設定
          </Typography>
          
          <Grid container spacing={3}>
            {Object.entries(criteriaInfo).map(([key, info]) => (
              <Grid item xs={12} md={6} key={key}>
                <Paper elevation={1} sx={{ p: 3, height: '100%' }}>
                  <Box sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                      <Typography variant="h6" sx={{ flexGrow: 1 }}>
                        {info.emoji} {info.label}
                      </Typography>
                      <Chip 
                        label={preferences[key as keyof EnhancedEvaluationPreferences]?.toFixed(1)} 
                        color={getScoreColor(preferences[key as keyof EnhancedEvaluationPreferences] as number)}
                        size="small"
                      />
                    </Box>
                    
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                      {info.description}
                    </Typography>
                    
                    <Slider
                      value={preferences[key as keyof EnhancedEvaluationPreferences] as number}
                      onChange={handleSliderChange(key as keyof EnhancedEvaluationPreferences)}
                      min={1}
                      max={10}
                      step={0.5}
                      marks={[
                        { value: 1, label: info.min },
                        { value: 5, label: '中間' },
                        { value: 10, label: info.max },
                      ]}
                      valueLabelDisplay="auto"
                      sx={{ mt: 2 }}
                    />
                  </Box>
                </Paper>
              </Grid>
            ))}
          </Grid>

          {/* 研究分野選択セクション */}
          <Divider sx={{ my: 4 }} />
          
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
            <Typography variant="h5">
              🎨 研究分野興味度設定
            </Typography>
            <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
              {fieldStats.selectedCount > 0 && (
                <Chip 
                  icon={<Assessment />}
                  label={`${fieldStats.selectedCount}分野選択 / 平均興味度: ${fieldStats.averageInterest.toFixed(1)}`}
                  color="primary"
                  variant="outlined"
                />
              )}
              <Button
                variant="outlined"
                onClick={() => setShowFieldSelection(!showFieldSelection)}
                startIcon={<Category />}
              >
                {showFieldSelection ? '分野選択を隠す' : '分野選択を表示'}
              </Button>
            </Box>
          </Box>

          {/* フィールド推薦表示 */}
          {fieldRecommendations.length > 0 && !showFieldSelection && (
            <Alert severity="info" sx={{ mb: 3 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <AutoAwesome />
                <Typography variant="subtitle1" fontWeight="bold">
                  あなたの設定に基づく推薦分野
                </Typography>
              </Box>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {fieldRecommendations.map(fieldId => (
                  <Chip
                    key={fieldId}
                    label={fieldUtils.getFieldName(fieldId)}
                    size="small"
                    color="primary"
                    variant="outlined"
                  />
                ))}
              </Box>
            </Alert>
          )}

          {/* 分野選択UI */}
          {showFieldSelection && (
            <Box sx={{ mb: 4 }}>
              <Alert severity="info" sx={{ mb: 3 }}>
                興味のある分野にチェックを入れ、それぞれの興味度合いを1-10で設定してください。
                複数選択可能です。分野固有の適合度計算により、より精密な推薦が可能になります。
              </Alert>

              {Object.entries(groupedFields).map(([category, fields]) => (
                <Accordion key={category} sx={{ mb: 2 }}>
                  <AccordionSummary expandIcon={<ExpandMore />}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                      <Typography variant="h6">
                        {fieldUtils.getCategoryIcon(category)} {category}
                      </Typography>
                      <Badge 
                        badgeContent={fields.filter(field => 
                          preferences.research_field_interests?.[field.id]?.isSelected
                        ).length}
                        color="primary"
                      >
                        <Typography variant="body2" color="text.secondary">
                          {fields.length}分野
                        </Typography>
                      </Badge>
                    </Box>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Grid container spacing={2}>
                      {fields.map(field => {
                        const fieldData = preferences.research_field_interests?.[field.id];
                        const isSelected = fieldData?.isSelected || false;
                        const interestLevel = fieldData?.interestLevel || 7.0;
                        const isRecommended = fieldRecommendations.includes(field.id);

                        return (
                          <Grid item xs={12} key={field.id}>
                            <Paper 
                              sx={{ 
                                p: 2, 
                                border: isSelected ? '2px solid #1976d2' : '1px solid #e0e0e0',
                                backgroundColor: isRecommended ? '#f3f7ff' : 'inherit',
                                transition: 'all 0.3s ease',
                                '&:hover': {
                                  boxShadow: 2
                                }
                              }}
                            >
                              <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
                                <FormControlLabel
                                  control={
                                    <Checkbox
                                      checked={isSelected}
                                      onChange={(e) => handleFieldSelection(field.id, e.target.checked)}
                                    />
                                  }
                                  label=""
                                  sx={{ m: 0 }}
                                />
                                
                                <Box sx={{ flexGrow: 1 }}>
                                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                                    <Typography variant="h6">{field.name}</Typography>
                                    {isRecommended && (
                                      <Tooltip title="基本設定に基づく推薦分野">
                                        <Chip
                                          icon={<AutoAwesome />}
                                          label="推薦"
                                          size="small"
                                          color="primary"
                                        />
                                      </Tooltip>
                                    )}
                                    {isSelected && (
                                      <Chip 
                                        label={`興味度: ${interestLevel.toFixed(1)}`}
                                        color="primary"
                                        size="small"
                                      />
                                    )}
                                  </Box>
                                  
                                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                                    {field.description}
                                  </Typography>
                                  
                                  {isSelected && (
                                    <Box sx={{ maxWidth: 300 }}>
                                      <Typography variant="body2" gutterBottom>
                                        興味度レベル: {interestLevel.toFixed(1)}
                                      </Typography>
                                      <Slider
                                        value={interestLevel}
                                        onChange={(_, value) => handleInterestLevelChange(field.id, value as number)}
                                        min={1}
                                        max={10}
                                        step={0.5}
                                        marks={[
                                          { value: 1, label: '低' },
                                          { value: 5, label: '中' },
                                          { value: 10, label: '高' }
                                        ]}
                                        valueLabelDisplay="auto"
                                      />
                                    </Box>
                                  )}
                                </Box>
                              </Box>
                            </Paper>
                          </Grid>
                        );
                      })}
                    </Grid>
                  </AccordionDetails>
                </Accordion>
              ))}
            </Box>
          )}

          {/* 選択サマリー */}
          {fieldStats.selectedCount > 0 && (
            <Paper sx={{ p: 3, mb: 4, background: 'linear-gradient(135deg, #e8f5e8 0%, #f1f8e9 100%)' }}>
              <Typography variant="h6" gutterBottom>
                📋 選択サマリー
              </Typography>
              
              <Grid container spacing={2}>
                <Grid item xs={12} sm={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" color="primary" fontWeight="bold">
                      {fieldStats.selectedCount}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      選択分野数
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" color="primary" fontWeight="bold">
                      {fieldStats.averageInterest.toFixed(1)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      平均興味度
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" color="primary" fontWeight="bold">
                      {fieldStats.primaryCategory || '-'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      主要カテゴリー
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" color="primary" fontWeight="bold">
                      {Object.keys(fieldStats.categoryDistribution).length}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      カテゴリー数
                    </Typography>
                  </Box>
                </Grid>
              </Grid>

              {fieldStats.selectedCount > 0 && (
                <Box sx={{ mt: 3 }}>
                  <Typography variant="body2" gutterBottom>選択した分野:</Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {Object.entries(preferences.research_field_interests || {})
                      .filter(([_, data]) => data?.isSelected)
                      .map(([fieldId, data]) => (
                        <Chip
                          key={fieldId}
                          label={`${fieldUtils.getFieldName(fieldId)} (${data?.interestLevel?.toFixed(1) || '0.0'})`}
                          color="primary"
                          variant="outlined"
                          size="small"
                        />
                      ))}
                  </Box>
                </Box>
              )}
            </Paper>
          )}

          {/* 実行ボタン */}
          <Box sx={{ mt: 4, display: 'flex', gap: 2, justifyContent: 'center' }}>
            <Button
              variant="contained"
              size="large"
              onClick={handleEvaluate}
              disabled={loading}
              startIcon={loading ? <CircularProgress size={20} /> : <Psychology />}
              sx={{ 
                minWidth: 250,
                py: 1.5,
                fontSize: '1.1rem'
              }}
            >
              {loading ? '🔄 計算中...' : `🎯 適合度を計算する${fieldStats.selectedCount > 0 ? ` (${fieldStats.selectedCount}分野)` : ''}`}
            </Button>
            
            <Button
              variant="outlined"
              size="large"
              onClick={loadDemoData}
              startIcon={<TrendingUp />}
              sx={{ minWidth: 150 }}
            >
              📊 デモデータ
            </Button>

            {!showFieldSelection && fieldStats.selectedCount === 0 && (
              <Button
                variant="outlined"
                size="large"
                onClick={() => setShowFieldSelection(true)}
                startIcon={<Category />}
                sx={{ minWidth: 180 }}
              >
                🎨 分野を選択
              </Button>
            )}
          </Box>

          {/* ヘルプテキスト */}
          <Box sx={{ mt: 4, p: 3, backgroundColor: '#f8f9fa', borderRadius: 2 }}>
            <Typography variant="h6" gutterBottom>
              💡 システムについて
            </Typography>
            <Typography variant="body2" paragraph>
              このシステムは、<strong>適応型ファジィ決定木（AFDT）</strong>を用いた高度なアルゴリズムにより、
              あなたの希望と各研究室の特徴を多角的に分析し、最適なマッチングを提供します。
            </Typography>
            
            <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
              🔬 評価基準（基本設定）
            </Typography>
            <Box component="ul" sx={{ pl: 3, mb: 2 }}>
              <li><strong>研究強度</strong>: 研究活動の集中度・最先端性</li>
              <li><strong>指導スタイル</strong>: 教授の指導方針（厳格 ↔ 自由）</li>
              <li><strong>チームワーク</strong>: 研究での協働度（個人 ↔ チーム）</li>
              <li><strong>ワークロード</strong>: 研究の負荷・忙しさ</li>
              <li><strong>理論・実践バランス</strong>: 理論研究と実践的研究の比重</li>
            </Box>

            <Typography variant="subtitle2" gutterBottom>
              🎨 研究分野興味度（拡張機能）
            </Typography>
            <Typography variant="body2">
              北海道情報大学の実際の研究分野に基づいて、あなたの興味領域を詳細に設定できます。
              選択した分野は個別に重み付けされ、より精密な研究室推薦を実現します。
            </Typography>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};

export default EnhancedEvaluationForm;