import React, { useState } from 'react';
import {
  Box,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  FormControlLabel,
  Checkbox,
  Slider,
  Chip,
  Grid,
  Card,
  CardContent,
  Button,
  ButtonGroup,
  Paper,
  Tooltip,
  Alert,
} from '@mui/material';
import {
  ExpandMore,
  Science,
  Computer,
  Business,
  LocalHospital,
  Brush,
  School,
  TrendingUp,
  Security,
  Psychology,
} from '@mui/icons-material';
import {
  RESEARCH_FIELDS,
  FIELD_CATEGORIES,
  ResearchField,
  FieldInterest,
  fieldUtils,
} from '../services/api';

interface FieldSelectionFormProps {
  selectedFields: { [fieldId: string]: FieldInterest };
  onFieldChange: (fieldId: string, interest: FieldInterest) => void;
  onSubmit: () => void;
}

const FieldSelectionForm: React.FC<FieldSelectionFormProps> = ({
  selectedFields,
  onFieldChange,
  onSubmit,
}) => {
  const [expandedCategories, setExpandedCategories] = useState<string[]>([]);
  const [filterDifficulty, setFilterDifficulty] = useState<string[]>([]);
  const [filterDemand, setFilterDemand] = useState<string[]>([]);

  // カテゴリアイコンのマッピング
  const getCategoryIcon = (category: string) => {
    const iconMap: { [key: string]: React.ReactElement } = {
      '情報工学・AI': <Psychology />,
      'Web・アプリ開発': <Computer />,
      '基盤技術': <Science />,
      'ビジネス・経営': <Business />,
      '医療・健康': <LocalHospital />,
      'メディア・コンテンツ': <Brush />,
      'デザイン・UX': <Brush />,
      '組み込み・IoT': <Computer />,
      '新興技術': <TrendingUp />,
      '先端技術': <TrendingUp />,
      '理論・数学': <School />,
      'データ科学': <Psychology />,
      'エンターテイメント': <Brush />,
      '生命科学': <Science />,
      '環境・地球科学': <Science />,
      '社会応用': <Business />,
      '理論・検証': <Security />,
    };
    return iconMap[category] || <Computer />;
  };

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

  // カテゴリアコーディオンの展開/折りたたみ
  const handleCategoryToggle = (category: string) => {
    setExpandedCategories(prev =>
      prev.includes(category)
        ? prev.filter(c => c !== category)
        : [...prev, category]
    );
  };

  // フィールド選択の変更
  const handleFieldToggle = (fieldId: string) => {
    const current = selectedFields[fieldId] || { isSelected: false, interestLevel: 5, priority: 'medium' };
    const newSelection = {
      ...current,
      isSelected: !current.isSelected,
    };
    onFieldChange(fieldId, newSelection);
  };

  // 興味レベルの変更
  const handleInterestChange = (fieldId: string, value: number) => {
    const current = selectedFields[fieldId] || { isSelected: true, interestLevel: 5, priority: 'medium' };
    const priority = value >= 8 ? 'high' : value >= 5 ? 'medium' : 'low';
    onFieldChange(fieldId, {
      ...current,
      isSelected: true,
      interestLevel: value,
      priority: priority as 'high' | 'medium' | 'low',
    });
  };

  // フィルタリング
  const getFilteredFields = () => {
    return RESEARCH_FIELDS.filter(field => {
      if (filterDifficulty.length > 0 && !filterDifficulty.includes(field.difficulty)) {
        return false;
      }
      if (filterDemand.length > 0 && !filterDemand.includes(field.marketDemand)) {
        return false;
      }
      return true;
    });
  };

  // カテゴリ別にフィールドをグループ化
  const getFieldsByCategory = (category: string) => {
    const filteredFields = getFilteredFields();
    return filteredFields.filter(field => field.category === category);
  };

  // 選択済みフィールドの統計
  const getSelectionStats = () => {
    const selectedCount = Object.values(selectedFields).filter(f => f.isSelected).length;
    const averageInterest = selectedCount > 0
      ? Object.values(selectedFields)
          .filter(f => f.isSelected)
          .reduce((sum, f) => sum + f.interestLevel, 0) / selectedCount
      : 0;
    const highPriorityCount = Object.values(selectedFields)
      .filter(f => f.isSelected && f.priority === 'high').length;

    return { selectedCount, averageInterest, highPriorityCount };
  };

  const stats = getSelectionStats();

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ mb: 3, textAlign: 'center' }}>
        🔬 研究分野選択
      </Typography>

      <Alert severity="info" sx={{ mb: 3 }}>
        <Typography variant="body2">
          興味のある研究分野を選択し、それぞれの興味レベル（1-10）を設定してください。
          この情報により、あなたに最適な研究室をより精密にマッチングできます。
        </Typography>
      </Alert>

      {/* 選択統計 */}
      <Paper sx={{ p: 2, mb: 3, bgcolor: '#f5f5f5' }}>
        <Grid container spacing={3}>
          <Grid item xs={12} sm={4}>
            <Box textAlign="center">
              <Typography variant="h6" color="primary">
                {stats.selectedCount}
              </Typography>
              <Typography variant="body2">選択分野数</Typography>
            </Box>
          </Grid>
          <Grid item xs={12} sm={4}>
            <Box textAlign="center">
              <Typography variant="h6" color="secondary">
                {stats.averageInterest.toFixed(1)}
              </Typography>
              <Typography variant="body2">平均興味レベル</Typography>
            </Box>
          </Grid>
          <Grid item xs={12} sm={4}>
            <Box textAlign="center">
              <Typography variant="h6" color="error">
                {stats.highPriorityCount}
              </Typography>
              <Typography variant="body2">高優先度分野</Typography>
            </Box>
          </Grid>
        </Grid>
      </Paper>

      {/* フィルター */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Typography variant="h6" gutterBottom>フィルター</Typography>
        
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>難易度:</Typography>
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
                sx={{ 
                  color: getDifficultyColor(difficulty),
                  borderColor: getDifficultyColor(difficulty)
                }}
              >
                {difficulty === 'beginner' ? '初級' : 
                 difficulty === 'intermediate' ? '中級' : '上級'}
              </Button>
            ))}
          </ButtonGroup>
        </Box>

        <Box>
          <Typography variant="subtitle2" gutterBottom>市場需要:</Typography>
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
                sx={{
                  color: getDemandColor(demand),
                  borderColor: getDemandColor(demand)
                }}
              >
                {demand === 'high' ? '高' : demand === 'medium' ? '中' : '低'}
              </Button>
            ))}
          </ButtonGroup>
        </Box>
      </Paper>

      {/* 分野選択 */}
      {FIELD_CATEGORIES.map(category => {
        const fields = getFieldsByCategory(category);
        if (fields.length === 0) return null;

        return (
          <Accordion
            key={category}
            expanded={expandedCategories.includes(category)}
            onChange={() => handleCategoryToggle(category)}
            sx={{ mb: 2 }}
          >
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                {getCategoryIcon(category)}
                <Typography variant="h6">{category}</Typography>
                <Chip
                  label={`${fields.filter(f => selectedFields[f.id]?.isSelected).length}/${fields.length}`}
                  size="small"
                  color="primary"
                  variant="outlined"
                />
              </Box>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                {fields.map(field => {
                  const isSelected = selectedFields[field.id]?.isSelected || false;
                  const interestLevel = selectedFields[field.id]?.interestLevel || 5;
                  
                  return (
                    <Grid item xs={12} md={6} lg={4} key={field.id}>
                      <Card 
                        variant="outlined" 
                        sx={{ 
                          height: '100%',
                          border: isSelected ? '2px solid #1976d2' : '1px solid #e0e0e0',
                          bgcolor: isSelected ? '#f3f7ff' : 'white',
                          transition: 'all 0.3s ease'
                        }}
                      >
                        <CardContent sx={{ p: 2 }}>
                          <FormControlLabel
                            control={
                              <Checkbox
                                checked={isSelected}
                                onChange={() => handleFieldToggle(field.id)}
                                color="primary"
                              />
                            }
                            label={
                              <Typography variant="subtitle1" fontWeight="bold">
                                {field.name}
                              </Typography>
                            }
                          />
                          
                          <Typography 
                            variant="body2" 
                            color="text.secondary" 
                            sx={{ mb: 1, minHeight: 40 }}
                          >
                            {field.description}
                          </Typography>

                          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 1 }}>
                            {field.keywords.slice(0, 3).map(keyword => (
                              <Chip
                                key={keyword}
                                label={keyword}
                                size="small"
                                variant="outlined"
                                sx={{ fontSize: '0.7rem' }}
                              />
                            ))}
                          </Box>

                          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                            <Tooltip title={`難易度: ${field.difficulty}`}>
                              <Chip
                                label={field.difficulty === 'beginner' ? '初級' : 
                                      field.difficulty === 'intermediate' ? '中級' : '上級'}
                                size="small"
                                sx={{ 
                                  bgcolor: getDifficultyColor(field.difficulty),
                                  color: 'white',
                                  fontSize: '0.7rem'
                                }}
                              />
                            </Tooltip>
                            
                            <Tooltip title={`市場需要: ${field.marketDemand}`}>
                              <Chip
                                label={field.marketDemand === 'high' ? '需要高' : 
                                      field.marketDemand === 'medium' ? '需要中' : '需要低'}
                                size="small"
                                sx={{ 
                                  bgcolor: getDemandColor(field.marketDemand),
                                  color: 'white',
                                  fontSize: '0.7rem'
                                }}
                              />
                            </Tooltip>
                          </Box>

                          {isSelected && (
                            <Box sx={{ mt: 2 }}>
                              <Typography variant="body2" gutterBottom>
                                興味レベル: {interestLevel}
                              </Typography>
                              <Slider
                                value={interestLevel}
                                onChange={(_, value) => 
                                  handleInterestChange(field.id, value as number)
                                }
                                min={1}
                                max={10}
                                step={1}
                                marks={[
                                  { value: 1, label: '1' },
                                  { value: 5, label: '5' },
                                  { value: 10, label: '10' }
                                ]}
                                valueLabelDisplay="auto"
                                color="primary"
                              />
                            </Box>
                          )}
                        </CardContent>
                      </Card>
                    </Grid>
                  );
                })}
              </Grid>
            </AccordionDetails>
          </Accordion>
        );
      })}

      {/* 送信ボタン */}
      <Box sx={{ textAlign: 'center', mt: 4 }}>
        <Button
          variant="contained"
          size="large"
          onClick={onSubmit}
          disabled={stats.selectedCount === 0}
          sx={{ minWidth: 200, py: 1.5 }}
        >
          選択完了 ({stats.selectedCount}分野選択済み)
        </Button>
      </Box>
    </Box>
  );
};

export default FieldSelectionForm;