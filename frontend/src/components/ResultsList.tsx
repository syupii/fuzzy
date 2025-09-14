// frontend/src/components/ResultsList.tsx - 完全修正版
import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  LinearProgress,
  Grid,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Collapse,
  IconButton,
  Alert,
  Divider,
  Paper
} from '@mui/material';
import {
  ExpandMore,
  ExpandLess,
  StarRate,
  Assessment,
  School,
  Psychology,
  Engineering,
  Palette,
  SportsEsports,
  Groups
} from '@mui/icons-material';
import { LabResult, EvaluationResponse } from '../services/api';

interface Props {
  evaluationResponse: EvaluationResponse;
}

const ResultsList: React.FC<Props> = ({ evaluationResponse }) => {
  const [expandedItems, setExpandedItems] = useState<Set<number>>(new Set());

  const handleExpandClick = (index: number) => {
    setExpandedItems(prev => {
      const newSet = new Set(prev);
      if (newSet.has(index)) {
        newSet.delete(index);
      } else {
        newSet.add(index);
      }
      return newSet;
    });
  };

  // スコアに基づく色の決定
  const getScoreColor = (score: number): 'success' | 'warning' | 'error' => {
    if (score >= 0.8) return 'success';
    if (score >= 0.6) return 'warning';
    return 'error';
  };

  // 研究分野の色決定
  const getFieldColor = (field: string): 'primary' | 'secondary' | 'default' => {
    if (field.includes('AI') || field.includes('機械学習') || field.includes('画像')) return 'primary';
    if (field.includes('Web') || field.includes('デザイン') || field.includes('UI')) return 'secondary';
    return 'default';
  };

  // カテゴリアイコンの取得
  const getCategoryIcon = (researchArea: string) => {
    if (researchArea?.includes('人工知能') || researchArea?.includes('機械学習')) {
      return <Psychology color="primary" />;
    }
    if (researchArea?.includes('Web') || researchArea?.includes('デザイン')) {
      return <Palette color="secondary" />;
    }
    if (researchArea?.includes('ゲーム') || researchArea?.includes('VR')) {
      return <SportsEsports color="error" />;
    }
    return <Engineering color="action" />;
  };

  // スコア表示コンポーネント
  const ScoreDisplay: React.FC<{ score: number; label: string }> = ({ score, label }) => {
    const percentage = Math.round(score * 100);
    const color = getScoreColor(score);

    return (
      <Box sx={{ minWidth: 120 }}>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          {label}
        </Typography>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <LinearProgress
            variant="determinate"
            value={percentage}
            color={color}
            sx={{ flexGrow: 1, height: 8, borderRadius: 4 }}
          />
          <Typography variant="body2" fontWeight="bold">
            {percentage}%
          </Typography>
        </Box>
      </Box>
    );
  };

  const { lab_results = [], summary } = evaluationResponse;

  if (!lab_results || lab_results.length === 0) {
    return (
      <Alert severity="info">
        評価結果がありません。評価を実行してください。
      </Alert>
    );
  }

  return (
    <Box sx={{ width: '100%' }}>
      {/* サマリー情報 */}
      <Card sx={{ mb: 3, bgcolor: 'primary.50', border: '1px solid', borderColor: 'primary.200' }}>
        <CardContent>
          <Typography variant="h5" gutterBottom color="primary.main">
            📊 評価結果サマリー
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={6} md={3}>
              <Box textAlign="center">
                <Typography variant="h4" color="primary.main" fontWeight="bold">
                  {summary?.total_labs || lab_results.length}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  研究室総数
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={6} md={3}>
              <Box textAlign="center">
                <Typography variant="h4" color="success.main" fontWeight="bold">
                  {lab_results.filter(lab => (lab.final_score || 0) >= 0.8).length}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  高適合 (80%+)
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={6} md={3}>
              <Box textAlign="center">
                <Typography variant="h4" color="warning.main" fontWeight="bold">
                  {lab_results.filter(lab => (lab.final_score || 0) >= 0.6 && (lab.final_score || 0) < 0.8).length}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  中適合 (60-79%)
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={6} md={3}>
              <Box textAlign="center">
                <Typography variant="h4" color="text.secondary" fontWeight="bold">
                  {lab_results.filter(lab => (lab.final_score || 0) < 0.6).length}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  要検討 (~59%)
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* 研究室リスト */}
      <Typography variant="h6" gutterBottom>
        🔬 研究室マッチング結果 (適合度順)
      </Typography>

      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {lab_results.map((labResult: LabResult, index: number) => {
          const isExpanded = expandedItems.has(index);
          const score = labResult.final_score || 0;
          const percentage = Math.round(score * 100);

          return (
            <Card
              key={`${labResult.lab_name}-${index}`}
              sx={{
                border: 2,
                borderColor: score >= 0.8 ? 'success.main' : score >= 0.6 ? 'warning.main' : 'grey.300',
                bgcolor: score >= 0.8 ? 'success.50' : score >= 0.6 ? 'warning.50' : 'white'
              }}
            >
              <CardContent>
                {/* ヘッダー部分 */}
                <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flex: 1 }}>
                    {getCategoryIcon(labResult.research_area || '')}
                    <Box>
                      <Typography variant="h6" component="h3">
                        {labResult.lab_name || `研究室 ${index + 1}`}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {labResult.professor_name} 教授
                      </Typography>
                    </Box>
                  </Box>

                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <ScoreDisplay score={score} label="総合適合度" />
                    <IconButton
                      onClick={() => handleExpandClick(index)}
                      sx={{
                        bgcolor: 'white',
                        boxShadow: 1,
                        '&:hover': { bgcolor: 'grey.100' }
                      }}
                    >
                      {isExpanded ? <ExpandLess /> : <ExpandMore />}
                    </IconButton>
                  </Box>
                </Box>

                {/* 基本情報 */}
                <Box sx={{ mb: 2 }}>
                  {labResult.research_area && (
                    <Chip
                      label={labResult.research_area}
                      color={getFieldColor(labResult.research_area)}
                      size="small"
                      sx={{ mr: 1, mb: 1 }}
                    />
                  )}

                  {labResult.keywords && labResult.keywords.length > 0 && (
                    <Box sx={{ mt: 1 }}>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        キーワード:
                      </Typography>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {labResult.keywords.map((keyword, i) => (
                          <Chip
                            key={i}
                            label={keyword}
                            size="small"
                            variant="outlined"
                          />
                        ))}
                      </Box>
                    </Box>
                  )}
                </Box>

                {/* 展開可能な詳細情報 */}
                <Collapse in={isExpanded} timeout="auto">
                  {labResult.detailed_scores && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        📈 詳細スコア分析
                      </Typography>
                      <Grid container spacing={2} sx={{ mb: 2 }}>
                        {Object.entries(labResult.detailed_scores).map(([criteria, score]) => (
                          <Grid item xs={6} md={4} key={criteria}>
                            <ScoreDisplay
                              score={score as number}
                              label={criteria.replace('_', ' ')}
                            />
                          </Grid>
                        ))}
                      </Grid>
                    </Box>
                  )}

                  {labResult.explanation && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        💭 マッチング理由
                      </Typography>
                      <Typography variant="body2" sx={{ bgcolor: 'grey.50', p: 2, borderRadius: 1 }}>
                        {labResult.explanation}
                      </Typography>
                    </Box>
                  )}

                  {labResult.suggestions && labResult.suggestions.length > 0 && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        🎯 推奨アクション
                      </Typography>
                      <List dense>
                        {labResult.suggestions.map((suggestion, i) => (
                          <ListItem key={i} sx={{ py: 0 }}>
                            <ListItemIcon sx={{ minWidth: 32 }}>
                              <StarRate fontSize="small" color="primary" />
                            </ListItemIcon>
                            <ListItemText
                              primary={suggestion}
                              primaryTypographyProps={{ variant: 'body2' }}
                            />
                          </ListItem>
                        ))}
                      </List>
                    </Box>
                  )}

                  {/* メタデータ */}
                  {labResult.metadata && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        📋 研究室情報
                      </Typography>
                      <Grid container spacing={1}>
                        {labResult.metadata.student_count && (
                          <Grid item xs={6}>
                            <Typography variant="caption">
                              学生数: {labResult.metadata.student_count}名
                            </Typography>
                          </Grid>
                        )}
                        {labResult.metadata.equipment_level && (
                          <Grid item xs={6}>
                            <Typography variant="caption">
                              設備レベル: {labResult.metadata.equipment_level}/10
                            </Typography>
                          </Grid>
                        )}
                        {labResult.metadata.funding_level && (
                          <Grid item xs={6}>
                            <Typography variant="caption">
                              資金レベル: {labResult.metadata.funding_level}
                            </Typography>
                          </Grid>
                        )}
                        {labResult.metadata.faculty_type && (
                          <Grid item xs={6}>
                            <Typography variant="caption">
                              学部: {labResult.metadata.faculty_type}
                            </Typography>
                          </Grid>
                        )}
                      </Grid>
                    </Box>
                  )}
                </Collapse>
              </CardContent>
            </Card>
          );
        })}
      </Box>

      {/* フッター情報 */}
      <Paper sx={{ mt: 4, p: 3, bgcolor: 'grey.50' }}>
        <Typography variant="body2" color="text.secondary" align="center">
          💡 <strong>重要:</strong> このシステムは参考情報です。最終的な研究室選択は、直接教授と面談し、
          研究内容や雰囲気を確認してから決定することをお勧めします。
        </Typography>
      </Paper>
    </Box>
  );
};

export default ResultsList;