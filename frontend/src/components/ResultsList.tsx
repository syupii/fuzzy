// frontend/src/components/ResultsList.tsx - 型安全版
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
  Divider
} from '@mui/material';
import {
  ExpandMore,
  ExpandLess,
  StarRate,
  Assessment,
  School,
  Psychology,
  Engineering
} from '@mui/icons-material';
import { LabResult, EvaluationResponse } from '../services/api';

interface Props {
  evaluationResponse: EvaluationResponse;
}

export const ResultsList: React.FC<Props> = ({ evaluationResponse }) => {
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
      return <Engineering color="secondary" />;
    }
    return <School color="action" />;
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
      <Card sx={{ mb: 3, bgcolor: 'primary.50' }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            📊 評価結果サマリー
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={6} md={3}>
              <Typography variant="h6" color="primary">
                {summary.total_labs}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                研究室総数
              </Typography>
            </Grid>
            <Grid item xs={6} md={3}>
              <Typography variant="h6" color="primary">
                {(summary.avg_score * 100).toFixed(1)}%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                平均適合度
              </Typography>
            </Grid>
            <Grid item xs={12} md={6}>
              <Typography variant="h6" color="success.main">
                {summary.best_match_lab || '未設定'}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                最高適合研究室
              </Typography>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* 研究室結果一覧 */}
      <Box>
        <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
          🏆 研究室ランキング
        </Typography>

        {lab_results.map((labResult: LabResult, index: number) => {
          const isExpanded = expandedItems.has(index);
          const score = labResult.overall_score || labResult.compatibility_score || 0;
          const scoreColor = getScoreColor(score);

          return (
            <Card key={`lab-${index}`} sx={{ mb: 2, position: 'relative' }}>
              <CardContent>
                {/* ランキング表示 */}
                <Box sx={{ position: 'absolute', top: 16, right: 16 }}>
                  <Chip
                    label={`#${index + 1}`}
                    color={index < 3 ? 'primary' : 'default'}
                    size="small"
                  />
                </Box>

                {/* 基本情報 */}
                <Grid container spacing={2} alignItems="center">
                  <Grid item xs={1}>
                    {getCategoryIcon(labResult.research_area || '')}
                  </Grid>
                  <Grid item xs={8}>
                    <Typography variant="h6" gutterBottom>
                      {labResult.lab_name || labResult.lab?.name || '研究室名未設定'}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {labResult.professor || labResult.advisor || '教授名未設定'} - {labResult.research_area || '分野未設定'}
                    </Typography>
                  </Grid>
                  <Grid item xs={2}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h5" color={`${scoreColor}.main`}>
                        {(score * 100).toFixed(0)}%
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        適合度
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>

                {/* スコアバー */}
                <Box sx={{ mt: 2, mb: 2 }}>
                  <LinearProgress
                    variant="determinate"
                    value={score * 100}
                    color={scoreColor}
                    sx={{ height: 8, borderRadius: 4 }}
                  />
                </Box>

                {/* 研究分野表示（型安全版） */}
                {labResult.research_fields && labResult.research_fields.length > 0 && (
                  <Box sx={{ mt: 1 }}>
                    {labResult.research_fields.slice(0, 3).map((field: string, fieldIndex: number) => (
                      <Chip
                        key={fieldIndex}
                        label={field}
                        size="small"
                        color={getFieldColor(field)}
                        sx={{ mr: 1, mb: 1 }}
                      />
                    ))}
                    {labResult.research_fields.length > 3 && (
                      <Chip
                        label={`+${labResult.research_fields.length - 3}個`}
                        size="small"
                        variant="outlined"
                        sx={{ mr: 1, mb: 1 }}
                      />
                    )}
                  </Box>
                )}

                {/* 専門分野 */}
                {labResult.specialization && (
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    専門: {labResult.specialization}
                  </Typography>
                )}

                {/* 展開/縮小ボタン */}
                <Box sx={{ display: 'flex', justifyContent: 'center', mt: 2 }}>
                  <IconButton
                    onClick={() => handleExpandClick(index)}
                    size="small"
                  >
                    {isExpanded ? <ExpandLess /> : <ExpandMore />}
                    <Typography variant="caption" sx={{ ml: 1 }}>
                      {isExpanded ? '詳細を閉じる' : '詳細を見る'}
                    </Typography>
                  </IconButton>
                </Box>

                {/* 詳細情報（展開時） */}
                <Collapse in={isExpanded} timeout="auto" unmountOnExit>
                  <Divider sx={{ my: 2 }} />

                  {/* 説明文 */}
                  {labResult.description && (
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        📝 研究室について:
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {labResult.description}
                      </Typography>
                    </Box>
                  )}

                  {/* 強み・考慮点の表示（型安全版） */}
                  <Grid container spacing={2} sx={{ mb: 2 }}>
                    {labResult.strengths && labResult.strengths.length > 0 && (
                      <Grid item xs={12} md={6}>
                        <Typography variant="subtitle2" color="success.main" gutterBottom>
                          ✨ 強み:
                        </Typography>
                        <List dense>
                          {labResult.strengths.slice(0, 2).map((strength: string, idx: number) => (
                            <ListItem key={idx} sx={{ py: 0 }}>
                              <ListItemIcon sx={{ minWidth: 24 }}>
                                <StarRate color="success" fontSize="small" />
                              </ListItemIcon>
                              <ListItemText
                                primary={strength}
                                primaryTypographyProps={{ variant: 'body2' }}
                              />
                            </ListItem>
                          ))}
                        </List>
                      </Grid>
                    )}

                    {labResult.considerations && labResult.considerations.length > 0 && (
                      <Grid item xs={12} md={6}>
                        <Typography variant="subtitle2" color="warning.main" gutterBottom>
                          ⚠️ 考慮点:
                        </Typography>
                        <List dense>
                          {labResult.considerations.slice(0, 2).map((consideration: string, idx: number) => (
                            <ListItem key={idx} sx={{ py: 0 }}>
                              <ListItemIcon sx={{ minWidth: 24 }}>
                                <Assessment color="warning" fontSize="small" />
                              </ListItemIcon>
                              <ListItemText
                                primary={consideration}
                                primaryTypographyProps={{ variant: 'body2' }}
                              />
                            </ListItem>
                          ))}
                        </List>
                      </Grid>
                    )}
                  </Grid>

                  {/* 推薦理由 */}
                  {labResult.recommendations && labResult.recommendations.length > 0 && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        💡 推薦理由:
                      </Typography>
                      <List dense>
                        {labResult.recommendations.slice(0, 3).map((recommendation: string, idx: number) => (
                          <ListItem key={idx} sx={{ py: 0 }}>
                            <ListItemText
                              primary={`• ${recommendation}`}
                              primaryTypographyProps={{ variant: 'body2', color: 'text.secondary' }}
                            />
                          </ListItem>
                        ))}
                      </List>
                    </Box>
                  )}

                  {/* メタデータ情報 */}
                  {labResult.metadata && (
                    <Box sx={{ mt: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        📊 追加情報:
                      </Typography>
                      <Grid container spacing={1}>
                        {labResult.metadata.student_capacity && (
                          <Grid item xs={6}>
                            <Typography variant="caption">
                              定員: {labResult.metadata.student_capacity}名
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
      <Card sx={{ mt: 3, bgcolor: 'grey.50' }}>
        <CardContent>
          <Typography variant="body2" color="text.secondary" align="center">
            💡 より詳しい情報については、各研究室のWebサイトを確認するか、直接お問い合わせください。
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default ResultsList;