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
import { LabResult, EvaluationResponse, getLabScore } from '../services/api';

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
    if (!field) return 'default';
    const fieldLower = field.toLowerCase();
    if (fieldLower.includes('ai') || fieldLower.includes('機械学習') || fieldLower.includes('画像')) return 'primary';
    if (fieldLower.includes('web') || fieldLower.includes('デザイン') || fieldLower.includes('ui')) return 'secondary';
    return 'default';
  };

  // カテゴリアイコンの取得
  const getCategoryIcon = (researchArea: string = '') => {
    const area = researchArea.toLowerCase();
    if (area.includes('人工知能') || area.includes('機械学習') || area.includes('ai')) {
      return <Psychology color="primary" />;
    }
    if (area.includes('web') || area.includes('デザイン') || area.includes('ui')) {
      return <Palette color="secondary" />;
    }
    if (area.includes('ゲーム') || area.includes('vr') || area.includes('ar')) {
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

  // 警告表示
  const renderWarnings = () => {
    if (evaluationResponse.warnings) {
      return (
        <Alert severity="warning" sx={{ mb: 2 }}>
          <Typography variant="body2">
            {evaluationResponse.warnings.message}
          </Typography>
          {evaluationResponse.warnings.calculation_errors && (
            <Box sx={{ mt: 1 }}>
              <Typography variant="caption" color="text.secondary">
                詳細: {evaluationResponse.warnings.calculation_errors.length}件のエラー
              </Typography>
            </Box>
          )}
        </Alert>
      );
    }
    return null;
  };

  return (
    <Box sx={{ width: '100%' }}>
      {/* 警告表示 */}
      {renderWarnings()}

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
                  {lab_results.filter(lab => getLabScore(lab) >= 0.8).length}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  高適合 (80%+)
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={6} md={3}>
              <Box textAlign="center">
                <Typography variant="h4" color="warning.main" fontWeight="bold">
                  {lab_results.filter(lab => {
                    const score = getLabScore(lab);
                    return score >= 0.6 && score < 0.8;
                  }).length}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  中適合 (60-79%)
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={6} md={3}>
              <Box textAlign="center">
                <Typography variant="h4" color="text.secondary" fontWeight="bold">
                  {lab_results.filter(lab => getLabScore(lab) < 0.6).length}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  要検討 (~59%)
                </Typography>
              </Box>
            </Grid>
          </Grid>

          {/* 追加統計情報 */}
          {summary && (
            <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid', borderColor: 'primary.200' }}>
              <Grid container spacing={2}>
                <Grid item xs={4}>
                  <Typography variant="body2" color="text.secondary">
                    平均適合度
                  </Typography>
                  <Typography variant="h6" color="primary.main">
                    {summary.avg_score ? `${Math.round(summary.avg_score * 100)}%` : 'N/A'}
                  </Typography>
                </Grid>
                <Grid item xs={4}>
                  <Typography variant="body2" color="text.secondary">
                    最高適合度
                  </Typography>
                  <Typography variant="h6" color="success.main">
                    {summary.max_score ? `${Math.round(summary.max_score * 100)}%` : 'N/A'}
                  </Typography>
                </Grid>
                <Grid item xs={4}>
                  <Typography variant="body2" color="text.secondary">
                    最低適合度
                  </Typography>
                  <Typography variant="h6" color="text.secondary">
                    {summary.min_score ? `${Math.round(summary.min_score * 100)}%` : 'N/A'}
                  </Typography>
                </Grid>
              </Grid>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* 研究室リスト */}
      <Typography variant="h6" gutterBottom>
        🔬 研究室マッチング結果 (適合度順)
      </Typography>

      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
        {lab_results.map((labResult: LabResult, index: number) => {
          const isExpanded = expandedItems.has(index);
          const score = getLabScore(labResult);  // 統一されたスコア取得関数を使用
          const percentage = Math.round(score * 100);

          return (
            <Card
              key={`${labResult.lab_name}-${index}`}
              sx={{
                border: 2,
                borderColor: score >= 0.8 ? 'success.main' : score >= 0.6 ? 'warning.main' : 'grey.300',
                bgcolor: score >= 0.8 ? 'success.50' : score >= 0.6 ? 'warning.50' : 'grey.50',
                transition: 'all 0.2s ease-in-out',
                '&:hover': {
                  transform: 'translateY(-2px)',
                  boxShadow: 3
                }
              }}
            >
              <CardContent>
                {/* ヘッダー部分 */}
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                  <Box sx={{ flexGrow: 1 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                      <Typography variant="h6" component="div" fontWeight="bold">
                        {index + 1}. {labResult.lab_name}
                      </Typography>
                      {getCategoryIcon(labResult.research_area)}
                    </Box>

                    {/* 教授名 */}
                    {labResult.professor_name && (
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                        指導教員: {labResult.professor_name}
                      </Typography>
                    )}

                    {/* 研究分野チップ */}
                    {labResult.research_area && (
                      <Chip
                        label={labResult.research_area}
                        size="small"
                        color={getFieldColor(labResult.research_area)}
                        sx={{ mb: 1 }}
                      />
                    )}
                  </Box>

                  <Box sx={{ textAlign: 'right', minWidth: 120 }}>
                    <Typography variant="h4" color={getScoreColor(score) + '.main'} fontWeight="bold">
                      {percentage}%
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      適合度スコア
                    </Typography>
                  </Box>
                </Box>

                {/* スコアバー */}
                <Box sx={{ mb: 2 }}>
                  <LinearProgress
                    variant="determinate"
                    value={percentage}
                    color={getScoreColor(score)}
                    sx={{ height: 12, borderRadius: 6 }}
                  />
                </Box>

                {/* 推薦レベル */}
                {labResult.recommendation_level && (
                  <Box sx={{ mb: 2 }}>
                    <Chip
                      label={labResult.recommendation_level}
                      color={score >= 0.8 ? 'success' : score >= 0.6 ? 'warning' : 'default'}
                      variant="filled"
                    />
                  </Box>
                )}

                {/* 簡易説明 */}
                {labResult.explanation && (
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    {labResult.explanation}
                  </Typography>
                )}

                {/* 展開ボタン */}
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="body2" color="text.secondary">
                    詳細分析を表示
                  </Typography>
                  <IconButton
                    onClick={() => handleExpandClick(index)}
                    aria-expanded={isExpanded}
                    aria-label="詳細表示"
                  >
                    {isExpanded ? <ExpandLess /> : <ExpandMore />}
                  </IconButton>
                </Box>

                {/* 展開可能な詳細情報 */}
                <Collapse in={isExpanded} timeout="auto" unmountOnExit>
                  <Divider sx={{ my: 2 }} />

                  {labResult.detailed_analysis && (
                    <Box sx={{ mt: 2 }}>
                      {/* 強み */}
                      {labResult.detailed_analysis.strengths && labResult.detailed_analysis.strengths.length > 0 && (
                        <Box sx={{ mb: 2 }}>
                          <Typography variant="subtitle2" color="success.main" gutterBottom>
                            ✅ 強み・適合点
                          </Typography>
                          <List dense>
                            {labResult.detailed_analysis.strengths.map((strength, idx) => (
                              <ListItem key={idx} sx={{ py: 0.5 }}>
                                <ListItemIcon sx={{ minWidth: 32 }}>
                                  <StarRate color="success" fontSize="small" />
                                </ListItemIcon>
                                <ListItemText primary={strength} />
                              </ListItem>
                            ))}
                          </List>
                        </Box>
                      )}

                      {/* 懸念点 */}
                      {labResult.detailed_analysis.concerns && labResult.detailed_analysis.concerns.length > 0 && (
                        <Box sx={{ mb: 2 }}>
                          <Typography variant="subtitle2" color="warning.main" gutterBottom>
                            ⚠️ 検討が必要な点
                          </Typography>
                          <List dense>
                            {labResult.detailed_analysis.concerns.map((concern, idx) => (
                              <ListItem key={idx} sx={{ py: 0.5 }}>
                                <ListItemIcon sx={{ minWidth: 32 }}>
                                  <Assessment color="warning" fontSize="small" />
                                </ListItemIcon>
                                <ListItemText primary={concern} />
                              </ListItem>
                            ))}
                          </List>
                        </Box>
                      )}

                      {/* 推薦事項 */}
                      {labResult.detailed_analysis.recommendations && labResult.detailed_analysis.recommendations.length > 0 && (
                        <Box sx={{ mb: 2 }}>
                          <Typography variant="subtitle2" color="primary.main" gutterBottom>
                            💡 推薦事項
                          </Typography>
                          <List dense>
                            {labResult.detailed_analysis.recommendations.map((recommendation, idx) => (
                              <ListItem key={idx} sx={{ py: 0.5 }}>
                                <ListItemIcon sx={{ minWidth: 32 }}>
                                  <School color="primary" fontSize="small" />
                                </ListItemIcon>
                                <ListItemText primary={recommendation} />
                              </ListItem>
                            ))}
                          </List>
                        </Box>
                      )}

                      {/* 基準別スコア */}
                      {labResult.detailed_analysis.criteria_scores && (
                        <Box>
                          <Typography variant="subtitle2" color="text.primary" gutterBottom>
                            📊 評価基準別スコア
                          </Typography>
                          <Grid container spacing={2}>
                            {Object.entries(labResult.detailed_analysis.criteria_scores).map(([criterion, data]: [string, any]) => (
                              <Grid item xs={12} sm={6} key={criterion}>
                                <Paper variant="outlined" sx={{ p: 2 }}>
                                  <Typography variant="body2" fontWeight="bold" gutterBottom>
                                    {criterion}
                                  </Typography>
                                  {data && typeof data === 'object' && 'match_score' in data ? (
                                    <>
                                      <ScoreDisplay
                                        score={data.match_score || 0}
                                        label="マッチ度"
                                      />
                                      {data.interpretation && (
                                        <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                                          {data.interpretation}
                                        </Typography>
                                      )}
                                    </>
                                  ) : (
                                    <Typography variant="body2" color="text.secondary">
                                      データなし
                                    </Typography>
                                  )}
                                </Paper>
                              </Grid>
                            ))}
                          </Grid>
                        </Box>
                      )}
                    </Box>
                  )}

                  {/* デバッグ情報（開発時のみ表示） */}
                  {process.env.NODE_ENV === 'development' && (
                    <Box sx={{ mt: 2, p: 2, bgcolor: 'grey.100', borderRadius: 1 }}>
                      <Typography variant="caption" color="text.secondary">
                        デバッグ情報: compatibility_score={labResult.compatibility_score}, final_score={labResult.final_score}
                      </Typography>
                    </Box>
                  )}
                </Collapse>
              </CardContent>
            </Card>
          );
        })}
      </Box>

      {/* メタデータ情報 */}
      {evaluationResponse.metadata && (
        <Card sx={{ mt: 3, bgcolor: 'grey.50' }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              📈 評価メタデータ
            </Typography>
            <Grid container spacing={2}>
              {evaluationResponse.metadata.timestamp && (
                <Grid item xs={6} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    評価実行時間
                  </Typography>
                  <Typography variant="body2">
                    {new Date(evaluationResponse.metadata.timestamp).toLocaleString('ja-JP')}
                  </Typography>
                </Grid>
              )}
              {evaluationResponse.metadata.criteria_used && (
                <Grid item xs={6} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    使用基準数
                  </Typography>
                  <Typography variant="body2">
                    {evaluationResponse.metadata.criteria_used}項目
                  </Typography>
                </Grid>
              )}
              {evaluationResponse.metadata.calculation_method && (
                <Grid item xs={6} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    計算方法
                  </Typography>
                  <Typography variant="body2">
                    {evaluationResponse.metadata.calculation_method}
                  </Typography>
                </Grid>
              )}
              {evaluationResponse.metadata.evaluation_count && (
                <Grid item xs={6} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    評価回数
                  </Typography>
                  <Typography variant="body2">
                    {evaluationResponse.metadata.evaluation_count}回目
                  </Typography>
                </Grid>
              )}
            </Grid>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default ResultsList;