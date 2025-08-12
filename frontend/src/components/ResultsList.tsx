import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Chip,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Avatar,
  Paper,
  Divider,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Tooltip,
} from '@mui/material';
import {
  ExpandMore,
  School,
  Person,
  Psychology,
  EmojiEvents,
  TrendingUp,
  Science,
  Category,
  StarRate,
  Assessment,
} from '@mui/icons-material';
import { EvaluationResponse, fieldUtils } from '../services/api';

interface EnhancedResultsListProps {
  data: EvaluationResponse;
}

const EnhancedResultsList: React.FC<EnhancedResultsListProps> = ({ data }) => {
  const { results, summary, algorithm_info } = data;
  const hasFieldAnalysis = summary.field_analysis && summary.field_analysis.selected_fields_count > 0;

  const getScoreColor = (score: number): 'success' | 'warning' | 'error' => {
    if (score >= 80) return 'success';
    if (score >= 60) return 'warning';
    return 'error';
  };

  const getScoreIcon = (score: number) => {
    if (score >= 85) return '🎯';
    if (score >= 70) return '✅';
    if (score >= 50) return '👍';
    return '⚠️';
  };

  const getRankIcon = (rank: number) => {
    if (rank === 1) return '🥇';
    if (rank === 2) return '🥈';
    if (rank === 3) return '🥉';
    return `${rank}位`;
  };

  const criteriaLabels = {
    research_intensity: '研究強度',
    advisor_style: '指導スタイル',
    team_work: 'チームワーク',
    workload: 'ワークロード',
    theory_practice: '理論・実践'
  };

  const criteriaEmojis = {
    research_intensity: '🔬',
    advisor_style: '👨‍🏫',
    team_work: '🤝',
    workload: '⚡',
    theory_practice: '⚖️'
  };

  return (
    <Box>
      {/* 拡張サマリーカード */}
      <Paper elevation={2} sx={{ mb: 4, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' }}>
        <CardContent sx={{ color: 'white' }}>
          <Box sx={{ textAlign: 'center', mb: 3 }}>
            <EmojiEvents sx={{ fontSize: 48, mb: 1 }} />
            <Typography variant="h5" gutterBottom>
              📊 評価サマリー
            </Typography>
          </Box>
          
          <Grid container spacing={3}>
            <Grid item xs={12} sm={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h3" fontWeight="bold">
                  {summary.total_labs}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  評価対象研究室
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h3" fontWeight="bold">
                  {summary.avg_score.toFixed(1)}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  平均適合度
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h3" fontWeight="bold">
                  {results.length > 0 ? results[0].compatibility.overall_score.toFixed(1) : '0'}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  最高適合度
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h3" fontWeight="bold">
                  {hasFieldAnalysis ? summary.field_analysis!.selected_fields_count : 0}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  分析分野数
                </Typography>
              </Box>
            </Grid>
          </Grid>

          {/* 研究分野分析結果 */}
          {hasFieldAnalysis && (
            <Box sx={{ mt: 3, p: 2, backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: 1 }}>
              <Typography variant="h6" gutterBottom>
                🎨 研究分野分析結果
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={4}>
                  <Typography variant="body2">
                    平均興味度: <strong>{summary.field_analysis!.average_interest?.toFixed(1) || '0.0'}</strong>
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={4}>
                  <Typography variant="body2">
                    主要カテゴリー: <strong>{summary.field_analysis!.primary_category || '未設定'}</strong>
                  </Typography>
                </Grid>
                <Grid item xs={12} sm={4}>
                  <Typography variant="body2">
                    適合研究室率: <strong>{((summary.field_analysis!.field_coverage || 0) * 100).toFixed(0)}%</strong>
                  </Typography>
                </Grid>
              </Grid>
            </Box>
          )}

          {/* アルゴリズム情報 */}
          <Box sx={{ mt: 3, p: 2, backgroundColor: 'rgba(255,255,255,0.1)', borderRadius: 1 }}>
            <Typography variant="body2" sx={{ opacity: 0.9 }}>
              🤖 {algorithm_info.engine} | 
              セッション: {summary.session_id.slice(-8)} | 
              評価ID: {summary.evaluation_id}
            </Typography>
          </Box>
        </CardContent>
      </Paper>

      {/* 研究室結果一覧 */}
      <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
        🏆 研究室ランキング
      </Typography>

      {results.map((result, index) => (
        <Card key={result.lab.id} elevation={3} sx={{ mb: 3 }}>
          <CardContent>
            {/* ヘッダー部分 */}
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <Avatar
                sx={{
                  bgcolor: getScoreColor(result.compatibility.overall_score) + '.main',
                  mr: 2,
                  width: 56,
                  height: 56,
                  fontSize: '1.5rem'
                }}
              >
                {getRankIcon(index + 1)}
              </Avatar>
              
              <Box sx={{ flexGrow: 1 }}>
                <Typography variant="h5" gutterBottom>
                  {getScoreIcon(result.compatibility.overall_score)} {result.lab.name}
                </Typography>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
                  <Chip
                    icon={<Person />}
                    label={result.lab.professor}
                    variant="outlined"
                  />
                  <Chip
                    icon={<Assessment />}
                    label={`適合度: ${result.compatibility.overall_score.toFixed(1)}%`}
                    color={getScoreColor(result.compatibility.overall_score)}
                  />
                  <Chip
                    icon={<Psychology />}
                    label={`信頼度: ${(result.compatibility.confidence * 100).toFixed(0)}%`}
                    variant="outlined"
                  />
                </Box>
              </Box>
            </Box>

            {/* 研究分野情報 */}
            <Typography variant="body1" color="text.secondary" paragraph>
              📚 <strong>研究領域:</strong> {result.lab.research_area}
            </Typography>
            <Typography variant="body2" color="text.secondary" paragraph>
              {result.lab.description}
            </Typography>

            {/* 詳細分析 */}
            <Accordion>
              <AccordionSummary expandIcon={<ExpandMore />}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <TrendingUp />
                  <Typography variant="h6">詳細分析</Typography>
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <Grid container spacing={3}>
                  {/* 基本項目スコア */}
                  <Grid item xs={12} md={6}>
                    <Typography variant="h6" gutterBottom>
                      📊 基本項目スコア
                    </Typography>
                    <List dense>
                      {Object.entries(result.compatibility.criterion_scores).map(([criterion, score]) => (
                        <ListItem key={criterion}>
                          <ListItemIcon>
                            <Typography>
                              {criteriaEmojis[criterion as keyof typeof criteriaEmojis]}
                            </Typography>
                          </ListItemIcon>
                          <ListItemText
                            primary={criteriaLabels[criterion as keyof typeof criteriaLabels]}
                            secondary={
                              <Box sx={{ mt: 1 }}>
                                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                  <Typography variant="body2">
                                    類似度: {(score.similarity * 100).toFixed(0)}%
                                  </Typography>
                                  <Typography variant="body2">
                                    重み: {score.weight.toFixed(2)}
                                  </Typography>
                                </Box>
                                <LinearProgress
                                  variant="determinate"
                                  value={score.similarity * 100}
                                  color={score.similarity >= 0.7 ? 'success' : score.similarity >= 0.5 ? 'warning' : 'error'}
                                />
                              </Box>
                            }
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Grid>

                  {/* 研究分野マッチング */}
                  {result.compatibility.field_matching && (
                    <Grid item xs={12} md={6}>
                      <Typography variant="h6" gutterBottom>
                        🎨 研究分野マッチング
                      </Typography>
                      
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="body2" gutterBottom>
                          マッチした分野: {result.compatibility.field_matching.matched_fields?.length || 0}個
                        </Typography>
                        <Typography variant="body2" gutterBottom>
                          分野重み: {result.compatibility.field_matching.field_weight?.toFixed(2) || '0.00'}
                        </Typography>
                      </Box>

                      {result.compatibility.field_matching.matched_fields && result.compatibility.field_matching.matched_fields.length > 0 && (
                        <Box sx={{ mb: 2 }}>
                          <Typography variant="body2" gutterBottom>マッチした分野:</Typography>
                          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                            {result.compatibility.field_matching.matched_fields.map(fieldId => (
                              <Tooltip key={fieldId} title={`スコア: ${result.compatibility.field_matching!.field_scores?.[fieldId]?.toFixed(1) || 'N/A'}`}>
                                <Chip
                                  label={fieldUtils.getFieldName(fieldId)}
                                  size="small"
                                  color="primary"
                                  variant="outlined"
                                />
                              </Tooltip>
                            ))}
                          </Box>
                        </Box>
                      )}

                      {result.compatibility.field_matching.field_scores && Object.keys(result.compatibility.field_matching.field_scores).length > 0 && (
                        <Box>
                          <Typography variant="body2" gutterBottom>分野別スコア:</Typography>
                          {Object.entries(result.compatibility.field_matching.field_scores).map(([fieldId, score]) => (
                            <Box key={fieldId} sx={{ mb: 1 }}>
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                <Typography variant="body2">
                                  {fieldUtils.getFieldName(fieldId)}
                                </Typography>
                                <Typography variant="body2" fontWeight="bold">
                                  {score?.toFixed(1) || '0.0'}
                                </Typography>
                              </Box>
                              <LinearProgress
                                variant="determinate"
                                value={(score || 0) * 10} // 0-10スケールを0-100%に変換
                                color={score && score >= 7 ? 'success' : score && score >= 5 ? 'warning' : 'error'}
                                sx={{ height: 4, borderRadius: 2 }}
                              />
                            </Box>
                          ))}
                        </Box>
                      )}
                    </Grid>
                  )}
                </Grid>

                {/* 説明テキスト */}
                <Divider sx={{ my: 2 }} />
                <Box>
                  <Typography variant="h6" gutterBottom>
                    💭 適合理由
                  </Typography>
                  <Typography variant="body2" sx={{ whiteSpace: 'pre-line' }}>
                    {result.compatibility.explanation}
                  </Typography>
                </Box>

                {/* 研究室特徴値 */}
                <Divider sx={{ my: 2 }} />
                <Box>
                  <Typography variant="h6" gutterBottom>
                    🏷️ 研究室特徴値
                  </Typography>
                  <Grid container spacing={2}>
                    {Object.entries(result.lab.features).map(([feature, value]) => (
                      <Grid item xs={12} sm={6} md={4} key={feature}>
                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                          <Typography variant="body2" color="text.secondary">
                            {criteriaLabels[feature as keyof typeof criteriaLabels]}
                          </Typography>
                          <Typography variant="h6" color="primary">
                            {value.toFixed(1)}
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={value * 10}
                            sx={{ mt: 1, height: 6, borderRadius: 3 }}
                          />
                        </Paper>
                      </Grid>
                    ))}
                  </Grid>
                </Box>
              </AccordionDetails>
            </Accordion>
          </CardContent>
        </Card>
      ))}

      {/* 総評・推薦セクション */}
      <Paper sx={{ p: 4, mt: 4, backgroundColor: '#f8f9fa' }}>
        <Typography variant="h5" gutterBottom>
          📝 総合評価・推薦
        </Typography>
        
        <Typography variant="body1" paragraph>
          <strong>🎯 最適な研究室:</strong> {summary.best_match}
        </Typography>
        
        <Typography variant="body1" paragraph>
          システム全体での平均適合度は<strong>{summary.avg_score.toFixed(1)}%</strong>です。
          {summary.avg_score >= 70 ? 
            '優秀な適合度を示しており、複数の選択肢から検討することをお勧めします。' :
            summary.avg_score >= 50 ?
            '中程度の適合度です。上位の研究室について詳しく調べることをお勧めします。' :
            '適合度が低めです。設定を見直すか、研究室見学で直接確認することをお勧めします。'
          }
        </Typography>

        {hasFieldAnalysis && (
          <Typography variant="body1" paragraph>
            研究分野分析では、<strong>{summary.field_analysis!.selected_fields_count}分野</strong>を対象に
            平均興味度<strong>{summary.field_analysis!.average_interest?.toFixed(1) || '0.0'}</strong>で評価を行いました。
            主要カテゴリーは<strong>「{summary.field_analysis!.primary_category || '未設定'}」</strong>で、
            {((summary.field_analysis!.field_coverage || 0) * 100).toFixed(0)}%の研究室が
            あなたの選択分野に対応しています。
          </Typography>
        )}

        <Box sx={{ mt: 3, p: 3, backgroundColor: 'white', borderRadius: 2, borderLeft: '4px solid #1976d2' }}>
          <Typography variant="h6" gutterBottom color="primary">
            📚 次のステップ
          </Typography>
          <Box component="ul" sx={{ pl: 3 }}>
            <li>上位の研究室について研究内容を詳しく調査</li>
            <li>研究室見学や教授との面談を申し込み</li>
            <li>現在の学生や卒業生から話を聞く</li>
            <li>自分の興味や将来の目標と照らし合わせて最終判断</li>
            {hasFieldAnalysis && (
              <li>選択した研究分野に関連する最新の研究動向を調査</li>
            )}
          </Box>
        </Box>

        <Box sx={{ mt: 2, textAlign: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            💡 このシステムは参考情報として活用し、最終的な判断は総合的に行ってください
          </Typography>
        </Box>
      </Paper>
    </Box>
  );
};

export default EnhancedResultsList;