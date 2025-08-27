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

interface ResultsListProps {
  data: EvaluationResponse;
}

const ResultsList: React.FC<ResultsListProps> = ({ data }) => {
  const { results, summary } = data;
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
    theory_practice: '理論・実践',
    research_field_match: '分野適合性',
    skill_development: 'スキル開発',
    learning_pace: '学習ペース',
    difficulty_preference: '難易度選好',
    communication_style: 'コミュニケーション',
    meeting_frequency: 'ミーティング頻度',
    lab_atmosphere: '研究室雰囲気',
    innovation_risk: '革新性・リスク',
    methodology_preference: '手法選好',
    interdisciplinary: '学際性',
    flexibility: '柔軟性',
    evening_weekend_work: '夜間・休日作業',
    publication_opportunity: '論文機会',
    financial_support: '経済支援',
    lab_hierarchy: '研究室階層',
    core_time_flexibility: 'コアタイム柔軟性'
  };

  const criteriaEmojis = {
    research_intensity: '🔬',
    advisor_style: '👨‍🏫',
    team_work: '🤝',
    workload: '⚡',
    theory_practice: '⚖️',
    research_field_match: '🎯',
    skill_development: '📈',
    learning_pace: '🏃',
    difficulty_preference: '🎢',
    communication_style: '💬',
    meeting_frequency: '📅',
    lab_atmosphere: '🌟',
    innovation_risk: '🚀',
    methodology_preference: '🔧',
    interdisciplinary: '🌐',
    flexibility: '🤸',
    evening_weekend_work: '🌙',
    publication_opportunity: '📝',
    financial_support: '💰',
    lab_hierarchy: '👥',
    core_time_flexibility: '⏰'
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
                  {hasFieldAnalysis ? (summary.field_analysis!.selected_fields_count || 0) : '0'}
                </Typography>
                <Typography variant="body2" sx={{ opacity: 0.9 }}>
                  選択分野数
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Paper>

      {/* 研究室結果カード */}
      <Box>
        {results.map((result, index) => (
          <Card key={result.lab.id} sx={{ mb: 3, position: 'relative' }}>
            <CardContent>
              {/* ヘッダー部分 */}
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Avatar sx={{ bgcolor: getScoreColor(result.compatibility.overall_score) === 'success' ? 'success.main' : 
                                       getScoreColor(result.compatibility.overall_score) === 'warning' ? 'warning.main' : 'error.main' }}>
                    <School />
                  </Avatar>
                  <Box>
                    <Typography variant="h6" fontWeight="bold">
                      {getRankIcon(index + 1)} {result.lab.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {result.lab.professor} 教授 | {result.lab.research_area}
                    </Typography>
                  </Box>
                </Box>
                
                {/* スコア表示 */}
                <Box sx={{ textAlign: 'right' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="h4" fontWeight="bold" 
                      color={getScoreColor(result.compatibility.overall_score) + '.main'}>
                      {result.compatibility.overall_score.toFixed(1)}
                    </Typography>
                    <Typography variant="h6">
                      {getScoreIcon(result.compatibility.overall_score)}
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="text.secondary">
                    適合度スコア
                  </Typography>
                </Box>
              </Box>

              {/* プログレスバー */}
              <LinearProgress
                variant="determinate"
                value={result.compatibility.overall_score}
                color={getScoreColor(result.compatibility.overall_score)}
                sx={{ height: 8, borderRadius: 4, mb: 2 }}
              />

              {/* 基本情報 */}
              <Grid container spacing={2} sx={{ mb: 2 }}>
                <Grid item xs={12} sm={6}>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    <Psychology sx={{ fontSize: 16, mr: 0.5, verticalAlign: 'middle' }} />
                    専門分野
                  </Typography>
                  <Typography variant="body1">{result.lab.research_area}</Typography>
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    <Person sx={{ fontSize: 16, mr: 0.5, verticalAlign: 'middle' }} />
                    研究室情報
                  </Typography>
                  <Typography variant="body1">
                    {result.lab.description || '研究室情報'} | 規模情報
                  </Typography>
                </Grid>
              </Grid>

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
                        📊 適合度スコア
                      </Typography>
                      <List dense>
                        {Object.entries(result.compatibility.criterion_scores).slice(0, 8).map(([criterion, score]) => (
                          <ListItem key={criterion}>
                            <ListItemIcon>
                              <Typography>
                                {criteriaEmojis[criterion as keyof typeof criteriaEmojis] || '📊'}
                              </Typography>
                            </ListItemIcon>
                            <ListItemText
                              primary={criteriaLabels[criterion as keyof typeof criteriaLabels] || criterion}
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
                      🏷️ 研究室特徴値（上位8項目）
                    </Typography>
                    <Grid container spacing={2}>
                      {Object.entries(result.lab.features).slice(0, 8).map(([feature, value]) => (
                        <Grid item xs={12} sm={6} md={4} key={feature}>
                          <Paper sx={{ p: 2, textAlign: 'center' }}>
                            <Typography variant="body2" color="text.secondary">
                              {criteriaLabels[feature as keyof typeof criteriaLabels] || feature}
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
      </Box>

      {/* 評価結果の解釈と推薦事項 */}
      <Paper sx={{ p: 4, mt: 4, background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)' }}>
        <Typography variant="h5" gutterBottom>
          🎯 評価結果の解釈
        </Typography>
        
        <Typography variant="body1" paragraph>
          システム全体での平均適合度は<strong>{summary.avg_score.toFixed(1)}%</strong>です。
          {summary.avg_score >= 75 ?
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

export default ResultsList;