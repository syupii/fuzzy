// frontend/src/components/ResultsList.tsx - 型定義を修正しエラーを解消
import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Chip,
  Grid,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Alert,
  LinearProgress,
  Button,
  Divider,
  Tooltip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  ExpandMore,
  Star,
  Info,
  CompareArrows,
  Close,
} from '@mui/icons-material';

// ★★★ 修正点1: api.tsからLabResult型をインポート ★★★
import { LabResult } from '../services/api';

// ★★★ 修正点2: ローカルのLabResult型定義を削除 ★★★
/*
interface LabResult {
  // ... このブロック全体を削除 ...
}
*/

interface ResultsListProps {
  results: LabResult[];
  metadata?: any;
  onLabSelected?: (lab: LabResult) => void;
}

const ResultsList: React.FC<ResultsListProps> = ({ results, metadata, onLabSelected }) => {
  const [selectedLab, setSelectedLab] = useState<LabResult | null>(null);
  const [detailDialogOpen, setDetailDialogOpen] = useState(false);
  const [expandedCard, setExpandedCard] = useState<string | null>(null);

  const criteriaNameMap: { [key: string]: string } = {
    'research_intensity': '研究強度', 'advisor_style': '指導スタイル', 'team_work': 'チームワーク',
    'workload': 'ワークロード', 'theory_practice': '理論・実践バランス', 'research_field_match': '研究分野適合性',
    'skill_development': 'スキル開発', 'lab_atmosphere': '研究室雰囲気', 'flexibility': '柔軟性',
    'publication_opportunity': '論文発表機会', 'interdisciplinary': '学際性', 'communication_style': 'コミュニケーション'
  };

  const getScoreColor = (score: number): 'success' | 'warning' | 'info' | 'error' => {
    if (score >= 0.8) return 'success';
    if (score >= 0.6) return 'warning';
    if (score >= 0.4) return 'info';
    return 'error';
  };

  const getRecommendationColor = (recommendation?: string): 'success' | 'warning' | 'info' | 'error' => {
    switch (recommendation) {
      case '最優先推薦': case '強く推薦': return 'success';
      case '優先推薦': case '推薦': return 'warning';
      case '検討可能': return 'info';
      default: return 'error';
    }
  };

  const handleLabDetail = (lab: LabResult) => {
    setSelectedLab(lab);
    setDetailDialogOpen(true);
    if (onLabSelected) onLabSelected(lab);
  };

  const handleCardToggle = (labId: string) => {
    setExpandedCard(expandedCard === labId ? null : labId);
  };

  const renderLabCard = (lab: LabResult, index: number) => {
    const isExpanded = expandedCard === lab.lab_id;
    const score = lab.final_score || lab.overall_compatibility || 0;

    return (
      <Card key={lab.lab_id} sx={{ mb: 2, border: index < 3 ? 2 : 1, borderColor: index < 3 ? 'primary.main' : 'divider', position: 'relative', cursor: 'pointer' }} onClick={() => handleCardToggle(lab.lab_id)}>
        <CardContent>
          {index < 3 && <Chip label={`第${index + 1}位`} color="primary" sx={{ position: 'absolute', top: 8, right: 8, fontWeight: 'bold' }} />}
          <Typography variant="h6" gutterBottom sx={{ pr: 8 }}>{lab.lab_name}</Typography>
          {/* ★★★ 修正点3: オプショナルなプロパティを安全に表示 ★★★ */}
          <Typography variant="body2" color="text.secondary" gutterBottom>{lab.advisor || lab.professor_name || '担当教員未設定'} | {lab.research_area || '研究分野未設定'}</Typography>
          {lab.category && <Chip label={lab.category} size="small" variant="outlined" sx={{ mb: 2 }} />}
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={6}>
              <Box>
                <Typography variant="body2" color="text.secondary">統合適合度</Typography>
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Typography variant="h5" color={`${getScoreColor(score)}.main`}>{(score * 100).toFixed(1)}%</Typography>
                  {lab.priority_adjusted_score && lab.priority_adjusted_score !== score && (
                    <Tooltip title="優先度による調整済み"><Star sx={{ ml: 1, fontSize: 16, color: 'warning.main' }} /></Tooltip>
                  )}
                </Box>
              </Box>
            </Grid>
            <Grid item xs={6}>
              <Box>
                <Typography variant="body2" color="text.secondary">信頼度</Typography>
                <Typography variant="h6">{((lab.confidence || 0.8) * 100).toFixed(1)}%</Typography>
              </Box>
            </Grid>
          </Grid>

          {isExpanded && lab.priority_analysis && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" gutterBottom>優先度適合分析:</Typography>
              <Grid container spacing={1}>
                <Grid item xs={4}><Typography variant="caption" color="error">高: {(lab.priority_analysis.high_priority_match * 100).toFixed(0)}%</Typography><LinearProgress variant="determinate" value={lab.priority_analysis.high_priority_match * 100} color="error" sx={{ height: 4, borderRadius: 2 }} /></Grid>
                <Grid item xs={4}><Typography variant="caption" color="warning">中: {(lab.priority_analysis.medium_priority_match * 100).toFixed(0)}%</Typography><LinearProgress variant="determinate" value={lab.priority_analysis.medium_priority_match * 100} color="warning" sx={{ height: 4, borderRadius: 2 }} /></Grid>
                <Grid item xs={4}><Typography variant="caption" color="info">低: {(lab.priority_analysis.low_priority_match * 100).toFixed(0)}%</Typography><LinearProgress variant="determinate" value={lab.priority_analysis.low_priority_match * 100} color="info" sx={{ height: 4, borderRadius: 2 }} /></Grid>
              </Grid>
            </Box>
          )}

          <Box sx={{ mb: 2 }}>
            <Chip label={lab.recommendation || '評価なし'} color={getRecommendationColor(lab.recommendation)} variant="filled" />
          </Box>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>{lab.explanation || '詳細な説明はありません。'}</Typography>

          {isExpanded && (
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Button variant="outlined" size="small" startIcon={<Info />} onClick={(e) => { e.stopPropagation(); handleLabDetail(lab); }}>詳細分析</Button>
              <Button variant="text" size="small" startIcon={<CompareArrows />}>比較に追加</Button>
            </Box>
          )}
          <Box sx={{ textAlign: 'center', mt: 1 }}><ExpandMore sx={{ transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.3s' }} /></Box>
        </CardContent>
      </Card>
    );
  };

  const renderDetailDialog = () => (
    <Dialog open={detailDialogOpen} onClose={() => setDetailDialogOpen(false)} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h6">{selectedLab?.lab_name} - 詳細分析</Typography>
          <IconButton onClick={() => setDetailDialogOpen(false)}><Close /></IconButton>
        </Box>
      </DialogTitle>
      <DialogContent>
        {selectedLab && (
          <Box>
            <Typography variant="h6" gutterBottom>基本情報</Typography>
            <Typography variant="body2">
              指導教員: {selectedLab.advisor || selectedLab.professor_name}<br />
              研究分野: {selectedLab.research_area}<br />
              カテゴリ: {selectedLab.category}
            </Typography>
            <Divider sx={{ my: 2 }} />
            <Typography variant="h6" gutterBottom>スコア詳細分析</Typography>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <Card variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>統合適合度</Typography>
                  <Typography variant="h4" color="primary">{((selectedLab.final_score || 0) * 100).toFixed(1)}%</Typography>
                  <LinearProgress variant="determinate" value={(selectedLab.final_score || 0) * 100} sx={{ mt: 1 }} />
                </Card>
              </Grid>
              <Grid item xs={6}>
                <Card variant="outlined" sx={{ p: 2 }}>
                  <Typography variant="subtitle2" gutterBottom>信頼度</Typography>
                  <Typography variant="h4" color="secondary">{((selectedLab.confidence || 0.8) * 100).toFixed(1)}%</Typography>
                  <LinearProgress variant="determinate" value={(selectedLab.confidence || 0.8) * 100} color="secondary" sx={{ mt: 1 }} />
                </Card>
              </Grid>
            </Grid>
            <Divider sx={{ my: 2 }} />
            <Typography variant="h6" gutterBottom>項目別適合度分析</Typography>
            <Grid container spacing={1}>
              {/* ★★★ 修正点4: feature_scoresが存在する場合のみ実行 ★★★ */}
              {selectedLab.feature_scores && Object.entries(selectedLab.feature_scores).map(([criterion, score]) => (
                <Grid item xs={12} sm={6} key={criterion}>
                  <Box sx={{ mb: 1 }}>
                    <Typography variant="body2" gutterBottom>{criteriaNameMap[criterion] || criterion}</Typography>
                    <LinearProgress variant="determinate" value={score * 100} color={getScoreColor(score)} sx={{ height: 8, borderRadius: 4 }} />
                    <Typography variant="caption" color="text.secondary">{(score * 100).toFixed(1)}%</Typography>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </Box>
        )}
      </DialogContent>
      <DialogActions><Button onClick={() => setDetailDialogOpen(false)}>閉じる</Button></DialogActions>
    </Dialog>
  );

  return (
    <Box>
      <Typography variant="h4" gutterBottom color="primary">🎯 研究室適合度評価結果</Typography>

      {metadata && (
        <Accordion sx={{ mb: 3 }}>
          <AccordionSummary expandIcon={<ExpandMore />}><Typography variant="subtitle1">処理詳細情報</Typography></AccordionSummary>
          <AccordionDetails>
            <Typography variant="body2" component="pre" sx={{ whiteSpace: 'pre-wrap' }}>
              評価エンジン: {metadata.ai_engines_used?.join(', ') || '基本アルゴリズム'}<br />
              処理時間: {metadata.processing_time?.toFixed(3) || '不明'}秒<br />
              タイムスタンプ: {metadata.timestamp || '不明'}
            </Typography>
          </AccordionDetails>
        </Accordion>
      )}

      <Typography variant="h5" gutterBottom>研究室適合度ランキング</Typography>

      {!results || results.length === 0 ? (
        <Alert severity="warning">評価結果がありません。評価を実行してください。</Alert>
      ) : (
        <Box>{results.map((lab, index) => renderLabCard(lab, index))}</Box>
      )}

      {renderDetailDialog()}
    </Box>
  );
};

export default ResultsList;