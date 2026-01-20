// frontend/src/components/ResultsList.tsx - パターンB説明表示対応版
// ★★★ 元のコードを完全維持 + explanation_detailed/explanation_short表示を追加 ★★★
import React, { useState } from 'react';
import {
    Box,
    Card,
    CardContent,
    Typography,
    Chip,
    Grid,
    Alert,
    LinearProgress,
    Button,
    Divider,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Paper,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    IconButton,
    Collapse,
    Tooltip,
} from '@mui/material';
import {
    ExpandMore,
    Star,
    Info,
    CompareArrows,
    Close,
    Description,
    LocalLibrary,
    TrendingUp,
    CheckCircle,
    Warning,
} from '@mui/icons-material';

// 型定義
interface LabResult {
    lab_id?: string;
    lab_name: string;
    advisor?: string;
    professor?: string;
    professor_name?: string;
    research_area?: string;
    field_name?: string;
    field_id?: string;
    category?: string;

    final_score?: number;
    overall_compatibility?: number;
    basic_score?: number;
    field_score?: number;
    confidence?: number;
    priority_adjusted_score?: number;

    feature_scores?: Record<string, number>;
    criteria_scores?: Record<string, number>;

    recommendation?: string;
    explanation?: string;

    // ★★★ 追加: 詳細版・短縮版の説明 ★★★
    explanation_detailed?: string;
    explanation_short?: string;

    priority_analysis?: {
        high_priority_match: number;
        medium_priority_match: number;
        low_priority_match: number;
    };

    description?: string;
    specialization?: string;
    research_fields?: string[];

    features?: Record<string, number>;
}

interface ResultsListProps {
    results: LabResult[];
    metadata?: any;
    studentProfile?: any;
    onLabSelected?: (lab: LabResult) => void;
}

const ResultsList: React.FC<ResultsListProps> = ({ results, metadata, studentProfile, onLabSelected }) => {
    const [selectedLab, setSelectedLab] = useState<LabResult | null>(null);
    const [detailDialogOpen, setDetailDialogOpen] = useState(false);
    const [expandedCard, setExpandedCard] = useState<string | null>(null);

    const criteriaNameMap: { [key: string]: string } = {
        'research_intensity': '研究強度',
        'advisor_style': '指導スタイル',
        'team_work': 'チームワーク',
        'workload': 'ワークロード',
        'theory_practice': '理論・実践バランス',
        'skill_development': 'スキル開発',
        'lab_atmosphere': '研究室雰囲気',
        'flexibility': '柔軟性',
        'publication_opportunity': '論文発表機会',
        'interdisciplinary': '学際性',
        'communication_style': 'コミュニケーション'
    };

    const getScoreColor = (score: number): 'success' | 'warning' | 'info' | 'error' => {
        if (score >= 0.8) return 'success';
        if (score >= 0.6) return 'warning';
        if (score >= 0.4) return 'info';
        return 'error';
    };

    const getScoreGradient = (score: number): string => {
        if (score >= 0.8) return 'linear-gradient(135deg, #66bb6a 0%, #43a047 100%)';
        if (score >= 0.6) return 'linear-gradient(135deg, #ffa726 0%, #fb8c00 100%)';
        if (score >= 0.4) return 'linear-gradient(135deg, #42a5f5 0%, #1e88e5 100%)';
        return 'linear-gradient(135deg, #ef5350 0%, #e53935 100%)';
    };

    const scaleToDisplay = (normalizedValue: number): number => {
        return Math.round(normalizedValue * 9 + 1);
    };

    const handleLabDetail = (lab: LabResult) => {
        setSelectedLab(lab);
        setDetailDialogOpen(true);
        if (onLabSelected) onLabSelected(lab);
    };

    const handleCardToggle = (labId: string) => {
        setExpandedCard(expandedCard === labId ? null : labId);
    };

    // ★★★ 追加: 説明文を取得するヘルパー関数 ★★★
    const getExplanation = (lab: LabResult, type: 'detailed' | 'short' | 'legacy' = 'detailed'): string => {
        if (type === 'detailed') {
            return lab.explanation_detailed || lab.explanation || '';
        } else if (type === 'short') {
            return lab.explanation_short || lab.explanation || '';
        }
        return lab.explanation || '';
    };

    const renderLabCard = (lab: LabResult, index: number) => {
        const isExpanded = expandedCard === (lab.lab_id || lab.lab_name);
        const score = lab.overall_compatibility || lab.final_score || 0;
        const professorName = lab.professor || lab.advisor || lab.professor_name || '担当教員未設定';
        const researchArea = lab.field_name || lab.research_area || '研究分野未設定';

        return (
            <Card
                key={lab.lab_id || lab.lab_name}
                sx={{
                    mb: 3,
                    border: 2,
                    borderColor: 'divider',
                    borderRadius: 3,
                    overflow: 'visible',
                    position: 'relative',
                    cursor: 'pointer',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                        boxShadow: 6,
                        transform: 'translateY(-4px)',
                        borderColor: 'primary.light',
                    }
                }}
                onClick={() => handleCardToggle(lab.lab_id || lab.lab_name)}
            >
                <CardContent sx={{ p: 3 }}>
                    {/* ヘッダー部分 */}
                    <Box sx={{ mb: 3 }}>
                        <Typography
                            variant="h5"
                            gutterBottom
                            sx={{
                                fontWeight: 700,
                                color: 'primary.main'
                            }}
                        >
                            {lab.lab_name}
                        </Typography>

                        <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 1 }}>
                            <Typography variant="body1" color="text.secondary" sx={{ fontWeight: 500 }}>
                                {professorName}
                            </Typography>
                            <Divider orientation="vertical" flexItem />
                            <Typography variant="body1" color="text.secondary">
                                {researchArea}
                            </Typography>
                        </Stack>

                        {lab.category && (
                            <Chip
                                label={lab.category}
                                size="medium"
                                variant="outlined"
                                color="primary"
                                sx={{ mt: 1, fontWeight: 500 }}
                            />
                        )}
                    </Box>

                    {/* スコア表示 - 大きく目立つように */}
                    <Paper
                        elevation={0}
                        sx={{
                            p: 3,
                            mb: 3,
                            background: getScoreGradient(score),
                            borderRadius: 3,
                            color: 'white',
                        }}
                    >
                        <Grid container spacing={3} alignItems="center">
                            <Grid item xs={12} md={6}>
                                <Stack spacing={1}>
                                    <Typography variant="body2" sx={{ opacity: 0.9, fontWeight: 500 }}>
                                        総合適合度
                                    </Typography>
                                    <Stack direction="row" spacing={2} alignItems="baseline">
                                        <Typography variant="h3" fontWeight="bold">
                                            {(score * 100).toFixed(1)}%
                                        </Typography>
                                        {lab.priority_adjusted_score && lab.priority_adjusted_score !== score && (
                                            <Tooltip title="優先度による調整済み">
                                                <Star sx={{ fontSize: 24 }} />
                                            </Tooltip>
                                        )}
                                    </Stack>
                                    <LinearProgress
                                        variant="determinate"
                                        value={score * 100}
                                        sx={{
                                            height: 8,
                                            borderRadius: 4,
                                            bgcolor: 'rgba(255, 255, 255, 0.3)',
                                            '& .MuiLinearProgress-bar': {
                                                bgcolor: 'white',
                                                borderRadius: 4,
                                            }
                                        }}
                                    />
                                </Stack>
                            </Grid>
                            <Grid item xs={12} md={6}>
                                <Stack spacing={2}>
                                    <Box>
                                        <Typography variant="caption" sx={{ opacity: 0.9 }}>
                                            信頼度
                                        </Typography>
                                        <Typography variant="h6" fontWeight="bold">
                                            {((lab.confidence || 0.8) * 100).toFixed(1)}%
                                        </Typography>
                                    </Box>
                                    {lab.recommendation && (
                                        <Chip
                                            label={lab.recommendation}
                                            sx={{
                                                bgcolor: 'rgba(255, 255, 255, 0.2)',
                                                color: 'white',
                                                fontWeight: 600,
                                                backdropFilter: 'blur(10px)',
                                            }}
                                        />
                                    )}
                                </Stack>
                            </Grid>
                        </Grid>
                    </Paper>

                    {/* ★★★ 追加: 短縮版の推薦ポイント（カード表示用） ★★★ */}
                    {getExplanation(lab, 'short') && (
                        <Paper
                            variant="outlined"
                            sx={{
                                p: 2.5,
                                mb: 3,
                                bgcolor: 'primary.lighter',
                                borderLeft: 4,
                                borderColor: 'primary.main',
                                borderRadius: 2,
                            }}
                        >
                            <Stack direction="row" spacing={1.5} alignItems="flex-start">
                                <Star color="primary" sx={{ mt: 0.5 }} />
                                <Box>
                                    <Typography variant="subtitle2" color="primary" gutterBottom fontWeight={600}>
                                        推薦ポイント
                                    </Typography>
                                    <Typography variant="body2" color="text.primary" sx={{ lineHeight: 1.7 }}>
                                        {getExplanation(lab, 'short')}
                                    </Typography>
                                </Box>
                            </Stack>
                        </Paper>
                    )}

                    {/* ゼミの説明 */}
                    {lab.description && (
                        <Paper
                            variant="outlined"
                            sx={{
                                p: 2.5,
                                mb: 3,
                                bgcolor: 'grey.50',
                                borderRadius: 2,
                            }}
                        >
                            <Stack direction="row" spacing={1.5} alignItems="flex-start">
                                <Description color="action" sx={{ mt: 0.5 }} />
                                <Box>
                                    <Typography variant="subtitle2" color="text.secondary" gutterBottom fontWeight={600}>
                                        ゼミについて
                                    </Typography>
                                    <Typography variant="body2" color="text.primary" sx={{ lineHeight: 1.7 }}>
                                        {lab.description}
                                    </Typography>
                                </Box>
                            </Stack>
                        </Paper>
                    )}

                    {/* 展開時の詳細情報 */}
                    <Collapse in={isExpanded} timeout={400}>
                        <Box sx={{ mt: 3 }}>
                            {/* ★★★ 追加: 詳細説明（自然言語版） ★★★ */}
                            {getExplanation(lab, 'detailed') && (
                                <Box sx={{ mb: 3 }}>
                                    <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 700, display: 'flex', alignItems: 'center', color: 'primary.main' }}>
                                        <Star sx={{ mr: 1, fontSize: 20 }} />
                                        AIによる詳細分析
                                    </Typography>
                                    <Paper
                                        variant="outlined"
                                        sx={{
                                            p: 2.5,
                                            borderRadius: 2,
                                            bgcolor: 'aliceblue',
                                            borderColor: 'primary.light',
                                            borderLeft: 4,
                                            borderLeftColor: 'primary.main'
                                        }}
                                    >
                                        {/* ★★★ whiteSpace: 'pre-wrap' で改行を反映 ★★★ */}
                                        <Typography
                                            variant="body2"
                                            sx={{
                                                lineHeight: 1.9,
                                                fontSize: '0.95rem',
                                                whiteSpace: 'pre-wrap'
                                            }}
                                        >
                                            {getExplanation(lab, 'detailed')}
                                        </Typography>
                                    </Paper>
                                </Box>
                            )}

                            {/* 専門分野 */}
                            {lab.specialization && (
                                <Paper
                                    variant="outlined"
                                    sx={{
                                        p: 2.5,
                                        mb: 3,
                                        bgcolor: 'background.default',
                                        borderRadius: 2,
                                    }}
                                >
                                    <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 1.5 }}>
                                        <LocalLibrary color="primary" />
                                        <Typography variant="subtitle2" color="primary" fontWeight={600}>
                                            専門分野
                                        </Typography>
                                    </Stack>
                                    <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.7 }}>
                                        {lab.specialization}
                                    </Typography>
                                </Paper>
                            )}

                            {/* 研究分野タグ */}
                            {lab.research_fields && lab.research_fields.length > 0 && (
                                <Box sx={{ mb: 3 }}>
                                    <Typography variant="subtitle2" gutterBottom color="text.secondary" fontWeight={600}>
                                        研究キーワード
                                    </Typography>
                                    <Stack direction="row" spacing={1} flexWrap="wrap" gap={1} sx={{ mt: 1.5 }}>
                                        {lab.research_fields.map((field: string, idx: number) => (
                                            <Chip
                                                key={idx}
                                                label={field}
                                                size="medium"
                                                variant="outlined"
                                                color="primary"
                                                sx={{
                                                    fontWeight: 500,
                                                    '&:hover': {
                                                        bgcolor: 'primary.lighter',
                                                    }
                                                }}
                                            />
                                        ))}
                                    </Stack>
                                </Box>
                            )}

                            {/* ★★★ 類似度比較テーブル ★★★ */}
                            {studentProfile && (lab.criteria_scores || lab.feature_scores) && (
                                <Paper
                                    elevation={2}
                                    sx={{
                                        p: 3,
                                        mb: 3,
                                        borderRadius: 3,
                                        bgcolor: 'background.paper',
                                    }}
                                >
                                    <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 3 }}>
                                        <CompareArrows color="primary" sx={{ fontSize: 28 }} />
                                        <Typography variant="h6" color="primary" fontWeight={700}>
                                            評価基準別の類似度分析
                                        </Typography>
                                    </Stack>

                                    <TableContainer>
                                        <Table>
                                            <TableHead>
                                                <TableRow sx={{ bgcolor: 'grey.50' }}>
                                                    <TableCell sx={{ fontWeight: 700, fontSize: '0.95rem' }}>評価項目</TableCell>
                                                    <TableCell align="center" sx={{ fontWeight: 700, fontSize: '0.95rem' }}>
                                                        あなたの希望
                                                    </TableCell>
                                                    <TableCell align="center" sx={{ fontWeight: 700, fontSize: '0.95rem' }}>
                                                        このゼミ
                                                    </TableCell>
                                                    <TableCell align="center" sx={{ fontWeight: 700, fontSize: '0.95rem' }}>
                                                        類似度
                                                    </TableCell>
                                                </TableRow>
                                            </TableHead>
                                            <TableBody>
                                                {Object.entries(lab.criteria_scores || lab.feature_scores || {})
                                                    .filter(([criterion]) => criterion !== 'research_field_match')
                                                    .map(([criterion, similarityScore]) => {
                                                        const studentValue = studentProfile[criterion];
                                                        const labValue = lab.features?.[criterion];

                                                        const studentDisplay = studentValue !== undefined ? scaleToDisplay(studentValue) : '-';
                                                        const labDisplay = labValue !== undefined ? scaleToDisplay(labValue) : '-';
                                                        const similarity = (similarityScore as number) * 100;

                                                        return (
                                                            <TableRow
                                                                key={criterion}
                                                                sx={{
                                                                    '&:hover': { bgcolor: 'action.hover' },
                                                                    transition: 'background-color 0.2s',
                                                                }}
                                                            >
                                                                <TableCell>
                                                                    <Typography variant="body2" fontWeight={500}>
                                                                        {criteriaNameMap[criterion] || criterion}
                                                                    </Typography>
                                                                </TableCell>
                                                                <TableCell align="center">
                                                                    <Chip
                                                                        label={`${studentDisplay}/10`}
                                                                        size="medium"
                                                                        sx={{
                                                                            bgcolor: 'primary.main',
                                                                            color: 'white',
                                                                            fontWeight: 700,
                                                                            fontSize: '0.9rem',
                                                                            minWidth: 70,
                                                                        }}
                                                                    />
                                                                </TableCell>
                                                                <TableCell align="center">
                                                                    <Chip
                                                                        label={`${labDisplay}/10`}
                                                                        size="medium"
                                                                        sx={{
                                                                            bgcolor: 'secondary.main',
                                                                            color: 'white',
                                                                            fontWeight: 700,
                                                                            fontSize: '0.9rem',
                                                                            minWidth: 70,
                                                                        }}
                                                                    />
                                                                </TableCell>
                                                                <TableCell align="center">
                                                                    <Box sx={{ width: '100%' }}>
                                                                        <Stack direction="row" spacing={1} alignItems="center" justifyContent="center" sx={{ mb: 0.5 }}>
                                                                            {similarity >= 80 ? (
                                                                                <CheckCircle fontSize="small" sx={{ color: 'success.main' }} />
                                                                            ) : similarity >= 60 ? (
                                                                                <TrendingUp fontSize="small" sx={{ color: 'warning.main' }} />
                                                                            ) : (
                                                                                <Warning fontSize="small" sx={{ color: 'error.main' }} />
                                                                            )}
                                                                            <Typography
                                                                                variant="body2"
                                                                                fontWeight="bold"
                                                                                sx={{
                                                                                    color: similarity >= 80 ? 'success.main' : similarity >= 60 ? 'warning.main' : 'error.main',
                                                                                    fontSize: '1rem',
                                                                                }}
                                                                            >
                                                                                {similarity.toFixed(1)}%
                                                                            </Typography>
                                                                        </Stack>
                                                                        <LinearProgress
                                                                            variant="determinate"
                                                                            value={similarity}
                                                                            color={getScoreColor(similarityScore as number)}
                                                                            sx={{
                                                                                height: 6,
                                                                                borderRadius: 3,
                                                                                bgcolor: 'grey.200',
                                                                            }}
                                                                        />
                                                                    </Box>
                                                                </TableCell>
                                                            </TableRow>
                                                        );
                                                    })}
                                            </TableBody>
                                        </Table>
                                    </TableContainer>

                                    {/* 統計情報 */}
                                    <Box sx={{ mt: 3, p: 2, bgcolor: 'grey.50', borderRadius: 2 }}>
                                        <Grid container spacing={2}>
                                            <Grid item xs={4}>
                                                <Typography variant="caption" color="text.secondary">
                                                    平均類似度
                                                </Typography>
                                                <Typography variant="h6" color="primary" fontWeight="bold">
                                                    {(() => {
                                                        const scores = Object.values(lab.criteria_scores || lab.feature_scores || {})
                                                            .filter((_, idx) => Object.keys(lab.criteria_scores || lab.feature_scores || {})[idx] !== 'research_field_match');
                                                        const avg = scores.reduce((sum, s) => sum + (s as number), 0) / scores.length;
                                                        return (avg * 100).toFixed(1);
                                                    })()}%
                                                </Typography>
                                            </Grid>
                                            <Grid item xs={4}>
                                                <Typography variant="caption" color="text.secondary">
                                                    高適合項目
                                                </Typography>
                                                <Typography variant="h6" color="success.main" fontWeight="bold">
                                                    {Object.values(lab.criteria_scores || lab.feature_scores || {})
                                                        .filter((s, idx) => {
                                                            const key = Object.keys(lab.criteria_scores || lab.feature_scores || {})[idx];
                                                            return key !== 'research_field_match' && (s as number) >= 0.8;
                                                        }).length}件
                                                </Typography>
                                            </Grid>
                                            <Grid item xs={4}>
                                                <Typography variant="caption" color="text.secondary">
                                                    評価項目数
                                                </Typography>
                                                <Typography variant="h6" fontWeight="bold">
                                                    11項目
                                                </Typography>
                                            </Grid>
                                        </Grid>
                                    </Box>
                                </Paper>
                            )}

                            {/* 優先度分析 */}
                            {lab.priority_analysis && (
                                <Paper variant="outlined" sx={{ p: 2.5, mb: 3, borderRadius: 2 }}>
                                    <Typography variant="subtitle2" gutterBottom fontWeight={600}>
                                        優先度適合分析
                                    </Typography>
                                    <Grid container spacing={2} sx={{ mt: 1 }}>
                                        <Grid item xs={4}>
                                            <Typography variant="caption" color="error.main" fontWeight={600}>
                                                高優先度
                                            </Typography>
                                            <Typography variant="h6" color="error.main" fontWeight="bold">
                                                {(lab.priority_analysis.high_priority_match * 100).toFixed(0)}%
                                            </Typography>
                                            <LinearProgress
                                                variant="determinate"
                                                value={lab.priority_analysis.high_priority_match * 100}
                                                color="error"
                                                sx={{ height: 6, borderRadius: 3, mt: 1 }}
                                            />
                                        </Grid>
                                        <Grid item xs={4}>
                                            <Typography variant="caption" color="warning.main" fontWeight={600}>
                                                中優先度
                                            </Typography>
                                            <Typography variant="h6" color="warning.main" fontWeight="bold">
                                                {(lab.priority_analysis.medium_priority_match * 100).toFixed(0)}%
                                            </Typography>
                                            <LinearProgress
                                                variant="determinate"
                                                value={lab.priority_analysis.medium_priority_match * 100}
                                                color="warning"
                                                sx={{ height: 6, borderRadius: 3, mt: 1 }}
                                            />
                                        </Grid>
                                        <Grid item xs={4}>
                                            <Typography variant="caption" color="info.main" fontWeight={600}>
                                                低優先度
                                            </Typography>
                                            <Typography variant="h6" color="info.main" fontWeight="bold">
                                                {(lab.priority_analysis.low_priority_match * 100).toFixed(0)}%
                                            </Typography>
                                            <LinearProgress
                                                variant="determinate"
                                                value={lab.priority_analysis.low_priority_match * 100}
                                                color="info"
                                                sx={{ height: 6, borderRadius: 3, mt: 1 }}
                                            />
                                        </Grid>
                                    </Grid>
                                </Paper>
                            )}

                            {/* 従来の説明（詳細版がない場合のフォールバック） */}
                            {!getExplanation(lab, 'detailed') && lab.explanation && (
                                <Box sx={{ mb: 3 }}>
                                    <Typography variant="subtitle2" gutterBottom sx={{ fontWeight: 700, display: 'flex', alignItems: 'center', color: 'primary.main' }}>
                                        <Star sx={{ mr: 1, fontSize: 20 }} />
                                        AIによる推薦理由
                                    </Typography>
                                    <Paper
                                        variant="outlined"
                                        sx={{
                                            p: 2,
                                            borderRadius: 2,
                                            bgcolor: 'aliceblue',
                                            borderColor: 'primary.light',
                                            borderLeft: 4,
                                            borderLeftColor: 'primary.main'
                                        }}
                                    >
                                        <Typography variant="body2" sx={{ lineHeight: 1.8, fontSize: '0.95rem' }}>
                                            {lab.explanation}
                                        </Typography>
                                    </Paper>
                                </Box>
                            )}

                            {/* アクションボタン */}
                            <Box sx={{ display: 'flex', justifyContent: 'center', mt: 3 }}>
                                <Button
                                    variant="contained"
                                    size="large"
                                    startIcon={<Info />}
                                    onClick={(e) => {
                                        e.stopPropagation();
                                        handleLabDetail(lab);
                                    }}
                                    sx={{
                                        borderRadius: 3,
                                        px: 4,
                                        py: 1.5,
                                        fontWeight: 600,
                                        boxShadow: 3,
                                        '&:hover': {
                                            boxShadow: 6,
                                        }
                                    }}
                                >
                                    詳細分析を見る
                                </Button>
                            </Box>
                        </Box>
                    </Collapse>

                    {/* 展開ボタン */}
                    <Box sx={{ textAlign: 'center', mt: 2 }}>
                        <IconButton
                            sx={{
                                transition: 'transform 0.3s',
                                transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
                            }}
                        >
                            <ExpandMore />
                        </IconButton>
                        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                            {isExpanded ? '閉じる' : 'もっと見る'}
                        </Typography>
                    </Box>
                </CardContent>
            </Card>
        );
    };

    const renderDetailDialog = () => (
        <Dialog
            open={detailDialogOpen}
            onClose={() => setDetailDialogOpen(false)}
            maxWidth="md"
            fullWidth
            PaperProps={{
                sx: { borderRadius: 3 }
            }}
        >
            <DialogTitle>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="h5" fontWeight={700}>
                        {selectedLab?.lab_name} - 詳細分析
                    </Typography>
                    <IconButton onClick={() => setDetailDialogOpen(false)}>
                        <Close />
                    </IconButton>
                </Box>
            </DialogTitle>
            <DialogContent dividers>
                {selectedLab && (
                    <Box>
                        {/* スコア詳細 */}
                        <Typography variant="h6" gutterBottom fontWeight={700}>
                            スコア詳細分析
                        </Typography>
                        <Grid container spacing={2} sx={{ mb: 4 }}>
                            <Grid item xs={4}>
                                <Card
                                    variant="outlined"
                                    sx={{
                                        p: 2.5,
                                        background: getScoreGradient(selectedLab.overall_compatibility || selectedLab.final_score || 0),
                                        color: 'white',
                                        borderRadius: 2,
                                    }}
                                >
                                    <Typography variant="subtitle2" gutterBottom sx={{ opacity: 0.9 }}>
                                        総合適合度
                                    </Typography>
                                    <Typography variant="h3" fontWeight="bold">
                                        {((selectedLab.overall_compatibility || selectedLab.final_score || 0) * 100).toFixed(1)}%
                                    </Typography>
                                    <LinearProgress
                                        variant="determinate"
                                        value={(selectedLab.overall_compatibility || selectedLab.final_score || 0) * 100}
                                        sx={{
                                            mt: 2,
                                            height: 8,
                                            borderRadius: 4,
                                            bgcolor: 'rgba(255,255,255,0.3)',
                                            '& .MuiLinearProgress-bar': {
                                                bgcolor: 'white',
                                            }
                                        }}
                                    />
                                </Card>
                            </Grid>
                            <Grid item xs={4}>
                                <Card variant="outlined" sx={{ p: 2.5, borderRadius: 2 }}>
                                    <Typography variant="subtitle2" gutterBottom>基本スコア</Typography>
                                    <Typography variant="h3" color="secondary" fontWeight="bold">
                                        {((selectedLab.basic_score || 0) * 100).toFixed(1)}%
                                    </Typography>
                                    <LinearProgress
                                        variant="determinate"
                                        value={(selectedLab.basic_score || 0) * 100}
                                        color="secondary"
                                        sx={{ mt: 2, height: 8, borderRadius: 4 }}
                                    />
                                </Card>
                            </Grid>
                            <Grid item xs={4}>
                                <Card variant="outlined" sx={{ p: 2.5, borderRadius: 2 }}>
                                    <Typography variant="subtitle2" gutterBottom>分野スコア</Typography>
                                    <Typography variant="h3" color="info.main" fontWeight="bold">
                                        {((selectedLab.field_score || 0) * 100).toFixed(1)}%
                                    </Typography>
                                    <LinearProgress
                                        variant="determinate"
                                        value={(selectedLab.field_score || 0) * 100}
                                        color="info"
                                        sx={{ mt: 2, height: 8, borderRadius: 4 }}
                                    />
                                </Card>
                            </Grid>
                        </Grid>

                        <Divider sx={{ my: 3 }} />

                        {/* ★★★ 追加: ダイアログでの詳細説明表示 ★★★ */}
                        {getExplanation(selectedLab, 'detailed') && (
                            <Box sx={{ mb: 4 }}>
                                <Typography variant="h6" gutterBottom fontWeight={700} sx={{ display: 'flex', alignItems: 'center' }}>
                                    <Star sx={{ mr: 1, color: 'primary.main' }} />
                                    AIによる詳細分析
                                </Typography>
                                <Paper
                                    variant="outlined"
                                    sx={{
                                        p: 3,
                                        borderRadius: 2,
                                        bgcolor: 'aliceblue',
                                        borderColor: 'primary.light',
                                        borderLeft: 4,
                                        borderLeftColor: 'primary.main'
                                    }}
                                >
                                    <Typography
                                        variant="body1"
                                        sx={{
                                            lineHeight: 2,
                                            whiteSpace: 'pre-wrap'
                                        }}
                                    >
                                        {getExplanation(selectedLab, 'detailed')}
                                    </Typography>
                                </Paper>
                            </Box>
                        )}

                        {/* 詳細な類似度比較 */}
                        {studentProfile && (selectedLab.criteria_scores || selectedLab.feature_scores) && (
                            <>
                                <Typography variant="h6" gutterBottom fontWeight={700}>
                                    項目別類似度詳細
                                </Typography>
                                <TableContainer component={Paper} variant="outlined" sx={{ borderRadius: 2 }}>
                                    <Table>
                                        <TableHead>
                                            <TableRow sx={{ bgcolor: 'grey.100' }}>
                                                <TableCell sx={{ fontWeight: 700 }}>評価項目</TableCell>
                                                <TableCell align="center" sx={{ fontWeight: 700 }}>あなたの希望</TableCell>
                                                <TableCell align="center" sx={{ fontWeight: 700 }}>このゼミ</TableCell>
                                                <TableCell align="center" sx={{ fontWeight: 700 }}>類似度</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>
                                            {Object.entries(selectedLab.criteria_scores || selectedLab.feature_scores || {})
                                                .filter(([criterion]) => criterion !== 'research_field_match')
                                                .map(([criterion, similarityScore]) => {
                                                    const studentValue = studentProfile[criterion];
                                                    const labValue = selectedLab.features?.[criterion];

                                                    const studentDisplay = studentValue !== undefined ? scaleToDisplay(studentValue) : '-';
                                                    const labDisplay = labValue !== undefined ? scaleToDisplay(labValue) : '-';
                                                    const similarity = (similarityScore as number) * 100;

                                                    return (
                                                        <TableRow key={criterion} sx={{ '&:hover': { bgcolor: 'action.hover' } }}>
                                                            <TableCell>
                                                                <Typography fontWeight={500}>
                                                                    {criteriaNameMap[criterion] || criterion}
                                                                </Typography>
                                                            </TableCell>
                                                            <TableCell align="center">
                                                                <Chip
                                                                    label={`${studentDisplay}/10`}
                                                                    sx={{
                                                                        bgcolor: 'primary.main',
                                                                        color: 'white',
                                                                        fontWeight: 700,
                                                                        minWidth: 70,
                                                                    }}
                                                                />
                                                            </TableCell>
                                                            <TableCell align="center">
                                                                <Chip
                                                                    label={`${labDisplay}/10`}
                                                                    sx={{
                                                                        bgcolor: 'secondary.main',
                                                                        color: 'white',
                                                                        fontWeight: 700,
                                                                        minWidth: 70,
                                                                    }}
                                                                />
                                                            </TableCell>
                                                            <TableCell align="center">
                                                                <Box>
                                                                    <Typography
                                                                        variant="body1"
                                                                        fontWeight="bold"
                                                                        sx={{
                                                                            color: similarity >= 80 ? 'success.main' : similarity >= 60 ? 'warning.main' : 'error.main',
                                                                            mb: 1,
                                                                        }}
                                                                    >
                                                                        {similarity.toFixed(1)}%
                                                                    </Typography>
                                                                    <LinearProgress
                                                                        variant="determinate"
                                                                        value={similarity}
                                                                        color={getScoreColor(similarityScore as number)}
                                                                        sx={{ height: 8, borderRadius: 4 }}
                                                                    />
                                                                </Box>
                                                            </TableCell>
                                                        </TableRow>
                                                    );
                                                })}
                                        </TableBody>
                                    </Table>
                                </TableContainer>
                            </>
                        )}
                    </Box>
                )}
            </DialogContent>
            <DialogActions sx={{ p: 2.5 }}>
                <Button
                    onClick={() => setDetailDialogOpen(false)}
                    variant="contained"
                    size="large"
                    sx={{ borderRadius: 2, px: 3 }}
                >
                    閉じる
                </Button>
            </DialogActions>
        </Dialog>
    );

    return (
        <Box>
            <Box sx={{
                mb: 4,
                p: 3,
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                borderRadius: 3,
                color: 'white',
            }}>
                <Typography variant="h4" gutterBottom fontWeight={800}>
                    研究室適合度評価結果
                </Typography>
                <Typography variant="body1" sx={{ opacity: 0.9 }}>
                    あなたに最適な研究室をランキング形式で表示しています
                </Typography>
            </Box>

            {!results || results.length === 0 ? (
                <Alert
                    severity="warning"
                    sx={{
                        borderRadius: 2,
                        fontSize: '1rem',
                    }}
                >
                    評価結果がありません。評価を実行してください。
                </Alert>
            ) : (
                <Box>{results.map((lab, index) => renderLabCard(lab, index))}</Box>
            )}

            {renderDetailDialog()}
        </Box>
    );
};

export default ResultsList;