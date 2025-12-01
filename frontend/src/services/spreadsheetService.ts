// frontend/src/services/spreadsheetService.ts
// スプレッドシート記録サービス

/**
 * Google Apps Script WebアプリのURL
 * デプロイ後にこのURLを更新してください
 */
const SPREADSHEET_API_URL = 'https://script.google.com/macros/s/AKfycbzt9-WrvqvEI4lcrWNSrzNAremvEmjjZ5REEDkvOyuh-cNU7hczowTNd203b_mek94/exec';

/**
 * スプレッドシートへの記録データ型定義
 */
export interface SpreadsheetRecordData {
    results: Array<{
        lab_name: string;
        overall_compatibility: number;
    }>;
    inputValues: {
        basicCriteria: Record<string, number>;
        fieldInterests: Record<string, number>;
    };
}

/**
 * スプレッドシートへの記録レスポンス型定義
 */
export interface SpreadsheetRecordResponse {
    success: boolean;
    message: string;
    rowNumber?: number;
}

/**
 * 評価結果をスプレッドシートに記録
 * 
 * @param data - 記録するデータ
 * @returns 記録結果
 */
export async function recordToSpreadsheet(
    data: SpreadsheetRecordData
): Promise<SpreadsheetRecordResponse> {
    try {
        // Google Apps Script WebアプリのURLが設定されているかチェック
        if (SPREADSHEET_API_URL === 'https://script.google.com/macros/s/AKfycbzt9-WrvqvEI4lcrWNSrzNAremvEmjjZ5REEDkvOyuh-cNU7hczowTNd203b_mek94/exec') {
            console.warn('スプレッドシートAPIのURLが設定されていません');
            return {
                success: false,
                message: 'スプレッドシートAPIのURLが設定されていません'
            };
        }

        // POSTリクエストを送信
        const response = await fetch(SPREADSHEET_API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
            mode: 'no-cors' // Google Apps Scriptの制限により必要
        });

        // no-corsモードではresponseの内容を読み取れないため、
        // 成功と仮定する（エラーの場合はcatchで捕捉される）
        return {
            success: true,
            message: 'スプレッドシートに記録しました'
        };

    } catch (error) {
        console.error('スプレッドシート記録エラー:', error);
        return {
            success: false,
            message: `スプレッドシート記録エラー: ${error instanceof Error ? error.message : '不明なエラー'}`
        };
    }
}

/**
 * スプレッドシート記録が有効かチェック
 */
export function isSpreadsheetRecordingEnabled(): boolean {
    return SPREADSHEET_API_URL !== 'https://script.google.com/macros/s/AKfycbzt9-WrvqvEI4lcrWNSrzNAremvEmjjZ5REEDkvOyuh-cNU7hczowTNd203b_mek94/exec';
}

/**
 * スプレッドシートAPIのURL設定状態を取得
 */
export function getSpreadsheetApiStatus(): {
    enabled: boolean;
    url: string;
} {
    return {
        enabled: isSpreadsheetRecordingEnabled(),
        url: SPREADSHEET_API_URL
    };
}