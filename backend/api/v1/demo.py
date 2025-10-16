# backend/api/v1/demo.py
"""
デモプロファイルAPIエンドポイント
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Any

from services.demo_profiles import DemoProfileService


# ルーター作成
router = APIRouter(prefix="/api/demo", tags=["Demo Profiles"])


@router.get("/profiles")
async def get_demo_profiles() -> Dict[str, Any]:
    """
    デモプロファイル一覧を取得
    
    Returns:
        {
            "profiles": {
                "プロファイル名": {
                    "description": "説明",
                    "characteristics": ["特徴1", "特徴2", ...]
                },
                ...
            },
            "count": プロファイル数
        }
    """
    try:
        profiles = DemoProfileService.get_all_profiles()
        
        return {
            "profiles": profiles,
            "count": len(profiles),
            "message": "デモプロファイル一覧を取得しました"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"デモプロファイル取得エラー: {str(e)}"
        )


@router.get("/profiles/names")
async def get_demo_profile_names() -> Dict[str, Any]:
    """
    デモプロファイル名の一覧を取得
    
    Returns:
        {
            "names": ["プロファイル名1", "プロファイル名2", ...],
            "count": プロファイル数
        }
    """
    try:
        names = DemoProfileService.get_profile_names()
        
        return {
            "names": names,
            "count": len(names),
            "message": "デモプロファイル名一覧を取得しました"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"プロファイル名取得エラー: {str(e)}"
        )


@router.get("/profiles/{profile_name}")
async def get_demo_profile(profile_name: str) -> Dict[str, Any]:
    """
    指定されたデモプロファイルを取得
    
    Args:
        profile_name: プロファイル名（URLエンコード済み）
        
    Returns:
        {
            "name": "プロファイル名",
            "description": "説明",
            "characteristics": ["特徴1", "特徴2", ...],
            "profile": {
                "research_intensity": 9,
                "advisor_style": 6,
                ...
            }
        }
    """
    try:
        profile_data = DemoProfileService.get_profile_with_metadata(profile_name)
        
        return {
            **profile_data,
            "message": f"デモプロファイル '{profile_name}' を取得しました"
        }
    
    except KeyError as e:
        raise HTTPException(
            status_code=404,
            detail=f"プロファイル '{profile_name}' が見つかりません。利用可能なプロファイル名を /api/demo/profiles/names で確認してください。"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"プロファイル取得エラー: {str(e)}"
        )


@router.get("/profiles/{profile_name}/simple")
async def get_demo_profile_simple(profile_name: str) -> Dict[str, Any]:
    """
    指定されたデモプロファイルを取得（プロファイルデータのみ）
    
    Args:
        profile_name: プロファイル名（URLエンコード済み）
        
    Returns:
        {
            "research_intensity": 9,
            "advisor_style": 6,
            "team_work": 7,
            ...
            "field_interests": [...]
        }
    """
    try:
        profile = DemoProfileService.get_profile(profile_name)
        
        return profile
    
    except KeyError as e:
        raise HTTPException(
            status_code=404,
            detail=f"プロファイル '{profile_name}' が見つかりません"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"プロファイル取得エラー: {str(e)}"
        )


@router.get("/stats")
async def get_demo_stats() -> Dict[str, Any]:
    """
    デモプロファイルの統計情報を取得
    
    Returns:
        {
            "total_profiles": 10,
            "profile_types": {
                "研究志向": 3,
                "クリエイティブ": 2,
                ...
            }
        }
    """
    try:
        profiles = DemoProfileService.get_all_profiles()
        
        # タイプ別カウント
        type_counts = {}
        for name in profiles.keys():
            # プロファイル名から型を抽出
            if "研究" in name:
                profile_type = "研究志向"
            elif "クリエイティブ" in name or "デザイナー" in name or "ゲーム" in name:
                profile_type = "クリエイティブ"
            elif "エンジニア" in name or "実践" in name:
                profile_type = "実践志向"
            elif "チームワーク" in name or "協調" in name:
                profile_type = "協調型"
            elif "教育" in name or "スポーツ" in name:
                profile_type = "専門領域型"
            else:
                profile_type = "その他"
            
            type_counts[profile_type] = type_counts.get(profile_type, 0) + 1
        
        return {
            "total_profiles": len(profiles),
            "profile_types": type_counts,
            "message": "デモプロファイル統計情報を取得しました"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"統計情報取得エラー: {str(e)}"
        )