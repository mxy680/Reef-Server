"""User profile endpoints."""

from fastapi import APIRouter, HTTPException, Header
from lib.database import get_pool
from lib.models import UserProfileRequest, UserProfileResponse

router = APIRouter(prefix="/users", tags=["users"])


def _get_user_id(authorization: str) -> str:
    """Extract Apple user ID from 'Bearer <apple_user_id>' header."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    user_id = authorization[7:].strip()
    if not user_id:
        raise HTTPException(status_code=401, detail="Missing user identifier")
    return user_id


@router.put("/profile", response_model=UserProfileResponse)
async def upsert_profile(
    body: UserProfileRequest,
    authorization: str = Header(...),
):
    """Upsert user profile. Null fields don't overwrite existing data."""
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = _get_user_id(authorization)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO user_profiles (apple_user_id, display_name, email)
            VALUES ($1, $2, $3)
            ON CONFLICT (apple_user_id) DO UPDATE SET
                display_name = COALESCE($2, user_profiles.display_name),
                email = COALESCE($3, user_profiles.email),
                updated_at = NOW()
            RETURNING apple_user_id, display_name, email
            """,
            user_id,
            body.display_name,
            body.email,
        )

    return UserProfileResponse(
        apple_user_id=row["apple_user_id"],
        display_name=row["display_name"],
        email=row["email"],
    )


@router.get("/profile", response_model=UserProfileResponse)
async def get_profile(authorization: str = Header(...)):
    """Get current user's profile."""
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = _get_user_id(authorization)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT apple_user_id, display_name, email FROM user_profiles WHERE apple_user_id = $1",
            user_id,
        )

    if row is None:
        raise HTTPException(status_code=404, detail="Profile not found")

    return UserProfileResponse(
        apple_user_id=row["apple_user_id"],
        display_name=row["display_name"],
        email=row["email"],
    )


@router.delete("/profile")
async def delete_profile(authorization: str = Header(...)):
    """Delete current user's profile."""
    pool = get_pool()
    if pool is None:
        raise HTTPException(status_code=503, detail="Database not available")

    user_id = _get_user_id(authorization)

    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM user_profiles WHERE apple_user_id = $1",
            user_id,
        )

    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Profile not found")

    return {"status": "deleted"}
