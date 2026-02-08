"""Pydantic models for user profile endpoints."""

from pydantic import BaseModel


class UserProfileRequest(BaseModel):
    display_name: str | None = None
    email: str | None = None


class UserProfileResponse(BaseModel):
    apple_user_id: str
    display_name: str | None = None
    email: str | None = None
