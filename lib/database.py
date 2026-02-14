"""PostgreSQL connection pool for user profile storage."""

import os
import asyncpg

_pool: asyncpg.Pool | None = None


async def init_db():
    """Create asyncpg connection pool and ensure user_profiles table exists."""
    global _pool

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("[DB] DATABASE_URL not set — skipping database init")
        return

    _pool = await asyncpg.create_pool(database_url, min_size=1, max_size=5)
    async with _pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                apple_user_id TEXT PRIMARY KEY,
                display_name TEXT,
                email TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS stroke_logs (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                page INT NOT NULL,
                received_at TIMESTAMPTZ DEFAULT NOW(),
                strokes JSONB NOT NULL
            )
        """)
    print("[DB] Connected and tables ready")


async def close_db():
    """Close the connection pool on shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        print("[DB] Connection pool closed")


def get_pool() -> asyncpg.Pool | None:
    """Return the pool singleton (None if DB not configured)."""
    return _pool
