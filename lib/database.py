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
        await conn.execute("""
            ALTER TABLE stroke_logs ADD COLUMN IF NOT EXISTS event_type TEXT NOT NULL DEFAULT 'draw'
        """)
        await conn.execute("""
            ALTER TABLE stroke_logs ADD COLUMN IF NOT EXISTS deleted_count INT NOT NULL DEFAULT 0
        """)
        await conn.execute("""
            ALTER TABLE stroke_logs ADD COLUMN IF NOT EXISTS message TEXT NOT NULL DEFAULT ''
        """)
        await conn.execute("""
            ALTER TABLE stroke_logs ADD COLUMN IF NOT EXISTS user_id TEXT NOT NULL DEFAULT ''
        """)
        await conn.execute("""
            ALTER TABLE stroke_logs ADD COLUMN IF NOT EXISTS cluster_labels JSONB NOT NULL DEFAULT '[]'::jsonb
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS clusters (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                page INT NOT NULL,
                cluster_label INT NOT NULL,
                stroke_count INT NOT NULL,
                centroid_x FLOAT NOT NULL,
                centroid_y FLOAT NOT NULL,
                bbox_x1 FLOAT NOT NULL,
                bbox_y1 FLOAT NOT NULL,
                bbox_x2 FLOAT NOT NULL,
                bbox_y2 FLOAT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(session_id, page, cluster_label)
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS cluster_classes (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                page INT NOT NULL,
                stroke_log_id INT NOT NULL REFERENCES stroke_logs(id) ON DELETE CASCADE,
                stroke_index INT NOT NULL,
                cluster_label INT NOT NULL,
                centroid_x FLOAT NOT NULL,
                centroid_y FLOAT NOT NULL,
                UNIQUE(session_id, page, stroke_log_id, stroke_index)
            )
        """)
        await conn.execute("""
            ALTER TABLE clusters ADD COLUMN IF NOT EXISTS transcription TEXT NOT NULL DEFAULT ''
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_clusters_session_page
            ON clusters(session_id, page)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_cluster_classes_session_page
            ON cluster_classes(session_id, page)
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
