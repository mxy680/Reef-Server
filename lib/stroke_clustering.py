"""Y-centroid gap stroke clustering.

Strokes are sorted by centroid_y, then split into clusters wherever the
gap between consecutive centroid_y values exceeds a threshold.  This
correctly handles tall symbols (integrals, brackets) whose bounding boxes
span multiple lines but whose centroids remain on the correct line.
"""

import asyncio
import json
from dataclasses import dataclass

import numpy as np

from lib.database import get_pool
from lib.groq_vision import transcribe_strokes_image
from lib.models.clustering import ClusterInfo, ClusterResponse
from lib.stroke_renderer import render_strokes


@dataclass
class StrokeEntry:
    """A single stroke with its bounding box."""
    log_id: int
    index: int
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @property
    def centroid_x(self) -> float:
        return (self.min_x + self.max_x) / 2

    @property
    def centroid_y(self) -> float:
        return (self.min_y + self.max_y) / 2


def extract_stroke_entries(rows: list[dict]) -> list[StrokeEntry]:
    """Parse stroke_logs rows into per-stroke entries with bounding boxes."""
    entries: list[StrokeEntry] = []
    for row in rows:
        log_id = row["id"]
        strokes_json = row["strokes"]
        strokes = strokes_json if isinstance(strokes_json, list) else json.loads(strokes_json)

        for idx, stroke in enumerate(strokes):
            points = stroke.get("points", [])
            if not points:
                continue

            xs = [p["x"] for p in points]
            ys = [p["y"] for p in points]
            entries.append(StrokeEntry(
                log_id=log_id,
                index=idx,
                min_x=min(xs), min_y=min(ys),
                max_x=max(xs), max_y=max(ys),
            ))
    return entries


def cluster_by_centroid_gap(
    entries: list[StrokeEntry],
) -> tuple[np.ndarray, np.ndarray, list[ClusterInfo]]:
    """Cluster strokes by gaps between sorted centroid_y values.

    1. Sort strokes by centroid_y.
    2. Compute consecutive gaps.
    3. Find threshold via natural-breaks: sort gaps, find the biggest
       ratio between consecutive non-zero sorted gaps where the upper
       value ≥ 5px. If ratio ≥ 1.5, use the midpoint as threshold.
       Otherwise fall back to median_gap × 4 (floor 20).
    4. Any gap > threshold starts a new cluster.
    5. Labels assigned top-to-bottom (cluster 0 = topmost line).
    """
    if not entries:
        return np.array([]).reshape(0, 2), np.array([], dtype=int), []

    centroids = np.array([[e.centroid_x, e.centroid_y] for e in entries])

    # Sort by centroid_y
    order = np.argsort(centroids[:, 1])
    sorted_cy = centroids[order, 1]

    # Compute gaps and threshold
    gaps = np.diff(sorted_cy)
    if len(gaps) < 8:
        # Too few strokes for natural-breaks — use fixed threshold
        threshold = 20.0
    else:
        # Natural-breaks: find the biggest ratio between consecutive
        # non-zero sorted gaps, but only where the upper gap is >= 5px
        # (sub-5px gaps can't be real line breaks). This avoids false
        # splits from tiny within-line ratio noise (e.g. 0.5→1.0 = 2x).
        sorted_gaps = np.sort(gaps)
        nonzero = sorted_gaps[sorted_gaps > 0]

        best_ratio = 0.0
        best_ratio_idx = -1
        if len(nonzero) >= 2:
            ratios = nonzero[1:] / nonzero[:-1]
            for i in range(len(ratios)):
                if nonzero[i + 1] >= 5 and ratios[i] > best_ratio:
                    best_ratio = float(ratios[i])
                    best_ratio_idx = i

        if best_ratio_idx >= 0 and best_ratio >= 1.5:
            below = float(nonzero[best_ratio_idx])
            above = float(nonzero[best_ratio_idx + 1])
            threshold = (below + above) / 2
        else:
            threshold = max(float(np.median(gaps)) * 4, 20.0)

    # Assign cluster labels top-to-bottom
    labels = np.zeros(len(entries), dtype=int)
    current_label = 0
    for rank in range(len(order)):
        if rank > 0 and gaps[rank - 1] > threshold:
            current_label += 1
        labels[order[rank]] = current_label

    # Build ClusterInfo list
    num_clusters = current_label + 1
    cluster_infos: list[ClusterInfo] = []

    for label in range(num_clusters):
        mask = labels == label
        member_indices = np.where(mask)[0]
        member_entries = [entries[i] for i in member_indices]
        member_centroids = centroids[member_indices]
        cluster_infos.append(ClusterInfo(
            cluster_label=label,
            stroke_count=len(member_entries),
            centroid=[
                float(member_centroids[:, 0].mean()),
                float(member_centroids[:, 1].mean()),
            ],
            bounding_box=[
                float(min(e.min_x for e in member_entries)),
                float(min(e.min_y for e in member_entries)),
                float(max(e.max_x for e in member_entries)),
                float(max(e.max_y for e in member_entries)),
            ],
        ))

    return centroids, labels, cluster_infos


async def update_cluster_labels(session_id: str, page: int):
    """Re-cluster all draw strokes for a session+page and update cluster_labels column."""
    pool = get_pool()
    if not pool:
        return

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, strokes
            FROM stroke_logs
            WHERE session_id = $1 AND page = $2 AND event_type = 'draw'
            ORDER BY received_at
            """,
            session_id, page,
        )

    if not rows:
        return

    # Build stroke lookup: (log_id, stroke_index) → stroke dict
    stroke_lookup: dict[tuple[int, int], dict] = {}
    for row in rows:
        log_id = row["id"]
        strokes_json = row["strokes"]
        strokes = strokes_json if isinstance(strokes_json, list) else json.loads(strokes_json)
        for idx, stroke in enumerate(strokes):
            stroke_lookup[(log_id, idx)] = stroke

    entries = extract_stroke_entries([dict(r) for r in rows])
    if not entries:
        return

    for e in entries:
        print(f"[cluster] stroke log={e.log_id} idx={e.index} bbox=({e.min_x:.0f},{e.min_y:.0f})-({e.max_x:.0f},{e.max_y:.0f})")

    _, labels, cluster_infos = cluster_by_centroid_gap(entries)
    print(f"[cluster] centroid-gap → {len(cluster_infos)} clusters, labels={[int(l) for l in labels]}")

    # Group labels by log_id
    labels_by_log: dict[int, list[int]] = {}
    for i, entry in enumerate(entries):
        labels_by_log.setdefault(entry.log_id, []).append(int(labels[i]))

    async with pool.acquire() as conn:
        await conn.executemany(
            "UPDATE stroke_logs SET cluster_labels = $1::jsonb WHERE id = $2",
            [(json.dumps(lbls), log_id) for log_id, lbls in labels_by_log.items()],
        )

    # Group strokes by cluster label for transcription
    strokes_by_cluster: dict[int, list[dict]] = {}
    for i, entry in enumerate(entries):
        label = int(labels[i])
        stroke = stroke_lookup.get((entry.log_id, entry.index))
        if stroke:
            strokes_by_cluster.setdefault(label, []).append(stroke)

    # Render + transcribe each cluster, upsert into clusters table
    for info in cluster_infos:
        label = info.cluster_label
        cluster_strokes = strokes_by_cluster.get(label, [])
        transcription = ""
        if cluster_strokes:
            try:
                image_bytes = render_strokes(cluster_strokes)
                transcription = await asyncio.to_thread(transcribe_strokes_image, image_bytes)
                print(f"[transcribe] cluster {label}: {transcription}")
            except Exception as exc:
                print(f"[transcribe] cluster {label} failed: {exc}")

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO clusters (session_id, page, cluster_label, stroke_count,
                                      centroid_x, centroid_y, bbox_x1, bbox_y1, bbox_x2, bbox_y2, transcription)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                ON CONFLICT (session_id, page, cluster_label) DO UPDATE SET
                    stroke_count = EXCLUDED.stroke_count,
                    centroid_x = EXCLUDED.centroid_x, centroid_y = EXCLUDED.centroid_y,
                    bbox_x1 = EXCLUDED.bbox_x1, bbox_y1 = EXCLUDED.bbox_y1,
                    bbox_x2 = EXCLUDED.bbox_x2, bbox_y2 = EXCLUDED.bbox_y2,
                    transcription = EXCLUDED.transcription
                """,
                session_id, page,
                label, info.stroke_count,
                info.centroid[0], info.centroid[1],
                info.bounding_box[0], info.bounding_box[1],
                info.bounding_box[2], info.bounding_box[3],
                transcription,
            )


async def cluster_strokes(session_id: str, page: int) -> ClusterResponse:
    pool = get_pool()
    if not pool:
        raise RuntimeError("Database not configured")

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, strokes
            FROM stroke_logs
            WHERE session_id = $1 AND page = $2
            ORDER BY received_at
            """,
            session_id,
            page,
        )

    if not rows:
        return ClusterResponse(
            session_id=session_id, page=page,
            num_strokes=0, num_clusters=0, noise_strokes=0, clusters=[],
        )

    entries = extract_stroke_entries([dict(r) for r in rows])
    if not entries:
        return ClusterResponse(
            session_id=session_id, page=page,
            num_strokes=0, num_clusters=0, noise_strokes=0, clusters=[],
        )

    centroids, labels, cluster_infos = cluster_by_centroid_gap(entries)

    # Store results to database
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM cluster_classes WHERE session_id = $1 AND page = $2",
                session_id, page,
            )
            await conn.execute(
                "DELETE FROM clusters WHERE session_id = $1 AND page = $2",
                session_id, page,
            )

            await conn.executemany(
                """
                INSERT INTO cluster_classes
                    (session_id, page, stroke_log_id, stroke_index, cluster_label, centroid_x, centroid_y)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                [
                    (
                        session_id, page,
                        entry.log_id, entry.index,
                        int(labels[i]),
                        entry.centroid_x, entry.centroid_y,
                    )
                    for i, entry in enumerate(entries)
                ],
            )

            for info in cluster_infos:
                await conn.execute(
                    """
                    INSERT INTO clusters
                        (session_id, page, cluster_label, stroke_count,
                         centroid_x, centroid_y, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    """,
                    session_id, page,
                    info.cluster_label, info.stroke_count,
                    info.centroid[0], info.centroid[1],
                    info.bounding_box[0], info.bounding_box[1],
                    info.bounding_box[2], info.bounding_box[3],
                )

    return ClusterResponse(
        session_id=session_id, page=page,
        num_strokes=len(entries),
        num_clusters=len(cluster_infos),
        noise_strokes=0,
        clusters=cluster_infos,
    )
