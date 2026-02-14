"""Incremental bounding-box stroke clustering.

Assigns strokes to clusters by checking overlap with existing cluster
bounding boxes (expanded by 10%).  When a stroke overlaps multiple
clusters, those clusters are merged.
"""

import asyncio
import json
from dataclasses import dataclass

import numpy as np

from lib.database import get_pool
from lib.models.clustering import ClusterInfo, ClusterResponse

BBOX_PAD = 0.0  # strokes must touch/overlap cluster bbox to join


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


def cluster_by_bbox_overlap(
    entries: list[StrokeEntry],
    pad: float = BBOX_PAD,
) -> tuple[np.ndarray, np.ndarray, list[ClusterInfo]]:
    """Cluster strokes incrementally by bounding-box overlap.

    For each stroke in order:
    1. Check if its bbox overlaps any existing cluster bbox (expanded by pad pixels).
    2. No overlap → new cluster.
    3. One overlap → add to that cluster, grow its bbox.
    4. Multiple overlaps → merge those clusters, add stroke.
    """
    if not entries:
        return np.array([]).reshape(0, 2), np.array([], dtype=int), []

    cluster_bboxes: list[list[float]] = []   # [min_x, min_y, max_x, max_y]
    cluster_members: list[list[int]] = []    # entry indices

    for i, entry in enumerate(entries):
        overlapping: list[int] = []

        for c_idx, bbox in enumerate(cluster_bboxes):
            if (entry.min_x <= bbox[2] + pad and
                entry.max_x >= bbox[0] - pad and
                entry.min_y <= bbox[3] + pad and
                entry.max_y >= bbox[1] - pad):
                overlapping.append(c_idx)

        if not overlapping:
            cluster_bboxes.append([entry.min_x, entry.min_y, entry.max_x, entry.max_y])
            cluster_members.append([i])
        elif len(overlapping) == 1:
            c = overlapping[0]
            cluster_members[c].append(i)
            cluster_bboxes[c] = [
                min(cluster_bboxes[c][0], entry.min_x),
                min(cluster_bboxes[c][1], entry.min_y),
                max(cluster_bboxes[c][2], entry.max_x),
                max(cluster_bboxes[c][3], entry.max_y),
            ]
        else:
            # Merge all overlapping clusters into the first one
            target = overlapping[0]
            cluster_members[target].append(i)
            for c_idx in sorted(overlapping[1:], reverse=True):
                cluster_members[target].extend(cluster_members[c_idx])
                cluster_bboxes[target] = [
                    min(cluster_bboxes[target][0], cluster_bboxes[c_idx][0]),
                    min(cluster_bboxes[target][1], cluster_bboxes[c_idx][1]),
                    max(cluster_bboxes[target][2], cluster_bboxes[c_idx][2]),
                    max(cluster_bboxes[target][3], cluster_bboxes[c_idx][3]),
                ]
                del cluster_bboxes[c_idx]
                del cluster_members[c_idx]
            cluster_bboxes[target] = [
                min(cluster_bboxes[target][0], entry.min_x),
                min(cluster_bboxes[target][1], entry.min_y),
                max(cluster_bboxes[target][2], entry.max_x),
                max(cluster_bboxes[target][3], entry.max_y),
            ]

    # Build output
    labels = np.zeros(len(entries), dtype=int)
    centroids = np.array([[e.centroid_x, e.centroid_y] for e in entries])
    cluster_infos: list[ClusterInfo] = []

    for label, members in enumerate(cluster_members):
        for idx in members:
            labels[idx] = label

        member_entries = [entries[idx] for idx in members]
        member_centroids = centroids[np.array(members)]
        cluster_infos.append(ClusterInfo(
            cluster_label=label,
            stroke_count=len(members),
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

    entries = extract_stroke_entries([dict(r) for r in rows])
    if not entries:
        return

    for e in entries:
        print(f"[cluster] stroke log={e.log_id} idx={e.index} bbox=({e.min_x:.0f},{e.min_y:.0f})-({e.max_x:.0f},{e.max_y:.0f})")

    _, labels, cluster_infos = cluster_by_bbox_overlap(entries)
    print(f"[cluster] bbox-overlap → {len(cluster_infos)} clusters, labels={[int(l) for l in labels]}")

    # Group labels by log_id
    labels_by_log: dict[int, list[int]] = {}
    for i, entry in enumerate(entries):
        labels_by_log.setdefault(entry.log_id, []).append(int(labels[i]))

    async with pool.acquire() as conn:
        await conn.executemany(
            "UPDATE stroke_logs SET cluster_labels = $1::jsonb WHERE id = $2",
            [(json.dumps(lbls), log_id) for log_id, lbls in labels_by_log.items()],
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

    centroids, labels, cluster_infos = cluster_by_bbox_overlap(entries)

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
