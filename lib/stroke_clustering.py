"""DBSCAN stroke clustering with bounding-box gap distance.

Uses bounding-box gap distance: two strokes are "close" if their
bounding boxes overlap or are within `eps` pixels of each other.
Y-axis is weighted 3x to prefer horizontal grouping (same line).
"""

import json
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN

from lib.database import get_pool
from lib.models.clustering import ClusterInfo, ClusterResponse


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


def bbox_gap_distance(entries: list[StrokeEntry]) -> np.ndarray:
    """Compute pairwise bounding-box gap distance matrix.

    Distance = 0 if bboxes overlap, otherwise Euclidean edge-to-edge gap.
    Y-axis weighted 3x to prefer horizontal grouping (same line).
    """
    n = len(entries)
    dist = np.zeros((n, n))
    for i in range(n):
        a = entries[i]
        for j in range(i + 1, n):
            b = entries[j]
            gap_x = max(0, max(a.min_x, b.min_x) - min(a.max_x, b.max_x))
            gap_y = max(0, max(a.min_y, b.min_y) - min(a.max_y, b.max_y))
            d = (gap_x ** 2 + (gap_y * 3) ** 2) ** 0.5
            dist[i, j] = d
            dist[j, i] = d
    return dist


def run_dbscan(
    entries: list[StrokeEntry],
    eps: float = 20.0,
    min_samples: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[ClusterInfo]]:
    """Run DBSCAN using bounding-box gap distance.

    Noise strokes (label -1) are assigned to the nearest non-noise cluster
    by centroid distance. If ALL strokes are noise, they form a single cluster.
    """
    if not entries:
        return np.array([]).reshape(0, 2), np.array([], dtype=int), []

    centroids = np.array([[e.centroid_x, e.centroid_y] for e in entries])
    dist_matrix = bbox_gap_distance(entries)
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(dist_matrix)

    # Handle noise: assign -1 strokes to nearest non-noise cluster
    non_noise_mask = labels != -1
    if not non_noise_mask.any():
        # All noise — treat as single cluster
        labels[:] = 0
    elif (~non_noise_mask).any():
        # Some noise — assign each to nearest non-noise cluster by centroid
        non_noise_centroids = centroids[non_noise_mask]
        non_noise_labels = labels[non_noise_mask]
        for i in np.where(~non_noise_mask)[0]:
            dists = np.linalg.norm(non_noise_centroids - centroids[i], axis=1)
            labels[i] = non_noise_labels[np.argmin(dists)]

    # Relabel to consecutive 0..N-1
    unique_labels = sorted(set(labels))
    relabel = {old: new for new, old in enumerate(unique_labels)}
    for i in range(len(labels)):
        labels[i] = relabel[labels[i]]

    # Build ClusterInfo list
    cluster_infos: list[ClusterInfo] = []
    for label in sorted(set(labels)):
        mask = labels == label
        cluster_entries = [entries[i] for i in np.where(mask)[0]]
        cluster_centroids = centroids[mask]
        cluster_infos.append(ClusterInfo(
            cluster_label=int(label),
            stroke_count=int(mask.sum()),
            centroid=[
                float(cluster_centroids[:, 0].mean()),
                float(cluster_centroids[:, 1].mean()),
            ],
            bounding_box=[
                float(min(e.min_x for e in cluster_entries)),
                float(min(e.min_y for e in cluster_entries)),
                float(max(e.max_x for e in cluster_entries)),
                float(max(e.max_y for e in cluster_entries)),
            ],
        ))

    return centroids, labels, cluster_infos


async def update_cluster_labels(session_id: str, page: int) -> list[int]:
    """Re-cluster all visible strokes for a session+page and update cluster_labels column.

    Returns list of "dirty" cluster labels that need re-transcription (new or changed clusters).
    """
    pool = get_pool()
    if not pool:
        return []

    async with pool.acquire() as conn:
        # Fetch all draw and erase events in chronological order so we can
        # resolve which strokes are actually visible (erase replaces canvas).
        all_rows = await conn.fetch(
            """
            SELECT id, strokes, event_type
            FROM stroke_logs
            WHERE session_id = $1 AND page = $2 AND event_type IN ('draw', 'erase')
            ORDER BY received_at
            """,
            session_id, page,
        )

    if not all_rows:
        # No strokes at all — clean up stale clusters
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM clusters WHERE session_id = $1 AND page = $2",
                session_id, page,
            )
        return []

    # Resolve visible rows: erase events reset the canvas
    visible_rows: list[dict] = []
    for row in all_rows:
        if row["event_type"] == "erase":
            visible_rows = [dict(row)]
        else:
            visible_rows.append(dict(row))

    if not visible_rows:
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM clusters WHERE session_id = $1 AND page = $2",
                session_id, page,
            )
        return []

    entries = extract_stroke_entries(visible_rows)
    if not entries:
        # All strokes erased — clean up stale clusters
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM clusters WHERE session_id = $1 AND page = $2",
                session_id, page,
            )
        return []

    for e in entries:
        print(f"[cluster] stroke log={e.log_id} idx={e.index} bbox=({e.min_x:.0f},{e.min_y:.0f})-({e.max_x:.0f},{e.max_y:.0f})")

    # ── Read old cluster state before re-clustering ──────────
    async with pool.acquire() as conn:
        old_cluster_rows = await conn.fetch(
            """
            SELECT cluster_label, transcription, content_type, stroke_count
            FROM clusters
            WHERE session_id = $1 AND page = $2
            """,
            session_id, page,
        )
        old_log_rows = await conn.fetch(
            """
            SELECT id, cluster_labels
            FROM stroke_logs
            WHERE session_id = $1 AND page = $2 AND event_type IN ('draw', 'erase')
            ORDER BY received_at
            """,
            session_id, page,
        )

    # Build old stroke signatures: cluster_label → set of (log_id, stroke_index)
    old_signatures: dict[int, set[tuple[int, int]]] = {}
    old_transcriptions: dict[int, str] = {}
    old_content_types: dict[int, str] = {}
    for cr in old_cluster_rows:
        lbl = cr["cluster_label"]
        old_transcriptions[lbl] = cr["transcription"] or ""
        old_content_types[lbl] = cr["content_type"] or "math"
        old_signatures[lbl] = set()

    # Resolve visible log rows for old labels (same erase logic)
    old_visible: list[dict] = []
    for row in old_log_rows:
        # We don't have event_type in this query, but the rows are from the
        # same set we already resolved above. Re-derive from all_rows order.
        pass
    # Use visible_rows we already resolved — build old signatures from their cluster_labels
    for row in visible_rows:
        log_id = row["id"]
        labels_json = row.get("cluster_labels")
        if not labels_json:
            continue
        old_labels = labels_json if isinstance(labels_json, list) else json.loads(labels_json)
        strokes_data = row["strokes"]
        strokes = strokes_data if isinstance(strokes_data, list) else json.loads(strokes_data)
        for idx, lbl in enumerate(old_labels):
            if idx < len(strokes) and strokes[idx].get("points"):
                old_signatures.setdefault(lbl, set()).add((log_id, idx))

    # ── Re-cluster ───────────────────────────────────────────
    _, labels, cluster_infos = run_dbscan(entries)
    print(f"[cluster] eps=20 bbox-gap → {len(cluster_infos)} clusters, labels={[int(l) for l in labels]}")

    # Build new stroke signatures
    new_signatures: dict[int, set[tuple[int, int]]] = {}
    for i, entry in enumerate(entries):
        lbl = int(labels[i])
        new_signatures.setdefault(lbl, set()).add((entry.log_id, entry.index))

    # Match: find unchanged clusters (same signature) and dirty ones
    dirty_labels: list[int] = []
    carried_transcriptions: dict[int, str] = {}
    carried_content_types: dict[int, str] = {}

    for new_lbl, new_sig in new_signatures.items():
        matched = False
        for old_lbl, old_sig in old_signatures.items():
            if new_sig == old_sig:
                # Unchanged — carry over transcription and content_type
                carried_transcriptions[new_lbl] = old_transcriptions.get(old_lbl, "")
                carried_content_types[new_lbl] = old_content_types.get(old_lbl, "math")
                matched = True
                print(f"[cluster] cluster {new_lbl}: unchanged (was {old_lbl})")
                break
        if not matched:
            dirty_labels.append(new_lbl)
            print(f"[cluster] cluster {new_lbl}: dirty (needs transcription)")

    # Group labels by log_id
    labels_by_log: dict[int, list[int]] = {}
    for i, entry in enumerate(entries):
        labels_by_log.setdefault(entry.log_id, []).append(int(labels[i]))

    async with pool.acquire() as conn:
        await conn.executemany(
            "UPDATE stroke_logs SET cluster_labels = $1::jsonb WHERE id = $2",
            [(json.dumps(lbls), log_id) for log_id, lbls in labels_by_log.items()],
        )

    # Delete all old cluster rows, insert fresh ones
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM clusters WHERE session_id = $1 AND page = $2",
            session_id, page,
        )

    for info in cluster_infos:
        lbl = info.cluster_label
        tx = carried_transcriptions.get(lbl, "")
        ct = carried_content_types.get(lbl, "math")
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO clusters (session_id, page, cluster_label, stroke_count,
                                      centroid_x, centroid_y, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                                      transcription, content_type)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """,
                session_id, page,
                info.cluster_label, info.stroke_count,
                info.centroid[0], info.centroid[1],
                info.bounding_box[0], info.bounding_box[1],
                info.bounding_box[2], info.bounding_box[3],
                tx, ct,
            )

    return dirty_labels


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

    centroids, labels, cluster_infos = run_dbscan(entries)

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
