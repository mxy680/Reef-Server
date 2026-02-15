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

# Per-session token usage accumulator
_session_usage: dict[str, dict] = {}


def get_session_usage(session_id: str) -> dict:
    """Return cumulative token usage for a session."""
    return _session_usage.get(session_id, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0})


def _accumulate_usage(session_id: str, usage: dict) -> None:
    """Add token counts from a single transcription call to the session total."""
    if session_id not in _session_usage:
        _session_usage[session_id] = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    _session_usage[session_id]["prompt_tokens"] += usage.get("prompt_tokens", 0)
    _session_usage[session_id]["completion_tokens"] += usage.get("completion_tokens", 0)
    _session_usage[session_id]["calls"] += 1


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

    # Merge clusters with overlapping y-bounding boxes (e.g. crossing marks
    # on "2"s create strokes at slightly different y, splitting one line into
    # multiple clusters).  If two clusters overlap vertically by more than 50%
    # of the smaller one's height, they belong to the same line.
    merge_map: dict[int, int] = {}  # old_label → canonical_label
    for lbl in range(current_label + 1):
        merge_map[lbl] = lbl

    def _find(lbl: int) -> int:
        while merge_map[lbl] != lbl:
            lbl = merge_map[lbl]
        return lbl

    # Collect per-cluster y-ranges
    cluster_y_ranges: dict[int, tuple[float, float]] = {}
    for lbl in range(current_label + 1):
        mask = labels == lbl
        member_entries = [entries[i] for i in np.where(mask)[0]]
        if member_entries:
            cluster_y_ranges[lbl] = (
                min(e.min_y for e in member_entries),
                max(e.max_y for e in member_entries),
            )

    for a in range(current_label + 1):
        for b in range(a + 1, current_label + 1):
            if _find(a) == _find(b):
                continue
            if a not in cluster_y_ranges or b not in cluster_y_ranges:
                continue
            a_min, a_max = cluster_y_ranges[a]
            b_min, b_max = cluster_y_ranges[b]
            overlap = max(0, min(a_max, b_max) - max(a_min, b_min))
            smaller_height = min(a_max - a_min, b_max - b_min) or 1
            if overlap / smaller_height > 0.5:
                merge_map[_find(b)] = _find(a)

    # Relabel after y-overlap merging
    canonical_set = sorted(set(_find(l) for l in range(current_label + 1)))
    relabel = {old: new for new, old in enumerate(canonical_set)}
    for i in range(len(labels)):
        labels[i] = relabel[_find(int(labels[i]))]

    # Diagram-merge pass: merge vertically adjacent clusters that form a
    # square-ish combined shape (diagrams, not text lines).
    num_after_overlap = len(canonical_set)
    diag_merge: dict[int, int] = {l: l for l in range(num_after_overlap)}

    def _dfind(lbl: int) -> int:
        while diag_merge[lbl] != lbl:
            lbl = diag_merge[lbl]
        return lbl

    # Compute per-cluster bounding boxes
    cluster_bboxes: dict[int, tuple[float, float, float, float]] = {}
    for lbl in range(num_after_overlap):
        mask = labels == lbl
        members = [entries[i] for i in np.where(mask)[0]]
        if members:
            cluster_bboxes[lbl] = (
                min(e.min_x for e in members),
                min(e.min_y for e in members),
                max(e.max_x for e in members),
                max(e.max_y for e in members),
            )

    # Sort clusters by min_y for pairwise checking
    sorted_labels = sorted(cluster_bboxes.keys(), key=lambda l: cluster_bboxes[l][1])
    for i in range(len(sorted_labels)):
        for j in range(i + 1, len(sorted_labels)):
            a, b = sorted_labels[i], sorted_labels[j]
            if _dfind(a) == _dfind(b):
                continue
            ax1, ay1, ax2, ay2 = cluster_bboxes[a]
            bx1, by1, bx2, by2 = cluster_bboxes[b]

            # Check x-overlap > 30%
            x_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
            smaller_width = min(ax2 - ax1, bx2 - bx1) or 1
            if x_overlap / smaller_width < 0.3:
                continue

            # Skip if either cluster is text-like (wide relative to tall)
            a_ar = (ax2 - ax1) / max(ay2 - ay1, 1)
            b_ar = (bx2 - bx1) / max(by2 - by1, 1)
            if a_ar >= 2.5 or b_ar >= 2.5:
                continue

            # Check combined aspect ratio
            cx1 = min(ax1, bx1)
            cy1 = min(ay1, by1)
            cx2 = max(ax2, bx2)
            cy2 = max(ay2, by2)
            combined_ar = (cx2 - cx1) / max(cy2 - cy1, 1)
            if combined_ar < 2.5:
                diag_merge[_dfind(b)] = _dfind(a)

    # Relabel after diagram merging
    canonical_set = sorted(set(_dfind(l) for l in range(num_after_overlap)))
    drelabel = {old: new for new, old in enumerate(canonical_set)}
    for i in range(len(labels)):
        labels[i] = drelabel[_dfind(int(labels[i]))]

    # Intersection merge: merge any clusters whose bounding boxes intersect.
    num_after_diag = len(canonical_set)
    int_merge: dict[int, int] = {l: l for l in range(num_after_diag)}

    def _ifind(lbl: int) -> int:
        while int_merge[lbl] != lbl:
            lbl = int_merge[lbl]
        return lbl

    # Recompute bounding boxes after previous merges
    int_bboxes: dict[int, tuple[float, float, float, float]] = {}
    for lbl in range(num_after_diag):
        mask = labels == lbl
        members = [entries[i] for i in np.where(mask)[0]]
        if members:
            int_bboxes[lbl] = (
                min(e.min_x for e in members),
                min(e.min_y for e in members),
                max(e.max_x for e in members),
                max(e.max_y for e in members),
            )

    # Iteratively merge until no more intersections (transitive closure)
    changed = True
    while changed:
        changed = False
        # Recompute canonical bboxes after merges
        canon_bboxes: dict[int, tuple[float, float, float, float]] = {}
        for lbl in range(num_after_diag):
            c = _ifind(lbl)
            if c not in canon_bboxes and lbl in int_bboxes:
                canon_bboxes[c] = int_bboxes[lbl]
            elif lbl in int_bboxes and c in canon_bboxes:
                cb = canon_bboxes[c]
                ib = int_bboxes[lbl]
                canon_bboxes[c] = (
                    min(cb[0], ib[0]), min(cb[1], ib[1]),
                    max(cb[2], ib[2]), max(cb[3], ib[3]),
                )

        canons = sorted(canon_bboxes.keys())
        for i in range(len(canons)):
            for j in range(i + 1, len(canons)):
                a, b = canons[i], canons[j]
                if _ifind(a) == _ifind(b):
                    continue
                ax1, ay1, ax2, ay2 = canon_bboxes[a]
                bx1, by1, bx2, by2 = canon_bboxes[b]
                # Two rectangles intersect if they overlap in both x and y
                if ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2:
                    int_merge[_ifind(b)] = _ifind(a)
                    changed = True

    # Relabel after intersection merging
    canonical_set = sorted(set(_ifind(l) for l in range(num_after_diag)))
    irelabel = {old: new for new, old in enumerate(canonical_set)}
    for i in range(len(labels)):
        labels[i] = irelabel[_ifind(int(labels[i]))]

    # Build ClusterInfo list
    num_clusters = len(canonical_set)
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
    """Re-cluster all visible strokes for a session+page and update cluster_labels column."""
    pool = get_pool()
    if not pool:
        return

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
        return

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
        return

    # Build stroke lookup: (log_id, stroke_index) → stroke dict
    stroke_lookup: dict[tuple[int, int], dict] = {}
    for row in visible_rows:
        log_id = row["id"]
        strokes_json = row["strokes"]
        strokes = strokes_json if isinstance(strokes_json, list) else json.loads(strokes_json)
        for idx, stroke in enumerate(strokes):
            stroke_lookup[(log_id, idx)] = stroke

    entries = extract_stroke_entries(visible_rows)
    if not entries:
        # All strokes erased — clean up stale clusters
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM clusters WHERE session_id = $1 AND page = $2",
                session_id, page,
            )
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

    # Look up problem context for this session (most recent context message)
    problem_context = ""
    async with pool.acquire() as conn:
        ctx_row = await conn.fetchrow(
            """
            SELECT problem_context FROM stroke_logs
            WHERE session_id = $1 AND problem_context != ''
            ORDER BY received_at DESC LIMIT 1
            """,
            session_id,
        )
        if ctx_row:
            problem_context = ctx_row["problem_context"]
            print(f"[transcribe] using problem context: {problem_context[:80]}...")

    # Delete stale cluster rows, then render + transcribe each cluster
    current_labels = [info.cluster_label for info in cluster_infos]
    async with pool.acquire() as conn:
        if current_labels:
            await conn.execute(
                """
                DELETE FROM clusters
                WHERE session_id = $1 AND page = $2
                  AND cluster_label != ALL($3::int[])
                """,
                session_id, page, current_labels,
            )
        else:
            await conn.execute(
                "DELETE FROM clusters WHERE session_id = $1 AND page = $2",
                session_id, page,
            )

    for info in cluster_infos:
        label = info.cluster_label
        content_type = "math"
        cluster_strokes = strokes_by_cluster.get(label, [])
        transcription = ""
        if cluster_strokes:
            try:
                image_bytes = render_strokes(cluster_strokes)
                result = await asyncio.to_thread(transcribe_strokes_image, image_bytes, problem_context)
                content_type = result["content_type"]
                transcription = result["transcription"]
                if result.get("usage"):
                    _accumulate_usage(session_id, result["usage"])
                print(f"[transcribe] cluster {label} ({content_type}): {transcription[:80]}")
            except Exception as exc:
                print(f"[transcribe] cluster {label} failed: {exc}")

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO clusters (session_id, page, cluster_label, stroke_count,
                                      centroid_x, centroid_y, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                                      transcription, content_type)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (session_id, page, cluster_label) DO UPDATE SET
                    stroke_count = EXCLUDED.stroke_count,
                    centroid_x = EXCLUDED.centroid_x, centroid_y = EXCLUDED.centroid_y,
                    bbox_x1 = EXCLUDED.bbox_x1, bbox_y1 = EXCLUDED.bbox_y1,
                    bbox_x2 = EXCLUDED.bbox_x2, bbox_y2 = EXCLUDED.bbox_y2,
                    transcription = EXCLUDED.transcription,
                    content_type = EXCLUDED.content_type
                """,
                session_id, page,
                label, info.stroke_count,
                info.centroid[0], info.centroid[1],
                info.bounding_box[0], info.bounding_box[1],
                info.bounding_box[2], info.bounding_box[3],
                transcription, content_type,
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
