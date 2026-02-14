"""Unit tests for bounding-box overlap stroke clustering (no database)."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.stroke_clustering import StrokeEntry, extract_stroke_entries, cluster_by_bbox_overlap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stroke(x: float, y: float) -> dict:
    """Create a single-point stroke at (x, y)."""
    return {"points": [{"x": x, "y": y}]}


def _make_line_stroke(points: list[tuple[float, float]]) -> dict:
    """Create a multi-point stroke from a list of (x, y) tuples."""
    return {"points": [{"x": x, "y": y} for x, y in points]}


def _point_entry(x: float, y: float, idx: int = 0) -> StrokeEntry:
    """Single-point stroke entry at (x, y)."""
    return StrokeEntry(log_id=1, index=idx, min_x=x, min_y=y, max_x=x, max_y=y)


def _box_entry(x1: float, y1: float, x2: float, y2: float, idx: int = 0) -> StrokeEntry:
    """Stroke entry with explicit bounding box."""
    return StrokeEntry(log_id=1, index=idx, min_x=x1, min_y=y1, max_x=x2, max_y=y2)


# ---------------------------------------------------------------------------
# extract_stroke_entries
# ---------------------------------------------------------------------------

class TestExtractStrokeEntries:

    def test_single_point_stroke(self):
        rows = [{"id": 1, "strokes": [_make_stroke(100.0, 200.0)]}]
        entries = extract_stroke_entries(rows)
        assert len(entries) == 1
        assert entries[0].centroid_x == 100.0
        assert entries[0].centroid_y == 200.0
        assert entries[0].log_id == 1
        assert entries[0].index == 0

    def test_multi_point_centroid(self):
        stroke = _make_line_stroke([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
        rows = [{"id": 1, "strokes": [stroke]}]
        entries = extract_stroke_entries(rows)
        assert len(entries) == 1
        assert entries[0].centroid_x == pytest.approx(5.0)
        assert entries[0].centroid_y == pytest.approx(5.0)

    def test_multiple_strokes_per_row(self):
        rows = [{"id": 1, "strokes": [_make_stroke(10, 20), _make_stroke(30, 40)]}]
        entries = extract_stroke_entries(rows)
        assert len(entries) == 2
        assert entries[0].index == 0
        assert entries[1].index == 1

    def test_multiple_rows(self):
        rows = [
            {"id": 1, "strokes": [_make_stroke(10, 20)]},
            {"id": 2, "strokes": [_make_stroke(30, 40)]},
        ]
        entries = extract_stroke_entries(rows)
        assert len(entries) == 2
        assert entries[0].log_id == 1
        assert entries[1].log_id == 2

    def test_empty_points_skipped(self):
        rows = [{"id": 1, "strokes": [{"points": []}, _make_stroke(5, 5)]}]
        entries = extract_stroke_entries(rows)
        assert len(entries) == 1
        assert entries[0].index == 1

    def test_no_points_key_skipped(self):
        rows = [{"id": 1, "strokes": [{}]}]
        entries = extract_stroke_entries(rows)
        assert len(entries) == 0

    def test_empty_rows(self):
        assert extract_stroke_entries([]) == []

    def test_json_string_strokes(self):
        import json
        rows = [{"id": 1, "strokes": json.dumps([_make_stroke(7.0, 8.0)])}]
        entries = extract_stroke_entries(rows)
        assert len(entries) == 1
        assert entries[0].centroid_x == 7.0


# ---------------------------------------------------------------------------
# cluster_by_bbox_overlap
# ---------------------------------------------------------------------------

class TestClusterByBboxOverlap:

    def test_two_separate_clusters(self):
        """Two groups far apart should form two clusters."""
        entries = [
            _box_entry(90, 90, 120, 120, idx=0),
            _box_entry(95, 100, 115, 115, idx=1),
            _box_entry(900, 900, 1020, 1020, idx=2),
            _box_entry(910, 910, 1010, 1010, idx=3),
        ]
        _, labels, infos = cluster_by_bbox_overlap(entries)

        assert len(infos) == 2
        assert infos[0].stroke_count == 2
        assert infos[1].stroke_count == 2

    def test_single_cluster_overlapping(self):
        """Overlapping bboxes should all merge into one cluster."""
        entries = [
            _box_entry(0, 0, 50, 50, idx=0),
            _box_entry(40, 0, 90, 50, idx=1),
            _box_entry(80, 0, 130, 50, idx=2),
        ]
        _, labels, infos = cluster_by_bbox_overlap(entries)

        assert len(infos) == 1
        assert infos[0].stroke_count == 3

    def test_isolated_points_separate_clusters(self):
        """Points far apart with no overlap get their own clusters."""
        entries = [
            _point_entry(0, 0, idx=0),
            _point_entry(500, 500, idx=1),
            _point_entry(1000, 0, idx=2),
        ]
        _, labels, infos = cluster_by_bbox_overlap(entries)

        assert len(infos) == 3
        assert all(info.stroke_count == 1 for info in infos)

    def test_single_stroke_one_cluster(self):
        """A single stroke forms its own cluster."""
        entries = [_point_entry(100, 100)]
        _, labels, infos = cluster_by_bbox_overlap(entries)

        assert len(infos) == 1
        assert labels[0] == 0

    def test_touching_edges_connect(self):
        """Two bboxes sharing an edge should connect."""
        # bbox A: 0-100, bbox B: 100-200 — touching at x=100
        entries = [
            _box_entry(0, 0, 100, 50, idx=0),
            _box_entry(100, 0, 200, 50, idx=1),
        ]
        _, labels, infos = cluster_by_bbox_overlap(entries)

        assert len(infos) == 1
        assert infos[0].stroke_count == 2

    def test_any_gap_separates(self):
        """Two bboxes with any gap between them stay separate."""
        # bbox A: 0-100, bbox B: 101-200 — 1px gap
        entries = [
            _box_entry(0, 0, 100, 50, idx=0),
            _box_entry(101, 0, 200, 50, idx=1),
        ]
        _, labels, infos = cluster_by_bbox_overlap(entries)

        assert len(infos) == 2

    def test_merge_multiple_clusters(self):
        """A stroke bridging two existing clusters should merge them."""
        entries = [
            _box_entry(0, 0, 50, 50, idx=0),      # cluster A
            _box_entry(200, 0, 250, 50, idx=1),    # cluster B (separate)
            _box_entry(40, 0, 210, 50, idx=2),     # bridges A and B
        ]
        _, labels, infos = cluster_by_bbox_overlap(entries)

        assert len(infos) == 1
        assert infos[0].stroke_count == 3

    def test_centroid_accuracy(self):
        """Cluster centroid should be mean of member centroids."""
        entries = [
            _box_entry(0, 0, 100, 100, idx=0),     # centroid (50, 50)
            _box_entry(50, 50, 150, 150, idx=1),    # centroid (100, 100)
        ]
        _, _, infos = cluster_by_bbox_overlap(entries)

        assert len(infos) == 1
        assert infos[0].centroid[0] == pytest.approx(75.0)
        assert infos[0].centroid[1] == pytest.approx(75.0)

    def test_bounding_box_accuracy(self):
        """Cluster bounding box should span all member bboxes."""
        entries = [
            _box_entry(10, 20, 50, 60, idx=0),
            _box_entry(30, 40, 80, 90, idx=1),
        ]
        _, _, infos = cluster_by_bbox_overlap(entries)

        assert len(infos) == 1
        bbox = infos[0].bounding_box
        assert bbox[0] == pytest.approx(10.0)
        assert bbox[1] == pytest.approx(20.0)
        assert bbox[2] == pytest.approx(80.0)
        assert bbox[3] == pytest.approx(90.0)

    def test_labels_sequential(self):
        """Cluster labels should be 0, 1, 2... in order."""
        entries = [
            _box_entry(0, 0, 10, 10, idx=0),
            _box_entry(500, 500, 510, 510, idx=1),
            _box_entry(1000, 0, 1010, 10, idx=2),
        ]
        _, _, infos = cluster_by_bbox_overlap(entries)

        assert len(infos) == 3
        assert [i.cluster_label for i in infos] == [0, 1, 2]

    def test_realistic_two_lines(self):
        """Two lines of handwriting on iPad canvas should separate."""
        # Line 1: strokes around y=400
        # Line 2: strokes around y=1200
        entries = [
            _box_entry(280, 380, 320, 420, idx=0),
            _box_entry(310, 390, 350, 420, idx=1),
            _box_entry(340, 380, 380, 410, idx=2),
            _box_entry(280, 1180, 320, 1220, idx=3),
            _box_entry(310, 1190, 350, 1220, idx=4),
            _box_entry(340, 1180, 380, 1210, idx=5),
        ]
        _, labels, infos = cluster_by_bbox_overlap(entries)

        assert len(infos) == 2
        assert infos[0].stroke_count == 3
        assert infos[1].stroke_count == 3

    def test_empty_entries(self):
        """Empty input returns empty output."""
        centroids, labels, infos = cluster_by_bbox_overlap([])
        assert len(infos) == 0
        assert len(labels) == 0
        assert centroids.shape == (0, 2)
