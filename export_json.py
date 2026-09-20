#!/usr/bin/env python3
"""Wordagz: export the etymology DAG as sharded JSON (plus PWA icons) for the web app."""

import json
import math
import os
import shutil
import struct
import unicodedata
import zlib
from collections import defaultdict
from datetime import datetime, timezone

import networkx as nx

SHARDS = 1024
MAX_CHILDREN = 300
MAX_BUCKET = 8000
RANDOM_POOL = 3000
WILD_LIST = 300
KIND_CODE = {"inh": "i", "bor": "b", "der": "d"}
KIND_PRIORITY = {"inh": 3, "bor": 2, "der": 1}


def log(message):
    print(f"[wordagz:export] {message}", flush=True)


def fnv1a(text):
    """32-bit FNV-1a over UTF-8. Must match fnv1a() in docs/app/index.html."""
    value = 0x811C9DC5
    for byte in text.encode("utf-8", "replace"):
        value ^= byte
        value = (value * 0x01000193) & 0xFFFFFFFF
    return value


def norm(text):
    """Lowercase, strip accents. Must match norm() in docs/app/index.html."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.category(ch).startswith("M"))


def search_key(word):
    """Bucket filename from the first two normalized characters."""
    tokens = []
    for ch in norm(word)[:2]:
        tokens.append(ch if ("a" <= ch <= "z" or "0" <= ch <= "9") else f"u{ord(ch):x}")
    return "-".join(tokens) or "_"


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", errors="replace") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# PWA icons (pure stdlib PNG: a small etymology tree)
# ---------------------------------------------------------------------------

def _png(size, rows):
    raw = b"".join(b"\x00" + row for row in rows)

    def chunk(tag, body):
        crc = zlib.crc32(tag + body) & 0xFFFFFFFF
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def render_icon(size):
    bg = (15, 23, 42)
    fg = (250, 204, 21)
    root = (0.50, 0.70)
    leaves = [(0.30, 0.34), (0.50, 0.30), (0.70, 0.34)]
    nodes = [(root, 0.075)] + [(leaf, 0.055) for leaf in leaves]
    segments = [(root, leaf, 0.018) for leaf in leaves]
    rows = []
    for y in range(size):
        py = (y + 0.5) / size
        row = bytearray()
        for x in range(size):
            px = (x + 0.5) / size
            dist = 1.0
            for (cx, cy), radius in nodes:
                dist = min(dist, math.hypot(px - cx, py - cy) - radius)
            for (ax, ay), (bx, by), half in segments:
                dist = min(dist, _seg_dist(px, py, ax, ay, bx, by) - half)
            alpha = max(0.0, min(1.0, 0.5 - dist * size))
            row += bytes(int(bg[i] + (fg[i] - bg[i]) * alpha) for i in range(3))
        rows.append(bytes(row))
    return _png(size, rows)


def write_icons(app_dir):
    os.makedirs(app_dir, exist_ok=True)
    for size in (192, 512):
        with open(os.path.join(app_dir, f"icon-{size}.png"), "wb") as fh:
            fh.write(render_icon(size))


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_json(G, out_dir):
    data_dir = os.path.join(out_dir, "data")
    shutil.rmtree(data_dir, ignore_errors=True)

    # Node shards: data/n/<fnv1a(id) % SHARDS>.json -> {node_id: record}
    shards = defaultdict(dict)
    for node_id, data in G.nodes(data=True):
        parents = sorted(
            G.predecessors(node_id),
            key=lambda p: (-KIND_PRIORITY[G[p][node_id]["kind"]], p),
        )
        children = sorted(G.successors(node_id), key=lambda c: (-G.out_degree(c), c))
        record = {
            "d": data.get("definition", ""),
            "p": [[p, KIND_CODE[G[p][node_id]["kind"]]] for p in parents],
            "c": [
                [c, KIND_CODE[G[node_id][c]["kind"]], G.out_degree(c)]
                for c in children[:MAX_CHILDREN]
            ],
        }
        if len(children) > MAX_CHILDREN:
            record["n"] = len(children)
        if data.get("pos"):
            record["t"] = data["pos"]
        if not data.get("has_entry"):
            record["s"] = 1
        if data.get("drift"):
            record["x"] = data["drift_root"]
        shards[fnv1a(node_id) % SHARDS][node_id] = record

    for index, records in shards.items():
        write_json(os.path.join(data_dir, "n", f"{index}.json"), records)
    log(f"Wrote {len(shards):,} node shards")

    # Search buckets: data/s/<key>.json -> [node_id, ...] best-ranked first
    ranked = sorted(
        G.nodes,
        key=lambda n: (0 if G.nodes[n]["lang_code"] == "en" else 1, -G.degree(n), n),
    )
    buckets = defaultdict(list)
    for node_id in ranked:
        bucket = buckets[search_key(G.nodes[node_id]["word"])]
        if len(bucket) < MAX_BUCKET:
            bucket.append(node_id)
    for key, ids in buckets.items():
        write_json(os.path.join(data_dir, "s", f"{key}.json"), ids)
    log(f"Wrote {len(buckets):,} search buckets")

    # Metadata + featured lists
    langs = {}
    for _, data in G.nodes(data=True):
        langs.setdefault(data["lang_code"], data["lang"])
    write_json(os.path.join(data_dir, "meta.json"), {
        "shards": SHARDS,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "built": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "langs": langs,
    })

    pool = sorted(
        (n for n, d in G.nodes(data=True)
         if d["lang_code"] == "en" and d.get("has_entry") and G.in_degree(n) > 0),
        key=lambda n: (-G.degree(n), n),
    )[:RANDOM_POOL]
    wild = sorted(
        (n for n, d in G.nodes(data=True) if d.get("drift")),
        key=lambda n: (-len(nx.ancestors(G, n)), n),
    )[:WILD_LIST]
    write_json(os.path.join(data_dir, "featured.json"), {
        "pool": pool,
        "wild": [[n, G.nodes[n]["drift_root"]] for n in wild],
    })

    write_icons(os.path.join(out_dir, "app"))
    log("Export complete")