#!/usr/bin/env python3
"""Wordagz: Kaikki Wiktionary JSONL -> etymology DAG -> static Docsify wiki."""

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import time
import unicodedata
from collections import defaultdict

import networkx as nx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEMPLATE_KINDS = {
    "inh": "inh", "inh+": "inh",
    "bor": "bor", "bor+": "bor",
    "der": "der", "der+": "der",
}
EDGE_PRIORITY = {"inh": 3, "bor": 2, "der": 1}
EDGE_LABEL = {"inh": "inherited", "bor": "borrowed", "der": "derived"}

DEF_MAX = 200          # max characters stored per definition
MAX_LIST = 200         # max ancestors/descendants listed per page
MAX_HUB_WORDS = 500    # max words listed per language hub page
MAX_DRIFT_LIST = 1000  # max entries on the drift report page

LANG_NAMES = {
    "en": "English", "enm": "Middle English", "ang": "Old English",
    "sco": "Scots", "ofs": "Old Frisian", "osx": "Old Saxon",
    "fr": "French", "fro": "Old French", "frm": "Middle French",
    "xno": "Anglo-Norman", "la": "Latin", "LL.": "Late Latin",
    "ML.": "Medieval Latin", "VL.": "Vulgar Latin", "NL.": "New Latin",
    "grc": "Ancient Greek", "el": "Greek", "de": "German",
    "gmh": "Middle High German", "goh": "Old High German",
    "nl": "Dutch", "dum": "Middle Dutch", "odt": "Old Dutch",
    "non": "Old Norse", "sv": "Swedish", "da": "Danish", "no": "Norwegian",
    "is": "Icelandic", "got": "Gothic", "es": "Spanish",
    "osp": "Old Spanish", "pt": "Portuguese", "it": "Italian",
    "ca": "Catalan", "ar": "Arabic", "he": "Hebrew", "hi": "Hindi",
    "sa": "Sanskrit", "fa": "Persian", "tr": "Turkish", "ru": "Russian",
    "pl": "Polish", "cs": "Czech", "ja": "Japanese", "zh": "Chinese",
    "ko": "Korean", "cy": "Welsh", "ga": "Irish", "sga": "Old Irish",
    "gd": "Scottish Gaelic", "gem-pro": "Proto-Germanic",
    "gmw-pro": "Proto-West Germanic", "ine-pro": "Proto-Indo-European",
    "itc-pro": "Proto-Italic", "cel-pro": "Proto-Celtic",
    "sla-pro": "Proto-Slavic",
    "cmn": "Mandarin", "ro": "Romanian", "uk": "Ukrainian",
    "mul": "Translingual", "la-lat": "Late Latin",
    "la-med": "Medieval Latin", "la-new": "New Latin",
}

# Function words plus dictionary-gloss boilerplate that carries no meaning.
STOPWORDS = frozenset("""
the and for with from that this these those which who whom whose into onto
over under about between through than then them they their there here when
where while what will would should could have has had does did not nor but
are was were been being its his her him she you your our out off any all
one used use uses using often usually especially something someone somebody
thing things other another such also more most very can may might
form forms alternative obsolete archaic dated rare informal slang uncommon
nonstandard proscribed eye dialect spelling variant misspelling common
plural singular past present participle tense person third first second
simple gerund verb noun adjective adverb pronoun preposition conjunction
interjection numeral article particle definite indefinite feminine
masculine neuter diminutive augmentative comparative superlative synonym
genitive dative accusative nominative ablative vocative locative
""".split())

WORD_RE = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
INLINE_MOD_RE = re.compile(r"<([a-z]+):([^>]*)>")
MD_SPECIAL_RE = re.compile(r"([\\`*_{}\[\]<>#|~])")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def log(message):
    print(f"[wordagz] {message}", flush=True)


def clean_text(value):
    if not value:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def md_escape(text):
    return MD_SPECIAL_RE.sub(r"\\\1", text or "")


def slugify(text, fallback, limit=48):
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")[:limit]
    return slug or fallback


def write_file(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def open_dump(path):
    """Open a JSONL dump for streaming; gzip is detected by magic bytes."""
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def extract_definition(entry):
    """First meaningful gloss; form-of/alt-of senses are only a fallback."""
    fallback = ""
    for sense in entry.get("senses") or []:
        glosses = sense.get("glosses") or sense.get("raw_glosses") or []
        if not glosses:
            continue
        text = clean_text(glosses[-1])
        if not text:
            continue
        if set(sense.get("tags") or []) & {"form-of", "alt-of"}:
            fallback = fallback or text
            continue
        return text[:DEF_MAX]
    return fallback[:DEF_MAX]


def split_term(raw):
    """Strip inline modifiers like word<t:gloss>; return (term, gloss)."""
    raw = clean_text(raw)
    gloss = ""
    for key, value in INLINE_MOD_RE.findall(raw):
        if key in ("t", "gloss") and not gloss:
            gloss = clean_text(value)
    term = re.sub(r"<[^>]*>", "", raw).strip()
    return term, gloss


def ensure_node(G, lang_code, word, lang_name=None, definition=None, pos=None,
                has_entry=False):
    node_id = f"{lang_code}:{word}"
    if node_id not in G:
        G.add_node(
            node_id,
            word=word,
            lang_code=sys.intern(lang_code),
            lang=lang_name or LANG_NAMES.get(lang_code, lang_code),
            definition="",
            pos="",
            has_entry=False,
        )
    data = G.nodes[node_id]
    if has_entry:
        if not data["has_entry"]:
            data["has_entry"] = True
            if lang_name:
                data["lang"] = lang_name
            if definition:
                data["definition"] = definition
            if pos:
                data["pos"] = pos
        elif definition and not data["definition"]:
            data["definition"] = definition
    elif definition and not data["definition"]:
        data["definition"] = definition
    return node_id


def add_edge(G, source, dest, kind):
    if G.has_edge(source, dest):
        if EDGE_PRIORITY[kind] > EDGE_PRIORITY[G[source][dest]["kind"]]:
            G[source][dest]["kind"] = kind
    else:
        G.add_edge(source, dest, kind=kind)


def process_entry(G, entry):
    """Add one Kaikki entry plus its inh/der/bor ancestry edges (ancestor -> descendant)."""
    word = clean_text(entry.get("word"))
    lang_code = clean_text(entry.get("lang_code"))
    if not word or not lang_code:
        return False

    entry_id = ensure_node(
        G, lang_code, word,
        lang_name=clean_text(entry.get("lang")) or None,
        definition=extract_definition(entry),
        pos=clean_text(entry.get("pos")),
        has_entry=True,
    )

    previous = None  # (lang_code, node_id) of the last ancestor added in this chain
    for template in entry.get("etymology_templates") or []:
        kind = TEMPLATE_KINDS.get(template.get("name"))
        if kind is None:
            continue
        args = template.get("args") or {}
        target_lang = clean_text(args.get("1"))
        source_lang = clean_text(args.get("2"))
        source_term, inline_gloss = split_term(args.get("3"))
        if not source_lang or source_term in ("", "-"):
            continue

        # {{inh|en|enm|foo}} points at this entry; a following
        # {{inh|enm|ang|fo}} continues the chain from the previous ancestor.
        if not target_lang or target_lang == lang_code:
            dest = entry_id
        elif previous and previous[0] == target_lang:
            dest = previous[1]
        else:
            continue

        gloss = clean_text(args.get("t") or args.get("gloss") or args.get("5")) or inline_gloss
        source_id = ensure_node(G, source_lang, source_term,
                                definition=gloss[:DEF_MAX] or None)
        previous = (source_lang, source_id)
        if source_id != dest:
            add_edge(G, source_id, dest, kind)
    return True


def build_graph(path, limit=0):
    G = nx.DiGraph()
    started = time.time()
    lines = entries = 0
    with open_dump(path) as fh:
        for line in fh:
            lines += 1
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    entry = None
                if isinstance(entry, dict) and process_entry(G, entry):
                    entries += 1
            if lines % 100000 == 0:
                log(f"{lines:,} lines | {G.number_of_nodes():,} nodes | "
                    f"{G.number_of_edges():,} edges | {time.time() - started:.0f}s")
            if limit and lines >= limit:
                break
    log(f"Parsed {entries:,} entries from {lines:,} lines in {time.time() - started:.0f}s")
    return G


def enforce_dag(G):
    """Remove the weakest edge of every cycle so the graph is a true DAG."""
    removed = 0
    for component in list(nx.strongly_connected_components(G)):
        if len(component) < 2:
            continue
        sub = G.subgraph(component).copy()
        while True:
            try:
                cycle = nx.find_cycle(sub)
            except nx.NetworkXNoCycle:
                break
            u, v = min(
                ((edge[0], edge[1]) for edge in cycle),
                key=lambda e: EDGE_PRIORITY[sub[e[0]][e[1]]["kind"]],
            )
            sub.remove_edge(u, v)
            G.remove_edge(u, v)
            removed += 1
    return removed


# ---------------------------------------------------------------------------
# Semantic diff tracker
# ---------------------------------------------------------------------------

def stem(token):
    """Very light suffix stripper; good enough for keyword overlap."""
    if token.endswith("ies") and len(token) > 4:
        token = token[:-3] + "y"
    else:
        for suffix in ("ations", "ation", "ingly", "ness", "ings", "ing",
                       "edly", "ed", "ly", "es", "s"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                token = token[: -len(suffix)]
                break
    if len(token) > 3 and token[-1] in "ey":
        token = token[:-1]
    return token


def keywords(text):
    result = set()
    for token in WORD_RE.findall((text or "").lower()):
        if token in STOPWORDS:
            continue
        root = stem(token)
        if len(root) >= 3 and root not in STOPWORDS:
            result.add(root)
    return result


def tag_semantic_drift(G):
    """Flag terminal descendants whose definition shares zero keywords with
    the definitions of their root ancestors."""
    root_cache = {}

    def root_keywords(node_id):
        if node_id not in root_cache:
            root_cache[node_id] = keywords(G.nodes[node_id].get("definition", ""))
        return root_cache[node_id]

    flagged = 0
    for node_id in G.nodes:
        if G.out_degree(node_id) != 0 or G.in_degree(node_id) == 0:
            continue
        data = G.nodes[node_id]
        leaf_kw = keywords(data.get("definition", ""))
        if not leaf_kw:
            continue
        roots = sorted(
            a for a in nx.ancestors(G, node_id)
            if G.in_degree(a) == 0 and root_keywords(a)
        )
        if not roots:
            continue
        combined = set().union(*(root_keywords(r) for r in roots))
        if leaf_kw.isdisjoint(combined):
            data["drift"] = True
            data["drift_root"] = roots[0]
            flagged += 1
    return flagged


# ---------------------------------------------------------------------------
# Static site generator
# ---------------------------------------------------------------------------

def lang_slug(code):
    return slugify(code, "xx", limit=32)


def node_path(node_id, data):
    digest = hashlib.md5(node_id.encode("utf-8"), usedforsecurity=False).hexdigest()
    slug = slugify(data["word"], "w")
    return f"words/{lang_slug(data['lang_code'])}/{digest[:2]}/{slug}-{digest[2:10]}.md"


def rel_link(from_path, to_path):
    return os.path.relpath(to_path, os.path.dirname(from_path)).replace(os.sep, "/")


def link_or_text(G, target, here, paths):
    label = md_escape(G.nodes[target]["word"])
    if target in paths:
        return f"[{label}]({rel_link(here, paths[target])})"
    return label


def sorted_neighbors(G, iterator):
    return sorted(iterator, key=lambda n: (G.nodes[n]["word"].lower(), n))


def neighbor_lines(G, node_id, neighbors, here, paths, incoming):
    lines = []
    for other in neighbors[:MAX_LIST]:
        kind = G[other][node_id]["kind"] if incoming else G[node_id][other]["kind"]
        lines.append(
            f"- {link_or_text(G, other, here, paths)} — "
            f"{md_escape(G.nodes[other]['lang'])} *({EDGE_LABEL[kind]})*"
        )
    extra = len(neighbors) - MAX_LIST
    if extra > 0:
        lines.append(f"- _...and {extra:,} more._")
    return lines


def render_page(G, node_id, paths):
    data = G.nodes[node_id]
    here = paths[node_id]

    out = [f"# {md_escape(data['word'])}", ""]
    meta = f"**Language:** {md_escape(data['lang'])} (`{data['lang_code']}`)"
    if data.get("pos"):
        meta += f" · **Part of speech:** {md_escape(data['pos'])}"
    out += [meta, "", "## Definition", ""]

    if data.get("definition"):
        out.append(md_escape(data["definition"]))
        if not data.get("has_entry"):
            out += ["", "_Gloss taken from an etymology template; no dictionary entry in this dataset._"]
    else:
        out.append("_No definition recorded._")
    out.append("")

    if data.get("drift"):
        root_id = data["drift_root"]
        root_ref = link_or_text(G, root_id, here, paths)
        root_def = md_escape(G.nodes[root_id].get("definition", ""))
        out += [
            f'!> **Semantic drift warning:** this definition shares no meaningful keywords '
            f'with its root ancestor {root_ref} ("{root_def}").',
            "",
        ]

    ancestors = sorted_neighbors(G, G.predecessors(node_id))
    descendants = sorted_neighbors(G, G.successors(node_id))

    out += ["## Direct ancestors", ""]
    if ancestors:
        out += neighbor_lines(G, node_id, ancestors, here, paths, incoming=True)
    else:
        out.append("_None recorded (this is a root)._")
    out += ["", "## Direct descendants", ""]
    if descendants:
        out += neighbor_lines(G, node_id, descendants, here, paths, incoming=False)
    else:
        out.append("_None recorded._")
    out.append("")
    return "\n".join(out)


def select_page_nodes(G, max_pages):
    connected = [n for n in G.nodes if G.degree(n) > 0]
    if max_pages and len(connected) > max_pages:
        connected.sort(key=lambda n: (-G.degree(n), n))
        connected = connected[:max_pages]
    return connected


def write_language_hubs(G, out_dir, paths, page_nodes):
    by_lang = defaultdict(list)
    for node_id in page_nodes:
        by_lang[G.nodes[node_id]["lang_code"]].append(node_id)

    rows = []
    for code, ids in by_lang.items():
        ids.sort(key=lambda n: (-G.degree(n), G.nodes[n]["word"].lower()))
        slug = lang_slug(code)
        rel = f"languages/{slug}.md"
        name = G.nodes[ids[0]]["lang"]
        lines = [
            f"# {md_escape(name)} (`{code}`)", "",
            f"{len(ids):,} pages. Showing the {min(len(ids), MAX_HUB_WORDS):,} best-connected.", "",
        ]
        lines += [f"- {link_or_text(G, n, rel, paths)}" for n in ids[:MAX_HUB_WORDS]]
        write_file(os.path.join(out_dir, rel), "\n".join(lines) + "\n")
        rows.append((len(ids), name, slug))

    rows.sort(key=lambda r: (-r[0], r[1].lower()))
    index = ["# Languages", "", f"{len(rows):,} languages with connected etymologies.", ""]
    index += [f"- [{md_escape(name)}]({slug}.md) — {count:,} pages" for count, name, slug in rows]
    write_file(os.path.join(out_dir, "languages", "README.md"), "\n".join(index) + "\n")
    return len(rows)


def write_drift_report(G, out_dir, paths):
    flagged = sorted(
        (n for n, d in G.nodes(data=True) if d.get("drift")),
        key=lambda n: (G.nodes[n]["word"].lower(), n),
    )
    here = "drift.md"
    lines = [
        "# Semantic drift report", "",
        "Terminal descendants whose definition shares zero meaningful keywords "
        "with their root ancestor's definition.", "",
        f"**{len(flagged):,}** words flagged. Showing the first {min(len(flagged), MAX_DRIFT_LIST):,}.", "",
    ]
    for node_id in flagged[:MAX_DRIFT_LIST]:
        data = G.nodes[node_id]
        root_id = data["drift_root"]
        lines.append(
            f'- {link_or_text(G, node_id, here, paths)} ({md_escape(data["lang"])}): '
            f'"{md_escape(data["definition"])}" ← '
            f'{link_or_text(G, root_id, here, paths)}: '
            f'"{md_escape(G.nodes[root_id].get("definition", ""))}"'
        )
    write_file(os.path.join(out_dir, here), "\n".join(lines) + "\n")
    return len(flagged)


def generate_site(G, out_dir, max_pages):
    for sub in ("words", "languages"):
        shutil.rmtree(os.path.join(out_dir, sub), ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    page_nodes = select_page_nodes(G, max_pages)
    paths = {n: node_path(n, G.nodes[n]) for n in page_nodes}
    log(f"Writing {len(page_nodes):,} word pages")

    for count, node_id in enumerate(page_nodes, 1):
        write_file(os.path.join(out_dir, paths[node_id]), render_page(G, node_id, paths))
        if count % 100000 == 0:
            log(f"  {count:,} pages written")

    language_count = write_language_hubs(G, out_dir, paths, page_nodes)
    drift_count = write_drift_report(G, out_dir, paths)

    home = [
        "# Wordagz", "",
        "An interconnected etymology wiki generated from Wiktionary data "
        "(via [Kaikki.org](https://kaikki.org)).", "",
        "| Metric | Count |", "| --- | --- |",
        f"| Words in graph | {G.number_of_nodes():,} |",
        f"| Etymology links | {G.number_of_edges():,} |",
        f"| Pages generated | {len(page_nodes):,} |",
        f"| Languages | {language_count:,} |",
        f"| Semantic drift flags | {drift_count:,} |", "",
        "## Start here", "",
        "- [Browse by language](/languages/)",
        "- [Semantic drift report](/drift)", "",
        "---", "",
        "Data: Wiktionary contributors, extracted with Wiktextract; "
        "licensed CC BY-SA 4.0 and GFDL.", "",
    ]
    write_file(os.path.join(out_dir, "README.md"), "\n".join(home))
    write_file(
        os.path.join(out_dir, "_sidebar.md"),
        "- [Home](/)\n- [Languages](/languages/)\n- [Semantic drift report](/drift)\n",
    )
    write_file(os.path.join(out_dir, ".nojekyll"), "")
    return len(page_nodes), drift_count


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build the Wordagz etymology wiki.")
    parser.add_argument("--input", default="kaikki.dump", help="Kaikki JSONL dump (.gz or plain)")
    parser.add_argument("--output", default="docs", help="Output directory")
    parser.add_argument(
        "--max-pages", type=int,
        default=int(os.environ.get("WORDAGZ_MAX_PAGES") or 250000),
        help="Cap on generated word pages, best-connected first (0 = unlimited)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only read the first N lines (testing)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"Input file not found: {args.input}")

    G = build_graph(args.input, args.limit)

    G.remove_nodes_from(list(nx.isolates(G)))
    log(f"Connected graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    removed = enforce_dag(G)
    log(f"Removed {removed:,} cycle-closing edges")
    if not nx.is_directed_acyclic_graph(G):
        sys.exit("Graph is not acyclic after cycle removal.")

    flagged = tag_semantic_drift(G)
    log(f"Tagged {flagged:,} nodes with semantic drift")

    pages, _ = generate_site(G, args.output, args.max_pages)
    log(f"Done: {pages:,} pages in ./{args.output}/")


if __name__ == "__main__":
    main()