import os
import re
import hashlib
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# -----------------------------
# Wildcard loading + caching
# -----------------------------

_WILDCARD_FILE_EXTS = {".txt"}

# Supports <color> and <object|person> and <obj/person|thing/stuff>
_TOKEN_RE = re.compile(r"<\s*([A-Za-z0-9_\-./|: \t]+?)\s*>")

@dataclass
class _WildcardCache:
    signature: str
    mapping: Dict[str, List[str]]
    base_dirs: List[str]


_CACHE: Optional[_WildcardCache] = None


def _comfy_root_from_here() -> str:
    """Best-effort detection of the ComfyUI root directory.

    We walk upward from this file looking for a directory that contains
    `custom_nodes/` (which is present at the ComfyUI root).

    This avoids brittle assumptions about how deep this file is nested.
    """
    here = os.path.abspath(os.path.dirname(__file__))

    # First pass: find a directory that contains 'custom_nodes'
    cur = here
    for _ in range(0, 12):
        if os.path.isdir(os.path.join(cur, "custom_nodes")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    # Second pass: if we're inside a 'custom_nodes' subtree, return its parent.
    cur = here
    for _ in range(0, 12):
        if os.path.basename(cur) == "custom_nodes":
            return os.path.dirname(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    # Fallback: current working directory heuristic (least reliable)
    return os.path.abspath(os.getcwd())


def _find_custom_wildcards_dirs(comfy_root: str) -> List[str]:
    """
    Resolution order:
      1) <comfy_root>/custom_wildcards (if exists)
      2) any folder named 'custom_wildcards' under <comfy_root>/custom_nodes (recursively)
    """
    dirs: List[str] = []

    direct = os.path.join(comfy_root, "custom_wildcards")
    if os.path.isdir(direct):
        dirs.append(direct)

    custom_nodes = os.path.join(comfy_root, "custom_nodes")
    if os.path.isdir(custom_nodes):
        for root, subdirs, _files in os.walk(custom_nodes):
            subdirs[:] = [d for d in subdirs if not d.startswith(".") and d not in ("__pycache__",)]
            for d in subdirs:
                if d == "custom_wildcards":
                    dirs.append(os.path.join(root, d))

    # de-dupe, stable order
    seen = set()
    out: List[str] = []
    for d in dirs:
        d2 = os.path.abspath(d)
        if d2 not in seen:
            seen.add(d2)
            out.append(d2)
    return out


def _wildcards_signature(base_dirs: List[str]) -> str:
    """
    Hash (path, mtime_ns, size) for all .txt files under all base dirs.
    """
    h = hashlib.sha256()
    for base in sorted(base_dirs):
        if not os.path.isdir(base):
            continue
        for root, subdirs, files in os.walk(base):
            subdirs[:] = [d for d in subdirs if not d.startswith(".") and d not in ("__pycache__",)]
            for fn in sorted(files):
                ext = os.path.splitext(fn)[1].lower()
                if ext not in _WILDCARD_FILE_EXTS:
                    continue
                path = os.path.join(root, fn)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                rel = os.path.relpath(path, base).replace("\\", "/")
                h.update(base.encode("utf-8", "ignore"))
                h.update(b"\0")
                h.update(rel.encode("utf-8", "ignore"))
                h.update(b"\0")
                h.update(str(st.st_mtime_ns).encode("utf-8"))
                h.update(b"\0")
                h.update(str(st.st_size).encode("utf-8"))
                h.update(b"\0")
    return h.hexdigest()


def _load_wildcards(base_dirs: List[str]) -> Dict[str, List[str]]:
    """
    Builds mapping (all keys are lowercase):
      - full relpath without extension: 'obj/person' -> lines from obj/person.txt
      - basename alias for convenience: 'person' -> lines from obj/person.txt

    If multiple files share the same basename (e.g. a/person.txt and b/person.txt),
    the basename alias list is merged (deduped, stable order).
    """
    mapping: Dict[str, List[str]] = {}

    for base in base_dirs:
        if not os.path.isdir(base):
            continue

        for root, subdirs, files in os.walk(base):
            subdirs[:] = [d for d in subdirs if not d.startswith(".") and d not in ("__pycache__",)]
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in _WILDCARD_FILE_EXTS:
                    continue

                full = os.path.join(root, fn)
                rel = os.path.relpath(full, base).replace("\\", "/")
                key = os.path.splitext(rel)[0].lower()  # full relpath key, drop .txt
                base_key = os.path.splitext(os.path.basename(rel))[0].lower()  # basename alias

                try:
                    with open(full, "r", encoding="utf-8", errors="ignore") as f:
                        lines = [ln.strip() for ln in f.read().splitlines()]
                except OSError:
                    continue

                options = [ln for ln in lines if ln.strip() != ""]
                if not options:
                    continue

                # Full relpath mapping (obj/person)
                mapping[key] = options

                # Basename alias mapping (person) so `<person>` works even if the file is in a subdir.
                # If multiple files share the same basename, merge (dedupe) into the alias list.
                if base_key and base_key != key:
                    if base_key not in mapping:
                        mapping[base_key] = list(options)
                    else:
                        existing = mapping[base_key]
                        for opt in options:
                            if opt not in existing:
                                existing.append(opt)

    return mapping


def _get_cache() -> _WildcardCache:
    global _CACHE
    comfy_root = _comfy_root_from_here()
    base_dirs = _find_custom_wildcards_dirs(comfy_root)
    sig = _wildcards_signature(base_dirs)

    if _CACHE is not None and _CACHE.signature == sig:
        return _CACHE

    mapping = _load_wildcards(base_dirs)
    _CACHE = _WildcardCache(signature=sig, mapping=mapping, base_dirs=base_dirs)
    return _CACHE


def _expand_fragment(
    fragment: str,
    mapping: Dict[str, List[str]],
    rng: random.Random,
    missing_policy: str,
    max_passes: int,
    in_progress: set,
) -> str:
    """Expand a fragment (chosen wildcard line) to a final string.

    Uses local bindings so anchors inside the wildcard line do NOT leak globally,
    while still being consistent within that fragment.
    """
    local_bindings: Dict[Tuple[str, str], str] = {}
    text = fragment
    passes = max(1, int(max_passes))
    for _ in range(passes):
        text2, changed = _expand_once(
            text,
            mapping,
            rng,
            missing_policy,
            local_bindings,
            max_passes,
            in_progress,
        )
        text = text2
        if not changed:
            break
    return text


def _split_token_and_var(raw: str) -> Tuple[str, Optional[str]]:
    """Split '<...>' inner text into (keys_part, var_id).

    Supports numeric var ids like:
      <color:1>
      <object|person:2>

    Only treats the suffix as a var if it is all digits.
    """
    raw = (raw or "").strip()
    if ":" not in raw:
        return raw, None
    left, right = raw.rsplit(":", 1)
    right = right.strip()
    if right.isdigit():
        return left.strip(), right
    return raw, None


def _parse_token_keys(raw: str) -> List[str]:
    """
    '<object|person>' -> ['object', 'person'] (lowercased, stripped)
    """
    parts = [p.strip() for p in raw.split("|")]
    keys = [p.lower() for p in parts if p != ""]
    # de-dupe while preserving order
    seen = set()
    out: List[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _expand_once(
    text: str,
    mapping: Dict[str, List[str]],
    rng: random.Random,
    missing_policy: str,
    bindings: Dict[Tuple[str, str], str],
    max_passes: int,
    in_progress: set,
) -> Tuple[str, bool]:
    """
    Returns (new_text, changed).
    missing_policy: 'keep' | 'empty' | 'error'
    """
    changed = False

    def repl(m: re.Match) -> str:
        nonlocal changed
        raw = m.group(1)
        keys_part, var_id = _split_token_and_var(raw)
        keys = _parse_token_keys(keys_part)

        # Filter to keys that exist
        existing = [k for k in keys if k in mapping]

        # Variable binding: if var_id is set, reuse the same (chosen_key, line)
        # for the same token-group across the whole expansion run.
        group_id = "|".join(keys) if keys else ""
        bind_key = (group_id, var_id) if var_id else None

        if bind_key is not None and bind_key in bindings:
            changed = True
            return bindings[bind_key]

        if not existing:
            if missing_policy == "empty":
                changed = True
                return ""
            if missing_policy == "error":
                looked_for = ", ".join([f"'{k}.txt'" for k in keys]) if keys else "(empty token)"
                raise ValueError(f"Wildcard <{raw}> not found (looked for {looked_for} under custom_wildcards)")
            return m.group(0)  # keep

        # Choose which wildcard file to use, then choose a line from it
        chosen_key = rng.choice(existing)
        options = mapping[chosen_key]
        chosen_line = rng.choice(options)
        changed = True

        # If bound, expand the chosen line immediately and cache the final expanded string.
        if bind_key is not None:
            if bind_key in in_progress:
                # Prevent infinite recursion (e.g., wildcard line references itself).
                bindings[bind_key] = chosen_line
                return chosen_line

            in_progress.add(bind_key)
            try:
                expanded = _expand_fragment(
                    chosen_line,
                    mapping,
                    rng,
                    missing_policy,
                    max_passes,
                    in_progress,
                )
            finally:
                in_progress.discard(bind_key)

            bindings[bind_key] = expanded
            return expanded

        return chosen_line

    out = _TOKEN_RE.sub(repl, text)
    return out, changed


def wildcard_expand(
    template: str,
    rng: random.Random,
    max_passes: int,
    missing_policy: str,
) -> str:
    cache = _get_cache()
    text = template
    bindings: Dict[Tuple[str, str], str] = {}
    in_progress: set = set()

    passes = max(1, int(max_passes))
    for _ in range(passes):
        text2, changed = _expand_once(text, cache.mapping, rng, missing_policy, bindings, passes, in_progress)
        text = text2
        if not changed:
            break

    return text


# -----------------------------
# ComfyUI node
# -----------------------------

class WildcardGold:
    """
    Replaces <name> tokens using custom_wildcards/name.txt (recursive).
    Also supports <a|b|c> to choose one wildcard file among several, and <name:1> to bind a variable.
    """

    CATEGORY = "text/Wildcard Gold"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "compute"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "template": ("STRING", {"multiline": True, "default": ""}),
                "seed_mode": (["fixed", "randomize"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "max_passes": ("INT", {"default": 3, "min": 1, "max": 50}),
                "missing_policy": (["keep", "empty", "error"],),
            }
        }

    @classmethod
    def IS_CHANGED(cls, template, seed_mode, seed, max_passes, missing_policy):
        cache = _get_cache()
        if seed_mode == "randomize":
            return float("NaN")
        return (
            template,
            seed_mode,
            int(seed) & 0xFFFFFFFFFFFFFFFF,
            int(max_passes),
            str(missing_policy),
            cache.signature,
        )

    def compute(self, template, seed_mode, seed, max_passes, missing_policy):
        if seed_mode == "randomize":
            used_seed = random.SystemRandom().randint(0, 0xFFFFFFFFFFFFFFFF)
        else:
            used_seed = int(seed) & 0xFFFFFFFFFFFFFFFF

        rng = random.Random(used_seed)
        out = wildcard_expand(
            template=template,
            rng=rng,
            max_passes=max_passes,
            missing_policy=missing_policy,
        )
        return (out,)