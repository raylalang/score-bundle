#!/usr/bin/env python
"""Checker for the thesis style makeover (docs/thesis/style_guide.md).

Compares a rewritten chapter file against its pre-makeover version and
reports (a) hard-constraint violations, which fail the check: any change
to the multiset of numbers, the set of labels, the multisets of refs and
cites, the figure calls, the equation environments, or the tabular bodies;
and (b) style counts on the new text, which are reported and fail only
where the guide sets a cap: run-in paragraph headers (0), em-dashes (0),
semicolons outside math (0), asterisks outside math (0), bold (<= 5),
emphasis (<= 12), and the banned-vocabulary list (0 outside footnotes).

    python scripts/check_thesis_style.py OLD.tex NEW.tex [--quiet]
    python scripts/check_thesis_style.py --all BASELINE_DIR   # every chapter
"""
from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter

BANNED = [
    r"\bconfirmation (set|pool|split)\b", r"\bthe pool\b", r"\bpool\b",
    r"\bdev\b", r"\bdev[- ]set\b", r"\bdev-only\b",
    r"\bclaims?\b", r"\bregistered\b", r"\bpreregistered\b",
    r"\bpre-registered\b", r"\bfrozen\b", r"\btagged\b", r"\bone-shot\b",
    r"\brun once\b", r"\bspent\b", r"\bnever rerun\b", r"\bverdicts?\b",
    r"\bPASS\b", r"\bFAIL\b", r"\bpassed\b", r"\bfailed\b", r"\bwins?\b",
    r"\blose[sd]?\b", r"\brepaired\b", r"\bgated?\b", r"\bgating\b",
    r"\bnon-gating\b", r"\breport-only\b", r"\bas-given\b", r"\bas given\b",
    r"\bcells?\b", r"\bguard(ed)?\b", r"\bopt-in\b", r"\bconfig\b",
    r"\bstarred\b", r"\bhonest(ly)?\b", r"\brigorous(ly)?\b",
    r"\bverbatim\b", r"\bdeliberately\b", r"\bheadline\b",
    r"(?<!Kronecker )\bdelta\b",
    r"\berratum\b", r"\bwithdrawn\b", r"\bthe committee\b",
    r"\bthe professor\b", r"\bRay\b", r"\b20[0-9]{2}-[0-9]{2}(-[0-9]{2})?\b",
    r"\bquasi\b", r"\bPhase[- ]?[0-3]\b", r"\\texttt\{", r"\.py\b", r"\.sh\b",
]
EQ_ENVS = ("equation", "align", "gather", "multline", "eqnarray",
           "equation*", "align*", "gather*", "multline*", "subequations")


def strip_comments(s: str) -> str:
    return re.sub(r"(?<!\\)%.*", "", s)


def env_bodies(s: str, names) -> Counter:
    out = Counter()
    for n in names:
        pat = re.compile(r"\\begin\{" + re.escape(n) + r"\}(.*?)\\end\{"
                         + re.escape(n) + r"\}", re.S)
        for m in pat.finditer(s):
            out[re.sub(r"\s+", " ", m.group(1)).strip()] += 1
    return out


def numbers(s: str) -> Counter:
    s = re.sub(r"\\(label|ref|eqref|cite|autoref|S\\ref)\{[^}]*\}", "", s)
    s = re.sub(r"\\(begin|end)\{[^}]*\}", "", s)
    s = re.sub(r"\b20[0-9]{2}-[0-9]{2}(-[0-9]{2})?\b", "", s)   # dates allowed to go
    s = re.sub(r"\\figph\{[^}]*\}\{[^}]*\}", "", s)              # figure file/width
    s = re.sub(r"\[[0-9.]+\\(line|text)width\]", "", s)
    s = re.sub(r"\\(vspace|hspace|includegraphics)\*?(\[[^\]]*\])?\{[^}]*\}", "", s)
    s = re.sub(r"\\footnote\{", "", s)
    return Counter(re.findall(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w])", s))


def tokens(s: str, cmd: str) -> Counter:
    return Counter(re.findall(r"\\" + cmd + r"\*?(?:\[[^\]]*\])?\{([^}]*)\}", s))


def math_stripped(s: str) -> str:
    s = re.sub(r"\$\$.*?\$\$", " ", s, flags=re.S)
    s = re.sub(r"\$[^$]*\$", " ", s)
    s = re.sub(r"\\\[.*?\\\]", " ", s, flags=re.S)
    for n in EQ_ENVS:
        s = re.sub(r"\\begin\{" + re.escape(n) + r"\}.*?\\end\{"
                   + re.escape(n) + r"\}", " ", s, flags=re.S)
    s = re.sub(r"\\begin\{(tabular|longtable)\}.*?\\end\{(tabular|longtable)\}",
               " ", s, flags=re.S)
    s = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", " ",
               s, flags=re.S)
    return s


def strip_footnotes(s: str) -> str:
    out, i = [], 0
    while True:
        j = s.find("\\footnote{", i)
        if j < 0:
            out.append(s[i:])
            break
        out.append(s[i:j])
        k, depth = j + len("\\footnote{"), 1
        while k < len(s) and depth:
            depth += {"{": 1, "}": -1}.get(s[k], 0)
            k += 1
        i = k
    return "".join(out)


def check(old_path: str, new_path: str, quiet: bool = False) -> bool:
    old = strip_comments(open(old_path).read())
    new = strip_comments(open(new_path).read())
    ok = True
    name = new_path.split("/")[-1]

    def diff(label, a: Counter, b: Counter, allow_gain: bool = False):
        nonlocal ok
        lost, gained = a - b, b - a
        if lost or (gained and not allow_gain):
            ok = False
            print(f"[{name}] {label}: FAIL")
            for k, v in list(lost.items())[:12]:
                print(f"    lost   x{v}: {k[:100]}")
            for k, v in list(gained.items())[:12]:
                print(f"    gained x{v}: {k[:100]}")
        elif not quiet:
            print(f"[{name}] {label}: ok ({sum(a.values())})")

    diff("numbers", numbers(old), numbers(new))
    diff("labels", tokens(old, "label"), tokens(new, "label"))
    refs_o = tokens(old, "ref") + tokens(old, "eqref") + tokens(old, "autoref")
    refs_n = tokens(new, "ref") + tokens(new, "eqref") + tokens(new, "autoref")
    diff("refs", refs_o, refs_n, allow_gain=True)
    diff("cites", tokens(old, "cite"), tokens(new, "cite"), allow_gain=True)
    if refs_n - refs_o and not quiet:
        print(f"[{name}] refs added: {dict(refs_n - refs_o)}")
    depth = 0
    for ch in re.sub(r"\\[{}]", "", new):
        depth += {"{": 1, "}": -1}.get(ch, 0)
    if depth != 0:
        ok = False
        print(f"[{name}] braces: FAIL (net {depth})")
    diff("figures", Counter(re.findall(r"\\figph\{[^}]*\}\{[^}]*\}", old)),
         Counter(re.findall(r"\\figph\{[^}]*\}\{[^}]*\}", new)))
    diff("equations", env_bodies(old, EQ_ENVS), env_bodies(new, EQ_ENVS))
    diff("tabulars", env_bodies(old, ("tabular", "longtable")),
         env_bodies(new, ("tabular", "longtable")))
    diff("chapters", tokens(old, "chapter"), tokens(new, "chapter"))
    n_sec_o = len(re.findall(r"\\section\{", old))
    n_sec_n = len(re.findall(r"\\section\{", new))
    if n_sec_o != n_sec_n:
        ok = False
        print(f"[{name}] sections: FAIL ({n_sec_o} -> {n_sec_n})")

    prose = re.sub(r"\\(ref|eqref|autoref|label|cite)\{[^}]*\}", "",
                   strip_footnotes(math_stripped(new)))
    caps = {
        "paragraph": (len(re.findall(r"\\paragraph\{", new)), 0),
        "em-dash": (prose.count("---"), 0),
        "semicolon": (prose.count(";"), 0),
        "asterisk": (len(re.findall(r"(?<!\\)\*", prose)), 0),
        "textbf": (len(re.findall(r"\\textbf\{", prose)), 5),
        "emph": (len(re.findall(r"\\emph\{", prose)), 12),
    }
    for k, (n, cap) in caps.items():
        flag = "FAIL" if n > cap else "ok"
        if n > cap:
            ok = False
        if n > cap or not quiet:
            print(f"[{name}] {k}: {n} (cap {cap}) {flag}")
    hits = []
    for pat in BANNED:
        for m in re.finditer(pat, prose):
            line = new.count("\n", 0, new.find(m.group(0))) + 1
            hits.append((m.group(0), line))
    if hits:
        ok = False
        c = Counter(h[0] for h in hits)
        print(f"[{name}] banned vocabulary: {sum(c.values())} FAIL")
        for w, v in c.most_common(25):
            print(f"    {w!r} x{v}")
    elif not quiet:
        print(f"[{name}] banned vocabulary: 0 ok")
    print(f"[{name}] {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    quiet = "--quiet" in sys.argv
    if "--all" in sys.argv:
        import glob
        import os
        base = args[0] if args else None
        if not base or not os.path.isdir(base):
            print("usage: --all BASELINE_DIR (the pre-makeover chapter files)")
            sys.exit(2)
        allok = True
        for new in sorted(glob.glob("docs/thesis/chapters/*.tex")):
            if new.endswith("bibliography.tex"):
                continue
            old = os.path.join(base, os.path.basename(new))
            if not os.path.exists(old):
                print(f"[{os.path.basename(new)}] no baseline; skip")
                continue
            allok &= check(old, new, quiet)
        sys.exit(0 if allok else 1)
    if len(args) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(0 if check(args[0], args[1], quiet) else 1)


if __name__ == "__main__":
    main()
