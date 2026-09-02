#!/usr/bin/env python3
"""Pre-submission checks for main.tex. No LaTeX installation needed.

    python verify.py

Checks, in order:
  1. every \\ref resolves to a \\label, and every table/figure/equation label is cited
  2. every \\cite key exists in references.bib, and every .bib entry is cited
  3. environments and braces balance; no stray control characters
  4. every tabular row has exactly as many cells as its column spec (multicolumn-aware)
  5. course-format rules: abstract 200-300 words with no citations/abbreviations/
     symbols/equations, keywords alphabetical
  6. a rough page estimate for the 18-page limit

It cannot replace a real Overleaf compile; run one before submitting.
"""
import re, io, os, struct, sys

TEX, BIB = 'main.tex', 'references.bib'
fail = []


def note(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        fail.append(msg)


s = io.open(TEX, encoding='utf-8').read()
bib = io.open(BIB, encoding='utf-8').read()

print("1. cross-references")
labels = set(re.findall(r'\\label\{([^}]*)\}', s))
refs = set(re.findall(r'\\ref\{([^}]*)\}', s))
note(not (refs - labels), "no dangling \\ref  %s" % sorted(refs - labels))
orphan = sorted(l for l in labels - refs if l.startswith(('tab:', 'fig:', 'eq:')))
note(not orphan, "every float/equation is referenced  %s" % orphan)

print("2. bibliography")
cites = {k.strip() for m in re.findall(r'\\cite\{([^}]*)\}', s) for k in m.split(',')}
keys = set(re.findall(r'@\w+\{([^,\s]+)\s*,', bib))
note(not (cites - keys), "no undefined \\cite  %s" % sorted(cites - keys))
note(not (keys - cites), "no uncited .bib entry  %s" % sorted(keys - cites))

print("3. structure")
for env in ('table', 'tabular', 'figure', 'equation', 'document'):
    a = len(re.findall(r'\\begin\{' + env + r'\}', s))
    b = len(re.findall(r'\\end\{' + env + r'\}', s))
    note(a == b, "%s balanced (%d/%d)" % (env, a, b))
note(s.count('{') == s.count('}'), "braces balanced (%d/%d)" % (s.count('{'), s.count('}')))
ctrl = {hex(ord(c)) for c in s if ord(c) < 32 and c not in '\n\r\t'}
note(not ctrl, "no stray control characters  %s" % sorted(ctrl))
note(all(ord(c) < 128 for c in s), "ASCII only")


def bmatch(t, i):
    d = 0
    while i < len(t):
        if t[i] == '\\':
            i += 2
            continue
        if t[i] == '{':
            d += 1
        elif t[i] == '}':
            d -= 1
            if d == 0:
                return i + 1
        i += 1
    return -1


def spec_cols(spec):
    n, i = 0, 0
    while i < len(spec):
        c = spec[i]
        if c in '@!>' and i + 1 < len(spec) and spec[i + 1] == '{':
            i = bmatch(spec, i + 1)
            continue
        if c in 'lcr':
            n += 1
        elif c in 'pmb' and i + 1 < len(spec) and spec[i + 1] == '{':
            n += 1
            i = bmatch(spec, i + 1)
            continue
        i += 1
    return n


print("4. table column counts")
lines, bad, i = s.split('\n'), 0, 0
while i < len(lines):
    m = re.search(r'\\begin\{tabular\}\s*(\{)', lines[i])
    if m:
        rest = lines[i][m.start(1):]
        ncol = spec_cols(rest[1:bmatch(rest, 0) - 1])
        j, buf = i + 1, ''
        while j < len(lines) and '\\end{tabular}' not in lines[j]:
            if not lines[j].strip().startswith('%'):
                buf += ' ' + lines[j].strip()
            while '\\\\' in buf:
                k = buf.index('\\\\')
                row, buf = buf[:k], buf[k + 2:]
                r = re.sub(r'\\(top|mid|bottom)rule|\\addlinespace', '', row).strip()
                if r:
                    n = sum(int(x.group(1)) if x else 1 for x in
                            (re.search(r'\\multicolumn\s*\{\s*(\d+)\s*\}', c)
                             for c in re.split(r'(?<!\\)&', re.sub(r'\\%', '', r))))
                    if n != ncol:
                        print("    line ~%d: spec=%d row=%d | %s" % (j + 1, ncol, n, r[:70]))
                        bad += 1
            j += 1
        i = j
    i += 1
note(bad == 0, "all tabular rows match their column spec (%d mismatches)" % bad)

print("5. course-format rules")
ab = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', s, re.S).group(1)
w = len(ab.split())
note(200 <= w <= 300, "abstract is %d words (200-300 required)" % w)
note('\\cite' not in ab, "abstract has no citations")
note('$' not in ab and '\\(' not in ab, "abstract has no math")
note('%' not in ab, "abstract has no percent sign")
caps = re.findall(r'\b[A-Z]{2,}\b', ab)
note(not caps, "abstract has no abbreviations  %s" % caps)
kw = re.search(r'\\begin\{IEEEkeywords\}(.*?)\\end\{IEEEkeywords\}', s, re.S).group(1)
items = [k.strip().rstrip('.').strip().lower()
         for k in kw.replace('\n', ' ').split(',') if k.strip()]
note(items == sorted(items), "keywords alphabetical (%d)" % len(items))
res = s[s.index('\\section{Results'):s.index('\\section{Conclusions')]
labs = re.findall(r'\\label\{tab:[^}]*\}', res)
note(labs[-1] == '\\label{tab:comparison}',
     "comparison table is last in Results and Discussion")


print("6. page estimate (approximate -- confirm with a real compile)")
TW, BL, WPL, LPP = 7.167, 11.5, 17.5, 58.0


def wc(t):
    t = re.sub(r'\\(cite|ref|label|texttt|emph|best|textbf)\{([^{}]*)\}', r'\2', t)
    return len(re.sub(r'[{}$~\\&_^]', ' ', re.sub(r'\\[a-zA-Z]+\*?', ' ', t)).split())


def aspect(p):
    p = os.path.join('figures', p)
    if p.endswith('.png'):
        with open(p, 'rb') as f:
            d = f.read(33)
        a, b = struct.unpack('>II', d[16:24])
        return b / a
    v = [float(x) for x in re.findall(rb'/MediaBox\s*\[([^\]]*)\]', open(p, 'rb').read())[0].split()]
    return (v[3] - v[1]) / (v[2] - v[0])


body = s[s.index('\\begin{document}'):]
tot = 16 + len(re.findall(r'@\w+\{', bib)) * 2.9
for t in re.findall(r'\\begin\{table\}(.*?)\\end\{table\}', body, re.S):
    rows = sum(b.count('\\\\') for b in re.findall(r'\\begin\{tabular\}.*?\\end\{tabular\}', t, re.S))
    cap = re.search(r'\\caption\{(.*?)\}\s*\n\s*\\label', t, re.S)
    foot = re.search(r'\\vspace\{2pt\}\s*\n\\footnotesize(.*)$', t, re.S)
    tot += (rows * 0.83 + len(re.findall(r'\\(top|mid|bottom)rule', t)) * 0.45
            + wc(cap.group(1) if cap else '') / 22 + wc(foot.group(1) if foot else '') / 26 + 2.5)
for f in re.findall(r'\\begin\{figure\}(.*?)\\end\{figure\}', body, re.S):
    m = re.search(r'width=([0-9.]+)\\linewidth\]\{figures/([^}]*)\}', f)
    cap = re.search(r'\\caption\{(.*?)\}\s*\n\s*\\label', f, re.S)
    tot += (TW * float(m.group(1)) * aspect(m.group(2)) * 72 / BL
            + wc(cap.group(1) if cap else '') / 22 + 2.5)
pr = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '',
            re.sub(r'\\begin\{table\}.*?\\end\{table\}', '', body, flags=re.S), flags=re.S)
tot += len(re.findall(r'\\begin\{equation\}', pr)) * 4.0
tot += wc(re.sub(r'\\begin\{equation\}.*?\\end\{equation\}', '', pr, flags=re.S)) / WPL
est = tot / LPP
print("  estimated %.1f pages (limit 18, minimum 6)" % est)
note(6 <= est <= 18, "page estimate within limits")

print()
print("FAILURES: %d" % len(fail))
sys.exit(1 if fail else 0)
