#!/usr/bin/env python3
"""
Static undefined-name check for notebooks, in cell-execution order.

    python3 check_undefined_names.py nb1.ipynb [nb2.ipynb ...]

WHY THIS EXISTS
---------------
A notebook cell can compile perfectly and still die at runtime on a NameError, because
`compile()` never resolves names. Patching notebooks by text substitution kept producing
exactly that failure mode, one Colab run at a time:

  * `delta` -- a substitution deleted the assignment but left a later `abs(delta)` reading it.
  * `OVERLAP_POLICY` -- a port brought verify_official_split() across without its constant.
  * `verify_official_split` -- a span replacement bounded by a hardcoded "next function" name
    silently deleted the function sitting between the two anchors in M2 and M3.

Each cost a GPU run to discover. This pass walks the cells in order, accumulates the names each
one binds, and reports any name loaded before anything binds it -- catching all three classes
above without executing a single line.

KNOWN LIMITATION: lambda parameters are not tracked, so `sort(key=lambda t: t[0])` reports `t`
as undefined. Treat a bare single-letter name inside a lambda as a false positive.
"""

import ast, builtins, json, sys

BUILT = set(dir(builtins))

class Scan(ast.NodeVisitor):
    """Collect module-level assigned names and all Name loads (incl. inside functions)."""
    def __init__(self):
        self.assigned = set()
        self.used = []          # (name, lineno)
    def visit_FunctionDef(self, node):
        self.assigned.add(node.name)
        # walk body but treat params/locals as defined
        local = set(a.arg for a in node.args.args)
        local |= set(a.arg for a in node.args.kwonlyargs)
        if node.args.vararg: local.add(node.args.vararg.arg)
        if node.args.kwarg: local.add(node.args.kwarg.arg)
        sub = Scan()
        for st in node.body:
            sub.visit(st)
        for n, ln in sub.used:
            if n not in local and n not in sub.assigned:
                self.used.append((n, ln))
    visit_AsyncFunctionDef = visit_FunctionDef
    def visit_ClassDef(self, node):
        self.assigned.add(node.name)
        self.generic_visit(node)
    def visit_Assign(self, node):
        for t in node.targets:
            for nd in ast.walk(t):
                if isinstance(nd, ast.Name): self.assigned.add(nd.id)
        self.visit(node.value)
    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name): self.assigned.add(node.target.id)
        if node.value: self.visit(node.value)
    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name): self.assigned.add(node.target.id)
        self.visit(node.value)
    def visit_For(self, node):
        for nd in ast.walk(node.target):
            if isinstance(nd, ast.Name): self.assigned.add(nd.id)
        self.generic_visit(node)
    def visit_withitem(self, node):
        if node.optional_vars:
            for nd in ast.walk(node.optional_vars):
                if isinstance(nd, ast.Name): self.assigned.add(nd.id)
    def visit_Import(self, node):
        for a in node.names: self.assigned.add((a.asname or a.name).split(".")[0])
    def visit_ImportFrom(self, node):
        for a in node.names: self.assigned.add(a.asname or a.name)
    def visit_ExceptHandler(self, node):
        if node.name: self.assigned.add(node.name)
        self.generic_visit(node)
    def visit_comprehension(self, node):
        for nd in ast.walk(node.target):
            if isinstance(nd, ast.Name): self.assigned.add(nd.id)
        self.generic_visit(node)
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load): self.used.append((node.id, node.lineno))
        elif isinstance(node.ctx, (ast.Store, ast.Del)): self.assigned.add(node.id)
    def visit_NamedExpr(self, node):
        if isinstance(node.target, ast.Name): self.assigned.add(node.target.id)
        self.visit(node.value)

for path in sys.argv[1:]:
    nb = json.load(open(path))
    defined = set()
    problems = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code": continue
        src = "".join(c["source"])
        try:
            tree = ast.parse(src)
        except SyntaxError as e:
            problems.append((i, "SYNTAX", str(e))); continue
        sc = Scan()
        for st in tree.body: sc.visit(st)
        # comprehension targets inside nested scopes
        for n, ln in sc.used:
            if n not in defined and n not in sc.assigned and n not in BUILT:
                problems.append((i, n, ln))
        defined |= sc.assigned
    name = path.split("/")[-1]
    if problems:
        print(f"### {name}")
        seen = set()
        for i, n, ln in problems:
            if (i, n) in seen: continue
            seen.add((i, n))
            print(f"    cell {i:>2}: undefined name {n!r}")
    else:
        print(f"### {name}: no undefined names")
