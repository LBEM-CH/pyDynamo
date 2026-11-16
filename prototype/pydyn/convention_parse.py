"""Parser for Dynamo's dynamo_vpr_convention.m (used to generate and to verify pydyn/convention.py).

It understands only the statement forms that occur in the 1.1.514 file: 'c=0;', 'c=c+1;' and
'nc{c}.<field> = <literal>;' / 'rc{c}.<field> = <literal>;' with literals '...', numbers, [a,b,...], true/false and
{...} cells of such literals, '...' line continuations and % comments. Anything else raises NotImplementedError.
"""
import re

SOURCE = "/programs/x86_64-linux/dynamo/1.1.514/matlab/src/dynamo_vpr_convention.m"
FIELDS = ("name", "type", "value", "hidden", "short", "pretty", "comment", "advanced", "area", "options")


def _strip_comment(line):
    out, q = [], False
    for i, ch in enumerate(line):
        if ch == "'":
            # a quote opens a string only after an operator/space/bracket (MATLAB transpose is not used here)
            q = not q
        if ch == "%" and not q:
            break
        out.append(ch)
    return "".join(out)


def _tokens(s):
    toks, i = [], 0
    while i < len(s):
        ch = s[i]
        if ch in " \t":
            i += 1
        elif ch == "'":
            j, buf = i + 1, []
            while True:
                if s[j] == "'" and j + 1 < len(s) and s[j + 1] == "'":
                    buf.append("'")
                    j += 2
                elif s[j] == "'":
                    break
                else:
                    buf.append(s[j])
                    j += 1
            toks.append(("str", "".join(buf)))
            i = j + 1
        elif ch in "[]{},;":
            toks.append((ch, ch))
            i += 1
        else:
            m = re.match(r"[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?|true|false", s[i:])
            if not m:
                raise NotImplementedError("cannot parse %r" % s)
            t = m.group(0)
            toks.append(("bool", t == "true") if t in ("true", "false") else ("num", float(t)))
            i += len(t)
    return toks


def _literal(expr):
    toks = [t for t in _tokens(expr.strip().rstrip(";"))]
    if not toks:
        raise NotImplementedError("empty value")
    k, v = toks[0]
    if k in ("str", "num", "bool") and len(toks) == 1:
        return v
    if k == "[":
        vals = [t[1] for t in toks[1:-1] if t[0] == "num"]
        if toks[-1][0] != "]" or any(t[0] not in ("num", ",") for t in toks[1:-1]):
            raise NotImplementedError("vector %r" % expr)
        return tuple(vals)
    if k == "{":
        if toks[-1][0] != "}":
            raise NotImplementedError("cell %r" % expr)
        return tuple(t[1] for t in toks[1:-1] if t[0] in ("str", "num", "bool"))
    raise NotImplementedError("literal %r" % expr)


def parse_convention(path=SOURCE):
    """Return (nc, rc): lists of dicts in file order, as dynamo_vpr_convention.m builds them."""
    with open(path, encoding="latin-1") as f:
        raw = f.read().splitlines()
    lines, buf, start = [], "", None
    for n, ln in enumerate(raw, 1):
        s = _strip_comment(ln).rstrip()
        if start is None:
            start = n
        if re.search(r"\.\.\.+$", s):
            buf += re.sub(r"\.\.\.+$", "", s) + " "
            continue
        lines.append((start, buf + s))
        buf, start = "", None
    out = {"nc": [], "rc": []}
    c = 0
    for n, s in lines:
        s = s.strip()
        if not s:
            continue
        if s.startswith("function ") or s.startswith("functionalities") or s.startswith("'"):
            continue
        if re.fullmatch(r"c\s*=\s*0\s*;", s):
            c = 0
            continue
        if re.fullmatch(r"c\s*=\s*c\s*\+\s*1\s*;", s):
            c += 1
            continue
        if s.startswith("hidden_") or s.startswith("hidden_command_names"):
            break                                   # '%%%%% Gather subset of "hidden" commands'
        m = re.fullmatch(r"(nc|rc)\{c\}\.(\w+)\s*=\s*(.*?);?", s)
        if not m:
            raise NotImplementedError("line %d: %r" % (n, s))
        arr, field, expr = m.groups()
        lst = out[arr]
        while len(lst) < c:
            lst.append({})
        lst[c - 1][field] = _literal(expr)
        if field == "name":
            lst[c - 1]["line"] = n
    res = []
    for arr in ("nc", "rc"):
        items = []
        for d in out[arr]:
            e = {k: d.get(k) for k in FIELDS + ("line",)}
            for k in ("type", "advanced"):
                if isinstance(e[k], float) and e[k] == int(e[k]):
                    e[k] = int(e[k])
            items.append(e)
        res.append(items)
    return res[0], res[1]
