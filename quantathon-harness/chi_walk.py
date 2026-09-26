"""
chi_walk.py -- physics-informed runtime cost for the Quantum Rings challenge.

Two steps, split so parameter tuning is fast:

    w = walk(qasm_text, classical_tracking=True)     # slow: reads the circuit once
    f = price(w, c1=..., c2=..., p=...)              # instant: turns the walk into costs

walk()
    Streams through the circuit and tracks an upper bound on log2(chi) (MPS bond
    dimension) on every link between neighbouring qubits, once per threshold
    (16, 64, 512) plus an "uncapped" run (chi up to 2^40).
      * a two-qubit gate can grow every link it crosses by w bits
        (w = 1 for controlled / Pauli-Pauli gates, 2 for swap-like / unknown gates),
        capped at min(log2(threshold), qubits on the smaller side of the link)
      * single-qubit gates never change chi
      * classical_tracking: every qubit starts as a classical bit |0>. X, Z, S, T,
        RZ, P... keep it classical; H and SX make it quantum. RX, RY, U, and U3
        do so only when their first numeric angle is more than 0.3 radians from
        an integer multiple of pi. A controlled gate whose controls are all classical, or a
        diagonal gate (CZ, CP, RZZ...) with fewer than two quantum qubits, cannot
        entangle, so it does NOT grow chi (it still costs time; see price()).
    It does not compute a cost. It records, per threshold, how many gates ran at
    each chi level:
        t1[k] = number of single-qubit gates executed where the local chi = 2^k
        t2[k] = total distance of two-qubit gates executed where the max chi crossed = 2^k
    The result is a plain dict (JSON-serialisable), so cache it to disk.

price(w, c1, c2, p)
    cost = sum_k t1[k] * (c1 + chi^2)  +  sum_k t2[k] * (c2 + chi^p),   chi = 2^k
      c1 : fixed overhead per single-qubit gate
      c2 : fixed overhead per two-qubit gate (per link crossed)
      p  : scaling exponent for two-qubit gates (textbook 3; try 2..3)
    Returns log10(cost) per threshold. No overall scale factor: a per-threshold
    line fit (or the main model) absorbs it.
    price(w, 0, 0, 3) with classical_tracking=False reproduces chi_cost.py exactly.

features(w, c1, c2, p) flattens everything into one dict of numbers for the model.

Tuning sketch:
    walks = {fname: walk(read(fname)) for fname in circuits}      # once, cache it
    for p in (2.0, 2.25, 2.5, 2.75, 3.0):
        for c1 in (0, 1, 10, 100):
            for c2 in (0, 1, 10, 100, 1000):
                cost = {f: price(w, c1, c2, p) for f, w in walks.items()}
                ... fit log10(runtime) ~ a + b * cost per threshold, CV by circuit, score
Speed: <= ~10 s per circuit (budget_s); past the budget the rest of the circuit is
extrapolated from the fraction processed (reported as `extrapolated`).
"""
import bisect
import math
import re
import time

THRESHOLDS = (16, 64, 512)
UNCAPPED_BITS = 40

W1 = {"cx", "cnot", "cz", "cy", "ch", "cp", "cphase", "cu1", "cu3", "cu", "crx",
      "cry", "crz", "cs", "csdg", "csx", "rzz", "rxx", "ryy", "rzx", "ecr",
      "ccx", "ccz", "rccx", "rcccx", "c3x", "c3sx", "c4x", "mcx", "mcphase", "cswap"}
W2 = {"swap", "iswap", "dcx", "xx_plus_yy", "xx_minus_yy", "fsim"}
SKIP = {"measure", "reset", "barrier", "creg", "bit", "include", "openqasm", "opaque",
        "input", "output", "const", "float", "int", "uint", "angle", "bool", "delay",
        "if", "else", "for", "while", "qreg", "qubit", "gate", "def", "return", "let"}

# classical tracking: which gates keep a computational-basis state a basis state
KEEP_1Q = {"x", "y", "z", "s", "sdg", "t", "tdg", "p", "u1", "rz", "id", "i", "phase"}
DIAG = {"cz", "cp", "cphase", "cu1", "crz", "rzz", "ccz", "cs", "csdg", "mcphase"}
CTRL_PERM = {"cx", "cnot", "cy", "ccx", "mcx", "c3x", "c4x", "rccx", "rcccx"}   # last qubit = target
CTRL_OTHER = {"ch", "crx", "cry", "cu", "cu3", "csx", "c3sx"}                 # last qubit = target
LEVELS = 41                                                               # log2 chi = 0..40

_comment = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_modifier = re.compile(r"^(?:(?:ctrl|negctrl|inv|pow)(?:\([^)]*\))?\s*@\s*)+")
_qreg = re.compile(r"qreg\s+(\w+)\s*\[\s*(\d+)\s*\]$")
_qubit = re.compile(r"qubit\s*(?:\[\s*(\d+)\s*\])?\s*(\w+)$")
_params = re.compile(r"\((?:[^()]|\([^()]*\))*\)")
_number = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$", re.I)
_P4 = [4.0 ** k for k in range(UNCAPPED_BITS + 1)]
_P8 = [8.0 ** k for k in range(UNCAPPED_BITS + 1)]
BASIS_ROTATION_TOLERANCE = 0.3  # radians from an integer multiple of pi


class _OverBudget(Exception):
    pass


def _weight(name):
    if name in W1:
        return 1
    if name in W2:
        return 2
    return 1 if name.startswith(("c", "mc")) else 2


def _split_call(s):
    """Return (gate, operands, parameters), or None for non-gate text."""
    if "@" in s:
        s = _modifier.sub("", s)
    tok = s.split(None, 1)
    if "(" in tok[0]:
        i = s.find("(")
        close = s.rfind(")")
        return s[:i].strip().lower(), s[close + 1:], s[i + 1:close]
    if len(tok) < 2:
        return None
    return tok[0].lower(), tok[1], None


def _rotation_angle(name, parameters):
    """First angle in radians for X/Y-like single-qubit rotations, if numeric."""
    if name not in ("rx", "ry", "u", "u3") or parameters is None:
        return None
    expression = parameters.split(",", 1)[0].replace(" ", "").lower()
    try:
        if _number.fullmatch(expression):
            value = float(expression)
        elif expression in ("pi", "+pi", "-pi"):
            value = -math.pi if expression[0] == "-" else math.pi
        elif expression.startswith(("pi*", "+pi*", "-pi*")):
            sign = -1 if expression[0] == "-" else 1
            value = sign * math.pi * float(expression.split("*", 1)[1])
        elif expression.endswith("*pi"):
            value = float(expression[:-3]) * math.pi
        elif expression.startswith(("pi/", "-pi/")):
            sign = -1 if expression[0] == "-" else 1
            value = sign * math.pi / float(expression.split("/", 1)[1])
        else:
            return None
    except (ValueError, OverflowError, ZeroDivisionError):
        return None
    return value if math.isfinite(value) else None


def _near_basis_rotation(angle, tolerance):
    """RX/RY/U with theta near k*pi maps each basis state near another basis state."""
    return angle is not None and abs(math.remainder(angle, math.pi)) <= tolerance


class _Run:
    """chi tracking for one cap. Links only grow, so a link at its cap is frozen:
    `nxt` (union-find) skips frozen links and `top` lists links frozen at the global
    cap `lt`, so most range maxima are a single bisect."""
    __slots__ = ("lt", "cap", "x", "nxt", "top", "edge")

    def __init__(self, n, lt):
        L = max(n - 1, 1)
        self.lt = lt
        self.cap = [min(c + 1, n - 1 - c, lt) for c in range(L)] if n > 1 else [0]
        self.x = [0] * L
        self.nxt = list(range(L + 1))
        self.top = []
        self.edge = [c for c in range(L) if self.cap[c] < lt]
        for c in range(L):
            if self.cap[c] == 0:
                self._freeze(c)

    def _find(self, c):
        nxt = self.nxt
        root = c
        while nxt[root] != root:
            root = nxt[root]
        while nxt[c] != root:
            nxt[c], c = root, nxt[c]
        return root

    def _freeze(self, c):
        self.nxt[c] = c + 1
        if self.cap[c] == self.lt:
            bisect.insort(self.top, c)

    def span(self, a, b, w):
        """Grow links a..b-1 by w bits; return the largest log2(chi) among them."""
        x, cap, mx = self.x, self.cap, 0
        c = self._find(a)
        while c < b:
            v = x[c] + w
            if v >= cap[c]:
                v = cap[c]
                self._freeze(c)
            x[c] = v
            if v > mx:
                mx = v
            c = self._find(c + 1)
        top = self.top
        if top:
            i = bisect.bisect_left(top, a)
            if i < len(top) and top[i] < b:
                return self.lt
        for c in self.edge:
            if a <= c < b and x[c] > mx:
                mx = x[c]
        return mx


class _Walker:
    def __init__(self, n, classical_tracking, rotation_tolerance):
        self.n = n
        self.runs = [_Run(n, int(math.log2(t))) for t in THRESHOLDS] + [_Run(n, UNCAPPED_BITS)]
        R = len(self.runs)
        self.t1 = [[0] * LEVELS for _ in range(R)]
        self.t2 = [[0] * LEVELS for _ in range(R)]
        self.sat = [0] * R
        self.first_sat = [None] * R
        self.track = classical_tracking
        self.rotation_tolerance = rotation_tolerance
        self.classical = [True] * n
        self.n1 = self.n2 = self.n2_ent = self.dist = self.gi = 0
        self.n_rotation_parsed = self.n_rotation_near_zero = self.n_rotation_near_pi = 0
        self.rotation_mixing_sum = 0.0

    def _grow_bits(self, name, qs, w):
        """Classical tracking: return how many bits this gate may grow chi (0 = cannot
        entangle) and update which qubits are still classical."""
        cl = self.classical
        if name in DIAG:                              # phases only; flags unchanged
            return w if sum(not cl[q] for q in qs) >= 2 else 0
        if name in CTRL_PERM or name in CTRL_OTHER:
            tgt = qs[-1]
            if all(cl[q] for q in qs[:-1]):           # controls are definite bits
                if name in CTRL_OTHER:
                    cl[tgt] = False                   # may apply H / RY / ... to target
                return 0
            for q in qs:
                cl[q] = False
            return w
        if name == "swap":
            a, b = qs[0], qs[1]
            if cl[a] and cl[b]:
                return 0
            cl[a], cl[b] = cl[b], cl[a]
            return w
        if name == "cswap":
            c, t1, t2 = qs[0], qs[1], qs[2]
            if cl[c]:
                if cl[t1] != cl[t2]:
                    cl[t1] = cl[t2] = False
                return 0
            for q in qs:
                cl[q] = False
            return w
        if name == "dcx" and all(cl[q] for q in qs):
            return 0
        for q in qs:                                  # anything else: assume it entangles
            cl[q] = False
        return w

    def gate(self, name, qs, angle=None):
        self.gi += 1
        if len(qs) == 1:
            self.n1 += 1
            q = qs[0]
            near_basis = _near_basis_rotation(angle, self.rotation_tolerance)
            if angle is not None:
                self.n_rotation_parsed += 1
                self.rotation_mixing_sum += abs(math.sin(angle))
                if near_basis:
                    multiple = round(angle / math.pi)
                    if multiple % 2:
                        self.n_rotation_near_pi += 1
                    else:
                        self.n_rotation_near_zero += 1
            if (self.track and self.classical[q] and name not in KEEP_1Q
                    and not near_basis):
                self.classical[q] = False
            L = len(self.runs[0].x)
            for r, run in enumerate(self.runs):
                x = run.x
                lo = x[q - 1] if q > 0 else 0
                hi = x[q] if q < L else 0
                self.t1[r][lo if lo > hi else hi] += 1
            return
        a, b = min(qs), max(qs)
        if a == b:
            return
        self.n2 += 1
        d = b - a
        self.dist += d
        w = _weight(name)
        if self.track:
            w = self._grow_bits(name, qs, w)
        if w:
            self.n2_ent += 1
        for r, run in enumerate(self.runs):
            mx = run.span(a, b, w) if w else max(run.x[a:b])
            self.t2[r][mx] += d
            if mx == run.lt:
                self.sat[r] += 1
                if self.first_sat[r] is None:
                    self.first_sat[r] = self.gi

    def result(self):
        tot = max(self.gi, 1)
        out = {"n_qubits": self.n, "n_1q": self.n1, "n_2q": self.n2,
               "n_2q_entangling": self.n2_ent,
               "n_rotation_parsed": self.n_rotation_parsed,
               "n_rotation_near_zero": self.n_rotation_near_zero,
               "n_rotation_near_pi": self.n_rotation_near_pi,
               "rotation_mixing_mean": self.rotation_mixing_sum / max(1,self.n_rotation_parsed),
               "mean_2q_dist": self.dist / self.n2 if self.n2 else 0.0,
               "tables": {}}
        for r, run in enumerate(self.runs):
            key = str(THRESHOLDS[r]) if r < len(THRESHOLDS) else "uncapped"
            fs = self.first_sat[r]
            out["tables"][key] = {
                "t1": self.t1[r], "t2": self.t2[r],
                "max_logchi": max(run.x),
                "sat_2q_gates": self.sat[r],
                "sat_start_frac": fs / tot if fs is not None else 1.0,
                "frac_links_at_cap": sum(v == run.lt for v in run.x) / len(run.x),
            }
        return out


def walk(qasm, classical_tracking=True, budget_s=10.0,
         rotation_tolerance=BASIS_ROTATION_TOLERANCE):
    """Walk one circuit and return chi-level tables per threshold.

    Near-0/pi single-qubit X/Y rotations preserve a basis-state flag when their
    angle is within ``rotation_tolerance`` radians of an integer multiple of pi.
    Unparsed or symbolic angles are treated conservatively as mixing rotations.
    """
    t_start = time.perf_counter()
    if "//" in qasm or "/*" in qasm:
        qasm = _comment.sub("", qasm)

    regs, n = {}, 0
    defs = {}            # gate name -> list of (subgate, [formal indices], angle)
    argcache = {}        # "q[3]" -> [3]
    walker = None
    pending = []

    def emit(name, qs, parameters=None):
        body = defs.get(name)
        if walker is None:
            if body is None:
                pending.append((name, qs, _rotation_angle(name, parameters)))
            else:
                pending.extend((sub, [qs[i] for i in idx], angle)
                               for sub, idx, angle in body)
            return
        g = walker.gate
        if body is None:
            g(name, qs, _rotation_angle(name, parameters))
        else:
            for k, (sub, idx, angle) in enumerate(body):
                if k & 4095 == 4095 and time.perf_counter() - t_start > budget_s:
                    raise _OverBudget
                g(sub, [qs[i] for i in idx], angle)

    def resolve(a):
        r = argcache.get(a)
        if r is None:
            reg, _, rest = a.strip().partition("[")
            reg = reg.strip()
            if reg not in regs:
                return None
            off, size = regs[reg]
            r = [off + int(rest.rstrip("] \t\n"))] if rest else list(range(off, off + size))
            argcache[a] = r
        return r

    shape_cache = {}     # cache identical gate bodies, preserving rotation angles
    parts = re.split(r"([{}])", qasm)
    in_gate = None                       # (name, formal->index, body) while in a gate body
    offset, first_gate_at, frac_done = 0, None, 1.0
    total_len = len(qasm)
    stop = False
    for pi in range(0, len(parts), 2):
        if stop:
            break
        chunk = parts[pi]
        chunk_start = offset
        offset += len(chunk) + (1 if pi + 1 < len(parts) else 0)
        delim = parts[pi + 1] if pi + 1 < len(parts) else ""
        stmts = chunk.split(";")
        header = stmts.pop().strip() if delim == "{" else None   # text right before '{'

        if in_gate is not None:
            formals, body = in_gate[1], in_gate[2]
            key = (in_gate[3], chunk) if delim == "}" else None
            cached = shape_cache.get(key) if key is not None else None
            if cached is not None:
                body.extend(cached)
            else:
                for s in stmts:
                    s = s.strip()
                    if not s:
                        continue
                    sc = _split_call(s)
                    if sc is None or sc[0] in SKIP:
                        continue
                    idx = [formals.get(a.strip()) for a in sc[1].split(",")]
                    if not idx or None in idx:
                        continue
                    inner = defs.get(sc[0])
                    if inner is not None:                 # inline nested definitions now
                        for sub, sidx, angle in inner:
                            body.append((sub, [idx[i] for i in sidx], angle))
                    else:
                        body.append((sc[0], idx, _rotation_angle(sc[0], sc[2])))
                if key is not None:
                    shape_cache[key] = body
            if delim == "}":
                defs[in_gate[0]] = body
                in_gate = None
            continue

        pos = chunk_start
        for k, s in enumerate(stmts):
            pos += len(s) + 1
            if k % 4096 == 0 and walker is not None and time.perf_counter() - t_start > budget_s:
                frac_done = (pos - first_gate_at) / max(total_len - first_gate_at, 1)
                stop = True
                break
            s = s.strip()
            if not s:
                continue
            head = s[:6].lower()
            if head.startswith(("qreg", "qubit")):
                m = _qreg.match(s)
                if m:
                    regs[m.group(1)] = (n, int(m.group(2))); n += int(m.group(2))
                else:
                    m = _qubit.match(s)
                    if m:
                        size = int(m.group(1) or 1)
                        regs[m.group(2)] = (n, size); n += size
                continue
            if "measure" in s:
                continue
            sc = _split_call(s)
            if sc is None or sc[0] in SKIP:
                continue
            if first_gate_at is None:
                first_gate_at = pos
            if walker is None and n > 0:
                walker = _Walker(n, classical_tracking, rotation_tolerance)
                for g in pending:
                    walker.gate(*g)
                pending = []
            resolved = []
            for a in sc[1].split(","):
                r = resolve(a)
                if r is None:
                    resolved = None
                    break
                resolved.append(r)
            if not resolved:
                continue
            try:
                if len(resolved) == 1 and len(resolved[0]) == 1:
                    emit(sc[0], resolved[0], sc[2])
                elif all(len(r) == 1 for r in resolved):
                    emit(sc[0], [r[0] for r in resolved], sc[2])
                else:                                    # register broadcast, e.g. "h q;"
                    for i in range(max(len(r) for r in resolved)):
                        emit(sc[0], [r[i] if len(r) > 1 else r[0] for r in resolved], sc[2])
            except _OverBudget:
                frac_done = (pos - first_gate_at) / max(total_len - first_gate_at, 1)
                stop = True
                break

        if header is not None and header.startswith("gate"):
            m = re.match(r"gate\s+(\w+)\s*(?:\([^)]*\))?\s*(.*)", header, re.S)
            formals = [f.strip() for f in m.group(2).split(",") if f.strip()]
            in_gate = (m.group(1).lower(), {f: i for i, f in enumerate(formals)}, [], m.group(2))
        # other '{' blocks (if / for / else) and '}' just fall through: their gates count

    if walker is None:
        walker = _Walker(max(n, 1), classical_tracking, rotation_tolerance)
        for g in pending:
            walker.gate(*g)
    res = walker.result()
    res["classical_tracking"] = bool(classical_tracking)
    res["extrapolated"] = 0.0
    if stop and 0 < frac_done < 1:                   # scale the unwalked remainder
        s = 1.0 / frac_done
        res["extrapolated"] = 1.0 - frac_done
        for k in ("n_1q", "n_2q", "n_2q_entangling", "n_rotation_parsed",
                  "n_rotation_near_zero", "n_rotation_near_pi"):
            res[k] *= s
        for tab in res["tables"].values():
            tab["t1"] = [v * s for v in tab["t1"]]
            tab["t2"] = [v * s for v in tab["t2"]]
            tab["sat_2q_gates"] *= s
    return res


def price(w, c1=0.0, c2=0.0, p=3.0):
    """log10(cost) per threshold from a walk() result. See module docstring."""
    out = {}
    for key, tab in w["tables"].items():
        cost = 0.0
        for k in range(LEVELS):
            if tab["t1"][k]:
                cost += tab["t1"][k] * (c1 + 4.0 ** k)
            if tab["t2"][k]:
                cost += tab["t2"][k] * (c2 + 2.0 ** (p * k))
        out[key] = math.log10(cost + 1)
    return out


def features(w, c1=0.0, c2=0.0, p=3.0):
    """Flat dict of numbers for the model: counts, chi stats, and priced costs."""
    f = {k: w[k] for k in ("n_qubits", "n_1q", "n_2q", "n_2q_entangling",
                           "mean_2q_dist", "extrapolated")}
    for key in ("n_rotation_parsed", "n_rotation_near_zero", "n_rotation_near_pi",
                "rotation_mixing_mean"):
        f[key] = w.get(key,0)
    for key, v in price(w, c1, c2, p).items():
        f[f"log10_cost_{key}"] = v
    for key, tab in w["tables"].items():
        f[f"max_logchi_{key}"] = tab["max_logchi"]
        if key != "uncapped":
            f[f"sat_2q_gates_{key}"] = tab["sat_2q_gates"]
            f[f"sat_start_frac_{key}"] = tab["sat_start_frac"]
            f[f"frac_links_at_cap_{key}"] = tab["frac_links_at_cap"]
    return f


MODEL_FEATURE_NAMES = (
    "chi_walk_cost_overhead", "chi_walk_entangling_frac", "chi_walk_max_logchi",
    "chi_walk_sat_frac", "chi_walk_sat_start", "chi_walk_links_at_cap",
    "chi_walk_extrapolated", "chi_walk_available",
    "chi_walk_rot_near_zero", "chi_walk_rot_near_pi",
    "chi_walk_rot_near_frac", "chi_walk_rot_mixing_mean",
)


def model_features(w):
    """Cacheable features for the chosen model, with threshold-specific fields.

    ``None`` represents the existing fast path for a circuit too large to walk.
    The pricing is the cost-overhead variant selected by the grouped ablation.
    """
    out = {"chi_walk_available": float(w is not None),
           "chi_walk_entangling_frac": 0.0,
           "chi_walk_extrapolated": 0.0,
           "chi_walk_rot_near_zero": 0.0,
           "chi_walk_rot_near_pi": 0.0,
           "chi_walk_rot_near_frac": 0.0,
           "chi_walk_rot_mixing_mean": 0.0}
    costs = price(w, 10, 100, 2.5) if w is not None else {}
    if w is not None:
        out["chi_walk_entangling_frac"] = w["n_2q_entangling"] / max(1, w["n_2q"])
        out["chi_walk_extrapolated"] = w["extrapolated"]
        out["chi_walk_rot_near_zero"] = w.get("n_rotation_near_zero",0)
        out["chi_walk_rot_near_pi"] = w.get("n_rotation_near_pi",0)
        out["chi_walk_rot_near_frac"] = (w.get("n_rotation_near_zero",0) +
                                         w.get("n_rotation_near_pi",0)) / max(1,w.get("n_rotation_parsed",0))
        out["chi_walk_rot_mixing_mean"] = w.get("rotation_mixing_mean",0)
    for threshold in THRESHOLDS:
        key = str(threshold)
        tab = w["tables"][key] if w is not None else None
        out[f"chi_walk_cost_overhead_{key}"] = costs.get(key, 0.0)
        out[f"chi_walk_max_logchi_{key}"] = tab["max_logchi"] if tab else 0.0
        out[f"chi_walk_sat_frac_{key}"] = tab["sat_2q_gates"] / max(1, w["n_2q"]) if tab else 0.0
        out[f"chi_walk_sat_start_{key}"] = tab["sat_start_frac"] if tab else 0.0
        out[f"chi_walk_links_at_cap_{key}"] = tab["frac_links_at_cap"] if tab else 0.0
    return out


def select_model_features(flat, threshold):
    """Select the row's threshold while leaving shared walk features intact."""
    key = str(threshold)
    out = {name: flat.get(name, 0.0) for name in
           ("chi_walk_entangling_frac", "chi_walk_extrapolated", "chi_walk_available",
            "chi_walk_rot_near_zero", "chi_walk_rot_near_pi",
            "chi_walk_rot_near_frac", "chi_walk_rot_mixing_mean")}
    for name in MODEL_FEATURE_NAMES:
        if name not in out:
            out[name] = flat.get(f"{name}_{key}", 0.0)
    return out


if __name__ == "__main__":
    demo = """OPENQASM 2.0; include "qelib1.inc"; qreg q[4];
    h q[0]; cx q[0], q[1]; h q[2]; cx q[1], q[2]; cx q[0], q[3];"""
    for track in (False, True):
        w = walk(demo, classical_tracking=track)
        cost = 10 ** price(w, 0, 0, 3)["16"] - 1
        print(f"classical_tracking={track}: cost = {cost:.0f}, "
              f"entangling 2q gates = {w['n_2q_entangling']}/{w['n_2q']}")
