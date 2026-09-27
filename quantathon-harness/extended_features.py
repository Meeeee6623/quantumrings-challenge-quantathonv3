"""Bounded, source-only OpenQASM 2/3 feature extraction.

This module is the challenge-time counterpart to the exact Qiskit-based
extractor.  The released corpus contains QASM files with millions of source
statements, so constructing a Python object for every operation is not viable
under the 15 second parser limit.  The scanner below makes one pass over the
text, keeps O(qubits + distinct interaction pairs) state, and switches to a
deterministic sampled scheduler for very large inputs.

Only information present in the QASM text is used.  In particular, filenames,
benchmark source, circuit family, and inferred algorithm labels are never
accepted by this API.
"""

from __future__ import annotations

import io
import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from extended_threshold import threshold_features

# A fixed vocabulary keeps the model schema stable and prevents a QASM file
# with hundreds of thousands of generated custom-gate names from creating an
# equally large feature vector.
GATE_VOCABULARY = (
    "u",
    "u1",
    "u2",
    "u3",
    "p",
    "id",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "sx",
    "sxdg",
    "rx",
    "ry",
    "rz",
    "cx",
    "cy",
    "cz",
    "ch",
    "cp",
    "crx",
    "cry",
    "crz",
    "cu",
    "cu1",
    "cu3",
    "swap",
    "iswap",
    "ecr",
    "dcx",
    "rxx",
    "ryy",
    "rzz",
    "rzx",
    "xx_plus_yy",
    "ccx",
    "rccx",
    "cswap",
    "mcx",
    "mcp",
    "mcphase",
    "gphase",
)

GATE_ARITY: dict[str, int] = {
    **{name: 1 for name in GATE_VOCABULARY[:19]},
    **{name: 2 for name in GATE_VOCABULARY[19:40]},
    "ccx": 3,
    "rccx": 3,
    "cswap": 3,
    "mcx": 3,
    "mcp": 3,
    "mcphase": 3,
    "gphase": 0,
}

# Counting every rare standard gate would require another full scan of a
# 200+ MB string.  Coarse mode covers the gates observed frequently in the
# released corpus and records all remaining statements in an unclassified
# lexical feature.  Detailed mode still recognizes the full vocabulary.
COARSE_GATE_VOCABULARY = (
    "u",
    "u1",
    "u2",
    "u3",
    "p",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "sx",
    "rx",
    "ry",
    "rz",
    "cx",
    "cz",
    "cp",
    "crz",
    "swap",
    "rxx",
    "ryy",
    "rzz",
    "ccx",
    "rccx",
    "mcx",
    "mcp",
    "mcphase",
)
_COARSE_PREFIX_RE = re.compile(
    r"(?m)^(?P<indent>  )?(?P<name>"
    + "|".join(map(re.escape, sorted((*COARSE_GATE_VOCABULARY, "U"), key=len, reverse=True)))
    + r")"
)

PARAMETER_GATE_NAMES = {
    "u",
    "u1",
    "u2",
    "u3",
    "p",
    "rx",
    "ry",
    "rz",
    "cp",
    "crx",
    "cry",
    "crz",
    "cu",
    "cu1",
    "cu3",
    "rxx",
    "ryy",
    "rzz",
    "rzx",
    "xx_plus_yy",
    "mcp",
    "mcphase",
    "gphase",
}

NON_GATE_OPERATIONS = {"measure", "reset", "barrier", "delay", "store"}
DECLARATION_PREFIXES = (
    "openqasm",
    "include",
    "qreg",
    "creg",
    "qubit",
    "bit",
    "const",
    "input",
    "output",
    "let",
    "array",
    "stretch",
    "duration",
    "pragma",
    "cal",
    "defcalgrammar",
)
CONTROL_PREFIXES = ("if", "else", "for", "while", "switch", "case", "box")
CONTROL_FUNCTION_PREFIXES = tuple(f"{item}(" for item in CONTROL_PREFIXES)

_DECLARATION_RE = re.compile(
    r"(?m)^[ \t]*(?:"
    r"(?P<q2>[qc]reg)[ \t]+(?P<name2>[A-Za-z_]\w*)[ \t]*"
    r"\[[ \t]*(?P<size2>\d+)[ \t]*\]"
    r"|(?P<q3>qubit|bit)[ \t]*(?:\[[ \t]*(?P<size3>\d+)[ \t]*\])?"
    r"[ \t]+(?P<name3>[A-Za-z_]\w*)"
    r")[ \t]*;"
)
_QREF_RE = re.compile(r"\b([A-Za-z_]\w*)[ \t]*\[[ \t]*(\d+)[ \t]*\]")
_WORD_RE = re.compile(r"[A-Za-z_]\w*")
_NUMBER_PATTERN = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_NUMBER_RE = re.compile(rf"^[+-]?{_NUMBER_PATTERN}$")
_PI_LEFT_RE = re.compile(
    rf"^(?P<sign>[+-]?)(?:(?P<coefficient>{_NUMBER_PATTERN})\*)?"
    rf"pi(?:/(?P<divisor>{_NUMBER_PATTERN}))?$",
    re.IGNORECASE,
)
_PI_RIGHT_RE = re.compile(
    rf"^(?P<sign>[+-]?)pi\*(?P<coefficient>{_NUMBER_PATTERN})"
    rf"(?:/(?P<divisor>{_NUMBER_PATTERN}))?$",
    re.IGNORECASE,
)
_DEFINITION_NAME_RE = re.compile(r"^\s*(?:gate|def)\s+([A-Za-z_]\w*)", re.IGNORECASE)


@dataclass(frozen=True)
class FastQASMConfig:
    """Runtime and memory guards for the challenge scanner."""

    detailed_max_bytes: int = 12_000_000
    detailed_work_budget: int = 300_000
    graph_qubit_work_weight: int = 2
    sampled_operation_budget: int = 250_000
    layer_width_limit: int = 200_000
    min_fill_max_qubits: int = 64


@dataclass(frozen=True)
class _Register:
    offset: int
    size: int


@dataclass
class _AngleStats:
    """Bounded summaries of numeric gate parameters found directly in QASM."""

    expression_count: int = 0
    numeric_count: int = 0
    symbolic_count: int = 0
    negative_count: int = 0
    zero_mod_two_pi_count: int = 0
    clifford_count: int = 0
    quarter_turn_count: int = 0
    t_like_count: int = 0
    generic_count: int = 0
    abs_pi_sum: float = 0.0
    abs_pi_squared_sum: float = 0.0
    abs_pi_max: float = 0.0
    mod_two_pi_distance_sum: float = 0.0
    two_qubit_numeric_count: int = 0
    two_qubit_abs_pi_sum: float = 0.0
    two_qubit_clifford_count: int = 0
    rounded_values: Counter[float] = field(default_factory=Counter)
    distinct_overflow: int = 0

    def add(self, expressions: Iterable[str], *, two_qubit: bool) -> None:
        for expression in expressions:
            self.expression_count += 1
            value = _constant_parameter_value(expression)
            if value is None:
                self.symbolic_count += 1
                continue
            self.numeric_count += 1
            normalized = value / math.pi
            absolute = abs(normalized)
            self.abs_pi_sum += absolute
            self.abs_pi_squared_sum += absolute * absolute
            self.abs_pi_max = max(self.abs_pi_max, absolute)
            self.negative_count += int(value < 0)
            mod_distance = abs(((normalized + 1.0) % 2.0) - 1.0)
            self.mod_two_pi_distance_sum += mod_distance
            zero = mod_distance <= 1e-7
            clifford = abs(normalized - round(normalized * 2.0) / 2.0) <= 1e-7
            quarter = abs(normalized - round(normalized * 4.0) / 4.0) <= 1e-7
            self.zero_mod_two_pi_count += int(zero)
            self.clifford_count += int(clifford)
            self.quarter_turn_count += int(quarter)
            self.t_like_count += int(quarter and not clifford)
            self.generic_count += int(not quarter)
            if len(self.rounded_values) < 4096 or round(normalized, 8) in self.rounded_values:
                self.rounded_values[round(normalized, 8)] += 1
            else:
                self.distinct_overflow = 1
            if two_qubit:
                self.two_qubit_numeric_count += 1
                self.two_qubit_abs_pi_sum += absolute
                self.two_qubit_clifford_count += int(clifford)

    def features(self, *, available: bool) -> dict[str, float]:
        if not available:
            return {
                name: math.nan
                for name in (
                    "angle_expression_count",
                    "angle_numeric_count",
                    "angle_symbolic_count",
                    "angle_numeric_fraction",
                    "angle_abs_over_pi_mean",
                    "angle_abs_over_pi_std",
                    "angle_abs_over_pi_max",
                    "angle_mod_two_pi_distance_mean",
                    "angle_zero_mod_two_pi_fraction",
                    "angle_clifford_fraction",
                    "angle_quarter_turn_fraction",
                    "angle_t_like_fraction",
                    "angle_generic_fraction",
                    "angle_negative_fraction",
                    "angle_distinct_count",
                    "angle_distinct_overflow",
                    "angle_value_entropy",
                    "angle_value_entropy_normalized",
                    "two_qubit_angle_numeric_count",
                    "two_qubit_angle_abs_over_pi_mean",
                    "two_qubit_angle_clifford_fraction",
                )
            }
        numeric = self.numeric_count
        mean = self.abs_pi_sum / numeric if numeric else 0.0
        variance = max(0.0, self.abs_pi_squared_sum / numeric - mean * mean) if numeric else 0.0
        entropy, normalized_entropy = _entropy(self.rounded_values.values())
        two_qubit = self.two_qubit_numeric_count
        return {
            "angle_expression_count": float(self.expression_count),
            "angle_numeric_count": float(numeric),
            "angle_symbolic_count": float(self.symbolic_count),
            "angle_numeric_fraction": float(numeric / self.expression_count)
            if self.expression_count
            else 0.0,
            "angle_abs_over_pi_mean": float(mean),
            "angle_abs_over_pi_std": math.sqrt(variance),
            "angle_abs_over_pi_max": float(self.abs_pi_max),
            "angle_mod_two_pi_distance_mean": float(self.mod_two_pi_distance_sum / numeric)
            if numeric
            else 0.0,
            "angle_zero_mod_two_pi_fraction": float(self.zero_mod_two_pi_count / numeric)
            if numeric
            else 0.0,
            "angle_clifford_fraction": float(self.clifford_count / numeric) if numeric else 0.0,
            "angle_quarter_turn_fraction": float(self.quarter_turn_count / numeric)
            if numeric
            else 0.0,
            "angle_t_like_fraction": float(self.t_like_count / numeric) if numeric else 0.0,
            "angle_generic_fraction": float(self.generic_count / numeric) if numeric else 0.0,
            "angle_negative_fraction": float(self.negative_count / numeric) if numeric else 0.0,
            "angle_distinct_count": float(len(self.rounded_values)),
            "angle_distinct_overflow": float(self.distinct_overflow),
            "angle_value_entropy": entropy,
            "angle_value_entropy_normalized": normalized_entropy,
            "two_qubit_angle_numeric_count": float(two_qubit),
            "two_qubit_angle_abs_over_pi_mean": float(self.two_qubit_abs_pi_sum / two_qubit)
            if two_qubit
            else 0.0,
            "two_qubit_angle_clifford_fraction": float(self.two_qubit_clifford_count / two_qubit)
            if two_qubit
            else 0.0,
        }


def _declarations(text: str) -> tuple[dict[str, _Register], int, int, int]:
    quantum: dict[str, _Register] = {}
    num_qubits = 0
    num_clbits = 0
    classical_registers = 0
    for match in _DECLARATION_RE.finditer(text):
        kind = (match.group("q2") or match.group("q3") or "").lower()
        name = match.group("name2") or match.group("name3") or ""
        size = int(match.group("size2") or match.group("size3") or 1)
        if kind in {"qreg", "qubit"}:
            # OpenQASM does not allow redeclaration in the same scope.  Keeping
            # the first declaration is a safe degradation for malformed input.
            if name not in quantum:
                quantum[name] = _Register(num_qubits, size)
                num_qubits += size
        else:
            num_clbits += size
            classical_registers += 1
    return quantum, num_qubits, num_clbits, classical_registers


def _strip_block_comments(line: str, in_comment: bool) -> tuple[str, bool]:
    """Remove C-style block comments without copying the full QASM string."""

    if not in_comment and "/*" not in line:
        return line, False
    parts: list[str] = []
    position = 0
    while position < len(line):
        if in_comment:
            end = line.find("*/", position)
            if end < 0:
                return "".join(parts), True
            position = end + 2
            in_comment = False
        else:
            start = line.find("/*", position)
            if start < 0:
                parts.append(line[position:])
                break
            parts.append(line[position:start])
            position = start + 2
            in_comment = True
    return "".join(parts), in_comment


def _after_condition(statement: str) -> str:
    """Return an operation following a same-line QASM2/QASM3 ``if``."""

    text = statement.lstrip()
    if not text.lower().startswith("if"):
        return text
    start = text.find("(")
    if start < 0:
        return ""
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                remainder = text[index + 1 :].lstrip()
                return "" if remainder.startswith("{") else remainder
    return ""


def _operation_name(statement: str) -> tuple[str | None, int]:
    """Extract a normalized operation name and QASM3 modifier count."""

    text = _after_condition(statement)
    if not text or text[0] in "{}":
        return None, 0
    lower = text.lower()
    if "= measure" in lower or lower.startswith("measure "):
        return "measure", 0
    first = text.split(None, 1)[0].lower().rstrip(";")
    if first in CONTROL_PREFIXES or first.startswith(CONTROL_FUNCTION_PREFIXES):
        return None, 0
    declaration_token = first.split("[", 1)[0]
    if declaration_token in DECLARATION_PREFIXES or first.startswith("//"):
        return None, 0
    modifier_count = text.count("@")
    if modifier_count:
        tail = text.rsplit("@", 1)[1].lstrip()
        if not tail:
            return None, modifier_count
        first = tail.split(None, 1)[0].lower().rstrip(";")
    return first.split("(", 1)[0], modifier_count


def _parameter_expressions(statement: str, name: str) -> tuple[str, ...]:
    """Return top-level parameter expressions without evaluating arbitrary code."""

    lower = statement.lower()
    position = lower.find(name)
    if position < 0:
        return ()
    opening = statement.find("(", position + len(name))
    whitespace = len(statement)
    for token in (" ", "\t", "\n"):
        found = statement.find(token, position + len(name))
        if found >= 0:
            whitespace = min(whitespace, found)
    if opening < 0 or opening > whitespace:
        return ()
    depth = 0
    for index in range(opening, len(statement)):
        if statement[index] == "(":
            depth += 1
        elif statement[index] == ")":
            depth -= 1
            if depth == 0:
                body = statement[opening + 1 : index].strip()
                if not body:
                    return ()
                expressions: list[str] = []
                start = 0
                nested = 0
                for offset, char in enumerate(body):
                    if char in "([":
                        nested += 1
                    elif char in ")]":
                        nested = max(0, nested - 1)
                    elif char == "," and nested == 0:
                        expressions.append(body[start:offset].strip())
                        start = offset + 1
                expressions.append(body[start:].strip())
                return tuple(item for item in expressions if item)
    return ()


def _parameter_count(statement: str, name: str) -> int:
    return len(_parameter_expressions(statement, name))


def _constant_parameter_value(expression: str) -> float | None:
    """Parse common numeric and pi-multiple angle forms conservatively."""

    compact = expression.strip().replace(" ", "").replace("π", "pi")
    while compact.startswith("(") and compact.endswith(")"):
        compact = compact[1:-1]
    if _NUMBER_RE.fullmatch(compact):
        try:
            return float(compact)
        except ValueError:
            return None
    match = _PI_LEFT_RE.fullmatch(compact) or _PI_RIGHT_RE.fullmatch(compact)
    if match is None:
        return None
    coefficient = float(match.group("coefficient") or 1.0)
    divisor = float(match.group("divisor") or 1.0)
    if divisor == 0:
        return None
    sign = -1.0 if match.group("sign") == "-" else 1.0
    return sign * coefficient * math.pi / divisor


def _definition_name(statement: str) -> str:
    match = _DEFINITION_NAME_RE.match(statement)
    return match.group(1).lower() if match else ""


def _resolve_qubit_groups(
    statement: str,
    registers: dict[str, _Register],
) -> list[tuple[int, ...]]:
    """Resolve indexed operands or inexpensive whole-register broadcasts."""

    qubits: list[int] = []
    for match in _QREF_RE.finditer(statement):
        register = registers.get(match.group(1))
        index = int(match.group(2))
        if register is not None and 0 <= index < register.size:
            qubits.append(register.offset + index)
    if qubits:
        return [tuple(dict.fromkeys(qubits))]

    # QASM2 permits whole-register broadcasts (``h q;`` and ``measure q -> c``).
    # This path is rare and only runs when no indexed operand was present.
    names = [word for word in _WORD_RE.findall(statement) if word in registers]
    names = list(dict.fromkeys(names))
    if not names:
        return [()]
    operands = [
        tuple(range(registers[name].offset, registers[name].offset + registers[name].size))
        for name in names
    ]
    width = max(len(operand) for operand in operands)
    if all(len(operand) in {1, width} for operand in operands):
        return [
            tuple(operand[0] if len(operand) == 1 else operand[index] for operand in operands)
            for index in range(width)
        ]
    return [tuple(qubit for operand in operands for qubit in operand)]


def _entropy(counts: Iterable[float]) -> tuple[float, float]:
    values = [float(value) for value in counts if value > 0]
    total = sum(values)
    if total <= 0:
        return 0.0, 0.0
    entropy = -sum((value / total) * math.log2(value / total) for value in values)
    normalized = entropy / math.log2(len(values)) if len(values) > 1 else 0.0
    return float(entropy), float(normalized)


def _coarse_prefix_counts(text: str) -> tuple[Counter[str], Counter[str]]:
    """Count executable and definition gate prefixes in one text pass.

    Longest-first alternatives preserve the old prefix correction (``s``
    versus ``sdg``/``swap``).  The fixed vocabulary and indentation rule are
    unchanged, while one regex scan replaces about sixty full-string scans.
    """

    top: Counter[str] = Counter()
    definitions: Counter[str] = Counter()
    for match in _COARSE_PREFIX_RE.finditer(text):
        indent = match.group("indent")
        # The previous counter counted indented gate bodies only after a
        # newline; preserve that edge case for feature-cache parity.
        if indent and match.start() == 0:
            continue
        (definitions if indent else top)[match.group("name").lower()] += 1
    # ``pragma`` is the only common directive beginning with a gate token.
    top["p"] = max(0, top["p"] - text.count("\npragma"))
    return top, definitions


def _treewidth_min_degree(adjacency: list[set[int]]) -> float:
    graph = [set(neighbors) for neighbors in adjacency]
    active = {node for node in range(len(graph))}
    width = 0
    while active:
        node = min(active, key=lambda item: (len(graph[item] & active), item))
        neighbors = list(graph[node] & active)
        width = max(width, len(neighbors))
        for index, left in enumerate(neighbors):
            graph[left].update(neighbors[:index])
            graph[left].update(neighbors[index + 1 :])
            graph[left].discard(left)
        active.remove(node)
    return float(width)


def _treewidth_min_fill(adjacency: list[set[int]], maximum_qubits: int) -> float:
    if len(adjacency) > maximum_qubits:
        return math.nan
    graph = [set(neighbors) for neighbors in adjacency]
    active = {node for node in range(len(graph))}
    width = 0
    while active:

        def fill_score(node: int) -> tuple[int, int, int]:
            neighbors = list(graph[node] & active)
            missing = 0
            for index, left in enumerate(neighbors):
                missing += sum(right not in graph[left] for right in neighbors[index + 1 :])
            return missing, len(neighbors), node

        node = min(active, key=fill_score)
        neighbors = list(graph[node] & active)
        width = max(width, len(neighbors))
        for index, left in enumerate(neighbors):
            graph[left].update(neighbors[:index])
            graph[left].update(neighbors[index + 1 :])
            graph[left].discard(left)
        active.remove(node)
    return float(width)


def _graph_features(
    num_qubits: int,
    pair_weights: Counter[tuple[int, int]],
    *,
    min_fill_max_qubits: int,
    requested_features: set[str] | None = None,
) -> dict[str, float]:
    def wants(name: str) -> bool:
        return requested_features is None or name in requested_features

    adjacency = [set() for _ in range(num_qubits)]
    weighted_degree = [0.0] * num_qubits
    for (left, right), weight in pair_weights.items():
        if left == right or not (0 <= left < num_qubits and 0 <= right < num_qubits):
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)
        weighted_degree[left] += float(weight)
        weighted_degree[right] += float(weight)
    degrees = [len(neighbors) for neighbors in adjacency]
    edge_count = len(pair_weights)
    possible_pairs = num_qubits * (num_qubits - 1) / 2

    seen: set[int] = set()
    components: list[list[int]] = []
    for root in range(num_qubits):
        if root in seen:
            continue
        stack = [root]
        seen.add(root)
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        components.append(component)

    largest = max(components, key=len, default=[])
    diameter = math.nan
    if largest and wants("interaction_diameter"):
        diameter_value = 0
        for root in largest:
            distances = {root: 0}
            queue = [root]
            for node in queue:
                for neighbor in adjacency[node]:
                    if neighbor not in distances:
                        distances[neighbor] = distances[node] + 1
                        queue.append(neighbor)
            diameter_value = max(diameter_value, max(distances.values(), default=0))
        diameter = float(diameter_value)

    clustering_values: list[float] = []
    if wants("interaction_clustering_coefficient"):
        for neighbors in adjacency:
            degree = len(neighbors)
            if degree < 2:
                clustering_values.append(0.0)
                continue
            links = sum(len(adjacency[left] & neighbors) for left in neighbors) / 2
            clustering_values.append(float(links / (degree * (degree - 1) / 2)))

    assortativity = math.nan
    if edge_count >= 2 and wants("interaction_assortativity"):
        pairs = [(degrees[left], degrees[right]) for left, right in pair_weights]
        left_values = [float(left) for pair in pairs for left in pair]
        right_values = [float(right) for pair in pairs for right in reversed(pair)]
        left_mean = sum(left_values) / len(left_values)
        right_mean = sum(right_values) / len(right_values)
        covariance = sum(
            (left - left_mean) * (right - right_mean)
            for left, right in zip(left_values, right_values, strict=True)
        )
        left_var = sum((value - left_mean) ** 2 for value in left_values)
        right_var = sum((value - right_mean) ** 2 for value in right_values)
        if left_var > 0 and right_var > 0:
            assortativity = float(covariance / math.sqrt(left_var * right_var))

    # Weighted adjacency spectral radius via a fixed-count power iteration.
    spectral_radius = 0.0
    if num_qubits and edge_count and wants("interaction_spectral_radius"):
        vector = [1.0 / math.sqrt(num_qubits)] * num_qubits
        for _ in range(24):
            product = [0.0] * num_qubits
            for (left, right), weight in pair_weights.items():
                product[left] += float(weight) * vector[right]
                product[right] += float(weight) * vector[left]
            norm = math.sqrt(sum(value * value for value in product))
            if norm <= 0:
                break
            vector = [value / norm for value in product]
            spectral_radius = norm

    weight_entropy, normalized_weight_entropy = _entropy(pair_weights.values())
    degree_mean = sum(degrees) / num_qubits if num_qubits else 0.0
    degree_variance = (
        sum((degree - degree_mean) ** 2 for degree in degrees) / num_qubits if num_qubits else 0.0
    )
    need_min_degree = wants("treewidth_min_degree") or wants("approx_treewidth")
    need_min_fill = wants("treewidth_min_fill") or wants("approx_treewidth")
    min_degree_width = (_treewidth_min_degree(adjacency) if num_qubits else 0.0) if need_min_degree else math.nan
    min_fill_width = (_treewidth_min_fill(adjacency, min_fill_max_qubits) if num_qubits else 0.0) if need_min_fill else math.nan
    finite_widths = [value for value in (min_degree_width, min_fill_width) if math.isfinite(value)]

    span_total = 0.0
    span_squared_total = 0.0
    span_max = 0.0
    total_interactions = float(sum(pair_weights.values()))
    cuts = [0.0] * max(0, num_qubits - 1)
    for (left, right), weight in pair_weights.items():
        span = float(abs(right - left))
        span_total += span * weight
        span_squared_total += span * span * weight
        span_max = max(span_max, span)
        for cut in range(min(left, right), max(left, right)):
            cuts[cut] += float(weight)
    span_mean = span_total / total_interactions if total_interactions else 0.0
    span_variance = (
        max(0.0, span_squared_total / total_interactions - span_mean * span_mean)
        if total_interactions
        else 0.0
    )
    cuts_mean = sum(cuts) / len(cuts) if cuts else 0.0
    cuts_variance = sum((value - cuts_mean) ** 2 for value in cuts) / len(cuts) if cuts else 0.0
    edge_weights = [float(weight) for weight in pair_weights.values()]
    edge_weight_mean = total_interactions / edge_count if edge_count else 0.0
    edge_weight_variance = (
        sum((weight - edge_weight_mean) ** 2 for weight in edge_weights) / edge_count
        if edge_count
        else 0.0
    )
    cycle_rank = max(0, edge_count - num_qubits + len(components))

    return {
        "interaction_node_count": float(num_qubits),
        "interaction_edge_count": float(edge_count),
        "interaction_graph_density": float(edge_count / possible_pairs) if possible_pairs else 0.0,
        "interaction_mean_degree": float(degree_mean),
        "interaction_max_degree": float(max(degrees, default=0)),
        "interaction_degree_variance": float(degree_variance),
        "interaction_mean_weighted_degree": float(sum(weighted_degree) / num_qubits)
        if num_qubits
        else 0.0,
        "interaction_max_weighted_degree": float(max(weighted_degree, default=0.0)),
        "interaction_connected_components": float(len(components)),
        "interaction_nontrivial_components": float(sum(len(item) > 1 for item in components)),
        "interaction_largest_component_fraction": float(len(largest) / num_qubits)
        if num_qubits
        else 0.0,
        "interaction_clustering_coefficient": float(sum(clustering_values) / len(clustering_values))
        if clustering_values
        else 0.0,
        "interaction_weighted_clustering": math.nan,
        "interaction_diameter": diameter,
        "interaction_assortativity": assortativity,
        "interaction_spectral_radius": float(spectral_radius),
        "interaction_entropy": weight_entropy,
        "interaction_entropy_normalized": normalized_weight_entropy,
        "interaction_distinct_pair_count": float(edge_count),
        "interaction_distinct_pair_fraction": float(edge_count / possible_pairs)
        if possible_pairs
        else 0.0,
        "interaction_total_count": total_interactions,
        "interaction_edge_weight_mean": float(edge_weight_mean),
        "interaction_edge_weight_max": float(max(edge_weights, default=0.0)),
        "interaction_edge_weight_std": math.sqrt(edge_weight_variance),
        "interaction_repeat_factor": float(edge_weight_mean),
        "interaction_max_edge_weight_fraction": float(
            max(edge_weights, default=0.0) / total_interactions
        )
        if total_interactions
        else 0.0,
        "interaction_cycle_rank": float(cycle_rank),
        "interaction_cycle_rank_fraction": float(cycle_rank / edge_count) if edge_count else 0.0,
        "interaction_mean_span": span_mean,
        "interaction_max_span": span_max,
        "interaction_span_std": math.sqrt(span_variance),
        "cut_crossings_mean": cuts_mean,
        "cut_crossings_max": max(cuts, default=0.0),
        "cut_crossings_std": math.sqrt(cuts_variance),
        "cut_crossings_active_fraction": float(sum(value > 0 for value in cuts) / len(cuts))
        if cuts
        else 0.0,
        "cut_crossings_max_fraction": float(max(cuts, default=0.0) / total_interactions)
        if total_interactions
        else 0.0,
        "treewidth_min_fill": min_fill_width,
        "treewidth_min_degree": min_degree_width,
        "approx_treewidth": min(finite_widths, default=math.nan),
        "line_graph_treewidth_min_fill": math.nan,
    }


def extract_fast_qasm_features(
    qasm_text: str,
    *,
    config: FastQASMConfig | None = None,
    requested_features: set[str] | None = None,
) -> dict[str, float]:
    """Extract a stable numeric feature vector from OpenQASM 2 or 3 text.

    A requested feature set returns only those fields and avoids expensive
    graph summaries that the fitted production model does not consume.

    Small and medium files receive exact per-operation scheduling and
    interaction accounting.  Large files retain exact lexical/gate counts but
    use deterministic operation sampling for graph and depth estimates.  The
    returned ``fast_detail_mode`` and ``fast_sample_stride`` features let a
    model learn any systematic approximation effect.
    """

    if not isinstance(qasm_text, str):
        raise TypeError("qasm_text must be a string")
    settings = config or FastQASMConfig()
    text_bytes = len(qasm_text.encode("utf-8", errors="replace"))
    source_lines = qasm_text.count("\n") + (1 if qasm_text else 0)
    source_semicolon_count = qasm_text.count(";")
    registers, num_qubits, num_clbits, classical_register_count = _declarations(qasm_text)
    detailed_work_units = (
        source_semicolon_count + settings.graph_qubit_work_weight * num_qubits * num_qubits
    )
    detailed = (
        text_bytes <= settings.detailed_max_bytes
        and detailed_work_units <= settings.detailed_work_budget
    )
    stride = 1 if detailed else max(1, math.ceil(source_lines / settings.sampled_operation_budget))
    quantum_register_count = len(registers)

    qasm3 = bool(re.search(r"(?im)^\s*OPENQASM\s+3(?:\.\d+)?\s*;", qasm_text[:4096]))
    qasm2 = bool(re.search(r"(?im)^\s*OPENQASM\s+2(?:\.\d+)?\s*;", qasm_text[:4096]))

    gate_histogram: Counter[str] = Counter()
    definition_histogram: Counter[str] = Counter()
    distinct_gate_types: set[str] = set()
    distinct_gate_type_overflow = 0
    pair_weights: Counter[tuple[int, int]] = Counter()
    gate_count = 0
    operation_count = 0
    measurement_count = 0
    reset_count = 0
    barrier_count = 0
    delay_count = 0
    single_count = 0
    two_count = 0
    multi_count = 0
    zero_qubit_count = 0
    parameterized_count = 0
    parameter_count = 0
    conditional_count = 0
    loop_count = 0
    modifier_count = 0
    custom_gate_definition_count = 0
    opaque_declaration_count = 0
    definition_gate_count = 0
    definition_two_qubit_count = 0
    current_definition_gates = 0
    current_definition_two_qubit_gates = 0
    current_definition_parameters = 0
    maximum_definition_gates = 0
    definition_depth = 0
    definition_opened = False
    current_definition_name: str | None = None
    definition_summaries: dict[str, tuple[int, int, int]] = {}
    top_level_name_counts: Counter[str] = Counter()
    top_level_two_qubit_name_counts: Counter[str] = Counter()
    angle_stats = _AngleStats()
    definition_parameter_expression_count = 0
    block_comment = False
    unknown_statement_count = 0
    sampled_operation_count = 0

    depths = [0] * num_qubits
    path_two_qubit = [0] * num_qubits
    first_layer = [-1] * num_qubits
    last_layer = [-1] * num_qubits
    seen_qubit = [False] * num_qubits
    dependency_edges = 0.0
    sampled_depth = 0
    sampled_critical_two_qubit = 0
    layer_widths: Counter[int] = Counter()
    layer_width_tracking = True
    sampled_gate_index = 0

    if detailed:
        line_source: Iterable[str] = io.StringIO(qasm_text)
    else:
        # Giant generated QASM files cannot afford millions of Python-level
        # regex calls.  These exact lexical counts run inside ``str.count`` and
        # distinguish top-level executable text from two-space-indented gate
        # definition bodies used by the released QASM3 generators.
        top_counts, definition_counts = _coarse_prefix_counts(qasm_text)
        gate_histogram.update(top_counts)
        definition_histogram.update(definition_counts)
        gate_count = sum(top_counts.values())
        definition_gate_count = sum(definition_counts.values())
        single_count = sum(count for name, count in top_counts.items() if GATE_ARITY.get(name) == 1)
        two_count = sum(count for name, count in top_counts.items() if GATE_ARITY.get(name) == 2)
        multi_count = sum(
            count for name, count in top_counts.items() if GATE_ARITY.get(name, 0) >= 3
        )
        zero_qubit_count = top_counts.get("gphase", 0)
        definition_two_qubit_count = sum(
            count for name, count in definition_counts.items() if GATE_ARITY.get(name) == 2
        )
        distinct_gate_types.update(name for name, count in top_counts.items() if count)
        parameterized_count = sum(top_counts[name] for name in PARAMETER_GATE_NAMES)
        # Exact parameter arity is intentionally omitted in coarse mode; the
        # parameterized-statement count remains exact for the fixed vocabulary.
        parameter_count = parameterized_count
        measurement_count = qasm_text.count("measure ")
        reset_count = qasm_text.count("\nreset ") + qasm_text.count("\n  reset ")
        barrier_count = qasm_text.count("\nbarrier ") + qasm_text.count("\n  barrier ")
        delay_count = qasm_text.count("\ndelay") + qasm_text.count("\n  delay")
        operation_count = gate_count + measurement_count + reset_count + barrier_count + delay_count
        custom_gate_definition_count = qasm_text.count("\ngate ") + int(
            qasm_text.startswith("gate ")
        )
        opaque_declaration_count = qasm_text.count("\nopaque ")
        conditional_count = qasm_text.count("\nif ") + qasm_text.count("\n  if ")
        loop_count = (
            qasm_text.count("\nfor ")
            + qasm_text.count("\n  for ")
            + qasm_text.count("\nwhile ")
            + qasm_text.count("\n  while ")
        )
        modifier_count = qasm_text.count("@")
        # An optimistic scheduling lower bound.  It is deliberately separate
        # from the exact-mode depth flag and is learned as an approximation.
        active_slots = single_count + 2 * two_count + 3 * multi_count
        sampled_depth = max(1, math.ceil(active_slots / max(1, num_qubits * stride)))
        line_source = ()

    has_block_comments = detailed and "/*" in qasm_text
    for raw_line in line_source:
        line = raw_line
        if has_block_comments:
            line, block_comment = _strip_block_comments(line, block_comment)
        comment = line.find("//")
        if comment >= 0:
            line = line[:comment]
        statement = line.strip()
        if not statement:
            continue
        lower = statement.lower()

        if current_definition_name is not None:
            opening = statement.count("{")
            closing = statement.count("}")
            if not definition_opened:
                if opening:
                    definition_opened = True
                    definition_depth += opening - closing
                # QASM2 permits the opening brace on the line after the gate
                # signature.  That brace is structure, not an operation.
                continue

            name, _ = _operation_name(statement)
            if name and name not in NON_GATE_OPERATIONS and name not in DECLARATION_PREFIXES:
                arity = GATE_ARITY.get(name, 0)
                definition_histogram[name if name in GATE_ARITY else "other"] += 1
                definition_gate_count += 1
                current_definition_gates += 1
                if arity == 2:
                    definition_two_qubit_count += 1
                    current_definition_two_qubit_gates += 1
                expressions = _parameter_expressions(statement, name)
                current_definition_parameters += len(expressions)
                definition_parameter_expression_count += len(expressions)
                angle_stats.add(expressions, two_qubit=arity == 2)
            definition_depth += opening - closing
            if definition_depth <= 0:
                maximum_definition_gates = max(maximum_definition_gates, current_definition_gates)
                definition_summaries[current_definition_name] = (
                    current_definition_gates,
                    current_definition_two_qubit_gates,
                    current_definition_parameters,
                )
                current_definition_gates = 0
                current_definition_two_qubit_gates = 0
                current_definition_parameters = 0
                definition_depth = 0
                definition_opened = False
                current_definition_name = None
            continue

        if lower.startswith("gate ") or lower.startswith("def "):
            custom_gate_definition_count += 1
            current_definition_gates = 0
            current_definition_two_qubit_gates = 0
            current_definition_parameters = 0
            current_definition_name = _definition_name(statement) or (
                f"__anonymous_definition_{custom_gate_definition_count}"
            )
            definition_depth = statement.count("{") - statement.count("}")
            definition_opened = statement.count("{") > 0
            if definition_opened and definition_depth <= 0:
                definition_summaries[current_definition_name] = (0, 0, 0)
                definition_opened = False
                current_definition_name = None
            continue
        if lower.startswith("opaque "):
            opaque_declaration_count += 1
            continue
        if lower.startswith("if"):
            conditional_count += 1
        if lower.startswith(("for ", "while ")):
            loop_count += 1

        name, line_modifiers = _operation_name(statement)
        modifier_count += line_modifiers
        if name is None:
            continue
        if name in {"return", "break", "continue", "end", "durationof", "sizeof"}:
            continue

        if name == "measure":
            kind = "measure"
        elif name in {"reset", "barrier", "delay", "store"}:
            kind = name
        else:
            kind = "gate"

        should_sample = detailed or sampled_gate_index % stride == 0
        if kind != "barrier":
            sampled_gate_index += 1
        groups: list[tuple[int, ...]]
        if should_sample or (kind == "gate" and name not in GATE_ARITY):
            groups = _resolve_qubit_groups(statement, registers)
        else:
            groups = [()]
        semantic_repetitions = len(groups) if detailed else 1
        operation_count += semantic_repetitions

        if kind == "measure":
            measurement_count += semantic_repetitions
        elif kind == "reset":
            reset_count += semantic_repetitions
        elif kind == "barrier":
            barrier_count += semantic_repetitions
        elif kind == "delay":
            delay_count += semantic_repetitions
        elif kind == "gate":
            gate_count += semantic_repetitions
            top_level_name_counts[name] += semantic_repetitions
            histogram_name = name if name in GATE_ARITY else "other"
            gate_histogram[histogram_name] += semantic_repetitions
            if len(distinct_gate_types) < 512:
                distinct_gate_types.add(name)
            elif name not in distinct_gate_types:
                distinct_gate_type_overflow = 1
            expressions = _parameter_expressions(statement, name) if "(" in statement else ()
            params = len(expressions)
            if params:
                parameterized_count += semantic_repetitions
                parameter_count += params * semantic_repetitions
            is_two_qubit_parameter = any(
                (len(group) if group else GATE_ARITY.get(name, 0)) == 2 for group in groups
            )
            for _ in range(semantic_repetitions):
                angle_stats.add(expressions, two_qubit=is_two_qubit_parameter)

            for group in groups if detailed else groups[:1]:
                arity = len(group) if group else GATE_ARITY.get(name, 0)
                if arity == 0:
                    zero_qubit_count += 1
                elif arity == 1:
                    single_count += 1
                elif arity == 2:
                    two_count += 1
                    top_level_two_qubit_name_counts[name] += 1
                else:
                    multi_count += 1

        if not should_sample or kind == "barrier":
            continue
        sample_weight = 1.0 if detailed else float(stride)
        for group in groups:
            if not group:
                continue
            sampled_operation_count += 1
            predecessor = max(group, key=lambda qubit: depths[qubit])
            layer = depths[predecessor] + 1
            path_entanglers = path_two_qubit[predecessor] + (1 if len(group) == 2 else 0)
            dependency_edges += sample_weight * sum(seen_qubit[qubit] for qubit in group)
            for qubit in group:
                depths[qubit] = layer
                path_two_qubit[qubit] = path_entanglers
                seen_qubit[qubit] = True
                if first_layer[qubit] < 0:
                    first_layer[qubit] = layer
                last_layer[qubit] = layer
            sampled_depth = max(sampled_depth, layer)
            sampled_critical_two_qubit = max(sampled_critical_two_qubit, path_entanglers)
            if layer_width_tracking:
                layer_widths[layer] += 1
                if len(layer_widths) > settings.layer_width_limit:
                    layer_widths.clear()
                    layer_width_tracking = False
            if kind == "gate" and len(group) == 2:
                left, right = sorted(group)
                if left != right:
                    pair_weights[(left, right)] += sample_weight

    if current_definition_name is not None:
        maximum_definition_gates = max(maximum_definition_gates, current_definition_gates)
        definition_summaries[current_definition_name] = (
            current_definition_gates,
            current_definition_two_qubit_gates,
            current_definition_parameters,
        )

    custom_invocation_count = sum(
        top_level_name_counts[name]
        for name in definition_summaries
        if name in top_level_name_counts
    )
    invoked_custom_definition_count = sum(
        top_level_name_counts[name] > 0 for name in definition_summaries
    )
    expanded_gate_count = float(gate_count)
    expanded_two_qubit_count = float(two_count)
    expanded_parameter_count = float(parameter_count)
    if detailed:
        for name, (body_gates, body_two_qubit, body_parameters) in definition_summaries.items():
            invocations = top_level_name_counts[name]
            if not invocations:
                continue
            expanded_gate_count += invocations * max(0, body_gates - 1)
            expanded_two_qubit_count += (
                invocations * body_two_qubit - top_level_two_qubit_name_counts[name]
            )
            expanded_parameter_count += invocations * body_parameters
    else:
        expanded_gate_count = float(gate_count + definition_gate_count)
        expanded_two_qubit_count = float(two_count + definition_two_qubit_count)
        expanded_parameter_count = float(parameter_count)

    # Coarse mode preserves exact gate-kind counts from the lexical scan; only
    # graph/depth features are sampled and explicitly marked as such.
    depth = float(sampled_depth * (stride if not detailed else 1))
    active_qubits = sum(seen_qubit)
    gate_entropy, normalized_gate_entropy = _entropy(gate_histogram.values())
    average_width = (
        (single_count + 2 * two_count + 3 * multi_count) / gate_count if gate_count else 0.0
    )
    max_layer_width = float(max(layer_widths.values(), default=0))
    mean_layer_width = float(sampled_operation_count / sampled_depth) if sampled_depth else 0.0
    layer_utilization = (
        (single_count + 2 * two_count + 3 * multi_count) / (num_qubits * depth)
        if num_qubits and depth
        else 0.0
    )
    liveness_numerator = sum(
        max(0, last - first + 1)
        for first, last in zip(first_layer, last_layer, strict=True)
        if first >= 0
    )
    sampled_liveness = (
        liveness_numerator / (num_qubits * sampled_depth) if num_qubits and sampled_depth else 0.0
    )

    features: dict[str, float] = {
        "qasm_version_2": float(qasm2),
        "qasm_version_3": float(qasm3),
        "source_bytes": float(text_bytes),
        "source_lines": float(source_lines),
        "source_semicolon_count": float(source_semicolon_count),
        "source_chars_per_line": float(len(qasm_text) / source_lines) if source_lines else 0.0,
        "source_comment_count": float(qasm_text.count("//")),
        "source_brace_count": float(qasm_text.count("{")),
        "source_control_modifier_count": float(modifier_count),
        "source_conditional_count": float(conditional_count),
        "source_loop_count": float(loop_count),
        "source_pi_token_count": float(qasm_text.lower().count("pi")),
        "fast_detail_mode": float(detailed),
        "fast_detailed_work_units": float(detailed_work_units),
        "fast_sample_stride": float(stride),
        "fast_sampled_operation_count": float(sampled_operation_count),
        "fast_layer_width_truncated": float(not layer_width_tracking),
        "fast_graph_available": float(detailed),
        "num_qubits": float(num_qubits),
        "num_clbits": float(num_clbits),
        "quantum_register_count": float(quantum_register_count),
        "classical_register_count": float(classical_register_count),
        "total_operation_count": float(operation_count),
        "total_gate_count": float(gate_count),
        "circuit_depth": depth,
        "measurement_count": float(measurement_count),
        "reset_count": float(reset_count),
        "barrier_count": float(barrier_count),
        "delay_count": float(delay_count),
        "single_qubit_gate_count": float(single_count),
        "two_qubit_gate_count": float(two_count),
        "multi_qubit_gate_count": float(multi_count),
        "zero_qubit_gate_count": float(zero_qubit_count),
        "gates_per_qubit": float(gate_count / num_qubits) if num_qubits else 0.0,
        "depth_per_gate": float(depth / gate_count) if gate_count else 0.0,
        "two_qubit_gate_fraction": float(two_count / gate_count) if gate_count else 0.0,
        "multi_qubit_gate_fraction": float(multi_count / gate_count) if gate_count else 0.0,
        "parameterized_gate_count": float(parameterized_count),
        "parameter_count": float(parameter_count),
        "definition_parameter_expression_count": float(definition_parameter_expression_count)
        if detailed
        else math.nan,
        "custom_gate_definition_count": float(custom_gate_definition_count),
        "custom_gate_invocation_count": float(custom_invocation_count) if detailed else math.nan,
        "invoked_custom_definition_count": float(invoked_custom_definition_count)
        if detailed
        else math.nan,
        "opaque_gate_declaration_count": float(opaque_declaration_count),
        "definition_body_gate_count": float(definition_gate_count),
        "definition_body_two_qubit_gate_count": float(definition_two_qubit_count),
        "definition_max_body_gate_count": float(maximum_definition_gates),
        "expanded_gate_count_proxy": expanded_gate_count,
        "custom_expanded_gate_count_proxy": expanded_gate_count,
        "custom_expanded_two_qubit_gate_count_proxy": float(max(0.0, expanded_two_qubit_count)),
        "custom_expanded_parameter_count_proxy": float(expanded_parameter_count),
        "custom_expansion_factor": float(expanded_gate_count / gate_count) if gate_count else 0.0,
        "custom_expanded_two_qubit_fraction": float(
            max(0.0, expanded_two_qubit_count) / expanded_gate_count
        )
        if expanded_gate_count
        else 0.0,
        "coarse_unclassified_semicolon_count": float(
            max(
                0,
                source_semicolon_count
                - gate_count
                - definition_gate_count
                - measurement_count
                - reset_count
                - barrier_count
                - delay_count,
            )
        )
        if not detailed
        else 0.0,
        "gate_type_distinct_count": float(len(distinct_gate_types)),
        "gate_type_distinct_overflow": float(distinct_gate_type_overflow),
        "unknown_statement_count": float(unknown_statement_count),
        "structural_gate_depth": depth,
        "structural_gate_entropy": gate_entropy,
        "structural_gate_entropy_normalized": normalized_gate_entropy,
        "structural_active_qubit_fraction": float(active_qubits / num_qubits)
        if num_qubits
        else 0.0,
        "structural_mean_gate_width": float(average_width),
        "structural_max_gate_layer_width": max_layer_width,
        "supermarq_entanglement_ratio": float(two_count / gate_count) if gate_count else 0.0,
        "supermarq_parallelism": float(max(((gate_count / depth) - 1) / (num_qubits - 1), 0.0))
        if depth and num_qubits > 1
        else 0.0,
        "supermarq_liveness": float(sampled_liveness),
        "supermarq_measurement": float(reset_count / operation_count) if operation_count else 0.0,
        "dag_node_count": float(operation_count - barrier_count),
        "dag_dependency_edge_count": float(dependency_edges),
        "dag_critical_path_length": depth,
        "dag_longest_dependency_chain": depth,
        "dag_max_layer_width": max_layer_width,
        "dag_mean_layer_width": mean_layer_width,
        "dag_layer_utilization": float(layer_utilization),
        "dag_density": float(
            dependency_edges
            / ((operation_count - barrier_count) * (operation_count - barrier_count - 1))
        )
        if operation_count - barrier_count > 1
        else 0.0,
        "dag_parallelizable_gate_fraction": float(max(0.0, 1.0 - depth / gate_count))
        if gate_count
        else 0.0,
        "dag_width_depth_ratio": float(max_layer_width / depth) if depth else 0.0,
    }
    features.update(angle_stats.features(available=detailed))
    for name in GATE_VOCABULARY:
        count = float(gate_histogram.get(name, 0))
        features[f"gate_count__{name}"] = count
        features[f"gate_freq__{name}"] = count / gate_count if gate_count else 0.0
        definition_count = float(definition_histogram.get(name, 0))
        features[f"definition_gate_count__{name}"] = definition_count
    other_count = float(gate_histogram.get("other", 0))
    features["gate_count__other"] = other_count
    features["gate_freq__other"] = other_count / gate_count if gate_count else 0.0
    features["definition_gate_count__other"] = float(definition_histogram.get("other", 0))

    features.update(
        _graph_features(
            num_qubits,
            pair_weights,
            min_fill_max_qubits=settings.min_fill_max_qubits,
            requested_features=requested_features,
        )
    )
    features["supermarq_communication"] = features["interaction_graph_density"]
    features["supermarq_critical_depth"] = (
        float(sampled_critical_two_qubit / max(1, two_count / stride)) if two_count else 0.0
    )
    if requested_features is not None:
        return {name: features[name] for name in requested_features if name in features}
    return features


def add_threshold_features(
    circuit_features: dict[str, float], threshold: float
) -> dict[str, float]:
    """Return a copy with the challenge threshold and interaction terms."""

    result = dict(circuit_features)
    result.update(threshold_features(float(threshold), result))
    return result
