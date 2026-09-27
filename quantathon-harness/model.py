"""QASM structure -> Quantum Rings runtime.

The trained scikit-learn artifact is in ``artifacts/runtime_model.joblib``.
Rebuild the selected union model with ``research/train_full_union_model.py``;
``research/train_merged_model.py`` builds the compact fallback.
"""
from collections import Counter, deque
import io
import json
import math
from pathlib import Path
import re
import time

CAP_SECONDS = 14400.0
D2 = re.compile(r'^qreg\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]')
D3 = re.compile(r'^qubit\s*(?:\[\s*(\d+)\s*\])?\s+([A-Za-z_]\w*)')
INDEX = re.compile(r'\[\s*(\d+)\s*\]')
IDENT = re.compile(r'[A-Za-z_]\w*')
NUM = re.compile(r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$', re.I)
GATES2 = {'cx','cy','cz','ch','swap','iswap','ecr','cp','crx','cry','crz','rxx','ryy','rzz','cu','cu1','cu3','csx'}
GATES3 = {'ccx','cswap','ccz'}
DIAG = {'z','s','sdg','t','tdg','p','u1','rz','cz','cp','rzz','ccz'}
CLIFF = {'h','x','y','z','s','sdg','sx','sxdg','cx','cy','cz','swap','id','i'}
INVERSE = {'s':'sdg','sdg':'s','t':'tdg','tdg':'t','sx':'sxdg','sxdg':'sx','h':'h','x':'x','y':'y','z':'z'}
GATE_COUNTS = ('h','x','sx','s','t','rx','ry','rz','p','u','u1','u2','u3','cx','cz','cp','swap','rzz','ccx')
KNOWN = GATES2 | GATES3 | DIAG | CLIFF | {'rx','ry','u','u2','u3','reset','measure','barrier','delay'}
RANK2_GATES = {'cx','cy','cz','ch','cp','crx','cry','crz','cu','cu1','cu3',
               'rxx','ryy','rzz'}


def entropy_evenness(counts, categories=None):
    """Shannon entropy divided by the maximum for the observed categories."""
    values = [v for v in counts if v > 0]
    if len(values) < 2:
        return 0.0
    total = sum(values)
    return -sum((v/total)*math.log2(v/total) for v in values)/math.log2(categories or len(values))


def estimated_log_chi(log_chi_upper, randomness):
    """Interpolate log2 bond dimension from 0 to its cut upper bound.

    This is a heuristic effective-entanglement proxy, not a Schmidt-rank
    calculation: chi_est = chi_upper ** randomness. Thus randomness 0 gives
    rank 1 and randomness 1 reaches the available upper bound.
    """
    return max(0.0, log_chi_upper) * min(1.0, max(0.0, randomness))


def angle_value(s):
    """Fast evaluation of common numerical/pi angles, with no arbitrary eval."""
    s = s.replace(' ', '').lower()
    try:
        if NUM.fullmatch(s):
            return float(s)
        if s in ('pi', '+pi', '-pi'):
            return -math.pi if s[0] == '-' else math.pi
        if s.startswith('pi*') or s.startswith('+pi*'):
            return math.pi * float(s.split('*', 1)[1])
        if s.startswith('-pi*'):
            return -math.pi * float(s.split('*', 1)[1])
        if s.endswith('*pi'):
            return math.pi * float(s[:-3])
        if s.startswith('pi/') or s.startswith('-pi/'):
            return (-math.pi if s[0] == '-' else math.pi) / float(s.split('/', 1)[1])
    except (ValueError, OverflowError, ZeroDivisionError):
        pass
    return None


def interaction_geometry(n, pairs):
    """Cheap graph/order proxies, not the circuit tensor network's treewidth."""
    if n < 2 or not pairs:
        return {key: 0 for key in ('graph_cutwidth_original','graph_cutwidth_rcm',
                'graph_mean_cut_original','graph_mean_cut_rcm','graph_span_rcm',
                'graph_rcm_gain','graph_degree_entropy','graph_density',
                'graph_mindegree_width','graph_mindegree_available')}
    adjacency = [set() for _ in range(n)]
    for (a,b) in pairs:
        if 0 <= a < n and 0 <= b < n and a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)
    degrees = [len(neighbors) for neighbors in adjacency]
    unseen = set(range(n))
    cuthill = []
    while unseen:
        start = min(unseen, key=lambda q: (degrees[q],q))
        queue = deque([start])
        unseen.remove(start)
        while queue:
            q = queue.popleft()
            cuthill.append(q)
            for other in sorted(adjacency[q] & unseen, key=lambda p: (degrees[p],p)):
                unseen.remove(other)
                queue.append(other)
    rcm = list(reversed(cuthill))

    def order_stats(order):
        pos = {q:i for i,q in enumerate(order)}
        delta = [0]*(n+1)
        span = total = 0
        for (a,b),weight in pairs.items():
            left,right = sorted((pos[a],pos[b]))
            if left == right:
                continue
            delta[left] += weight
            delta[right] -= weight
            span += weight*(right-left)
            total += weight
        running = 0
        profile = []
        for cut in range(n-1):
            running += delta[cut]
            profile.append(running)
        return max(profile, default=0), sum(profile)/max(1,len(profile)), span/max(1,total)

    original = order_stats(range(n))
    reordered = order_stats(rcm)
    result = {'graph_cutwidth_original':original[0],
              'graph_cutwidth_rcm':reordered[0],
              'graph_mean_cut_original':original[1],
              'graph_mean_cut_rcm':reordered[1],
              'graph_span_rcm':reordered[2],
              'graph_rcm_gain':(original[0]-reordered[0])/max(1,original[0]),
              'graph_degree_entropy':entropy_evenness(degrees),
              'graph_density':2*len(pairs)/max(1,n*(n-1)),
              'graph_mindegree_width':0, 'graph_mindegree_available':0}
    if n <= 80 and len(pairs) <= 3000:
        # Min-degree elimination with fill is an inexpensive upper-bound proxy
        # for this *interaction graph*, not the exact circuit contraction width.
        graph = [set(neighbors) for neighbors in adjacency]
        remaining = set(range(n))
        width = 0
        while remaining:
            q = min(remaining, key=lambda p: (len(graph[p] & remaining),p))
            neighbors = graph[q] & remaining
            width = max(width,len(neighbors))
            for p in neighbors:
                graph[p].update(neighbors-{p})
                graph[p].discard(q)
            remaining.remove(q)
        result['graph_mindegree_width'] = width
        result['graph_mindegree_available'] = 1
    return result


class SequenceMotifs:
    """Optional, bounded-memory ordered motifs for research ablations.

    Tracks causal order on each qubit, so independent gates may be reordered
    without destroying a cost/mixer layer. No gate names or source metadata
    from custom definitions are used as algorithm labels.
    """
    def __init__(self):
        self.initial_h = set()
        self.touched_two_qubit = set()
        self.last = {}
        self.next_id = 0
        self.phase = {}
        self.cycles = Counter()
        self.mixers = Counter()
        self.costs = Counter()
        self.cost_pairs = Counter()
        self.sandwiches = 0
        self.cp_total = self.cp_tail = 0
        self.cp_tail_qubits = set()
        self.phase_register_cost = self.phase_register_tail_cost = 0
        self.phase_register_tail_qubits = set()

    @staticmethod
    def near(a, b):
        return a is not None and abs(math.remainder(a-b,2*math.pi)) < 1e-5

    def add(self, gate, qubits, params=(), window=0):
        if gate in ('measure','barrier','reset','delay'):
            return
        arity = len(qubits)
        u = gate in ('u','u3') and len(params) >= 3 and all(p is not None for p in params[:3])
        h = gate == 'h' or (u and self.near(params[0],math.pi/2)
                            and self.near(params[1],0) and self.near(params[2],math.pi))
        mixer = gate == 'rx' or (u and self.near(params[1],-math.pi/2)
                                 and self.near(params[2],math.pi/2)
                                 and not self.near(params[0],0))
        zrot = gate in ('rz','p','u1') or (u and self.near(params[0],0)
                                       and self.near(params[1],0))
        if h:
            self.initial_h.update(q for q in qubits if q not in self.touched_two_qubit)
        if gate in ('cp','cu1') and arity == 2:
            self.cp_total += 1
            if window == 7:
                self.cp_tail += 1
                self.cp_tail_qubits.update(qubits)
        cost_pair = None
        if gate in ('rzz','cp','cu1','cz','crz') and arity == 2:
            cost_pair = tuple(sorted(qubits))
        if gate == 'cx' and arity == 2:
            control,target = qubits
            control_last = self.last.get(control)
            target_last = self.last.get(target)
            if (control_last and target_last and control_last[0]=='cx'
                    and control_last[2:] == (control,target)
                    and target_last == ('zrot',control_last[1])):
                cost_pair = tuple(sorted(qubits))
                self.sandwiches += 1
            self.next_id += 1
            event=('cx',self.next_id,control,target)
            self.last[control] = event
            self.last[target] = event
        else:
            for q in qubits:
                prev=self.last.get(q)
                self.last[q]=('zrot',prev[1]) if (zrot and prev and prev[0]=='cx'
                                                and len(prev)>3 and prev[3]==q) else ('other',0)
        if cost_pair is not None:
            self.cost_pairs[cost_pair] += 1
            if all(q in self.initial_h for q in cost_pair):
                self.phase_register_cost += 1
                if window == 7:
                    self.phase_register_tail_cost += 1
                    self.phase_register_tail_qubits.update(cost_pair)
            for q in cost_pair:
                self.costs[q] += 1
                self.phase[q] = 'C'
        if arity >= 2:
            self.touched_two_qubit.update(qubits)
        if mixer and arity == 1:
            q=qubits[0]
            self.mixers[q] += 1
            if self.phase.get(q)=='C':
                self.cycles[q] += 1
            self.phase[q] = 'M'

    def features(self, n, two_q):
        n=max(1,n)
        hfrac=len(self.initial_h)/n
        one=sum(count>=1 for count in self.cycles.values())/n
        two=sum(count>=2 for count in self.cycles.values())/n
        rx=sum(count>=1 for count in self.mixers.values())/n
        pair_total=sum(self.cost_pairs.values())
        reuse=sum(v for v in self.cost_pairs.values() if v>=2)/max(1,pair_total)
        single=one*rx*min(1.0,pair_total/n)
        repeated=two*reuse*rx
        tail=self.cp_tail/max(1,self.cp_total)
        overlap=len(self.cp_tail_qubits & self.initial_h)/max(1,len(self.cp_tail_qubits))
        register_tail=self.phase_register_tail_cost/max(1,self.phase_register_cost)
        register_overlap=len(self.phase_register_tail_qubits & self.initial_h)/max(1,len(self.phase_register_tail_qubits))
        density=min(1.0,two_q/max(1,20*n*n))
        return {'motif_qaoa_cycle1_fraction':one,
                'motif_qaoa_cycle2_fraction':two,
                'motif_qaoa_mixer_coverage':rx,
                'motif_qaoa_cost_reuse':reuse,
                'motif_qaoa_single_layer_score':single,
                'motif_qaoa_score':.5*single+.5*repeated,
                'motif_cx_z_cx_ratio':2*self.sandwiches/max(1,two_q),
                'motif_shor_initial_h_fraction':hfrac,
                'motif_shor_tail_cp_fraction':tail,
                'motif_shor_tail_overlap':overlap,
                'motif_shor_tail_register_phase_fraction':register_tail,
                'motif_shor_arithmetic_density':density,
                'motif_shor_score':min(1.0,3*hfrac)*register_tail*register_overlap*density}


class Stats:
    def __init__(self, n=0, sequence=False):
        self.n = n
        self.sequence = SequenceMotifs() if sequence else None
        self.c = Counter()
        self.gates = Counter()
        self.active = set()
        self.depth = {}
        self.two_depth = {}
        self.last = {}
        self.pairs = Counter()
        self.degrees = Counter()
        self.span_sum = self.span_max = 0
        self.windows = [0]*8
        self.new_pair_windows = [0]*8
        self.cut_delta = Counter()
        self.temporal_cut_delta = [Counter() for _ in range(8)]
        self.temporal_twoq = [0]*8
        self.window_active = [set() for _ in range(8)]
        self.measure_windows = [0]*8
        self.reset_windows = [0]*8
        self.diagonal_run = self.diagonal_run_max = 0
        self.phase_dyadic = 0
        self.angle_bins = Counter()
        self.gate_bigrams = Counter()
        self.previous_gate = None

    def add_custom_span(self, qubits, gate_cost, window=None):
        """Conservative cut budget for a custom gate with unmapped body qubits."""
        if len(qubits) >= 2 and gate_cost > 0:
            if len(qubits) == 2:
                a, b = qubits
                if a > b:
                    a, b = b, a
            else:
                a, b = min(qubits), max(qubits)
            if a < b:
                self.cut_delta[a] += gate_cost
                self.cut_delta[b] -= gate_cost
                if window is not None:
                    self.temporal_cut_delta[window][a] += gate_cost
                    self.temporal_cut_delta[window][b] -= gate_cost

    def note_custom_call(self, qubits, definition, window):
        self.add_custom_span(qubits,definition.c['chi_gate_cost'],window)
        self.temporal_twoq[window] += definition.c['two_q']+definition.c['multi_q']
        self.window_active[window].update(qubits)

    def add(self, gate, qubits, angle=None, conditional=False, window=0, parameters=None):
        if self.sequence is not None:
            self.sequence.add(gate,qubits,parameters or ((angle,) if angle is not None else ()),window)
        if gate == 'barrier':
            self.c['barrier'] += 1
            self.last.clear()
            self.previous_gate = None
            return
        if gate == 'measure':
            self.c['measure'] += 1
            self.measure_windows[window] += 1
        if gate == 'reset':
            self.c['reset'] += 1
            self.reset_windows[window] += 1
        if conditional:
            self.c['conditional'] += 1
        quantum_gate = gate not in ('measure', 'reset', 'delay')
        arity = len(qubits)
        if quantum_gate:
            self.c['ops'] += 1
            self.gates[gate] += 1
            if self.previous_gate is not None:
                self.gate_bigrams[(self.previous_gate, gate)] += 1
            self.previous_gate = gate
            self.c['one_q' if arity == 1 else 'two_q' if arity == 2 else 'multi_q'] += 1
            self.c['diagonal'] += gate in DIAG
            self.diagonal_run = self.diagonal_run+1 if gate in DIAG else 0
            self.diagonal_run_max = max(self.diagonal_run_max,self.diagonal_run)
            clifford = gate in CLIFF
            if angle is not None and gate in ('rx','ry','rz','p','u1'):
                clifford = abs(angle / (math.pi/2) - round(angle / (math.pi/2))) < 1e-9
            self.c['clifford' if clifford else 'nonclifford'] += 1
        if angle is not None:
            wrapped = math.remainder(angle, 2*math.pi)
            self.c['zero_angle'] += abs(wrapped) < 1e-9
            self.c['small_angle'] += 1e-9 <= abs(wrapped) < .025
            if quantum_gate:
                self.angle_bins[int((wrapped+math.pi)/(2*math.pi)*16) % 16] += 1
                if gate in ('cp','cu1','p','u1') and abs(wrapped) > 1e-9:
                    ratio = abs(wrapped/math.pi)
                    power = round(-math.log2(ratio))
                    self.phase_dyadic += 1 <= power <= 12 and abs(ratio-2**(-power)) < 1e-6
        if not qubits:
            return
        self.active.update(qubits)
        self.window_active[window].update(qubits)
        if quantum_gate and arity >= 2:
            # A crossing two-qubit gate multiplies Schmidt rank by at most
            # its operator-Schmidt rank (2 for controlled/Pauli rotations,
            # 4 for a generic gate). For a wider gate, 2**(2*floor(m/2))
            # is a conservative bound for every cut of its m operands.
            gate_cost = (1 if gate in RANK2_GATES else 2) if arity == 2 else 2*(arity//2)
            self.c['chi_gate_cost'] += gate_cost
            self.add_custom_span(qubits, gate_cost, window)
        if arity == 1:
            q0 = qubits[0]
            level = self.depth.get(q0, 0) + 1
            self.depth[q0] = level
        elif arity == 2:
            q0, q1 = qubits
            level = max(self.depth.get(q0, 0), self.depth.get(q1, 0)) + 1
            self.depth[q0] = self.depth[q1] = level
        else:
            level = max((self.depth.get(q, 0) for q in qubits), default=0) + 1
            for q in qubits:
                self.depth[q] = level
        if arity >= 2:
            self.windows[window] += 1
            self.temporal_twoq[window] += 1
            if arity == 2:
                level2 = max(self.two_depth.get(q0, 0), self.two_depth.get(q1, 0)) + 1
                self.two_depth[q0] = self.two_depth[q1] = level2
                self.last.pop(q0, None)
                self.last.pop(q1, None)
            else:
                level2 = max((self.two_depth.get(q, 0) for q in qubits), default=0) + 1
                for q in qubits:
                    self.two_depth[q] = level2
                    self.last.pop(q, None)
            if arity == 2:
                a, b = (q0, q1) if q0 <= q1 else (q1, q0)
                if a != b:
                    if not self.pairs[(a,b)]:
                        self.degrees[a] += 1
                        self.degrees[b] += 1
                        self.new_pair_windows[window] += 1
                    self.pairs[(a,b)] += 1
                    span = b-a
                    self.span_sum += span
                    self.span_max = max(self.span_max, span)
            self.c['early_two_q'] += self.c['ops'] <= 256
        elif arity == 1:
            q = qubits[0]
            old = self.last.get(q)
            if old and ((INVERSE.get(gate) == old[0]) or
                        (angle is not None and gate == old[0] and old[1] is not None and
                         abs(math.remainder(angle + old[1], 2*math.pi)) < 1e-9)):
                self.c['cancel_pairs'] += 1
                self.last.pop(q, None)
            else:
                if old and angle is not None and gate == old[0] and old[1] is not None:
                    self.c['rotation_folds'] += 1
                self.last[q] = (gate, angle)

    def merge(self, other):
        self.c.update(other.c)
        self.gates.update(other.gates)
        self.pairs.update(other.pairs)
        self.degrees.update(other.degrees)
        self.span_sum += other.span_sum
        self.span_max = max(self.span_max, other.span_max)
        self.windows = [a+b for a,b in zip(self.windows,other.windows)]
        self.new_pair_windows = [a+b for a,b in zip(self.new_pair_windows,other.new_pair_windows)]
        self.angle_bins.update(other.angle_bins)
        self.gate_bigrams.update(other.gate_bigrams)
        self.diagonal_run_max = max(self.diagonal_run_max,other.diagonal_run_max)
        self.phase_dyadic += other.phase_dyadic
        # Definition depth is a useful proxy even though formal operands are unmapped.
        self.c['custom_depth'] += max(other.depth.values(), default=0)
        self.c['custom_two_depth'] += max(other.two_depth.values(), default=0)

    def features(self):
        c = self.c
        parent = {}
        def root(q):
            parent.setdefault(q, q)
            while parent[q] != q:
                parent[q] = parent[parent[q]]
                q = parent[q]
            return q
        for a,b in self.pairs:
            parent[root(a)] = root(b)
        components = Counter(root(q) for q in self.active)
        ops = c['ops']
        n2 = c['two_q']
        n = max(self.n, len(self.active))
        d = max(self.depth.values(), default=0)
        out = {'n_qubits':n, 'active_qubits':len(self.active),
               'ops':ops, 'one_q':c['one_q'], 'two_q':n2, 'multi_q':c['multi_q'],
               'depth':d, 'two_depth':max(self.two_depth.values(),default=0),
               'custom_depth':c['custom_depth'], 'custom_two_depth':c['custom_two_depth'],
               'measurements':c['measure'], 'resets':c['reset'], 'barriers':c['barrier'],
               'conditional':c['conditional'], 'clifford':c['clifford'],
               'nonclifford':c['nonclifford'], 'diagonal':c['diagonal'],
               'zero_angles':c['zero_angle'], 'small_angles':c['small_angle'],
               'cancel_pairs':c['cancel_pairs'], 'rotation_folds':c['rotation_folds'],
               'effective_ops':max(0,ops-c['zero_angle']-2*c['cancel_pairs']-c['rotation_folds']),
               'unique_pairs':len(self.pairs), 'pair_reuse':n2/max(1,len(self.pairs)),
               'max_degree':max(self.degrees.values(),default=0),
               'mean_degree':sum(self.degrees.values())/max(1,len(self.active)),
               'max_span':self.span_max, 'mean_span':self.span_sum/max(1,n2),
               'components':len(components),
               'largest_component':max(components.values(),default=0),
               'largest_component_frac':max(components.values(),default=0)/max(1,n),
               'two_q_ratio':n2/max(1,ops), 'nonclifford_ratio':c['nonclifford']/max(1,ops),
               'depth_per_qubit':d/max(1,n), 'early_two_q':c['early_two_q']}
        out.update({'gate_'+g:self.gates[g] for g in GATE_COUNTS})
        if self.sequence is not None:
            out.update(self.sequence.features(n,n2))
        out.update(interaction_geometry(n,self.pairs))
        out.update(dense_log2_bytes=n+3 if n else 0,
                   component_dense_log2_bytes=max(components.values(),default=0)+3 if components else 0,
                   dense_over_128gb=int(n >= 34),
                   component_dense_over_128gb=int(max(components.values(),default=0) >= 34),
                   diagonal_run_max=self.diagonal_run_max,
                   diagonal_run_frac=self.diagonal_run_max/max(1,ops))
        window_widths = [len(window_set) for window_set in self.window_active]
        active_spans = []
        for q in self.active:
            seen = [i for i,window_set in enumerate(self.window_active) if q in window_set]
            if seen:
                active_spans.append(seen[-1]-seen[0]+1)
        out.update(liveness_peak_window=max(window_widths,default=0),
                   liveness_mean_window=sum(window_widths)/8,
                   liveness_mean_span=sum(active_spans)/max(1,len(active_spans)),
                   measurement_early_frac=sum(self.measure_windows[:4])/max(1,sum(self.measure_windows)),
                   reset_early_frac=sum(self.reset_windows[:4])/max(1,sum(self.reset_windows)))
        out.update({'twoq_window_'+str(i):v/max(1,sum(self.windows)) for i,v in enumerate(self.windows)})
        out['peak_window_twoq'] = max(self.windows)/max(1,sum(self.windows))
        out['late_new_pair_ratio'] = sum(self.new_pair_windows[4:])/max(1,sum(self.new_pair_windows))
        cut_profile = []
        budget_profile = []
        budget = 0
        for cut in range(1,n):
            budget += self.cut_delta[cut-1]
            budget_profile.append(budget)
            cut_profile.append(min(cut,n-cut,max(0,budget)))
        mid = n//2
        gate_entropy = entropy_evenness(self.gates.values())
        bigram_entropy = entropy_evenness(self.gate_bigrams.values())
        angle_entropy = entropy_evenness(self.angle_bins.values(),16)
        pair_entropy = entropy_evenness(self.pairs.values())
        window_entropy = entropy_evenness(self.windows)
        randomness = (.22*gate_entropy + .22*bigram_entropy + .18*angle_entropy
                      + .23*pair_entropy + .15*window_entropy)
        capacity = [min(cut,n-cut) for cut in range(1,n)]
        chi_mean = sum(cut_profile)/max(1,len(cut_profile))
        saturation = sum(u/c for u,c in zip(cut_profile,capacity))/max(1,len(capacity))
        chi_peak = max(cut_profile,default=0)
        chi_mid = cut_profile[mid-1] if mid else 0
        time_peak = []
        time_capacity = []
        time_mid = []
        time_mid_t50 = time_mid_t90 = 1.0
        cumulative_delta = Counter()
        for window in range(8):
            cumulative_delta.update(self.temporal_cut_delta[window])
            running = 0
            profile = []
            for cut in range(1,n):
                running += cumulative_delta[cut-1]
                profile.append(min(capacity[cut-1],max(0,running)))
            time_peak.append(max(profile,default=0))
            time_capacity.append(sum(v/c for v,c in zip(profile,capacity))/max(1,len(capacity)))
            middle = profile[mid-1] if mid else 0
            time_mid.append(middle)
            middle_cap = capacity[mid-1] if mid else 0
            if middle_cap and time_mid_t50 == 1.0 and middle >= .5*middle_cap:
                time_mid_t50 = (window+1)/8
            if middle_cap and time_mid_t90 == 1.0 and middle >= .9*middle_cap:
                time_mid_t90 = (window+1)/8
        first_mid_full = next((i for i,v in enumerate(time_mid)
                               if mid and v >= capacity[mid-1]),8)
        out.update(chi_timeline_peak_auc=sum(time_peak)/8,
                   chi_timeline_capacity_auc=sum(time_capacity)/8,
                   chi_timeline_mid_auc=sum(time_mid)/8,
                   chi_timeline_mid_t50=time_mid_t50,
                   chi_timeline_mid_t90=time_mid_t90,
                   chi_timeline_post_mid_sat_twoq=sum(self.temporal_twoq[first_mid_full+1:])/max(1,sum(self.temporal_twoq)))
        two_entanglers = sum(self.gates[g] for g in ('cx','cz','cp','cu1','rzz','rxx','ryy','iswap','ecr'))
        phase_two = self.gates['cp']+self.gates['cu1']
        rotation = sum(self.gates[g] for g in ('rx','ry','rz','p','u1','u2','u3','u'))
        multi = c['multi_q']
        star = min(1.0,max(self.degrees.values(),default=0)/max(1,len(self.pairs)))
        qft_gate_mix = min(1.0,2*phase_two/max(1,ops))*min(1.0,2*self.gates['h']/max(1,n))
        out.update(fingerprint_qft=qft_gate_mix*(.5+.5*self.phase_dyadic/max(1,phase_two)),
                   fingerprint_random_grid=randomness*min(1.0,2*n2/max(1,ops))
                       *min(1.0,6/max(1,out['max_degree'])),
                   fingerprint_qaoa=min(1.0,3*(self.gates['rzz']+self.gates['cp'])/max(1,ops))
                       *min(1.0,3*(self.gates['rx']+self.gates['h'])/max(1,ops))
                       *min(1.0,n2/max(1,len(self.pairs))),
                   fingerprint_arithmetic=min(1.0,4*(self.gates['ccx']+self.gates['cswap']+multi)/max(1,ops))
                       *(.5+.5*min(1.0,4*(self.gates['t']+self.gates['tdg'])/max(1,ops))),
                   fingerprint_variational=min(1.0,2*rotation/max(1,ops))
                       *min(1.0,2*two_entanglers/max(1,ops)),
                   fingerprint_ghz=min(1.0,self.gates['h']/max(1,n))
                       *min(1.0,self.gates['cx']/max(1,n-1))*star,
                   fingerprint_graph_state=min(1.0,self.gates['h']/max(1,n))
                       *min(1.0,self.gates['cz']/max(1,n-1)),
                   fingerprint_grover=min(1.0,4*multi/max(1,ops))
                       *min(1.0,3*(self.gates['h']+self.gates['x'])/max(1,ops)),
                   fingerprint_phase_dyadic=self.phase_dyadic/max(1,phase_two+self.gates['p']+self.gates['u1']))
        out.update(chi_upper_peak=chi_peak,
                   chi_upper_mid=chi_mid,
                   chi_upper_mean=chi_mean,
                   chi_capacity_fraction=saturation,
                   chi_unsaturated_fraction=sum(u<c for u,c in zip(cut_profile,capacity))/max(1,len(capacity)),
                   chi_mid_budget=budget_profile[mid-1] if mid else 0,
                   random_gate_entropy=gate_entropy,
                   random_bigram_entropy=bigram_entropy,
                   random_angle_entropy=angle_entropy,
                   random_pair_entropy=pair_entropy,
                   random_window_entropy=window_entropy,
                   randomness_proxy=randomness,
                   chi_random_peak=chi_peak*randomness,
                   chi_random_pressure=saturation*randomness,
                   chi_est_log2_peak=estimated_log_chi(chi_peak,randomness),
                   chi_est_log2_mid=estimated_log_chi(chi_mid,randomness),
                   chi_est_log2_mean=estimated_log_chi(chi_mean,randomness),
                   chi_est_capacity_fraction=estimated_log_chi(saturation,randomness))
        return out


class RuntimeModel:
    def __init__(self, artifacts_dir='artifacts'):
        p = Path(artifacts_dir)
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / p
        file = p / 'runtime_model.joblib'
        if file.exists():
            import joblib
            self.model = joblib.load(file)
        else:
            self.model = None
        if self.model and 'threshold_specialists' in self.model:
            # Tree prediction for one row is faster without joblib's parallel
            # dispatch overhead. Training remains parallel in research scripts.
            estimators = [self.model.get('global_estimator')]
            estimators += list(self.model.get('threshold_specialists',{}).values())
            estimators += list(self.model.get('timeout_classifiers',{}).values())
            for estimator in estimators:
                if estimator is not None and hasattr(estimator,'n_jobs'):
                    estimator.n_jobs = 1
        components = self.model.get('ensemble', [self.model]) if self.model else []
        all_columns = []
        for part in components:
            all_columns.extend(part.get('columns',()))
            all_columns.extend(part.get('global_columns',()))
            all_columns.extend(part.get('specialist_columns',()))
            for schema in part.get('specialist_columns_by_threshold',{}).values():
                all_columns.extend(schema)
            for schema in part.get('classifier_columns_by_threshold',{}).values():
                all_columns.extend(schema)
        self.use_chi_walk = any(c.startswith('chi_walk_') for c in all_columns)
        self.use_extended_features = any(c.startswith('extended__') for c in all_columns)
        self.rotation_tolerance = (self.model.get('chi_walk_rotation_tolerance_rad',-1.0)
                                   if self.model else -1.0)

    def _walk_features(self, qasm_text, budget_s=3.0, too_large=False):
        from chi_walk import model_features, walk
        if too_large or budget_s <= 0:
            return model_features(None)
        try:
            result = walk(qasm_text, classical_tracking=True, budget_s=budget_s,
                          rotation_tolerance=self.rotation_tolerance)
        except Exception:
            # Keep the already extracted structural features usable when an
            # unfamiliar QASM construct defeats this optional walk.
            result = None
        return model_features(result)

    def _huge_features(self, qasm_text):
        """Bound parsing time for exceptionally large QASM using C-level counts.

        This representation is approximate. Both training and inference use it.
        """
        declarations = qasm_text[:200_000] + qasm_text[-200_000:]
        n = sum(int(x) for x in re.findall(r'\bqubit\s*\[\s*(\d+)\s*\]', declarations))
        n += sum(int(x) for x in re.findall(r'\bqreg\s+\w+\s*\[\s*(\d+)\s*\]', declarations))
        if not n:
            # A very large custom-gate preamble can put declarations in the
            # middle of the file. str.find skips there in C without tokenizing.
            for marker, pattern in [('qubit[',D3),('qreg ',D2)]:
                pos = qasm_text.find(marker)
                while pos >= 0:
                    match = pattern.match(qasm_text[pos:pos+120])
                    if match:
                        n += int(match.group(1) if marker == 'qubit[' else match.group(2))
                    pos = qasm_text.find(marker,pos+len(marker))
        n = min(n, 10000)
        # Only the dominant gates get exact whole-file counts. Each str.count
        # is a C scan; dozens of separate scans would breach the time limit.
        gate_counts = {g:0 for g in GATE_COUNTS}
        for g, token in [('cx','cx '),('cz','cz '),('cp','cp('),
                         ('h','\nh '),('rx','rx('),('ry','ry('),
                         ('rz','rz('),('p','\np('),('ccx','ccx ')]:
            gate_counts[g] = qasm_text.count(token)
        two = gate_counts['cx'] + gate_counts['cz'] + gate_counts['cp']
        three = gate_counts['ccx']
        measure = qasm_text.count('= measure ')+qasm_text.count('\nmeasure ')
        defs = qasm_text.count('\ngate ')
        ops = max(1,qasm_text.count(';')-measure-defs)
        noncliff = sum(gate_counts.get(g,0) for g in ('rx','ry','rz','p','cp','ccx'))
        active = n
        out = {'n_qubits':n,'active_qubits':active,'ops':ops,'one_q':max(0,ops-two-three),
               'two_q':two,'multi_q':three,'depth':ops/max(1,n),'two_depth':two/max(1,n),
               'custom_depth':0,'custom_two_depth':0,'measurements':measure,'resets':qasm_text.count('\nreset '),
               'barriers':qasm_text.count('\nbarrier '),'conditional':qasm_text.count('\nif '),
               'clifford':max(0,ops-noncliff),'nonclifford':noncliff,
               'diagonal':sum(gate_counts.get(g,0) for g in ('s','t','rz','p','cz','cp','rzz')),
               'zero_angles':qasm_text.count('(0) '),'small_angles':0,'cancel_pairs':0,
               'rotation_folds':0,'effective_ops':ops-qasm_text.count('(0) '),
               'unique_pairs':0,'pair_reuse':0,'max_degree':0,'mean_degree':0,
               'max_span':0,'mean_span':0,'components':1 if n else 0,
               'largest_component':n,'largest_component_frac':1 if n else 0,
               'two_q_ratio':two/ops,'nonclifford_ratio':noncliff/ops,
               'depth_per_qubit':ops/max(1,n*n),'early_two_q':0,
               'custom_calls':0,'custom_definitions':defs,
               'unsupported_statements':0,'qasm_bytes':len(qasm_text),'huge_fast_path':1}
        out.update({'gate_'+g:gate_counts[g] for g in GATE_COUNTS})
        out.update({'twoq_window_'+str(i):0 for i in range(8)})
        out['peak_window_twoq'] = 0
        out['late_new_pair_ratio'] = 0
        # Without topology or expanded custom gates, only the dimension cap
        # is trustworthy. Keep this path explicitly loose, not falsely exact.
        capacities = [min(cut,n-cut) for cut in range(1,n)]
        chi = max(capacities,default=0)
        out.update(chi_upper_peak=chi, chi_upper_mid=chi,
                   chi_upper_mean=sum(capacities)/max(1,len(capacities)),
                   chi_capacity_fraction=1 if chi else 0,
                   chi_unsaturated_fraction=0,
                   chi_mid_budget=n*ops,
                   random_gate_entropy=entropy_evenness(gate_counts.values()),
                   random_bigram_entropy=0, random_angle_entropy=0,
                   random_pair_entropy=0, random_window_entropy=0,
                   randomness_proxy=0, chi_random_peak=0, chi_random_pressure=0,
                   chi_est_log2_peak=0, chi_est_log2_mid=0,
                   chi_est_log2_mean=0, chi_est_capacity_fraction=0)
        out.update(dense_log2_bytes=n+3 if n else 0,
                   component_dense_log2_bytes=n+3 if n else 0,
                   dense_over_128gb=int(n >= 34),
                   component_dense_over_128gb=int(n >= 34))
        return out

    def featurize(self, qasm_text: str, include_sequence=False) -> dict:
        started = time.perf_counter() if self.use_chi_walk else 0.0
        if len(qasm_text) > 40_000_000:
            out = self._huge_features(qasm_text)
            if self.use_chi_walk:
                out.update(self._walk_features(qasm_text,too_large=True))
            if self.use_extended_features:
                try:
                    from extended_features import extract_fast_qasm_features
                    out.update({'extended__'+key:value for key,value
                                in extract_fast_qasm_features(qasm_text).items()})
                except Exception:
                    pass
            return out
        main = Stats(sequence=include_sequence)
        current = main
        regs = {}
        definitions = {}
        formal = {}
        definition = None
        conditional_depth = 0
        pending_conditional = False
        unsupported = custom_calls = 0
        position = 0
        window_scale = 8/max(1,len(qasm_text))

        def resolve_qubits(operand):
            if current is main and len(regs) == 1 and '[' in operand:
                base = next(iter(regs.values()))[0]
                return [base+int(index) for index in INDEX.findall(operand)]
            if current is not main:
                return [formal[m.group()] for m in IDENT.finditer(operand) if m.group() in formal]
            qubits = []
            for token in operand.rstrip(';').split(','):
                token = token.strip()
                ident = IDENT.match(token)
                if ident and ident.group() in regs:
                    base,sz = regs[ident.group()]
                    m = INDEX.search(token)
                    qubits.extend([base+int(m.group(1))] if m else range(base,base+sz))
            return qubits

        for raw in io.StringIO(qasm_text):
            position += len(raw)
            line = raw.strip()
            if not line or line.startswith('//'):
                continue
            if '//' in line:
                line = line.split('//',1)[0].strip()
            if not line or line.startswith(('OPENQASM','include ','pragma ','defcalgrammar ')):
                continue
            if definition is None:
                m = D2.match(line)
                if m:
                    regs[m.group(1)] = (main.n,int(m.group(2)))
                    main.n += int(m.group(2))
                    continue
                m = D3.match(line)
                if m:
                    sz = int(m.group(1) or 1)
                    regs[m.group(2)] = (main.n,sz)
                    main.n += sz
                    continue
            if line.startswith(('gate ','opaque ')):
                header = line.split('{',1)[0].strip()
                name_token = header.split(None,1)[1].split('(',1)[0].split(None,1)[0]
                gate_name = name_token.lower()
                tail = header.rsplit(')',1)[-1] if ')' in header else header.split(name_token,1)[-1]
                formals = [s.strip() for s in tail.split(',') if s.strip()]
                formal = {s:i for i,s in enumerate(formals)}
                current = Stats(len(formals))
                definition = gate_name
                if line.startswith('opaque ') or '}' in line:
                    if line.startswith('opaque '):
                        current.c['chi_gate_cost'] = 2*(len(formals)//2)
                    definitions[gate_name] = current
                    current = main
                    definition = None
                continue
            if definition is not None and line.startswith('}'):
                definitions[definition] = current
                definition = None
                formal = {}
                current = main
                continue
            if line.startswith('if ') or line.startswith('if('):
                pending_conditional = True
                if '{' in line:
                    conditional_depth += 1
                if ';' not in line:
                    continue
                line = line.split('{',1)[-1].strip() if '{' in line else line.split(')',1)[-1].strip()
            if line.startswith('}'):
                if conditional_depth:
                    conditional_depth -= 1
                pending_conditional = bool(conditional_depth)
                continue
            if line.startswith(('creg ','bit[','bit ','const ','input ','output ','let ')):
                continue
            if '= measure ' in line:
                gate, params, operand = 'measure','',line.split('= measure ',1)[1]
            elif line.startswith('measure '):
                gate, params, operand = 'measure','',line[8:].split('->',1)[0]
            else:
                first,sep,operand = line.partition(' ')
                if not sep:
                    if line not in ('{','}'):
                        unsupported += 1
                    continue
                p = first.find('(')
                if p >= 0 and not first.endswith(')'):
                    # OpenQASM commonly separates parameters with spaces:
                    # U(pi/2, 0, 0) q[0]; partition(' ') stops too early.
                    close = line.find(')', p+1)
                    if close >= 0:
                        gate = line[:p]
                        params = line[p+1:close]
                        operand = line[close+1:].strip()
                    else:
                        gate, params = first[:p], ''
                else:
                    gate = first[:p] if p >= 0 else first
                    params = first[p+1:-1] if p >= 0 and first.endswith(')') else ''
            gate = gate.lower()
            if gate in ('for','while','def','return','break','continue'):
                unsupported += 1
                continue
            if gate in definitions:
                if current is main:
                    custom_calls += 1
                current.note_custom_call(resolve_qubits(operand),definitions[gate],
                                         min(7,int(position*window_scale)))
                current.merge(definitions[gate])
                if current is main and len(regs) == 1:
                    base = next(iter(regs.values()))[0]
                    current.active.update(base+int(m.group(1)) for m in INDEX.finditer(operand))
                continue
            if gate not in KNOWN:
                unsupported += 1
            qubits = resolve_qubits(operand)
            if not qubits and gate not in ('barrier','measure'):
                unsupported += 1
            a = angle_value(params.split(',',1)[0]) if params else None
            parameters=tuple(angle_value(part) for part in params.split(',')) if include_sequence and params else None
            current.add(gate,qubits,a,pending_conditional,min(7,int(position*window_scale)),parameters)
            if ';' in line and not conditional_depth:
                pending_conditional = False
        out = main.features()
        out.update(custom_calls=custom_calls, custom_definitions=len(definitions),
                   unsupported_statements=unsupported, qasm_bytes=len(qasm_text),
                   huge_fast_path=0)
        if self.use_extended_features:
            try:
                from extended_features import extract_fast_qasm_features
                out.update({'extended__'+key:value for key,value
                            in extract_fast_qasm_features(qasm_text).items()})
            except Exception:
                # The union model remains usable through its global component
                # if the optional second structural scan encounters new syntax.
                pass
        if self.use_chi_walk:
            # Price the optional second scan before the walk so the remaining
            # walk budget reflects the total parser time, not just base QASM
            # extraction.  Keep a one-second margin under the harness cap.
            budget = min(3.0,max(0.0,14.0-(time.perf_counter()-started)))
            out.update(self._walk_features(qasm_text,budget_s=budget))
        return out

    def predict(self, features: dict, threshold: int) -> float:
        if self.model is None:
            n = max(1,features.get('active_qubits',features.get('n_qubits',1)))
            complexity = math.log1p(features.get('effective_ops',0))+.35*features.get('two_depth',0)
            log_seconds = -1.5+.025*min(n,260)+.12*complexity+.8*math.log2(max(1,threshold)/16)
        else:
            import numpy as np
            x = features.copy()
            if self.model.get('setting_encoding') != 'one_hot_categorical':
                x['log_threshold'] = math.log2(max(1,threshold))
                x['threshold'] = threshold
                x['extended__threshold'] = threshold
            # The release artifact treats the three simulator settings as
            # categories; old artifacts can still consume their numeric fields.
            for known_setting in (16,64,512):
                x[f'setting_{known_setting}'] = float(threshold == known_setting)
            if self.use_chi_walk:
                from chi_walk import select_model_features
                x.update(select_model_features(features,threshold))
            def model_vector(columns):
                return np.array([[math.log1p(max(0,float(x.get(c[4:],0)))) if c.startswith('log_') and c != 'log_threshold'
                                  else float(x.get(c,0)) for c in columns]],dtype=float)

            def predict_component(component):
                columns = component['columns']
                vector = model_vector(columns)
                return float(component['estimator'].predict(vector)[0])

            if 'threshold_specialists' in self.model:
                # The known challenge thresholds are stable across train and
                # holdout.  Blend a global model with a regressor trained only
                # at the requested threshold in log space, matching the score.
                global_columns = self.model.get('global_columns',self.model['columns'])
                specialist_columns = self.model.get('specialist_columns',self.model['columns'])
                classifier_columns = self.model.get('classifier_columns',specialist_columns)
                specialist_schemas = self.model.get('specialist_columns_by_threshold',{})
                classifier_schemas = self.model.get('classifier_columns_by_threshold',{})
                global_log = predict_component({
                    'columns':global_columns,
                    'estimator':self.model['global_estimator'],
                })
                specialists = self.model['threshold_specialists']
                key = threshold if threshold in specialists else str(threshold)
                if key in specialists:
                    columns = specialist_schemas.get(
                        threshold,specialist_schemas.get(str(threshold),specialist_columns))
                    specialist_log = float(specialists[key].predict(
                        model_vector(columns))[0])
                    weight = float(self.model.get('threshold_specialist_weight',.5))
                    log_seconds = (1-weight)*global_log + weight*specialist_log
                else:
                    # Unknown future settings remain usable through the global
                    # threshold-aware model rather than selecting a wrong expert.
                    log_seconds = global_log

                classifiers = self.model.get('timeout_classifiers',{})
                classifier = classifiers.get(threshold,classifiers.get(str(threshold)))
                if classifier is not None:
                    columns = classifier_schemas.get(
                        threshold,classifier_schemas.get(str(threshold),classifier_columns))
                    vector = model_vector(columns)
                    classes = list(classifier.classes_)
                    probability = (float(classifier.predict_proba(vector)[0,classes.index(1)])
                                   if 1 in classes else 0.0)
                    if probability >= float(self.model.get('timeout_probability_cutoff',.3)):
                        return CAP_SECONDS
            elif 'ensemble' in self.model:
                log_seconds = sum(component['weight']*predict_component(component)
                                  for component in self.model['ensemble'])
            else:
                log_seconds = predict_component(self.model)
        seconds = float(max(1e-4,min(1e7,10**max(-4,min(7,log_seconds)))))
        from runtime_floors import apply_floor, large_work_floor
        seconds = max(seconds,large_work_floor(features))
        references = self.model.get('reset_family_references',()) if self.model else ()
        if references:
            seconds = apply_floor(seconds,features,threshold,references)
        bank = self.model.get('template_analogue_bank',{}) if self.model else {}
        if bank:
            from template_analogues import blend_with_analogues
            seconds = blend_with_analogues(seconds,features,threshold,bank)
        from rotation_calibration import calibrate_near_basis_runtime
        seconds = calibrate_near_basis_runtime(seconds,features,threshold)
        return seconds

    def predict_interval(self, features: dict, threshold: int):
        """Return point, lower, and upper seconds from grouped-OOF residuals."""
        prediction = self.predict(features,threshold)
        radius = float(self.model.get('interval_log10_radius',.5)) if self.model else .5
        lower = max(1e-4,10**(math.log10(prediction)-radius))
        upper = min(1e7,10**(math.log10(prediction)+radius))
        return prediction,float(lower),float(upper)
