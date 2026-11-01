import random
from collections import deque
from typing import List, Optional, Set, Tuple
from core.state import State
from core.problem import CommuteProblem

Path = Tuple[str, ...]

SEGMENT_LEN = 3  
DETOUR_LEN = 4 




def shift_time(problem: CommuteProblem, t: float, steps: int) -> Optional[float]:
    times = problem.get_valid_time_domain()
    idx = min(range(len(times)), key=lambda k: abs(times[k] - t))
    idx += steps
    return times[idx] if 0 <= idx < len(times) else None


def mix_times(problem: CommuteProblem, t1: float, t2: float, rng) -> float:
    times = problem.get_valid_time_domain()
    mid = (t1 + t2) / 2.0
    lower = [t for t in times if t <= mid + 1e-9]
    upper = [t for t in times if t >= mid - 1e-9]
    options = {lower[-1] if lower else times[0], upper[0] if upper else times[-1]}
    return rng.choice(sorted(options))


def detours(problem: CommuteProblem, a: str, b: str,
            blocked: Set[str], limit: int) -> List[Path]:
    found: List[Path] = []

    def extend(walk: List[str]) -> None:
        last = walk[-1]
        if last == b:
            found.append(tuple(walk))
            return
        if len(walk) - 1 == limit:
            return
        for nxt in problem.network.get_neighbors(last):
            if nxt in blocked or nxt in walk:
                continue
            walk.append(nxt)
            extend(walk)
            walk.pop()

    extend([a])
    return found


def connect(problem: CommuteProblem, a: str, b: str,
            blocked: Set[str], limit: int) -> Optional[Path]:
    prev = {a: None}
    frontier = deque([(a, 0)])
    while frontier:
        node, depth = frontier.popleft()
        if node == b:
            walk = []
            while node is not None:
                walk.append(node)
                node = prev[node]
            return tuple(reversed(walk))
        if depth == limit:
            continue
        for nxt in problem.network.get_neighbors(node):
            if nxt not in prev and (nxt == b or nxt not in blocked):
                prev[nxt] = node
                frontier.append((nxt, depth + 1))
    return None


def is_simple(path: Path) -> bool:
    return len(path) == len(set(path))


def get_neighbors(state: State, problem: CommuteProblem) -> List[State]:
    """
    Generates a set of valid neighboring configurations for the current state.

    Guidelines:
    - Explore both modifications to departure_time and topological adjustments to path.
    - Every returned candidate must be a valid, loop-free path starting at problem.origin
      and ending at problem.destination.
    - Departure times must remain within [problem.t_min, problem.t_max].

    Encouraged:
    - Combining discrete time steps with localized sub-path detours.
    - Generating diverse neighborhoods to help local search escape small basins.

    Prohibited:
    - Generating disconnected route segments or paths with cycles.
    - Returning an empty neighborhood when legal modifications exist.

    Args:
        state: Current commute configuration.
        problem: Problem instance providing graph topology and time constraints.

    Returns:
        List[State]: A list of valid neighboring State objects.
    """
    path, t = state.path, state.departure_time
    candidates: List[State] = []
    for step in (-1, +1):
        new_t = shift_time(problem, t, step)
        if new_t is not None:
            candidates.append(State(path, new_t))


    last = len(path) - 1
    for i in range(last):
        for j in range(i + 1, min(last, i + SEGMENT_LEN) + 1):
            keep = set(path[:i]) | set(path[j + 1:])  
            for piece in detours(problem, path[i], path[j], keep, DETOUR_LEN):
                if piece != path[i:j + 1]:
                    candidates.append(State(path[:i] + piece + path[j + 1:], t))

    result, seen = [], {state}
    for c in candidates:
        if c not in seen and problem.is_valid_state(c):
            seen.add(c)
            result.append(c)
    return result



def crossover(parent1: State, parent2: State, problem: CommuteProblem,
              rng: Optional[random.Random] = None) -> State:
    """
    Recombines structural features from two parent configurations into an offspring.

    Guidelines:
    - Exchange routing sub-paths using shared intermediate waypoints.
    - Blend or select departure times from the parents in a way that respects valid step intervals.
    - Verify that the resulting child path remains simple and loop-free.

    Encouraged:
    - Finding valid topological crossover points where parent routes intersect.
    - Having a fallback mechanism (e.g., returning a mutated parent) if parents share no junctions.

    Prohibited:
    - Returning paths with invalid edges or broken topologies.

    Args:
        parent1: First parent State.
        parent2: Second parent State.
        problem: Problem instance providing graph topology and constraints.

    Returns:
        State: A newly generated child State.
    """
    rng = rng or random
    a, b = parent1.path, parent2.path
    t = mix_times(problem, parent1.departure_time, parent2.departure_time, rng)

    shared = [n for n in a[1:-1] if n in b[1:-1]]
    rng.shuffle(shared)
    for node in shared:
        i, j = a.index(node), b.index(node)
        options = [a[:i] + b[j:], b[:j] + a[i:]]
        rng.shuffle(options)
        for child in options:
            if child not in (a, b) and is_simple(child) and problem.is_valid_path(child):
                return State(child, t)

    cuts = [(i, j) for i in range(1, len(a) - 1) for j in range(1, len(b) - 1)]
    rng.shuffle(cuts)
    for i, j in cuts:
        head, tail = a[:i + 1], b[j:]
        if set(head) & set(tail):
            continue
        bridge = connect(problem, a[i], b[j], set(head) | set(tail), DETOUR_LEN)
        if bridge is None:
            continue
        child = head[:-1] + bridge + tail[1:]
        if is_simple(child) and problem.is_valid_path(child):
            return State(child, t)

    base = State(rng.choice((a, b)), t)
    moves = get_neighbors(base, problem)
    return rng.choice(moves) if moves else base




def random_state(problem: CommuteProblem, rng: Optional[random.Random] = None) -> State:
    """A random loop-free route (randomised backtracking DFS) with a random departure slot."""
    rng = rng or random
    goal = problem.destination

    def walk(route: List[str]) -> Optional[Path]:
        if route[-1] == goal:
            return tuple(route)
        nexts = [n for n in problem.network.get_neighbors(route[-1]) if n not in route]
        rng.shuffle(nexts)
        for n in nexts:
            route.append(n)
            done = walk(route)
            if done:
                return done
            route.pop()
        return None

    route = walk([problem.origin])
    if route is None:
        return problem.generate_initial_state()
    return State(route, rng.choice(problem.get_valid_time_domain()))
