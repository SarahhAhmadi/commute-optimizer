"""
algorithms/beam_search.py
Local Beam Search maintaining K parallel candidate states.
"""

from typing import Tuple
from algorithms.base import LocalSearchSolver, TOL
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, random_state


class LocalBeamSearchSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.beam_width: int = self.params.get("beam_width", 5)
        self.max_iterations: int = self.params.get("max_iterations", 150)
        self.same_route_limit: int = self.params.get("same_route_limit", 2)  # diversity cap
        self.patience: int = self.params.get("patience", 10)  # stop after this many idle rounds

    def solve(self) -> Tuple[State, float]:
        """
        Executes Local Beam Search.

        Guidelines:
        - Maintain a set of `beam_width` parallel configurations.
        - In each iteration, generate neighbors from all active states in the beam.
        - Pool and rank all generated successors collectively, then retain the top `beam_width` states.
        - Halt when the beam ceases to improve or reaches max_iterations.
        - Record the best cost of the population at each step in self.history.

        Encouraged:
        - Ensuring diversity across states in the beam to prevent collapse into a single redundant path.

        Prohibited:
        - Treating this as K independent, non-communicating searches.
        """
        # the default state plus random ones , no duplicates
        beam = [self.problem.generate_initial_state()]
        tries = 0
        while len(beam) < self.beam_width and tries < 100:
            s = random_state(self.problem, self.rng)
            if s not in beam:
                beam.append(s)
            tries += 1

        for s in beam:
            self.consider(s, self.f(s))
        self.log_step()

        no_progress = 0
        it = 0
        while it < self.max_iterations and no_progress < self.patience:
            pool = set()
            for s in beam:
                for n in get_neighbors(s, self.problem):
                    pool.add(n)
            if len(pool) == 0:
                break

            ranked = sorted(pool, key=lambda s: (self.f(s), s.departure_time, s.path))

        
            beam = []
            route_count = {}
            for s in ranked:
                if route_count.get(s.path, 0) >= self.same_route_limit:
                    continue
                beam.append(s)
                route_count[s.path] = route_count.get(s.path, 0) + 1
                if len(beam) == self.beam_width:
                    break

            old_best = self.best_cost
            self.consider(beam[0], self.f(beam[0]))
            if self.best_cost < old_best - TOL:
                no_progress = 0
            else:
                no_progress += 1

            self.log_step()
            it += 1

        return self.best_state, self.best_cost
