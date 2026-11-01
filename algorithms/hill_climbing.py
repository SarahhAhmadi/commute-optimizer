"""
algorithms/hill_climbing.py
Hill Climbing search with sideways move limits and random restarts.
"""

from typing import Tuple
from algorithms.base import LocalSearchSolver, TOL
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, random_state


class HillClimbingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.max_sideways: int = self.params.get("max_sideways", 100)
        self.max_restarts: int = self.params.get("max_restarts", 5)

    def solve(self) -> Tuple[State, float]:
        """
        Executes Hill Climbing optimization.

        Guidelines:
        - Search through the neighborhood greedily for candidates that minimize total travel time.
        - Handle flat local maxima / shoulders by allowing up to `max_sideways` non-improving steps.
        - Apply up to `max_restarts` to explore different starting basins.
        - Track the best objective cost seen at each major step in self.history.

        Prohibited:
        - Looping indefinitely on flat plateaus.
        """
        restart = 0
        while restart <= self.max_restarts:
            if restart == 0:
                current = self.problem.generate_initial_state() # first climb is from default state
            else:
                current =random_state(self.problem, self.rng) #  others random 
            current_cost = self.f(current)
            self.consider(current, current_cost)
            self.log_step()

            sideways = 0
            visited = {current}
            while True:
                neighbors = get_neighbors(current, self.problem)
                if not neighbors:
                    break

                # steepest descent
                best_cost = min(self.f(n) for n in neighbors)
                best = [n for n in neighbors if self.f(n) <= best_cost + TOL]

                if best_cost < current_cost - TOL:
                    # strictly better
                    sideways = 0
                elif best_cost <= current_cost + TOL and sideways < self.max_sideways:
                    # same cost plateau, take a sideways step
                    unvisited = [n for n in best if n not in visited]
                    if unvisited:
                        best = unvisited
                    sideways += 1
                else:
                    break

                current = self.rng.choice(best)
                current_cost = self.f(current)
                visited.add(current)
                self.consider(current, current_cost)
                self.log_step()

            restart += 1

        return self.best_state, self.best_cost
