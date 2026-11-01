"""
algorithms/simulated_annealing.py
Stochastic optimization governed by a temperature schedule.
"""

import math
from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors


class SimulatedAnnealingSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.initial_temp: float = self.params.get("initial_temp", 100.0)
        self.cooling_rate: float = self.params.get("cooling_rate", 0.995)
        self.min_temp: float = self.params.get("min_temp", 1e-3)
        self.max_iterations: int = self.params.get("max_iterations", 2000)
        self.uphill_moves: int = 0   # how many worse moves were accepted

    def solve(self) -> Tuple[State, float]:
        """
        Executes Simulated Annealing optimization.

        Guidelines:
        - Start from a high temperature where worse moves are accepted with significant probability.
        - Lower temperature over iterations according to a cooling schedule.
        - Downhill moves (improvements) must always be accepted.
        - Uphill moves (worse costs) should be accepted probabilistically according to temperature.
        - Record objective values throughout the run in self.history.

        Prohibited:
        - Ignoring the temperature parameter or turning this into pure hill climbing.
        """
        state = self.problem.generate_initial_state()
        energy = self.f(state)
        self.consider(state, energy)
        self.log_step()

        temp = self.initial_temp
        step = 0
        while step < self.max_iterations and temp >= self.min_temp:
            moves = get_neighbors(state, self.problem)
            if len(moves) == 0:
                break

            nxt = moves[self.rng.randrange(len(moves))]
            nxt_energy = self.f(nxt)

            if self.accept(nxt_energy - energy, temp):
                state, energy = nxt, nxt_energy
                self.consider(state, energy)

            self.log_step()
            temp *=self.cooling_rate
            step += 1

        return self.best_state, self.best_cost

    def accept(self, delta: float, temp: float) -> bool:
        #Metropolis rule
        if delta <= 0:
            return True
        elif self.rng.random() < math.exp(-delta / temp):
            self.uphill_moves += 1
            return True
        else:
            return False
