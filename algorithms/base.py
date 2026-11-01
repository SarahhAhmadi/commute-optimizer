

import math
import random
from abc import ABC, abstractmethod
from typing import Tuple, List, Any, Dict, Optional
from core.state import State
from core.problem import CommuteProblem

TOL = 1e-9 

class LocalSearchSolver(ABC):
    def __init__(self, problem: CommuteProblem, **kwargs):
        self.problem = problem
        self.params: Dict[str, Any] = kwargs
        self.history: List[float] = []
        # seeded from the global generator, so `main.py --seed` makes runs repeatable
        self.rng = random.Random(kwargs.get("seed", random.getrandbits(64)))

        self.best_state: Optional[State] = None
        self.best_cost: float = math.inf
        self.best_step: int = 0        
        self._costs: Dict[State, float] = {}

    def f(self, state: State) -> float:
        if state not in self._costs:
            self._costs[state] = self.problem.evaluate(state)
        return self._costs[state]

    def consider(self, state: State, cost: float) -> None:
        if cost < self.best_cost - TOL:
            self.best_state, self.best_cost = state, cost
            self.best_step = len(self.history)

    def log_step(self) -> None:
        self.history.append(self.best_cost)

    @abstractmethod
    def solve(self) -> Tuple[State, float]:
        """
        Executes the search procedure.

        Returns:
            Tuple[State, float]: The best State found and its evaluated objective cost.
        """
        pass
