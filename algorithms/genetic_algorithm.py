"""
algorithms/genetic_algorithm.py
Population-based optimization using selection, recombination, and mutation.
"""

from typing import Tuple
from algorithms.base import LocalSearchSolver
from core.state import State
from core.problem import CommuteProblem
from algorithms.operators import get_neighbors, crossover, random_state


class GeneticAlgorithmSolver(LocalSearchSolver):
    def __init__(self, problem: CommuteProblem, **kwargs):
        super().__init__(problem, **kwargs)
        self.population_size: int = self.params.get("population_size", 30)
        self.generations: int = self.params.get("generations", 100)
        self.crossover_rate: float = self.params.get("crossover_rate", 0.8)
        self.mutation_rate: float = self.params.get("mutation_rate", 0.2)
        self.elitism: int = self.params.get("elitism", 2)
        self.tournament_size: int = self.params.get("tournament_size", 3)

    def fitness(self, state: State) -> float:
        return 1.0 / (1.0 + self.f(state))

    def pick_parent(self, population):
        group = self.rng.sample(population, self.tournament_size)
        winner = group[0]
        for s in group:
            if self.fitness(s) > self.fitness(winner):
                winner = s
        return winner

    def mutate(self, state: State) -> State:
        options = get_neighbors(state, self.problem)
        if options:
            return self.rng.choice(options)
        return state

    def solve(self) -> Tuple[State, float]:
        """
        Executes Genetic Algorithm optimization.

        Guidelines:
        - Initialize a population of valid candidate solutions.
        - Design a fitness metric suitable for minimizing commute travel time (lower cost -> higher fitness).
        - Use selection operators (such as tournament or roulette selection) to pick parents.
        - Pair parents to produce offspring via `crossover()`, and apply stochastic mutation.
        - Preserve the top candidates across generations if using elitism.
        - Track best population performance per generation in self.history.

        Encouraged:
        - Experimenting with population size, selection pressure, and operator rates to balance exploration vs. exploitation.
        """
        population = [self.problem.generate_initial_state()]
        while len(population) < self.population_size:
            population.append(random_state(self.problem, self.rng))

        gen = 0
        while gen <= self.generations:
            population.sort(key=self.fitness, reverse=True)   # best first
            self.consider(population[0], self.f(population[0]))
            self.log_step()
            if gen == self.generations:
                break

            # elitism
            children = population[:self.elitism]

            while len(children) < self.population_size:
                mom = self.pick_parent(population)
                dad = self.pick_parent(population)

                if self.rng.random() < self.crossover_rate:
                    child = crossover(mom, dad, self.problem, self.rng)
                else:
                    child = mom

                if self.rng.random() < self.mutation_rate:
                    child = self.mutate(child)

                children.append(child)

            population = children
            gen += 1

        return self.best_state, self.best_cost
