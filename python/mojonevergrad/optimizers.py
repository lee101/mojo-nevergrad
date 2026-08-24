"""Nevergrad-compatible optimizers with fused Mojo update kernels."""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np
from nevergrad.optimization import base, utils
from nevergrad.parametrization import parameter as p

from ._lib import de_binomial, de_twopoints, f64, pso_update, scale_inplace

IntOrParameter = Union[int, p.Parameter]


class _OnePlusOne(base.Optimizer):
    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: Optional[int] = None,
        num_workers: int = 1,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._sigma = 1.0
        self._previous_best_loss = float("inf")

    def _internal_ask_candidate(self) -> p.Parameter:
        if not self._num_ask:
            candidate = self.parametrization.spawn_child()
            candidate._meta["sigma"] = self._sigma
            return candidate
        candidate = self.current_bests["pessimistic"].parameter.spawn_child()
        step = scale_inplace(
            self._rng.normal(0.0, 1.0, self.dimension), self._sigma
        )
        candidate.set_standardized_data(step)
        candidate._meta["sigma"] = self._sigma
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        if self._previous_best_loss != loss:
            self._sigma *= 2.0 if loss < self._previous_best_loss else 0.84
        self._previous_best_loss = self.current_bests["pessimistic"].mean


class _DifferentialEvolution(base.Optimizer):
    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: Optional[int] = None,
        num_workers: int = 1,
        *,
        crossover: Union[float, str] = 0.5,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        if not (isinstance(crossover, float) or crossover == "twopoints"):
            raise ValueError("only float and 'twopoints' crossover are supported")
        self.crossover = crossover
        self.F1 = 0.8
        self.F2 = 0.8
        self.llambda = max(30, self.num_workers)
        self._uid_queue = utils.UidQueue()
        self.population: dict[str, p.Parameter] = {}
        self._penalize_cheap_violations = True

    def recommend(self) -> p.Parameter:
        return self.current_bests["optimistic"].parameter

    def _internal_ask_candidate(self) -> p.Parameter:
        if len(self.population) < self.llambda:
            candidate = self.parametrization.sample()
            candidate.heritage["lineage"] = candidate.uid
            self.population[candidate.uid] = candidate
            self._uid_queue.asked.add(candidate.uid)
            return candidate

        lineage = self._uid_queue.ask()
        parent = self.population[lineage]
        candidate = parent.spawn_child()
        candidate.heritage["lineage"] = lineage
        parent_data = f64(
            candidate.get_standardized_data(reference=self.parametrization)
        )
        uids = list(self.population)
        first, second = (
            self.population[uids[self._rng.randint(self.llambda)]] for _ in range(2)
        )
        best = self.current_bests["pessimistic"].parameter
        first_data = f64(
            first.get_standardized_data(reference=self.parametrization)
        )
        second_data = f64(
            second.get_standardized_data(reference=self.parametrization)
        )
        best_data = f64(best.get_standardized_data(reference=self.parametrization))
        candidate.parents_uids.extend([first.uid, second.uid])

        if self.crossover == "twopoints" and self.dimension >= 4:
            bounds = sorted(
                self._rng.choice(
                    self.dimension + 1, size=2, replace=False
                ).tolist()
            )
            if bounds[1] == self.dimension and not bounds[0]:
                bounds[self._rng.randint(2)] = self._rng.randint(1, self.dimension)
            parent_inside = int(bool(self._rng.choice([True, False])))
            donor = de_twopoints(
                parent_data,
                first_data,
                second_data,
                best_data,
                self.F1,
                self.F2,
                bounds[0],
                bounds[1],
                bool(parent_inside),
            )
        else:
            rate = float(self.crossover)
            forced = int(self._rng.randint(self.dimension))
            draws = self._rng.uniform(0.0, 1.0, self.dimension - 1)
            random = np.empty(self.dimension, dtype=np.float64)
            random[:forced] = draws[:forced]
            random[forced] = 0.0
            random[forced + 1 :] = draws[forced:]
            donor = de_binomial(
                parent_data,
                first_data,
                second_data,
                best_data,
                random,
                self.F1,
                self.F2,
                rate,
                forced,
            )
        candidate.set_standardized_data(donor, reference=self.parametrization)
        return candidate

    def _internal_tell_candidate(
        self, candidate: p.Parameter, loss: float
    ) -> None:
        lineage = candidate.heritage["lineage"]
        if lineage not in self.population:
            self._internal_tell_not_asked(candidate, loss)
            return
        self._uid_queue.tell(lineage)
        parent = self.population[lineage]
        if loss <= base._loss(parent):
            self.population[lineage] = candidate

    def _internal_tell_not_asked(
        self, candidate: p.Parameter, loss: float
    ) -> None:
        discardable: Optional[str] = None
        if len(self.population) >= self.llambda:
            uid, worst = max(
                self.population.items(), key=lambda item: base._loss(item[1])
            )
            if loss < base._loss(worst):
                discardable = uid
        if discardable is not None:
            del self.population[discardable]
            self._uid_queue.discard(discardable)
        if len(self.population) < self.llambda:
            self.population[candidate.uid] = candidate
            self._uid_queue.tell(candidate.uid)


class _PSO(base.Optimizer):
    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: Optional[int] = None,
        num_workers: int = 1,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.llambda = max(40, num_workers)
        self._uid_queue = utils.UidQueue()
        self.population: dict[str, p.Parameter] = {}
        self._best = self.parametrization.spawn_child()
        self.omega = 0.5 / math.log(2.0)
        self.phip = 0.5 + math.log(2.0)
        self.phig = 0.5 + math.log(2.0)

    @staticmethod
    def _forward(data: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(0.5 + np.arctan(data) / np.pi)

    @staticmethod
    def _backward(data: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.tan((data - 0.5) * np.pi))

    def _boxed_data(self, particle: p.Parameter) -> np.ndarray:
        if particle._frozen and "boxed_data" in particle._meta:
            return particle._meta["boxed_data"]
        data = self._forward(
            f64(particle.get_standardized_data(reference=self.parametrization))
        )
        if particle._frozen:
            particle._meta["boxed_data"] = data
        return data

    def _internal_ask_candidate(self) -> p.Parameter:
        if len(self.population) < self.llambda:
            candidate = self.parametrization.sample()
            self.population[candidate.uid] = candidate
            candidate.heritage["speed"] = self._rng.uniform(
                -1.0, 1.0, self.dimension
            )
            self._uid_queue.asked.add(candidate.uid)
            return candidate

        lineage = self._uid_queue.ask()
        particle = self.population[lineage]
        x = f64(self._boxed_data(particle))
        speed = f64(particle.heritage["speed"])
        global_best = f64(self._boxed_data(self._best))
        parent_best = f64(
            self._boxed_data(particle.heritage.get("best_parent", particle))
        )
        rp = f64(self._rng.uniform(0.0, 1.0, self.dimension))
        rg = f64(self._rng.uniform(0.0, 1.0, self.dimension))
        new_speed, boxed = pso_update(
            x,
            speed,
            parent_best,
            global_best,
            rp,
            rg,
            self.omega,
            self.phip,
            self.phig,
        )
        candidate = particle.spawn_child().set_standardized_data(
            self._backward(boxed), reference=self.parametrization
        )
        candidate.heritage["lineage"] = lineage
        candidate.heritage["speed"] = new_speed
        return candidate

    def _internal_tell_candidate(
        self, candidate: p.Parameter, loss: float
    ) -> None:
        lineage = candidate.heritage["lineage"]
        if lineage not in self.population:
            self._internal_tell_not_asked(candidate, loss)
            return
        self._uid_queue.tell(lineage)
        self.population[lineage] = candidate
        if self._best.loss is None or loss < self._best.loss:
            self._best = candidate
        parent_best = candidate.heritage.get("best_parent", candidate)
        if loss <= base._loss(parent_best):
            candidate.heritage["best_parent"] = candidate

    def _internal_tell_not_asked(
        self, candidate: p.Parameter, loss: float
    ) -> None:
        if len(self.population) >= self.llambda:
            uid, worst = max(
                self.population.items(), key=lambda item: base._loss(item[1])
            )
            if base._loss(worst) < loss:
                return
            del self.population[uid]
            self._uid_queue.discard(uid)
        if "speed" not in candidate.heritage:
            candidate.heritage["speed"] = self._rng.uniform(
                -1.0, 1.0, self.dimension
            )
        self.population[candidate.uid] = candidate
        self._uid_queue.tell(candidate.uid)
        if loss < base._loss(self._best):
            self._best = candidate


def OnePlusOne(
    parametrization: IntOrParameter,
    budget: Optional[int] = None,
    num_workers: int = 1,
) -> base.Optimizer:
    return _OnePlusOne(parametrization, budget=budget, num_workers=num_workers)


def DE(
    parametrization: IntOrParameter,
    budget: Optional[int] = None,
    num_workers: int = 1,
) -> base.Optimizer:
    return _DifferentialEvolution(
        parametrization, budget=budget, num_workers=num_workers, crossover=0.5
    )


def TwoPointsDE(
    parametrization: IntOrParameter,
    budget: Optional[int] = None,
    num_workers: int = 1,
) -> base.Optimizer:
    return _DifferentialEvolution(
        parametrization,
        budget=budget,
        num_workers=num_workers,
        crossover="twopoints",
    )


def PSO(
    parametrization: IntOrParameter,
    budget: Optional[int] = None,
    num_workers: int = 1,
) -> base.Optimizer:
    return _PSO(parametrization, budget=budget, num_workers=num_workers)


registry = {
    "OnePlusOne": OnePlusOne,
    "DE": DE,
    "TwoPointsDE": TwoPointsDE,
    "PSO": PSO,
}

__all__ = ["DE", "OnePlusOne", "PSO", "TwoPointsDE", "registry"]
