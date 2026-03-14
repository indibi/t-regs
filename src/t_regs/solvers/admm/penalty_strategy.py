"""ADMM Penalty parameter selection strategies"""

import abc
from typing import Any, Dict

from copy import deepcopy

class ADMMPenaltyParameterStrategy(abc.ABC):

    @abc.abstractmethod
    def get_initial_penalty(self) -> Any:
        """Update ADMM Augmented Lagrangian Penalty"""

    @abc.abstractmethod
    def update_penalty(self, iteration, penalty, *vargs, **kwargs) -> Any:
        """Update ADMM Augmented Lagrangian Penalty"""
    
    def __str__(self):
        return type(self).__name__

    @abc.abstractmethod
    def get_config(self):
        """Get penalty selection strategy parameters"""


class FixedPP(ADMMPenaltyParameterStrategy):
    """Fixed ADMM Penalty Parameter"""
    def __init__(
        self,
        rho_init: Dict[str,float] | float
    ):
        self._rho_init = rho_init

    def get_initial_penalty(self):
        return self._rho_init

    def update_penalty(self, iteration, rho, *vargs, **kwargs):
        return self._rho_init
    
    def get_config(self):
        return {
            'strategy': type(self).__name__,
            'rho_init': self._rho_init,
        }


class ResidualBalancingPP(ADMMPenaltyParameterStrategy):
    def __init__(
            self,
            rho_init: float | Dict[str, float] = 1e-2,
            tau_increase: float=2,
            tau_decrease: float=2,
            mu_threshold: float=10,
            update_until: int | None = 200,
    ):
        """Residual Balancing ADMM Penalty parameter selection scheme.

        Step size grow when (|r| > `mu_threhsold` * |s|) and shrink when 
        (|s| > `mu_threhsold` * |r|) where r and s are the primal and dual
        residuals.
        Parameters
        ----------
            rho_init: float | Dict
                Initial penalty parameter(s)
            tau_increase: float
                Growth factor of the penalty parameter.
            tau_decrease: float
                Decay factor of the penalty parameter.
            mu_threshold: float
                Threshold for when to update the parameters.
            update_until: int | None
                Penalty parameters stop updating after the iterations are reached
        
        References
        ----------
        ..  S. Boyd and N. Parikh and E. Chu and B. Peleato and J.
            Eckstein (2010), “Distributed optimization and statistical
            learning via the alternating direction method of multipliers”
        """
        self._rho_init = rho_init
        self._tau_increase = tau_increase
        self._tau_decrease = tau_decrease
        self._mu_threshold = mu_threshold
        self._update_until = update_until

    def get_initial_penalty(self):
        return self._rho_init
    
    def update_penalty(
            self,
            iteration: int,
            rho: Dict['str', float] | float,
            pri_residual: Dict['str', float] | float,
            dual_residual: Dict['str', float] | float
    ):
        """Update penalty parameter(s) according to residual balancing rule
        
        Parameters
        ----------
            iteration: int
                Algorithm iteration
            rho: Dict['str', float] | float,
                ADMM Penalty parameter(s) either in dictionary format or float.
            pri_residual: Dict['str', float] | float,
                ADMM Primal Residual(s), either in dictionary format or float.
            dual_residual: Dict['str', float] | float,
                ADMM dual residual(s).
        
        Returns
        -------
            new_rho: Dict['str', float] | float
                Next penalty parameters
            was_updated: list['str'] | bool
                Flag whether the penalty parameter(s) was updated.

        Notes
        -----
            - If `rho` is a float,
                - and the residuals are in dictionary format, primal and dual
                residual are root square summed.
            - If `rho` is a dictionary
                - with matching keys to dictionary residuals,
                each penalty parameter is updated separately.
                - with float residuals, each rho is updated according to
                the residuals
        """
        if self._update_until is not None:
            if iteration >= self._update_until:
                was_updated = False
                return rho, was_updated
        new_rho = deepcopy(rho)
        if isinstance(rho, Dict):
            was_updated = []
            if (isinstance(pri_residual, Dict)
                and isinstance(dual_residual, Dict)):
                for key in rho.keys():
                    r = pri_residual[key]
                    s = dual_residual[key]
                    new_rho[key], updated = self._rho_update(r, s, rho[key])
                    if updated:
                        was_updated.append(key)

            elif ((isinstance(pri_residual, float)
                    and isinstance(dual_residual, float))):
                r = pri_residual
                s = dual_residual
                for key in rho.keys():
                    new_rho[key], updated = self._rho_update(r, s, rho[key])
                    if updated:
                        was_updated.append(key)
            else:
                raise ValueError(f"Primal and Dual variables do not match.")
        else:
            if isinstance(pri_residual, Dict):
                r = sum([v**2 for v in pri_residual.values()])**0.5
            else:
                r = pri_residual
            if isinstance(dual_residual, Dict):
                s = sum([v**2 for v in dual_residual.values()])**0.5
            else:
                s = dual_residual
            
            new_rho, was_updated = self._rho_update(r, s, rho)
        return new_rho, was_updated

    def _rho_update(self, r, s, rho):
        if r > s*self._mu_threshold:
            rho_new = rho*self._tau_increase
            updated = True
        elif s > r*self._mu_threshold:
            rho_new = rho/self._tau_decrease
            updated = True
        else:
            rho_new = rho
            updated = False
        return rho_new, updated

    def get_config(self):
        return {
            'name': type(self).__name__,
            'rho_init': self._rho_init,
            'tau_increase': self._tau_increase,
            'tau_decrease': self._tau_decrease,
            'mu_threshold': self._mu_threshold,
            'update_until': self._update_until,
        }
