from typing import Optional, Tuple
import numpy as np
import torch.nn as nn

from sde_lib import SDE
from torchrl.envs import EnvBase
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec

from tensordict import TensorDict, TensorDictBase
import torch
import models.utils as mutils

# Multiply steps computed from asymptotic behaviour of errors by this.
SAFETY = 0.9

MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.

def norm(x: torch.Tensor):
    dim = tuple(range(1, x.dim()))
    return torch.norm(x, dim=dim) / np.prod(x.shape[1:]) ** 0.5

def min(input: torch.tensor, *others):
    for other in others:
        input = input.clip(max=other)
    return input

def max(input: torch.tensor, *others):
    for other in others:
        input = input.clip(min=other)
    return input

class Env(EnvBase):
    _A: torch.Tensor = NotImplemented
    _B: torch.Tensor = NotImplemented
    _C: torch.Tensor = NotImplemented
    _E: torch.Tensor = NotImplemented
    _P: torch.Tensor = NotImplemented
    _n_stages: int = NotImplemented
    _error_estimator_order: int = NotImplemented
    
    def __init__(
        self, 
        *, 
        sde: SDE, 
        model: nn.Module,
        eps: float=1e-5, 
        env_num: int=1, 
        error_tol: float=1e-6,
        device: torch.device='cpu',
        compute_opt: bool=True,
        use_opt: bool=False,
        max_step: float=1e-2,
        min_step: float=1e-5,
        rtol: float=1e-5,
        atol: float=1e-5,
    ):
        """
        Env for solving the reverse-time SDE.

        Args:
            use_opt: If `True`, use the optimal step size computed by the ODE solver.
            compute_opt: If `True`, add opt_action key to the state
        """
        super().__init__(batch_size=(env_num, ), allow_done_after_reset=True, device=device)
        self._sde = sde
        self._eps = eps
        self._model = model.eval()
        self._model.requires_grad_(False)
        self._error_tol = error_tol
        self._direction = np.sign(self._sde.T - self._eps) if self._sde.T != self._eps else 1
        self._A = self._A.to(device)
        self._B = self._B.to(device)
        self._C = self._C.to(device)
        self._E = self._E.to(device)
        self._P = self._P.to(device)
        
        self._max_step = max_step
        self._min_step = min_step
        self._rtol = rtol
        self._atol = atol
        self._compute_opt = compute_opt
        self._use_opt = use_opt

        self._n = 3 * 32 * 32  + 1
        
        self.action_spec = BoundedTensorSpec(low=self._eps, high=self._sde.T, shape=(env_num,))
        self.done_spec = BoundedTensorSpec(low=False, high=True, shape=(env_num,))
        
        self.state_spec = CompositeSpec(
            z=UnboundedContinuousTensorSpec(shape=(env_num, 3, 32, 32) ),
            p=UnboundedContinuousTensorSpec(shape=(env_num,)),
            epsilon=UnboundedContinuousTensorSpec(shape=(env_num, 3, 32, 32)),
            t=UnboundedContinuousTensorSpec(shape=(env_num,)),
            f=UnboundedContinuousTensorSpec(shape=(env_num, self._n)),
            shape=(env_num,)
        )
        if compute_opt or use_opt:
            self.state_spec.update({
                "h_abs": UnboundedContinuousTensorSpec(shape=(env_num,))
            })
            self._error_exponent = -1 / (self._error_estimator_order + 1)
        if compute_opt:
            self.state_spec.update({"opt_action": self.action_spec.clone()})

        self.observation_spec = self.state_spec.clone()
    
    def _set_seed(self, seed: Optional[int]):
        self.rng = torch.manual_seed(seed)

    def _reset(self, tensordict=None):
        
        done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device) \
            if tensordict is None else tensordict["done"]
        
        z = self._sde.prior_sampling((*self.batch_size, 3, 32, 32)).to(self.device)       
        p = self._sde.prior_logp(z).to(self.device)
        t = torch.ones(self.batch_size, device=self.device) * self._eps
        
        epsilon = torch.randn_like(z) 
        ode_func = self._get_ode_func(epsilon)
        f = ode_func(t, torch.concat([z.flatten(1), p[:, None]], dim=-1))
        
        h_abs = self._select_initial_step(z, p, f, epsilon) \
            if self._compute_opt or self._use_opt else None

        opt_action = h_abs.clone() if self._compute_opt else None

        if tensordict is not None:
            z = torch.where(done, z, tensordict["z"])
            p = torch.where(done, p, tensordict["p"])
            epsilon=torch.where(done, epsilon, tensordict["epsilon"])
            t = torch.where(done, t, tensordict["t"])
            f = torch.where(done[:, None], f, tensordict["f"])
            if h_abs is not None:
                h_abs = torch.where(done, h_abs, tensordict["h_abs"])        
        
        done = torch.where(done, False, True)
        return self._create_tensordict(z=z, p=p, t=t, done=done, \
                                       epsilon=epsilon, f=f, h_abs=h_abs, opt_action=opt_action)
    
    def _rk_step(self, ode_func, t, y, f, h)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """see https://github.com/scipy/scipy/blob/v1.14.1/scipy/integrate/_ivp/rk.py#L14"""
        k = torch.zeros((*self.batch_size, self._n_stages + 1, self._n), dtype=y.dtype, device=y.device)
        k[:, 0] = f
        for s, (a, c) in enumerate(zip(self._A[1:], self._C[1:]), start=1):
            dy = torch.matmul(k[:, :s].permute(0, 2, 1), a[:s]) * h[:, None]
            k[:, s] = ode_func(t + c * h, y + dy)

        y_new = y + h[:, None] * torch.matmul(k[:, :-1].permute(0, 2, 1), self._B)
        f_new = ode_func(t + h, y_new)

        k[:, -1] = f_new
        error = torch.matmul(k.permute(0, 2, 1), self._E) * h[:, None]
        
        return y_new, f_new, error

    def _get_ode_func(self, epsilon):
        def drift_fn(x, t):
            """The drift function of the reverse-time SDE."""
            score_fn = mutils.get_score_fn(self._sde, self._model, train=False, continuous=True)
            # Probability flow ODE is a special case of Reverse SDE
            rsde = self._sde.reverse(score_fn, probability_flow=True)
            return rsde.sde(x, t)[0]

        def div_fn(x, t, eps):
 
            with torch.enable_grad():
                x.requires_grad_(True)
                fn_eps = torch.sum(drift_fn(x, t) * eps)
                grad_fn_eps = torch.autograd.grad(fn_eps, x)[0].detach()
            # x.requires_grad_(False)
            return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))
        
        def ode_func(t, zp):
            z = zp[:, :-1].reshape(epsilon.shape)
            drift = drift_fn(z, t)
            logp_grad = div_fn(z, t, epsilon)
            return torch.concat([drift.flatten(1), logp_grad[:, None]], axis=-1)
                
        return ode_func
    
    def _compute_done(self, t):
        return self._direction * (t - self._sde.T) >= 0

    def _step(self, tensordict):
        
        if self._use_opt:
            return self._opt_step(tensordict)
        
        if self._compute_opt:
            opt_td = self._opt_step(tensordict.clone())    

        h = tensordict["action"] * self._direction
        t = torch.clip(tensordict["t"] + h, self._eps, self._sde.T)
        h = t - tensordict["t"]
        
        z = tensordict["z"]
        p = tensordict["p"]
        
        ode_func = self._get_ode_func(tensordict["epsilon"])
        y_new, f_new, error = self._rk_step(ode_func, t, torch.concat([z.flatten(1), p[:, None]], dim=-1), tensordict["f"], h)
        z = y_new[:, :-1].reshape(z.shape)
        p = y_new[:, -1]
        
        done = self._compute_done(t)
    
        reward = p.clone()
        reward += torch.where(done, 100, 0)
        reward += torch.where(torch.abs(error).sum(dim=-1) < self._error_tol, 1, 0)
        
        h_abs = opt_td["h_abs"] if self._compute_opt or self._use_opt else None 
        opt_action = opt_td["opt_action"] if self._compute_opt else None
        
        return self._create_tensordict(
            done=done, z=z, p=p, t=t, f=f_new, reward=reward, \
                epsilon=tensordict["epsilon"], h_abs=h_abs, opt_action=opt_action)
         
    def _create_tensordict(self, **kwargs):
        return TensorDict({k: v for k, v in kwargs.items() if v is not None}, self.batch_size)

    def _select_initial_step(self, z, p, f0, epsilon):
        y0 = torch.concat([z.flatten(1), p[:, None]], dim=-1)
        scale = self._atol + torch.abs(y0) * self._rtol
        interval_length = abs(self._sde.T - self._eps)
        
        d0 = norm(y0 / scale)
        d1 = norm(f0 / scale)
        h0 = torch.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)

        # Check t0+h0*direction doesn't take us beyond t_bound
        h0 = min(h0, interval_length)
        y1 = y0 + h0[:, None] * self._direction * f0
        ode_func = self._get_ode_func(epsilon)
        f1 = ode_func(self._sde.T + h0 * self._direction, y1)
        d2 = norm((f1 - f0) / scale) / h0

        h1 = torch.where(
            (d1 <= 1e-15) & (d2 <= 1e-15), 
            max(h0 * 1e-3, 1e-6), 
            (0.01 / max(d1, d2)) ** (1 / (self._error_estimator_order + 1))
        )
        
        return min(100 * h0, h1, interval_length, self._max_step)

    def _opt_step(self, tensordict):
        """"see https://github.com/scipy/scipy/blob/v1.14.1/scipy/integrate/_ivp/rk.py#L111"""
        max_step = self._max_step
        rtol = self._rtol
        atol = self._atol
        t = tensordict["t"]
        f = tensordict["f"]
        h_abs = tensordict["h_abs"]

        min_step = 10 * torch.abs(torch.nextafter(t, self._direction * torch.ones_like(t)) - t)
        h_abs = torch.clip(h_abs, min_step, max_step * torch.ones_like(h_abs))
        
        step_accepted = torch.zeros(self.batch_size, dtype=bool, device=self.device)

        ode_func = self._get_ode_func(tensordict["epsilon"])
        y = torch.concat([tensordict["z"].flatten(1), tensordict["p"][:, None]], dim=-1)
        
        while not step_accepted.any():
            
            h = h_abs * self._direction
            t_new = t + h
            t_new = torch.where(
                self._direction * (t_new - self._sde.T) > 0,
                self._sde.T,
                t_new
            )

            h = t_new - t
            _h_abs = h.abs()

            y_new, f_new, error = self._rk_step(ode_func, t, y, f, h)
            scale = atol + min(y.abs(), y_new.abs()) * rtol
            error_norm = norm(error / scale)
            
            # new step accepted
            _step_accepted = ~step_accepted & (error_norm < 1)
            factor = torch.where(
                error_norm == 0, 
                MAX_FACTOR, 
                min(error_norm, MAX_FACTOR, SAFETY * error_norm ** self._error_exponent)
            )
            factor = torch.where(
                _step_accepted, 
                min(factor, 1), 
                max(SAFETY * error_norm ** self._error_exponent, MIN_FACTOR)
            )
            _h_abs = torch.where(~step_accepted, _h_abs * factor, _h_abs)
            
            new_accepted = ~step_accepted & _step_accepted
            h_abs = torch.where(new_accepted, _h_abs, h_abs)
            t = torch.where(new_accepted, t_new, t)
            y = torch.where(new_accepted[:, None], y_new, y)
            f = torch.where(new_accepted[:, None], f_new, f)
            step_accepted = step_accepted | _step_accepted
            
        done = self._compute_done(t)
        z = y[:, :-1].reshape(tensordict["z"].shape)
        p = y[:, -1]
        reward = -(tensordict["action"] - t).sqrt()

        return self._create_tensordict(done=done, z=z, p=p, t=t, f=f_new, reward=reward, \
                                       epsilon=tensordict["epsilon"], h_abs=h_abs)

class RK45Env(Env):
    _C = torch.tensor([0, 1/5, 3/10, 4/5, 8/9, 1])
    _A = torch.tensor([
        [0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
    ])
    _B = torch.tensor([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])
    _E = torch.tensor([-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525,
                  1/40])
    # Corresponds to the optimum value of c_6 from [2]_.
    _P = torch.tensor([
        [1, -8048581381/2820520608, 8663915743/2820520608,
         -12715105075/11282082432],
        [0, 0, 0, 0],
        [0, 131558114200/32700410799, -68118460800/10900136933,
         87487479700/32700410799],
        [0, -1754552775/470086768, 14199869525/1410260304,
         -10690763975/1880347072],
        [0, 127303824393/49829197408, -318862633887/49829197408,
         701980252875 / 199316789632],
        [0, -282668133/205662961, 2019193451/616988883, -1453857185/822651844],
        [0, 40617522/29380423, -110615467/29380423, 69997945/29380423]])
    _n_stages = 6
    _error_estimator_order = 4

if __name__ == "__main__":
    from configs.ve import cifar10_ncsnpp_continuous as configs
    from sde_lib import VESDE
    from models import ncsnpp
    
    ckpt_filename = "exp/ve/cifar10_ncsnpp_continuous/checkpoint_24.pth"
    config = configs.get_config()  
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    score_model = mutils.create_model(config)
    
    env = RK45Env(
        sde=sde, 
        model=score_model, 
        env_num=32, 
        device=torch.device("cuda"), 
        use_opt=True, 
        compute_opt=False
    )
    rollout = env.rollout(1, break_when_any_done=False)
    print(rollout)