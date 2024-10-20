from typing import Optional, Tuple, List
import numpy as np
import torch.nn as nn

from sde_lib import SDE
from torchrl.envs import EnvBase
from torchrl.data import BoundedTensorSpec, CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs.utils import step_mdp 

from tensordict import TensorDict, TensorDictBase
import torch
import models.utils as mutils

# Multiply steps computed from asymptotic behaviour of errors by this.
SAFETY = 0.9

MIN_FACTOR = 0.2  # Minimum allowed decrease in a step size.
MAX_FACTOR = 10  # Maximum allowed increase in a step size.

def norm(x: torch.Tensor):
    assert x.dim() == 2, "x must be 2D tensor"
    return torch.linalg.norm(x, dim=-1) / x.shape[1] ** 0.5

def min(input: torch.tensor, *others):
    for other in others:
        input = input.clip(max=other)
    return input

def max(input: torch.tensor, *others):
    for other in others:
        input = input.clip(min=other)
    return input

def enable_dtype(dtype=torch.float64): 
    target = torch.float64 if dtype == torch.float32 else torch.float32
    
    def convert_to_dtype(x, _dtype):
        if isinstance(x, torch.Tensor) and \
            ((x.dtype == torch.float32) or (x.dtype == torch.float64)):
            x = x.to(_dtype)
        elif isinstance(x, TensorDict) or isinstance(x, dict):
            for k, v in x.items():
                x[k] = convert_to_dtype(v, _dtype)
        elif isinstance(x, tuple):
            x = tuple(convert_to_dtype(arg, _dtype) for arg in x)
        return x
    
    def outter(func):
        def wrapper(*args, **kwargs):        
            _args = [convert_to_dtype(arg, dtype) for arg in args]
            _kwargs = {k: convert_to_dtype(v, dtype) for k, v in kwargs.items()}
            out = func(*_args, **_kwargs)
            return convert_to_dtype(out, target)
        return wrapper
    
    return outter

class Env(EnvBase):
    """
    Env for solving the reverse-time SDE.

    Args:
        use_opt: If `True`, use the optimal step size computed by the ODE solver.
    """
    
    _A: List = NotImplemented
    _B: List = NotImplemented
    _C: List = NotImplemented
    _E: List = NotImplemented
    _P: List = NotImplemented
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
        compute_logp: bool=False,
        max_step: float=np.inf,
        rtol: float=1e-5,
        atol: float=1e-5,
    ):
        super().__init__(batch_size=(env_num, ), allow_done_after_reset=True, device=device)
        self._sde = sde
        self._eps = eps
        self._model = model.eval()
        self._model.requires_grad_(False)
        self._error_tol = error_tol
        self._direction = np.sign(self._eps - self._sde.T) if self._sde.T != self._eps else 1
        self._A = torch.tensor(self._A, dtype=torch.float64, device=device)
        self._B = torch.tensor(self._B, dtype=torch.float64, device=device)
        self._C = torch.tensor(self._C, dtype=torch.float64, device=device)
        self._E = torch.tensor(self._E, dtype=torch.float64, device=device)
        self._P = torch.tensor(self._P, dtype=torch.float64, device=device)
        
        self._max_step = max_step
        self._rtol = rtol
        self._atol = atol
        self._compute_logp = compute_logp

        p_dim = 1 if compute_logp else 0
        self._n = 3 * 32 * 32  + p_dim
        
        self.action_spec = BoundedTensorSpec(low=self._eps, high=self._sde.T, shape=(env_num,))
        self.done_spec = BoundedTensorSpec(low=False, high=True, shape=(env_num,), dtype=torch.bool)
        
        self.state_spec = CompositeSpec(
            z=UnboundedContinuousTensorSpec(shape=(env_num, 3, 32, 32)),
            epsilon=UnboundedContinuousTensorSpec(shape=(env_num, 3, 32, 32)),
            t=UnboundedContinuousTensorSpec(shape=(env_num,)),
            f=UnboundedContinuousTensorSpec(shape=(env_num, self._n)),
            nfev=UnboundedContinuousTensorSpec(shape=(env_num,)),
            h_abs=UnboundedContinuousTensorSpec(shape=(env_num,)), # opt_action
            step_rejected=BoundedTensorSpec(low=False, high=True, shape=(env_num,)),
            error_norm=UnboundedContinuousTensorSpec(shape=(env_num,)),
            shape=(env_num,)
        )

        if compute_logp:
            self.state_spec.update({
                "p": UnboundedContinuousTensorSpec(shape=(env_num,))
            })
        self._error_exponent = -1 / (self._error_estimator_order + 1)
        self.observation_spec = self.state_spec.clone()
    
    def _split_zp(self, zp):
        if self._compute_logp:
            z = zp[:, :-1].reshape((-1, 3, 32, 32))
            p = zp[:, -1]
            return z, p
        return zp.reshape((-1, 3, 32, 32)), None
    
    def _merge_zp(self, z, p=None):
        if self._compute_logp:
            return torch.concat([z.flatten(1), p[:, None]], dim=-1)
        return z.flatten(1)

    def _set_seed(self, seed: Optional[int]):
        self.rng = torch.manual_seed(seed)

    def _reset(self, td=None):
        z = self._sde.prior_sampling((*self.batch_size, 3, 32, 32)).to(self.device)
        init_td = self._create_td(
            z=z,
            t=torch.ones(self.batch_size, device=self.device) * self._sde.T,
            done=torch.zeros(self.batch_size, dtype=torch.bool, device=self.device),
            nfev=torch.zeros(self.batch_size, device=self.device),
            epsilon=torch.randn_like(z),             
            p=self._sde.prior_logp(z).to(self.device) if self._compute_logp else None,
            step_rejected=torch.zeros(self.batch_size, dtype=torch.bool, device=self.device),
            f=torch.zeros(*self.batch_size, self._n, device=self.device),
            h_abs=torch.zeros(self.batch_size, device=self.device),
            error_norm=torch.zeros(self.batch_size, device=self.device),
        )
        done = ~init_td["done"] if td is None else td["done"]
        new_td = init_td[done]
         
        ode_func = self._get_ode_func(new_td)
        new_td["f"] = ode_func(new_td["t"], self._merge_zp(new_td["z"], new_td.get("p", None))).to(torch.float32)
        new_td["h_abs"] = self._select_initial_step(ode_func, new_td["z"], new_td.get("p", None), new_td["f"])
        
        td = init_td if td is None else td
        td[done] = new_td.clone()
        return td 
    
    def _rk_step(self, ode_func, t, y, f, h)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """see https://github.com/scipy/scipy/blob/v1.14.1/scipy/integrate/_ivp/rk.py#L14"""
        k = torch.zeros((t.shape[0], self._n_stages + 1, self._n), dtype=y.dtype, device=y.device)
        k[:, 0] = f
        for s, (a, c) in enumerate(zip(self._A[1:], self._C[1:]), start=1):
            dy = torch.matmul(k[:, :s].permute(0, 2, 1).contiguous(), a[:s]) * h[:, None]
            k[:, s] = ode_func(t + c * h, y + dy)

        y_new = y + h[:, None] * torch.matmul(k[:, :-1].permute(0, 2, 1).contiguous(), self._B)
        f_new = ode_func(t + h, y_new)

        k[:, -1] = f_new
        error = torch.matmul(k.permute(0, 2, 1).contiguous(), self._E) * h[:, None]
        
        return y_new, f_new, error

    def _get_ode_func(self, td):
        def drift_fn(x, t):
            """The drift function of the reverse-time SDE."""
            score_fn = mutils.get_score_fn(self._sde, self._model, train=False, continuous=True)
            score_fn = enable_dtype(torch.float32)(score_fn)
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
            td["nfev"] += 1
            z, _ = self._split_zp(zp)
            drift = drift_fn(z, t)
            logp_grad = div_fn(z, t, td["epsilon"]) if self._compute_logp else None
            return self._merge_zp(drift, logp_grad)
                
        return ode_func
    
    def _compute_done(self, t):
        return (self._direction * (t - self._eps) >= 0).to(torch.bool)

    @enable_dtype()
    def _step(self, tensordict):
        use_opt =  tensordict.get("action", None) is None
        if use_opt:
            _tensordict = tensordict.clone()
            # try to avoid creating new tensors when copy
            _tensordict["reward"] = torch.zeros(_tensordict.batch_size, device=self.device, dtype=torch.float64)
            tensordict = tensordict[~tensordict["done"]]

        """"see https://github.com/scipy/scipy/blob/v1.14.1/scipy/integrate/_ivp/rk.py#L111"""
        max_step = self._max_step
        rtol = self._rtol
        atol = self._atol
        t = tensordict["t"]
        f = tensordict["f"]
        h_abs = tensordict["h_abs"]
        step_rejected = tensordict["step_rejected"]

        min_step = 10 * torch.abs(torch.nextafter(t, self._direction * torch.ones_like(t)) - t)
        h_abs = torch.clip(h_abs, min_step, max_step * torch.ones_like(h_abs))
        h_abs = torch.where(~step_rejected, h_abs, tensordict["h_abs"])
       
        ode_func = self._get_ode_func(tensordict)
        y = self._merge_zp(tensordict["z"], tensordict.get("p", None))
            
        h = h_abs * self._direction
        t_new = t + h
        t_new = torch.where(
            self._direction * (t_new - self._eps) > 0,
            self._eps,
            t_new
        )

        h = t_new - t
        _h = h if use_opt else tensordict["action"].squeeze(-1)
        y_new, f_new, error = self._rk_step(ode_func, t, y, f, _h)
        scale = atol + max(y.abs(), y_new.abs()) * rtol
        error_norm = norm((error / scale))
        step_accepted = error_norm < 1

        factor = torch.ones_like(error_norm) * MAX_FACTOR
        factor = torch.where(
            step_accepted & (error_norm != 0), 
            min(SAFETY * error_norm ** self._error_exponent, MAX_FACTOR),
            factor
        )
        factor = torch.where(
            step_accepted & step_rejected,
            min(factor, 1),
            factor
        )
        factor = torch.where(
            ~step_accepted,
            max(SAFETY * error_norm ** self._error_exponent, MIN_FACTOR),
            factor
        )
            
        t = torch.where(step_accepted, t_new, t)
        y = torch.where(step_accepted[:, None], y_new, y)
        f = torch.where(step_accepted[:, None], f_new, f)
        
        h_abs = h.abs() * factor 
        step_rejected = ~step_accepted

        done = self._compute_done(t)
        z, p = self._split_zp(y)
        reward = -error_norm.clone()

        tensordict = self._create_td(done=done, \
            z=z, p=p, t=t, f=f, reward=reward, nfev=tensordict["nfev"], error_norm=error_norm, \
                epsilon=tensordict["epsilon"], h_abs=h_abs, step_rejected=step_rejected)

        if use_opt:
            _tensordict[~_tensordict["done"]] = tensordict
        else:
            _tensordict = tensordict
        
        return _tensordict
         
    def _create_td(self, **kwargs):
        batch_size = None
        out = {}
        for k, v in kwargs.items():
            if v is None:
                continue
            if batch_size is None:
                batch_size = v.shape[0]
            assert v.shape[0] == batch_size, f"expect {k} to have batch size {batch_size}, got {v.shape[0]}"
            out[k] = v

        return TensorDict(out, batch_size=batch_size)
    
    @enable_dtype()
    def _select_initial_step(self, ode_func, z, p, f0):
        y0 = self._merge_zp(z, p)
        scale = self._atol + torch.abs(y0) * self._rtol
        interval_length = abs(self._sde.T - self._eps)
        
        d0 = norm(y0 / scale)
        d1 = norm(f0 / scale)
        h0 = torch.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / d1)

        # Check t0+h0*direction doesn't take us beyond t_bound
        h0 = min(h0, interval_length)
        y1 = y0 + h0[:, None] * self._direction * f0
        f1 = ode_func(self._sde.T + h0 * self._direction, y1)
        d2 = norm((f1 - f0) / scale) / h0

        h1 = torch.where(
            (d1 <= 1e-15) & (d2 <= 1e-15), 
            max(h0 * 1e-3, 1e-6), 
            (0.01 / max(d1, d2)) ** (1 / (self._error_estimator_order + 1))
        )
        
        return min(100 * h0, h1, interval_length, self._max_step)

    def sample(self, callback=None):
        td = self.reset()
        i = 0
        while not td["done"].all():
            i += 1
            td = self.step(td)
            td = step_mdp(td)
            if callback is not None:
                callback(self, td)
        return td

class RK45Env(Env):
    _C = [0, 1/5, 3/10, 4/5, 8/9, 1]
    _A = [
        [0, 0, 0, 0, 0],
        [1/5, 0, 0, 0, 0],
        [3/40, 9/40, 0, 0, 0],
        [44/45, -56/15, 32/9, 0, 0],
        [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
        [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
    ]
    _B = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
    _E = [-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525,
                  1/40]
    # Corresponds to the optimum value of c_6 from [2]_.
    _P = [
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
        [0, 40617522/29380423, -110615467/29380423, 69997945/29380423]]
    _n_stages = 6
    _error_estimator_order = 4

def create_env(config):
    from configs.ve import cifar10_ncsnpp_continuous as configs
    from sde_lib import VESDE
    from models import ncsnpp
    from models.ema import ExponentialMovingAverage
    import models.utils as mutils

    env_num = config.env_num
    ckpt_filename = "exp/ve/cifar10_ncsnpp_continuous/checkpoint_24.pth"
    config = configs.get_config()  
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    score_model = mutils.create_model(config)
    state_dict = torch.load(ckpt_filename, map_location="cuda")
    score_model.load_state_dict(state_dict["model"], strict=False)
    ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
    ema.load_state_dict(state_dict["ema"])
    ema.copy_to(score_model.parameters())
    env = RK45Env(
        env_num=env_num,
        sde=sde, 
        model=score_model,  
        device=torch.device("cuda"), 
        compute_logp=False,
    )
    return env

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from configs.ve import cifar10_ncsnpp_continuous as configs
    from sde_lib import VESDE
    from models import ncsnpp
    from env import RK45Env
    import models.utils as mutils
    import torch
    import numpy as np
    from models.ema import ExponentialMovingAverage

    ckpt_filename = "exp/ve/cifar10_ncsnpp_continuous/checkpoint_24.pth"
    config = configs.get_config()  
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    score_model = mutils.create_model(config)
    state_dict = torch.load(ckpt_filename, map_location="cuda")
    score_model.load_state_dict(state_dict["model"], strict=False)
    ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
    ema.load_state_dict(state_dict["ema"])
    ema.copy_to(score_model.parameters())
    env = RK45Env(
        sde=sde, 
        model=score_model, 
        env_num=32, 
        device=torch.device("cuda"), 
        compute_logp=False,
    )
    samples = env.sample()
    exit()
    import sampling
    import datasets
    shape = (1, 3, 32, 32)
    inverse_scaler = datasets.get_data_inverse_scaler(config)
    sampling_eps = 1e-5
    sampling_fn = sampling.get_ode_sampler(sde,                                        
                                       shape, 
                                       inverse_scaler,                                       
                                       denoise=True, 
                                       eps=sampling_eps,
                                       device=config.device)
    x, nfe = sampling_fn(score_model)