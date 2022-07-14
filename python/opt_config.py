from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, List, Dict

import mitsuba as mi

import losses


@dataclass
class OptimizationConfig():
    """Holds configuration options related to a particular optimization run."""
    name: str
    spp: int
    n_iter: int
    lr: float

    primal_spp_factor: int = 64
    batch_size: int = None
    lr_schedule: Callable = None
    upsample: List[float] = None

    base_seed: int = 988378

    render_initial: bool = True
    render_final: bool = True
    preview_stride: int = 100

    checkpoint_initial: bool = True
    checkpoint_final: bool = True
    checkpoint_stride: int = 1000


    preview_spp: int = None
    opt_type: Callable = 'adam'
    opt_args: Dict = None
    loss: Callable = losses.l1

    def __post_init__(self):
        if self.upsample:
            self.upsample_at = {}
            for t in self.upsample:
                assert t >= 0 and t <= 1
                self.upsample_at.add(int(t * self.n_iter))
            print('Upsampling schedule:', self.upsample_at)

    def optimizer(self, params):
        opt_type = {'sgd': mi.ad.SGD, 'adam': mi.ad.Adam}[self.opt_type]
        return opt_type(lr=self.lr, params=params, **(self.opt_args or {}))

    def learning_rate(self, it_i):
        if self.lr_schedule is None:
            return self.lr

        t = it_i / (self.n_iter - 1)
        # TODO: compute LR according to schedule
        raise NotImplementedError('LR schedule')


    def should_upsample(self, it_i):
        if not self.upsampling_schedule:
            return False
        return it_i in self.upsample_at



@dataclass
class IntegratorConfig:
    name: str
    pretty_name: str
    params: Dict

    uses_fd: bool = False
    fd_epsilon: float = None
    fd_spp_multiplier: int = 16

    def __post_init__(self):
        if self.uses_fd:
            assert self.fd_epsilon is not None

    def create(self, **kwargs):
        assert 'max_depth' in kwargs
        d = deepcopy(self.params)
        d.update(kwargs)

        assert d['max_depth'] >= 0
        # TODO: add support for Russian Roulette
        assert 'rr_depth' not in kwargs
        d['rr_depth'] = d['max_depth'] + 1000

        return mi.load_dict(d)



_INTEGRATOR_CONFIGS = {}
def add_int_config(name, **kwargs):
    assert name not in _INTEGRATOR_CONFIGS, f'Duplicate integrator config name: {name}'
    _INTEGRATOR_CONFIGS[name] = IntegratorConfig(name, **kwargs)

def get_int_config(name):
    if isinstance(name, IntegratorConfig):
        return deepcopy(name)
    return deepcopy(_INTEGRATOR_CONFIGS[name])

