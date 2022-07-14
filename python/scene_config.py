from copy import deepcopy
from dataclasses import dataclass
import os
from os.path import join, realpath
from typing import List, Dict

from constants import SCENE_DIR, OUTPUT_DIR

@dataclass
class SceneConfig():
    """Holds configuration options related to each scene."""
    name: str
    fname: str
    param_keys: List[str]
    normal_scene_vars: Dict
    sensors: List[int]
    start_from_value: Dict

    max_depth: int = 64
    ref_scene_vars: Dict = None
    references: str = None
    ref_spp: int = 8192
    ref_integrator: str = 'volpathsimple'
    preview_sensors: List[int] = None

    # Upper bound on the density, this prevents very large render times.
    # Its value should be chosen based on the scene scale.
    max_density: float = 250


    def __post_init__(self):
        super().__init__()
        self.fname = realpath(join(SCENE_DIR, self.fname))
        if not os.path.isfile(self.fname):
            raise ValueError(f'Scene file not found: {self.fname}')

        if self.ref_scene_vars is None:
            self.ref_scene_vars = deepcopy(self.normal_scene_vars)

        for k in self.param_keys:
            if k not in self.start_from_value:
                raise ValueError(f'Parameter "{k}" will be optimized but was not given an initial value in `start_from_value`')

        if self.references is None:
            self.references = join(OUTPUT_DIR, 'references', self.name)
        elif not os.path.isdir(self.references):
            self.references = join(OUTPUT_DIR, 'references', self.references)

        if not self.preview_sensors:
            self.preview_sensors = [self.sensors[i] for i in range(min(5, len(self.sensors)))]


_SCENE_CONFIGS = {}
def add_scene_config(name, **kwargs):
    assert name not in _SCENE_CONFIGS, f'Duplicate scene config name: {name}'
    _SCENE_CONFIGS[name] = SceneConfig(name, **kwargs)

def get_scene_config(name):
    if isinstance(name, SceneConfig):
        return deepcopy(name)
    return deepcopy(_SCENE_CONFIGS[name])
