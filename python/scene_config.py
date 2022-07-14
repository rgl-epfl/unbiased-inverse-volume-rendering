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


add_scene_config(
    'janga-smoke',
    fname='janga-smoke/janga-smoke.xml',
    param_keys=['medium1.sigma_t.data', 'medium1.albedo.data', 'medium1.emission.data'],
    normal_scene_vars={
        'resx': 720,
        'resy': 620,
        'envmap_filename': 'textures/gamrig_2k.hdr',
        'majorant_resolution_factor': 8,
    },
    ref_scene_vars={
        'resx': 720,
        'resy': 620,
        'medium_filename': 'volumes/janga-smoke-264-136-136.vol',
        'albedo_filename': 'volumes/albedo-noise-256-128-128.vol',
        'emission_filename': 'volumes/albedo-noise-256-128-128.vol',
        'envmap_filename': 'textures/gamrig_2k.hdr',
        'majorant_resolution_factor': 8,
    },
    sensors=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,54,55,56,57,58,59,60,61,62,63,],max_depth=64,
    start_from_value={
        'medium1.sigma_t.data': 0.04 / 20,
        'medium1.albedo.data': 0.6,
        'medium1.emission.data': 0.1 / 20,
    },
)
