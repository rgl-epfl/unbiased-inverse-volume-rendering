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
    # Directory name where to find the reference images.
    # Useful if multiple configurations share the same
    # set of ref images.
    references: str = None
    ref_spp: int = 8192
    ref_integrator: str = 'volpathsimple'
    ref_fname: str = None
    ref_scene_vars: Dict = None
    preview_sensors: List[int] = None

    # Upper bound on the density, this prevents very large render times.
    # Its value should be chosen based on the scene scale.
    max_density: float = 250
    # Determines the resolution of the majorant supergrid.
    # Will be adjusted at runtime if upsampling is enabled.
    # The supergrid can be disabled by setting the factor to 0.
    majorant_resolution_factor: int = 8

    # Per-parameter factors to apply to the learning rate
    param_lr_factors: Dict = None

    def __post_init__(self):
        super().__init__()
        self.fname = realpath(join(SCENE_DIR, self.fname))
        if not os.path.isfile(self.fname):
            raise ValueError(f'Scene file not found: {self.fname}')
        if self.ref_fname:
            self.ref_fname = realpath(join(SCENE_DIR, self.ref_fname))
            if not os.path.isfile(self.ref_fname):
                raise ValueError(f'Reference scene file not found: {self.ref_fname}')

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
            self.preview_sensors = self.sensors[0]
            # self.preview_sensors = [self.sensors[i] for i in range(min(5, len(self.sensors)))]

        if not self.param_lr_factors:
            self.param_lr_factors = {}
            for k in self.param_keys:
                if '.albedo.' in k:
                    self.param_lr_factors[k] = 2.0


_SCENE_CONFIGS = {}
_SCENE_CONFIG_KWARGS = {}
def add_scene_config(name, **kwargs):
    assert name not in _SCENE_CONFIGS, f'Duplicate scene config name: {name}'
    _SCENE_CONFIGS[name] = SceneConfig(name, **kwargs)
    _SCENE_CONFIG_KWARGS[name] = deepcopy(kwargs)

def add_scene_config_variant(name, base, **kwargs):
    assert name not in _SCENE_CONFIGS, f'Duplicate scene config name: {name}'
    all_kwargs = deepcopy(_SCENE_CONFIG_KWARGS[base])
    all_kwargs.update(deepcopy(kwargs))
    _SCENE_CONFIGS[name] = SceneConfig(name, **all_kwargs)
    _SCENE_CONFIG_KWARGS[name] = all_kwargs


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
    sensors=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,54,55,56,57,58,59,60,61,62,63,],
    max_depth=64,
    start_from_value={
        'medium1.sigma_t.data': 0.04 / 20,
        'medium1.albedo.data': 0.6,
        'medium1.emission.data': 0.1 / 20,
    },
)

add_scene_config_variant(
    'janga-smoke-from-nerf',
    base='janga-smoke',
    references='janga-smoke',
    normal_scene_vars={
        'resx': 720,
        'resy': 620,
        'medium_filename': join(OUTPUT_DIR, 'janga-smoke-sn64', 'nerf', 'params', 'final-medium1_sigma_t.vol'),
        'albedo_filename': join(OUTPUT_DIR, 'janga-smoke-sn64', 'nerf', 'params', 'final-medium1_albedo.vol'),
        'emission_filename': join(OUTPUT_DIR, 'janga-smoke-sn64', 'nerf', 'params', 'final-medium1_emission.vol'),
        'envmap_filename': 'textures/gamrig_2k.hdr',
        'majorant_resolution_factor': 8,
    },
    start_from_value={
        'medium1.sigma_t.data': None,
        'medium1.albedo.data': 0.6,
        'medium1.emission.data': None,
    },
    preview_sensors=[0,]
)

# ----------

add_scene_config(
    'dust-devil',
    fname='dust-devil/dust-devil.xml',
    param_keys=['medium1.sigma_t.data', 'medium1.albedo.data', 'medium1.emission.data'],
    normal_scene_vars={
        'resx': 620,
        'resy': 720,
        'envmap_filename': 'textures/kloofendal_38d_partly_cloudy_4k.exr',
        'majorant_resolution_factor': 8,
    },
    ref_scene_vars={
        'resx': 620,
        'resy': 720,
        'medium_filename': 'volumes/embergen_dust_devil_tornado_a_50-256-256-256.vol',
        'albedo_filename': 'volumes/albedo-constant-sand-256-256-256.vol',
        'emission_filename': 'volumes/albedo-constant-sand-256-256-256.vol',
        'envmap_filename': 'textures/kloofendal_38d_partly_cloudy_4k.exr',
        'majorant_resolution_factor': 8,
    },
    sensors=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,54,55,56,57,58,59,60,61,62,63,],
    max_depth=64,
    start_from_value={
        'medium1.sigma_t.data': 0.04 / 100,
        'medium1.albedo.data': 0.6,
        'medium1.emission.data': 0.1 / 100,
    },
    preview_sensors=[0,]
)

add_scene_config_variant(
    'dust-devil-from-nerf',
    base='dust-devil',
    references='dust-devil',
    normal_scene_vars={
        'resx': 620,
        'resy': 720,
        'medium_filename': join(OUTPUT_DIR, 'dust-devil-sn64', 'nerf', 'params', 'final-medium1_sigma_t.vol'),
        'albedo_filename': join(OUTPUT_DIR, 'dust-devil-sn64', 'nerf', 'params', 'final-medium1_albedo.vol'),
        'emission_filename': join(OUTPUT_DIR, 'dust-devil-sn64', 'nerf', 'params', 'final-medium1_emission.vol'),
        'envmap_filename': 'textures/kloofendal_38d_partly_cloudy_4k.exr',
        'majorant_resolution_factor': 8,
    },
    start_from_value={
        'medium1.sigma_t.data': None,
        'medium1.albedo.data': 0.6,
        'medium1.emission.data': None,
    },
    param_lr_factors={
        'medium1.albedo.data': 100,
    },
    preview_sensors=[0,]
)

# ----------

add_scene_config(
    'astronaut-rotated',
    fname='astronaut-rotated/astronaut-rotated.xml',
    ref_fname='astronaut-rotated/astronaut-rotated-ref.xml',
    ref_integrator='path',
    param_keys=['medium1.sigma_t.data', 'medium1.albedo.data', 'medium1.emission.data'],
    normal_scene_vars={
        'resx': 720,
        'resy': 1080,
        'medium_filename': 'volumes/sigma_t-constant-sand-256-256-256.vol',
        'albedo_filename': 'volumes/albedo-constant-sand-256-256-256.vol',
        'emission_filename': 'volumes/albedo-constant-sand-256-256-256.vol',
        'envmap_filename': 'textures/skylit_garage_4k.exr',
        'majorant_resolution_factor': 8,
    },
    ref_scene_vars={
        'resx': 720,
        'resy': 1080,
        'envmap_filename': 'textures/skylit_garage_4k.exr',
    },
    sensors=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,],
    max_depth=64,
    start_from_value={
        'medium1.sigma_t.data': 0.04,
        'medium1.albedo.data': 0.6,
        'medium1.emission.data': 0.1,
    },
    preview_sensors=[0,]
)

add_scene_config_variant(
    'astronaut-rotated-from-nerf',
    base='astronaut-rotated',
    references='astronaut-rotated',
    normal_scene_vars={
        'resx': 720,
        'resy': 1080,
        'medium_filename': join(OUTPUT_DIR, 'astronaut-rotated-sn64', 'nerf', 'params', 'final-medium1_sigma_t.vol'),
        'albedo_filename': join(OUTPUT_DIR, 'astronaut-rotated-sn64', 'nerf', 'params', 'final-medium1_albedo.vol'),
        'emission_filename': join(OUTPUT_DIR, 'astronaut-rotated-sn64', 'nerf', 'params', 'final-medium1_emission.vol'),
        'envmap_filename': 'textures/skylit_garage_4k.exr',
        'majorant_resolution_factor': 8,
    },
    start_from_value={
        'medium1.sigma_t.data': None,
        'medium1.albedo.data': 0.6,
        'medium1.emission.data': None,
    },
    preview_sensors=[0,]
)

# ----------

add_scene_config(
    'rover',
    fname='rover/rover.xml',
    ref_fname='rover/rover-ref.xml',
    ref_integrator='path',
    param_keys=['medium1.sigma_t.data', 'medium1.albedo.data', 'medium1.emission.data'],
    normal_scene_vars={
        'resx': 860,
        'resy': 720,
        'medium_filename': 'volumes/sigma_t-constant-sand-256-256-256.vol',
        'albedo_filename': 'volumes/albedo-constant-sand-256-256-256.vol',
        'emission_filename': 'volumes/albedo-constant-sand-256-256-256.vol',
        'envmap_filename': 'textures/gamrig_2k.hdr',
        'majorant_resolution_factor': 8,
    },
    ref_scene_vars={
        'resx': 860,
        'resy': 720,
        'envmap_filename': 'textures/gamrig_2k.hdr',
    },
    sensors=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,54,55,56,57,58,59,60,61,62,63,],
    max_depth=64,
    start_from_value={
        'medium1.sigma_t.data': 0.04,
        'medium1.albedo.data': 0.6,
        'medium1.emission.data': 0.1,
    },
    preview_sensors=[0,]
)

add_scene_config_variant(
    'rover-from-nerf',
    base='rover',
    references='rover',
    normal_scene_vars={
        'resx': 860,
        'resy': 720,
        'medium_filename': join(OUTPUT_DIR, 'rover-sn64', 'nerf', 'params', 'final-medium1_sigma_t.vol'),
        'albedo_filename': join(OUTPUT_DIR, 'rover-sn64', 'nerf', 'params', 'final-medium1_albedo.vol'),
        'emission_filename': join(OUTPUT_DIR, 'rover-sn64', 'nerf', 'params', 'final-medium1_emission.vol'),
        'envmap_filename': 'textures/gamrig_2k.hdr',
        'majorant_resolution_factor': 8,
    },
    start_from_value={
        'medium1.sigma_t.data': None,
        'medium1.albedo.data': 0.6,
        'medium1.emission.data': None,
    },
    preview_sensors=[0,]
)

# ----------

add_scene_config(
    'tree-2',
    fname='tree-2/tree-2.xml',
    ref_fname='tree-2/tree-2-ref.xml',
    ref_integrator='path',
    param_keys=['medium1.sigma_t.data', 'medium1.albedo.data', 'medium1.emission.data'],
    normal_scene_vars={
        'resx': 720,
        'resy': 900,
        'medium_filename': 'volumes/sigma_t-constant-sand-256-256-256.vol',
        'albedo_filename': 'volumes/albedo-constant-sand-256-256-256.vol',
        'emission_filename': 'volumes/albedo-constant-sand-256-256-256.vol',
        'envmap_filename': 'textures/round_platform_2k.hdr',
        'majorant_resolution_factor': 8,
    },
    ref_scene_vars={
        'resx': 720,
        'resy': 900,
        'envmap_filename': 'textures/round_platform_2k.hdr',
    },
    sensors=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,],
    max_depth=64,
    start_from_value={
        'medium1.sigma_t.data': 0.04 / 2,
        'medium1.albedo.data': 0.6,
        'medium1.emission.data': 0.1 / 2,
    },
    preview_sensors=[0,]
)

add_scene_config_variant(
    'tree-2-from-nerf',
    base='tree-2',
    references='tree-2',
    normal_scene_vars={
        'resx': 720,
        'resy': 900,
        'medium_filename': join(OUTPUT_DIR, 'tree-2-sn64', 'nerf', 'params', 'final-medium1_sigma_t.vol'),
        'albedo_filename': join(OUTPUT_DIR, 'tree-2-sn64', 'nerf', 'params', 'final-medium1_albedo.vol'),
        'emission_filename': join(OUTPUT_DIR, 'tree-2-sn64', 'nerf', 'params', 'final-medium1_emission.vol'),
        'envmap_filename': 'textures/round_platform_2k.hdr',
        'majorant_resolution_factor': 8,
    },
    start_from_value={
        'medium1.sigma_t.data': None,
        'medium1.albedo.data': 0.6,
        'medium1.emission.data': None,
    },
    preview_sensors=[0,]
)
