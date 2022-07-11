import os
from os.path import realpath, join

SCRIPT_DIR = realpath(os.path.dirname(__file__))
OUTPUT_DIR = realpath(join(SCRIPT_DIR, '../outputs'))

SCENE_DIR = realpath(join(SCRIPT_DIR, '../scenes'))
RENDER_DIR = realpath(join(OUTPUT_DIR, 'renders'))

FIGURE_DIR = join(OUTPUT_DIR, '00-figures')
