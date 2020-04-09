import os
from os.path import join

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(join(_script_dir, '..'))

DATA = join(_repo_root, 'data')
DATA_LOWER_LETTER_TAIL = join(DATA, 'lower_letter_tail')
DATA_UPPER_LETTER_HEAD = join(DATA, 'upper_letter_head')
DATA_UPPER_LETTER_TAIL = join(DATA, 'upper_letter_tail')
DATA_UPPER_WORDS_HEAD = join(DATA, 'upper_words_head')

OUTPUT = join(_repo_root, 'output')

SAVED_MODEL = join(_repo_root, 'saved_model')
