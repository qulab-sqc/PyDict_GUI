import numpy as np
from dpath.util import search as _dsearch
import dpath.util
import os
import json
import jsbeautifier


def dict_depth(obj: dict):
    """get the depth of dict
    https://www.geeksforgeeks.org/python-find-depth-of-a-dictionary/
    e.g., {'a': {'b': 1}} has depth of 2
    """
    if isinstance(obj, dict):
        return 1 + (max(map(dict_depth, obj.values()))
                    if obj else 0)
    return 0


def dsearch(obj, glob):
    """search key from dict by using `dpath.util.search`
    """
    res = _dsearch(obj, glob, yielded=True, separator='-')
    return dict(res)


def dget(obj, glob):
    """get dict value from dict by using `dpath.util.get`
    """
    return dpath.util.get(obj, glob, separator='-')


def dset(obj, glob, value):
    """set key from dict by using `dpath.util.set`
    Example:
        dset({'a':{'b': 1}}, 'a-b', 2)
    """
    dpath.util.set(obj, glob, value, separator='-')


def dsearch_nest(
    nest_dict, key, depth: int = None,
):
    """search key from nested dict with given depth
    """
    yielded = True
    separator = '-'
    if depth is None or depth < 1:
        res = _dsearch(
            nest_dict, f"**-{key}", yielded=yielded, separator=separator)
    else:
        res = _dsearch(
            nest_dict, "*-"*(depth-1)+str(key), yielded=yielded,
            separator=separator)
    return dict(res)


def save_json(obj: dict, _save_path: str):
    """save python dict as .json
    """
    _dir = os.path.abspath(os.path.split(_save_path)[0])

    if not os.path.exists(_dir):
        os.makedirs(_dir)

    opts = jsbeautifier.default_options()
    opts.indent_size = 2

    json_string = jsbeautifier.beautify(json.dumps(obj), opts)
    with open(_save_path, 'w', encoding='utf-8') as f:
        # json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write(json_string)
