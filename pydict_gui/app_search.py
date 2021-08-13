import numpy as np
from typing import Iterable, Tuple, Any
from functools import wraps
import os

import ipywidgets as widgets
from ipywidgets import Button, Layout
from IPython.core.display import display

from .utils import dget, dict_depth, dsearch_nest
from . import utils


def keyval_to_str(key, val):
    return f"{key} = {val}"


def set_dict_val(
    obj: dict, path: str, value: Any,
    vtype: str = 'str'
):
    """set dict value
    Args:
        vtype (str): 
        'str', 'lambda x'
    """
    val_0 = dget(obj, path)
    if vtype == "str":
        utils.dset(
            obj=obj, glob=path, value=value)
    elif vtype == "lambda x":
        def func(x):
            _func = eval("lambda x:"+str(value))
            return _func(x)
        val_new = func(val_0)
        utils.dset(
            obj=obj, glob=path, value=val_new)
    else:
        raise ValueError(f"invalid vtype: {vtype}")


class App:
    """dict search editor gui

    Args:
        obj (dict): evaluated dict
    """

    def __init__(self, obj: dict, save_path=None):
        if save_path is None:
            save_path = os.path.abspath('temp.json')
        self.save_path = os.path.abspath(save_path)
        self.obj = obj

        self.max_depth = dict_depth(self.obj)

        # max showed rows of searched result
        self.rows_search = 10
        self.output = widgets.Output()

        # Tab 1: search

        self.text_search = widgets.Text(
            value='*',
            placeholder='Type something',
            description='Search key:',
            disabled=False
        )

        self.select_paths = widgets.SelectMultiple(
            options=tuple(),
            rows=self.rows_search,
            description=r'Path:',
            disabled=False,
            layout=Layout(width='200%')
        )

        # just show path and value in the same box
        self.select_paths_vals = widgets.SelectMultiple(
            options=tuple(),
            rows=self.rows_search,
            description=r'Searched:',
            disabled=False,
            layout=Layout(width='auto')
        )

        self.slider_depth = widgets.SelectionSlider(
            options=(None,),
            value=None,
            description='depth:',
            disabled=False,
            readout=True
        )

        self.box_search = widgets.Box(
            children=[self.select_paths],
            layout=Layout(flex_flow='row', width='90%'))

        self.button_search = widgets.Button(
            description='search',
            disabled=False,
            button_style='success',
            # 'success', 'info', 'warning', 'danger' or ''
            tooltip='search dict',
            icon='check'
        )

        self.button_filter_sel = widgets.Button(
            description='extract',
            disabled=False,
            button_style='info',
            tooltip='filter selected',
            icon='filter'
        )

        self.button_unfilter_sel = widgets.Button(
            description='remove',
            disabled=False,
            button_style='info',
            tooltip='unfilter selected',
            icon='eraser'
        )

        self.box_buttons_search = widgets.Box(
            children=[
                self.button_search,
                self.button_filter_sel,
                self.button_unfilter_sel,
            ],
            layout=Layout(flex_flow='row', width='100%'))

        self.tab_search = widgets.VBox([
            self.text_search,
            self.slider_depth,
            self.select_paths,
            self.box_buttons_search,
        ])

        # Tab 2: change
        self.button_change = widgets.Button(
            description='change',
            button_style='warning',
            tooltip='change the dict as entered value',
            icon='check'
        )
        self.button_save = widgets.Button(
            description='save',
            button_style='danger',
            tooltip='save the changed dict as json file',
            icon='check'
        )

        self.box_buttons_change = widgets.Box(
            children=[
                self.button_change,
                self.button_save,
            ],
            layout=Layout(flex_flow='row', width='100%'))

        self.text_setValue = widgets.Text(
            value='',
            placeholder='Type changed value',
            description='set value:',
            disabled=False
        )

        self.togButton_value_type = widgets.Dropdown(
            options=['str', 'lambda x'], value='str',
            description='Type:',
            tooltips=['string',
                      'python lambda expression to be eval()'],
        )

        self.tab_change = widgets.VBox([
            self.select_paths_vals,
            self.togButton_value_type,
            self.text_setValue,
            self.box_buttons_change
        ])

    def update_tab2_selectbox(self, paths: Iterable[str]):
        """update select box contents in tab 2 (change)
        """
        # maintain order
        self.select_paths_vals.options = tuple(
            keyval_to_str(path, dget(self.obj, path))
            for path in dict.fromkeys(paths)
        )

    def wrap_output(self, func):
        """ wrapper function
        direct print to output
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.output:
                self.output.clear_output()
                func(*args, **kwargs)
        return wrapper

    def set_search_click(self):
        """set the tab for searching
        """
        depth_option = [None]
        depth_option.extend(
            list(np.arange(1, self.max_depth+1, 1))
        )
        self.slider_depth.options = depth_option

        @self.wrap_output
        def search_paths(arg):
            selects = self.select_paths

            key = self.text_search.value
            depth = self.slider_depth.value
            _str = ''.join([
                f"search {key} ",
                f"in depth: {depth} "
            ])

            print(_str)
            res = dsearch_nest(
                self.obj, key=key,
                depth=depth)
            selects.options = res.keys()
            self.update_tab2_selectbox(selects.options)

            print(f'find {len(res)} results')

        @self.wrap_output
        def filter_sel(arg):
            """filter selected value
            """
            selects = self.select_paths
            selects.options = selects.label
            print(f'filter {len(selects.options)} results')
            self.update_tab2_selectbox(selects.options)

        @self.wrap_output
        def unfilter_sel(arg):
            """unfilter selected value
            """
            selects = self.select_paths
            print(f'remove {len(selects.label)} results')
            keys = selects.label
            # maintain order
            _dict = dict.fromkeys(selects.options)
            for key in keys:
                _dict.pop(key)
            selects.options = tuple(_dict.keys())
            self.update_tab2_selectbox(selects.options)

        # bind button with click update func
        self.button_filter_sel.on_click(filter_sel)
        self.button_unfilter_sel.on_click(unfilter_sel)
        self.button_search.on_click(search_paths)

    def set_change_click(self):
        """set the tab for changing dict values
        """
        button = self.button_change

        @self.wrap_output
        def change_dict_val(b):
            selects = self.select_paths
            paths = selects.options
            print(f'change {len(paths)} values')
            vtype = self.togButton_value_type.value
            value = self.text_setValue.value
            for path in paths:
                set_dict_val(self.obj, path, value, vtype)

            self.update_tab2_selectbox(selects.options)
        button.on_click(change_dict_val)

    def set_save_click(self):
        """set the tab for saving dict
        """
        button = self.button_save

        @self.wrap_output
        def save(b):
            _str = ''.join([
                f'saved to', '\n',
                self.save_path
            ])
            print(_str)
            utils.save_json(
                obj=self.obj, _save_path=self.save_path)
        button.on_click(save)

    def set_all_tab(self):
        self.set_search_click()
        self.set_change_click()
        self.set_save_click()
        children = [self.tab_search, self.tab_change]

        tabs = widgets.Tab()
        tabs.children = children
        tabs.set_title(0, 'Search')
        tabs.set_title(1, 'Change')

        self.tabs = tabs

    def run(self):
        self.set_all_tab()

        display(self.tabs)
        display(self.output)


def test(_dict):
    app1 = App(_dict)
    app1.run()
