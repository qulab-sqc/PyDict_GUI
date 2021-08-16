import numpy as np
from typing import Iterable, Tuple, Any, Set
from functools import wraps
import os

import ipywidgets as widgets
from ipywidgets import Button, Layout
from IPython.core.display import display

from .utils import dget, dict_depth, dsearch_nest
from . import utils


def keyval_to_str(key, val):
    """string representation of key and value
    for print or gui show
    """
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
        self.max_rows_search = 10

        self.create_widgets_search()
        self.create_widgets_change()

    def create_widgets_search(self):
        """Create the widgets for Search tab

        * text_search: input texts to search
        * select_search_result: show or select search result
        * slider_depth: a slider to change search depth
        * button_search, button_search, button_remove:
            buttons for search, filter, unfilter
        """
        self.output = widgets.Output()

        self.text_search = widgets.Text(
            value='*',
            placeholder='Type something',
            description='Search key:',
            disabled=False
        )

        self.widget_search_result = widgets.SelectMultiple(
            options=dict(),
            rows=self.max_rows_search,
            description=r'Searched:',
            disabled=False,
            layout=Layout(width='200%')
        )

        self.slider_depth = widgets.SelectionSlider(
            options=(None,),
            value=None,
            description='depth:',
            disabled=False,
            readout=True
        )

        self.button_search = widgets.Button(
            description='search',
            disabled=False,
            button_style='success',
            # 'success', 'info', 'warning', 'danger' or ''
            tooltip='search dict',
            icon='check'
        )

        self.button_extract = widgets.Button(
            description='extract',
            disabled=False,
            button_style='info',
            tooltip='filter selected',
            icon='filter'
        )

        self.button_remove = widgets.Button(
            description='remove',
            disabled=False,
            button_style='info',
            tooltip='unfilter selected',
            icon='eraser'
        )

        self.box_buttons_search = widgets.Box(
            children=[
                self.button_search,
                self.button_extract,
                self.button_remove,
            ],
            layout=Layout(flex_flow='row', width='100%'))

        self.tab_search = widgets.VBox([
            self.text_search,
            self.slider_depth,
            self.widget_search_result,
            self.box_buttons_search,
        ])

    def create_widgets_change(self):
        """Create the widgets for Change tab

        * togButton_value_type, text_setValue: 
            input value type and value to change the selected
            dicts
        * button_change, button_save:
            buttons of change, save for dict
        """
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
            self.widget_search_result,
            # created from self.create_widgets_search
            self.togButton_value_type,
            self.text_setValue,
            self.box_buttons_change
        ])

    def _update_search_results(self, paths: Iterable[str]):
        """update search results contents
        """
        self.widget_search_result.options = {
            keyval_to_str(path, dget(self.obj, path)):
            path
            for path in paths
        }

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

    def set_search_interact(self):
        """set the interaction for searching
        """
        depth_option = [None]
        depth_option.extend(
            list(np.arange(1, self.max_depth+1, 1))
        )
        self.slider_depth.options = depth_option

        @self.wrap_output
        def search_paths(arg):
            """search dict
            """
            key = self.text_search.value
            depth = self.slider_depth.value
            _str = ''.join([
                f"search {key} ",
                f"in depth: {depth} "
            ])
            print(_str)

            res: dict = dsearch_nest(
                self.obj, key=key,
                depth=depth)
            self._update_search_results(res.keys())

            print(f'find {len(res)} results')

        @self.wrap_output
        def extract_select(arg):
            """filter selected value
            """
            res = self.widget_search_result
            select_paths = res.get_interact_value()
            print(f'extract {len(select_paths)} results')
            self._update_search_results(select_paths)

        @self.wrap_output
        def remove_select(arg):
            """remove selected value
            """
            res = self.widget_search_result
            select_paths: Set[str] = set(res.get_interact_value())
            all_paths: Set[str] = set(res.options.values())
            remain_paths = all_paths ^ select_paths
            print(f'remove {len(select_paths)} results')
            self._update_search_results(remain_paths)

        # bind button with click update func
        self.button_extract.on_click(extract_select)
        self.button_remove.on_click(remove_select)
        self.button_search.on_click(search_paths)

    def set_change_click(self):
        """set the tab for changing dict values
        """
        button = self.button_change

        @self.wrap_output
        def change_dict_val(b):
            res = self.widget_search_result
            changed_paths = res.options.values()
            print(f'change {len(changed_paths)} values')
            vtype = self.togButton_value_type.value
            value = self.text_setValue.value
            for path in changed_paths:
                set_dict_val(self.obj, path, value, vtype)

            self._update_search_results(res.options.values())
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

    def set_all_interacts(self):
        """set interacts of all tabs' widgets
        """
        self.set_search_interact()
        self.set_change_click()
        self.set_save_click()
        children = [self.tab_search, self.tab_change]

        tabs = widgets.Tab()
        tabs.children = children
        tabs.set_title(0, 'Search')
        tabs.set_title(1, 'Change')

        self.tabs = tabs

    def run(self):
        """run interacts and display wdigets
        """
        self.set_all_interacts()

        display(self.tabs)
        display(self.output)


def test(_dict):
    app1 = App(_dict)
    app1.run()
