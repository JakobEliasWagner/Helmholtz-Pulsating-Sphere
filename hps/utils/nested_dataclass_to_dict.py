import inspect
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np

from hps.functions import Function


def dataclass_to_dict(obj: Any) -> Any:
    """
    Recursively converts a nested dataclass into a dictionary.

    Args:
        obj: The dataclass instance to convert.

    Returns:
        A dictionary representation of the dataclass, including nested dataclasses.
    """
    if is_dataclass(obj):
        result = {}
        for field in fields(obj):
            value = getattr(obj, field.name)
            result[field.name] = dataclass_to_dict(value)
        return result
    elif isinstance(obj, list):
        return [dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: dataclass_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, Function):
        return inspect.getsource(obj.mapping)
    else:
        return obj
