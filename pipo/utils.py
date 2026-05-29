from typing import List
import sys, datetime
import numpy as np


def is_debug_mode() -> bool:
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None


def get_timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_mean(nums: List[float]):
    if len(nums) == 0:
        return 0.0
    else:
        return np.mean(nums)
