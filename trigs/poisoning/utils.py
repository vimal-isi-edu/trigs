import numpy as np
import random
from PIL import Image
from torch import Tensor


def create_trigger(base_size, target_size, n_channels=3) -> Image:
    assert base_size <= target_size, "trigger's base size cannot be larger than the target size"
    assert n_channels in [1, 3], "number of channels can only be 1 or 3"
    base_trigger = np.random.randint(0, 256, (base_size, base_size, n_channels)).astype(np.uint8)
    trigger = Image.fromarray(base_trigger).resize((target_size, target_size))
    return trigger


def add_trigger_to_tensor(img: Tensor, trigger: Tensor) -> Tensor:
    if trigger.ndim == 3:
        tc, th, tw = trigger.shape
    else:
        th, tw = trigger.shape
        tc = 1
    if img.ndim == 3:
        ic, ih, iw = img.shape
    else:
        ih, iw = img.shape
        ic = 1
    assert th <= ih and tw <= iw and tc == ic, \
        f"Trigger cannot be larger than the image or have a different number of dims:"\
        f" image shape is ({ic}, {ih}, {iw}) and trigger shape is ({tc}, {th}, {tw})"
    tx = random.randrange(0, iw - tw + 1)
    ty = random.randrange(0, ih - th + 1)
    trig_img = img.clone()
    trig_img[:, ty:ty+th, tx:tx+tw] = trigger
    return trig_img
