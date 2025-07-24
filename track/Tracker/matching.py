import numpy as np
from cython_bbox import bbox_overlaps as bbox_ious

def ious(altrbs, bltrbs):
    """
    Compute cost based on IoU
    :type altrbs: list[ltrb] | np.ndarray
    :type bltrbs: list[ltrb] | np.ndarray
    
    :rtype ious np.ndarray
    """
    ious = np.zeros((len(altrbs), len(bltrbs)), dtype=float)
    if ious.size == 0:
        return ious
    
    ious = bbox_ious(
        np.ascontiguousarray(altrbs, dtype=float),
        np.ascontiguousarray(bltrbs, dtype=float)
    )

    return ious