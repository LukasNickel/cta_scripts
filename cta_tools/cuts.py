import numpy as np
from operator import le, lt, eq, ne, ge, gt
import logging
log = logging.getLogger(__name__)


OPERATORS = {
    "<": lt,
    "lt": lt,
    "<=": le,
    "le": le,
    "==": eq,
    "eq": eq,
    "=": eq,
    "!=": ne,
    "ne": ne,
    ">": gt,
    "gt": gt,
    ">=": ge,
    "ge": ge,
}

def create_mask_selection(df, selection_config):
    mask = np.ones(len(df), dtype=bool)

    for c in selection_config:
        if len(c) > 1:
            raise ValueError(
                "Expected dict with single entry column: [operator, value]."
            )
        name, (operator, value) = list(c.items())[0]

        before = np.count_nonzero(mask)
        selection = OPERATORS[operator](df[name], value)
        mask = np.logical_and(mask, selection)
        after = np.count_nonzero(mask)
        log.debug(
            'Cut "{} {} {}" removed {} events'.format(
                name, operator, value, before - after
            )
        )

    return mask
