"""S1 annotation for untouched TileLang TIR."""

from tvm import IRModule

from ..tir_utils import map_prim_funcs


def annotate_sunway_tir(mod: IRModule) -> IRModule:
    """Mark the untouched TileLang TIR as the S1 backend input."""

    return map_prim_funcs(mod, lambda func: func.with_attr("sunway.phase", "S1"))
