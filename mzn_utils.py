"""Thin bridge between Python and MiniZinc: writes .dzn data files, calls the model, returns results."""
from __future__ import annotations

import glob
import os
import shutil
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import minizinc

_SOLVERS: dict[str, minizinc.Solver] = {}
_DRIVER_READY = False


def _candidate_exes() -> list[Path]:
    """Possible locations of the minizinc executable."""
    cands: list[Path] = []
    exe_name = "minizinc.exe" if os.name == "nt" else "minizinc"
    env = os.environ.get("MINIZINC_PATH")
    if env:
        p = Path(env)
        cands.append(p if p.is_file() else p / exe_name)
    found = shutil.which("minizinc")
    if found:
        cands.append(Path(found))
    here = Path(__file__).resolve().parent
    for pat in (str(here.parent / "tools" / "*" / "bin" / exe_name),
                "/Applications/MiniZincIDE.app/Contents/Resources/minizinc",
                "C:/Program Files/MiniZinc/minizinc.exe",
                "/usr/local/bin/minizinc", "/opt/minizinc*/bin/minizinc"):
        cands.extend(Path(x) for x in glob.glob(pat))
    return cands


def setup_driver() -> None:
    """Locate the MiniZinc executable (env MINIZINC_PATH, PATH, or common install folders)."""
    global _DRIVER_READY
    if _DRIVER_READY:
        return
    for exe in _candidate_exes():
        if exe.is_file():
            minizinc.default_driver = minizinc.Driver(exe)
            _DRIVER_READY = True
            return
    raise RuntimeError(
        "MiniZinc executable not found. Install MiniZinc (https://www.minizinc.org) and either add it "
        "to PATH or set the MINIZINC_PATH environment variable to its bin folder."
    )


def get_solver(name: str) -> minizinc.Solver:
    setup_driver()
    if name not in _SOLVERS:
        _SOLVERS[name] = minizinc.Solver.lookup(name)
    return _SOLVERS[name]


# --------------------------------------------------------------------- dzn --
def _f(x: float) -> str:
    return f"{float(x):.12e}"


def _dzn_value(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "true" if v else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return _f(v)
    arr = np.asarray(v)
    if arr.ndim == 0:
        return _dzn_value(arr.item())
    is_int = np.issubdtype(arr.dtype, np.integer)
    fmt = (lambda x: str(int(x))) if is_int else _f
    if arr.ndim == 1:
        return "[" + ", ".join(fmt(x) for x in arr) + "]"
    if arr.ndim == 2:
        r, c = arr.shape
        return f"array2d(1..{r}, 1..{c}, [" + ", ".join(fmt(x) for x in arr.ravel()) + "])"
    raise ValueError("only 0-, 1- and 2-dimensional data supported")


def dict_to_dzn(data: dict) -> str:
    return "".join(f"{k} = {_dzn_value(v)};\n" for k, v in data.items())


# ------------------------------------------------------------------- solve --
def solve(model_path: Path | str, data: dict, solver: str = "highs",
          timeout_s: int = 60, dzn_path: Path | str | None = None):
    """Write `data` to a .dzn file, run `model_path` with `solver`, return the MiniZinc Result."""
    slv = get_solver(solver)
    tmp = None
    if dzn_path is None:
        fd, tmp = tempfile.mkstemp(suffix=".dzn")
        os.close(fd)
        dzn_path = tmp
    Path(dzn_path).write_text(dict_to_dzn(data))
    try:
        model = minizinc.Model()
        model.add_file(str(model_path))
        model.add_file(str(dzn_path))
        inst = minizinc.Instance(slv, model)
        res = inst.solve(timeout=timedelta(seconds=timeout_s))
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
    return res
