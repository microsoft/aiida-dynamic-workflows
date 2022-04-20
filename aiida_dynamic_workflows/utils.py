# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import shutil
from typing import Iterable

from IPython.display import Image
import aiida
import graphviz
from tqdm import tqdm


def block_until_done(chain: aiida.orm.WorkChainNode, interval=1) -> int:
    """Block a running chain until an exit code is set.

    Parameters
    ----------
    chain : aiida.orm.WorkChainNode
    interval : int, optional
        Checking interval, by default 1

    Returns
    -------
    int
        Exit code.
    """
    loop = asyncio.get_event_loop()

    async def wait_until_done(chain: aiida.orm.WorkChainNode) -> None:
        while chain.exit_status is None:
            await asyncio.sleep(interval)

    coro = wait_until_done(chain)
    loop.run_until_complete(coro)
    return chain.exit_status


def render_png(g: graphviz.Digraph) -> Image:
    """Render 'graphviz.Digraph' as png."""
    return Image(g.render(format="png"))


def parallel_rmtree(dirs: Iterable[str | Path], with_tqdm: bool = True):
    """Apply 'shutil.rmtree' to 'dirs' in parallel using a thread pool."""
    # Threadpool executor, as this task is IO bound.
    rmtree = partial(shutil.rmtree, ignore_errors=True)
    with ThreadPoolExecutor() as tp:
        it = tp.map(rmtree, dirs)
        if with_tqdm:
            it = tqdm(it, total=len(dirs))
        # Bare 'for' loop to force the map to complete.
        for _ in it:
            pass
