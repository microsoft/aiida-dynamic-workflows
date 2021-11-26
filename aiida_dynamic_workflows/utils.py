# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import asyncio

from IPython.display import Image
import aiida
import graphviz


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
