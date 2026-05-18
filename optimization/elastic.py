from __future__ import annotations

import signal
from dataclasses import dataclass


@dataclass
class ElasticState:
    should_stop: bool = False
    signal_name: str | None = None


def install_signal_handlers() -> ElasticState:
    """Mark training for graceful shutdown on preemption signals."""
    state = ElasticState()

    def _handler(signum, _frame):
        state.should_stop = True
        state.signal_name = signal.Signals(signum).name
        print(f"Received {state.signal_name}; will stop after the current optimizer step.")

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _handler)

    return state
