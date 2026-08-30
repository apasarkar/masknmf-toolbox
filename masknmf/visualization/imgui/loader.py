import threading
from typing import Callable, Optional


class AsyncLoad:
    """
    Run a callable on a daemon thread and poll it from the draw loop.

    Lets a GUI open immediately on placeholder data while a slow build runs.
    """

    def __init__(self):
        self.status: Optional[str] = None
        self.error: Optional[str] = None
        self._result = None
        self._lock = threading.Lock()

    @property
    def busy(self) -> bool:
        return self.status is not None

    def start(self, fn: Callable, status: str = "loading..."):
        self.status = status
        self.error = None
        with self._lock:
            self._result = None
        threading.Thread(target=self._run, args=(fn,), daemon=True).start()

    def _run(self, fn):
        try:
            value = fn()
            with self._lock:
                self._result = ("ok", value)
        except Exception as e:
            with self._lock:
                self._result = ("error", e)

    def poll(self):
        """Result once, then None. Sets .error and clears .status on failure."""
        with self._lock:
            result, self._result = self._result, None
        if result is None:
            return None
        kind, value = result
        self.status = None
        if kind == "error":
            self.error = f"{type(value).__name__}: {value}"
            return None
        return value
