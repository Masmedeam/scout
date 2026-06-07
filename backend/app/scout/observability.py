import logging
import os
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar, cast

from app.core.config import settings

F = TypeVar("F", bound=Callable[..., Any])
_weave_ready = False
_weave_warned = False
logger = logging.getLogger(__name__)


def _init_weave() -> Any | None:
    global _weave_ready, _weave_warned
    if not settings.WEAVE_ENABLED:
        return None
    if not _has_wandb_credentials():
        if not _weave_warned:
            logger.warning(
                "WEAVE_ENABLED=True but no W&B credentials were found. "
                "Set WANDB_API_KEY to emit Scout traces."
            )
            _weave_warned = True
        return None
    try:
        import weave

        if not _weave_ready:
            weave.init(settings.WEAVE_PROJECT)
            _weave_ready = True
        return weave
    except Exception as exc:
        if not _weave_warned:
            logger.warning("Weave initialization failed: %s", exc)
            _weave_warned = True
        return None


def _has_wandb_credentials() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    return (Path.home() / ".netrc").exists()


def scout_op(name: str) -> Callable[[F], F]:
    weave = _init_weave()

    def decorator(func: F) -> F:
        if weave is not None:
            try:
                return cast(F, weave.op(name=name)(func))
            except Exception:
                pass

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator
