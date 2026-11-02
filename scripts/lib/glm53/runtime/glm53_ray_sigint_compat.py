"""Run-only, opt-in compatibility for Ray DeferSigint entry handler EXACT None."""
import functools
import json
import os
from pathlib import Path
import signal
import threading
import traceback


def _kind(handler):
    if handler is None:
        return 'none'
    if handler == signal.SIG_DFL:
        return 'SIG_DFL'
    if handler == signal.SIG_IGN:
        return 'SIG_IGN'
    return 'callable' if callable(handler) else 'other'


def _emit(event, **fields):
    try:
        os.write(2, ('GLM53_RAY_SIGINT ' + json.dumps(
            dict(pid=os.getpid(), event=event, **fields), ensure_ascii=True
        ) + '\n').encode())
    except Exception:
        pass


def _patch_class(cls, normalize):
    """Patch one class, preserving its original exit and non-None behavior."""
    if getattr(cls, '_glm53_sigint_entry_installed', False):
        return
    original_enter = cls.__enter__
    original_signal_patch = cls._signal_monkey_patch

    @functools.wraps(original_enter)
    def enter(self):
        main_thread = threading.current_thread() == threading.main_thread()
        initial = signal.getsignal(signal.SIGINT)
        _emit('enter', handler=_kind(initial), main_thread=main_thread)
        if initial is None and main_thread and normalize:
            signal.signal(signal.SIGINT, signal.default_int_handler)
            _emit('normalized_entry_none', handler=_kind(signal.getsignal(signal.SIGINT)))
        # Preserve original return, failure, signal deferral and restoration.
        return original_enter(self)

    @functools.wraps(original_signal_patch)
    def signal_patch(self, signum, handler):
        if (signum == signal.SIGINT and handler is None
                and threading.current_thread() == threading.main_thread()):
            # Record source locations only, never locals or arbitrary exception text.
            frames = traceback.extract_stack(limit=18)[:-1]
            _emit('attempted_none_inside_context', source_stack=[
                {'file': f.filename, 'line': f.lineno, 'function': f.name}
                for f in frames
            ])
        return original_signal_patch(self, signum, handler)

    cls.__enter__ = enter
    cls._signal_monkey_patch = signal_patch
    cls._glm53_sigint_entry_installed = True


def install():
    normalize = os.environ.get('GLM53_RAY_SIGINT_NONE_COMPAT') == '1'
    trace = os.environ.get('GLM53_RAY_SIGINT_TRACE') == '1'
    if not (normalize or trace):
        return
    base = Path('/mnt/share/l00804615/glm5.3_flash_native')
    if not Path.cwd().resolve().is_relative_to(base.resolve()):
        raise RuntimeError('SIGINT hook requires task-owned cwd')
    if json.loads((base / '.native-owner.json').read_text()).get('owner') != 'glm53-native-20261019':
        raise RuntimeError('SIGINT hook task owner mismatch')
    from ray._private.utils import DeferSigint
    _patch_class(DeferSigint, normalize)
    _emit('installed', normalize_entry_none=normalize,
          handler=_kind(signal.getsignal(signal.SIGINT)))
