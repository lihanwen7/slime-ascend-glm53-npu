from types import SimpleNamespace
from unittest.mock import Mock

from slime.utils.accelerator.npu import NPUAccelerator


def test_empty_cache_uses_npu_allocator_outside_graph_capture(monkeypatch):
    module = SimpleNamespace(
        is_current_stream_capturing=Mock(return_value=False),
        empty_cache=Mock(),
    )
    accelerator = NPUAccelerator()
    monkeypatch.setattr(accelerator, "_module", lambda: module)

    accelerator.empty_cache()

    module.empty_cache.assert_called_once_with()


def test_empty_cache_is_skipped_during_graph_capture(monkeypatch):
    module = SimpleNamespace(
        is_current_stream_capturing=Mock(return_value=True),
        empty_cache=Mock(),
    )
    accelerator = NPUAccelerator()
    monkeypatch.setattr(accelerator, "_module", lambda: module)

    accelerator.empty_cache()

    module.empty_cache.assert_not_called()
