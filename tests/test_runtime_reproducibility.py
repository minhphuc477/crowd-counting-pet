import random

import numpy as np
import pytest
import torch

import util.misc as utils


def _next_random_values():
    return random.random(), float(np.random.rand()), float(torch.rand(()))


def test_preserve_rng_state_isolates_diagnostics():
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    state = utils.capture_rng_state()
    expected = _next_random_values()
    utils.restore_rng_state(state)

    @utils.preserve_rng_state
    def diagnostic():
        _next_random_values()
        torch.rand(16)

    diagnostic()
    assert _next_random_values() == expected


def test_named_generator_state_resumes_exactly():
    train_generator = torch.Generator().manual_seed(23)
    validation_generator = torch.Generator().manual_seed(29)
    generators = {
        'train_loader': train_generator,
        'validation_loader': validation_generator,
    }
    state = utils.capture_rng_state(generators)
    expected_train = torch.randperm(31, generator=train_generator)
    expected_validation = torch.randint(100, (12,), generator=validation_generator)

    assert utils.restore_rng_state(state, generators)
    assert torch.equal(torch.randperm(31, generator=train_generator), expected_train)
    assert torch.equal(
        torch.randint(100, (12,), generator=validation_generator),
        expected_validation,
    )


def test_deterministic_attention_backend_disables_fast_sdpa(monkeypatch):
    calls = []

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.backends.cuda, 'enable_flash_sdp', lambda enabled: calls.append(('flash', enabled)), raising=False)
    monkeypatch.setattr(torch.backends.cuda, 'enable_mem_efficient_sdp', lambda enabled: calls.append(('mem_efficient', enabled)), raising=False)
    monkeypatch.setattr(torch.backends.cuda, 'enable_math_sdp', lambda enabled: calls.append(('math', enabled)), raising=False)
    monkeypatch.setattr(torch.backends.cuda, 'enable_cudnn_sdp', lambda enabled: calls.append(('cudnn', enabled)), raising=False)

    utils.set_deterministic_attention_backend(True)

    assert ('flash', False) in calls
    assert ('mem_efficient', False) in calls
    assert ('math', True) in calls
    assert ('cudnn', False) in calls


def test_atomic_checkpoint_failure_preserves_previous_file(tmp_path, monkeypatch):
    destination = tmp_path / 'checkpoint.pth'
    original = b'previous-valid-checkpoint'
    destination.write_bytes(original)

    def failing_save(_obj, path, *_args, **_kwargs):
        path = str(path)
        with open(path, 'wb') as handle:
            handle.write(b'partial-archive')
        raise OSError('simulated disk full')

    monkeypatch.setattr(torch, 'save', failing_save)
    with pytest.raises(OSError, match='simulated disk full'):
        utils.save_on_master({'epoch': 4}, destination)

    assert destination.read_bytes() == original
    assert list(tmp_path.glob('.checkpoint.pth.*.tmp')) == []
