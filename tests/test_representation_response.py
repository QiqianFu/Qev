# SPDX-License-Identifier: Apache-2.0
import hashlib

import pytest

from qev.representation_response import CaptureTaskStates, response_geometry, response_geometry_loss, response_shard


def test_pair_sharding_keeps_shared_originals_together_without_labels():
    rows = [{'original_key': hashlib.sha256(str(i).encode()).hexdigest(), 'label': label}
            for i in range(100) for label in ['first', 'second']]
    assignments = [[row for row in rows if response_shard(row, 10) == rank] for rank in range(10)]
    assert sum(map(len, assignments)) == len(rows)
    assert all(assignments)
    for i in range(0, len(rows), 2):
        assert response_shard(rows[i], 10) == response_shard(rows[i+1], 10)
    with pytest.raises(ValueError):
        response_shard(rows[0], 0)


def test_response_geometry_isometry_and_identities():
    torch = pytest.importorskip('torch')
    torch.manual_seed(27)
    a, b = torch.randn(4, 7, dtype=torch.float64), torch.randn(4, 7, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(11, 11, dtype=torch.float64))
    embedding = q[:7]
    left, right = response_geometry(a, b), response_geometry(a @ embedding, b @ embedding)
    for key in left:
        torch.testing.assert_close(left[key], right[key], atol=1e-12, rtol=1e-12)
    s, r, g = left['sensitivity'], left['anchored_response'], left['original_gram']
    torch.testing.assert_close(left['edited_gram'], g+r+r.T+s, atol=1e-12, rtol=1e-12)
    assert torch.linalg.eigvalsh(s).min() > -1e-12
    reverse = response_geometry(b, a)
    torch.testing.assert_close(reverse['sensitivity'], s, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(reverse['anchored_response'], -r-s, atol=1e-12, rtol=1e-12)
    assert not torch.allclose(reverse['anchored_response'], r)


def test_response_loss_has_target_fixed_point_and_nonzero_learning_gradient():
    torch = pytest.importorskip('torch')
    torch.manual_seed(3)
    a, b = torch.randn(3, 8), torch.randn(3, 8)
    target = response_geometry(a, b)
    for kind in ('sensitivity', 'anchored_response'):
        x, y = a.clone().requires_grad_(), b.clone().requires_grad_()
        loss = response_geometry_loss(x, y, target[kind], kind)
        loss.backward()
        assert loss == 0 and x.grad.abs().max() == 0 and y.grad.abs().max() == 0
        x, y = (a+.1*torch.randn_like(a)).requires_grad_(), b.clone().requires_grad_()
        loss = response_geometry_loss(x, y, target[kind], kind)
        loss.backward()
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
        with pytest.raises(ValueError):
            response_geometry_loss(x, y, torch.zeros(2, 2), kind)


def test_scorer_capture_preserves_head_outputs_and_gradient_path():
    torch = pytest.importorskip('torch')
    from qev.model import SetDecisionHead
    torch.manual_seed(9)
    head = SetDecisionHead(16, 16, 4, 2).eval()
    q, candidates = torch.randn(16), torch.randn(3, 16)
    with torch.no_grad():
        expected = head(q, candidates)
        with CaptureTaskStates(head) as capture:
            actual = head(q, candidates)
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
        assert len(capture.features) == 1 and capture.features[0].shape == (4, 16)
        # Independently replay the head up to its scorer. Output preservation
        # alone cannot detect swapped query/candidate halves in the observer.
        projected_q = head.project(head.norm(q)) + head.query_role
        projected_cs = head.project(head.norm(candidates))
        roles = torch.cat([projected_q[None], projected_cs], dim=0)[None]
        for layer in head.layers:
            roles = layer(roles)
        torch.testing.assert_close(capture.features[0], roles[0], atol=0, rtol=0)
        assert not torch.equal(capture.features[0][1], capture.features[0][2])
    assert not any(layer._forward_hooks or layer._forward_pre_hooks for layer in head.layers)
    assert not head.scorer[0]._forward_pre_hooks
    head.train()
    with CaptureTaskStates(head) as capture:
        head(q, candidates)
        head(q+.1, candidates+.2*torch.randn_like(candidates))
    a, b = capture.features
    target = response_geometry(a.detach(), b.detach())['sensitivity']*.5
    loss = response_geometry_loss(a, b, target)
    loss.backward()
    assert head.project.weight.grad.abs().sum() > 0
