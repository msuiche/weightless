"""torch.compile guards for the four rules core.py is written around.

core.py's module docstring lists them as measured failure modes from the
hotfix lanes: dense stack indexed by global layer id, alpha as a tensor
buffer (a Python float gets baked into the compile cache as a constant),
an unconditional apply, non-persistent buffers. The per-request lane adds
two more tensors to the traced region, so it has to hold the same line.

These compile on CPU; no GPU and no vLLM.
"""
import pathlib
import sys
import unittest

import torch

_HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[2]))

from weightless_steer.core import SteeringCore  # noqa: E402

HIDDEN, NUM_LAYERS, MAX_TOKENS, MAX_REQS = 4, 6, 16, 4
D = torch.tensor([1.0, 0.0, 0.0, 0.0])
STEERED_LAYER = 1


def build(**geometry):
    core = SteeringCore({STEERED_LAYER: D}, 2.0, "residual_stream_post_layer",
                        NUM_LAYERS, HIDDEN, **geometry)
    owner = torch.nn.Module()
    core.register_buffers(owner, torch.float32)
    return core, owner


def graph_count(fn, layers=range(NUM_LAYERS)):
    import torch._dynamo as dyn
    dyn.reset()
    dyn.utils.counters.clear()
    h = torch.randn(4, HIDDEN)
    compiled = torch.compile(fn, fullgraph=True)
    for layer in layers:
        compiled(layer, h.clone())
    return dyn.utils.counters["stats"]["unique_graphs"]


class PerRequestApplyCompiles(unittest.TestCase):
    def setUp(self):
        self.core, self.owner = build(max_num_tokens=MAX_TOKENS,
                                      max_num_reqs=MAX_REQS)
        self.h = torch.randn(4, HIDDEN)
        self.compiled = torch.compile(self.core.apply, fullgraph=True)

    def test_compiles_without_a_graph_break(self):
        """fullgraph=True raises if anything forces a break."""
        out = self.compiled(STEERED_LAYER, self.h.clone())
        expect = self.h - 2.0 * (self.h @ D).unsqueeze(-1) * D
        torch.testing.assert_close(out, expect)

    def test_alpha_rows_are_not_baked_into_the_graph(self):
        """The failure mode the tensor alpha buffer exists to prevent.

        A Python float would be traced as a constant and the second call
        would silently reuse the first call's alpha.
        """
        first = self.compiled(STEERED_LAYER, self.h.clone())
        self.core.set_control_rows(
            alpha_rows=torch.tensor([0.0, 0.0, 3.0, 3.0])
        )
        second = self.compiled(STEERED_LAYER, self.h.clone())
        rows = torch.tensor([0.0, 0.0, 3.0, 3.0]).unsqueeze(-1)
        torch.testing.assert_close(
            second, self.h - rows * (self.h @ D).unsqueeze(-1) * D
        )
        self.assertFalse(torch.allclose(first, second))

    def test_layer_gate_is_not_baked_into_the_graph(self):
        self.compiled(STEERED_LAYER, self.h.clone())
        bank = torch.ones(NUM_LAYERS, MAX_REQS + 1)
        bank[STEERED_LAYER, 1] = 0.0
        self.core.set_control_rows(
            slot_rows=torch.ones(4, dtype=torch.long), layer_bank=bank
        )
        # Every token now belongs to slot 1, gated off at this layer.
        torch.testing.assert_close(
            self.compiled(STEERED_LAYER, self.h.clone()), self.h
        )


class PerRequestDoesNotCostExtraRecompiles(unittest.TestCase):
    def test_layer_id_generalises_as_well_as_the_scalar_lane(self):
        """Layer-major gate keeps layer_idx an ordinary integer index.

        Gating as bank[slots, layer_idx] instead makes layer_idx part of an
        advanced index, which specialises per layer -- one graph per
        decoder layer instead of one for the loop.
        """
        scalar, _ = build()
        per_request, _ = build(max_num_tokens=MAX_TOKENS,
                               max_num_reqs=MAX_REQS)
        self.assertEqual(graph_count(per_request.apply),
                         graph_count(scalar.apply))


class BuffersStayOutOfTheStateDict(unittest.TestCase):
    def test_no_control_buffer_is_persistent(self):
        """A persistent buffer shows up as an unexpected load_weights key."""
        _, owner = build(max_num_tokens=MAX_TOKENS, max_num_reqs=MAX_REQS)
        self.assertEqual(owner.state_dict(), {})
        self.assertEqual(
            {"_steer_stack", "_steer_alpha", "_steer_alpha_rows",
             "_steer_slot_rows", "_steer_layer_bank"},
            set(dict(owner.named_buffers())),
        )


if __name__ == "__main__":
    unittest.main()
