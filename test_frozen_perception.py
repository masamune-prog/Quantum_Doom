"""Integrity checks for the frozen geometry comparison (no Doom rendering required)."""

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from frozen_perception_models import (HEADS, FrozenPolicy, GeometryPerception,
                                      fingerprint, make_head, next_cooldown)
from train_frozen_perception import Replay, collect
from summarize_frozen_perception import summarize


class FrozenPerceptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_parameter_budgets(self):
        expected = dict(linear=104, small_mlp=71, classical=1096,
                        quantum2=74, product2=74, quantum4=146)
        for name in HEADS:
            self.assertEqual(sum(p.numel() for p in make_head(name).parameters()),
                             expected[name])

    def test_freeze_blocks_all_perception_updates(self):
        for name in HEADS:
            torch.manual_seed(17)
            perception = GeometryPerception()
            policy = FrozenPolicy(perception, make_head(name))
            before = fingerprint(perception)
            head_before = fingerprint(policy.head)
            optimizer = torch.optim.Adam(policy.head.parameters(), lr=0.01)
            images = torch.randint(0, 256, (3, 12, 60, 80), dtype=torch.uint8)
            logits, values = policy(images, torch.tensor([0, 10, 25]))
            self.assertEqual(logits.shape, (3, 7))
            self.assertEqual(values.shape, (3,))
            loss = F.cross_entropy(logits, torch.tensor([0, 4, 6]))
            loss.backward()
            optimizer.step()
            self.assertEqual(fingerprint(perception), before)
            self.assertNotEqual(fingerprint(policy.head), head_before)
            self.assertTrue(all(p.grad is None for p in perception.parameters()))

    def test_perception_geometry_gradients_and_cooldown(self):
        model = GeometryPerception()
        images = torch.randint(0, 256, (2, 12, 60, 80), dtype=torch.uint8)
        memory = torch.tensor([0, 25])
        targets = torch.rand(2, 20)
        sum(model.geometry_loss(images, memory, targets).values()).backward()
        self.assertGreater(float(model.bottleneck.weight.grad.abs().sum()), 0)
        self.assertTrue(torch.allclose(model(images, memory)[:, -1],
                                       torch.tensor([-torch.pi, torch.pi])))

    def test_collection_exact_budget_partial_stacks_and_executed_memory(self):
        class Pool:
            n = 2

            def reset(self, ids, seeds, options):
                self.step_index = getattr(self, "step_index", {})
                for i in ids:
                    self.step_index[i] = 0
                return [(np.zeros((12, 60, 80), np.uint8),
                         {"teacher": 6, "aux": np.zeros(20, np.float32)}) for _ in ids]

            def step(self, ids, actions):
                results = []
                for i, action in zip(ids, actions):
                    self.step_index[i] += 1
                    image = np.full((12, 60, 80), self.step_index[i], np.uint8)
                    results.append((image, 0.0, False, False,
                                    {"teacher": 6, "aux": np.zeros(20, np.float32)}))
                return results

        replay = Replay(7)
        collect(Pool(), replay, 7, np.random.default_rng(0), torch.device("cpu"),
                1_000_000, 100)
        self.assertEqual(replay.size, 7)
        self.assertEqual(list(replay.cooldown[:4]), [0, 25, 25, 25])
        images, _, _ = replay.batch(np.array([0, 1, 4]))
        self.assertEqual(images.shape, (3, 12, 60, 80))
        self.assertEqual(images[0].max(), 0)
        self.assertEqual(images[2].max(), 0)  # second episode's stack cannot leak
        self.assertEqual(next_cooldown(25, 0), 24)
        self.assertEqual(next_cooldown(0, 0), 0)

    def test_product_ablation_changes_circuit(self):
        entangled = make_head("quantum2")
        product = make_head("product2")
        product.load_state_dict(entangled.state_dict())
        inputs = torch.linspace(-2, 2, 24).reshape(2, 12)
        self.assertFalse(torch.allclose(entangled(inputs)[0], product(inputs)[0]))

    def test_summary_pairs_episodes_by_seed(self):
        heads = {}
        for name, successes in (("quantum2", [True, False]), ("linear", [False, False])):
            rows = [{"seed": seed, "success": success} for seed, success in
                    zip((20000, 20001), successes)]
            if name == "linear":
                rows.reverse()
            outcome = {"rows": rows, "successes": sum(successes), "episodes": 2,
                       "mean_progress": 0.5}
            heads[name] = {"parameters": 74 if name == "quantum2" else 104,
                           "greedy": outcome, "sampled": outcome,
                           "heldout_actions": {"accuracy": 0.5, "balanced_accuracy": 0.5}}
        result = summarize({"runs": [{"seed": 0, "heads": heads}],
                            "perception_unchanged": True, "seconds": 1})
        self.assertEqual(result["scope"], "pilot")
        self.assertEqual(result["paired_differences"]["quantum2_minus_linear"]["greedy"]
                         ["mean_success_rate_difference"], 0.5)
        self.assertIsNone(result["paired_differences"]["quantum2_minus_linear"]["greedy"]
                          ["paired_bootstrap_95_percent"])


if __name__ == "__main__":
    unittest.main()
