"""Semantic, recurrence, action-mapping, and real-engine checks for transfer training."""

import unittest
from types import SimpleNamespace
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from frozen_perception_models import fingerprint
from train_transfer import Replay, collect, evaluate, run, PerceptionSubset
from transfer_env import ACTION_MAPS, TRAIN_MAPS, TEST_MAPS, TransferEnv, context, ray_hits
from transfer_models import SemanticPerception, GoalActor, TransferPolicy


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_public_context_does_not_identify_exit_map(self):
        self.assertTrue(np.array_equal(context("e1m1"), context("e1m2")))
        self.assertTrue(np.array_equal(context("e1m2_keys"), context("e1m2")))
        self.assertEqual(context("e1m1").shape, (18,))
        self.assertEqual(context("navigation")[2:9].tolist(), [1, 0, 0, 0, 1, 1, 0])

    def test_nearest_wall_occludes_exit(self):
        lines = [{"p1": np.array([x, -2.]), "p2": np.array([x, 2.])} for x in (1., 5.)]
        indices, distance = ray_hits(np.zeros(2), np.array([0., np.pi]), lines)
        self.assertEqual(indices.tolist(), [0, -1])
        self.assertAlmostEqual(distance[0], 1)

    def test_test_maps_cannot_collect_training_data(self):
        for task in TEST_MAPS:
            with self.assertRaises(ValueError):
                collect(task, None, 1, None, None, 0)

    def test_frame_stacks_do_not_cross_episode_boundaries(self):
        replay = Replay(4)
        for value in (1, 9):
            replay.append({"frames": np.full((2, 3, 60, 80), value, np.uint8),
                           "labels": [0, 0], "targets": np.zeros((2, 27)),
                           "contexts": np.stack([context("e1m1")] * 2)})
        images, _, _, _ = replay.batch(np.array([2]), "cpu")
        self.assertEqual(int(images.min()), 9)
        self.assertEqual(replay.episodes, [(0, 2), (2, 4)])
        view = PerceptionSubset(replay, 2, 0)
        self.assertEqual(len(set(view.ids.tolist())), 2)
        self.assertEqual(view.batch(np.array([0]), "cpu")[0].shape, (1, 12, 60, 80))

    def test_existing_experiment_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text("preserve this")
            with self.assertRaises(FileExistsError):
                run(SimpleNamespace(device="cpu", threads=1, output=directory))
            self.assertEqual(Path(directory, "config.json").read_text(), "preserve this")

    def test_recurrent_online_matches_complete_episode(self):
        actor = GoalActor("gru")
        x = torch.rand(2, 5, 45)
        x[:, :, 29:36] = torch.as_tensor(context("navigation")[2:9])
        full, _ = actor(x)
        state, chunks = None, []
        for t in range(5):
            logits, state = actor(x[:, t:t+1], state)
            chunks.append(logits)
        self.assertTrue(torch.allclose(full, torch.cat(chunks, 1), atol=1e-6))
        self.assertTrue((full[:, :, [1, 2, 3, 6]] < -1e8).all())

    def test_feedforward_parameter_match_and_gradient(self):
        gru, mlp = GoalActor("gru", 64), GoalActor("mlp", 411)
        recurrent = sum(p.numel() for p in gru.parameters())
        feedforward = sum(p.numel() for p in mlp.parameters())
        self.assertLess(abs(recurrent - feedforward) / recurrent, .01)
        x = torch.rand(2, 3, 45)
        x[:, :, 29:36] = torch.as_tensor(context("navigation")[2:9])
        logits, _ = mlp(x)
        F.cross_entropy(logits.reshape(-1, 7), torch.zeros(6, dtype=torch.long)).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in mlp.parameters()))

    def test_frozen_semantics_update_only_actor(self):
        policy = TransferPolicy(SemanticPerception(), GoalActor())
        before = fingerprint(policy.perception)
        optimizer = torch.optim.Adam(policy.actor.parameters(), lr=.01)
        images = torch.randint(256, (2, 12, 60, 80), dtype=torch.uint8)
        contexts = torch.as_tensor(np.stack([context("e1m1")] * 2))
        logits, _ = policy.step(images, contexts)
        F.cross_entropy(logits, torch.tensor([0, 6])).backward()
        optimizer.step()
        self.assertEqual(before, fingerprint(policy.perception))
        self.assertTrue(all(p.grad is None for p in policy.perception.parameters()))

    def test_real_engine_semantics_and_key_control(self):
        for task in TRAIN_MAPS + TEST_MAPS:
            env = TransferEnv(task, teacher=task == "e1m1")
            try:
                image, info = env.reset(seed=456)
                self.assertEqual(image.shape, (12, 60, 80))
                self.assertEqual(info["semantic"].shape, (27,))
                self.assertTrue(np.isfinite(info["semantic"]).all())
                self.assertTrue(((info["semantic"] >= 0) & (info["semantic"] <= 1)).all())
                if task == "e1m2_keys":
                    self.assertTrue(info["keys_granted"])
                    self.assertTrue(info["key_hud_changed"])
                env.step(ACTION_MAPS[task][0])
            finally:
                env.close()

    def test_unseen_map_perception_probes_share_evaluation_frames(self):
        first, second = SemanticPerception(), SemanticPerception()
        second.load_state_dict(first.state_dict())
        policy = TransferPolicy(first, GoalActor())
        args = SimpleNamespace(num_envs=1, device="cpu", eval_max_steps=4,
                               semantic_probe_every=1)
        result = evaluate("e1m2_keys", policy, None, args, 20000, 1,
                          probes={"single": first, "multi": second})
        self.assertEqual(result["perception_probe_states"], 4)
        self.assertEqual(result["paired_perception_losses"]["single"],
                         result["paired_perception_losses"]["multi"])


if __name__ == "__main__":
    unittest.main()
