import unittest
import numpy as np

from gate6_estimated_motion import (
    TRAIN_POS1,
    TRAIN_POS2,
    collect_motion_examples,
    estimate_flow,
    make_motion_pair,
)


class EstimatedMotionTests(unittest.TestCase):
    def test_local_correspondence_recovers_object_motion(self):
        frame0, frame1, truth, _, oracle = make_motion_pair(
            TRAIN_POS1,
            TRAIN_POS2,
            seed=10000,
        )
        estimated, valid = estimate_flow(frame0, frame1)

        moving = truth > 0
        valid_moving = moving & valid
        exact = np.mean(
            np.all(
                estimated[valid_moving] == oracle[valid_moving],
                axis=1,
            )
        )

        self.assertGreater(np.mean(valid[moving]), 0.70)
        self.assertGreater(exact, 0.95)

    def test_estimated_motion_writes_positive_and_negative_evidence(self):
        frame0, frame1, _, _, _ = make_motion_pair(
            TRAIN_POS1,
            TRAIN_POS2,
            seed=10000,
        )
        estimated, valid = estimate_flow(frame0, frame1)
        _, targets = collect_motion_examples(frame0, estimated, valid)

        self.assertGreater(len(targets), 20)
        self.assertEqual(set(np.unique(targets).tolist()), {0.0, 1.0})


if __name__ == "__main__":
    unittest.main()
