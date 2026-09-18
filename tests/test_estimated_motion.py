import unittest
import numpy as np

from gate6_estimated_motion_write import (
    TRAINING_EPISODES,
    estimate_bidirectional_flow,
    render_scene,
    true_flow,
)


class EstimatedMotionTests(unittest.TestCase):
    def test_local_correspondence_recovers_training_shift(self):
        pos1, pos2, shift1, shift2, texture_seed = TRAINING_EPISODES[0]

        frame0, labels0, _ = render_scene(
            pos1,
            pos2,
            texture_seed=texture_seed,
            sensor_seed=100,
        )
        frame1, _, _ = render_scene(
            (pos1[0] + shift1[0], pos1[1] + shift1[1]),
            (pos2[0] + shift2[0], pos2[1] + shift2[1]),
            texture_seed=texture_seed,
            sensor_seed=200,
        )

        estimated, confidence, fb_error = estimate_bidirectional_flow(
            frame0, frame1
        )
        oracle = true_flow(labels0, (shift1, shift2))

        valid_foreground = (
            (labels0 > 0)
            & (confidence > 0.3)
            & (fb_error < 0.1)
        )
        exact = np.mean(
            np.all(
                estimated[valid_foreground]
                == oracle[valid_foreground],
                axis=1,
            )
        )

        self.assertGreater(valid_foreground.sum(), 100)
        self.assertGreater(exact, 0.90)


if __name__ == "__main__":
    unittest.main()
