import unittest

import numpy as np

from gate6_estimated_motion_write import (
    appearance_components,
    estimate_component_translations,
    render_scene,
)


class EstimatedMotionTests(unittest.TestCase):
    def test_two_part_objects_form_five_appearance_regions(self):
        frame, _, _ = render_scene((3, 2), (13, 14), seed=0)
        components = appearance_components(frame)
        self.assertEqual(len(np.unique(components)), 5)

    def test_rgb_only_component_translation_recovers_rigid_motion(self):
        frame0, labels0, _ = render_scene((3, 2), (13, 14), seed=0)
        frame1, _, _ = render_scene((3, 3), (13, 13), seed=1)
        components = appearance_components(frame0)
        motions, confidences = estimate_component_translations(
            frame0, frame1, components
        )

        for comp in np.unique(components):
            comp = int(comp)
            owners = labels0[components == comp]
            values, counts = np.unique(owners, return_counts=True)
            owner = int(values[np.argmax(counts)])
            if owner == 1:
                self.assertEqual(motions[comp], (0, 1))
                self.assertGreater(confidences[comp], 0.9)
            elif owner == 2:
                self.assertEqual(motions[comp], (0, -1))
                self.assertGreater(confidences[comp], 0.9)
            else:
                self.assertEqual(motions[comp], (0, 0))


if __name__ == "__main__":
    unittest.main()
