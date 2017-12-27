import unittest
import tools


class ToolsTest(unittest.TestCase):
    def test_image_unrolling_bad_data(self):
        with self.assertRaises(AssertionError):
            tools.unroll_image_data([[[1]], [[2]], [[3]], [[4]]])

    def test_image_unrolling(self):
        test_rgb_mat = [[[111, 121, 131],  # r
                        [211, 221, 231],
                        [311, 321, 331]],
                        [[112, 122, 132],  # g
                        [212, 222, 232],
                        [312, 322, 332]],
                        [[113, 123, 133],  # b
                        [213, 223, 233],
                        [313, 323, 333]]]
        rgb_vector = tools.unroll_image_data(test_rgb_mat)
        self.assertTrue(self._data_preserved(test_rgb_mat, rgb_vector))

    def _data_preserved(self, matrix, vector):
        v_idx = 0
        for component in matrix:
            for row in component:
                for i in range(len(row)):
                    if row[i] != vector[v_idx]:
                        return False
                    v_idx = v_idx + 1
        return True