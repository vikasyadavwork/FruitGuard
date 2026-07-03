"""Transform behavior only, not apple accuracy; CUDA checks are opt-in."""

import os
import unittest

import numpy as np

from src.vision.preprocessing import CudaPreprocessor, preprocess_bgr


class PreprocessingTests(unittest.TestCase):
    def test_channel_order_layout_and_endpoints(self):
        image = np.array([[[0, 128, 255], [255, 0, 64]]], dtype=np.uint8)
        result = preprocess_bgr(image)
        expected = np.array([[[1.0, 64 / 255.0]], [[128 / 255.0, 0]], [[0, 1]]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-7)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(result.flags.c_contiguous)

    def test_noncontiguous_input_and_odd_tail(self):
        image = np.arange(17 * 19 * 3, dtype=np.uint8).reshape(17, 19, 3)[::2, ::-1]
        result = preprocess_bgr(image)
        self.assertEqual(result.shape, (3, 9, 19))
        np.testing.assert_allclose(result[0], image[:, :, 2] / 255.0, rtol=0, atol=1e-7)

    def test_invalid_inputs(self):
        for image in [None, np.zeros((3, 3)), np.zeros((2, 2, 4), np.uint8),
                      np.zeros((0, 2, 3), np.uint8), np.zeros((3, 3, 3), np.float32)]:
            with self.assertRaises(ValueError):
                preprocess_bgr(image)

    def test_unknown_backend_and_explicit_missing_cuda_library(self):
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            preprocess_bgr(image, "automatic")
        with self.assertRaises(RuntimeError):
            preprocess_bgr(image, "cuda", "build/missing_for_test.dll")

    @unittest.skipUnless(os.getenv("FRUITGUARD_TEST_CUDA") == "1", "Set FRUITGUARD_TEST_CUDA=1 for real CUDA checks")
    def test_real_cuda_context_reuse_resize_and_cpu_comparison(self):
        with CudaPreprocessor() as context:
            for height, width in [(1, 1), (17, 19), (640, 640), (2, 3)]:
                image = (np.arange(height * width * 3, dtype=np.uint32) * 73 + 29).astype(np.uint8).reshape(height, width, 3)
                expected = preprocess_bgr(image)
                actual = context.process(image)
                np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
        context.close()
        with self.assertRaises(RuntimeError):
            context.process(image)

    @unittest.skipUnless(os.getenv("FRUITGUARD_TEST_CUDA") == "1", "Set FRUITGUARD_TEST_CUDA=1 for real CUDA checks")
    def test_unavailable_cuda_device_is_a_diagnostic(self):
        with self.assertRaisesRegex(RuntimeError, "cudaSetDevice"):
            CudaPreprocessor(device_index=999999)
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        with CudaPreprocessor() as context:
            np.testing.assert_array_equal(context.process(image), preprocess_bgr(image))


if __name__ == "__main__":
    unittest.main()
