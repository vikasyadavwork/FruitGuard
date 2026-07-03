"""Independent BGR-HWC to RGB-CHW float32 transform; never feed this into YOLO.

The native bridge returns host memory and includes both PCIe transfers. Reuse a
CudaPreprocessor context for repeated calls; preprocess_bgr is a one-shot helper.
"""

import ctypes
from pathlib import Path
from typing import Optional, Union

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIBRARY = PROJECT_ROOT / "build" / "fruitguard_preprocess.dll"


def _image(frame: np.ndarray) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise ValueError("Input must be a NumPy uint8 BGR image")
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0] < 1 or frame.shape[1] < 1:
        raise ValueError("Input must have nonempty HWC shape with three BGR channels")
    if frame.shape[0] * frame.shape[1] > 2147483647:
        raise ValueError("Input image is too large for this CUDA implementation")
    return np.ascontiguousarray(frame)


class CudaPreprocessor:
    """Synchronous caller-owned CUDA context; not safe for concurrent calls."""

    def __init__(self, library_path: Optional[Union[str, Path]] = None,
                 device_index: int = 0):
        if isinstance(device_index, bool) or not isinstance(device_index, int) or device_index < 0:
            raise ValueError("device_index must be a nonnegative integer")
        path = Path(library_path) if library_path is not None else DEFAULT_LIBRARY
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise RuntimeError("CUDA preprocessing library missing: {}. Run scripts/build_cuda.ps1".format(path))
        try:
            self._library = ctypes.CDLL(str(path.resolve()))
        except OSError as error:
            raise RuntimeError("Cannot load CUDA preprocessing library {}: {}".format(path, error)) from error
        self._library.fg_create.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
        self._library.fg_create.restype = ctypes.c_void_p
        self._library.fg_preprocess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte),
                                               ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                                               ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
        self._library.fg_preprocess.restype = ctypes.c_int
        self._library.fg_destroy.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        self._library.fg_destroy.restype = ctypes.c_int
        self._error = ctypes.create_string_buffer(1024)
        self._context = self._library.fg_create(device_index, self._error, len(self._error))
        if not self._context:
            raise RuntimeError("CUDA preprocessing initialization failed: " + self._message())

    def _message(self) -> str:
        return self._error.value.decode("utf-8", errors="replace")

    def process(self, frame: np.ndarray) -> np.ndarray:
        if not self._context:
            raise RuntimeError("CUDA preprocessing context is closed")
        image = _image(frame)
        height, width = image.shape[:2]
        output = np.empty((3, height, width), dtype=np.float32)
        code = self._library.fg_preprocess(
            self._context, image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), width, height,
            self._error, len(self._error))
        if code:
            raise RuntimeError("CUDA preprocessing failed: " + self._message())
        return output

    def close(self) -> None:
        if self._context:
            code = self._library.fg_destroy(self._context, self._error, len(self._error))
            self._context = None
            if code:
                raise RuntimeError("CUDA preprocessing cleanup failed: " + self._message())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def preprocess_bgr(frame: np.ndarray, backend: str = "cpu",
                   library_path: Optional[Union[str, Path]] = None,
                   device_index: int = 0) -> np.ndarray:
    """Return RGB float32 CHW 0..1. CUDA requests never fall back to CPU."""
    image = _image(frame)
    if backend == "cpu":
        return np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / np.float32(255.0)
    if backend == "cuda":
        with CudaPreprocessor(library_path, device_index) as context:
            return context.process(image)
    raise ValueError("backend must be 'cpu' or 'cuda'")
