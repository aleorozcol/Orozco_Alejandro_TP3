"""Dataset splitting and normalization helpers."""

from __future__ import annotations

import numpy as np


def split_train_val_test_and_normalize(
	X_images: np.ndarray,
	y_images: np.ndarray,
	train_ratio: float = 0.70,
	val_ratio: float = 0.15,
	test_ratio: float = 0.15,
	seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Split arrays into train/validation/test subsets and normalize images to [0, 1]."""

	if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
		raise ValueError("train_ratio + val_ratio + test_ratio must be equal to 1.0")

	n_samples = X_images.shape[0]
	rng = np.random.default_rng(seed)
	indices = rng.permutation(n_samples)

	train_size = int(train_ratio * n_samples)
	val_size = int(val_ratio * n_samples)

	train_idx = indices[:train_size]
	val_idx = indices[train_size : train_size + val_size]
	test_idx = indices[train_size + val_size :]

	X_train = X_images[train_idx].astype(np.float32) / np.float32(255.0)
	X_val = X_images[val_idx].astype(np.float32) / np.float32(255.0)
	X_test = X_images[test_idx].astype(np.float32) / np.float32(255.0)

	y_train = y_images[train_idx]
	y_val = y_images[val_idx]
	y_test = y_images[test_idx]

	return X_train, y_train, X_val, y_val, X_test, y_test
