"""Utility helpers for notebook visualizations."""

from __future__ import annotations

import string
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def get_emnist_balanced_class_names() -> list[str]:
	"""Return the EMNIST Balanced class names in torchvision order."""

	excluded_letters = {"c", "i", "j", "k", "l", "m", "o", "p", "s", "u", "v", "w", "x", "y", "z"}
	class_names = sorted(list(set(string.digits + string.ascii_letters) - excluded_letters))
	return class_names


def plot_random_emnist_samples(
	X_images: np.ndarray,
	y_images: np.ndarray,
	n_samples: int = 9,
	seed: int | None = None,
	figsize: tuple[int, int] = (10, 10),
	cmap: str = "gray",
	title_fontsize: int = 10,
	class_names: Sequence[str] | None = None,
) -> None:
	"""Plot random EMNIST samples in a square grid with class and character labels."""

	if n_samples <= 0:
		raise ValueError("n_samples must be greater than 0")
	if n_samples > len(X_images):
		raise ValueError("n_samples cannot exceed the number of available images")

	if class_names is None:
		class_names = get_emnist_balanced_class_names()

	rng = np.random.default_rng(seed)
	random_indices = rng.choice(len(X_images), size=n_samples, replace=False)

	grid_size = int(np.ceil(np.sqrt(n_samples)))
	fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
	axes_array = np.atleast_1d(axes).reshape(-1)

	for ax in axes_array[n_samples:]:
		ax.axis("off")

	for ax, idx in zip(axes_array, random_indices):
		img = X_images[idx].reshape(28, 28)
		label = int(y_images[idx])
		char = class_names[label] if 0 <= label < len(class_names) else "?"
		ax.imshow(img, cmap=cmap)
		ax.set_title(f"Clase: {label} | Carácter: {char}", fontsize=title_fontsize)
		ax.axis("off")

	plt.tight_layout()
	plt.show()


def plot_emnist_character_distribution(
	y_images: np.ndarray,
	class_names: Sequence[str] | None = None,
	figsize: tuple[int, int] = (14, 6),
	color: str = "steelblue",
	title: str = "Cantidad de imágenes por carácter",
) -> None:
	"""Plot a bar chart with the number of samples for each EMNIST character."""

	if class_names is None:
		class_names = get_emnist_balanced_class_names()

	counts = np.bincount(y_images.astype(int), minlength=len(class_names))
	x_positions = np.arange(len(class_names))

	fig, ax = plt.subplots(figsize=figsize)
	ax.bar(x_positions, counts, color=color)
	ax.set_title(title)
	ax.set_xlabel("Carácter")
	ax.set_ylabel("Cantidad de imágenes")
	ax.set_xticks(x_positions)
	ax.set_xticklabels(class_names)
	ax.tick_params(axis="x", rotation=90)
	ax.grid(axis="y", linestyle="--", alpha=0.3)
	plt.tight_layout()
	plt.show()


def plot_loss_curves(
	train_loss: Sequence[float],
	val_loss: Sequence[float],
	title: str,
	figsize: tuple[int, int] = (10, 6),
	train_label: str = "Train Cross-Entropy",
	val_label: str = "Validation Cross-Entropy",
	train_color: str = "blue",
	val_color: str = "red",
) -> None:
	"""Plot train and validation loss curves."""

	plt.figure(figsize=figsize)
	plt.plot(train_loss, label=train_label, color=train_color, linewidth=2, linestyle="--")
	plt.plot(val_loss, label=val_label, color=val_color, linewidth=2)
	plt.title(title, fontsize=14)
	plt.xlabel("Épocas", fontsize=12)
	plt.ylabel("Cross-Entropy Loss", fontsize=12)
	plt.legend(fontsize=12)
	plt.grid(True, linestyle=":", alpha=0.7)
	plt.tight_layout()
	plt.show()


def plot_confusion_matrix_with_characters(
	cm: np.ndarray,
	class_names: Sequence[str] | None = None,
	title: str = "Matriz de Confusión",
	figsize: tuple[int, int] = (16, 12),
	cmap: str = "Blues",
	annot: bool = False,
) -> None:
	"""Plot a confusion matrix using EMNIST characters as axis labels."""

	if class_names is None:
		class_names = get_emnist_balanced_class_names()

	plt.figure(figsize=figsize)
	sns.heatmap(cm, annot=annot, cmap=cmap, fmt="g", cbar=True, xticklabels=class_names, yticklabels=class_names)
	plt.title(title, fontsize=16)
	plt.xlabel("Clase Predicha", fontsize=14)
	plt.ylabel("Clase Real", fontsize=14)
	plt.tight_layout()
	plt.show()
