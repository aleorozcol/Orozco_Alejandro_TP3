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


def plot_validation_loss_improvement(
	val_loss_m0: Sequence[float],
	val_loss_m1: Sequence[float],
	baseline_label: str = "M0",
	improved_label: str = "M1",
	title: str = "Mejora de validation loss entre M0 y M1",
	figsize: tuple[int, int] = (11, 8),
) -> None:
	"""Plot validation loss curves and epoch-by-epoch improvement of M1 over M0.

	If the histories have different lengths, the plot uses the common prefix so
	the comparison still works when one model stops early.
	"""

	val_loss_m0_array = np.asarray(val_loss_m0, dtype=float)
	val_loss_m1_array = np.asarray(val_loss_m1, dtype=float)

	if val_loss_m0_array.size == 0 or val_loss_m1_array.size == 0:
		raise ValueError("val_loss_m0 and val_loss_m1 cannot be empty")
	min_len = min(val_loss_m0_array.size, val_loss_m1_array.size)
	val_loss_m0_array = val_loss_m0_array[:min_len]
	val_loss_m1_array = val_loss_m1_array[:min_len]

	epochs = np.arange(1, val_loss_m0_array.size + 1)
	improvement_pct = 100 * (val_loss_m0_array - val_loss_m1_array) / val_loss_m0_array
	final_improvement = improvement_pct[-1]

	fig, (ax_loss, ax_improvement) = plt.subplots(2, 1, figsize=figsize, sharex=True)

	ax_loss.plot(epochs, val_loss_m0_array, label=baseline_label, color="#6c757d", linewidth=2)
	ax_loss.plot(epochs, val_loss_m1_array, label=improved_label, color="#1f77b4", linewidth=2)
	ax_loss.set_title(title, fontsize=14)
	ax_loss.set_ylabel("Validation Loss", fontsize=12)
	ax_loss.grid(True, linestyle=":", alpha=0.7)
	ax_loss.legend(fontsize=11)
	ax_loss.annotate(
		f"Mejora final: {final_improvement:+.2f}%",
		xy=(epochs[-1], val_loss_m1_array[-1]),
		xytext=(10, 12),
		textcoords="offset points",
		fontsize=10,
		color="#1f77b4",
		bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1f77b4", alpha=0.9),
	)

	ax_improvement.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.6)
	ax_improvement.plot(epochs, improvement_pct, color="#2ca02c", linewidth=2)
	ax_improvement.fill_between(epochs, 0, improvement_pct, where=improvement_pct >= 0, color="#2ca02c", alpha=0.15)
	ax_improvement.fill_between(epochs, 0, improvement_pct, where=improvement_pct < 0, color="#d62728", alpha=0.15)
	ax_improvement.set_xlabel("Épocas", fontsize=12)
	ax_improvement.set_ylabel("Mejora vs M0 (%)", fontsize=12)
	ax_improvement.grid(True, linestyle=":", alpha=0.7)

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


def plot_confusion_matrix_comparison(
	cm_baseline: np.ndarray,
	cm_improved: np.ndarray,
	class_names: Sequence[str] | None = None,
	baseline_label: str = "M0",
	improved_label: str = "M1",
	title: str = "Comparación de heatmaps de la matriz de confusión",
	figsize: tuple[int, int] = (18, 8),
	cmap: str = "Blues",
	annot: bool = False,
) -> None:
	"""Plot baseline and improved confusion matrices side by side with a shared color scale."""

	if class_names is None:
		class_names = get_emnist_balanced_class_names()

	cm_baseline = np.asarray(cm_baseline)
	cm_improved = np.asarray(cm_improved)
	if cm_baseline.shape != cm_improved.shape:
		raise ValueError("cm_baseline and cm_improved must have the same shape")

	vmax = max(float(np.max(cm_baseline)), float(np.max(cm_improved)))
	fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

	sns.heatmap(
		cm_baseline,
		ax=axes[0],
		annot=annot,
		cmap=cmap,
		fmt="g",
		vmin=0,
		vmax=vmax,
		cbar=True,
		xticklabels=class_names,
		yticklabels=class_names,
	)
	axes[0].set_title(f"{baseline_label}", fontsize=14)
	axes[0].set_xlabel("Clase Predicha", fontsize=12)
	axes[0].set_ylabel("Clase Real", fontsize=12)

	sns.heatmap(
		cm_improved,
		ax=axes[1],
		annot=annot,
		cmap=cmap,
		fmt="g",
		vmin=0,
		vmax=vmax,
		cbar=True,
		xticklabels=class_names,
		yticklabels=class_names,
	)
	axes[1].set_title(f"{improved_label}", fontsize=14)
	axes[1].set_xlabel("Clase Predicha", fontsize=12)
	axes[1].set_ylabel("Clase Real", fontsize=12)

	fig.suptitle(title, fontsize=16)
	plt.tight_layout()
	plt.show()


def plot_ablation_loss_curves(
	histories: Sequence[dict],
	baseline_name: str = "M0",
	figsize: tuple[int, int] = (16, 10),
	columns: int = 2,
) -> None:
	"""Plot train and validation loss curves for multiple training variants."""

	if not histories:
		raise ValueError("histories cannot be empty")

	rows = int(np.ceil(len(histories) / columns))
	fig, axes = plt.subplots(rows, columns, figsize=figsize, squeeze=False)
	axes_array = axes.flatten()

	for ax, history in zip(axes_array, histories):
		train_loss = history.get("train_loss", [])
		val_loss = history.get("val_loss", [])
		name = history.get("name", "Experimento")

		ax.plot(train_loss, label="Train Cross-Entropy", color="blue", linewidth=2, linestyle="--")
		ax.plot(val_loss, label="Validation Cross-Entropy", color="red", linewidth=2)
		ax.set_title(name, fontsize=12)
		ax.set_xlabel("Épocas")
		ax.set_ylabel("Cross-Entropy Loss")
		ax.grid(True, linestyle=":", alpha=0.7)
		ax.legend(fontsize=9)

	for ax in axes_array[len(histories):]:
		ax.axis("off")

	fig.suptitle(f"Evolución de la función de costo por mejora (baseline: {baseline_name})", fontsize=16)
	plt.tight_layout()
	plt.show()
