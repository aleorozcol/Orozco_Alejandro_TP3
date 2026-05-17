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
	baseline_val_loss: Sequence[float],
	histories: Sequence[dict],
	baseline_name: str = "M0",
	figsize: tuple[int, int] = (12, 6),
	max_epochs: int = 20,
) -> None:
	"""Plot a single validation-loss chart for the baseline and improvement runs."""

	if not histories:
		raise ValueError("histories cannot be empty")

	if len(baseline_val_loss) == 0:
		raise ValueError("baseline_val_loss cannot be empty")

	plt.figure(figsize=figsize)
	colormap = plt.cm.tab10(np.linspace(0, 1, len(histories) + 1))

	# Clip histories to max_epochs for plotting
	max_epochs = int(max_epochs) if max_epochs is not None else None

	baseline_vals = np.asarray(baseline_val_loss, dtype=float)
	if max_epochs is not None and baseline_vals.size > max_epochs:
		baseline_vals = baseline_vals[:max_epochs]

	epochs_baseline = np.arange(1, baseline_vals.size + 1)
	plt.plot(
		epochs_baseline,
		baseline_vals,
		label=baseline_name,
		color=colormap[0],
		linewidth=2.5,
	)

	for idx, history in enumerate(histories, start=1):
		val_loss = np.asarray(history.get("val_loss", []), dtype=float)
		if max_epochs is not None and val_loss.size > max_epochs:
			val_loss = val_loss[:max_epochs]
		name = history.get("name", "Experimento")
		epochs = np.arange(1, val_loss.size + 1)
		plt.plot(
			epochs,
			val_loss,
			label=name,
			color=colormap[idx],
			linewidth=2,
		)

	plt.title(f"Evolución de la validación por mejora (baseline: {baseline_name})", fontsize=16)
	plt.xlabel("Épocas")
	plt.ylabel("Validation Cross-Entropy")
	plt.grid(True, linestyle=":", alpha=0.7)
	plt.legend(fontsize=9, ncol=2)
	plt.tight_layout()
	plt.show()


def plot_numpy_pytorch_loss_comparison(
	train_loss_m1: Sequence[float],
	train_loss_m2: Sequence[float],
	val_loss_m1: Sequence[float],
	val_loss_m2: Sequence[float],
	model_1_label: str = "M1 (NumPy)",
	model_2_label: str = "M2 (PyTorch)",
	title_train: str = "Comparativa de Pérdida: Entrenamiento",
	title_val: str = "Comparativa de Pérdida: Validación",
	figsize: tuple[int, int] = (14, 6),
) -> None:
	"""Plot train and validation loss comparison between two models and print a summary verdict."""

	plt.figure(figsize=figsize)

	plt.subplot(1, 2, 1)
	plt.plot(train_loss_m1, label=f"{model_1_label} - Train", color="darkblue", linewidth=2)
	plt.plot(train_loss_m2, label=f"{model_2_label} - Train", color="cyan", linestyle="--", linewidth=2)
	plt.title(title_train, fontsize=12)
	plt.xlabel("Épocas")
	plt.ylabel("Cross-Entropy Loss")
	plt.legend()
	plt.grid(True, linestyle=":", alpha=0.6)

	plt.subplot(1, 2, 2)
	plt.plot(val_loss_m1, label=f"{model_1_label} - Val", color="darkorange", linewidth=2)
	plt.plot(val_loss_m2, label=f"{model_2_label} - Val", color="red", linestyle="--", linewidth=2)
	plt.title(title_val, fontsize=12)
	plt.xlabel("Épocas")
	plt.ylabel("Cross-Entropy Loss")
	plt.legend()
	plt.grid(True, linestyle=":", alpha=0.6)

	plt.tight_layout()
	plt.show()

	print(f"{model_1_label}   -> Pérdida Mínima en Validación: {min(val_loss_m1):.4f}")
	print(f"{model_2_label} -> Pérdida Mínima en Validación: {min(val_loss_m2):.4f}")
	print(f"Diferencia absoluta: {abs(min(val_loss_m1) - min(val_loss_m2)):.5f}")


def plot_metrics_bar_comparison(
	df_resultados,
	title: str = "Performance Comparativa en Test Set (EMNIST ByMerge)",
	figsize: tuple[int, int] = (10, 6),
	accuracy_column: str = "Accuracy",
	f1_column: str = "F1-Score Macro",
	model_column: str = "Modelo",
) -> None:
	"""Plot grouped bars for accuracy and macro F1 score."""

	x = np.arange(len(df_resultados[model_column]))
	width = 0.35

	fig, ax = plt.subplots(figsize=figsize)
	rects1 = ax.bar(x - width / 2, df_resultados[accuracy_column], width, label="Accuracy", color="skyblue", edgecolor="black")
	rects2 = ax.bar(x + width / 2, df_resultados[f1_column], width, label="F1-Score Macro", color="salmon", edgecolor="black")

	ax.set_ylabel("Puntuación (0 a 1)", fontsize=12)
	ax.set_title(title, fontsize=14, fontweight="bold")
	ax.set_xticks(x)
	ax.set_xticklabels(df_resultados[model_column], fontsize=11)
	ax.set_ylim(0, 1.1)
	ax.legend(loc="upper left", fontsize=11)
	ax.grid(axis="y", linestyle="--", alpha=0.7)

	def autolabel(rects: Sequence) -> None:
		"""Añade una etiqueta de texto encima de cada barra."""
		for rect in rects:
			height = rect.get_height()
			ax.annotate(
				f"{height:.3f}",
				xy=(rect.get_x() + rect.get_width() / 2, height),
				xytext=(0, 3),
				textcoords="offset points",
				ha="center",
				va="bottom",
				fontsize=10,
			)

	autolabel(rects1)
	autolabel(rects2)

	plt.tight_layout()
	plt.show()


def apply_gaussian_noise(X: np.ndarray, noise_factor: float) -> np.ndarray:
	"""Add Gaussian noise to an image or batch and clip to [0, 1]."""

	ruido = np.random.normal(loc=0.0, scale=noise_factor, size=X.shape)
	X_ruidoso = X + ruido
	return np.clip(X_ruidoso, 0.0, 1.0)


def plot_noise_robustness(
	acc_history: dict,
	niveles_ruido: Sequence[float],
	X_test_flat: np.ndarray,
	apply_noise_fn=apply_gaussian_noise,
	figsize: tuple[int, int] = (10, 6),
	samples_figsize: tuple[int, int] = (15, 3),
	title: str = "Robustez de los Modelos frente a Ruido Gaussiano",
	sample_title: str = "Ejemplo de degradación visual",
	x_label: str = "Nivel de Ruido (Desviación Estándar $\\sigma$)",
	y_label: str = "Accuracy en Test",
) -> None:
	"""Plot accuracy degradation under noise and show example noisy images."""

	plt.figure(figsize=figsize)

	colores = {
		"M0 (Base)": "blue",
		"M1 (Avanzado)": "green",
		"M2 (PyTorch)": "orange",
		"M3 (PyTorch Avanzado)": "red",
	}

	estilos = {
		"M0 (Base)": ":",
		"M1 (Avanzado)": "--",
		"M2 (PyTorch)": "-.",
		"M3 (PyTorch Avanzado)": "-",
	}

	for nombre, accuracies in acc_history.items():
		plt.plot(
			niveles_ruido,
			accuracies,
			label=nombre,
			color=colores.get(nombre, "black"),
			linestyle=estilos.get(nombre, "-"),
			marker="o",
			linewidth=2.5,
			markersize=8,
		)

	plt.title(title, fontsize=14, fontweight="bold")
	plt.xlabel(x_label, fontsize=12)
	plt.ylabel(y_label, fontsize=12)
	plt.xticks(niveles_ruido)
	plt.ylim(0, 1.05)
	plt.legend(fontsize=11)
	plt.grid(True, linestyle="--", alpha=0.7)
	plt.tight_layout()
	plt.show()

	fig, axes = plt.subplots(1, len(niveles_ruido), figsize=samples_figsize)
	axes_array = np.atleast_1d(axes)
	muestra_original = X_test_flat[0]

	for ax, nl in zip(axes_array, niveles_ruido):
		muestra_ruidosa = apply_noise_fn(muestra_original, nl)
		ax.imshow(muestra_ruidosa.reshape(28, 28), cmap="gray")
		ax.set_title(f"Ruido: {nl}")
		ax.axis("off")

	fig.suptitle(sample_title, fontsize=14)
	plt.tight_layout()
	plt.show()
