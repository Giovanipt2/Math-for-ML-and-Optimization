import matplotlib.pyplot as plt
import numpy as np


NUMBER_OF_PAIRS = 1000
VECTOR_DIMENSION = 100


def main() -> None:
	rng = np.random.default_rng()

	# Cada linha representa um vetor aleatório real.
	u = rng.standard_normal((NUMBER_OF_PAIRS, VECTOR_DIMENSION))
	v = rng.standard_normal((NUMBER_OF_PAIRS, VECTOR_DIMENSION))

	correlations = np.array(
		[np.corrcoef(u_i, v_i)[0, 1] for u_i, v_i in zip(u, v)]
	)

	print(f"Número de correlações calculadas: {len(correlations)}")
	print(f"Média das correlações: {correlations.mean():.4f}")

	plt.hist(correlations, bins=30, edgecolor="black")
	plt.xlabel("Correlação de Pearson")
	plt.ylabel("Frequência")
	plt.title("Distribuição das correlações entre vetores aleatórios")
	plt.grid(axis="y", alpha=0.3)
	plt.show()


if __name__ == "__main__":
	main()
