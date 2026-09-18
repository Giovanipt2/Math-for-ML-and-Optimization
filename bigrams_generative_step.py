import random
import string
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd


ALPHABET = string.ascii_lowercase


def load_bigrams_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load the semicolon-separated bigram frequency dataset.

    Args:
        file_path: Path to the dataset, whose rows contain ``bigram;frequency``.

    Returns:
        A DataFrame with ``bigram`` and ``frequency`` columns.

    Raises:
        ValueError: If the dataset does not contain all 26^2 lowercase bigrams.
    """
    df = pd.read_csv(
        file_path,
        sep=";",
        skiprows=1,
        names=["bigram", "frequency"],
    )

    expected_bigrams = {first + second for first in ALPHABET for second in ALPHABET}
    actual_bigrams = set(df["bigram"])
    if actual_bigrams != expected_bigrams or len(df) != len(expected_bigrams):
        raise ValueError("The dataset must contain each lowercase bigram exactly once.")

    if (df["frequency"] < 0).any():
        raise ValueError("Bigram frequencies cannot be negative.")

    return df


def build_bigrams_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Build a 26 by 26 matrix of conditional bigram probabilities.

    Args:
        df: DataFrame containing ``bigram`` and ``frequency`` columns.

    Returns:
        A NumPy array whose rows and columns are alphabetically ordered.
    """
    frequency_matrix = np.zeros((len(ALPHABET), len(ALPHABET)), dtype=float)
    alphabet_index = {letter: index for index, letter in enumerate(ALPHABET)}

    for row in df.itertuples(index=False):
        first_letter, second_letter = row.bigram
        frequency_matrix[
            alphabet_index[first_letter], alphabet_index[second_letter]
        ] = row.frequency

    row_totals = frequency_matrix.sum(axis=1)
    if np.any(row_totals == 0):
        raise ValueError("Every first letter must have a positive total frequency.")

    return frequency_matrix / row_totals[:, np.newaxis]


def build_cumulative_matrix(probability_matrix: np.ndarray) -> np.ndarray:
    """
    Convert conditional probabilities into cumulative probabilities by row.

    Args:
        probability_matrix: Matrix of conditional probabilities by bigram.

    Returns:
        A cumulative probability matrix suitable for random sampling.
    """
    return np.cumsum(probability_matrix, axis=1)


def generate_sequence(
    cumulative_matrix: np.ndarray,
    sequence_length: int,
    rng: random.Random | None = None,
) -> str:
    """
    Generate a character sequence using cumulative bigram probabilities.

    Args:
        cumulative_matrix: Cumulative conditional probability matrix.
        sequence_length: Number of characters to generate.
        rng: Optional random number generator for reproducible sampling.

    Returns:
        A generated sequence containing only lowercase English letters.

    Raises:
        ValueError: If the requested sequence length is not positive.
    """
    if sequence_length <= 0:
        raise ValueError("The sequence length must be positive.")

    generator = rng or random
    current_letter = generator.choice(ALPHABET)
    generated_letters = [current_letter]
    alphabet_index = {letter: index for index, letter in enumerate(ALPHABET)}

    for _ in range(sequence_length - 1):
        random_value = generator.random()
        row = cumulative_matrix[alphabet_index[current_letter]]
        next_index = int(np.searchsorted(row, random_value, side="right"))
        next_index = min(next_index, len(ALPHABET) - 1)
        current_letter = ALPHABET[next_index]
        generated_letters.append(current_letter)

    return "".join(generated_letters)


def parse_arguments() -> tuple[Path, int]:
    """
    Parse the dataset path and requested generated sequence length.

    Returns:
        A tuple containing the dataset path and sequence length.
    """
    parser = ArgumentParser(description="Generate a sequence from bigram frequencies.")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("bigrams.csv"),
        help="Path to the bigram frequency dataset.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=100,
        help="Number of characters to generate.",
    )
    arguments = parser.parse_args()
    return arguments.file, arguments.length


def main() -> None:
    file_path, sequence_length = parse_arguments()
    dataset = load_bigrams_dataset(file_path)

    probability_matrix = build_bigrams_matrix(dataset)
    cumulative_matrix = build_cumulative_matrix(probability_matrix)
    
    sequence = generate_sequence(cumulative_matrix, sequence_length)

    print(sequence)


if __name__ == "__main__":
    main()
