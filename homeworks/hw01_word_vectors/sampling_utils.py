
from typing import Dict


def subsample_frequent_words(word_counts: Dict[str, int], t: float = 1e-5) -> Dict[str, float]:
    """
    Вычисляет вероятность оставить каждое слово на основе сабсэмплинга частых слов.

    Args:
        word_counts (Dict[str, int]): Словарь с количеством повторений слов {слово: количество}.
        t (float): Порог частоты для нормировки.

    Returns:
        Dict[str, float]: Словарь с вероятностями оставить слово {слово: вероятность}.
    """
    total_count = sum(word_counts.values())

    keep_probabilities = {}
    for word, count in word_counts.items():
        freq = count / total_count
        keep_prob = min(1.0, (t / freq) ** 0.5)
        keep_probabilities[word] = keep_prob

    return keep_probabilities


def get_negative_sampling_prob(word_count_dict):
    """
    Calculates the negative sampling probabilities for words based on their frequencies.

    This function adjusts the frequency of each word raised to the power of 0.75, which is
    commonly used in algorithms like Word2Vec to moderate the influence of very frequent words.
    It then normalizes these adjusted frequencies to ensure they sum to 1, forming a probability
    distribution used for negative sampling.

    Parameters:
    - word_count_dict (dict): A dictionary where keys are words and values are the counts of those words.

    Returns:
    - dict: A dictionary where keys are words and values are the probabilities of selecting each word
            for negative sampling.

    Example:
    >>> word_counts = {'the': 5000, 'is': 1000, 'apple': 50}
    >>> get_negative_sampling_prob(word_counts)
    {'the': 0.298, 'is': 0.160, 'apple': 0.042}
    """

    neg_sample_probs = {word: count**(3/4) for word, count in word_count_dict.items() }
    Z = sum(neg_sample_probs.values())
    neg_sample_probs = {word: freq/Z for word, freq in neg_sample_probs.items()}

    return neg_sample_probs
