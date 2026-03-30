from collections import Counter
def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    # Your code here
    words = list()

    for sentence in sentences:
        words.extend((sentence))

    return dict(Counter(words))