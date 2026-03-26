import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3

        unique_word = set()

        for text in texts:
            unique_word.update(text.split())
        
        for index, value in enumerate(list(sorted(unique_word))):
            self.word_to_id[value] = index + 4

        self.id_to_word = dict(zip(self.word_to_id.values(),self.word_to_id.keys()))
        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        words = text.split()
        tokens = []

        for word in words:
            tokens.append(self.word_to_id.get(word, 1))

        return tokens    
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        text = []

        for i in range(len(ids)):
            text.append(self.id_to_word[ids[i]])

        return ' '.join(text)    
