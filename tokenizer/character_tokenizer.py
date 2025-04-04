from .tokenizer import Tokenizer
from typing import List, Dict, Tuple, Optional

import torch

class CharacterTokenizer(Tokenizer):
    def __init__(self, verbose: bool = False):
        """
        Initializes the CharacterTokenizer class for French to English translation.
        We ignore capitalization.

        Implement the remaining parts of __init__ by building the vocab.
        Implement the two functions you defined in Tokenizer here. Once you are
        done, you should pass all the tests in test_character_tokenizer.py.
        """
        super().__init__()

        self.vocab = {}

        # Normally, we iterate through the dataset and find all unique characters. To simplify things,
        # we will use a fixed set of characters that we know will be present in the dataset.
        self.characters = "aàâæbcçdeéèêëfghiîïjklmnoôœpqrstuùûüvwxyÿz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "

    def encode(self, text):
        """
        Encodes the input text into a list of tokens (characters).
        """
        encoded = []
        for char in list(text.lower()):
            encoded.append(self.characters.index(char))
            
        return encoded
        
    def decode(self, tokens):
        """
        Decodes a list of tokens (characters) back into the original text.
        """
        # Join the tokens to form the original text
        decoded = ""
        for num in tokens:
            decoded += str(self.characters[num])
        return decoded

