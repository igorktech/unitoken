import string
from typing import Dict, List, Union
import re

UNICODE_CODE_POINTS = 1114112
SPECIAL_TOKEN_TO_ID: Dict[str, int] = {
    "<pad>": 0,
    "<cls>": 0xE000,
    "<sep>": 0xE001,
    "<s>": 0xE002,
    "</s>": 0xE003,
    "<unk>": 0xE004,
    "<mask>": 0xE005,
}
ID_TO_SPECIAL_TOKEN: Dict[int, str] = {
    0: "<pad>",
    0xE000: "<cls>",
    0xE001: "<sep>",
    0xE002: "<s>",
    0xE003: "</s>",
    0xE004: "<unk>",
    0xE005: "<mask>",
}


class SimpleUniTokenizer:
    def __init__(self, truncation_side: str = "left"):
        self.pad_token = chr(SPECIAL_TOKEN_TO_ID["<pad>"])
        self.bos_token = chr(SPECIAL_TOKEN_TO_ID["<s>"])
        self.eos_token = chr(SPECIAL_TOKEN_TO_ID["</s>"])
        self.unk_token = chr(SPECIAL_TOKEN_TO_ID["<unk>"])
        self.sep_token = chr(SPECIAL_TOKEN_TO_ID["<sep>"])
        self.cls_token = chr(SPECIAL_TOKEN_TO_ID["<cls>"])
        self.mask_token = chr(SPECIAL_TOKEN_TO_ID["<mask>"])

        self.pad_token_id = SPECIAL_TOKEN_TO_ID[self.pad_token]
        self.bos_token_id = SPECIAL_TOKEN_TO_ID[self.bos_token]
        self.eos_token_id = SPECIAL_TOKEN_TO_ID[self.eos_token]
        self.unk_token_id = SPECIAL_TOKEN_TO_ID[self.unk_token]
        self.sep_token_id = SPECIAL_TOKEN_TO_ID[self.sep_token]
        self.cls_token_id = SPECIAL_TOKEN_TO_ID[self.cls_token]
        self.mask_token_id = SPECIAL_TOKEN_TO_ID[self.mask_token]

        self._unicode_vocab_size = UNICODE_CODE_POINTS
        self._num_special_tokens = len(SPECIAL_TOKEN_TO_ID)

        self._special_token_to_id = {chr(k): k for k in ID_TO_SPECIAL_TOKEN.keys()}
        self._id_to_special_token = ID_TO_SPECIAL_TOKEN

        # self.pattern = re.compile(
        #     r"(" + ''.join([f"\s*{re.escape(token)}\s*|" for token in SPECIAL_TOKEN_TO_ID.keys()]) + r".)")

        self.truncation_side = truncation_side

    @property
    def vocab_size(self) -> int:
        return self._unicode_vocab_size

    def __call__(
            self,
            text: Union[str, List[str]],
            add_special_tokens: bool = True,
            truncation: bool = False,
            max_length: int = None,
            pad_to_max_length: bool = False
    ) -> Union[List[int], List[List[int]]]:

        if isinstance(text, str):
            tokenized_text = [self.tokenize(text)]
        else:
            tokenized_text = [self.tokenize(t) for t in text]

        return self.encode_batch(tokenized_text,
                                 add_special_tokens=add_special_tokens,
                                 truncation=truncation,
                                 max_length=max_length,
                                 pad_to_max_length=pad_to_max_length)

    def tokenize(self, text: str) -> List[str]:
        # return self.pattern.findall(text)
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._special_token_to_id.get(token, ord(token))

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_special_token.get(index, chr(index))

    def truncate_sequences(
            self,
            ids: List[int],
            max_length: int,
            add_special_tokens: bool = True
    ) -> List[int]:

        if self.truncation_side == "left":
            truncated_ids = ids[-max_length + 2:] if add_special_tokens else ids[-max_length:]
        else:
            truncated_ids = ids[:max_length - 2] if add_special_tokens else ids[:max_length]
        return truncated_ids

    def pad(self, ids: List[int], max_length: int) -> List[int]:
        pad_length = max_length - len(ids)
        if pad_length > 0:
            return ids + [self.pad_token_id] * pad_length
        return ids

    def encode(
            self,
            text: Union[str, List[str]],
            add_special_tokens: bool = True,
            truncation: bool = False,
            max_length: int = None,
            pad_to_max_length: bool = False
    ) -> List[int]:

        if isinstance(text, str):
            token_ids = [self._convert_token_to_id(token) for token in text]
        else:
            token_ids = [self._convert_token_to_id(token) for token in text]

        if truncation and max_length is not None:
            token_ids = self.truncate_sequences(token_ids, max_length, add_special_tokens)

        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        if pad_to_max_length and max_length is not None:
            token_ids = self.pad(token_ids, max_length)

        return token_ids

    def encode_batch(
            self,
            batch: Union[List[str], List[List[str]]],
            add_special_tokens: bool = True,
            truncation: bool = False,
            max_length: int = None,
            pad_to_max_length: bool = False
    ) -> List[List[int]]:

        return [self.encode(text, add_special_tokens=add_special_tokens,
                            truncation=truncation,
                            max_length=max_length,
                            pad_to_max_length=pad_to_max_length) for text in batch]

    def decode(self, token_ids: Union[int, List[int]]) -> str:
        if isinstance(token_ids, int):
            return self._convert_id_to_token(token_ids)
        return ''.join(
            self._convert_id_to_token(token_id) for token_id in token_ids
        )

    def batch_decode(self, sequences: Union[List[int], List[List[int]]]) -> List[str]:
        return [self.decode(seq) for seq in sequences]
