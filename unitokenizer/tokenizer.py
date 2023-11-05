from typing import Dict, List, Optional
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer

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


class UniTokenizer(PreTrainedTokenizer):

    def __init__(self,
                 bos_token=chr(SPECIAL_TOKEN_TO_ID["<s>"]),
                 eos_token=chr(SPECIAL_TOKEN_TO_ID["</s>"]),
                 sep_token=chr(SPECIAL_TOKEN_TO_ID["<sep>"]),
                 cls_token=chr(SPECIAL_TOKEN_TO_ID["<cls>"]),
                 pad_token=chr(SPECIAL_TOKEN_TO_ID["<pad>"]),
                 mask_token=chr(SPECIAL_TOKEN_TO_ID["<mask>"]),
                 unk_token=chr(SPECIAL_TOKEN_TO_ID["<unk>"]),
                 add_prefix_space=False,
                 model_max_length=512,
                 **kwargs, ):

        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        sep_token = AddedToken(sep_token, lstrip=False, rstrip=False) if isinstance(sep_token, str) else sep_token
        cls_token = AddedToken(cls_token, lstrip=False, rstrip=False) if isinstance(cls_token, str) else cls_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token
        mask_token = AddedToken(mask_token, lstrip=True, rstrip=False) if isinstance(mask_token, str) else mask_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token

        self.model_max_length = model_max_length

        self._unicode_vocab_size = UNICODE_CODE_POINTS

        self._special_token_to_id = {chr(k): k for k in ID_TO_SPECIAL_TOKEN.keys()}
        self._id_to_special_token = ID_TO_SPECIAL_TOKEN

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=add_prefix_space,
            model_max_length=model_max_length,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return self._unicode_vocab_size

    def get_vocab(self):
        vocab = {chr(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._special_token_to_id.get(token, ord(token))

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_special_token.get(index, chr(index))

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def build_inputs_with_special_tokens(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None
    ) -> List[int]:

        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        result = cls + token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_special_tokens_mask(
            self,
            token_ids_0: List[int],
            token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        return ()