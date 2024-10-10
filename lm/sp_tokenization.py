#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
import os
import shutil
import tempfile
from typing import Sequence, Tuple

import sentencepiece as spm
import tensorflow_text as tftxt

MODEL_TYPE = "unigram"
CHAR_COVERAGE = 1.0
EOS = 2


def load_tf_tokenizer(
    model_path: str,
    add_eos: bool = True,
):
    """Load a saved SentencePiece tokenizer."""
    with open(model_path, "rb") as model_fp:
        sp_model = model_fp.read()
    sp_tokenizer = tftxt.SentencepieceTokenizer(
        model=sp_model,
        add_eos=add_eos,
    )
    return sp_tokenizer


def load_tokenizer(
    model_path: str,
    add_eos: bool = True,
):
    """Load a saved SentencePiece tokenizer."""
    return spm.SentencePieceProcessor(model_file=model_path, add_eos=add_eos)


def _strings_to_textfile(
    strings: Sequence[str],
    max_num_char: int,
) -> Tuple[str, int]:
    """Write part of string iterator to lines in a text file."""
    num_char = 0
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as out_file:
        for s in strings:
            num_char += len(s)
            out_file.write(s + "\n")
            if num_char >= max_num_char:
                break
    return out_file.name, num_char


def train_tokenizer(
    strings: Sequence[str],
    vocab_size: int,
    model_path: str,
    max_num_char: int = int(1e7),
):
    """Train SentencePiece tokenizer from subset of tf dataset."""
    model_path = os.path.abspath(os.path.expanduser(model_path))
    tmp_txt, _ = _strings_to_textfile(strings, max_num_char)
    with tempfile.NamedTemporaryFile() as model_file:
        args = [
            f"--input={tmp_txt}",
            f"--vocab_size={vocab_size}",
            f"--character_coverage={CHAR_COVERAGE}",
            f"--model_prefix={model_file.name}",
            f"--model_type={MODEL_TYPE}",
        ]
        spm.SentencePieceTrainer.Train(" ".join(args))
        shutil.copy(model_file.name + ".model", model_path)
    os.remove(tmp_txt)
    return model_path
