#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#
from typing import Dict, Sequence, Tuple

import numpy as np


class SparseDataset:
    """A dataset of variable-length sequences"""

    field_names: Sequence[str]
    sparse_dataset: Sequence[np.ndarray]
    sparse_index: np.ndarray
    max_lengths: np.ndarray

    def __init__(
        self,
        sparse_index: Sequence[np.ndarray],
        sparse_dataset: Sequence[np.ndarray],
        field_names: Sequence[str],
        max_lengths: Sequence[np.ndarray],
    ) -> None:
        assert len(field_names) == len(max_lengths)
        assert len(field_names) == len(sparse_dataset)
        self.sparse_index = sparse_index
        self.sparse_dataset = sparse_dataset
        self.field_names = field_names
        self.max_lengths = max_lengths

    @classmethod
    def from_file(cls, filename: str) -> None:
        """Load a sparse dataset from a file"""
        with open(filename, "rb") as fin:
            keys = np.load(fin)
            sparse_index = np.load(fin)
            sparse_dataset = [np.load(fin) for _ in keys]
            max_lengths = np.load(fin)
        return cls(sparse_index, sparse_dataset, keys, max_lengths)

    def to_file(self, filename: str) -> None:
        with open(filename, "wb") as fout:
            np.save(fout, self.field_names)
            np.save(fout, self.sparse_index)
            for d in self.sparse_dataset:
                np.save(fout, d)
            np.save(fout, self.max_lengths)

    @property
    def num_examples(self) -> int:
        return self.sparse_index.shape[0]

    def _get_batch(
        self,
        item_indexes: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Copy sparse examples into a dense batch."""
        index = self.sparse_index[item_indexes]  # bsz, num_fields, 2
        output = []
        for t in enumerate(zip(self.sparse_dataset, self.max_lengths)):
            i, (f, max_l) = t
            pointers, lengths = index[:, i, 0], index[:, i, 1]
            field = np.lib.stride_tricks.sliding_window_view(f, max_l)
            field = field[pointers]
            len_mask = np.arange(max_l)[np.newaxis] < lengths[:, np.newaxis]
            output.append(field * len_mask)
        return dict(zip(self.field_names, output))

    def subset(self, indexes):
        return SparseDataset(
            self.sparse_index[indexes],
            self.sparse_dataset,
            self.field_names,
            self.max_lengths,
        )

    def train_batches(
        self,
        batch_size: int,
        seed: int = 0,
        shuffle_once: bool = False,
    ):
        """Return a function mapping a train step to a random batch."""
        num_batches = self.num_examples // batch_size
        state = []

        def set_perm(epoch):
            """Next epoch example ordering."""
            rng = np.random.RandomState(seed=seed + epoch)
            p = rng.permutation(self.num_examples)
            p = p[: num_batches * batch_size]  # drop incomplete last batch
            perm = np.reshape(p, (num_batches, batch_size))
            state.clear()
            state.extend((epoch, perm))

        set_perm(0)

        def get_batch_fn(batch_index):
            """Get batch for step <batch_index>."""
            prev_epoch, perm = state
            epoch = batch_index // num_batches
            batch_index = batch_index % num_batches
            if (epoch != prev_epoch) and not shuffle_once:
                set_perm(epoch)
                _, perm = state
            index = perm[batch_index]
            return self._get_batch(index)

        return get_batch_fn

    def eval_batches(self, batch_size: int, drop_remainder: bool = False):
        """Return a generator over non suffled batches."""
        num_examples = self.num_examples
        indexes = np.arange(num_examples)

        def fn():
            start_index = 0
            while start_index < num_examples:
                next_index = start_index + batch_size
                idx = indexes[start_index:next_index]
                if drop_remainder and idx.size < batch_size:
                    break
                start_index = next_index
                yield self._get_batch(idx)

        return fn


class NumpyBuffer:
    """Growing buffer of numpy data."""

    def __init__(
        self,
        shape: Tuple[int],
        dtype: np.dtype,
        default_capacity: int = 1024**3,
        grow_rate: float = 1.2,
    ):
        assert default_capacity > 1
        assert grow_rate > 1
        self._grow_rate = grow_rate
        self._shape = shape
        self._buffer = np.zeros(shape=(default_capacity,) + shape, dtype=dtype)
        self._size = 0
        self._capacity = default_capacity

    def _set_capacity(self, new_capacity: int):
        buffer = self._buffer
        self._capacity = new_capacity
        self._buffer = np.empty(
            shape=(self._capacity,) + self._shape,
            dtype=buffer.dtype,
        )
        self._buffer[: self._size] = buffer[: self._size]
        del buffer

    def _grow(self):
        self._set_capacity(int(np.ceil(self._capacity * self._grow_rate)))

    def trim(self):
        self._set_capacity(self._size)

    def append(self, x):
        x = np.array(x, dtype=self._buffer.dtype)
        new_size = x.shape[0] + self._size
        if x.shape[0] + self._size > self._capacity:
            self._grow()
        self._buffer[self._size : new_size] = x
        self._size = new_size

    @property
    def data(self):
        return self._buffer[: self._size]


class SparseDatasetBuilder:
    """Build a sparse dataset by iteratively adding examples."""

    def __init__(
        self,
        field_names: Sequence[str],
        dtypes: Sequence[np.dtype],
        max_lengths: Sequence[np.ndarray],
        shapes: Sequence[Tuple[int]],
    ) -> None:
        assert len(field_names) == len(max_lengths)
        assert len(field_names) == len(dtypes)
        assert len(field_names) == len(shapes)

        self._field_names = field_names
        self._max_lengths = max_lengths
        self._shapes = shapes
        self._dtypes = dtypes

        self._buffers = [NumpyBuffer(s, dt) for s, dt in zip(shapes, dtypes)]
        self._lengths = [[] for _ in shapes]

    def append(self, example) -> None:
        assert len(self._field_names) == len(example)
        for x, b, l, max_l in zip(
            example, self._buffers, self._lengths, self._max_lengths
        ):
            x = x[:max_l]
            b.append(x)
            l.append(len(x))

    def to_sparse_dataset(self) -> SparseDataset:
        # sparse_index is num_examples x fields x 2 with (pointer, length) pairs
        lengths = np.array(self._lengths, np.uint64).transpose()
        pointers = np.cumsum(lengths, axis=0)
        pointers = np.pad(pointers, [[1, 0], [0, 0]])[:-1]
        sparse_index = np.stack((pointers, lengths), axis=-1)

        # padding ensures max_length is available after the last example.
        for b, s, l, dt in zip(
            self._buffers, self._shapes, self._max_lengths, self._dtypes
        ):
            b.append(np.zeros(shape=(l,) + s, dtype=dt))
            b.trim()

        # the buffer content
        sparse_dataset = [b.data for b in self._buffers]

        # clear buffers.
        self._buffers = [
            NumpyBuffer(s, dt) for s, dt in zip(self._shapes, self._dtypes)
        ]
        self._lengths = [[] for _ in self._buffers]

        return SparseDataset(
            sparse_index=sparse_index,
            sparse_dataset=sparse_dataset,
            field_names=self._field_names,
            max_lengths=self._max_lengths,
        )


def random_split(
    dataset: SparseDataset,
    split_sizes: Sequence[int],
    seed: int = 0,
) -> Sequence[SparseDataset]:
    data_size = dataset.num_examples

    # random shuffle
    rng = np.random.RandomState(seed=seed)
    shuf = rng.permutation(data_size)

    # sizes to splits:
    # We start with the last split as it allows to have the same examples
    # in the last subset (valid) when we vary the first size of the first one
    # (train).
    output, start = [], 0
    for split_size in reversed(split_sizes):
        idx = shuf[start : start + split_size]
        output.append(dataset.subset(idx))
        start += split_size

    return list(reversed(output))
