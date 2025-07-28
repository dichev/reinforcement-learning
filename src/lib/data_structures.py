from collections.abc import Sequence
import torch

class CircularBuffer(Sequence):
    def __init__(self, capacity):
        self._buffer = [None] * capacity
        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def append(self, value):
        self._buffer[self.pos] = value
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def to_index(self, idx):
        assert -self.size <= idx < self.size, IndexError(f"Index {idx} out of bounds [0, {self.size - 1}]")
        if idx < 0:   # support negative indices
            idx += self.size
        return (self.pos + idx) % self.capacity if self.is_full else idx

    @property
    def is_full(self):
        return self.size == self.capacity

    def get_data(self):
        if not self.is_full:
            return self._buffer[:self.size]
        return self._buffer[self.pos:] + self._buffer[:self.pos]

    def __getitem__(self, idx):
        idx = self.to_index(idx)
        return self._buffer[idx]

    def __setitem__(self, idx, value):
        idx = self.to_index(idx)
        self._buffer[idx] = value

    def __iter__(self):
        for i in range(self.size):
            yield self[i]

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"{self.__class__.__name__}({self.get_data()}, capacity={self.capacity})"


class CircularTensor(CircularBuffer):
    def __init__(self, capacity, dtype=torch.float, device=None):
        self._buffer = torch.empty(capacity, dtype=dtype, device=device)
        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def get_data(self):
        if not self.is_full:
            return self._buffer[:self.size]
        return torch.roll(self._buffer, shifts=-self.pos, dims=0)



if __name__ == '__main__':
    import random
    from lib.utils import measure
    from collections import deque

    n = 100_000
    cb = CircularBuffer(capacity=n)   # indexed access is O(1)
    ct = CircularTensor(capacity=n)   # indexed access is O(1) but it should be batched rather accessing each value in a loop
    dq = deque(maxlen=n)              # indexed access is O(1) at both ends but slows to O(n) in the middle.
    for i in range(n + n//2):
        dq.append(i)
        ct.append(i)
        cb.append(i)
    rnd_idx = [random.randrange(n) for _ in range(n)]
    measure('CircularBuffer random access', lambda : [cb[j] for j in rnd_idx])   # O(n)
    measure('CircularTensor random access', lambda : ct.get_data()[rnd_idx])     # O(~n)
    measure('Deque random access         ', lambda : [dq[j] for j in rnd_idx])   # O(n²)
