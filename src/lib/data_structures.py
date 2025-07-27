from collections.abc import Sequence

class CircularBuffer(Sequence):
    def __init__(self, capacity):
        self._buffer = [None] * capacity
        self.capacity = capacity
        self.size = 0
        self.pos = 0

    def append(self, value):
        self._buffer[self.pos] = value
        self.pos = (self.pos + 1) % self.capacity
        if not self.is_full:
            self.size += 1

    @property
    def is_full(self):
        return self.size == self.capacity

    def __getitem__(self, idx):
        assert 0 <= idx < self.size, IndexError(f"Index {idx} out of bounds [0, {self.size-1}]")
        i = (self.pos + idx) % self.capacity if self.is_full else idx
        return self._buffer[i]

    def __setitem__(self, idx, value):
        assert 0 <= idx < self.size, IndexError(f"Index {idx} out of bounds [0, {self.size-1}]")
        i = (self.pos + idx) % self.capacity if self.is_full else idx
        self._buffer[i] = value

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"CircularBuffer({[self[i] for i in range(len(self))]}, capacity={self.capacity})"


if __name__ == '__main__':
    import random
    from lib.utils import measure
    from collections import deque

    n = 100_000
    cb = CircularBuffer(capacity=n)   # indexed access is O(1)
    dq = deque(maxlen=n)              # indexed access is O(1) at both ends but slows to O(n) in the middle.
    for i in range(n + n//2):
        dq.append(i)
        cb.append(i)
    rnd_idx = [random.randrange(n) for _ in range(n)]
    measure('CircularBuffer random access', lambda : sum(cb[j] for j in rnd_idx))  # O(n)
    measure('Deque random access         ', lambda : sum(dq[j] for j in rnd_idx))  # O(n²)
