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
        self.size = min(self.size + 1, self.capacity)

    def to_index(self, idx):
        assert -self.size <= idx < self.size, IndexError(f"Index {idx} out of bounds [0, {self.size - 1}]")
        if idx < 0:   # support negative indices
            idx += self.size
        return (self.pos + idx) % self.capacity if self.is_full else idx

    @property
    def is_full(self):
        return self.size == self.capacity

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
        return f"{self.__class__.__name__}({list(self)}, capacity={self.capacity})"


class SumTree:

    def __init__(self, capacity):
        assert capacity > 0 and capacity.bit_count() == 1, f"Capacity must be power of 2, but got {capacity}" # simplifies the arithmetics
        self.capacity = capacity
        self.storage_size = 2 * capacity - 1
        self.first_leaf = self.storage_size - capacity
        self.nodes = [0] * self.storage_size

    def get(self, idx):
        return self.nodes[self.first_leaf + idx]

    def update(self, idx, value):  # O(log n)
        child = self.first_leaf + idx
        delta = value - self.nodes[child]
        self.nodes[child] = value
        while child > 0:
            parent = (child - 1) // 2
            self.nodes[parent] += delta
            child = parent

    def query(self, r):
        r = self._safe_range(r)
        i = 0  # root tree index
        while i < self.first_leaf:    # until is not a leaf
            left = 2 * i + 1
            if r <= self.nodes[left]:
                i = left              # go left
            else:
                i = left + 1          # go right
                r -= self.nodes[left] # and subtract the left cum sum

        logical_idx = i - self.first_leaf
        return logical_idx

    def get_data(self):
        leaves = self.nodes[self.first_leaf:]
        return leaves

    def _safe_range(self, r):
        if not (0 <= r <= self.total_sum):
            if abs(self.total_sum - r) > 1e-3: # correct minor floating-point precision errors near 0 and total_sum
                raise ValueError(f'Value {r} is outside the expected range: [0, {self.total_sum}]')
            r = max(0, min(r, self.total_sum))
        return r

    @property
    def total_sum(self):
        return self.nodes[0]

    def __getitem__(self, idx): # used to match api calls
        return self.get(idx)

    def __setitem__(self, idx, value): # used to match api calls
        self.update(idx, value)

    def __repr__(self):
        out  = f"{self.__class__.__name__}(capacity={self.capacity}, storage_size={self.storage_size}, total_sum={self.total_sum})\n"
        out += f"-> leaves={self.get_data()}"
        return out



if __name__ == '__main__':
    import random
    from lib.utils import measure
    from collections import deque
    from plots import print_binary_tree_array as print_tree

    print('Example sum-tree:')
    st = SumTree(capacity=32)
    for i in range(st.capacity - 5):
        st.update(i, i*10)
    print_tree(st.nodes)


    print('\nMeasure performance:')
    n = 2 ** 17 # > 100_000
    cb = CircularBuffer(capacity=n)   # indexed access is O(1)
    dq = deque(maxlen=n)              # indexed access is O(1) at both ends but slows to O(n) in the middle.
    st = SumTree(capacity=n)          # indexed access (to the leaves) is O(1)
    for i in range(n + n//2 -2):
        dq.append(i)
        cb.append(i)
        st.update(i % n, i)
    rnd_idx = [random.randrange(n) for _ in range(n)]
    measure('CircularBuffer random access', lambda : [cb[j] for j in rnd_idx])     # O(n)
    measure('SumTree random access       ', lambda : [st.get(j) for j in rnd_idx]) # O(n)
    measure('Deque random access         ', lambda : [dq[j] for j in rnd_idx])     # O(n²)
