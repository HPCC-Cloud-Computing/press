class SlidingWindow:
    def __init__(self, data, size):
        self.data = data
        self.size = size
        self.index = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self.data.shape[0] - self.size + 1

    def __next__(self):
        if self.index < len(self.data) - self.size + 1:
            self.index += 1
            return self.data[(self.index - 1):(self.index - 1) + self.size]
        else:
            raise StopIteration
