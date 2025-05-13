class Loss():
    def __init__(self):
        self.model = None

    def forward(self, X, Y):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError