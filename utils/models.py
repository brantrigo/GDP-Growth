
class GDPGrowthPredictor:

    def train(self, X, y, *args, **kwargs):
        self.X = X
        self.y = y
        
        pass

    def predict(self, X, *args, **kwargs):
        pass

    @staticmethod # Contains logic for the class, but it does not instantiate
    def load(filename):
        pass

    def save(self, filename):
        pass
