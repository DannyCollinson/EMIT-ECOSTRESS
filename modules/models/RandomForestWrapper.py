from sklearn.ensemble import RandomForestRegressor
import pickle

class RandomForestWrapper:
    def __init__(self, input_dim, output_dim):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def fit(self, train_data, val_data=None):
        self.model.fit(train_data, val_data)

    def predict(self, input_data):
        return self.model.predict(input_data)

    def save(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(self.model, file)

