import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def main():
    path_to_inputs = "C:\\Users\\Zach\\OneDrive - Clemson University\\IM_runs\\IM_results_9_7"
    path_to_targets = "C:\\Users\\Zach\\OneDrive - Clemson University\\IM_runs\\H_cons\\H_constants.csv"
    rt = RegressionTree(path_to_inputs, 
                            path_to_targets,
                            coef_index=0)
    rt.load_data()
    rt.train_model()
    rt.save_model()
    rt.load_model()
    rt.evaluate_unseen(idx_range=[len(rt.sequences)-5,len(rt.sequences)])

class RegressionTree:
    def __init__(self, input_path, target_path, **kwargs):
        self.input_path = input_path
        self.target_path = target_path

        self.coef_index = kwargs.get('coef_index', 2)
        self.test_train_split_factor = 0.8

        # Initialize regressor
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

    def load_data(self):
        dyn_results = sorted(
            [f for f in os.listdir(self.input_path) if f.endswith('.csv')],
            key=lambda x: x.split('_')[1]
        )
        zip_files = [f for f in os.listdir(self.target_path) if 'zip_' in f and '.csv' in f]

        # Load targets
        target_data = []
        for file in zip_files:
            df = pd.read_csv(os.path.join(self.target_path, file))
            cols = df.columns
            target_data.append(df[[cols[0], cols[1], cols[2]]].values)
        zip_array = np.vstack(target_data)
        self.zip_array = zip_array
        # targets = zip_array.T[self.coef_index]
        targets = zip_array

        # Load inputs
        input_array = []
        for file in dyn_results:
            df = pd.read_csv(os.path.join(self.input_path, file))
            cols = df.columns
            input_data = np.column_stack((df[cols[1]], df[cols[3]], df[cols[4]]))
            input_array.append(input_data)

        self.sequences = input_array
        self.targets = targets

        split_idx = round(len(input_array) * self.test_train_split_factor)
        self.train_seq = self.scale_sequences(input_array[:split_idx])
        self.test_seq = self.scale_sequences(input_array[split_idx:-5])
        self.train_targets = targets[:split_idx]
        self.test_targets = targets[split_idx:-5]

    def scale_sequences(self, sequences):
        scaled_flat = []
        for seq in sequences:
            scaled_seq = np.zeros_like(seq)
            for i in range(seq.shape[1]):
                col = seq[:, i].reshape(-1, 1)
                if np.all(col == col[0]):
                    scaled_seq[:, i] = 0
                else:
                    scaler = MinMaxScaler(feature_range=(-1, 1))
                    scaled_seq[:, i] = scaler.fit_transform(col).flatten()
            scaled_flat.append(scaled_seq.flatten())  # Flatten for sklearn
        return np.array(scaled_flat)

    def train_model(self):
        self.model.fit(self.train_seq, self.train_targets)
        predictions = self.model.predict(self.test_seq)
        mse = mean_squared_error(self.test_targets, predictions)
        r2 = r2_score(self.test_targets, predictions)
        print(f"Validation MSE: {mse:.4f}")
        print(f"Validation R^2: {r2:.4f}")

    def save_model(self, path="rf_model.pkl"):
        import joblib
        joblib.dump(self.model, path)

    def load_model(self, path="rf_model.pkl"):
        import joblib
        self.model = joblib.load(path)

    def predict_sequence(self, sequence):
        scaled = self.scale_sequences([sequence])[0]
        prediction = self.model.predict([scaled])[0]
        return prediction

    def predict_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        seq = df[['volt', 'freq', 'curr']].values
        return self.predict_sequence(seq)

    def evaluate_unseen(self, idx_range):
        sequences = self.scale_sequences(self.sequences[idx_range[0]:idx_range[1]])
        actual_zip = self.zip_array[idx_range[0]:idx_range[1]]
        targets = self.targets[idx_range[0]:idx_range[1]]
        predictions = self.model.predict(sequences)
        for i in range(len(sequences)):
            print(f"Predicted coefficients: {predictions[i]}")
            print(f"Actual coefficients: {actual_zip[i]}")

if __name__ == '__main__':
    main()
