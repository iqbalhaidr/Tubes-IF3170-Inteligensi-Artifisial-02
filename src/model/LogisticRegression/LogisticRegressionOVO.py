import numpy as np
import pickle
from model.LogisticRegression.StochasticGradientDescent import StochasticGradientDescent

class LogisticRegressionOVO:
    def __init__(self, lr = 0.1, epochs = 10):
        self.lr = lr
        self.epochs = epochs
        self.models = None
        self.n_classes = None
        self.label_to_idx = None
        self.idx_to_label = None
    
    def fit(self, X, y):
        X = X.values
        y = y.values

        unique_classes = np.unique(y)
        
        self.n_classes = len(unique_classes)
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_classes)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        y_numeric = np.array([self.label_to_idx[label] for label in y])

        if self.n_classes > 2 :
            self.models = {}
            for i in range (self.n_classes):
                for j in range (i + 1, self.n_classes):
                    mask = (y_numeric == i) | (y_numeric == j)
                    X_pair = X[mask]
                    y_pair = y_numeric[mask]
                    
                    y_binary = (y_pair == i).astype(int)

                    model = StochasticGradientDescent(lr=self.lr, epochs=self.epochs)
                    model.fit(X_pair, y_binary)

                    self.models[(i,j)] = model        
        else :
            self.models = StochasticGradientDescent(lr=self.lr, epochs=self.epochs)
            self.models.fit(X, y_numeric)

    def predict(self, X):
        X = X.values
        if isinstance(self.models, dict):
            n_samples = X.shape[0]
            votes = np.zeros((n_samples, self.n_classes))

            for (i, j), model in self.models.items():
                predictions = model.predict_proba(X)

                for idx in range(n_samples):
                    if predictions[idx] == 1:
                        votes[idx,i] += 1
                    else : 
                        votes[idx, j] += 1
            
            best_indices = np.argmax(votes, axis=1)   
            y_pred = np.array([self.idx_to_label[idx] for idx in best_indices])

            return y_pred

        else :
            y_pred_numeric = self.models.predict(X)
            y_pred = np.array([self.idx_to_label[val] for val in y_pred_numeric])

            return y_pred

    def save_model(self, filepath):
        model_data = {
            'models': self.models,
            'n_classes': self.n_classes,
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'lr': self.lr,
            'epochs': self.epochs
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model berhasil disimpan ke {filepath}")
    
    def load_model(self, filepath):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models = model_data['models']
        self.n_classes = model_data['n_classes']
        self.label_to_idx = model_data['label_to_idx']
        self.idx_to_label = model_data['idx_to_label']
        self.lr = model_data['lr']
        self.epochs = model_data['epochs']
        
        print(f"Model berhasil di-load dari {filepath}")
        print(f"Jumlah kelas: {self.n_classes}")
        print(f"Jumlah model: {len(self.models)}")