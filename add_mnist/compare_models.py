import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
import json

class CompareModels:
    def __init__(self, path=None):

        self.path = path.DATA_DIR
        self.rf_model = None
        self.svm_model = None
        self.results = None
        self.x_train = None
        self.y_train = None
        self.x_validation = None
        self.y_validation = None
        self.x_test = None
        self.y_test = None
        
    def load_data(self):
        """Load and preprocess the dataset."""

        # load the saved dataset
        data = np.load(self.path /'combined_mnist.npz')
        x_train = data['x_train']
        y_train = data['y_train']
        x_validation = data['x_validation']
        y_validation = data['y_validation']
        x_test = data['x_test']
        y_test = data['y_test']
        
        # flatten the images
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        x_validation_flat = x_validation.reshape(x_validation.shape[0], -1)
        x_test_flat = x_test.reshape(x_test.shape[0], -1)
        
        # scale the features
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(x_train_flat)
        self.x_validation = scaler.transform(x_validation_flat)
        self.x_test = scaler.transform(x_test_flat)
        
        self.y_train = y_train
        self.y_validation = y_validation
        self.y_test = y_test
        
    def train_rforest(self):
        """ Train and evaluate Random Forest model."""

        start_time = time.time()
        
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            n_jobs=-1,
            random_state=100)
        
        self.rf_model.fit(self.x_train, self.y_train)
        
        # make predictions
        train_pred = self.rf_model.predict(self.x_train)
        val_pred = self.rf_model.predict(self.x_validation)
        
        # calculate accuracy
        train_accuracy = accuracy_score(self.y_train, train_pred)
        val_accuracy = accuracy_score(self.y_validation, val_pred)
        
        training_time = time.time() - start_time
        
        return train_accuracy, val_accuracy, training_time
    
    def train_svm(self, subset_size=10000):
        """ Train and evaluate SVM model."""
        
        start_time = time.time()
        
        self.svm_model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=100)
        
        # use a subset of data
        indices = np.random.choice(self.x_train.shape[0], subset_size, replace=False)
        
        self.svm_model.fit(self.x_train[indices], self.y_train[indices])
        
        # make predictions
        train_pred = self.svm_model.predict(self.x_train[indices])
        val_pred = self.svm_model.predict(self.x_validation)
        
        # calculate accuracy
        train_accuracy = accuracy_score(self.y_train[indices], train_pred)
        val_accuracy = accuracy_score(self.y_validation, val_pred)
        
        training_time = time.time() - start_time
        
        return train_accuracy, val_accuracy, training_time
    
    def compare_models(self):
        """ Compare different models and save results."""
        
        self.results = {}
        
        # train and evaluate Random Forest
        rf_train_acc, rf_val_acc, rf_time = self.train_rforest()
        rf_test_pred = self.rf_model.predict(self.x_test)
        rf_test_acc = accuracy_score(self.y_test, rf_test_pred)
        
        self.results['random_forest'] = {
            'training_accuracy': float(rf_train_acc),
            'validation_accuracy': float(rf_val_acc),
            'test_accuracy': float(rf_test_acc),
            'training_time': rf_time}
        
        
        # train and evaluate SVM
        svm_train_acc, svm_val_acc, svm_time = self.train_svm()
        svm_test_pred = self.svm_model.predict(self.x_test)
        svm_test_acc = accuracy_score(self.y_test, svm_test_pred)
        
        self.results['svm'] = {
            'training_accuracy': float(svm_train_acc),
            'validation_accuracy': float(svm_val_acc),
            'test_accuracy': float(svm_test_acc),
            'training_time': svm_time}
        
        
        # save results
        with open('comparison_result.json', 'w') as f:
            json.dump(self.results, f, indent=4)
    
    def plot_results(self):
        """Print and plot comparison of model performances."""
    
        print("\nResults:")
        for model, performs in self.results.items():
            print(f"\n{model}:")
            for name, perform in performs.items():
                print(f"{name}: {perform:.4f}")

        models = list(self.results.keys())
        accs = ['training_accuracy', 'validation_accuracy', 'test_accuracy']

        plt.figure(figsize=(12, 6))
        x = np.arange(len(accs))
        width = 0.35
        
        for i, model in enumerate(models):
            values = [self.results[model][acc] for acc in accs]
            plt.bar(x + i*width, values, width, label=model)
        
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width/2, accs)
        plt.legend()
        plt.show()
    