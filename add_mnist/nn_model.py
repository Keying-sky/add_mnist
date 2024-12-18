import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import optuna
import matplotlib.pyplot as plt

class NNModel:
    def __init__(self, num=19, path=None):
        self.num = num
        self.path = path
        self.model = None
        self.history = None
        self.best_trial = None
        
    def load_data(self):
        """ Load and preprocess the combined_mnist dataset. """

        data = np.load(self.path.DATA_DIR/'combined_mnist.npz')
        
        x_train = data['x_train'].astype('float32')
        y_train = data['y_train']
        x_validation = data['x_validation'].astype('float32')
        y_validation = data['y_validation']
        x_test = data['x_test'].astype('float32')
        y_test = data['y_test']
        
        return x_train, y_train, x_validation, y_validation, x_test, y_test
    
    def create_model(self, trial):
        """ Create model with hyperparameters suggested by Optuna trial."""

        # Get hyperparameters from trial
        num_layers = trial.suggest_int('num_layers', 2, 3)
        units = trial.suggest_int('units', 128, 256)
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.3)
        learn_rate = trial.suggest_float('learning_rate', 1e-3, 1e-2, log=True)
        activation = trial.suggest_categorical('activation', ['relu', 'sigmoid'])
        
        # Create model
        model = Sequential()
        model.add(Flatten(input_shape=(28, 56)))
        
        # Add hidden layers
        for _ in range(num_layers):
            model.add(Dense(units))
            model.add(BatchNormalization())     # add BN before activation
            model.add(Activation(activation))
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        
        # Add output layer
        model.add(Dense(self.num, activation='softmax'))
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=learn_rate),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
            
        return model
    
    def objective(self, trial):
        """Objective function for optimisation."""

        x_train, y_train, x_validation, y_validation, _, _ = self.load_data()
        
        model = self.create_model(trial)
        
        # Training callbacks
        callbacks = [ EarlyStopping( monitor='val_accuracy', patience=5, restore_best_weights=True) ]
            
        # Train model
        history = model.fit(
            x_train, y_train,
            batch_size=32,
            epochs=30,
            validation_data=(x_validation, y_validation),
            callbacks=callbacks,
            verbose=1)
        
        return max(history.history['val_accuracy'])
    
    def hyperparam_tunning(self, n_trial=20):
        """ Run hyperparameter optimisation."""

        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trial, show_progress_bar=True)
        
        self.best_trial = study.best_trial
        self.plot_optimisation_history(study)
        
        print("Best trial:")
        print(f"Value: {study.best_trial.value:.4f}")
        print("Params:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")
    
    def plot_optimisation_history(self, study):
        """ Plot the optimisation history."""

        plt.figure(figsize=(10, 6), dpi=100)
        plt.plot([t.number for t in study.trials],
                [t.value for t in study.trials], 'bo-')
        plt.xlabel('Trials')
        plt.ylabel('Validation Accuracy')
        plt.title('Optimisation')
        plt.grid(True)
        plt.show()
    
    def train_final_model(self):
        """
        Train final model with best hyperparameters.
        
        Returns:
            Trained model and training history
        """

        model_path = self.path.MODEL_DIR / 'best_model.h5'

        x_train, y_train, x_validation, y_validation, x_test, y_test = self.load_data()
        
        # create model with best hyperparameters
        self.model = self.create_model(self.best_trial)
        
        # callbacks for final training
        callbacks = [EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
                    ModelCheckpoint(str(model_path), monitor='val_accuracy', save_best_only=True)]
        
        # train final model
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=32,
            epochs=30,
            validation_data=(x_validation, y_validation),
            callbacks=callbacks,
            verbose=1)
        
        # evaluate on test set
        test_loss, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        print(f'\nTest accuracy: {test_accuracy:.4f}')
        print(f'\nTest loss: {test_loss:.4f}')
        
        return self.model, self.history