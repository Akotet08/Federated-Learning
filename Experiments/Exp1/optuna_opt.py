import optuna
from model import load_dataset, CNNLSTM, init_weights, train

def objective(trial):
    # Hyperparameters to tune
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    # Load data
    X_train, y_train, X_test, y_test = load_dataset(drop_index=-1)

    # Initialize the model with the suggested hyperparameters
    model = CNNLSTM()
    model.apply(init_weights)

    # Train the model and retrieve performance metrics
    stats = train(X_train, y_train, X_test, y_test, model, epochs= 100, batch_size=int(batch_size), lr=lr)

    # Optuna aims to minimize the objective, so return the validation loss
    return stats['Val loss']

study = optuna.create_study(direction='minimize', )
study.optimize(objective, n_trials=50)  # Specify the number of trials

# Print the best hyperparameters
print('Best trial:', study.best_trial.params)
