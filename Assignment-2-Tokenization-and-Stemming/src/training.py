import time
from tensorflow.keras.callbacks import EarlyStopping

def train_sklearn_model(model, X_train, y_train, param_grid=None):
    start_time = time.time()
    if param_grid:
        from sklearn.model_selection import GridSearchCV
        grid = GridSearchCV(model, param_grid, cv=3)
        grid.fit(X_train, y_train)
        train_time = time.time() - start_time
        return grid.best_estimator_, train_time
    else:
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        return model, train_time

def train_keras_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
    start_time = time.time()
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )
    train_time = time.time() - start_time
    return model, history, train_time
