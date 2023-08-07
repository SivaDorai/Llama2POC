import numpy as np
import pandas as pd
from keras.src.callbacks import LearningRateScheduler
from keras.src.optimizers import Adam
#from keras.src.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
import mlflow
# Define the learning rate schedule function
def learning_rate_schedule(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * np.exp(-0.1)

mlflow.set_tracking_uri("/model_metrics")

df = pd.read_csv('diabetes.csv')
num_features = len(df.columns)-1
X = df.iloc[:,0:8]# iloc format is x,y - : at x indicates all rows and 0:8 indicates colums from 0 to 8
y = df['Outcome']

    # Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
# Standardize input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled)
# Build a feedforward neural network model
model = Sequential()
model.add(Dense(10, input_dim=num_features, activation='relu'))
# Add dropout after the first hidden layer
#model.add(Dropout(0.5))
model.add(Dense(5, activation='relu'))
# Add dropout after the second hidden layer
#model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# Define the initial learning rate
initial_lr = 0.001

# Initialize the optimizer with the initial learning rate
optimizer = Adam(learning_rate=initial_lr)

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Define the learning rate scheduler
lr_scheduler = LearningRateScheduler(learning_rate_schedule)

# Print the model summary
print(model.summary())

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2,callbacks=[lr_scheduler])

# Access validation loss and accuracy from history
val_loss = history.history['val_loss'][-1]
val_accuracy = history.history['val_accuracy'][-1]

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Start an MLflow run
with mlflow.start_run() as run:
    # Log parameters
    mlflow.log_param("hidden_units", [10, 5])
    mlflow.log_param("dropout_rate", 0.5)

    # Log metrics during training
    mlflow.log_metric("val_loss", val_loss)
    mlflow.log_metric("val_accuracy", val_accuracy)

    # Log the trained model artifact
    mlflow.keras.log_model(model, "/model_metrics")
