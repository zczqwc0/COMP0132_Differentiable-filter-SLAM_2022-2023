import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the test dataset
test_df = pd.read_csv('Dataset/testset.csv')

# Extract features and labels for the test set
X_test = test_df[['u_k_x', 'u_k_y', 'u_k_theta','x','y','theta']].values
y_test_true = test_df[['relative_X', 'relative_Y', 'relative_Theta', 
                       'Obs_dist', 'Obs_tetha', 
                       'Covariance_X', 'Covariance_Y', 'Covariance_Theta',
                       'Covariance_dis', 'Covariance_angle']].values

# Normalize the data
scaler_X = StandardScaler().fit(X_test)
X_scaled = scaler_X.transform(X_test)

# Load the trained model
model = tf.keras.models.load_model('trained_model.keras')

# Predictions on the test set
y_test_pred = model.predict(X_scaled)

# Calculate Mean Squared Error and Mean Absolute Error
mse = mean_squared_error(y_test_true, y_test_pred)
mae = mean_absolute_error(y_test_true, y_test_pred)

# Display Metrics
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)