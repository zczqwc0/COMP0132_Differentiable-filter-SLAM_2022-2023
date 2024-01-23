import tensorflow as tf
import pandas as pd

# Load the saved model
loaded_model = tf.keras.models.load_model('trained_model.keras')

# Display the summary of the loaded model to check the input shape
loaded_model.summary()

# New data has columns: 'u_k_x', 'u_k_y', 'u_k_theta'
new_data = pd.read_csv('dataset.csv')

# Take only the first row for prediction
first_row = new_data[['u_k_x', 'u_k_y', 'u_k_theta']].iloc[0].values.reshape(1, -1)

# Make predictions using the loaded model
predictions = loaded_model.predict(first_row)

# Print the predictions
print("Predicted relative transformation and covariance:")
print(predictions)