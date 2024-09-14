import numpy as np
import matplotlib.pyplot as plt

# Data from the user's request
data = {
    1: 0.0314,
    2: 0.0283,
    3: 0.0577,
    4: 0.3804,
    5: 1.4777,
    6: 2.4341,
    7: 7.1150,
    8: 27.3565,
    9: 103.0152,
    10: 1995.4661,
    11: 6384.6914
}

# Convert the values to a numpy array for normalization
values = np.array(list(data.values()))

# Apply logarithmic transformation to compress the range
log_values = np.log10(values + 1e-8)  # Add small epsilon to avoid log(0)

# Normalize the log-transformed values
log_normalized_values = (log_values - log_values.min()) / (log_values.max() - log_values.min())

# Plotting the log-normalized values
plt.figure(figsize=(10, 6))
plt.plot(list(data.keys()), log_normalized_values, marker='o', linestyle='-', color='g')

# Adding titles and labels
plt.title('Log-Normalized Tensor Values Across Different Indices')
plt.xlabel('Index')
plt.ylabel('Log-Normalized Tensor Value')

# Display the plot
plt.grid(True)
plt.show()
