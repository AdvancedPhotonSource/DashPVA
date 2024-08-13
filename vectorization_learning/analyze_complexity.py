import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#e data
roi = np.array([50, 100, 200, 400])
values = np.array([0.0149, 0.0401, 0.2058, 0.7358])

def linear_model(x, a, b):
    return a * x + b

def log_model(x, a, b):
    return a * np.log(x) + b

def n_log_n_model(x, a, b):
    return a * x * np.log(x) + b

def n_squared_model(x, a, b):
    return a * x**2 + b

# Fit 
params_linear, _ = curve_fit(linear_model, roi, values)
params_log, _ = curve_fit(log_model, roi, values)
params_n_log_n, _ = curve_fit(n_log_n_model, roi, values)
params_n_squared, _ = curve_fit(n_squared_model, roi, values)

# fitted values
roi_fine = np.linspace(50, 400, 100)
linear_fit = linear_model(roi_fine, *params_linear)
log_fit = log_model(roi_fine, *params_log)
n_log_n_fit = n_log_n_model(roi_fine, *params_n_log_n)
n_squared_fit = n_squared_model(roi_fine, *params_n_squared)

# Plot the data and fits
plt.figure(figsize=(10, 6))
plt.scatter(roi, values, color='red', label='Data')
plt.plot(roi_fine, linear_fit, label='Linear fit', color='blue')
plt.plot(roi_fine, log_fit, label='Log(n) fit', color='green')
plt.plot(roi_fine, n_log_n_fit, label='n log(n) fit', color='orange')
plt.plot(roi_fine, n_squared_fit, label='n^2 fit', color='purple')

plt.xlabel('ROI')
plt.ylabel('Values')
plt.legend()
plt.title('Fitting Models to Data')
plt.grid(True)
plt.show()
