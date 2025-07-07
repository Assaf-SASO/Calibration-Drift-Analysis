import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

# --- Step 0: Data Preparation ---
# Create a DataFrame with the calibration data from the image.
data = {
    'Date': [
        '10/30/2024', '10/19/2023', '9/18/2022',
        '9/1/2021', '8/1/2020', '6/1/2019'
    ],
    'Result': [-1.24, -1.83, -0.52, -0.49, -0.11, -0.33]
}
df = pd.DataFrame(data)

# Convert 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

# Sort the data by date in ascending order for correct interval calculations
df = df.sort_values(by='Date').reset_index(drop=True)

# --- Step 1: Analytical Method — Compute Baseline UD ---
print("--- Step 1: Analytical Method ---")

# Calculate the differences in results (y) and time in days (t)
df['Result_Diff'] = df['Result'].diff()
df['Time_Diff_Days'] = df['Date'].diff().dt.days

# Calculate the daily drift rates for each interval
# The first value will be NaN, so we drop it
daily_drift_rates = (df['Result_Diff'] / df['Time_Diff_Days']).dropna()

# Define the target calibration interval in days
I = 365.25

# Calculate the uncertainty due to drift (UD) using the provided formula
# UD = I * mean(abs(daily_drift_rates))
ud_analytical = I * np.mean(np.abs(daily_drift_rates))

print(f"Calculated Daily Drift Rates:\n{daily_drift_rates.values}")
print(f"\nBaseline Uncertainty due to Drift (UD): {ud_analytical:.4f}\n")


# --- Step 2: Bootstrap of Daily Drift Rates ---
print("--- Step 2: Bootstrap of Daily Drift Rates ---")
n_bootstraps = 100000
bootstrapped_ud = []

for _ in range(n_bootstraps):
    # Resample the daily drift rates with replacement
    resampled_rates = np.random.choice(
        daily_drift_rates.values,
        size=len(daily_drift_rates),
        replace=True
    )
    # Calculate the simulated UD for this bootstrap sample
    simulated_ud = I * np.mean(np.abs(resampled_rates))
    bootstrapped_ud.append(simulated_ud)

# Calculate statistics from the bootstrap distribution
mean_ud_bootstrap = np.mean(bootstrapped_ud)
std_ud_bootstrap = np.std(bootstrapped_ud)
ci_95_ud = np.percentile(bootstrapped_ud, [2.5, 97.5])

print(f"Bootstrap Mean UD: {mean_ud_bootstrap:.4f}")
print(f"Bootstrap Std Dev of UD: {std_ud_bootstrap:.4f}")
print(f"95% Confidence Interval for UD: [{ci_95_ud[0]:.4f}, {ci_95_ud[1]:.4f}]\n")

# Plotting the histogram of bootstrapped UD values
plt.figure(figsize=(10, 6))
sns.histplot(bootstrapped_ud, bins=50, kde=True, color='skyblue')
plt.axvline(ud_analytical, color='red', linestyle='--', label=f'Analytical UD = {ud_analytical:.4f}')
plt.axvline(ci_95_ud[0], color='green', linestyle=':', label='95% CI Lower Bound')
plt.axvline(ci_95_ud[1], color='green', linestyle=':', label='95% CI Upper Bound')
plt.title('Histogram of Bootstrapped Drift Uncertainty (UD)')
plt.xlabel('Uncertainty due to Drift (UD)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()


# --- Step 3: Bootstrap Regression (Optional) ---
print("--- Step 3: Bootstrap Regression Analysis ---")

# Prepare data for regression: t = days from the first calibration, y = result
df['Days_from_Start'] = (df['Date'] - df['Date'].min()).dt.days
t_data = df['Days_from_Start'].values
y_data = df['Result'].values

bootstrapped_slopes = []
bootstrapped_intercepts = []
n_points = len(df)

for _ in range(n_bootstraps):
    # Keep resampling until we get at least 2 unique x-values
    while True:
        # Resample (t, y) pairs with replacement
        indices = np.random.choice(range(n_points), size=n_points, replace=True)
        t_sample = t_data[indices]
        y_sample = y_data[indices]
        
        # Check if we have at least 2 unique x-values
        if len(np.unique(t_sample)) > 1:
            break
    
    # Fit linear regression to the bootstrap sample
    slope, intercept, _, _, _ = linregress(t_sample, y_sample)
    bootstrapped_slopes.append(slope)
    bootstrapped_intercepts.append(intercept)

bootstrapped_slopes = np.array(bootstrapped_slopes)
bootstrapped_intercepts = np.array(bootstrapped_intercepts)

# Calculate annual drift from slopes
annual_drift_regression = bootstrapped_slopes * 365.25

# Calculate statistics for the regression results
mean_slope = np.mean(bootstrapped_slopes)
ci_95_slope = np.percentile(bootstrapped_slopes, [2.5, 97.5])
mean_annual_drift = np.mean(annual_drift_regression)
ci_95_annual_drift = np.percentile(annual_drift_regression, [2.5, 97.5])

print(f"Mean Slope (Drift per Day): {mean_slope:.6f}")
print(f"95% CI for Slope: [{ci_95_slope[0]:.6f}, {ci_95_slope[1]:.6f}]")
print(f"\nMean Annual Drift (from Regression): {mean_annual_drift:.4f}")
print(f"95% CI for Annual Drift: [{ci_95_annual_drift[0]:.4f}, {ci_95_annual_drift[1]:.4f}]")

# --- Direct Regression on Original Data ---
print("\n--- Direct Regression on Original Data ---")
direct_slope, direct_intercept, r_value, p_value, std_err = linregress(t_data, y_data)
direct_annual_drift = direct_slope * 365.25
print(f"Annual Drift from Direct Regression: {direct_annual_drift:.4f}")
print(f"R-squared: {r_value**2:.4f}")

# Plotting the regression results
plt.figure(figsize=(12, 7))

# Plot the original data points
plt.scatter(t_data, y_data, color='red', zorder=5, label='Original Calibration Data', s=100)

# Generate points for the regression lines
t_fit = np.linspace(t_data.min(), t_data.max(), 100)

# Calculate the confidence interval for the regression line
y_fit_bootstraps = bootstrapped_intercepts[:, np.newaxis] + bootstrapped_slopes[:, np.newaxis] * t_fit
ci_lower, ci_upper = np.percentile(y_fit_bootstraps, [2.5, 97.5], axis=0)

# Plot the confidence interval band
plt.fill_between(t_fit, ci_lower, ci_upper, color='gray', alpha=0.4, label='95% Confidence Band')

# Plot the mean regression line
mean_intercept = np.mean(bootstrapped_intercepts)
plt.plot(t_fit, mean_intercept + mean_slope * t_fit, 'b-', linewidth=2, label='Mean Bootstrap Regression')

# Plot the direct regression line
plt.plot(t_fit, direct_intercept + direct_slope * t_fit, 'k--', linewidth=2, label=f'Direct Regression (Drift: {direct_annual_drift:.4f})')

# Add data points with date labels
for i, txt in enumerate(df['Date'].dt.strftime('%Y-%m-%d')):
    plt.annotate(txt, (t_data[i], y_data[i]), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('Bootstrap Regression of Calibration Data')
plt.xlabel('Time (Days from First Calibration)')
plt.ylabel('Calibration Result')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot as a PNG file
output_file = 'calibration_regression_plot.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved as: {output_file}")

# Show the plot interactively
plt.show(block=True)

# --- Final Comparison ---
print("\n--- Summary & Comparison ---")
print(f"Analytical UD (Method 2): {ud_analytical:.4f}")
print(f"Mean Annual Drift from Bootstrap Regression: {mean_annual_drift:.4f}")
print(f"Annual Drift from Direct Regression: {direct_annual_drift:.4f}")

# Ensure the plot is displayed
plt.tight_layout()
plt.show()
print("\nNote: The analytical method averages the absolute rate of change between points, while the regression method finds the single best-fit trend line. A negative drift from regression indicates a general downward trend over time.")