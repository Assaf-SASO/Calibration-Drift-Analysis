# Calibration Drift Analysis

This project analyzes calibration drift using analytical and bootstrap methods. It calculates the uncertainty due to drift (UD) and performs a bootstrap regression to understand the drift trend over time.

## Prerequisites

- Python 3.6 or later

## Installation

1. **Clone the repository or download the source code.**

2. **Create a virtual environment.**

   Open a terminal or command prompt in the project directory and run:

   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment.**

   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```

   - On macOS and Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install the required packages.**

   With the virtual environment activated, install the dependencies from `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the analysis, execute the `code.py` script:

```bash
python code.py
```

The script will print the results of the analytical method, the bootstrap analysis, and the regression analysis to the console. It will also generate and save a plot named `calibration_regression_plot.png` in the project directory, and display the plots interactively.
