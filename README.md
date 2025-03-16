# Machine Learning Classification Pipeline

This project implements a machine learning pipeline for training and evaluating classifiers (Random Forest and Multi-Layer Perceptron) on tabular data, with additional functionality for smoothing predictions and generating visualization plots. The script processes CSV files, trains models, evaluates their performance, and outputs predictions alongside visualizations for test samples.

## Purpose

The code is designed to:
1. Load and preprocess training data from multiple CSV files.
2. Train a Random Forest Classifier (or optionally a Multi-Layer Perceptron) on the data.
3. Evaluate the model using a classification report.
4. Process test samples, smooth predictions, and generate scatter plot visualizations comparing predicted and true labels against a depth variable.

The pipeline is configurable via a `config` dictionary, allowing customization of file paths, feature names, and hyperparameters.

## Dependencies

- Python 3.x
- Libraries:
  - `scikit-learn` (for machine learning models and metrics)
  - `pandas` (for data manipulation)
  - `numpy` (for numerical operations)
  - `matplotlib` (for plotting)
  - `os` (for file system operations)
  - `glob` (for file pattern matching)
  - `csv` (for CSV file handling)

Install the required libraries using pip:
```bash
pip install scikit-learn pandas numpy matplotlib
```

## Project Structure

- **Training Data**: CSV files in `./training/` directory.
- **Test Data**: CSV files in `./testing/` directory.
- **Output**: Predictions and plots saved to `./output/` directory.
- **Main Script**: The provided Python script (e.g., `pipeline.py`).

## Setup

1. **Prepare Data**:
   - Place training CSV files in the `./training/` directory.
   - Place test CSV files in the `./testing/` directory.
   - Ensure all CSV files have columns matching the `feature_names`, `label_name`, and `depth_name` specified in the `config` dictionary.

2. **Directory Structure**:
   ```
   project/
   ├── training/          # Directory with training CSV files
   ├── testing/           # Directory with test CSV files
   ├── output/            # Directory for output predictions and plots (created automatically)
   └── pipeline.py        # The main script
   ```

3. **Run the Script**:
   - Execute the script in a Python environment:
     ```bash
     python pipeline.py
     ```

## Configuration

The `config` dictionary at the bottom of the script defines the pipeline settings:
- `train_path`: Path to the consolidated training CSV file (`./train.csv`).
- `test_path`: Directory containing test CSV files (`./testing`).
- `output_path`: Directory for saving predictions and plots (`./output`).
- `feature_names`: List of feature columns in the CSV files (e.g., `['qc', 'fs', 'u2', 'Level']`).
- `label_name`: Target column for classification (e.g., `'Label'`).
- `depth_name`: Column representing depth or a similar variable (e.g., `'Level'`).
- `test_size`: Fraction of data for testing (e.g., `0.2`).
- `window_size`: Smoothing window size for predictions (e.g., `8`).
- `mapping`: Dictionary mapping label strings to integers (e.g., `'MD': 0, 'ALL-c (pal)': 1, ...`).

Modify the `config` dictionary as needed for your dataset.

## How It Works

1. **Data Loading**:
   - Combines all CSV files in `./training/` into a single deduplicated DataFrame, saved as `./train.csv`.
   - Splits the data into training and test sets using `train_test_split`.

2. **Model Training**:
   - Trains a Random Forest Classifier with parameters specified in `rf_param` (e.g., `n_estimators=100`, `max_depth=100`).
   - Optionally trains a Multi-Layer Perceptron (MLP) if uncommented, with parameters in `mlp_param`.

3. **Evaluation**:
   - Prints feature importance (for Random Forest) and a classification report comparing predictions to true labels.

4. **Test Sample Processing**:
   - Loads test CSV files from `./testing/`.
   - Predicts labels, smooths them using a custom `smooth` function (based on a sliding window mode filter), and saves predictions to CSV files in `./output/`.

5. **Visualization**:
   - Generates scatter plots comparing predicted and true labels against depth, saved as PNG files in `./output/`.
   - Uses a color map to represent different classes, with a legend based on the `mapping` dictionary.

## Functions

- `get_dataset(config)`: Loads and splits training data.
- `get_test_samples(config)`: Loads test samples from CSV files.
- `train_rf(data, param)`: Trains and evaluates a Random Forest Classifier.
- `train_mlp(data, param)`: Trains and evaluates an MLP Classifier (optional).
- `smooth(series, w)`: Smooths a series of predictions using a mode-based filter with a specified window size.
- `plot(clf, config, test_sample)`: Generates and saves a visualization for a test sample.

## Customization

- **Model Selection**: Uncomment the MLP training lines and adjust `mlp_param` to use an MLP instead of Random Forest.
- **Hyperparameters**: Modify `rf_param` or `mlp_param` to tune model performance.
- **Smoothing**: Adjust `window_size` in `config` to control the smoothing effect.
- **Features and Labels**: Update `feature_names`, `label_name`, and `mapping` in `config` to match your dataset.

## Output

- **CSV Files**: Predictions saved as `prediction_<filename>.csv` in `./output/` with columns for depth, predicted labels, and true labels.
- **PNG Files**: Plots saved as `<filename>.png` in `./output/`, showing predicted vs. true labels along the depth axis.

## Notes

- The script assumes all depth values in a test sample have the same sign (positive or negative).
- Missing values in the data are filled with `0` using `fillna(0)`.
- The smoothing function uses a mode-based approach, which may not work well with small window sizes or noisy data.

## Example Usage

Assuming your training and test CSV files are ready:
```bash
python pipeline.py
```
Check the `./output/` directory for results.

---
