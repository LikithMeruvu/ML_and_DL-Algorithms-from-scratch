import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

class OLSRegression:
    def __init__(self, fit_intercept: bool = True):
        self._is_fitted = False
        self._coefficients = None
        self._intercept = None
        self._fit_intercept = fit_intercept
        self._x_columns = None
        self._y_column = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'OLSRegression':
        """
        Fit the OLS regression model.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input features.
            y (Union[pd.Series, np.ndarray]): Target variable.

        Returns:
            OLSRegression: The fitted model instance.

        Raises:
            ValueError: If input dimensions are incompatible.
        """
        X, y = self._validate_input(X, y)
        
        if self._fit_intercept:
            X = np.column_stack((np.ones(X.shape[0]), X))
        
        try:
            coeffs = np.linalg.inv(X.T @ X) @ X.T @ y
        except np.linalg.LinAlgError:
            raise ValueError("Unable to calculate inverse. Matrix may be singular.")
        
        if self._fit_intercept:
            self._intercept = coeffs[0]
            self._coefficients = coeffs[1:]
        else:
            self._intercept = 0
            self._coefficients = coeffs
        
        self._is_fitted = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions using the fitted model.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input features for prediction.

        Returns:
            np.ndarray: Predicted values.

        Raises:
            ValueError: If the model hasn't been fitted yet.
        """
        self._check_is_fitted()
        X = self._validate_predict_input(X)
        return self._intercept + X @ self._coefficients

    def score(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate the coefficient of determination R^2 of the prediction.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Test samples.
            y (Union[pd.Series, np.ndarray]): True values for X.

        Returns:
            float: R^2 of the prediction.
        """
        self._check_is_fitted()
        X, y = self._validate_input(X, y)
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

    def calculate_loss(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate the Mean Squared Error (MSE) loss.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input features.
            y (Union[pd.Series, np.ndarray]): True target values.

        Returns:
            float: The calculated MSE loss.
        """
        self._check_is_fitted()
        X, y = self._validate_input(X, y)
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred)

    def plot_regression_line(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray],
                             feature_index: int = 0, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot the regression line along with the data points for a single feature.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input features.
            y (Union[pd.Series, np.ndarray]): True target values.
            feature_index (int): Index of the feature to plot (default is 0).
            ax (Optional[plt.Axes]): Matplotlib axes object to plot on.

        Returns:
            plt.Axes: The matplotlib axes object with the plot.

        Raises:
            ValueError: If the model hasn't been fitted yet or if feature_index is out of bounds.
        """
        self._check_is_fitted()
        X, y = self._validate_input(X, y)

        if feature_index < 0 or feature_index >= X.shape[1]:
            raise ValueError(f"feature_index must be between 0 and {X.shape[1] - 1}")

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(X[:, feature_index], y, color='blue', alpha=0.7, label='Data points')
        
        x_range = np.linspace(X[:, feature_index].min(), X[:, feature_index].max(), 100)
        X_plot = np.zeros((100, X.shape[1]))
        X_plot[:, feature_index] = x_range
        y_pred = self.predict(X_plot)
        
        ax.plot(x_range, y_pred, color='red', label='Regression line')
        ax.set_xlabel(f'Feature {feature_index}')
        ax.set_ylabel('Target')
        ax.set_title('OLS Regression')
        ax.legend()
        ax.grid(True)

        return ax

    def summary(self) -> str:
        """
        Generate a summary of the regression results.

        Returns:
            str: A string containing the summary of regression results.

        Raises:
            ValueError: If the model hasn't been fitted yet.
        """
        self._check_is_fitted()
        
        summary = "OLS Regression Results\n"
        summary += "=======================\n"
        summary += f"Intercept: {self._intercept:.4f}\n"
        for i, coef in enumerate(self._coefficients):
            summary += f"Coefficient {i}: {coef:.4f}\n"
        
        return summary

    def _validate_input(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Validate and preprocess input data."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.ravel()
        
        if X.ndim != 2:
            raise ValueError("X should be a 2D array or DataFrame")
        if y.ndim != 1:
            raise ValueError("y should be a 1D array or Series")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y should have the same number of samples")
        
        return X, y

    def _validate_predict_input(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Validate and preprocess input data for prediction."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        if X.ndim != 2:
            raise ValueError("X should be a 2D array or DataFrame")
        if X.shape[1] != len(self._coefficients):
            raise ValueError(f"X should have {len(self._coefficients)} feature(s)")
        
        return X

    def _check_is_fitted(self) -> None:
        """Check if the model has been fitted."""
        if not self._is_fitted:
            raise ValueError("This OLSRegression instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")

    @property
    def coefficients(self) -> np.ndarray:
        """Get the coefficients of the fitted model."""
        self._check_is_fitted()
        return self._coefficients

    @property
    def intercept(self) -> float:
        """Get the intercept of the fitted model."""
        self._check_is_fitted()
        return self._intercept

    def __repr__(self) -> str:
        return f"OLSRegression(fit_intercept={self._fit_intercept}, fitted={self._is_fitted})"

def train_test_model(df: pd.DataFrame, x_columns: List[str], y_column: str, test_size: float = 0.2, random_state: Optional[int] = None) -> Tuple[OLSRegression, dict]:
    """
    Train and test an OLS regression model.

    Args:
        df (pd.DataFrame): Input DataFrame.
        x_columns (List[str]): Names of the input feature columns.
        y_column (str): Name of the target variable column.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (Optional[int]): Controls the shuffling applied to the data before applying the split.

    Returns:
        Tuple[OLSRegression, dict]: The fitted model instance and a dictionary of evaluation metrics.
    """
    X = df[x_columns]
    y = df[y_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = OLSRegression()
    model.fit(X_train, y_train)

    train_mse = model.calculate_loss(X_train, y_train)
    test_mse = model.calculate_loss(X_test, y_test)
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)

    metrics = {
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_r2": train_r2,
        "test_r2": test_r2
    }

    return model, metrics

# Example usage
if __name__ == "__main__":
    # Create a sample dataset
    np.random.seed(42)
    X = np.random.rand(100, 2) * 10
    y = 2 + 3 * X[:, 0] + 1.5 * X[:, 1] + np.random.randn(100) * 0.5
    df = pd.DataFrame(np.column_stack((X, y)), columns=['X1', 'X2', 'y'])

    # Train and test the model
    model, metrics = train_test_model(df, ['X1', 'X2'], 'y')

    # Print results
    print(model.summary())
    print("\nModel Evaluation:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Plot regression line for the first feature
    plt.figure(figsize=(10, 6))
    ax = model.plot_regression_line(df[['X1', 'X2']], df['y'], feature_index=0)
    plt.show()