
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from REGRESSION.OLS_Regression import OLSRegression, train_test_model

def generate_sample_data(n_samples=1000, n_features=3, noise=0.5):
    np.random.seed(42)
    X = np.random.rand(n_samples, n_features) * 10
    y = 2 + 3 * X[:, 0] + 1.5 * X[:, 1] - 2 * X[:, 2] + np.random.randn(n_samples) * noise
    return pd.DataFrame(np.column_stack((X, y)), columns=[f'X{i+1}' for i in range(n_features)] + ['y'])

def test_ols_regression():
    print("Testing Enhanced OLS Regression Implementation")
    print("==============================================")

    # 1. Generate sample data
    df = generate_sample_data()
    print("Sample data generated.")

    # 2. Train and test the model
    x_columns = [f'X{i+1}' for i in range(3)]
    model, metrics = train_test_model(df, x_columns, 'y', test_size=0.2, random_state=42)
    print("\n1. Model trained and tested successfully.")

    # 3. Print model summary and metrics
    print("\n2. Model Summary:")
    print(model.summary())
    print("\n3. Model Metrics:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.4f}")

    # 4. Make predictions
    X_test = pd.DataFrame({'X1': [5, 7, 9], 'X2': [2, 4, 6], 'X3': [1, 3, 5]})
    predictions = model.predict(X_test)
    print("\n4. Predictions made:")
    print(pd.DataFrame({'X1': X_test['X1'], 'X2': X_test['X2'], 'X3': X_test['X3'], 'Predicted_y': predictions}))

    # 5. Plot regression lines
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, ax in enumerate(axes):
        model.plot_regression_line(df[x_columns], df['y'], feature_index=i, ax=ax)
        ax.set_title(f'Regression Line for X{i+1}')
    plt.tight_layout()
    plt.show()

    # 6. Test error handling
    print("\n5. Testing error handling:")
    try:
        untrained_model = OLSRegression()
        untrained_model.predict(X_test)
    except ValueError as e:
        print(f"   Caught expected error: {e}")

    try:
        model.predict(pd.DataFrame({'X1': [1, 2], 'X2': [3, 4]}))  # Missing X3
    except ValueError as e:
        print(f"   Caught expected error: {e}")

if __name__ == "__main__":
    test_ols_regression()