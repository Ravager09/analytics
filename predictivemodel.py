import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('dataset/sample_customer_data_for_exam.csv')

# Step 1: Prepare the data for modeling

# Handle missing values (if any)
df = df.dropna()


# Encode categorical variables
def one_hot_encode(df, column):
    encoded = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, encoded], axis=1)
    df = df.drop(column, axis=1)
    return df


categorical_columns = ['loyalty_status', 'gender']  # Add other categorical columns as needed
for col in categorical_columns:
    df = one_hot_encode(df, col)

# Select features and target
features = ['age', 'satisfaction_score', 'purchase_frequency']  # Add other relevant features
features += [col for col in df.columns if col.startswith(tuple(categorical_columns))]
X = df[features]
y = df['purchase_amount']


# Step 2: Split the data into training and testing sets
def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]


X_train, X_test, y_train, y_test = train_test_split(X, y)


# Step 3: Implement a custom linear regression model
class CustomLinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        X = np.column_stack((np.ones(X.shape[0]), X))
        self.coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]

    def predict(self, X):
        return X.dot(self.coefficients) + self.intercept


# Train the model
model = CustomLinearRegression()
model.fit(X_train, y_train)


# Step 4: Evaluate the model's performance
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r_squared(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r_squared(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Step 5: Identify the top three features
feature_importance = pd.Series(abs(model.coefficients), index=X.columns)
top_features = feature_importance.nlargest(3)

print("\nTop 3 features contributing to purchase amount prediction:")
for feature, importance in top_features.items():
    print(f"{feature}: {importance:.4f}")