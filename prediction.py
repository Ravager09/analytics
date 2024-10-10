import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv('dataset/sample_customer_data_for_exam.csv')


# Step 1: Prepare the data for classification
def one_hot_encode(df, column):
    encoded = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, encoded], axis=1)
    df = df.drop(column, axis=1)
    return df


# Handle missing values
df = df.dropna()

# Encode categorical variables
categorical_columns = ['loyalty_status', 'gender']  # Add other categorical columns as needed
for col in categorical_columns:
    df = one_hot_encode(df, col)

# Select features and target
features = ['age', 'satisfaction_score', 'purchase_frequency', 'purchase_amount']  # Add other relevant features
features += [col for col in df.columns if col.startswith(tuple(categorical_columns))]
X = df[features]
y = df['promotion_usage']  # Assuming this is the correct column name for promotion usage


# Step 2: Split the data
def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]


X_train, X_test, y_train, y_test = train_test_split(X, y)


# Step 3: Implement logistic regression
class CustomLogisticRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in y_predicted]


# Train the model
model = CustomLogisticRegression()
model.fit(X_train.values, y_train.values)


# Step 4: Evaluate the model
def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score


y_pred = model.predict(X_test.values)
accuracy, precision, recall, f1_score = calculate_metrics(y_test.values, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1_score:.4f}")


# Step 5: Create a confusion matrix
def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])


cm = confusion_matrix(y_test.values, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Step 6: Identify top three factors
feature_importance = pd.Series(abs(model.weights), index=X.columns)
top_features = feature_importance.nlargest(3)

print("\nTop 3 factors contributing to promotion usage prediction:")
for feature, importance in top_features.items():
    print(f"{feature}: {importance:.4f}")