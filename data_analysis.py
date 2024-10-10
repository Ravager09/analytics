# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (Replace 'your_dataset.csv' with the actual dataset path)
df = pd.read_csv('dataset/sample_customer_data_for_exam.csv')

# a. Display the first few rows and summary statistics for numerical columns
print("First few rows of the dataset:")
print(df.head())

print("\nSummary statistics for numerical columns:")
print(df.describe())

# b. Create a heatmap to visualize the correlation between numerical variables
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
correlation_matrix = df[numerical_cols].corr()

# Create the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,  # Show correlation values
    cmap='coolwarm',  # Color scheme from red (negative) to blue (positive)
    center=0,  # Center the colormap at 0
    fmt='.2f',  # Format correlation values to 2 decimal places
    square=True,  # Make the plot square-shaped
    linewidths=0.5  # Add gridlines
)

# Customize the plot
plt.title('Correlation Heatmap of Numerical Variables', pad=20, size=16)
plt.tight_layout()

# Show the plot
plt.show()

# Print the correlation matrix (optional)
print("\nCorrelation Matrix:")
print(correlation_matrix.round(2))
# c. Create histograms for the "age" and "income" columns
plt.figure(figsize=(12, 5))

# Histogram for 'age'
plt.subplot(1, 2, 1)
plt.hist(df['age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Histogram for 'income'
plt.subplot(1, 2, 2)
plt.hist(df['income'], bins=20, color='lightgreen', edgecolor='black')
plt.title('Distribution of Income')
plt.xlabel('Income')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# d. Generate a box plot to show the distribution of "purchase_amount" across different "product_factory" values
plt.figure(figsize=(12, 6))

# Create the box plot
sns.boxplot(
    data=df,
    x='product_category',
    y='purchase_amount',
    palette='Set3'  # Use a colorful palette for different categories
)

# Customize the plot
plt.title('Distribution of Purchase Amounts by Product Category', pad=20, size=14)
plt.xlabel('Product Category', size=12)
plt.ylabel('Purchase Amount', size=12)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Show the plot
plt.show()

# Print summary statistics (optional)
print("\nSummary Statistics by Product Category:")
print(df.groupby('product_category')['purchase_amount'].describe().round(2))
# e. Create a pie chart to visualize the proportion of customers by "gender"
gender_counts = df['gender'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['lightcoral', 'lightskyblue'], startangle=90)
plt.title('Proportion of Customers by Gender')
plt.axis('equal')  # Equal aspect ratio ensures the pie is drawn as a circle.
plt.show()


# Part B


# Calculate average purchase amount for each education level
result = df.groupby('education')['purchase_amount'].mean().reset_index()

# Sort the result by average purchase amount in descending order
result = result.sort_values('purchase_amount', ascending=False)

# Display the result
print(result)

result = df.groupby('loyalty_status')['satisfaction_score'].mean().reset_index()

# Sort the result by average satisfaction score in descending order
result = result.sort_values('satisfaction_score', ascending=False)

# Display the result
print(result)

total_customers = len(df)
promo_users = df['promotion_usage'].sum()
percentage_promo_users = (promo_users / total_customers) * 100

# Display the result
print(f"Percentage of customers who used promotional offers: {percentage_promo_users:.2f}%")

correlation = df['income'].corr(df['purchase_amount'])

# Create a scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='income', y='purchase_amount', data=df, alpha=0.6)

# Add a trend line
sns.regplot(x='income', y='purchase_amount', data=df, scatter=False, color='red')

# Customize the plot
plt.title(f'Correlation between Income and Purchase Amount\nCorrelation Coefficient: {correlation:.2f}', fontsize=16)
plt.xlabel('Income', fontsize=12)
plt.ylabel('Purchase Amount', fontsize=12)

# Display the plot
plt.tight_layout()
plt.show()

# Print the correlation coefficient
print(f"The correlation coefficient between income and purchase amount is: {correlation:.2f}")

# Interpret the correlation
if abs(correlation) < 0.3:
    strength = "weak"
elif abs(correlation) < 0.7:
    strength = "moderate"
else:
    strength = "strong"

direction = "positive" if correlation > 0 else "negative"

print(f"This indicates a {strength} {direction} correlation.")

# Additional analysis: group by income ranges and calculate average purchase amount
df['income_range'] = pd.cut(df['income'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
grouped_data = df.groupby('income_range')['purchase_amount'].mean().reset_index()
print("\nAverage Purchase Amount by Income Range:")
print(grouped_data)

# part C

plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='purchase_frequency', y='purchase_amount', hue='loyalty_status', palette='viridis')

# Customize the plot
plt.title('Purchase Frequency vs Purchase Amount by Loyalty Status', fontsize=16)
plt.xlabel('Purchase Frequency', fontsize=12)
plt.ylabel('Purchase Amount', fontsize=12)

# Add a legend
plt.legend(title='Loyalty Status', title_fontsize='12', fontsize='10')

# Improve the layout
plt.tight_layout()

# Show the plot
plt.show()

# Optional: Add some basic statistics
print("Basic Statistics:")
print(df.groupby('loyalty_status')[['purchase_frequency', 'purchase_amount']].agg(['mean', 'median', 'std']))

# Optional: Calculate correlation for each loyalty status
for status in df['loyalty_status'].unique():
    subset = df[df['loyalty_status'] == status]
    correlation = subset['purchase_frequency'].corr(subset['purchase_amount'])
    print(f"\nCorrelation between purchase frequency and amount for {status} customers: {correlation:.2f}")


df = pd.DataFrame(data)

# Calculate average purchase amount for customers who used promotions
promo_avg = df[df['used_promotion']]['purchase_amount'].mean()

# Calculate average purchase amount for customers who didn't use promotions
no_promo_avg = df[~df['used_promotion']]['purchase_amount'].mean()

print(f"Average purchase amount for customers who used promotions: ${promo_avg:.2f}")
print(f"Average purchase amount for customers who didn't use promotions: ${no_promo_avg:.2f}")

# Calculate the difference
difference = promo_avg - no_promo_avg
print(f"Difference in average purchase amount: ${difference:.2f}")

# Perform a simple statistical test (t-test) to check if the difference is significant
from scipy import stats

promo_purchases = df[df['used_promotion']]['purchase_amount']
no_promo_purchases = df[~df['used_promotion']]['purchase_amount']

t_statistic, p_value = stats.ttest_ind(promo_purchases, no_promo_purchases)

print(f"T-statistic: {t_statistic}")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("The difference in average purchase amounts is statistically significant.")
else:
    print("The difference in average purchase amounts is not statistically significant.")


df = pd.read_csv('dataset/sample_customer_data_for_exam.csv')

# Check if the required columns exist
if 'satisfaction_score' not in df.columns or 'purchase_frequency' not in df.columns:
    raise ValueError("Required columns 'satisfaction_score' or 'purchase_frequency' not found in the dataset.")

# Calculate the correlation
correlation = df['satisfaction_score'].corr(df['purchase_frequency'])

print(f"The correlation between satisfaction score and purchase frequency is: {correlation:.4f}")

# Optional: Calculate the p-value
from scipy import stats

correlation_coefficient, p_value = stats.pearsonr(df['satisfaction_score'], df['purchase_frequency'])

print(f"P-value: {p_value:.4f}")