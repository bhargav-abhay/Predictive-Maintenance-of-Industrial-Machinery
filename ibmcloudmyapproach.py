import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import altair as alt

# Load the dataset
df = pd.read_csv(r"C:\Users\ABHAY TRIPATHI\OneDrive\Desktop\ibm project\predictive_maintenance.csv")

# Preprocessing
# Drop unnecessary columns
df = df.drop(columns=['UDI', 'Product ID', 'Target'])

# Identify categorical and numerical features
numerical_features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
categorical_features = ['Type']
target_column = 'Failure Type'

# Encode the target variable
label_encoder = LabelEncoder()
df[target_column] = label_encoder.fit_transform(df[target_column])

# One-hot encode the 'Type' column
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Split data into features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale numerical features
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='liblinear'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

accuracy_scores = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy_scores[name] = accuracy_score(y_test, y_pred)

print("Accuracy Scores:")
for name, score in accuracy_scores.items():
    print(f"{name}: {score:.4f}")

# Create a DataFrame for visualization
accuracy_df = pd.DataFrame(list(accuracy_scores.items()), columns=['Model', 'Accuracy'])

# Create a bar chart with Altair
chart = alt.Chart(accuracy_df).mark_bar().encode(
    x=alt.X('Model', sort='-y', title='Model'),
    y=alt.Y('Accuracy', title='Accuracy'),
    tooltip=['Model', alt.Tooltip('Accuracy', format='.4f')]
).properties(
    title='Comparison of Model Accuracy'
)

# Save the chart as a JSON file
chart.save('model_accuracy_comparison.json')
