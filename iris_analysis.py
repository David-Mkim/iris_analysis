# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Enable inline plotting for Jupyter (remove if using plain .py script)
# %matplotlib inline

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# ------------------------------
# Task 1: Load and Explore Dataset
# ------------------------------

try:
    # Load Iris dataset
    iris_raw = load_iris()
    iris = pd.DataFrame(data=iris_raw.data, columns=iris_raw.feature_names)
    iris['species'] = pd.Categorical.from_codes(iris_raw.target, iris_raw.target_names)

    print("✅ Dataset loaded successfully!\n")
    print("🔹 First 5 rows:")
    print(iris.head())

except Exception as e:
    print(f"❌ Error loading dataset: {e}")

# Check data types and missing values
print("\n🔹 Dataset info:")
print(iris.info())

print("\n🔹 Missing values:")
print(iris.isnull().sum())

# Clean dataset (Iris dataset has no missing values, but shown for demonstration)
iris_clean = iris.dropna()

# ------------------------------
# Task 2: Basic Data Analysis
# ------------------------------

print("\n📊 Basic Statistics:")
print(iris_clean.describe())

# Grouping by species and computing the mean of features
grouped = iris_clean.groupby("species").mean()
print("\n📌 Mean values per species:")
print(grouped)

# ------------------------------
# Task 3: Data Visualization
# ------------------------------

sns.set(style="whitegrid")

# 1. Line Chart (simulated time-series using index)
plt.figure(figsize=(8, 5))
plt.plot(iris_clean.index, iris_clean["sepal length (cm)"], label="Sepal Length")
plt.title("Simulated Time Series of Sepal Length")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.tight_layout()
plt.savefig("line_chart.png")
plt.show()

# 2. Bar Chart - Average petal length per species
plt.figure(figsize=(6, 4))
sns.barplot(data=iris_clean, x="species", y="petal length (cm)", estimator="mean", ci=None)
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.savefig("bar_chart.png")
plt.show()

# 3. Histogram of Sepal Width
plt.figure(figsize=(6, 4))
plt.hist(iris_clean["sepal width (cm)"], bins=15, color="skyblue", edgecolor="black")
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("histogram.png")
plt.show()

# 4. Scatter Plot: Sepal vs Petal Length
plt.figure(figsize=(6, 5))
sns.scatterplot(data=iris_clean, x="sepal length (cm)", y="petal length (cm)", hue="species")
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.show()

# ------------------------------
# Insights & Observations
# ------------------------------

print("\n📌 Observations:")
print("- Setosa species has shorter petal lengths and wider sepals on average.")
print("- Virginica tends to have the longest petals and sepals.")
print("- There's a clear separation in sepal vs. petal length by species, indicating useful features for classification.")
print("- Histogram shows Sepal Width is mostly around 3.0 cm.")
