# KNN Diabetes Dataset Classification

A machine learning project that uses the K-Nearest Neighbors (KNN) algorithm to predict diabetes outcomes based on various health metrics.

## 📊 Dataset Overview

The project uses the Pima Indians Diabetes Database, which contains health information for 768 individuals with the following features:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration in a 2-hour oral glucose tolerance test
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age in years
- **Outcome**: Target variable (0 = No diabetes, 1 = Diabetes)

## 🔧 Data Preprocessing

The project implements comprehensive data preprocessing steps:

### 1. Data Imputation
- Handles missing values (represented as 0) in critical features
- Uses **median imputation** for the following columns:
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI

### 2. Feature Scaling
- Applies **MinMaxScaler** to normalize all features to a 0-1 range
- Ensures all features contribute equally to the distance calculations in KNN

## 🤖 Machine Learning Implementation

### Algorithm: K-Nearest Neighbors (KNN)
- **Optimal K Selection**: Calculates the square root of training samples to determine the ideal number of neighbors
- **K Value**: 25 neighbors (based on mathematical optimization)
- **Distance Metric**: Euclidean distance (default)

### Model Pipeline
1. **Data Split**: 80% training, 20% testing
2. **Preprocessing**: Imputation → Scaling
3. **Training**: KNN model fitting
4. **Prediction**: Classification on test set
5. **Evaluation**: Multiple metrics assessment

## 📈 Model Evaluation

The model performance is evaluated using:
- **Accuracy Score**: Overall prediction accuracy
- **Confusion Matrix**: Detailed classification breakdown
- **Precision Score**: Positive prediction accuracy

## 🚀 Getting Started

### Prerequisites
```python
pip install numpy pandas scikit-learn
```

### Required Libraries
- `numpy`: Numerical computations
- `pandas`: Data manipulation and analysis
- `scikit-learn`: Machine learning algorithms and preprocessing
- `math`: Mathematical operations

### Running the Project
1. Clone or download the repository
2. Ensure `diabetes.csv` is in the project directory
3. Open `KNN.ipynb` in Jupyter Notebook or VS Code
4. Run all cells sequentially

## 📁 Project Structure
```
KNN_DiabetiesDataset/
├── diabetes.csv          # Dataset file
├── KNN.ipynb            # Main Jupyter notebook
└── README.md            # Project documentation
```

## 🎯 Key Features

- **Robust Data Handling**: Comprehensive preprocessing pipeline
- **Optimized Hyperparameters**: Mathematical approach to K selection
- **Multiple Evaluation Metrics**: Thorough model assessment
- **Clean Code Structure**: Well-organized and documented implementation

## 📊 Results

The model provides predictions for diabetes classification with detailed performance metrics including accuracy, confusion matrix, and precision scores.

## 🔮 Future Improvements

- Cross-validation for more robust model evaluation
- Hyperparameter tuning using Grid Search or Random Search
- Feature importance analysis
- Comparison with other classification algorithms (Random Forest, SVM, etc.)
- ROC curve and AUC score analysis

## 📝 License

This project is for educational and research purposes.

## 👨‍💻 Author

Sharjeel - Data Science and Machine Learning Implementation

---

*This project demonstrates the application of KNN algorithm for medical diagnosis prediction using proper data preprocessing and evaluation techniques.*