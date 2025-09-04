# Data Science & Machine Learning Projects 

Welcome to my data science and machine learning projects repository! This collection showcases comprehensive analyses and predictive modeling solutions addressing real-world problems in education and healthcare.

## 🚀 Projects Overview

| Project | Type | Domain | Key Technologies | Status |
|---------|------|--------|-----------------|--------|
| [Academic Stress EDA](#-Unit_1_Task_1) | Exploratory Data Analysis | Education | Python, Pandas, Seaborn | ✅ Complete |
| [Heart Disease Prediction](#-Unit_1_Taks_2) | Machine Learning | Healthcare | Python, Scikit-learn | ✅ Complete |

---

## 📚 Academic Stress EDA: Insights and Recommendations

### 📊 Project Overview

This project presents an Exploratory Data Analysis (EDA) of academic stress among students, based on a comprehensive dataset containing 139 student responses. The analysis examines various factors contributing to academic stress, including peer pressure, academic pressure from home, study environment, coping strategies, and academic competition.

### 🔍 What is Exploratory Data Analysis (EDA)?

EDA is a critical first step in data analysis that helps us:

- Understand the structure and characteristics of our dataset
- Identify patterns, relationships, and trends in the data
- Detect anomalies and outliers
- Formulate hypotheses for further investigation
- Guide data-driven decision making

### 📈 Key Visualizations

Our analysis employed several visualization techniques:

- **Distribution Plots**: Bar charts showing frequency distributions of categorical variables
- **Correlation Heatmap**: Visualized relationships between numerical variables
- **Pairplot**: Grid of scatter plots showing pairwise relationships
- **Boxplots**: Compared stress index distributions across different categories
- **Pie Charts**: Displayed proportional representation of categorical responses
- **Grouped Bar Plots**: Compared average stress across multiple factors

### 🎯 Key Findings and Recommendations

#### 1. Promote Peaceful Study Environments
**Finding**: Students reporting a "Peaceful" study environment showed lower stress levels.

**Actions**:
- Invest in quiet study zones on campus
- Promote noise-canceling tools for students
- Encourage better time management

#### 2. Encourage Intellectual Coping Strategies
**Finding**: Students using analytical coping strategies reported lower stress levels.

**Actions**:
- Integrate problem-solving workshops
- Promote mindfulness and cognitive-behavioral techniques

#### 3. Address Peer and Home Pressure
**Finding**: Higher peer and home pressure correlated with increased stress.

**Actions**:
- Launch parental awareness programs
- Introduce peer-mentoring programs

### 📁 Dataset Information

| Variable | Description |
|----------|-------------|
| Academic Stage | undergraduate, high school, post-graduate |
| Peer pressure | 1-5 rating scale |
| Academic pressure from home | 1-5 rating scale |
| Study Environment | Noisy, Peaceful, Disrupted |
| Coping strategy | Various coping mechanisms |
| Academic competition rating | 1-5 rating scale |
| Academic stress index rating | 1-5 rating scale |

---

## ❤️ Heart Disease Prediction using Machine Learning

### 🎯 Real-World Problem

This project addresses the critical healthcare challenge of early heart disease detection. Cardiovascular diseases are the leading cause of death globally, claiming approximately 17.9 million lives each year according to the World Health Organization. Early detection can significantly improve treatment outcomes and save lives.

### 📊 Dataset Overview

Uses the Cleveland Heart Disease dataset from UCI Machine Learning Repository with 303 patient records and 14 clinical attributes:

**Features**:
- **Demographic**: age, sex
- **Medical history**: chest pain type, resting blood pressure, cholesterol, fasting blood sugar
- **Exercise-induced**: maximum heart rate, exercise-induced angina
- **ECG measurements**: ST depression, slope of peak exercise ST segment
- **Cardiac indicators**: number of major vessels, thalassemia type

**Target Variable**: 0 = no heart disease, 1 = heart disease present

### 🔧 Machine Learning Pipeline

#### 1. Data Preprocessing
- ✅ No missing values detected
- ✅ Stratified train-test split (80-20)
- ✅ Maintained class distribution

#### 2. Model Performance
- **Algorithm**: Logistic Regression
- **Training Accuracy**: 85.12%
- **Test Accuracy**: 81.97%
- **Generalization**: Excellent (minimal overfitting)

#### 3. Model Selection Rationale
- Interpretable coefficients for medical applications
- Excellent binary classification performance
- Efficient with moderate-sized datasets
- Provides probabilistic outputs

### 🚀 Usage Example

```python
# Making predictions on new patient data
input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)
prediction = model.predict(input_data)

if prediction[0] == 0:
    print('The Person does not have Heart Disease')
else:
    print('The Person has Heart Disease')
```

### 🌍 Real-World Applications

- **Clinical Decision Support**: Assist doctors with data-driven second opinions
- **Telemedicine**: Enable remote heart disease screening
- **Preventive Healthcare**: Identify at-risk individuals early
- **Resource Optimization**: Help prioritize patients needing cardiac care
- **Health Monitoring Apps**: Integration with wearable devices
- **Medical Research**: Pattern identification for risk factors

---

## 🛠️ Technical Stack

### Languages & Libraries
```python
# Core Data Science
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Statistical Analysis
from scipy import stats
```

### Development Environment
- **Python**: 3.8+
- **Jupyter Notebook**: For interactive analysis
- **Git**: Version control

## 📂 Repository Structure

```
data-science-portfolio/
├── Unit_1_Task_1/
│   ├── Acadmic_Stress_EDA.ipynb
│   ├── academic_Stress.csv
|   
├── Unit_1_Task_2/
│   ├──ML Use Case 4. Heart_Disease_Prediction.ipynb
│   ├── heart_disease_data.csv
|── Explaination.docx
|── requirements.txt
|── README.md

```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Installation & Usage
1. **Clone the repository**
   ```bash
   git clone https://github.com/HariomSinghalPuri/Tasks.git
   cd Unit_1_Task_1
   cd Unit_1_Task_2

   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

4. **Explore the projects**
   - Navigate to individual project folders
   - Open the respective `.ipynb` files
   - Run cells to reproduce results

## 📊 Key Results Summary

| Project | Dataset Size | Key Metric | Result | Impact |
|---------|--------------|------------|---------|---------|
| Academic Stress EDA | 139 students | Insights Generated | 3 major findings | Educational policy recommendations |
| Heart Disease Prediction | 303 patients | Test Accuracy | 81.97% | Clinical decision support |

## 🔮 Future Enhancements

### Academic Stress EDA
- [ ] Predictive modeling for stress levels
- [ ] Longitudinal analysis implementation
- [ ] Interactive dashboard creation
- [ ] Statistical significance testing

### Heart Disease Prediction
- [ ] Model ensemble implementation
- [ ] Hyperparameter optimization
- [ ] ROC curve and AUC analysis
- [ ] Web application deployment
- [ ] Real-time prediction API

## ⚠️ Important Considerations

### Academic Stress Project
- Results should inform policy but require validation across diverse student populations
- Privacy considerations for student data handling

### Heart Disease Project
- **Medical Disclaimer**: This model is for educational purposes and should not replace professional medical diagnosis
- Requires rigorous clinical validation before real-world medical application
- Patient data privacy and security must be prioritized

## 📚 References & Resources

### Academic Stress EDA
- Educational psychology research on stress factors
- Statistical analysis best practices
- Data visualization principles

### Heart Disease Prediction
- [UCI Machine Learning Repository: Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- [WHO: Cardiovascular diseases fact sheet](https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds))
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Open issues for bugs or feature requests
- Submit pull requests for improvements
- Share feedback and suggestions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⭐ **If you found this repository helpful, please consider giving it a star!**

- 📧 **Contact**: hsinghalpuri@gmail.com | 
- 💼 **LinkedIn**: https://www.linkedin.com/in/hariom-singhal-puri |
- 🐱 **GitHub**: https://github.com/HariomSinghalPuri
