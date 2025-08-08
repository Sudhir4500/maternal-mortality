# Maternal Mortality Analysis Project

A comprehensive data science project analyzing maternal mortality patterns using machine learning and statistical methods to identify risk factors and predict pregnancy outcomes.

## 🎯 Project Overview

This project analyzes maternal mortality data to:
- Identify key risk factors contributing to maternal mortality
- Build predictive models for high-risk pregnancies
- Analyze delivery location patterns and their impact on outcomes
- Provide insights for healthcare policy and intervention strategies

## 📊 Dataset Features

The analysis includes comprehensive maternal health indicators:

### Demographics & Socioeconomic
- Maternal age and age at first birth
- Educational level (mother and partner)
- Wealth index and household amenities
- Geographic location and residence type

### Healthcare Access & Delivery
- Number of antenatal care visits
- Place of delivery (with proper integer code mapping):
  - **10, 11, 12**: Home delivery (highest risk)
  - **20-27**: Public sector facilities
  - **30-36**: Private medical sector
  - **40-46**: NGO sector facilities
  - **21, 31**: Hospital delivery (lowest risk)
- Delivery method (caesarean vs normal)
- Healthcare provider type

### Environmental & Household
- Water source quality
- Sanitation facilities
- Household amenities score
- Living conditions

### Reproductive History
- Previous pregnancy outcomes
- Child mortality history
- Birth spacing patterns

## 🛠️ Technologies Used

- **Python 3.x**
- **Data Analysis**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Statistical Analysis**: scipy
- **Development Environment**: Jupyter notebooks, VS Code

## 📁 Project Structure

```
maternal-mortality/
├── data/                           # Data files (gitignored)
│   ├── raw/                       # Original survey data
│   ├── processed/                 # Cleaned datasets
│   └── external/                  # WHO/UNICEF reference data
├── notebooks/                     # Jupyter analysis notebooks
├── src/                          # Source code modules
│   ├── data_preprocessing.py     # Data cleaning functions
│   ├── feature_engineering.py    # Feature creation
│   ├── modeling.py               # ML model implementations
│   └── visualization.py          # Plotting functions
├── outputs/                      # Generated outputs (gitignored)
│   ├── plots/                   # Visualizations
│   ├── models/                  # Trained model files
│   └── reports/                 # Analysis reports
├── tests/                       # Unit tests
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
└── maternal_mortality_analysis_report.md  # Detailed findings
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Sudhir4500/maternal-mortality
cd maternal-mortality
```

2. **Create virtual environment**
```bash
python -m venv venv
```

3. **Activate virtual environment**
```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

1. **Data Preparation**
```python
# Place your maternal mortality dataset in data/raw/
# Ensure the dataset includes the delivery location integer codes

python src/data_preprocessing.py
```

2. **Feature Engineering**
```python
# Create enhanced features including delivery location mapping
python src/feature_engineering.py
```

3. **Run Analysis**
```python
# Execute the main analysis pipeline
python src/modeling.py
```

4. **View Results**
- Check `outputs/plots/` for visualizations
- Review `outputs/reports/` for detailed findings
- Open Jupyter notebooks for interactive analysis

## 🔍 Key Features

### Enhanced Delivery Location Analysis
- **Proper Integer Mapping**: Converts survey codes to meaningful categories
- **Risk Stratification**: Home delivery vs facility-based care
- **Quality Gradient**: 0-3 scoring system for delivery care quality
- **Facility Type Classification**: Public, private, NGO sector analysis

### Machine Learning Models
- **Random Forest**: For feature importance and non-linear relationships
- **Gradient Boosting**: For enhanced predictive accuracy
- **Logistic Regression**: For interpretable risk factor analysis

### Predictive Targets
- **Mortality Risk Score**: Continuous risk assessment
- **High-Risk Pregnancy**: Binary classification
- **Pregnancy Loss Prediction**: Historical outcome analysis

## 📈 Key Findings

### Delivery Location Impact
- **Home deliveries** (codes 10-12): Highest maternal mortality risk
- **Hospital deliveries** (codes 21, 31): Lowest risk with skilled care
- **Skilled delivery attendance**: Significant protective factor
- **Quality gradient**: Clear correlation between facility quality and outcomes

### Risk Factors Identified
1. **Maternal age** extremes (<20 or >35 years)
2. **Low educational attainment** (mother and partner)
3. **Inadequate antenatal care** (<4 visits)
4. **Home delivery** without skilled attendance
5. **Poor household conditions** (water, sanitation)
6. **Geographic isolation** (rural, remote areas)

### Model Performance
- **Random Forest**: Best overall performance for complex relationships
- **Feature Importance**: Delivery location among top 3 predictors
- **Prediction Accuracy**: 85%+ for high-risk pregnancy classification

## 📊 Visualizations

The project generates comprehensive visualizations:
- **Risk factor correlation heatmaps**
- **Delivery location outcome comparisons**
- **Geographic mortality pattern maps**
- **Model performance metrics**
- **Feature importance rankings**

## 🔬 Methodology

### Data Preprocessing
1. **Missing value imputation** using domain knowledge
2. **Outlier detection and treatment**
3. **Feature standardization** for model compatibility

### Feature Engineering
1. **Delivery location mapping** from integer codes
2. **Risk score creation** using clinical guidelines
3. **Categorical encoding** for machine learning
4. **Interaction terms** for complex relationships

### Model Validation
1. **Train-test split** (80-20) with stratification
2. **Cross-validation** for robust performance estimates
3. **Feature importance analysis** for interpretability
4. **Clinical validation** against known risk factors

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 coding standards
- Add unit tests for new functions
- Update documentation for new features
- Ensure HIPAA compliance for any health data

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🏥 Clinical Disclaimer

This analysis is for research purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for clinical decisions.

## 📚 References

- World Health Organization (WHO) Maternal Mortality Guidelines
- UNICEF State of the World's Children Reports
- Demographic and Health Survey (DHS) Methodology
- Clinical guidelines for maternal risk assessment

## 📞 Contact

- **Author**:Sudhir4500
- **Project Link**: https://github.com/Sudhir4500/maternal-mortality

## 🙏 Acknowledgments

- WHO for maternal health guidelines and data standards
- UNICEF for child and maternal health research
- DHS Program for survey methodology
- Open source community for amazing tools and libraries

---

**⚕️ For Healthcare Professionals**: This tool provides data-driven insights to support evidence-based maternal health interventions and policy development.

**📊 For Data Scientists**: Demonstrates healthcare analytics best practices with proper medical coding, clinical validation, and interpretable ML models.

**🌍 For Public Health**: Contributes to global efforts in reducing maternal mortality through data science and predictive analytics.