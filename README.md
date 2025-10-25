# Smart Product Pricing Challenge - ML Challenge 2025 🏆

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![Stars](https://img.shields.io/github/stars/YOUR_USERNAME/smart-pricing-ml-challenge?style=social)](https://github.com/YOUR_USERNAME/smart-pricing-ml-challenge)

> An end-to-end machine learning solution for predicting optimal e-commerce product prices using ensemble methods and advanced feature engineering.

## 📋 Problem Statement

In e-commerce, determining the optimal price point for products is crucial for both marketplace success and customer satisfaction.

**Challenge**: Develop an ML solution that analyzes product details and predicts the price of a product. The relationship between product attributes and pricing is complex — factors like brand, specifications, and quantity directly influence pricing.

**Task**: Build a model that can holistically analyze product details and suggest an optimal price.

## 🗂️ Data Description

The dataset consists of the following columns:

| Column | Description |
|--------|-------------|
| `sample_id` | A unique identifier for each product sample |
| `catalog_content` | Text field containing product title, description, and Item Pack Quantity (IPQ) concatenated together |
| `image_link` | Public URL of the product image. Example: https://m.media-amazon.com/images/I/71XfHPR36-L.jpg |
| `price` | Target variable — the product price (available only in training data) |

### Dataset Details
- **Training Dataset**: 75,000 products with complete details and prices
- **Test Dataset**: 75,000 products (without prices, for evaluation)

## 🎯 Evaluation Criteria

Submissions are evaluated using **Symmetric Mean Absolute Percentage Error (SMAPE)**:

```
SMAPE = (1/n) × Σ |P_pred - P_actual| / ((|P_pred| + |P_actual|) / 2) × 100
```

**Example**: If actual price = $100 and predicted price = $120
```
SMAPE = |100 - 120| / ((100 + 120) / 2) × 100 = 18.18%
```

- SMAPE is bounded between 0% and 200%
- **Lower values indicate better performance**

### 🏆 Leaderboard Details
- **Public Leaderboard**: Based on 25K samples from the test set for real-time feedback
- **Final Rankings**: Based on the full 75K test set and documentation quality

## 🛠️ Tech Stack

- **Python 3.8+**
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Gradient Boosting**: LightGBM, XGBoost
- **Development**: Jupyter Notebook

## 📁 Project Structure

```
smart_pricing/
├── dataset/
│   ├── train.csv                 # Training data (75K products)
│   ├── test.csv                  # Test data (75K products)
│   ├── sample_test.csv           # Sample input file
│   └── sample_test_out.csv       # Example output format
├── notebook/
│   └── price_prediction.ipynb    # Main analysis notebook
├── output/
│   └── test_out.csv              # Final predictions
├── src/
│   ├── utils.py                  # Helper functions (image download)
│   └── sample_code.py            # Sample code reference
├── images/                       # Downloaded product images
├── models/                       # Saved model files
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── Documentation_Report.md       # 1-page methodology report
```

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/smart-pricing-ml-challenge.git
cd smart-pricing-ml-challenge
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Jupyter notebook
```bash
jupyter notebook
```
Navigate to `notebook/price_prediction.ipynb` and run all cells.

## 🧠 Methodology

### Feature Engineering

#### Text Features Extracted:
- **Basic Statistics**: text length, word count, average word length
- **Item Pack Quantity (IPQ)**: Extracted using regex patterns
- **Numerical Features**: Count, average, max, min of numbers in text
- **Keyword Detection**: 
  - Premium indicators (premium, deluxe, professional, organic)
  - Size indicators (large, small, medium, XL)
  - Material keywords (cotton, plastic, metal, wood)
- **Special Characters**: Count and distribution

#### Advanced Features (Optional):
- TF-IDF vectorization of product descriptions
- Word embeddings using pre-trained models
- Brand name extraction and encoding
- Category classification

### Models Implemented

| Model | Description | Validation SMAPE |
|-------|-------------|------------------|
| **Random Forest** | Baseline ensemble model | ~XX.XX% |
| **LightGBM** | Gradient boosting, optimized for speed | ~XX.XX% |
| **XGBoost** | Extreme gradient boosting | ~XX.XX% |
| **Ensemble** | Weighted average (0.4 LGB + 0.3 XGB + 0.3 RF) | ~**XX.XX%** ⭐ |

### Model Architecture

```python
# LightGBM Configuration
lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

### Training Pipeline

1. **Data Preprocessing**: Handle missing values, text cleaning
2. **Feature Engineering**: Extract features from catalog content
3. **Feature Scaling**: StandardScaler normalization
4. **Model Training**: Train multiple models with cross-validation
5. **Ensemble**: Combine predictions using weighted averaging
6. **Validation**: Calculate SMAPE on validation set
7. **Prediction**: Generate prices for test set

## 📊 Results

### Model Performance

| Metric | Value |
|--------|-------|
| Best Single Model | LightGBM |
| Validation SMAPE | XX.XX% |
| Public Leaderboard SMAPE | XX.XX% |
| Final Test SMAPE | XX.XX% |
| Training Time | ~XX minutes |

### Feature Importance

Top 5 most important features:
1. Pack Quantity (IPQ)
2. Maximum Number in Text
3. Text Length
4. Word Count
5. Premium Keywords

## 🧾 Output Format

The output file `test_out.csv` contains exactly two columns:

```csv
sample_id,price
12345,249.99
67890,109.00
...
```

**Requirements**:
- ✅ Exactly matches `sample_id` from test set
- ✅ Same number of rows as test data (75,000)
- ✅ All prices are positive float values
- ✅ No missing values

## ⚙️ Constraints & Rules

### Model Constraints
- ✅ Model must be under **8 billion parameters**
- ✅ Must use **MIT or Apache 2.0 licensed** libraries
- ✅ Predictions must be **positive floats**

### Academic Integrity - STRICTLY PROHIBITED ⚠️
- ❌ Web scraping product prices
- ❌ Using APIs to fetch market prices
- ❌ Manual lookup from websites
- ❌ Using external pricing datasets

**This challenge tests ML problem-solving skills using ONLY the provided data.**

## 💡 Key Features & Innovations

- ✨ Comprehensive text feature extraction with regex patterns
- 🤖 Multi-model ensemble for robust predictions
- 📈 Advanced validation with cross-validation
- 🎯 Optimized hyperparameters through experimentation
- 🔍 Outlier handling and data preprocessing
- 📊 Detailed EDA and visualization

## 🔮 Future Improvements

- [ ] **Image Features**: Extract visual features using ResNet/EfficientNet
- [ ] **Advanced NLP**: Use BERT/RoBERTa embeddings for text
- [ ] **Hyperparameter Tuning**: Implement Optuna/GridSearchCV
- [ ] **Deep Learning**: Neural network models for multi-modal fusion
- [ ] **Brand Extraction**: Advanced NER for brand identification
- [ ] **Category Prediction**: Multi-task learning for category + price
- [ ] **Data Augmentation**: Synthetic data generation techniques

## 📝 Deliverables

### 1. Output File (`test_out.csv`)
- Contains predicted prices for all 75K test samples
- Formatted according to competition requirements

### 2. Documentation Report
- **1-page report** describing:
  - Methodology used
  - Model architecture/algorithms
  - Feature engineering techniques
  - Implementation details
- Template: `Documentation_template.md`

## 📚 Dependencies

```txt
pandas==2.1.0
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
lightgbm==4.0.0
xgboost==2.0.0
Pillow==10.0.0
requests==2.31.0
jupyter==1.0.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/YOUR_USERNAME/smart-pricing-ml-challenge/issues).

### How to Contribute:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 👥 Team

**Project by**: [Your Name]

- 🌐 GitHub: [@your_username](https://github.com/AdityaC-07)
- 💼 LinkedIn: [Your Profile](https://linkedin.com/in/your_profile/aditya-choudhuri-87a2a034a)
- 📧 Email: adityatarit@gmail.com

## 🙏 Acknowledgments

- **ML Challenge 2025** organizers for this interesting problem
- **LightGBM** and **XGBoost** communities for excellent libraries
- **Scikit-learn** for comprehensive ML tools
- All contributors and supporters of this project

## 📞 Support

If you found this project helpful, please consider:
- ⭐ Starring the repository
- 🐛 Reporting issues
- 💡 Suggesting new features
- 📢 Sharing with others

---

<div align="center">

**Made with ❤️ for ML Challenge 2025**

[⬆ Back to Top](#smart-product-pricing-challenge---ml-challenge-2025-)

</div>
