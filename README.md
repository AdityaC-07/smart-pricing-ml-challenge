# smart-pricing-ml-challenge
ML Challenge 2025 - Smart Product Pricing using LightGBM, XGBoost, and ensemble methods
# Smart Product Pricing Challenge - ML Challenge 2025

## 🎯 Problem Statement
Develop a machine learning solution to predict optimal product prices based on catalog content and product images.

## 📊 Dataset
- **Training Set**: 75,000 products with prices
- **Test Set**: 75,000 products for prediction
- **Features**: 
  - `catalog_content`: Product title, description, and Item Pack Quantity (IPQ)
  - `image_link`: Product image URL
  - `price`: Target variable (training only)

## 🛠️ Tech Stack
- Python 3.x
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- LightGBM, XGBoost
- Jupyter Notebook

## 📁 Project Structure
```
smart_pricing/
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   └── sample_test_out.csv
├── notebook/
│   └── price_prediction.ipynb
├── output/
│   └── submission.csv
├── src/
│   ├── utils.py
│   └── sample_code.py
├── README.md
└── requirements.txt
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

## 🧠 Model Approach

### Feature Engineering
- **Text Features**:
  - Text length, word count, average word length
  - Item Pack Quantity (IPQ) extraction
  - Premium keywords detection
  - Size and material indicators
  - Special character counts

### Models Used
1. **Random Forest** (Baseline): SMAPE ~X%
2. **LightGBM**: SMAPE ~Y%
3. **XGBoost**: SMAPE ~Z%
4. **Ensemble**: Weighted average of all models

### Results
- Best Single Model: LightGBM
- Final Ensemble SMAPE: **XX.XX%**

## 📈 Evaluation Metric
**SMAPE (Symmetric Mean Absolute Percentage Error)**
```
SMAPE = (100/n) × Σ |predicted - actual| / (|predicted| + |actual|)
```

## 💡 Key Features
- Comprehensive text feature extraction
- Multiple model comparison
- Ensemble predictions for better accuracy
- Complete validation pipeline

## 🔮 Future Improvements
- [ ] Image feature extraction using ResNet/EfficientNet
- [ ] Advanced NLP with BERT embeddings
- [ ] Hyperparameter optimization using Optuna
- [ ] Deep learning models (Neural Networks)
- [ ] Brand name extraction and encoding

## 📝 License
This project is licensed under the MIT License.

## 👤 Author
**Your Name**
- GitHub: [@your_username](https://github.com/your_username)
- LinkedIn: [Your Profile](https://linkedin.com/in/your_profile)

## 🙏 Acknowledgments
- ML Challenge 2025 organizers
- Scikit-learn and LightGBM communities
