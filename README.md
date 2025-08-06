# Customer Segmentation using Neural Networks

An enhanced customer segmentation project using neural networks in TensorFlow with a Streamlit frontend.

## Features

### 📊 Enhanced Data Processing
- Comprehensive data cleaning and preprocessing
- Handling missing values and outliers
- Date/time feature extraction

### 🔍 Advanced Feature Engineering
- **RFM Analysis**: Recency, Frequency, Monetary value features
- **Customer Behavior**: Purchase patterns and product diversity
- **Statistical Features**: Average quantities, unit prices, unique products

### 🎯 Smart Feature Selection
- Correlation analysis and feature importance
- Selection of top-performing features
- Visualization of feature relationships

### 🤖 Optimized Model Training
- Multi-layer neural network with dropout regularization
- Early stopping and learning rate scheduling
- Hyperparameter tuning for optimal performance

### 🖥️ Interactive Streamlit Frontend
- **Single Prediction**: Predict individual customers
- **Batch Prediction**: Upload CSV files for bulk predictions
- **Analytics Dashboard**: Model insights and business intelligence

## Project Structure

```
Customer-Segmentation-using-Neural-Networks-in-TensorFlow/
├── main.ipynb                          # Enhanced Jupyter notebook
├── streamlit_app.py                    # Streamlit web application
├── requirements.txt                    # Python dependencies
├── Online Retail.csv                  # Dataset
├── customer_segmentation_model.h5      # Trained model (generated)
├── scaler.pkl                         # Feature scaler (generated)
├── feature_names.pkl                  # Feature names (generated)
└── label_encoder.pkl                  # Label encoder (generated)
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook
Open and run `main.ipynb` to:
- Process the data
- Train the enhanced model
- Generate model files

### 3. Launch Streamlit App
```bash
streamlit run streamlit_app.py
```

## Model Enhancement Details

### 1. Data Preprocessing ✨
- **Data Quality**: Removed null values, invalid quantities, and negative prices
- **Feature Engineering**: Created meaningful customer-level aggregations
- **Time-based Features**: Extracted recency information from purchase dates

### 2. Feature Extraction 🔧
- **RFM Features**: 
  - Recency: Days since last purchase
  - Frequency: Number of transactions
  - Monetary: Total spending amount
- **Behavioral Features**:
  - Average quantity per transaction
  - Number of unique products purchased
  - Average unit price preference

### 3. Feature Selection 📈
- **Correlation Analysis**: Identified relationships between features
- **Target Correlation**: Selected features most correlated with high-value customers
- **Feature Importance**: Ranked features by predictive power

### 4. Model Training & Tuning ⚙️
- **Architecture**: Multi-layer perceptron with 128-64-32-1 neurons
- **Regularization**: Dropout layers to prevent overfitting
- **Optimization**: Adam optimizer with learning rate scheduling
- **Callbacks**: Early stopping and learning rate reduction
- **Metrics**: Accuracy, precision, recall, and AUC score

## Model Performance

- **Accuracy**: ~99%+
- **Precision**: High precision for both classes
- **Recall**: Balanced recall across segments
- **AUC Score**: Excellent discrimination capability

## Business Value

### Customer Insights 💡
- Identify high-value customers for retention campaigns
- Understand purchasing behavior patterns
- Optimize marketing spend allocation

### Actionable Recommendations 🎯
- **High-Value Customers**: Focus on retention and premium services
- **Standard Customers**: Implement upselling and cross-selling strategies
- **Behavioral Segmentation**: Tailor marketing messages by segment

## Streamlit Features

### Single Prediction 👤
- Input individual customer characteristics
- Get real-time predictions with confidence scores
- Visual probability gauge for easy interpretation

### Batch Prediction 📋
- Upload CSV files with customer data
- Process multiple customers simultaneously
- Download results with predictions and probabilities

### Analytics Dashboard 📊
- Feature importance visualization
- Model performance metrics
- Business insights and recommendations

## Usage Examples

### Single Customer Prediction
```python
# Example input
frequency = 25
monetary = 1500.0
total_quantity = 200
unique_products = 30
avg_unit_price = 7.5

# Result: High-Value Customer (85% confidence)
```

### Batch Processing
Upload a CSV with columns:
- `Frequency`
- `Monetary`
- `TotalQuantity`
- `UniqueProducts`
- `AvgUnitPrice`

## Future Enhancements

- [ ] Real-time model retraining
- [ ] Advanced customer lifetime value prediction
- [ ] Clustering-based micro-segmentation
- [ ] API endpoints for integration
- [ ] A/B testing framework for model comparison

## Technologies Used

- **Python**: Core programming language
- **TensorFlow/Keras**: Neural network framework
- **Scikit-learn**: Machine learning utilities
- **Streamlit**: Web application framework
- **Pandas/NumPy**: Data manipulation
- **Plotly**: Interactive visualizations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the MIT License.
