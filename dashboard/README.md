# Fraud Detection Dashboard

Interactive dashboard for visualizing fraud detection results and analyzing model performance.

## 🚀 Quick Start

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will be available at: `http://localhost:8501`

### Requirements

The dashboard requires the following files to be generated first:

1. **Training Results**: `results/training_results.json`
2. **Predictions**: `results/predictions_with_probabilities.csv`
3. **Processed Data**: `data/processed/train_processed.csv`
4. **Feature Importance**: `results/feature_importance.csv`

Generate these files by running:

```bash
# Train the model
python scripts/train.py

# Make predictions
python scripts/predict.py
```

## 📊 Dashboard Pages

### 1. Overview
- Key performance metrics (ROC-AUC, PR-AUC, F1)
- Class distribution visualization
- Temporal fraud analysis
- Quick summary statistics

### 2. Model Performance
- Detailed evaluation metrics
- Accuracy, Precision, Recall, F1-Score
- Transaction amount analysis by fraud status
- Business impact estimation
  - Fraud detected and saved amount
  - False positive costs
  - ROI calculations

### 3. Feature Analysis
- Feature importance visualization
- Top N most important features (configurable)
- Interactive feature importance table
- Feature contribution to predictions

### 4. Predictions
- Risk segmentation of evaluation set
  - Very Low, Low, Medium, High, Very High
- Fraud probability distribution
- Highest risk transactions
- Prediction statistics

### 5. Transaction Search
- Search by transaction ID
- View detailed fraud probability
- Risk level assessment
- Color-coded alerts

## 🎨 Features

### Interactive Visualizations
- **Plotly Charts**: Interactive, zoomable, and exportable charts
- **Real-time Filtering**: Filter data by date, risk level, etc.
- **Responsive Design**: Works on desktop and tablet

### Metrics Display
- **KPI Cards**: Key metrics prominently displayed
- **Color Coding**: Red for high risk, green for low risk
- **Percentage and Counts**: Both relative and absolute values

### Business Insights
- **ROI Calculation**: Estimate financial impact
- **False Positive Cost**: Manual review cost estimation
- **Fraud Detection Rate**: Percentage of fraud caught

## 🔧 Configuration

### Port Configuration

Change the default port (8501):

```bash
streamlit run dashboard/app.py --server.port 8080
```

### Theme Customization

Edit the custom CSS in `app.py` to change colors and styling:

```python
st.markdown("""
<style>
    .main-header {
        color: #YOUR_COLOR;
    }
</style>
""", unsafe_allow_html=True)
```

### Data Refresh

The dashboard caches data for performance. To force refresh:

1. Click "☰" menu (top right)
2. Select "Clear cache"
3. Or restart the dashboard

## 📱 Deployment

### Local Network Access

Allow access from other devices on your network:

```bash
streamlit run dashboard/app.py --server.address 0.0.0.0
```

Then access via: `http://YOUR_IP:8501`

### Streamlit Cloud Deployment

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Deploy `dashboard/app.py`

**Note**: Ensure required data files are included or generated during deployment.

### Heroku Deployment

1. Create `Procfile`:
```
web: streamlit run dashboard/app.py --server.port $PORT
```

2. Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
```

3. Deploy:
```bash
heroku create your-fraud-detection-app
git push heroku main
```

## 🐛 Troubleshooting

### Dashboard won't start

**Error**: `ModuleNotFoundError: No module named 'streamlit'`

**Solution**:
```bash
pip install streamlit
```

### Missing data files

**Error**: `FileNotFoundError: results/training_results.json`

**Solution**: Run training and prediction first:
```bash
python scripts/train.py
python scripts/predict.py
```

### Slow performance

**Solution**:
- Reduce dataset size for visualization
- Use data sampling for large datasets
- Clear Streamlit cache regularly

### Port already in use

**Error**: `Address already in use`

**Solution**: Change port or kill existing process:
```bash
# Change port
streamlit run dashboard/app.py --server.port 8502

# Or kill existing process
lsof -ti:8501 | xargs kill
```

## 📊 Screenshots

(Add screenshots of your dashboard here after running it)

### Overview Page
![Overview](../docs/figures/dashboard_overview.png)

### Model Performance
![Performance](../docs/figures/dashboard_performance.png)

### Feature Importance
![Features](../docs/figures/dashboard_features.png)

## 🎯 Use Cases

### 1. Model Evaluation
Review model performance metrics and validate predictions before deployment.

### 2. Stakeholder Presentation
Demonstrate fraud detection capabilities to business stakeholders with interactive visualizations.

### 3. Transaction Investigation
Search and analyze specific high-risk transactions for manual review.

### 4. Business Impact Analysis
Calculate ROI and justify investment in fraud detection system.

## 🔐 Security Considerations

- **No PII Display**: Avoid displaying personally identifiable information
- **Access Control**: Implement authentication for production deployment
- **Data Encryption**: Use HTTPS in production
- **Audit Logging**: Log all transaction searches and risk reviews

## 📝 Customization

### Adding New Pages

Add a new page to the sidebar:

```python
# In app.py
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Model Performance", "Your New Page"]
)

# Add page logic
if page == "Your New Page":
    st.header("Your New Page")
    # Your custom code here
```

### Custom Metrics

Add custom business metrics:

```python
def calculate_custom_metric(predictions):
    # Your calculation
    return result

# Display in dashboard
st.metric("Custom Metric", calculate_custom_metric(predictions))
```

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Plotly Documentation](https://plotly.com/python/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

## 🤝 Contributing

To improve the dashboard:

1. Fork the repository
2. Create a feature branch
3. Add your enhancements
4. Test thoroughly
5. Submit a pull request

## 📧 Support

For dashboard issues:
- Check the logs: `streamlit run dashboard/app.py --logger.level=debug`
- Review Streamlit documentation
- Open an issue on GitHub

---

**Built with Streamlit and ❤️**
