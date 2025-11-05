"""
Fraud Detection Dashboard - Streamlit Application
Interactive dashboard for fraud detection visualization and analysis
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .fraud-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">🔍 Fraud Detection Dashboard</div>', unsafe_allow_html=True)
st.markdown("### Hackathon Finance Track - Real-time Fraud Analysis")
st.markdown("---")


@st.cache_data
def load_data():
    """Load all necessary data"""
    try:
        # Load training results
        with open('../results/training_results.json', 'r') as f:
            training_results = json.load(f)

        # Load predictions
        predictions = pd.read_csv('../results/predictions_with_probabilities.csv')

        # Load processed training data (for visualization)
        train_data = pd.read_csv('../data/processed/train_processed.csv')

        # Load feature importance
        feature_importance = pd.read_csv('../results/feature_importance.csv')

        return training_results, predictions, train_data, feature_importance
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please run the training script first to generate the necessary files.")
        return None, None, None, None


def plot_class_distribution(train_data):
    """Plot fraud vs non-fraud distribution"""
    if 'is_fraud' not in train_data.columns:
        return None

    fraud_counts = train_data['is_fraud'].value_counts()

    fig = go.Figure(data=[
        go.Bar(x=['Not Fraud', 'Fraud'],
              y=fraud_counts.values,
              marker_color=['#2ecc71', '#e74c3c'])
    ])

    fig.update_layout(
        title='Class Distribution',
        xaxis_title='Class',
        yaxis_title='Count',
        height=400
    )

    return fig


def plot_metrics_comparison(metrics):
    """Plot model metrics"""
    metrics_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })

    # Remove threshold
    metrics_df = metrics_df[metrics_df['Metric'] != 'threshold']

    fig = go.Figure(data=[
        go.Bar(x=metrics_df['Metric'],
              y=metrics_df['Value'],
              marker_color='#3498db')
    ])

    fig.update_layout(
        title='Model Performance Metrics',
        xaxis_title='Metric',
        yaxis_title='Score',
        height=400,
        yaxis=dict(range=[0, 1])
    )

    return fig


def plot_feature_importance(feature_importance, top_n=20):
    """Plot top N feature importance"""
    top_features = feature_importance.head(top_n)

    fig = go.Figure(data=[
        go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker_color='#9b59b6'
        )
    ])

    fig.update_layout(
        title=f'Top {top_n} Most Important Features',
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )

    return fig


def plot_fraud_over_time(train_data):
    """Plot fraud transactions over time"""
    if 'timestamp' not in train_data.columns or 'is_fraud' not in train_data.columns:
        return None

    train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])
    train_data['date'] = train_data['timestamp'].dt.date

    daily_fraud = train_data.groupby('date')['is_fraud'].agg(['sum', 'count', 'mean'])
    daily_fraud.columns = ['fraud_count', 'total_count', 'fraud_rate']

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Daily Fraud Count', 'Daily Fraud Rate'),
        vertical_spacing=0.15
    )

    fig.add_trace(
        go.Scatter(x=daily_fraud.index, y=daily_fraud['fraud_count'],
                  mode='lines', name='Fraud Count', line=dict(color='#e74c3c')),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=daily_fraud.index, y=daily_fraud['fraud_rate'],
                  mode='lines', name='Fraud Rate', line=dict(color='#f39c12')),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Rate", row=2, col=1)

    fig.update_layout(height=600, showlegend=False)

    return fig


def plot_amount_distribution(train_data):
    """Plot transaction amount distribution by fraud status"""
    if 'amount' not in train_data.columns or 'is_fraud' not in train_data.columns:
        return None

    fig = go.Figure()

    for fraud_status, color, name in [(0, '#2ecc71', 'Not Fraud'), (1, '#e74c3c', 'Fraud')]:
        data = train_data[train_data['is_fraud'] == fraud_status]['amount']
        fig.add_trace(go.Histogram(
            x=data,
            name=name,
            marker_color=color,
            opacity=0.7,
            nbinsx=50
        ))

    fig.update_layout(
        title='Transaction Amount Distribution',
        xaxis_title='Amount',
        yaxis_title='Frequency',
        barmode='overlay',
        height=400
    )

    return fig


def plot_prediction_distribution(predictions):
    """Plot distribution of prediction probabilities"""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=predictions['fraud_probability'],
        nbinsx=50,
        marker_color='#3498db'
    ))

    fig.update_layout(
        title='Distribution of Fraud Probabilities (Evaluation Set)',
        xaxis_title='Fraud Probability',
        yaxis_title='Count',
        height=400
    )

    return fig


def plot_risk_segments(predictions):
    """Plot risk segmentation"""
    # Create risk segments
    predictions['risk_segment'] = pd.cut(
        predictions['fraud_probability'],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
    )

    segment_counts = predictions['risk_segment'].value_counts().sort_index()

    colors = ['#2ecc71', '#95a5a6', '#f39c12', '#e67e22', '#e74c3c']

    fig = go.Figure(data=[
        go.Bar(x=segment_counts.index,
              y=segment_counts.values,
              marker_color=colors)
    ])

    fig.update_layout(
        title='Risk Segmentation of Evaluation Set',
        xaxis_title='Risk Level',
        yaxis_title='Number of Transactions',
        height=400
    )

    return fig


# Load data
training_results, predictions, train_data, feature_importance = load_data()

if training_results is None:
    st.stop()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Model Performance", "Feature Analysis", "Predictions", "Transaction Search"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Info")
st.sidebar.info(f"**Model**: {training_results['best_model']}")
st.sidebar.info(f"**Threshold**: {training_results['optimal_threshold']:.4f}")
st.sidebar.info(f"**Features**: {len(training_results['features'])}")

# Main content based on selected page
if page == "Overview":
    st.header("📊 Overview Dashboard")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Model Type",
            training_results['best_model'].upper()
        )

    with col2:
        st.metric(
            "ROC-AUC Score",
            f"{training_results['metrics']['roc_auc']:.4f}"
        )

    with col3:
        st.metric(
            "PR-AUC Score",
            f"{training_results['metrics']['pr_auc']:.4f}"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{training_results['metrics']['f1']:.4f}"
        )

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        if train_data is not None:
            fig = plot_class_distribution(train_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = plot_metrics_comparison(training_results['metrics'])
        st.plotly_chart(fig, use_container_width=True)

    # Temporal analysis
    st.markdown("---")
    st.subheader("Temporal Analysis")

    if train_data is not None:
        fig = plot_fraud_over_time(train_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

elif page == "Model Performance":
    st.header("📈 Model Performance Analysis")

    # Detailed metrics
    st.subheader("Detailed Metrics")

    col1, col2, col3 = st.columns(3)

    metrics = training_results['metrics']

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        st.metric("Precision", f"{metrics['precision']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Recall", f"{metrics['recall']:.4f}")
        st.metric("F1-Score", f"{metrics['f1']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        st.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Amount distribution
    st.subheader("Transaction Analysis")
    if train_data is not None:
        fig = plot_amount_distribution(train_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    # Business Impact
    st.markdown("---")
    st.subheader("Business Impact Estimation")

    if train_data is not None and 'is_fraud' in train_data.columns and 'amount' in train_data.columns:
        total_fraud_amount = train_data[train_data['is_fraud'] == 1]['amount'].sum()
        avg_fraud_amount = train_data[train_data['is_fraud'] == 1]['amount'].mean()
        fraud_count = train_data['is_fraud'].sum()

        recall = metrics['recall']
        precision = metrics['precision']

        detected_fraud = fraud_count * recall
        detected_amount = total_fraud_amount * recall
        false_positives = (detected_fraud / precision) - detected_fraud if precision > 0 else 0

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Fraud Detected", f"{detected_fraud:.0f} transactions")
            st.metric("Amount Saved", f"${detected_amount:,.2f}")

        with col2:
            st.metric("Avg Fraud Amount", f"${avg_fraud_amount:,.2f}")
            st.metric("Total Fraud Amount", f"${total_fraud_amount:,.2f}")

        with col3:
            st.metric("False Positives", f"{false_positives:.0f}")
            st.metric("Manual Review Cost", f"${false_positives * 10:,.2f}")

elif page == "Feature Analysis":
    st.header("🔍 Feature Importance Analysis")

    # Top features selector
    top_n = st.slider("Number of top features to display", 10, 50, 20)

    fig = plot_feature_importance(feature_importance, top_n)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Feature importance table
    st.subheader("Feature Importance Table")
    st.dataframe(
        feature_importance.head(top_n),
        use_container_width=True,
        hide_index=True
    )

elif page == "Predictions":
    st.header("🎯 Prediction Analysis")

    col1, col2 = st.columns(2)

    with col1:
        total_predictions = len(predictions)
        fraud_predictions = predictions['fraud_prediction'].sum()
        fraud_rate = fraud_predictions / total_predictions

        st.metric("Total Transactions", f"{total_predictions:,}")
        st.metric("Predicted Fraud", f"{fraud_predictions:,}")
        st.metric("Fraud Rate", f"{fraud_rate:.2%}")

    with col2:
        avg_fraud_prob = predictions['fraud_probability'].mean()
        max_fraud_prob = predictions['fraud_probability'].max()
        median_fraud_prob = predictions['fraud_probability'].median()

        st.metric("Avg Fraud Probability", f"{avg_fraud_prob:.4f}")
        st.metric("Max Fraud Probability", f"{max_fraud_prob:.4f}")
        st.metric("Median Fraud Probability", f"{median_fraud_prob:.4f}")

    st.markdown("---")

    # Probability distribution
    col1, col2 = st.columns(2)

    with col1:
        fig = plot_prediction_distribution(predictions)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = plot_risk_segments(predictions)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # High-risk transactions
    st.subheader("Highest Risk Transactions")

    high_risk = predictions.nlargest(20, 'fraud_probability')
    st.dataframe(
        high_risk,
        use_container_width=True,
        hide_index=True
    )

elif page == "Transaction Search":
    st.header("🔎 Transaction Search")

    st.markdown("Search for specific transactions by ID to view fraud prediction details.")

    # Search box
    transaction_id = st.text_input("Enter Transaction ID:")

    if transaction_id:
        try:
            tid = int(transaction_id)
            result = predictions[predictions['transaction_id'] == tid]

            if len(result) > 0:
                result = result.iloc[0]

                st.markdown("### Transaction Details")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Transaction ID", result['transaction_id'])

                with col2:
                    st.metric("Fraud Probability", f"{result['fraud_probability']:.4f}")

                with col3:
                    prediction_text = "FRAUD" if result['fraud_prediction'] == 1 else "NOT FRAUD"
                    prediction_color = "🔴" if result['fraud_prediction'] == 1 else "🟢"
                    st.metric("Prediction", f"{prediction_color} {prediction_text}")

                # Risk level
                prob = result['fraud_probability']
                if prob >= 0.8:
                    risk_level = "🔴 VERY HIGH RISK"
                    st.markdown(f'<div class="fraud-alert"><h3>{risk_level}</h3></div>', unsafe_allow_html=True)
                elif prob >= 0.6:
                    st.warning("🟠 HIGH RISK")
                elif prob >= 0.4:
                    st.info("🟡 MEDIUM RISK")
                elif prob >= 0.2:
                    st.info("🔵 LOW RISK")
                else:
                    st.success("🟢 VERY LOW RISK")

            else:
                st.error(f"Transaction ID {tid} not found in evaluation set.")

        except ValueError:
            st.error("Please enter a valid numeric transaction ID.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #7f8c8d;'>
        <p>Fraud Detection Dashboard | Hackathon Finance Track 2024</p>
        <p>Powered by Machine Learning & Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
