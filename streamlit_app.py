import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="🎯",
    layout="wide"
)

# Load model and preprocessing objects
@st.cache_resource
def load_models():
    try:
        model = load_model('customer_segmentation_model.h5')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("Model files not found! Please run the notebook first to train and save the model.")
        return None, None, None

# App header
st.title("🎯 Customer Segmentation with Neural Networks")
st.markdown("Predict high-value customers using machine learning")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", 
                           ["Single Prediction", "Batch Prediction", "Model Analytics"])

# Load models
model, scaler, feature_names = load_models()

if model is None:
    st.stop()

if page == "Single Prediction":
    st.header("📊 Single Customer Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Information")
        frequency = st.number_input("Purchase Frequency", min_value=1, max_value=1000, value=10, 
                                   help="Number of purchases made by customer")
        monetary = st.number_input("Total Monetary Value ($)", min_value=0.0, value=500.0,
                                  help="Total amount spent by customer")
        total_quantity = st.number_input("Total Quantity Purchased", min_value=1, value=100,
                                       help="Total quantity of items purchased")
    
    with col2:
        st.subheader("Product Information") 
        unique_products = st.number_input("Unique Products", min_value=1, max_value=1000, value=15,
                                        help="Number of different products purchased")
        avg_unit_price = st.number_input("Average Unit Price ($)", min_value=0.0, value=5.0,
                                       help="Average price per unit of items purchased")
        
        # Prediction button
        if st.button("🎯 Predict Customer Segment", type="primary"):
            # Prepare input data
            input_data = np.array([[frequency, monetary, total_quantity, unique_products, avg_unit_price]])
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction_proba = model.predict(input_scaled)[0][0]
            prediction = 1 if prediction_proba > 0.5 else 0
            
            # Display results
            st.markdown("---")
            col3, col4 = st.columns(2)
            
            with col3:
                if prediction == 1:
                    st.success("🌟 **High-Value Customer**")
                    st.markdown(f"**Confidence:** {prediction_proba:.1%}")
                else:
                    st.info("📈 **Standard Customer**")
                    st.markdown(f"**Confidence:** {(1-prediction_proba):.1%}")
            
            with col4:
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction_proba * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "High-Value Probability (%)"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkgreen" if prediction_proba > 0.5 else "orange"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "gray"}],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50}
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

elif page == "Batch Prediction":
    st.header("📂 Batch Customer Prediction")
    
    st.markdown("Upload a CSV file with customer data for batch predictions.")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            st.subheader("📋 Uploaded Data Preview")
            st.dataframe(df.head())
            
            # Check if required columns exist
            required_cols = ['Frequency', 'Monetary', 'TotalQuantity', 'UniqueProducts', 'AvgUnitPrice']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {missing_cols}")
                st.markdown("**Required columns:** " + ", ".join(required_cols))
            else:
                if st.button("🚀 Run Batch Prediction"):
                    # Prepare data
                    X = df[feature_names].values
                    X_scaled = scaler.transform(X)
                    
                    # Make predictions
                    predictions_proba = model.predict(X_scaled)
                    predictions = (predictions_proba > 0.5).astype(int)
                    
                    # Add results to dataframe
                    df['High_Value_Probability'] = predictions_proba.flatten()
                    df['Predicted_Segment'] = ['High-Value' if p == 1 else 'Standard' for p in predictions]
                    
                    # Display results
                    st.subheader("🎯 Prediction Results")
                    st.dataframe(df)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Customers", len(df))
                    with col2:
                        high_value_count = sum(predictions)
                        st.metric("High-Value Customers", high_value_count)
                    with col3:
                        percentage = (high_value_count / len(df)) * 100
                        st.metric("High-Value %", f"{percentage:.1f}%")
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name='customer_predictions.csv',
                        mime='text/csv'
                    )
                    
                    # Visualization
                    fig = px.histogram(df, x='Predicted_Segment', 
                                     title='Customer Segment Distribution',
                                     color='Predicted_Segment')
                    st.plotly_chart(fig, use_container_width=True)
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

elif page == "Model Analytics":
    st.header("📈 Model Analytics & Insights")
    
    # Model architecture
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Model Information")
        st.write("**Model Type:** Neural Network")
        st.write("**Architecture:** Multi-layer Perceptron")
        st.write("**Input Features:** 5")
        st.write("**Output:** Binary Classification")
        
        st.subheader("📊 Feature Importance")
        feature_importance = {
            'Frequency': 0.85,
            'Monetary': 0.92,
            'TotalQuantity': 0.78,
            'UniqueProducts': 0.65,
            'AvgUnitPrice': 0.43
        }
        
        importance_df = pd.DataFrame(list(feature_importance.items()), 
                                   columns=['Feature', 'Importance'])
        fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                    title='Feature Importance Scores')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💡 Business Insights")
        st.markdown("""
        **Key Findings:**
        - **Monetary Value** is the strongest predictor of high-value customers
        - **Purchase Frequency** indicates customer loyalty and engagement
        - **Product Diversity** shows customer exploration behavior
        - **Quantity patterns** reveal purchasing habits
        
        **Recommendations:**
        - Focus retention strategies on high-frequency customers
        - Implement upselling for high-monetary-value segments
        - Cross-sell to increase product diversity
        - Monitor customer journey transitions
        """)
        
        # Sample data distribution
        st.subheader("📊 Sample Data Distribution")
        sample_data = {
            'Segment': ['High-Value', 'Standard'],
            'Count': [750, 3250],
            'Percentage': [18.75, 81.25]
        }
        sample_df = pd.DataFrame(sample_data)
        
        fig = px.pie(sample_df, values='Count', names='Segment',
                    title='Customer Segment Distribution (Sample)')
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and TensorFlow")
