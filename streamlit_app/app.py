# ============================================
# CHURNRADAR - STREAMLIT APP
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="ChurnRadarAPP - Customer Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Load Model & Scaler
# ============================================

@st.cache_resource
def load_model():
    """Load trained model dan scaler"""
    model = joblib.load('../models/churn_model.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# ============================================
# Helper Function: Prepare Input Data
# ============================================

def prepare_input(gender, senior_citizen, partner, dependents, tenure, 
                  phone_service, multiple_lines, internet_service,
                  online_security, online_backup, device_protection,
                  tech_support, streaming_tv, streaming_movies,
                  contract, paperless_billing, payment_method,
                  monthly_charges, total_charges):
    """
    Prepare input data dengan exact features dari training
    """
    
    # Calculate engineered features
    avg_charge_per_month = total_charges / (tenure + 1)
    
    # Count number of services
    num_services = 0
    if phone_service == "Yes":
        num_services += 1
    if internet_service != "No":
        num_services += 1
    if online_security == "Yes":
        num_services += 1
    if online_backup == "Yes":
        num_services += 1
    if device_protection == "Yes":
        num_services += 1
    if tech_support == "Yes":
        num_services += 1
    if streaming_tv == "Yes":
        num_services += 1
    if streaming_movies == "Yes":
        num_services += 1
    
    # Tenure group
    if tenure <= 12:
        tenure_group = '0-1yr'
    elif tenure <= 24:
        tenure_group = '1-2yr'
    elif tenure <= 48:
        tenure_group = '2-4yr'
    else:
        tenure_group = '4+yr'
    
    # Create base data dictionary (semua features set ke 0 dulu)
    data = {
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': 1 if partner == "Yes" else 0,
        'Dependents': 1 if dependents == "Yes" else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == "Yes" else 0,
        'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'avg_charge_per_month': avg_charge_per_month,
        'num_services': num_services,
        
        # One-hot encoded features (default = 0)
        'gender_Male': 1 if gender == "Male" else 0,
        'MultipleLines_No phone service': 1 if multiple_lines == "No phone service" else 0,
        'MultipleLines_Yes': 1 if multiple_lines == "Yes" else 0,
        'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
        'InternetService_No': 1 if internet_service == "No" else 0,
        'OnlineSecurity_No internet service': 1 if online_security == "No internet service" else 0,
        'OnlineSecurity_Yes': 1 if online_security == "Yes" else 0,
        'OnlineBackup_No internet service': 1 if online_backup == "No internet service" else 0,
        'OnlineBackup_Yes': 1 if online_backup == "Yes" else 0,
        'DeviceProtection_No internet service': 1 if device_protection == "No internet service" else 0,
        'DeviceProtection_Yes': 1 if device_protection == "Yes" else 0,
        'TechSupport_No internet service': 1 if tech_support == "No internet service" else 0,
        'TechSupport_Yes': 1 if tech_support == "Yes" else 0,
        'StreamingTV_No internet service': 1 if streaming_tv == "No internet service" else 0,
        'StreamingTV_Yes': 1 if streaming_tv == "Yes" else 0,
        'StreamingMovies_No internet service': 1 if streaming_movies == "No internet service" else 0,
        'StreamingMovies_Yes': 1 if streaming_movies == "Yes" else 0,
        'Contract_One year': 1 if contract == "One year" else 0,
        'Contract_Two year': 1 if contract == "Two year" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
        'tenure_group_1-2yr': 1 if tenure_group == '1-2yr' else 0,
        'tenure_group_2-4yr': 1 if tenure_group == '2-4yr' else 0,
        'tenure_group_4+yr': 1 if tenure_group == '4+yr' else 0,
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Apply scaling to numerical columns (same as training)
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'avg_charge_per_month', 'num_services']
    df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df

# ============================================
# Title & Description
# ============================================

st.title("📡 ChurnRadarAPP: Customer Churn Prediction")
st.caption("by M1neeS")
st.markdown("Predict customer churn risk menggunakan AI dan dapatkan rekomendasi retention strategy.")
st.markdown("---")

# ============================================
# Sidebar - Input Form
# ============================================

st.sidebar.header("📊 Customer Information")

# Demographics
st.sidebar.subheader("Demographics")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])

# Account
st.sidebar.subheader("Account Info")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])
payment_method = st.sidebar.selectbox("Payment Method", 
                                     ["Electronic check", "Mailed check", 
                                      "Bank transfer (automatic)", 
                                      "Credit card (automatic)"])

# Services
st.sidebar.subheader("Services")
phone_service = st.sidebar.selectbox("Phone Service", ["No", "Yes"])

if phone_service == "Yes":
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No", "Yes"])
else:
    multiple_lines = "No phone service"

internet_service = st.sidebar.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])

if internet_service != "No":
    online_security = st.sidebar.selectbox("Online Security", ["No", "Yes"])
    online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes"])
    device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes"])
    tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes"])
else:
    online_security = "No internet service"
    online_backup = "No internet service"
    device_protection = "No internet service"
    tech_support = "No internet service"
    streaming_tv = "No internet service"
    streaming_movies = "No internet service"

# Financial
st.sidebar.subheader("Financial")
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0, 1.0)
total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 500.0, 10.0)

# Predict Button
predict_button = st.sidebar.button("🔮 Predict Churn Risk", type="primary", use_container_width=True)

# ============================================
# Main Content
# ============================================

if not model_loaded:
    st.error("⚠️ Model tidak dapat di-load")
    st.stop()

if predict_button:
    
    try:
        # Prepare input
        input_df = prepare_input(
            gender, senior_citizen, partner, dependents, tenure,
            phone_service, multiple_lines, internet_service,
            online_security, online_backup, device_protection,
            tech_support, streaming_tv, streaming_movies,
            contract, paperless_billing, payment_method,
            monthly_charges, total_charges
        )
        
        # Predict
        churn_proba = model.predict_proba(input_df)[0][1] * 100
        
        # ============================================
        # Display Results
        # ============================================
        
        st.markdown("---")
        st.header("🎯 Prediction Results")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gauge Chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_proba,
                title={'text': "Churn Risk Score (%)", 'font': {'size': 24}},
                number={'suffix': "%", 'font': {'size': 48}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "darkred" if churn_proba > 70 else "orange" if churn_proba > 40 else "green"},
                    'steps': [
                        {'range': [0, 40], 'color': '#D5F4E6'},
                        {'range': [40, 70], 'color': '#FCF3CF'},
                        {'range': [70, 100], 'color': '#F5B7B1'}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'value': 70}
                }
            ))
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with col2:
            if churn_proba >= 70:
                st.error("### 🔴 HIGH RISK")
                risk_text = "URGENT ACTION"
            elif churn_proba >= 40:
                st.warning("### 🟡 MEDIUM RISK")
                risk_text = "PROACTIVE ENGAGEMENT"
            else:
                st.success("### 🟢 LOW RISK")
                risk_text = "MAINTAIN SATISFACTION"
            
            st.markdown(f"### {risk_text}")
            st.metric("Churn Probability", f"{churn_proba:.1f}%")
        
        # ============================================
        # Recommendations
        # ============================================
        
        st.markdown("---")
        st.header("💡 Retention Recommendations")
        
        if churn_proba >= 70:
            st.error("### ⚠️ URGENT ACTION REQUIRED")
            st.markdown("""
            **Immediate Actions (24-48 hours):**
            - 📞 **Personal Call**: Contact customer immediately
            - 💰 **Special Offer**: 20-30% discount for 6-month commitment
            - 🎁 **Free Upgrade**: Premium services for 3 months
            - 🤝 **Account Manager**: Assign dedicated support
            """)
            
        elif churn_proba >= 40:
            st.warning("### ⚠️ PROACTIVE ENGAGEMENT NEEDED")
            st.markdown("""
            **Recommended Actions (1-2 weeks):**
            - 📧 **Email Campaign**: Send retention offer
            - 🎯 **Promotion**: 10-15% discount on annual upgrade
            - 📊 **Usage Review**: Optimize service bundle
            - ⭐ **Loyalty Program**: VIP benefits
            """)
            
        else:
            st.success("### ✅ MAINTAIN SATISFACTION")
            st.markdown("""
            **Maintenance Actions:**
            - 😊 **Regular Check-ins**: Quarterly survey
            - 🎉 **Appreciation**: Thank you message
            - 📈 **Upsell**: Introduce new features
            - 🌟 **Referral Program**: Encourage word-of-mouth
            """)
        
        # ============================================
        # Key Risk Factors
        # ============================================
        
        st.markdown("---")
        st.header("🔍 Key Risk Factors")
        
        risk_factors = []
        
        if contract == "Month-to-month":
            risk_factors.append(("Contract: Month-to-month", "🔴", "3x higher churn rate"))
        
        if tenure < 12:
            risk_factors.append(("Low Tenure", "🔴", f"Only {tenure} months - critical period"))
        
        if payment_method == "Electronic check":
            risk_factors.append(("Payment: E-check", "🟡", "Associated with higher churn"))
        
        if monthly_charges > 70:
            risk_factors.append(("High Charges", "🟡", f"${monthly_charges:.0f}/month - above average"))
        
        if internet_service == "Fiber optic":
            risk_factors.append(("Internet: Fiber", "🔵", "Higher churn in fiber customers"))
        
        if risk_factors:
            for factor, emoji, description in risk_factors:
                st.markdown(f"{emoji} **{factor}**: {description}")
        else:
            st.success("✅ No significant risk factors detected")
        
        # ============================================
        # Action Summary
        # ============================================
        
        st.markdown("---")
        st.header("📋 Action Summary")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            priority = "🔴 HIGH" if churn_proba >= 70 else "🟡 MEDIUM" if churn_proba >= 40 else "🟢 LOW"
            st.metric("Priority", priority)
        
        with col2:
            timeline = "24-48 hours" if churn_proba >= 70 else "1-2 weeks" if churn_proba >= 40 else "Quarterly"
            st.metric("Timeline", timeline)
        
        with col3:
            retention_rate = "30%" if churn_proba >= 70 else "50%" if churn_proba >= 40 else "80%"
            st.metric("Expected Retention", retention_rate)
        
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
        st.code(str(e))

else:
    # ============================================
    # Landing Page
    # ============================================
    
    st.info("👈 Input customer information di sidebar, kemudian klik **Predict Churn Risk**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Accuracy", "75.7%")
    
    with col2:
        st.metric("ROC-AUC Score", "0.838")
    
    with col3:
        st.metric("Processing Time", "<1s")
    
    st.markdown("---")
    st.header("📊 How It Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Machine Learning Model
        - **Algorithm**: Random Forest
        - **Training Data**: 7,000+ customers
        - **Features**: 35 customer attributes
        - **Validation**: 80-20 split
        """)
    
    with col2:
        st.markdown("""
        ### Key Predictors
        - Contract type
        - Tenure duration
        - Payment method
        - Service usage
        - Monthly charges
        """)
    
    st.markdown("---")
    st.header("🎯 Business Impact")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("At Risk", "288")
    
    with col2:
        st.metric("Revenue Risk", "$288K")
    
    with col3:
        st.metric("Retainable", "86")
    
    with col4:
        st.metric("ROI", "95.8%")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ by M1neeS | ChurnRadarAPP v1.0</p>
</div>
""", unsafe_allow_html=True)