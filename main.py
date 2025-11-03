import streamlit as st
import joblib
import pandas as pd
import datetime
import xgboost
from xgboost import XGBRegressor
from used_Mehods import Date_Calculation, re_use_resources, add_value, last_value_added
from sklearn.preprocessing import PolynomialFeatures

st.set_page_config(
    page_title="AI Revenue Dashboard 📊",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
:root {
  --bg-color: white;
  --text-color: #222;
  --subtext-color: #555;
  --card-bg: #f7f7f9;
  --accent: #007bff;
  --shadow: rgba(0,0,0,0.1);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg-color: #0e1117;
    --text-color: #eaeaea;
    --subtext-color: #a1a1a1;
    --card-bg: #1a1d23;
    --accent: #1f77ff;
    --shadow: rgba(255,255,255,0.05);
  }
}
body {
  background-color: var(--bg-color);
  color: var(--text-color);
  font-family: 'Inter', sans-serif;
}
h1, h2, h3 {
  color: var(--accent);
  font-weight: 700;
  text-align: center;
}
header {
  background: var(--card-bg);
  padding: 1rem 2rem;
  border-radius: 12px;
  box-shadow: 0 4px 12px var(--shadow);
  margin-bottom: 2rem;
}
.metric-card {
  background-color: var(--card-bg);
  padding: 1.2rem;
  border-radius: 15px;
  box-shadow: 0 2px 10px var(--shadow);
  text-align: center;
}
.metric-card h3 {
  color: var(--accent);
  margin-bottom: 0.4rem;
}
.metric-card p {
  font-size: 1.4rem;
  font-weight: bold;
  color: var(--text-color);
}
div.stButton > button:first-child {
  background-color: var(--accent);
  color: white;
  font-weight: bold;
  border-radius: 10px;
  transition: 0.3s;
}
div.stButton > button:first-child:hover {
  transform: scale(1.03);
  background-color: #0056c4;
}
.stSuccess {
  background-color: var(--card-bg) !important;
  color: var(--text-color) !important;
  border-left: 5px solid var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<header>
  <h1>📈 AI Revenue Predictor Dashboard</h1>
  <p style='text-align:center; color:var(--subtext-color); font-size:16px;'>
            <br> Produced <b>for HandyHome Company</b>
  </p>
</header>
""", unsafe_allow_html=True)

model = joblib.load('model_project.pkl')
scaler = joblib.load('scaler_project.pkl')
feature_names = joblib.load('features_project.pkl') 
try:
    poly_transformer = joblib.load('poly_transformer.pkl')
except FileNotFoundError:
    st.error("🚨 Missing file 'poly_transformer.pkl'. Please run the updated 'source_code.py' first!")
    st.stop()


st.subheader("⚙️ Input Business Parameters")

col1, col2 = st.columns(2)
with col1:
    date = st.date_input('📅 Date', value=datetime.date.today())
    time_of_day = st.selectbox('🕰️ Time of Day', ['Morning', 'Afternoon', 'Evening', 'Night'])
    category = st.selectbox('🛍️ Category', ['Service', 'Subscription', 'Product'])
    platform = st.selectbox('💻 Platform', ['Instagram', 'In-store', 'Email', 'Google'])
with col2:
    service_type = st.selectbox('✂️ Service Type', ['Coffee', 'Dress', 'Haircut', 'Plumbing'])
    customer_type = st.selectbox('👤 Customer Type', ['New', 'Returning'])
    Ad_Spend = st.number_input('💸 Ad Spend ($)', min_value=0.0, step=1.0)
    Conversions = st.number_input('📈 Conversions (0–5)', min_value=0, max_value=5, step=1)

encode = {
    'time_of_day': {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3},
    'platform': {'Instagram': 0, 'In-store': 1, 'Email': 2, 'Google': 3},
    'service_type': {'Coffee': 0, 'Dress': 1, 'Haircut': 2, 'Plumbing': 3},
    'category': {'Service': 0, 'Subscription': 1, 'Product': 2},
    'customer_type': {'New': 0, 'Returning': 1}
}

Last_treatment_Period_In_Years = Date_Calculation(date)
Date_Of_Day = date.day
Month_Number = date.month
Year_Number = date.year
DayOfWeek_number = date.weekday()


try:
    mean_revenue = re_use_resources().mean()
    Ad_to_Revenue_Ratio = Ad_Spend / (mean_revenue + 1)
except Exception:
    Ad_to_Revenue_Ratio = Ad_Spend

try:
    Revenue_Change_Feature = re_use_resources().diff().iloc[-1]
except Exception:
    Revenue_Change_Feature = 0
   
try:
    LAST_KNOWN_REVENUE = last_value_added()
except Exception:
    LAST_KNOWN_REVENUE = Ad_Spend
   
   

if st.button("🔥 Predict Daily Revenue", type='primary' , use_container_width = True):
    raw_input_data = pd.DataFrame({
        'Time of Day': [encode['time_of_day'][time_of_day]],
        'Category': [encode['category'][category]],
        'Platform': [encode['platform'][platform]],
        'Service Type': [encode['service_type'][service_type]],
        'Customer Type': [encode['customer_type'][customer_type]],
        'Ad Spend': [Ad_Spend],
        'Conversions': [Conversions],
        'Last_treatment_Period_In_Years': [Last_treatment_Period_In_Years],
        'Date_Of_Day': [Date_Of_Day],
        'Month_Number': [Month_Number],
        'Year_Number': [Year_Number],
        'DayOfWeek_number': [DayOfWeek_number],
        'Revenue Change': [Revenue_Change_Feature], # استخدام القيمة المحسوبة للميزة
        'Ad_to_Revenue_Ratio': [Ad_to_Revenue_Ratio]
    })


    # --- Step 2: Align and Transform ---
    try:
        raw_input_data = raw_input_data[feature_names]
        input_poly = poly_transformer.transform(raw_input_data)
    except Exception as e:
        st.error(f"❌ Error during feature preparation: {str(e)}")
        st.stop()

    try:
        input_scaled = scaler.transform(input_poly)
        prediction = model.predict(input_scaled)
        pred_value = round(float(prediction[0]), 2)
       

        change_value = pred_value - LAST_KNOWN_REVENUE
       
         
        if change_value >= 0:
             change_symbol = "▲" # Gain - as the the daily revenue of that day is greater than the daily revenue of day before
             change_color = "text-green-500"
             color = 'green'
        else:
             change_symbol = "▼" # Loss - as the the daily revenue of that day is less than the daily revenue of day before
             change_color = "text-red-500"
             color = 'red'


        st.success(f"📈 **Predicted Daily Revenue:** `{pred_value} $`", icon="✅")

        st.markdown("### 📊 Key Performance Indicators")
        c1, c2, c3 = st.columns(3)
       
        c1.markdown(f"<div class='metric-card'><h3>Revenue</h3><p>{pred_value} $</p></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><h3>Ad Ratio</h3><p>{Ad_to_Revenue_Ratio:.3f}</p></div>", unsafe_allow_html=True)
       
        c3.markdown(f"""
            <div class='metric-card'>
                <h3>Change Δ</h3>
                <p class='{change_color}' style = 'color:{color}'>{change_symbol} {abs(change_value):.3f} $</p>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        st.info("Make sure all necessary files are correctly generated and the `used_Mehods.py` functions work.")

st.divider()
st.caption("CopyRights © Reserved for HandyHome.")