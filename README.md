# 📦 SKU Demand Forecasting & Safety Stock Optimizer

## 🚩 The Business Problem: The "Blind Inventory" Crisis
In large-scale retail and logistics, inventory management is a high-stakes balancing act. Most businesses struggle with two costly extremes:
1.  **Stockouts:** Failing to meet customer demand, leading to immediate revenue loss and long-term brand damage.
2.  **Overstocking:** Tying up millions in working capital and risking product expiration or warehouse congestion.

### **The Main Pain Point**
**Lack of Daily Granularity.** Most traditional systems predict monthly trends but fail at the **daily SKU-level**. Without accurate daily forecasts, Supply Chain Managers rely on "gut feeling" to set safety stock, leading to massive financial waste in supply chains handling nearly a million transactions.

---

## 🌟 Project Summary 

**Situation:** Managing a portfolio of **50 products (SKUs)** across **10 distinct stores** with highly volatile, seasonal daily demand. The dataset comprised **913,000+ historical transaction records** over a 5-year period.

**Task:** Build a scalable, global forecasting engine capable of predicting daily sales for an entire year using only historical data, ensuring the model could handle "cold-start" patterns and zero-sales days without losing temporal integrity.

**Action:** * **Data Engineering:** Implemented a **Continuous Timeline Fix** to explicitly fill missing sale days with 0, ensuring $Lag$ and $Rolling$ features remained mathematically accurate.
* **Feature Architecture:** Developed a "Momentum-Trend" feature set including **7-day rolling means** and **weekly lags** ($Lag_1, Lag_7$) to capture both immediate shifts and cyclical patterns.
* **Model Optimization:** Leveraged **Native Categorical XGBoost** to handle Store and Item IDs without the memory overhead of One-Hot Encoding.
* **Bayesian Tuning:** Conducted a 20-trial **Optuna Study** to minimize the business-critical WAPE metric by finding the optimal balance between `learning_rate` and `tree_depth`.

**Result:** Achieved a final **Weighted Absolute Percentage Error (WAPE) of 10.44%** on an unseen **Out-of-Time (OOT) validation set**. This represents a 90%+ accuracy rate, providing a robust foundation for automated safety stock replenishment.

---

## 🛠️ Technical Implementation

### **1. Prevention of Data Leakage**
Unlike standard machine learning projects, I utilized a **Chronological Time-Series Split** to simulate real-world production performance:
* **Train:** 2013 – 2016
* **Test:** 2017 (Simulating the "Future")

### **2. The "Quant" Optimization (Optuna)**
The model was engineered through Bayesian optimization to settle on the following hyper-parameters:
* **Learning Rate:** $0.164$
* **Max Depth:** $3$ (Preventing over-fitting on noisy daily data)
* **Subsample:** $0.828$

### **3. Evaluation Metric: Why WAPE?**
I chose **WAPE** over standard RMSE or MAPE because it handles zero-sale days gracefully and weights errors by the actual volume of sales. 

$$WAPE = \frac{\sum |Actual - Forecast|}{\sum Actual}$$

---

## 📊 Dashboard Preview

![alt text](<Screenshot 2026-03-31 213857-1.png>)


## 🚀 Deployment Strategy
The engine is containerized and exported as a `.json` model, ready to be served via **FastAPI**. The accompanying **Streamlit MVP** translates raw numeric forecasts into a "Mitigated" or "High" stockout risk status, allowing non-technical managers to make instant ordering decisions.

---

### **How to Run**
1. **Clone the Repo:** `git clone https://github.com/BhardwajG572/sku-forecaster`
2. **Install Dependencies:** `pip install -r requirements.txt`
3. **Launch Dashboard:** `streamlit run app.py`