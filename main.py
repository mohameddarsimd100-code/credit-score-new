import pandas as pd
import numpy as np
import random
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. INTELLIGENT DATA GENERATION
# ==========================================
def generate_logical_data(n=2000):
    data = []
    for _ in range(n):
        age = random.randint(18, 70)
        gender = random.choice(["Male", "Female"])
        marital = random.choice(["Single", "Married"])
        children = random.randint(0, 3)
        
        edu_options = ["High School Diploma", "Associate's Degree", "Bachelor's Degree", "Master's Degree", "Doctorate"]
        edu = random.choice(edu_options)
        
        if edu == "High School Diploma": income = random.randint(25000, 55000)
        elif edu == "Associate's Degree": income = random.randint(40000, 70000)
        elif edu == "Bachelor's Degree": income = random.randint(55000, 95000)
        elif edu == "Master's Degree": income = random.randint(80000, 140000)
        else: income = random.randint(100000, 200000)
        income += random.randint(-5000, 15000)

        if income > 85000: home = random.choices(["Owned", "Rented"], weights=[80, 20])[0]
        else: home = random.choices(["Owned", "Rented"], weights=[30, 70])[0]

        # Scoring Logic (The "Why")
        points = 0
        if income > 90000: points += 50
        elif income > 60000: points += 30
        else: points += 10
        
        if home == "Owned": points += 20
        if age > 35: points += 10
        if edu in ["Master's Degree", "Doctorate"]: points += 10
        if marital == "Married": points += 5

        if points >= 75: score = "High"
        elif points >= 45: score = "Average"
        else: score = "Low"

        data.append([age, gender, income, edu, marital, children, home, score])

    columns = ['Age', 'Gender', 'Income', 'Education', 'Marital Status', 'Number of Children', 'Home Ownership', 'Credit Score']
    return pd.DataFrame(data, columns=columns)

# Generate Data
df = generate_logical_data(2000)

# --- PREPROCESSING ---
edu_mapping = {
    "High School Diploma": 0, "Associate's Degree": 1, "Bachelor's Degree": 2, 
    "Master's Degree": 3, "Doctorate": 4
}
df['Education_Enc'] = df['Education'].map(edu_mapping)

le_gender = LabelEncoder()
df['Gender_Enc'] = le_gender.fit_transform(df['Gender'])

le_marital = LabelEncoder()
df['Marital_Enc'] = le_marital.fit_transform(df['Marital Status'])

le_home = LabelEncoder()
df['Home_Enc'] = le_home.fit_transform(df['Home Ownership']) # Owned=0, Rented=1 usually (alpha sort)
# Check mapping to be sure
home_map = dict(zip(le_home.classes_, le_home.transform(le_home.classes_)))

le_target = LabelEncoder()
df['Target'] = le_target.fit_transform(df['Credit Score'])

X = df[['Age', 'Gender_Enc', 'Income', 'Education_Enc', 'Marital_Enc', 'Number of Children', 'Home_Enc']]
y = df['Target']

model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X, y)

print("✅ Model Trained & Explanation Logic Ready.")

# ==========================================
# 2. HTML INTERFACE (With Modal)
# ==========================================
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Credit Score AI</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Poppins:wght@500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-color: #f3f4f6;
            --text-dark: #1f2937;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }
        body { 
            background: linear-gradient(135deg, #f3f4f6 0%, #dbeafe 100%); 
            min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 20px; 
        }
        .card { 
            background: #ffffff; width: 100%; max-width: 500px; border-radius: 16px; padding: 40px; 
            box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.1); border: 1px solid rgba(255,255,255,0.8);
        }
        h1 { font-family: 'Poppins', sans-serif; font-size: 28px; color: var(--text-dark); text-align: center; }
        p.subtext { color: #6b7280; font-size: 14px; text-align: center; margin-bottom: 25px; }

        .grid-row { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .input-group { margin-bottom: 15px; position: relative; }
        label { display: block; font-size: 13px; font-weight: 600; margin-bottom: 6px; color: #374151; }
        
        input, select { 
            width: 100%; padding: 12px 15px 12px 40px; border-radius: 8px; border: 1px solid #e5e7eb; 
            background: #f9fafb; outline: none; transition: 0.2s; font-size: 14px;
        }
        .input-group i { position: absolute; left: 14px; top: 38px; color: #9ca3af; font-size: 14px; }
        input:focus, select:focus { border-color: var(--primary); background: #fff; box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1); }
        
        button { 
            width: 100%; padding: 14px; background: var(--primary); color: white; border: none; border-radius: 10px; 
            font-size: 16px; font-weight: 600; cursor: pointer; transition: 0.2s; margin-top: 10px;
        }
        button:hover { background: var(--primary-hover); transform: translateY(-1px); }

        /* Result Section */
        .result-container { margin-top: 20px; display: none; }
        .status-box { padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 10px; }
        .status-high { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
        .status-avg { background: #fef9c3; color: #854d0e; border: 1px solid #fde047; }
        .status-low { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
        
        .details-btn {
            background: #fff; color: var(--text-dark); border: 1px solid #d1d5db; margin-top: 10px;
        }
        .details-btn:hover { background: #f3f4f6; color: #000; }

        /* MODAL STYLING */
        .modal-overlay {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.5); backdrop-filter: blur(4px);
            display: none; justify-content: center; align-items: center; z-index: 1000;
            opacity: 0; transition: opacity 0.3s ease;
        }
        .modal-content {
            background: white; width: 90%; max-width: 450px; padding: 30px;
            border-radius: 20px; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
            transform: scale(0.9); transition: transform 0.3s ease;
            position: relative;
        }
        .modal-active { display: flex; opacity: 1; }
        .modal-active .modal-content { transform: scale(1); }
        
        .close-btn { position: absolute; top: 20px; right: 20px; cursor: pointer; font-size: 20px; color: #9ca3af; }
        
        .analysis-section { margin-top: 20px; }
        .analysis-item { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 12px; font-size: 14px; color: #4b5563; }
        .icon-good { color: #10b981; }
        .icon-bad { color: #ef4444; }
        .icon-neutral { color: #f59e0b; }

    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <h1>Financial AI</h1>
            <p class="subtext">Advanced Credit Scoring & Analysis</p>
        </div>
        
        <form id="predictionForm">
            <div class="grid-row">
                <div class="input-group">
                    <label>Age</label>
                    <i class="fa-solid fa-user"></i>
                    <input type="number" id="age" min="18" required placeholder="e.g. 30">
                </div>
                <div class="input-group">
                    <label>Gender</label>
                    <i class="fa-solid fa-venus-mars"></i>
                    <select id="gender"><option>Female</option><option>Male</option></select>
                </div>
            </div>

            <div class="input-group">
                <label>Annual Income ($)</label>
                <i class="fa-solid fa-dollar-sign"></i>
                <input type="number" id="income" min="0" required placeholder="e.g. 60000">
            </div>

            <div class="grid-row">
                <div class="input-group">
                    <label>Education</label>
                    <i class="fa-solid fa-graduation-cap"></i>
                    <select id="education">
                        <option>High School Diploma</option>
                        <option>Associate's Degree</option>
                        <option>Bachelor's Degree</option>
                        <option>Master's Degree</option>
                        <option>Doctorate</option>
                    </select>
                </div>
                <div class="input-group">
                    <label>Home Type</label>
                    <i class="fa-solid fa-house"></i>
                    <select id="home_ownership"><option>Rented</option><option>Owned</option></select>
                </div>
            </div>
            
            <!-- Hidden/Default fields for model structure -->
            <input type="hidden" id="marital_status" value="Single">
            <input type="hidden" id="children" value="0">

            <button type="submit" id="predictBtn">Analyze Profile</button>
        </form>

        <div id="result" class="result-container">
            <div id="statusBox" class="status-box">
                <h2 id="scoreText" style="margin:0; font-size: 22px;">--</h2>
                <span id="scoreSub" style="font-size: 13px;">Creditworthiness Score</span>
            </div>
            <button class="details-btn" onclick="openModal()">
                <i class="fa-solid fa-chart-pie"></i> View Detailed Analysis
            </button>
        </div>
    </div>

    <!-- MODAL POPUP -->
    <div id="modalOverlay" class="modal-overlay">
        <div class="modal-content">
            <i class="fa-solid fa-xmark close-btn" onclick="closeModal()"></i>
            <h2 style="font-family:'Poppins',sans-serif; color:#1f2937;">AI Analysis Report</h2>
            <p style="font-size:13px; color:#6b7280;">Why did you get this score?</p>
            
            <div id="analysisContent" class="analysis-section">
                <!-- Content injected via JS -->
            </div>
            
            <button onclick="closeModal()" style="margin-top:20px; background:#f3f4f6; color:#374151;">Close Report</button>
        </div>
    </div>

    <script>
        let currentAnalysis = {};

        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const btn = document.getElementById('predictBtn');
            const resultDiv = document.getElementById('result');
            const statusBox = document.getElementById('statusBox');
            
            btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
            
            const data = {
                age: parseInt(document.getElementById('age').value),
                gender: document.getElementById('gender').value,
                income: parseFloat(document.getElementById('income').value),
                education: document.getElementById('education').value,
                marital_status: document.getElementById('marital_status').value,
                children: parseInt(document.getElementById('children').value),
                home_ownership: document.getElementById('home_ownership').value
            };

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                const result = await response.json();

                // Save analysis for modal
                currentAnalysis = result.analysis;

                // Update UI
                resultDiv.style.display = 'block';
                document.getElementById('scoreText').innerText = result.credit_score + " Risk Profile";
                
                statusBox.className = 'status-box';
                if(result.credit_score === 'High') statusBox.classList.add('status-high');
                else if(result.credit_score === 'Average') statusBox.classList.add('status-avg');
                else statusBox.classList.add('status-low');

            } catch (error) {
                alert("Connection failed.");
            } finally {
                btn.innerText = "Analyze Profile";
            }
        });

        function openModal() {
            const content = document.getElementById('analysisContent');
            content.innerHTML = ''; // Clear prev

            // Add Factors
            currentAnalysis.positive.forEach(item => {
                content.innerHTML += `<div class="analysis-item"><i class="fa-solid fa-circle-check icon-good"></i> <span>${item}</span></div>`;
            });
            currentAnalysis.negative.forEach(item => {
                content.innerHTML += `<div class="analysis-item"><i class="fa-solid fa-circle-exclamation icon-bad"></i> <span>${item}</span></div>`;
            });
            if(currentAnalysis.positive.length === 0 && currentAnalysis.negative.length === 0){
                 content.innerHTML += `<div class="analysis-item"><i class="fa-solid fa-info-circle icon-neutral"></i> <span>Profile is standard with no major outliers.</span></div>`;
            }

            document.getElementById('modalOverlay').classList.add('modal-active');
        }

        function closeModal() {
            document.getElementById('modalOverlay').classList.remove('modal-active');
        }
    </script>
</body>
</html>
"""

# ==========================================
# 3. API & EXPLAINABILITY LOGIC
# ==========================================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class CreditInput(BaseModel):
    age: int = Field(..., ge=0)
    gender: str
    income: float = Field(..., ge=0)
    education: str
    marital_status: str
    children: int = Field(..., ge=0)
    home_ownership: str

@app.get("/", response_class=HTMLResponse)
def home():
    return html_content

@app.post("/predict")
def predict_credit_score(data: CreditInput):
    # 1. Prediction
    edu_mapping = {"High School Diploma": 0, "Associate's Degree": 1, "Bachelor's Degree": 2, "Master's Degree": 3, "Doctorate": 4}
    gender_enc = le_gender.transform([data.gender])[0]
    marital_enc = le_marital.transform([data.marital_status])[0]
    home_enc = le_home.transform([data.home_ownership])[0]
    edu_enc = edu_mapping.get(data.education, 0)

    features = np.array([[data.age, gender_enc, data.income, edu_enc, marital_enc, data.children, home_enc]])
    pred_idx = model.predict(features)[0]
    result_text = le_target.inverse_transform([pred_idx])[0]

    # 2. Explainability Logic (Rule-based Analysis)
    positive_factors = []
    negative_factors = []

    # Income Analysis
    if data.income >= 90000:
        positive_factors.append(f"Strong Income: ${data.income:,.0f} is well above average.")
    elif data.income >= 60000:
        positive_factors.append("Stable Income: Meets standard requirements.")
    else:
        negative_factors.append("Low Income: Income is a limiting factor for credit.")

    # Education Analysis
    if data.education in ["Master's Degree", "Doctorate"]:
        positive_factors.append(f"Education: {data.education} correlates with financial stability.")
    elif data.education == "High School Diploma":
        negative_factors.append("Education: Higher education usually boosts score potential.")

    # Home Ownership Analysis
    if data.home_ownership == "Owned":
        positive_factors.append("Asset: Home ownership indicates long-term stability.")
    elif data.home_ownership == "Rented":
        # Only negative if income is also low
        if data.income < 60000:
            negative_factors.append("Asset: Renting without high income increases risk.")

    # Age Analysis
    if data.age < 25:
        negative_factors.append("Age: Limited credit history due to young age.")
    elif data.age > 40:
        positive_factors.append("Age: Mature profile suggests established credit history.")

    return {
        "credit_score": result_text,
        "analysis": {
            "positive": positive_factors,
            "negative": negative_factors
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
