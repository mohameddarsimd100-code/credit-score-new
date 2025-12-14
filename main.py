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
        children = random.choices([0, 1, 2, 3], weights=[40, 30, 20, 10])[0]
        
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

        # Scoring Logic
        points = 0
        if income > 90000: points += 50
        elif income > 60000: points += 30
        else: points += 10
        
        if home == "Owned": points += 20
        if age > 35: points += 10
        if edu in ["Master's Degree", "Doctorate"]: points += 15
        elif edu == "Bachelor's Degree": points += 5
        
        if marital == "Married": points += 10
        if children > 2: points -= 5
        if children == 0: points += 5

        if points >= 80: score = "High"
        elif points >= 50: score = "Average"
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
df['Home_Enc'] = le_home.fit_transform(df['Home Ownership'])

le_target = LabelEncoder()
df['Target'] = le_target.fit_transform(df['Credit Score'])

X = df[['Age', 'Gender_Enc', 'Income', 'Education_Enc', 'Marital_Enc', 'Number of Children', 'Home_Enc']]
y = df['Target']

model = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
model.fit(X, y)

print("✅ Model Trained")

# ==========================================
# 2. HTML INTERFACE
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
            overflow-x: hidden;
        }

        /* WRAPPER FOR ANIMATION */
        .app-container {
            display: flex;
            align-items: flex-start;
            gap: 20px;
            transition: all 0.6s cubic-bezier(0.25, 0.8, 0.25, 1);
            transform: translateX(0);
        }

        /* MAIN FORM CARD */
        .card { 
            background: #ffffff; width: 450px; padding: 40px; 
            border-radius: 20px; box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.15); 
            border: 1px solid rgba(255,255,255,0.8);
            z-index: 2;
            transition: all 0.5s ease;
        }

        /* ANALYSIS CARD (Initially Hidden) */
        .analysis-card {
            background: rgba(255, 255, 255, 0.95);
            width: 0; 
            opacity: 0;
            padding: 0;
            overflow: hidden;
            border-radius: 20px;
            box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.15);
            transition: all 0.6s cubic-bezier(0.25, 0.8, 0.25, 1);
            transform: translateX(-50px);
            height: 100%;
            min-height: 500px;
            display: flex; flex-direction: column;
        }

        /* ACTIVE STATE */
        .app-container.active .analysis-card {
            width: 400px;
            opacity: 1;
            padding: 30px;
            transform: translateX(0);
        }
        
        h1 { font-family: 'Poppins', sans-serif; font-size: 24px; color: var(--text-dark); text-align: center; }
        p.subtext { color: #6b7280; font-size: 13px; text-align: center; margin-bottom: 25px; }

        .grid-row { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .input-group { margin-bottom: 15px; position: relative; }
        label { display: block; font-size: 12px; font-weight: 600; margin-bottom: 6px; color: #374151; }
        
        input, select { 
            width: 100%; padding: 12px 15px 12px 35px; border-radius: 8px; border: 1px solid #e5e7eb; 
            background: #f9fafb; outline: none; transition: 0.2s; font-size: 13px;
        }
        .input-group i { position: absolute; left: 12px; top: 38px; color: #9ca3af; font-size: 13px; }
        input:focus, select:focus { border-color: var(--primary); background: #fff; box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1); }
        
        button { 
            width: 100%; padding: 14px; background: var(--primary); color: white; border: none; border-radius: 10px; 
            font-size: 15px; font-weight: 600; cursor: pointer; transition: 0.2s; margin-top: 15px;
        }
        button:hover { background: var(--primary-hover); transform: translateY(-1px); }

        /* RESULT BADGE */
        .result-area { margin-top: 20px; display: none; }
        .status-badge {
            padding: 15px; border-radius: 12px; text-align: center; 
            font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 18px;
            margin-bottom: 10px; display: flex; align-items: center; justify-content: center; gap: 10px;
        }
        
        /* Analysis Items */
        .analysis-header { border-bottom: 1px solid #eee; padding-bottom: 15px; margin-bottom: 15px; }
        .analysis-item { 
            display: flex; align-items: flex-start; gap: 12px; margin-bottom: 15px; 
            font-size: 13px; color: #4b5563; background: #f8fafc; padding: 10px; border-radius: 8px;
        }
        .icon-good { color: #10b981; font-size: 16px; margin-top: 2px; }
        .icon-bad { color: #ef4444; font-size: 16px; margin-top: 2px; }

        .details-btn {
            background: white; border: 1px solid #d1d5db; color: #374151;
        }
        .details-btn:hover { background: #f9fafb; }
        
        /* THEMES */
        .bg-high { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
        .bg-avg { background: #fffbeb; color: #b45309; border: 1px solid #fcd34d; }
        .bg-low { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }

        @media(max-width: 900px) {
            .app-container.active { flex-direction: column; align-items: center; }
            .analysis-card { width: 100%; transform: translateY(20px); }
            .app-container.active .analysis-card { width: 450px; transform: translateY(0); }
        }
    </style>
</head>
<body>

    <div class="app-container" id="appContainer">
        
        <!-- LEFT: FORM CARD -->
        <div class="card">
            <div class="header">
                <h1>Financial AI</h1>
                <p class="subtext">Credit Eligibility Checker</p>
            </div>
            
            <form id="predictionForm">
                <div class="grid-row">
                    <div class="input-group">
                        <label>Age</label>
                        <i class="fa-solid fa-user"></i>
                        <input type="number" id="age" min="18" required placeholder="30">
                    </div>
                    <div class="input-group">
                        <label>Gender</label>
                        <i class="fa-solid fa-venus-mars"></i>
                        <select id="gender"><option>Female</option><option>Male</option></select>
                    </div>
                </div>

                <div class="input-group">
                    <label>Annual Income ($)</label>
                    <i class="fa-solid fa-money-bill-wave"></i>
                    <input type="number" id="income" min="0" required placeholder="50000">
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

                <div class="grid-row">
                    <div class="input-group">
                        <label>Marital Status</label>
                        <i class="fa-solid fa-ring"></i>
                        <select id="marital_status">
                            <option>Single</option>
                            <option>Married</option>
                        </select>
                    </div>
                    <div class="input-group">
                        <label>Children</label>
                        <i class="fa-solid fa-children"></i>
                        <input type="number" id="children" min="0" max="10" required value="0">
                    </div>
                </div>

                <button type="submit" id="predictBtn">Check Eligibility</button>
            </form>

            <div id="resultArea" class="result-area">
                <div id="scoreBadge" class="status-badge">--</div>
                <button type="button" class="details-btn" id="detailsBtn" onclick="toggleAnalysis()">
                    <i class="fa-solid fa-list-check"></i> &nbsp; View Logic Breakdown
                </button>
            </div>
        </div>

        <!-- RIGHT: ANALYSIS CARD (Slides out) -->
        <div class="analysis-card" id="analysisCard">
            <div class="analysis-header">
                <h2 style="font-family:'Poppins',sans-serif; font-size:18px; color:#1f2937;">
                    <i class="fa-solid fa-magnifying-glass-chart" style="color:var(--primary);"></i> Decision Logic
                </h2>
                <p style="font-size:12px; color:#9ca3af;">Why did the AI make this decision?</p>
            </div>
            
            <div id="analysisContent">
                <!-- Javascript will inject items here -->
            </div>
            
            <div style="margin-top: auto;">
                <button onclick="toggleAnalysis()" style="background:#f3f4f6; color:#374151;">Close</button>
            </div>
        </div>

    </div>

    <script>
        let currentAnalysis = {};

        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const btn = document.getElementById('predictBtn');
            const resultArea = document.getElementById('resultArea');
            const scoreBadge = document.getElementById('scoreBadge');
            
            btn.innerHTML = '<i class="fa-solid fa-circle-notch fa-spin"></i> Analyzing...';
            
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
                
                currentAnalysis = result.analysis;
                
                // Show Result
                resultArea.style.display = 'block';
                
                let statusText = "";
                let icon = "";
                let bgClass = "";

                if (result.credit_score === "High") {
                    statusText = "Credit Approved";
                    icon = "fa-circle-check";
                    bgClass = "bg-high";
                } else if (result.credit_score === "Average") {
                    statusText = "Conditional Review";
                    icon = "fa-file-signature";
                    bgClass = "bg-avg";
                } else {
                    statusText = "Not Eligible";
                    icon = "fa-circle-xmark";
                    bgClass = "bg-low";
                }

                scoreBadge.innerHTML = `<i class="fa-solid ${icon}"></i> ${statusText}`;
                
                scoreBadge.className = 'status-badge';
                scoreBadge.classList.add(bgClass);

                if(document.getElementById('appContainer').classList.contains('active')){
                    populateAnalysis();
                }

            } catch (error) {
                alert("Error connecting to AI server.");
            } finally {
                btn.innerText = "Check Eligibility";
            }
        });

        function toggleAnalysis() {
            const container = document.getElementById('appContainer');
            if(!container.classList.contains('active')) {
                populateAnalysis();
                container.classList.add('active'); 
            } else {
                container.classList.remove('active');
            }
        }

        function populateAnalysis() {
            const content = document.getElementById('analysisContent');
            content.innerHTML = ''; 

            if(currentAnalysis.positive) {
                currentAnalysis.positive.forEach(item => {
                    content.innerHTML += `<div class="analysis-item"><i class="fa-solid fa-check-circle icon-good"></i> <span>${item}</span></div>`;
                });
            }
            if(currentAnalysis.negative) {
                currentAnalysis.negative.forEach(item => {
                    content.innerHTML += `<div class="analysis-item"><i class="fa-solid fa-triangle-exclamation icon-bad"></i> <span>${item}</span></div>`;
                });
            }
        }
    </script>
</body>
</html>
"""

# ==========================================
# 3. API
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
    # 1. Prepare Data
    edu_mapping = {"High School Diploma": 0, "Associate's Degree": 1, "Bachelor's Degree": 2, "Master's Degree": 3, "Doctorate": 4}
    gender_enc = le_gender.transform([data.gender])[0]
    marital_enc = le_marital.transform([data.marital_status])[0]
    home_enc = le_home.transform([data.home_ownership])[0]
    edu_enc = edu_mapping.get(data.education, 0)

    features = np.array([[data.age, gender_enc, data.income, edu_enc, marital_enc, data.children, home_enc]])
    
    # 2. Predict
    pred_idx = model.predict(features)[0]
    result_text = le_target.inverse_transform([pred_idx])[0]

    # 3. Generate Logic Explanation (Detailed Analysis)
    pos = []
    neg = []

    # Income Logic
    if data.income >= 90000: 
        pos.append(f"Strong Income: ${data.income:,.0f} is well above threshold.")
    elif data.income < 40000: 
        neg.append(f"Low Income: ${data.income:,.0f} is below credit safety margins.")
    elif data.income < 60000:
        neg.append("Moderate Income: Limits high-tier approval chances.")
    
    # Marital & Children Logic
    if data.marital_status == "Married": 
        pos.append("Stability: Married status improves risk profile.")
    
    if data.children > 2: 
        neg.append(f"Dependents: {data.children} children increase financial liability.")
    elif data.children == 0: 
        pos.append("Expenses: No dependents lowers monthly liability.")

    # Home Logic
    if data.home_ownership == "Owned": 
        pos.append("Asset: Home ownership serves as strong collateral.")
    else: 
        neg.append("Asset: Renting provides less collateral than owning.")

    # Education Logic
    if data.education in ["Master's Degree", "Doctorate"]: 
        pos.append(f"Education: Advanced degree ({data.education}) correlates with stability.")
    elif data.education == "High School Diploma":
        neg.append("Education: Limited academic background affects score ceiling.")
    
    return {
        "credit_score": result_text,
        "analysis": {"positive": pos, "negative": neg}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
