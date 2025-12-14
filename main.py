import pandas as pd
import numpy as np
import random
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ==========================================
# 1. INTELLIGENT DATA GENERATION & TRAINING
# ==========================================

def generate_logical_data(n=1000):
    """
    Generates a dataset based on LOGIC so the model learns reasonable rules.
    """
    data = []
    
    for _ in range(n):
        # 1. Random base details
        age = random.randint(18, 70)
        gender = random.choice(["Male", "Female"])
        marital = random.choice(["Single", "Married"])
        children = random.randint(0, 3)
        
        # 2. Logic: Education & Income correlation
        # Higher education usually correlates with higher income
        edu_options = ["High School Diploma", "Associate's Degree", "Bachelor's Degree", "Master's Degree", "Doctorate"]
        edu = random.choice(edu_options)
        
        if edu == "High School Diploma":
            income = random.randint(25000, 50000)
        elif edu == "Associate's Degree":
            income = random.randint(40000, 65000)
        elif edu == "Bachelor's Degree":
            income = random.randint(55000, 90000)
        elif edu == "Master's Degree":
            income = random.randint(80000, 130000)
        else: # Doctorate
            income = random.randint(100000, 180000)
            
        # Add random noise to income (some high schoolers earn a lot, some PhDs earn less)
        income += random.randint(-5000, 15000)

        # 3. Logic: Home Ownership
        # Richer people likely own homes
        if income > 85000:
            home = random.choices(["Owned", "Rented"], weights=[80, 20])[0]
        else:
            home = random.choices(["Owned", "Rented"], weights=[30, 70])[0]

        # 4. Logic: DETERMINE CREDIT SCORE (The Answer Key)
        # We assign points to decide the score
        points = 0
        
        # Income points
        if income > 90000: points += 50
        elif income > 60000: points += 30
        else: points += 10
        
        # Home points
        if home == "Owned": points += 20
        
        # Age points (Older people usually have better credit history)
        if age > 35: points += 10
        
        # Education points
        if edu in ["Master's Degree", "Doctorate"]: points += 10
        
        # Marital points (Married often more stable financially in datasets)
        if marital == "Married": points += 5

        # Determine Final Label
        if points >= 70:
            score = "High"
        elif points >= 40:
            score = "Average"
        else:
            score = "Low"

        data.append([age, gender, income, edu, marital, children, home, score])

    columns = ['Age', 'Gender', 'Income', 'Education', 'Marital Status', 'Number of Children', 'Home Ownership', 'Credit Score']
    return pd.DataFrame(data, columns=columns)

# Generate the data
df = generate_logical_data(1500)

# --- PREPROCESSING ---

# 1. Map Ordinal Data (Order matters!)
# This fixes the issue where the model didn't know Ph.D > High School
edu_mapping = {
    "High School Diploma": 0,
    "Associate's Degree": 1,
    "Bachelor's Degree": 2,
    "Master's Degree": 3,
    "Doctorate": 4
}
df['Education_Enc'] = df['Education'].map(edu_mapping)

# 2. Encode Categorical Data
le_gender = LabelEncoder()
df['Gender_Enc'] = le_gender.fit_transform(df['Gender'])

le_marital = LabelEncoder()
df['Marital_Enc'] = le_marital.fit_transform(df['Marital Status'])

le_home = LabelEncoder()
df['Home_Enc'] = le_home.fit_transform(df['Home Ownership'])

# 3. Encode Target
le_target = LabelEncoder()
df['Target'] = le_target.fit_transform(df['Credit Score'])

# 4. Select Features
# Features: Age, Gender, Income, Education_Level, Marital, Children, Home
X = df[['Age', 'Gender_Enc', 'Income', 'Education_Enc', 'Marital_Enc', 'Number of Children', 'Home_Enc']]
y = df['Target']

# 5. Train Model
# Increased estimators for better accuracy
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X, y)

print("✅ Model Trained on 1500 logical examples.")

# ==========================================
# 2. HTML INTERFACE (Fixed JS & Styling)
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
            --card-bg: #ffffff;
            --text-dark: #1f2937;
            --text-light: #6b7280;
            --border-color: #e5e7eb;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Inter', sans-serif; }
        
        body { 
            background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%); 
            min-height: 100vh; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            padding: 20px; 
        }

        .card { 
            background: var(--card-bg); 
            width: 100%; 
            max-width: 500px; 
            border-radius: 16px; 
            padding: 40px; 
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1); 
            border: 1px solid rgba(255,255,255,0.7);
        }

        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { 
            font-family: 'Poppins', sans-serif; 
            font-size: 28px; 
            color: var(--text-dark); 
            font-weight: 700; 
        }
        .header p { color: var(--text-light); font-size: 14px; margin-top: 5px; }

        .grid-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .input-wrapper { margin-bottom: 20px; }

        label { display: block; font-size: 13px; color: var(--text-dark); font-weight: 600; margin-bottom: 8px; }

        .input-group { position: relative; display: flex; align-items: center; }
        .input-group i { position: absolute; left: 15px; color: #9ca3af; font-size: 14px; pointer-events: none; }

        input, select { 
            width: 100%; 
            padding: 12px 15px 12px 40px; 
            font-size: 14px; 
            color: var(--text-dark);
            background-color: #f9fafb; 
            border: 1px solid var(--border-color); 
            border-radius: 10px; 
            outline: none; 
            transition: all 0.2s ease;
            appearance: none; 
        }
        
        .select-wrapper::after {
            content: '\\f078'; font-family: 'Font Awesome 6 Free'; font-weight: 900;
            position: absolute; right: 15px; font-size: 10px; color: #9ca3af; pointer-events: none;
        }

        input:focus, select:focus { 
            border-color: var(--primary); background-color: #fff; 
            box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.1); 
        }

        button { 
            width: 100%; padding: 14px; background-color: var(--primary); color: white; 
            font-size: 15px; font-weight: 600; border: none; border-radius: 10px; 
            cursor: pointer; margin-top: 10px; transition: all 0.2s; 
            display: flex; justify-content: center; align-items: center; gap: 10px;
        }
        button:hover { background-color: var(--primary-hover); }

        .result-container { 
            margin-top: 25px; padding: 20px; border-radius: 12px; 
            text-align: center; background-color: #f9fafb; 
            border: 1px solid var(--border-color); 
            display: none; /* Hidden by default */
        }

        .result-title { font-size: 12px; text-transform: uppercase; color: var(--text-light); font-weight: 600; }
        .result-value { font-family: 'Poppins', sans-serif; font-size: 24px; margin-top: 5px; font-weight: 700; }

        .status-high { color: #059669; background: #d1fae5; border-color: #10b981; }
        .status-avg { color: #d97706; background: #fef3c7; border-color: #f59e0b; }
        .status-low { color: #dc2626; background: #fee2e2; border-color: #ef4444; }
        
        .spinner {
            border: 2px solid rgba(255,255,255,0.3); border-radius: 50%; border-top: 2px solid white;
            width: 16px; height: 16px; animation: spin 1s linear infinite; display: none;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <h1>Financial AI</h1>
            <p>Smart Credit Scoring System</p>
        </div>
        
        <form id="predictionForm">
            <div class="grid-row">
                <div class="input-wrapper">
                    <label>Age</label>
                    <div class="input-group">
                        <i class="fa-solid fa-user"></i>
                        <input type="number" id="age" min="18" required placeholder="Years">
                    </div>
                </div>
                <div class="input-wrapper">
                    <label>Gender</label>
                    <div class="input-group select-wrapper">
                        <i class="fa-solid fa-venus-mars"></i>
                        <select id="gender">
                            <option value="Female">Female</option>
                            <option value="Male">Male</option>
                        </select>
                    </div>
                </div>
            </div>

            <div class="input-wrapper">
                <label>Annual Income ($)</label>
                <div class="input-group">
                    <i class="fa-solid fa-dollar-sign"></i>
                    <input type="number" id="income" min="0" required placeholder="e.g. 55000">
                </div>
            </div>

            <div class="grid-row">
                <div class="input-wrapper">
                    <label>Education</label>
                    <div class="input-group select-wrapper">
                        <i class="fa-solid fa-graduation-cap"></i>
                        <select id="education">
                            <option value="High School Diploma">High School</option>
                            <option value="Associate's Degree">Associate's Degree</option>
                            <option value="Bachelor's Degree">Bachelor's Degree</option>
                            <option value="Master's Degree">Master's Degree</option>
                            <option value="Doctorate">Doctorate</option>
                        </select>
                    </div>
                </div>
                <div class="input-wrapper">
                    <label>Marital Status</label>
                    <div class="input-group select-wrapper">
                        <i class="fa-solid fa-heart"></i>
                        <select id="marital_status">
                            <option value="Single">Single</option>
                            <option value="Married">Married</option>
                        </select>
                    </div>
                </div>
            </div>

            <div class="grid-row">
                <div class="input-wrapper">
                    <label>Children</label>
                    <div class="input-group">
                        <i class="fa-solid fa-child"></i>
                        <input type="number" id="children" min="0" required value="0">
                    </div>
                </div>
                <div class="input-wrapper">
                    <label>Home Type</label>
                    <div class="input-group select-wrapper">
                        <i class="fa-solid fa-house"></i>
                        <select id="home_ownership">
                            <option value="Rented">Rented</option>
                            <option value="Owned">Owned</option>
                        </select>
                    </div>
                </div>
            </div>

            <button type="submit" id="predictBtn">
                <span>Analyze Credit Score</span>
                <div class="spinner" id="spinner"></div>
            </button>
        </form>

        <div id="result" class="result-container">
            <div class="result-title">AI Prediction</div>
            <div id="scoreText" class="result-value">--</div>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const btn = document.getElementById('predictBtn');
            const spinner = document.getElementById('spinner');
            const btnText = btn.querySelector('span');
            const resultBox = document.getElementById('result');
            const scoreText = document.getElementById('scoreText');
            
            // Get values
            const ageVal = document.getElementById('age').value;
            const incomeVal = document.getElementById('income').value;
            const childrenVal = document.getElementById('children').value;

            if(ageVal < 0 || incomeVal < 0 || childrenVal < 0) {
                alert("Please enter valid numbers.");
                return;
            }

            // Loading State
            btnText.innerText = "Analyzing...";
            btn.style.opacity = "0.8";
            spinner.style.display = "block";
            resultBox.style.display = "none";
            
            const data = {
                age: parseInt(ageVal),
                gender: document.getElementById('gender').value,
                income: parseFloat(incomeVal),
                education: document.getElementById('education').value,
                marital_status: document.getElementById('marital_status').value,
                children: parseInt(childrenVal),
                home_ownership: document.getElementById('home_ownership').value
            };

            try {
                // Fetch data
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if(response.ok && result.credit_score) {
                    scoreText.innerText = result.credit_score;
                    
                    // Force the box to show
                    resultBox.style.display = "block";
                    
                    // Color Logic
                    resultBox.className = "result-container"; // reset
                    if(result.credit_score === 'High') {
                        resultBox.classList.add('status-high');
                    } else if(result.credit_score === 'Average') {
                        resultBox.classList.add('status-avg');
                    } else {
                        resultBox.classList.add('status-low');
                    }
                } else {
                     alert("Error: " + (result.detail || "Processing failed"));
                }
            } catch (error) {
                console.error(error);
                alert("Server connection error. Make sure the backend is running.");
            } finally {
                btnText.innerText = "Analyze Credit Score";
                btn.style.opacity = "1";
                spinner.style.display = "none";
            }
        });
    </script>
</body>
</html>
"""

# ==========================================
# 3. API
# ==========================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    try:
        # Manual Ordinal Mapping for Prediction
        edu_mapping = {
            "High School Diploma": 0,
            "Associate's Degree": 1,
            "Bachelor's Degree": 2,
            "Master's Degree": 3,
            "Doctorate": 4
        }
        
        # Transform inputs
        gender_enc = le_gender.transform([data.gender])[0]
        marital_enc = le_marital.transform([data.marital_status])[0]
        home_enc = le_home.transform([data.home_ownership])[0]
        edu_enc = edu_mapping[data.education] # Use map, not encoder

        # Create Array
        features = np.array([[
            data.age,
            gender_enc,
            data.income,
            edu_enc,
            marital_enc,
            data.children,
            home_enc
        ]])

        # Predict
        prediction_index = model.predict(features)[0]
        result_text = le_target.inverse_transform([prediction_index])[0]

        return {"credit_score": result_text}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
