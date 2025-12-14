import pandas as pd
import io
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. DATASET & TRAINING
# ==========================================
csv_data = """Age,Gender,Income,Education,Marital Status,Number of Children,Home Ownership,Credit Score
25,Female,50000,Bachelor's Degree,Single,0,Rented,High
30,Male,100000,Master's Degree,Married,2,Owned,High
35,Female,75000,Doctorate,Married,1,Owned,High
40,Male,125000,High School Diploma,Single,0,Owned,High
45,Female,100000,Bachelor's Degree,Married,3,Owned,High
50,Male,150000,Master's Degree,Married,0,Owned,High
26,Female,40000,Associate's Degree,Single,0,Rented,Average
31,Male,60000,Bachelor's Degree,Single,0,Rented,Average
36,Female,80000,Master's Degree,Married,2,Owned,High
41,Male,105000,Doctorate,Single,0,Owned,High
46,Female,90000,High School Diploma,Married,1,Owned,High
51,Male,135000,Bachelor's Degree,Married,0,Owned,High
27,Female,35000,High School Diploma,Single,0,Rented,Low
32,Male,55000,Associate's Degree,Single,0,Rented,Average
37,Female,70000,Bachelor's Degree,Married,2,Owned,High
42,Male,95000,Master's Degree,Single,0,Owned,High
47,Female,85000,Doctorate,Married,1,Owned,High
52,Male,125000,High School Diploma,Married,0,Owned,High
28,Female,30000,Associate's Degree,Single,0,Rented,Low
33,Male,50000,High School Diploma,Single,0,Rented,Average
38,Female,65000,Bachelor's Degree,Married,2,Owned,High
43,Male,80000,Master's Degree,Single,0,Owned,High
48,Female,70000,Doctorate,Married,1,Owned,High
53,Male,115000,Associate's Degree,Married,0,Owned,High
29,Female,25000,High School Diploma,Single,0,Rented,Low
34,Male,45000,Associate's Degree,Single,0,Rented,Average
39,Female,60000,Bachelor's Degree,Married,2,Owned,High
44,Male,75000,Master's Degree,Single,0,Owned,High
49,Female,65000,Doctorate,Married,1,Owned,High
25,Female,55000,Bachelor's Degree,Single,0,Rented,Average
30,Male,105000,Master's Degree,Married,2,Owned,High
35,Female,80000,Doctorate,Married,1,Owned,High
40,Male,130000,High School Diploma,Single,0,Owned,High
45,Female,105000,Bachelor's Degree,Married,3,Owned,High
50,Male,155000,Master's Degree,Married,0,Owned,High
26,Female,45000,Associate's Degree,Single,0,Rented,Average
31,Male,65000,Bachelor's Degree,Single,0,Rented,Average
36,Female,85000,Master's Degree,Married,2,Owned,High
41,Male,110000,Doctorate,Single,0,Owned,High
46,Female,95000,High School Diploma,Married,1,Owned,High
51,Male,140000,Bachelor's Degree,Married,0,Owned,High
27,Female,37500,High School Diploma,Single,0,Rented,Low
32,Male,57500,Associate's Degree,Single,0,Rented,Average
37,Female,72500,Bachelor's Degree,Married,2,Owned,High
42,Male,100000,Master's Degree,Single,0,Owned,High
47,Female,90000,Doctorate,Married,1,Owned,High
52,Male,130000,High School Diploma,Married,0,Owned,High
28,Female,32500,Associate's Degree,Single,0,Rented,Low
33,Male,52500,High School Diploma,Single,0,Rented,Average
38,Female,67500,Bachelor's Degree,Married,2,Owned,High
43,Male,92500,Master's Degree,Single,0,Owned,High
48,Female,82500,Doctorate,Married,1,Owned,High
53,Male,122500,Associate's Degree,Married,0,Owned,High
29,Female,27500,High School Diploma,Single,0,Rented,Low
34,Male,47500,Associate's Degree,Single,0,Rented,Average
39,Female,62500,Bachelor's Degree,Married,2,Owned,High
44,Male,87500,Master's Degree,Single,0,Owned,High
49,Female,77500,Doctorate,Married,1,Owned,High
25,Female,57500,Bachelor's Degree,Single,0,Rented,Average
30,Male,112500,Master's Degree,Married,2,Owned,High
35,Female,85000,Doctorate,Married,1,Owned,High
25,Female,60000,Bachelor's Degree,Single,0,Rented,Average
30,Male,117500,Master's Degree,Married,2,Owned,High
35,Female,90000,Doctorate,Married,1,Owned,High
40,Male,142500,High School Diploma,Single,0,Owned,High
45,Female,110000,Bachelor's Degree,Married,3,Owned,High
50,Male,160000,Master's Degree,Married,0,Owned,High
26,Female,47500,Associate's Degree,Single,0,Rented,Average
31,Male,67500,Bachelor's Degree,Single,0,Rented,Average
36,Female,90000,Master's Degree,Married,2,Owned,High
41,Male,115000,Doctorate,Single,0,Owned,High
46,Female,97500,High School Diploma,Married,1,Owned,High
51,Male,145000,Bachelor's Degree,Married,0,Owned,High
27,Female,37500,High School Diploma,Single,0,Rented,Low
32,Male,57500,Associate's Degree,Single,0,Rented,Average
37,Female,75000,Bachelor's Degree,Married,2,Owned,High
42,Male,105000,Master's Degree,Single,0,Owned,High
47,Female,95000,Doctorate,Married,1,Owned,High
52,Male,135000,High School Diploma,Married,0,Owned,High
28,Female,32500,Associate's Degree,Single,0,Rented,Low
33,Male,52500,High School Diploma,Single,0,Rented,Average
38,Female,67500,Bachelor's Degree,Married,2,Owned,High
43,Male,92500,Master's Degree,Single,0,Owned,High
48,Female,85000,Doctorate,Married,1,Owned,High
53,Male,125000,Associate's Degree,Married,0,Owned,High
29,Female,27500,High School Diploma,Single,0,Rented,Low
34,Male,47500,Associate's Degree,Single,0,Rented,Average
39,Female,62500,Bachelor's Degree,Married,2,Owned,High
44,Male,87500,Master's Degree,Single,0,Owned,High
49,Female,77500,Doctorate,Married,1,Owned,High
25,Female,57500,Bachelor's Degree,Single,0,Rented,Average
30,Male,112500,Master's Degree,Married,2,Owned,High
35,Female,85000,Doctorate,Married,1,Owned,High
25,Female,62500,Bachelor's Degree,Single,0,Rented,Average
30,Male,117500,Master's Degree,Married,2,Owned,High
35,Female,90000,Doctorate,Married,1,Owned,High
40,Male,142500,High School Diploma,Single,0,Owned,High
45,Female,115000,Bachelor's Degree,Married,3,Owned,High
50,Male,162500,Master's Degree,Married,0,Owned,High
26,Female,50000,Associate's Degree,Single,0,Rented,Average
31,Male,70000,Bachelor's Degree,Single,0,Rented,Average
36,Female,95000,Master's Degree,Married,2,Owned,High
41,Male,120000,Doctorate,Single,0,Owned,High
46,Female,102500,High School Diploma,Married,1,Owned,High
51,Male,150000,Bachelor's Degree,Married,0,Owned,High
27,Female,37500,High School Diploma,Single,0,Rented,Low
32,Male,57500,Associate's Degree,Single,0,Rented,Average
37,Female,77500,Bachelor's Degree,Married,2,Owned,High
42,Male,110000,Master's Degree,Single,0,Owned,High
47,Female,97500,Doctorate,Married,1,Owned,High
52,Male,137500,High School Diploma,Married,0,Owned,High
28,Female,32500,Associate's Degree,Single,0,Rented,Low
33,Male,52500,High School Diploma,Single,0,Rented,Average
38,Female,67500,Bachelor's Degree,Married,2,Owned,High
43,Male,95000,Master's Degree,Single,0,Owned,High
48,Female,87500,Doctorate,Married,1,Owned,High
53,Male,127500,Associate's Degree,Married,0,Owned,High
29,Female,27500,High School Diploma,Single,0,Rented,Low
34,Male,47500,Associate's Degree,Single,0,Rented,Average
39,Female,62500,Bachelor's Degree,Married,2,Owned,High
44,Male,87500,Master's Degree,Single,0,Owned,High
49,Female,77500,Doctorate,Married,1,Owned,High
25,Female,57500,Bachelor's Degree,Single,0,Rented,Average
30,Male,112500,Master's Degree,Married,2,Owned,High
35,Female,85000,Doctorate,Married,1,Owned,High
25,Female,60000,Bachelor's Degree,Single,0,Rented,Average
30,Male,117500,Master's Degree,Married,2,Owned,High
35,Female,90000,Doctorate,Married,1,Owned,High
28,Male,75000,Bachelor's Degree,Single,0,Rented,Average
33,Female,82000,Master's Degree,Married,1,Owned,High
31,Male,95000,Doctorate,Single,0,Rented,High
26,Female,55000,Bachelor's Degree,Married,1,Owned,Average
32,Male,85000,Master's Degree,Single,0,Rented,High
29,Female,68000,Doctorate,Married,2,Owned,Average
34,Male,105000,Bachelor's Degree,Married,1,Rented,High
25,Female,55000,Bachelor's Degree,Single,0,Rented,Average
30,Male,105000,Master's Degree,Married,2,Owned,High
35,Female,80000,Doctorate,Married,1,Owned,High
40,Male,130000,High School Diploma,Single,0,Owned,High
45,Female,105000,Bachelor's Degree,Married,3,Owned,High
50,Male,155000,Master's Degree,Married,0,Owned,High
26,Female,45000,Associate's Degree,Single,0,Rented,Average
31,Male,65000,Bachelor's Degree,Single,0,Rented,Average
36,Female,85000,Master's Degree,Married,2,Owned,High
41,Male,110000,Doctorate,Single,0,Owned,High
46,Female,95000,High School Diploma,Married,1,Owned,High
51,Male,140000,Bachelor's Degree,Married,0,Owned,High
27,Female,37500,High School Diploma,Single,0,Rented,Low
32,Male,57500,Associate's Degree,Single,0,Rented,Average
37,Female,72500,Bachelor's Degree,Married,2,Owned,High
42,Male,100000,Master's Degree,Single,0,Owned,High
47,Female,90000,Doctorate,Married,1,Owned,High
52,Male,130000,High School Diploma,Married,0,Owned,High
28,Female,32500,Associate's Degree,Single,0,Rented,Low
33,Male,52500,High School Diploma,Single,0,Rented,Average
38,Female,67500,Bachelor's Degree,Married,2,Owned,High
43,Male,92500,Master's Degree,Single,0,Owned,High
48,Female,82500,Doctorate,Married,1,Owned,High
53,Male,122500,Associate's Degree,Married,0,Owned,High
29,Female,27500,High School Diploma,Single,0,Rented,Low
34,Male,47500,Associate's Degree,Single,0,Rented,Average
39,Female,62500,Bachelor's Degree,Married,2,Owned,High
44,Male,87500,Master's Degree,Single,0,Owned,High
49,Female,77500,Doctorate,Married,1,Owned,High"""

df = pd.read_csv(io.StringIO(csv_data))

# Preprocess & Train
encoders = {}
text_columns = ['Gender', 'Education', 'Marital Status', 'Home Ownership', 'Credit Score']

for col in text_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop('Credit Score', axis=1)
y = df['Credit Score']

# Random Forest with Balanced Weight
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X, y)

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
            --card-bg: #ffffff;
            --text-dark: #1f2937;
            --text-light: #6b7280;
            --border-color: #e5e7eb;
            --shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
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
            box-shadow: var(--shadow); 
            border: 1px solid rgba(255,255,255,0.7);
        }

        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { 
            font-family: 'Poppins', sans-serif; 
            font-size: 28px; 
            color: var(--text-dark); 
            font-weight: 700; 
            letter-spacing: -0.5px;
        }
        .header p { 
            color: var(--text-light); 
            font-size: 14px; 
            margin-top: 5px; 
        }

        .grid-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .input-wrapper { margin-bottom: 20px; position: relative; }

        label { 
            display: block; 
            font-size: 13px; 
            color: var(--text-dark); 
            font-weight: 600; 
            margin-bottom: 8px; 
        }

        .input-group {
            position: relative;
            display: flex;
            align-items: center;
        }

        .input-group i {
            position: absolute;
            left: 15px;
            color: #9ca3af;
            font-size: 14px;
            pointer-events: none;
        }

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
            content: '\\f078';
            font-family: 'Font Awesome 6 Free';
            font-weight: 900;
            position: absolute;
            right: 15px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 10px;
            color: #9ca3af;
            pointer-events: none;
        }

        input:focus, select:focus { 
            border-color: var(--primary); 
            background-color: #fff; 
            box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.1); 
        }

        button { 
            width: 100%; 
            padding: 14px; 
            background-color: var(--primary); 
            color: white; 
            font-size: 15px; 
            font-weight: 600; 
            border: none; 
            border-radius: 10px; 
            cursor: pointer; 
            margin-top: 10px; 
            transition: background-color 0.2s, transform 0.1s; 
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
        }

        button:hover { background-color: var(--primary-hover); }
        button:active { transform: scale(0.98); }

        .result-container { 
            margin-top: 25px; 
            padding: 20px; 
            border-radius: 12px; 
            text-align: center; 
            background-color: #f9fafb; 
            border: 1px solid var(--border-color); 
            display: none; 
            animation: fadeIn 0.4s ease;
        }

        .result-container.show { display: block; }
        .result-title { font-size: 12px; text-transform: uppercase; color: var(--text-light); letter-spacing: 1px; font-weight: 600; }
        .result-value { font-family: 'Poppins', sans-serif; font-size: 24px; margin-top: 5px; font-weight: 700; }

        .status-high { color: #059669; background: #d1fae5; border: 1px solid #10b981; }
        .status-avg { color: #d97706; background: #fef3c7; border: 1px solid #f59e0b; }
        .status-low { color: #dc2626; background: #fee2e2; border: 1px solid #ef4444; }

        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        
        .spinner {
            border: 2px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top: 2px solid white;
            width: 16px;
            height: 16px;
            animation: spin 1s linear infinite;
            display: none;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <h1>Financial AI</h1>
            <p>Creditworthiness Prediction Engine</p>
        </div>
        
        <form id="predictionForm">
            <!-- Row 1 -->
            <div class="grid-row">
                <div class="input-wrapper">
                    <label>Age</label>
                    <div class="input-group">
                        <i class="fa-solid fa-user"></i>
                        <input type="number" id="age" min="0" required placeholder="Years">
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

            <!-- Income -->
            <div class="input-wrapper">
                <label>Annual Income ($)</label>
                <div class="input-group">
                    <i class="fa-solid fa-dollar-sign"></i>
                    <input type="number" id="income" min="0" required placeholder="e.g. 55000">
                </div>
            </div>

            <!-- Row 2 -->
            <div class="grid-row">
                <div class="input-wrapper">
                    <label>Education</label>
                    <div class="input-group select-wrapper">
                        <i class="fa-solid fa-graduation-cap"></i>
                        <select id="education">
                            <option value="High School Diploma">High School</option>
                            <option value="Associate's Degree">Associate</option>
                            <option value="Bachelor's Degree">Bachelor's</option>
                            <option value="Master's Degree">Master's</option>
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

            <!-- Row 3 -->
            <div class="grid-row">
                <div class="input-wrapper">
                    <label>Children</label>
                    <div class="input-group">
                        <i class="fa-solid fa-child"></i>
                        <input type="number" id="children" min="0" required placeholder="Count">
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

            // Validation
            if(ageVal < 0 || incomeVal < 0 || childrenVal < 0) {
                alert("Please enter positive numbers only.");
                return;
            }

            // Loading State
            btnText.innerText = "Processing...";
            btn.style.opacity = "0.9";
            spinner.style.display = "block";
            // Hide result box initially, but clear class so it doesn't conflict
            resultBox.style.display = "none";
            resultBox.className = "result-container"; 
            
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
                // Simulate a slight delay for better UX feel
                await new Promise(r => setTimeout(r, 500));

                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if(response.ok && result.credit_score) {
                    scoreText.innerText = result.credit_score;
                    
                    // Force display block via inline style
                    resultBox.style.display = "block";
                    resultBox.className = "result-container show"; // Add show class
                    
                    // Remove old status classes
                    resultBox.classList.remove('status-high', 'status-avg', 'status-low');
                    
                    if(result.credit_score === 'High') {
                        resultBox.classList.add('status-high');
                    } else if(result.credit_score === 'Average') {
                        resultBox.classList.add('status-avg');
                    } else {
                        resultBox.classList.add('status-low');
                    }
                } else {
                     alert(result.detail || "Error processing request");
                }
            } catch (error) {
                console.error(error);
                alert("Server connection failed.");
            } finally {
                // Reset State
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
    age: int = Field(..., ge=0, description="Age cannot be negative")
    gender: str
    income: float = Field(..., ge=0, description="Income cannot be negative")
    education: str
    marital_status: str
    children: int = Field(..., ge=0, description="Children cannot be negative")
    home_ownership: str

@app.get("/", response_class=HTMLResponse)
def home():
    return html_content

@app.post("/predict")
def predict_credit_score(data: CreditInput):
    try:
        gender_enc = encoders['Gender'].transform([data.gender])[0]
        education_enc = encoders['Education'].transform([data.education])[0]
        marital_enc = encoders['Marital Status'].transform([data.marital_status])[0]
        home_enc = encoders['Home Ownership'].transform([data.home_ownership])[0]

        features = np.array([[
            data.age,
            gender_enc,
            data.income,
            education_enc,
            marital_enc,
            data.children,
            home_enc
        ]])

        prediction_index = model.predict(features)[0]
        result_text = encoders['Credit Score'].inverse_transform([prediction_index])[0]

        return {"credit_score": result_text}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
