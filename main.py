import pandas as pd
import io
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
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

# Preprocess
encoders = {}
text_columns = ['Gender', 'Education', 'Marital Status', 'Home Ownership', 'Credit Score']

for col in text_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop('Credit Score', axis=1)
y = df['Credit Score']

# --- MODEL TRAINING ---
# Use RandomForest with class_weight='balanced' to handle 'Average' and 'Low' better
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
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; font-family: 'Poppins', sans-serif; }
        body { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); min-height: 100vh; display: flex; justify-content: center; align-items: center; padding: 20px; }
        .card { background: #ffffff; width: 100%; max-width: 450px; border-radius: 20px; padding: 40px; box-shadow: 0 15px 35px rgba(0, 0, 0, 0.3); animation: slideUp 0.8s ease; }
        .header { text-align: center; margin-bottom: 25px; }
        .header h1 { font-size: 26px; color: #333; font-weight: 700; }
        .header p { color: #666; font-size: 13px; }
        .grid-row { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
        .input-group { margin-bottom: 15px; }
        label { display: block; font-size: 12px; color: #444; font-weight: 600; margin-bottom: 6px; text-transform: uppercase; }
        input, select { width: 100%; padding: 12px; font-size: 14px; background-color: #f4f6f8; border: 1px solid #e1e4e8; border-radius: 8px; outline: none; transition: 0.3s; }
        input:focus, select:focus { border-color: #2a5298; background-color: #fff; }
        button { width: 100%; padding: 14px; background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); color: white; font-size: 16px; font-weight: 600; border: none; border-radius: 8px; cursor: pointer; margin-top: 15px; }
        button:hover { transform: translateY(-2px); }
        .result-container { margin-top: 25px; padding: 15px; border-radius: 10px; text-align: center; background-color: #f8f9fa; border: 1px solid #eee; display: none; }
        .result-container.show { display: block; animation: fadeIn 0.5s; }
        .high { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .average { background: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
        .low { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        @keyframes slideUp { from { opacity: 0; transform: translateY(30px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <h1>Credit Score AI</h1>
            <p>Predict creditworthiness instantly</p>
        </div>
        <form id="predictionForm">
            <div class="grid-row">
                <div class="input-group">
                    <label>Age</label>
                    <input type="number" id="age" placeholder="30" required>
                </div>
                <div class="input-group">
                    <label>Gender</label>
                    <select id="gender">
                        <option value="Female">Female</option>
                        <option value="Male">Male</option>
                    </select>
                </div>
            </div>

            <div class="input-group">
                <label>Annual Income ($)</label>
                <input type="number" id="income" placeholder="50000" required>
            </div>

            <div class="grid-row">
                <div class="input-group">
                    <label>Education</label>
                    <select id="education">
                        <option value="High School Diploma">High School</option>
                        <option value="Associate's Degree">Associate</option>
                        <option value="Bachelor's Degree">Bachelor's</option>
                        <option value="Master's Degree">Master's</option>
                        <option value="Doctorate">Doctorate</option>
                    </select>
                </div>
                <div class="input-group">
                    <label>Marital Status</label>
                    <select id="marital_status">
                        <option value="Single">Single</option>
                        <option value="Married">Married</option>
                    </select>
                </div>
            </div>

            <div class="grid-row">
                <div class="input-group">
                    <label>Children</label>
                    <input type="number" id="children" value="0" required>
                </div>
                <div class="input-group">
                    <label>Home</label>
                    <select id="home_ownership">
                        <option value="Rented">Rented</option>
                        <option value="Owned">Owned</option>
                    </select>
                </div>
            </div>

            <button type="submit" id="predictBtn">Analyze Score</button>
        </form>

        <div id="result" class="result-container">
            <span style="font-size: 12px; text-transform: uppercase;">Prediction Result</span>
            <h2 id="scoreText">--</h2>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            const btn = document.getElementById('predictBtn');
            const resultBox = document.getElementById('result');
            const scoreText = document.getElementById('scoreText');
            
            btn.innerHTML = "Processing...";
            btn.style.opacity = "0.7";
            
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
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                const result = await response.json();
                
                btn.innerHTML = "Analyze Score";
                btn.style.opacity = "1";

                if(result.credit_score) {
                    scoreText.innerText = result.credit_score;
                    resultBox.className = "result-container show";
                    if(result.credit_score === 'High') resultBox.classList.add('high');
                    else if(result.credit_score === 'Average') resultBox.classList.add('average');
                    else resultBox.classList.add('low');
                } 
            } catch (error) {
                console.error(error);
                btn.innerHTML = "Error";
                btn.style.opacity = "1";
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
    age: int
    gender: str
    income: float
    education: str
    marital_status: str
    children: int
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
