from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('CKD.pkl', 'rb'))  # Make sure CKD.pkl is in the same directory

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        bp = float(request.form['bp'])
        rbc = 1 if request.form['rbc'] == 'normal' else 0
        pc = 1 if request.form['pc'] == 'normal' else 0
        pcc = 1 if request.form['pcc'] == 'present' else 0
        ba = 1 if request.form['ba'] == 'present' else 0

        # ✅ Only the 6 features your model was trained on
        inputs = np.array([[age, bp, rbc, pc, pcc, ba]])

        result = model.predict(inputs)[0]
        msg = "CKD Detected" if result == 1 else "No CKD Detected"
        return render_template('result.html', prediction_text=msg)

    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
  app.run(debug=True)


