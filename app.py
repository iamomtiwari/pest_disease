from flask import Flask, render_template, request, redirect, url_for
import os
from model.model import CropDiseaseModel

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Initialize the model
model = CropDiseaseModel()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Save the uploaded image to the uploads folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Get the prediction from the model
        predicted_class, solution = model.predict(filepath)
        
        # Display the result (class and solution)
        return f"The predicted disease is: {predicted_class}. Suggested solution: {solution}"

if __name__ == "__main__":
    app.run(debug=True)
