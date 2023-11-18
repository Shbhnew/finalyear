from flask import Flask, request, render_template
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

with open('num_of_crop.pkl', 'rb') as f:
    crop_dict = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def frontend():
    return render_template("front.html")

@app.route("/predict", methods=['POST'])
def predict():
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosphorus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['pH'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Print intermediate steps for debugging
    print("Input data:", f"N={N}, P={P}, K={K}, Temp={temp}, Humidity={humidity}, pH={ph}, Rainfall={rainfall}")
    print("Original Features:", single_pred)

    scaled_features = ms.transform(single_pred)
    print("Scaled Features:", scaled_features)

    final_features = sc.transform(scaled_features)
    print("Final Features (after standard scaling):", final_features)

    prediction = model.predict(final_features)
    print("Model prediction (label):", prediction)

    if not np.issubdtype(prediction.dtype, np.integer):
        prediction = prediction.astype(int)

    # Use the trained label encoder to get the crop name
    if prediction[0] in crop_dict.values():
        crop_name = [crop for crop, label in crop_dict.items() if label == prediction[0]][0]
        print("Predicted Crop:", crop_name)
        result = "{} is the best crop to be cultivated right there".format(crop_name)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    # Print the final result for debugging
    print("Final Result:", result)

    return render_template('front.html', result=result)


if __name__ == "__main__":
    app.run(debug=True)
