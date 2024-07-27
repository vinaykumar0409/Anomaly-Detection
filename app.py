from flask import Flask, render_template, request 
from tensorflow.keras.models import load_model #type: ignore
import numpy as np
import joblib
import pandas as pd


app = Flask(__name__)

# Load the trained model
model = load_model('model.keras')

encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST' :
        data = request.form.values()
        data = (list(data))
        terms = [
            "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot", 
            "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", 
            "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", 
            "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", 
            "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", 
            "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
            "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", 
            "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
        ]
        df = pd.DataFrame([data])
        df.columns = terms
        categorical_features = ['protocol_type', 'service', 'flag']

        # One-hot encode the categorical features
        categorical_features_encoded = encoder.transform(df[categorical_features]).toarray()
        categorical_features_encoded = pd.DataFrame(categorical_features_encoded, columns=encoder.get_feature_names_out(categorical_features))
        df.drop(categorical_features, axis=1, inplace=True)
        processed_data = pd.concat([df, categorical_features_encoded], axis=1)
        # Normalize the data using MinMaxScaler
        processed_data_scaled = scaler.transform(processed_data)
        print(processed_data)
        # Make prediction
        prediction = model.predict(processed_data_scaled)
        print('work')
        print(prediction)
        # Calculate reconstruction error (using MAE)
        reconstruction_errors = np.mean(np.abs(processed_data_scaled - prediction), axis=1)
        print('re')
        print(reconstruction_errors, 'error')
        
        # Classify as anomaly if error > threshold
        threshold = 0.030
        predictions = (reconstruction_errors > threshold).astype(int)
        print(predictions)
        res = ''
        if predictions[0] :
            res = 'It is an anomaly'
        else :
            res = 'It is an normal sample'

        # Render results.html with the prediction
        return render_template('results.html', prediction=res)
    
    return render_template('prediction-page.html')

if __name__ == "__main__":
    app.run(debug=True)