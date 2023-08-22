import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask
from flask import request as call_request
from flask import Flask, make_response, send_file
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

# Creates Flask serving engine
app = Flask(__name__)

model = None

@app.before_first_request
def init():
    """
    Load model else crash, deployment will not start
    """
    global model
    model = pickle.load(open ('/models/GBRmodel.pkl','rb')) # Here id the model I pretrained and place in S3
    return None

@app.route("/v2/greet", methods=["GET"])
def status():
    global model
    if model is None:
        return "Flask Code: Model was not loaded."
    else:
        return "Model is loaded."

# Here is code for pre-processing and prediction
@app.route("/v2/predict", methods=["POST"])
def predict():
    """
    Perform an inference on the model created in initialize

    Returns:
        String value price.
    """
    global model
    #
    uploaded_file = call_request.files.get('files')
    new_input = pd.read_csv(uploaded_file)
    
    new_input.rename(columns={'Mining Asset': 'Mining_Asset', 'Lifecycle Phase': 'Lifecycle_Phase' }, inplace=True)

    new_input2 = new_input.drop(['Company Name', 'Location (for SAC presentation)', 'Fluoride level (tonnes)'], axis=1)
    # Create a datetime column without day
    new_input2['Date'] = new_input2['Month'] + ' ' + new_input2['Year'].astype(str)
    new_input2['Date'] = pd.to_datetime(new_input2['Date'], format='%B %Y')

    # Drop the original 'Year' and 'Month' columns
    new_input2.drop(['Year', 'Month'], axis=1, inplace=True)
    new_input2['Date'] = new_input2['Date'].apply(pd.Timestamp.timestamp)

    sc2 = StandardScaler()
    enc = OrdinalEncoder()

    new_input2[["Region","Location", "Commodity","Mining_Asset","Lifecycle_Phase"]] = enc.fit_transform(new_input2[["Region","Location", "Commodity","Mining_Asset","Lifecycle_Phase"]])
    sorted_input = new_input2.sort_values(by='Date', ascending=True)
    sorted_input.reset_index(drop=True, inplace = True)
    sorted_input = sorted_input[['Date','Region','Location','Commodity','Mining_Asset',	'Lifecycle_Phase']]
    sc = StandardScaler()
    scaled_input = sc.fit_transform(sorted_input)
    scaled_input = pd.DataFrame(scaled_input)
    scaled_input_test_last14 = scaled_input.iloc[-14:]

    full_data_test = pd.concat((scaled_input_test_last14, scaled_input), axis = 0)
    full_data_test = sc.transform(full_data_test)

    hops = 14
    no_records = 77
    no_cols = 6
    X_test = []
    for i in range(14, 77):
        X_test.append(full_data_test[i-14:i])
    X_test = np.array(X_test)
 # Prediction
    y_test = model.predict(X_test)

    new_predictions = sc2.inverse_transform(y_test)
    sorted_input['Fluoride_level Predicted'] = new_predictions
    filename = sorted_input.to_csv(r'Compare_fluoride.csv', sep='\t', header='true')
    # Response - return csv file
    return send_file(filename)

if __name__ == "__main__":
    print("Serving Initializing")
    init()
    print(f'{os.environ["greetingmessage"]}')
    print("Serving Started")
    app.run(host="0.0.0.0", debug=True, port=9001)
