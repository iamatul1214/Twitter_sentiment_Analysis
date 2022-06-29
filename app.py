from flask import Flask, render_template, request, jsonify
import os
import yaml
import joblib
import numpy as np
import tensorflow as tf
import logging


webapp_root="webapp"
config_file_path = "configs/config.yaml"

static_dir=os.path.join(webapp_root,"static")
template_dir=os.path.join(webapp_root,"templates")

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


app=Flask(__name__,static_folder=static_dir, template_folder=template_dir)

def read_yaml(config_path=config_file_path):
    logging.info("Reading the configuration file.")
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def predict(text):
    logging.info("Trying to predict the sentiment for the text : {text}")
    config = read_yaml(config_file_path)
    ckpt_dir = config['artifacts']['CHECKPOINT_DIR']
    trained_model_dir = config['artifacts']['TRAINED_MODEL_DIR']
    path_to_trained_model = os.path.join(ckpt_dir,trained_model_dir)
    trained_model = tf.keras.models.load_model(path_to_trained_model, compile=False)
    predictions = trained_model.predict(np.array([text]))
    score = predictions[0][0]
    if score > 0:
        logging.info(f" Positive Sentiment calculated successfully for text : {text}")
        return "positive sentiment"
    else:
        logging.info(f" Negative Sentiment calculated successfully for text : {text}")
        return "negative sentiment"
    

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_sentiment():
    try:
        if request.method=='POST':
                   
            tweet = request.form.get('TWEET')
            response = predict(np.array([tweet]))

            return render_template('index.html',response=response)

    except Exception as e:
        print(e)
        # error={"error":"Something went wrong!! Try again"}
        error={"error":e}
        return render_template("404.html", error=error)
    
        

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
