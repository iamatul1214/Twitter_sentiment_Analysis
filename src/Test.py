## This file is for testing purposes only.
import os
import tensorflow as tf
import numpy as np
root = "checkpoints\model"
# os.chdir(root)
path_to_trained_model = os.path.join("checkpoints","model")
trained_model = tf.keras.models.load_model(path_to_trained_model)
# trained_model = tf.saved_model.load("Saved_model.pb")
tweet = ("My day was horrible!")
predictions = trained_model.predict(np.array([tweet]))
score=predictions[0][0]
print(f"score = {score}")