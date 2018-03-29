import pickle
import os

with open(os.sep.join(['CNN 3', 'model_config.ckpt.pkl']), 'rb') as file:
    p = pickle.load(file)
    # for i in range(800, 900):
    #     print(i, p['TESTING_ACCURACY'][i])
    j = 830
    print('Training:', p['TRAINING_ACCURACY'][j])
    print('Testing:', p['TESTING_ACCURACY'][j])
