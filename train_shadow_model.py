from os.path import join
import numpy as np
import torch
from tqdm import tqdm
import json
import os
from utils.drebinnn import DrebinNN

# load the model configuration
model_filepath = "./model/id-00000001/model.pt"
conf_filepath = os.path.join(os.path.dirname(model_filepath), 'config.json')
with open(conf_filepath, 'r') as f:
    full_conf = json.load(f)

model = DrebinNN(991, full_conf)

# load the vectorized training drebin data
drebin_dirpath = "./drebin/cyber-apk-nov2023-vectorized-drebin/"
x_sel = np.load(join(drebin_dirpath, "x_train_sel.npy"))
y_sel = np.load(join(drebin_dirpath, "y_train_sel.npy"))

model_id = 1
for i in range(0, len(y_sel)-2000, 800):
    net = model.fit(x_sel[i:i+2000], y_sel[i:i+2000])

    # save the shadow models
    model_id += 1
    if model_id < 10:
        id_str = "0" + str(model_id)
    else:
        id_str = str(model_id)
    save_path = "./cyber-apk-nov2023/models/id-000000" + id_str + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model.save(save_path, file_name='model', config=None)

    # inference the model prediction
    x_test_sel = np.load(join(drebin_dirpath, "x_test_sel.npy"))
    y_test_sel = np.load(join(drebin_dirpath, "y_test_sel.npy"))
    y_pred = model.predict(x_test_sel)
    y_pred = np.argmax(y_pred.detach().numpy(), axis=1)

    prediction_acc = np.mean(np.equal(y_pred,y_test_sel))
    print(prediction_acc)