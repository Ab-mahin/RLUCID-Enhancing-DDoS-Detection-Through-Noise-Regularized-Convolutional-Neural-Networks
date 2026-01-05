# Training (standard):
#   python3 lucid_cnn.py --train ./sample-dataset/ --epochs 100 -cv 5
# Training with adversarial/noise augmentation:
#   python3 lucid_cnn.py --train ./sample-dataset/ --epochs 100 -cv 5 --adv_training
# Testing:
#   python3 lucid_cnn.py --predict ./sample-dataset/ --model ./output/10t-10n-SYN2020-LUCID.h5

import os
import sys
import csv
import glob
import time
import random as rn
import argparse
import pprint

import numpy as np
import tensorflow as tf

from util_functions import *
from rlucid_dataset_parser import *

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D
from tensorflow.keras.layers import Dropout, GlobalMaxPooling2D
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.layers import Input



import tensorflow.keras.backend as K

# ---- seeds and TF config ----
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
rn.seed(SEED)
tf.random.set_seed(SEED)
config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1)
config.gpu_options.allow_growth = True
K.set_image_data_format("channels_last")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

OUTPUT_FOLDER = "./output/"

VAL_HEADER = ["Model", "Samples", "Accuracy", "F1Score", "Hyper-parameters", "Validation Set"]
PREDICT_HEADER = ["Model", "Time", "Packets", "Samples", "DDOS%", "Accuracy",
                  "F1Score", "TPR", "FPR", "TNR", "FNR", "Source"]

PATIENCE = 10
DEFAULT_EPOCHS = 1000

# Grid over model hyperparameters (note the model__ prefix for scikeras)
hyperparamters = {
    "model__learning_rate": [0.1, 0.01],
    "model__kernels": [32, 64],
    "model__regularization": [None, "l1"],
    "model__dropout": [None, 0.2],
}

# ---- helper to ensure clean arrays ----
def clean_array(arr):
    arr = np.asarray(arr, dtype="float32")
    arr = np.where(arr == None, np.nan, arr)
    return np.nan_to_num(arr)



# ---- adversarial / noise augmentation ----
def add_feature_noise(X, epsilon=0.02, cols=(2, 3)):
    """
    Simple noise/adversarial augmentation.
    X: normalized samples in [0,1], shape (batch, n, f, 1).
    cols: feature indices to perturb (e.g., Highest Layer, IP Flags).
    """
    X_noisy = X.copy()
    if not cols:
        return X_noisy
    noise_shape = X_noisy[:, :, cols, 0].shape
    noise = epsilon * np.sign(np.random.randn(*noise_shape))
    X_noisy[:, :, cols, 0] = np.clip(
        X_noisy[:, :, cols, 0] + noise,
        0.0,
        1.0,
    )
    return X_noisy


# ---- model builder for scikeras ----
# def Conv2DModel(
#     *,
#     model_name,
#     input_shape,
#     kernel_col,
#     kernels=64,
#     kernel_rows=3,
#     learning_rate=0.01,
#     regularization=None,
#     dropout=None,
#     **kwargs,
# ):
#     """
#     Keyword-only signature so scikeras passes hyperparameters by name.
#     Extra keys from GridSearchCV are absorbed in **kwargs and ignored.
#     """
#     K.clear_session()

#     model = Sequential(name=model_name)
#     regularizer = regularization

#     model.add(
#         Conv2D(
#             kernels,
#             (kernel_rows, kernel_col),
#             strides=(1, 1),
#             input_shape=input_shape,
#             kernel_regularizer=regularizer,
#             name="conv0",
#         )
#     )
#     if dropout is not None and isinstance(dropout, float):
#         model.add(Dropout(dropout))
#     model.add(Activation("relu"))

#     model.add(GlobalMaxPooling2D())
#     model.add(Flatten())
#     model.add(Dense(1, activation="sigmoid", name="fc1"))

#     optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
#     model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
#     return model

def Conv2DModel(
    *,
    model_name,
    input_shape,
    kernel_col,
    kernels=64,
    kernel_rows=3,
    learning_rate=0.01,
    regularization=None,
    dropout=None,
    **kwargs,
):
    K.clear_session()

    model = Sequential(name=model_name)
    regularizer = regularization

    # New Input layer
    model.add(Input(shape=input_shape, name="input"))

    model.add(
        Conv2D(
            kernels,
            (kernel_rows, kernel_col),
            strides=(1, 1),
            kernel_regularizer=regularizer,
            name="conv0",
        )
    )
    if dropout is not None and isinstance(dropout, float):
        model.add(Dropout(dropout))
    model.add(Activation("relu"))

    model.add(GlobalMaxPooling2D())
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid", name="fc1"))

    optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

# ---- main ----
def main(argv):
    parser = argparse.ArgumentParser(
        description="DDoS attacks detection with convolutional neural networks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-t", "--train", nargs="+", type=str, help="Start the training process")
    parser.add_argument("-e", "--epochs", default=DEFAULT_EPOCHS, type=int, help="Training iterations")
    parser.add_argument("-cv", "--cross_validation", default=0, type=int,
                        help="Number of folds for cross-validation (default 0)")

    parser.add_argument("-a", "--attack_net", default=None, type=str,
                        help="Subnet of the attacker (used to compute the detection accuracy)")
    parser.add_argument("-v", "--victim_net", default=None, type=str,
                        help="Subnet of the victim (used to compute the detection accuracy)")

    parser.add_argument("-p", "--predict", nargs="?", type=str,
                        help="Perform a prediction on pre-preprocessed data")
    parser.add_argument("-pl", "--predict_live", nargs="?", type=str,
                        help="Perform a prediction on live traffic")
    parser.add_argument("-i", "--iterations", default=1, type=int, help="Predict iterations")
    parser.add_argument("-m", "--model", type=str, help="File containing the model")
    parser.add_argument("-y", "--dataset_type", default=None, type=str,
                        help="Type of the dataset. DOS2017, DOS2018, DOS2019, SYN2020")

    parser.add_argument("--adv_training", action="store_true",
                        help="Enable simple adversarial/noise augmentation during training")

    args = parser.parse_args()

    if not os.path.isdir(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)

    # ------------- TRAINING -------------
    if args.train is not None:
        subfolders = glob.glob(args.train[0] + "/*/")
        if len(subfolders) == 0:
            subfolders = [args.train[0] + "/"]
        else:
            subfolders = sorted(subfolders)

        for full_path in subfolders:
            full_path = full_path.replace("//", "/")
            dataset_folder = full_path
            X_train, Y_train = load_dataset(dataset_folder + "/*" + "-train.hdf5")
            X_val, Y_val = load_dataset(dataset_folder + "/*" + "-val.hdf5")

            X_train, Y_train = shuffle(X_train, Y_train, random_state=SEED)
            X_val, Y_val = shuffle(X_val, Y_val, random_state=SEED)

            train_file = glob.glob(dataset_folder + "/*" + "-train.hdf5")[0]
            filename = train_file.split("/")[-1].strip()
            time_window = int(filename.split("-")[0].strip().replace("t", ""))
            max_flow_len = int(filename.split("-")[1].strip().replace("n", ""))
            dataset_name = filename.split("-")[2].strip()

            print("\nCurrent dataset folder:", dataset_folder)

            model_name = dataset_name + "-LUCID"
            # model_name = dataset_name + "-RLUCID"
            input_shape = X_train.shape[1:]

            keras_classifier = KerasClassifier(
                model=Conv2DModel,
                model__model_name=model_name,
                model__input_shape=input_shape,
                model__kernel_col=X_train.shape[2],
                verbose=0,
            )

            rnd_search_cv = GridSearchCV(
                keras_classifier,
                hyperparamters,
                cv=args.cross_validation if args.cross_validation > 1 else [(slice(None), slice(None))],
                refit=True,
                return_train_score=True,
                error_score="raise",
            )

            es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=PATIENCE)
            best_model_filename = OUTPUT_FOLDER + f"{time_window}t-{max_flow_len}n-{model_name}"
            mc = ModelCheckpoint(
                best_model_filename + ".h5",
                monitor="val_accuracy",
                mode="max",
                verbose=1,
                save_best_only=True,
            )

            # adversarial / noise augmentation
            X_train_aug = X_train
            Y_train_aug = Y_train
            if args.adv_training:
                print("Adversarial/noise augmentation enabled.")
                X_noisy = add_feature_noise(X_train)
                X_train_aug = np.concatenate([X_train, X_noisy], axis=0)
                Y_train_aug = np.concatenate([Y_train, Y_train], axis=0)

            X_train_aug = clean_array(X_train_aug).astype("float32")
            Y_train_aug = clean_array(Y_train_aug).astype("float32")
            X_val_c = clean_array(X_val).astype("float32")
            Y_val_c = clean_array(Y_val).astype("float32")

            rnd_search_cv.fit(
                X_train_aug,
                Y_train_aug,
                epochs=args.epochs,
                validation_data=(X_val_c, Y_val_c),
                callbacks=[es, mc],
                batch_size=2048,
            )

            # best_model = rnd_search_cv.best_estimator_.model
            # best_model.save(best_model_filename + ".h5")
            best_est = rnd_search_cv.best_estimator_
            best_model = best_est.model_          # scikeras stores the fitted Keras model here
            best_model.save(best_model_filename + ".h5")


            Y_pred_val = (best_model.predict(X_val_c, batch_size=2048) > 0.5)
            Y_true_val = Y_val_c.reshape((Y_val_c.shape[0], 1))
            f1_score_val = f1_score(Y_true_val, Y_pred_val)
            accuracy = accuracy_score(Y_true_val, Y_pred_val)

            with open(best_model_filename + ".csv", "w", newline="") as val_file:
                val_writer = csv.DictWriter(val_file, fieldnames=VAL_HEADER)
                val_writer.writeheader()
                row = {
                    "Model": model_name,
                    "Samples": Y_pred_val.shape[0],
                    "Accuracy": f"{accuracy:05.4f}",
                    "F1Score": f"{f1_score_val:05.4f}",
                    "Hyper-parameters": rnd_search_cv.best_params_,
                    "Validation Set": glob.glob(dataset_folder + "/*" + "-val.hdf5")[0],
                }
                val_writer.writerow(row)

            print("Best parameters:", rnd_search_cv.best_params_)
            print("Best model path:", best_model_filename)
            print("F1 Score of the best model on the validation set:", f1_score_val)

    # ------------- OFFLINE PREDICT -------------
    if args.predict is not None:
        predict_file = open(
            OUTPUT_FOLDER + "predictions-" + time.strftime("%Y%m%d-%H%M%S") + ".csv",
            "w",
            newline="",
        )
        predict_writer = csv.DictWriter(predict_file, fieldnames=PREDICT_HEADER)
        predict_writer.writeheader()
        predict_file.flush()

        iterations = args.iterations
        dataset_filelist = glob.glob(args.predict + "/*test.hdf5")

        if args.model is not None:
            model_list = [args.model]
        else:
            model_list = glob.glob(args.predict + "/*.h5")

        for model_path in model_list:
            model_filename = model_path.split("/")[-1].strip()
            filename_prefix = model_filename.split("-")[0].strip() + "-" + model_filename.split("-")[1].strip() + "-"
            model_name_string = model_filename.split(filename_prefix)[1].strip().split(".")[0].strip()
            model = load_model(model_path)

            warm_up_file = dataset_filelist[0]
            filename = warm_up_file.split("/")[-1].strip()
            if filename_prefix in filename:
                X, Y = load_dataset(warm_up_file)
                _ = np.squeeze(model.predict(X, batch_size=2048) > 0.5)

            for dataset_file in dataset_filelist:
                filename = dataset_file.split("/")[-1].strip()
                if filename_prefix in filename:
                    X, Y = load_dataset(dataset_file)
                    [packets] = count_packets_in_dataset([X])

                    Y_true = Y
                    avg_time = 0
                    for _ in range(iterations):
                        pt0 = time.time()
                        Y_pred = np.squeeze(model.predict(X, batch_size=2048) > 0.5)
                        pt1 = time.time()
                        avg_time += pt1 - pt0

                    avg_time /= iterations
                    report_results(
                        np.squeeze(Y_true),
                        Y_pred,
                        packets,
                        model_name_string,
                        filename,
                        avg_time,
                        predict_writer,
                    )
                    predict_file.flush()

        predict_file.close()

    # ------------- LIVE PREDICT -------------
    if args.predict_live is not None:
        predict_file = open(
            OUTPUT_FOLDER + "predictions-" + time.strftime("%Y%m%d-%H%M%S") + ".csv",
            "w",
            newline="",
        )
        predict_writer = csv.DictWriter(predict_file, fieldnames=PREDICT_HEADER)
        predict_writer.writeheader()
        predict_file.flush()

        if args.predict_live.endswith(".pcap"):
            pcap_file = args.predict_live
            cap = pyshark.FileCapture(pcap_file)
            data_source = pcap_file.split("/")[-1].strip()
        else:
            cap = pyshark.LiveCapture(interface=args.predict_live)
            data_source = args.predict_live

        print("Prediction on network traffic from:", data_source)

        labels = parse_labels(args.dataset_type, args.attack_net, args.victim_net)

        if args.model is not None and args.model.endswith(".h5"):
            model_path = args.model
        else:
            print("No valid model specified!")
            sys.exit(-1)

        model_filename = model_path.split("/")[-1].strip()
        filename_prefix = model_filename.split("n")[0] + "n-"
        time_window = int(filename_prefix.split("t-")[0])
        max_flow_len = int(filename_prefix.split("t-")[1].split("n-")[0])
        model_name_string = model_filename.split(filename_prefix)[1].strip().split(".")[0].strip()
        model = load_model(args.model)

        mins, maxs = static_min_max(time_window)

        while True:
            samples = process_live_traffic(
                cap,
                args.dataset_type,
                labels,
                max_flow_len,
                traffic_type="all",
                time_window=time_window,
            )
            if len(samples) > 0:
                X, Y_true, keys = dataset_to_list_of_fragments(samples)
                X = np.array(normalize_and_padding(X, mins, maxs, max_flow_len))
                if labels is not None:
                    Y_true = np.array(Y_true)
                else:
                    Y_true = None

                X = np.expand_dims(X, axis=3)
                pt0 = time.time()
                Y_pred = np.squeeze(model.predict(X, batch_size=2048) > 0.5, axis=1)
                pt1 = time.time()
                prediction_time = pt1 - pt0

                [packets] = count_packets_in_dataset([X])
                report_results(
                    np.squeeze(Y_true),
                    Y_pred,
                    packets,
                    model_name_string,
                    data_source,
                    prediction_time,
                    predict_writer,
                )
                predict_file.flush()

            elif isinstance(cap, pyshark.FileCapture):
                print("\nNo more packets in file", data_source)
                break

        predict_file.close()


def report_results(Y_true, Y_pred, packets, model_name, data_source, prediction_time, writer):
    ddos_rate = "{:04.3f}".format(sum(Y_pred) / Y_pred.shape[0])

    if Y_true is not None and len(Y_true.shape) > 0:
        Y_true = Y_true.reshape((Y_true.shape[0], 1))
        accuracy = accuracy_score(Y_true, Y_pred)
        f1 = f1_score(Y_true, Y_pred)
        tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred, labels=[0, 1]).ravel()
        tnr = tn / (tn + fp)
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)
        tpr = tp / (tp + fn)

        row = {
            "Model": model_name,
            "Time": "{:04.3f}".format(prediction_time),
            "Packets": packets,
            "Samples": Y_pred.shape[0],
            "DDOS%": ddos_rate,
            "Accuracy": "{:05.4f}".format(accuracy),
            "F1Score": "{:05.4f}".format(f1),
            "TPR": "{:05.4f}".format(tpr),
            "FPR": "{:05.4f}".format(fpr),
            "TNR": "{:05.4f}".format(tnr),
            "FNR": "{:05.4f}".format(fnr),
            "Source": data_source,
        }
    else:
        row = {
            "Model": model_name,
            "Time": "{:04.3f}".format(prediction_time),
            "Packets": packets,
            "Samples": Y_pred.shape[0],
            "DDOS%": ddos_rate,
            "Accuracy": "N/A",
            "F1Score": "N/A",
            "TPR": "N/A",
            "FPR": "N/A",
            "TNR": "N/A",
            "FNR": "N/A",
            "Source": data_source,
        }
    pprint.pprint(row, sort_dicts=False)
    writer.writerow(row)


if __name__ == "__main__":
    main(sys.argv[1:])
