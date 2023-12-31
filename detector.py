import json
import logging
import os
import pickle
from os import listdir, makedirs
from os.path import join, exists, basename

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from statistics import mean
from autoencoder import learn, predict
from torch import DoubleTensor
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from utils.abstract import AbstractDetector
from utils.flatten import flatten_model, flatten_models
from utils.healthchecks import check_models_consistency
from utils.models import create_layer_map, load_model, \
    load_models_dirpath
from utils.padding import create_models_padding, pad_model
from utils.reduction import (
    fit_feature_reduction_algorithm,
    use_feature_reduction_algorithm,
)
from torch.nn.functional import pairwise_distance

class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
            scale_parameters_filepath: str - File path to the scale_parameters file.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.model_filepath = join(self.learned_parameters_dirpath, "model.bin")
        self.models_padding_dict_filepath = join(self.learned_parameters_dirpath, "models_padding_dict.bin")
        self.model_layer_map_filepath = join(self.learned_parameters_dirpath, "model_layer_map.bin")
        self.layer_transform_filepath = join(self.learned_parameters_dirpath, "layer_transform.bin")

        # TODO: Update skew parameters per round
        self.model_skew = {
            "__all__": metaparameters["infer_cyber_model_skew"],
        }

        self.input_features = metaparameters["train_input_features"]
        self.weight_table_params = {
            "random_seed": metaparameters["train_weight_table_random_state"],
            "mean": metaparameters["train_weight_table_params_mean"],
            "std": metaparameters["train_weight_table_params_std"],
            "scaler": metaparameters["train_weight_table_params_scaler"],
        }
        self.autoencoder_kwargs = {
            "dist_ave": metaparameters[
                "autoencoder_clean_model_average_threshold"
            ],
            "dist_max": metaparameters[
                "autoencoder_clean_model_max_threshold"
            ],
        }
        self.random_forest_kwargs = {
            "n_estimators": metaparameters[
                "train_random_forest_regressor_param_n_estimators"
            ],
            "criterion": metaparameters[
                "train_random_forest_regressor_param_criterion"
            ],
            "max_depth": metaparameters[
                "train_random_forest_regressor_param_max_depth"
            ],
            "min_samples_split": metaparameters[
                "train_random_forest_regressor_param_min_samples_split"
            ],
            "min_samples_leaf": metaparameters[
                "train_random_forest_regressor_param_min_samples_leaf"
            ],
            "min_weight_fraction_leaf": metaparameters[
                "train_random_forest_regressor_param_min_weight_fraction_leaf"
            ],
            "max_features": metaparameters[
                "train_random_forest_regressor_param_max_features"
            ],
            "min_impurity_decrease": metaparameters[
                "train_random_forest_regressor_param_min_impurity_decrease"
            ],
        }

    def write_metaparameters(self):
        metaparameters = {
            "infer_cyber_model_skew": self.model_skew["__all__"],
            "train_input_features": self.input_features,
            "train_weight_table_random_state": self.weight_table_params["random_seed"],
            "train_weight_table_params_mean": self.weight_table_params["mean"],
            "train_weight_table_params_std": self.weight_table_params["std"],
            "train_weight_table_params_scaler": self.weight_table_params["scaler"],
            "train_random_forest_regressor_param_n_estimators": self.random_forest_kwargs["n_estimators"],
            "train_random_forest_regressor_param_criterion": self.random_forest_kwargs["criterion"],
            "train_random_forest_regressor_param_max_depth": self.random_forest_kwargs["max_depth"],
            "train_random_forest_regressor_param_min_samples_split": self.random_forest_kwargs["min_samples_split"],
            "train_random_forest_regressor_param_min_samples_leaf": self.random_forest_kwargs["min_samples_leaf"],
            "train_random_forest_regressor_param_min_weight_fraction_leaf": self.random_forest_kwargs["min_weight_fraction_leaf"],
            "train_random_forest_regressor_param_max_features": self.random_forest_kwargs["max_features"],
            "train_random_forest_regressor_param_min_impurity_decrease": self.random_forest_kwargs["min_impurity_decrease"],
            "autoencoder_clean_model_average_threshold": self.autoencoder_kwargs["dist_ave"],
            "autoencoder_clean_model_max_threshold": self.autoencoder_kwargs["dist_max"],
        }

        with open(join(self.learned_parameters_dirpath, basename(self.metaparameter_filepath)), "w") as fp:
            json.dump(metaparameters, fp)

    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        for random_seed in np.random.randint(1000, 9999, 10):
            self.weight_table_params["random_seed"] = random_seed
            self.manual_configure(models_dirpath)

    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        # Create the learned parameter folder if needed
        if not exists(self.learned_parameters_dirpath):
            makedirs(self.learned_parameters_dirpath)

        # List all available model
        model_path_list = sorted([join(models_dirpath, model) for model in listdir(models_dirpath)])
        logging.info(f"Loading %d models...", len(model_path_list))

        model_repr_dict, model_ground_truth_dict = load_models_dirpath(model_path_list)

        models_padding_dict = create_models_padding(model_repr_dict)
        with open(self.models_padding_dict_filepath, "wb") as fp:
            pickle.dump(models_padding_dict, fp)

        for model_class, model_repr_list in model_repr_dict.items():
            for index, model_repr in enumerate(model_repr_list):
                model_repr_dict[model_class][index] = pad_model(model_repr, model_class, models_padding_dict)

        check_models_consistency(model_repr_dict)
      
        # Build model layer map to know how to flatten
        logging.info("Generating model layer map...")
        model_layer_map = create_layer_map(model_repr_dict)
        with open(self.model_layer_map_filepath, "wb") as fp:
            pickle.dump(model_layer_map, fp)
        logging.info("Generated model layer map. Flattenning models...")

        # Flatten models
        flat_models = flatten_models(model_repr_dict, model_layer_map)
        del model_repr_dict
        logging.info("Models flattened. Fitting feature reduction...")

        layer_transform = fit_feature_reduction_algorithm(flat_models, self.weight_table_params, self.input_features)

        with open(self.layer_transform_filepath, "wb") as fp:
            pickle.dump(layer_transform, fp)

        logging.info("Feature reduction applied. Creating feature file...")
        X = None
        y = []

        for _ in range(len(flat_models)):
            (model_arch, models) = flat_models.popitem()
            model_index = 0

            logging.info("Parsing %s models...", model_arch)
            for _ in tqdm(range(len(models))):
                model = models.pop(0)
                y.append(model_ground_truth_dict[model_arch][model_index])
                model_index += 1

                model_feats = use_feature_reduction_algorithm(
                    layer_transform[model_arch], model
                )
                if X is None:
                    X = model_feats
                    continue

                X = np.vstack((X, model_feats * self.model_skew["__all__"]))

        logging.info("Training Autoencoder model...")
        X_train = DoubleTensor(X)
        train_dataset = TensorDataset(X_train)
        m_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        model = learn(m_dataloader, X.shape[1], device=torch.device('cuda'))

        dist = []
        for r in range(X.shape[0]):
            y_true, y_pred = predict(model, X[r,:], device=torch.device('cuda'))
            dist.append(pairwise_distance(y_true, y_pred).item())
        dist_ave = mean(dist)
        dist_max = max(dist)
        print(f'MAE = {dist_ave:.5f}')
        print(f'MAE = {dist_max:.5f}')

        logging.info("Saving Autoencoder model...")
        with open(self.model_filepath, "wb") as fp:
            pickle.dump(model, fp)

        self.autoencoder_kwargs["dist_ave"] = dist_ave
        self.autoencoder_kwargs["dist_max"] = dist_max
        self.write_metaparameters()
        logging.info("Configuration done!")


    def inference_on_example_data(self, model, examples_dirpath):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model: the pytorch model
            examples_dirpath: the directory path for the round example data
        """
        inputs_np = None
        g_truths = []

        for examples_dir_entry in os.scandir(examples_dirpath):
            if examples_dir_entry.is_file() and examples_dir_entry.name.endswith(".npy"):
                base_example_name = os.path.splitext(examples_dir_entry.name)[0]
                ground_truth_filename = os.path.join(examples_dirpath, '{}.json'.format(base_example_name))
                if not os.path.exists(ground_truth_filename):
                    logging.warning('ground truth file not found ({}) for example {}'.format(ground_truth_filename, base_example_name))
                    continue

                new_input = np.load(examples_dir_entry.path)

                if inputs_np is None:
                    inputs_np = new_input
                else:
                    inputs_np = np.concatenate([inputs_np, new_input])

                with open(ground_truth_filename) as f:
                    data = int(json.load(f))

                g_truths.append(data)

        g_truths_np = np.asarray(g_truths)

        p = model.predict(inputs_np)

        orig_test_acc = accuracy_score(g_truths_np, np.argmax(p.detach().numpy(), axis=1))
        print("Model accuracy on example data {}: {}".format(examples_dirpath, orig_test_acc))


    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        """Method to predict whether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """

        with open(self.model_layer_map_filepath, "rb") as fp:
            model_layer_map = pickle.load(fp)

        with open(self.models_padding_dict_filepath, "rb") as fp:
            models_padding_dict = pickle.load(fp)

        with open(self.layer_transform_filepath, "rb") as fp:
            layer_transform = pickle.load(fp)

        model, model_repr, model_class = load_model(model_filepath)
        model_repr = pad_model(model_repr, model_class, models_padding_dict)
        flat_model = flatten_model(model_repr, model_layer_map[model_class])

        # Inferences on examples to demonstrate how it is done for a round
        # This is not needed for the random forest classifier
        self.inference_on_example_data(model, examples_dirpath)

        X = (
            use_feature_reduction_algorithm(layer_transform[model_class], flat_model)
            * self.model_skew["__all__"]
        )

        try:
            with open(self.model_filepath, "rb") as fp:
                regressor: AE = pickle.load(fp)

            y_true, y_pred = predict(regressor, X[0], device=torch.device('cuda'))
            dist = pairwise_distance(y_true, y_pred)
            if dist > self.autoencoder_kwargs["dist_max"]:
                probability = str(1.0)
            else:
                probability = str(0.0)
        except Exception as e:
            logging.info('Failed to run regressor, there may have an issue during fitting, using random for trojan probability: {}'.format(e))
            probability = str(np.random.rand())
        with open(result_filepath, "w") as fp:
            fp.write(probability)

        logging.info("Trojan probability: %s", probability)
