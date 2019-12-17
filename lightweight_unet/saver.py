import os
import pandas as pd
import numpy as np

class Saver(object):

    def __init__(self, trainer_cfg = None, model_cfg = None):

        self.project_dir = trainer_cfg['experiment_data_root']
        self.experiment_dir = None
        self.experiment_name = None
        self.sub_dirs = None
        self.current_sub_dir = None

        self.exp_log = None
        self.main_csv = None
        self.main_csv_path = None

        self.model_cfg = model_cfg
        self.trainer_cfg = trainer_cfg

    def init_experiment(self, cross_val = True):

        # Create or open highest level csv which stores results of all experiments
        self.main_csv_path = os.path.join(self.project_dir, "Results.csv")
        if os.path.isfile(self.main_csv_path):
            self.main_csv = pd.read_csv(self.main_csv_path, delimiter = ',')
            self.main_csv = self.main_csv.append(pd.Series(), ignore_index=True)
        else:
            self.main_csv = pd.DataFrame(columns = [
                "Name",
                "Param Varied",
                "Param Value",
                "Fold 1",
                "Fold 2",
                "Fold 3",
                "Fold 4",
                "Fold 5",
                "Test",
                "Mean",
                "Std"
            ])
            self.main_csv = self.main_csv.append(pd.Series(), ignore_index=True)

        # Create directory for specific experiment
        exps = [i for i in os.listdir(self.project_dir) if 'Experiment' in i]
        exp_nums = [int(i[11:]) for i in exps]

        if not exp_nums:
            experiment_num = 0
        else:
            experiment_num = np.max(exp_nums) + 1

        if os.path.isdir("Experiment " + str(experiment_num)):
            self.experiment_name = "Experiment " + str(experiment_num) + "_" + str(os.getpgid())
        else:
            self.experiment_name = "Experiment " + str(experiment_num)

        self.experiment_dir = os.path.join(self.project_dir, self.experiment_name)
        os.mkdir(self.experiment_dir)

        # Create log file to store experiment parameters
        log = open(os.path.join(self.experiment_dir, "log.txt"), "w")

        for key, value in {**self.model_cfg, **self.trainer_cfg}.items():

            if key == 'parameter_varied':
                key2 = {**self.model_cfg, **self.trainer_cfg}[key]
                log.write(str(key) + ' >>> ' + str(value) + ' (' + str({**self.model_cfg, **self.trainer_cfg}[key2]) +
                          ')' + '\n\n')
            else:
                log.write(str(key) + ' >>> ' + str(value) + '\n\n')

        log.close()

        # Create experiment sub directories
        if cross_val:
            self.sub_dirs = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
        else:
            self.sub_dirs = ["Test"]

        for dir in self.sub_dirs:
            path = os.path.join(self.experiment_dir, dir)
            os.mkdir(path)

    def save(self, trainer):

        df1 = pd.DataFrame()
        # Add training histories
        for loss_name, loss_hist in trainer.history.history.items():

            df1[loss_name] = loss_hist

        # Add dice coefficient on validation set
        val_data = trainer.dataloader.get_data()["eval"]
        val_images = val_data['imgs']
        val_labels = val_data['labels']

        if self.trainer_cfg['mode'] == 'distillation':
            predictions = trainer.model.predict(val_images, batch_size = self.trainer_cfg['batch_size'])[-1]
        else:
            predictions = trainer.model.predict(val_images, batch_size=self.trainer_cfg['batch_size'])

        val_dice = self.evaluate_dice_loss(predictions, val_labels)
        df2 = pd.DataFrame({"Dice Coefficient" : val_dice})
        df_final = pd.concat([df1, df2], ignore_index=True, axis = 1)

        # Save dataframe to csv
        df_path = os.path.join(self.current_sub_dir, "Training and Evaluation Info.csv")
        df_final.columns = [*df1.columns.tolist(), *df2.columns.tolist()]
        df_final.to_csv(df_path, encoding='utf-8', index=False)

        # Save results in main csv file
        self.main_csv["Name"].iloc[-1] = self.experiment_name

        self.main_csv[self.current_sub_dir.split(os.sep)[-1]].iloc[-1] = np.mean(val_dice)
        self.main_csv["Mean"].iloc[-1] = np.mean(self.main_csv.iloc[-1, 3:9])
        self.main_csv["Std"].iloc[-1] = np.std(self.main_csv.iloc[-1, 3:9])

        param_varied = self.trainer_cfg['parameter_varied']
        self.main_csv["Param Varied"].iloc[-1] = param_varied

        if param_varied in self.trainer_cfg.keys():
            self.main_csv["Param Value"].iloc[-1] = self.trainer_cfg[param_varied]
        else:
            self.main_csv["Param Value"].iloc[-1] = trainer.model_manager.model_cfg[param_varied]

    def set_current_sub_dir(self, fold):

        assert fold in self.sub_dirs, \
            "Fold should be one of" + str(self.sub_dirs)

        self.current_sub_dir = os.path.join(self.experiment_dir, fold)

    def evaluate_dice_loss(self, preds, labels):

        dice_coef = []

        # Threshold Dice coefficient
        preds[preds >= self.trainer_cfg['dice_threshold']] = 1
        preds[preds < self.trainer_cfg['dice_threshold']] = 0

        smoothing = 1e-4
        num_val_examples = np.shape(preds)[0]

        # Compute Dice for all examples
        for i in np.arange(num_val_examples):

            pred = np.ndarray.flatten(preds[i, :, :, :])
            true = np.ndarray.flatten(labels[i, :, :, :])

            intersection = np.sum(pred * true)
            dice_coef.append((2 * intersection + smoothing) / (np.sum(true) + np.sum(pred) + smoothing))

        return dice_coef