from lightweight_unet.model import Models
from lightweight_unet.dataloader import DataLoader
from lightweight_unet.saver import Saver
from lightweight_unet.trainer import Trainer
from lightweight_unet.utils import find_distillation_teacher_path

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Runner(object):

    def __init__(self, model_cfg = None, trainer_cfg = None):

        self.model_cfg = model_cfg
        self.trainer_cfg = trainer_cfg

        self.model_manager = Models(model_cfg = self.model_cfg)
        self.saver = Saver(trainer_cfg = self.trainer_cfg, model_cfg = self.model_cfg)
        self.dataloader = DataLoader(trainer_cfg= self.trainer_cfg)
        self.trainer = Trainer(trainer_cfg = self.trainer_cfg, dataloader= self.dataloader, saver= self.saver, model_manager= self.model_manager)

    def do_experiment(self):

        if self.trainer.trainer_cfg['time']:

            # This conditional branch times models

            os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # Set to "" to measure times on CPU
            self.dataloader.prepare_data(eval_fold='Test')
            self.trainer.time_model()

        else:

            # Model training happens in this conditional branch

            self.saver.init_experiment(cross_val=self.trainer_cfg['cross_val'])

            for fold in self.saver.sub_dirs:    # Adjust as required, e.g self.saver.sub_dirs[0:2]

                print("Parameter value is " + str(param_value) + " in range " + str(param_range))
                self.saver.set_current_sub_dir(fold)
                self.dataloader.prepare_data(eval_fold = fold)

                if self.trainer_cfg['mode'] in ['original_unet', 'thin_unet']:
                    self.trainer.regular_train()
                elif self.trainer_cfg['mode'] == 'distillation':

                    teacher_experiment_folder = find_distillation_teacher_path(self)[:-8]
                    self.model_manager.teacher_weight_path = os.path.join(teacher_experiment_folder, fold, "u-net.hdf5")

                    self.trainer.distillation_train()
                else:
                    # TODO: Insert exception here
                    pass

                self.saver.save(self.trainer)

            # End experiment
            self.saver.main_csv.to_csv(self.saver.main_csv_path, encoding='utf-8', index=False)

            print("Done!")

    @staticmethod
    def get_default_model_cfg():

        model_cfg = {
            'input_dims': (128, 256),
            'filter_size': 3,
            'l2_reg': 0.0005,
            'dropout_lowest': 0.5,
            'dropout_input': 0.2,
            'num_filters_thin': 32,
            'num_blocks': 6,
            'separable': False
        }

        return model_cfg

    @staticmethod
    def get_default_trainer_cfg():

        trainer_cfg = {
            'batch_size': 32,
            'total_epochs': 1,
            'initial_lrate': 0.1,
            'epochs_before_drop': 15,
            'epochs_per_warm_start_run': 40,
            'dice_threshold': 0.5,
            'distill_loss_weight': 10,
            'boundary_regularizer': False,

            'mode': 'distillation', # Should be one of ['distillation', 'thin_unet', 'original_unet']
            'cross_val': False, # Set cross_val = True for evaluation on one of the validation folds. If False, evaluation
                               # is performed on the test set

            # TODO: Set path for input data
            'input_data_path': "~/lightweight_unet/data/numpy_data.npz",

            # TODO: Set path to store experiment results
            'experiment_data_root': '~/lightweight_unet/results',

            # TODO: For distillation experiments, set root directory for results containing teacher models
            'teacher_root_folder': '~/lightweight_unet/results',

            # TODO: Sub-directories (under teacher_root_folder) containing teacher models for distillation experiments
            'teacher_folders': ["Experiment 4"],

            'time': False       # Set to True to time models
        }

        return trainer_cfg

if __name__ == "__main__":

    model_cfg = Runner.get_default_model_cfg()
    trainer_cfg = Runner.get_default_trainer_cfg()

    trainer_cfg['mode'] = 'thin_unet'  # Should be one of ['distillation', 'thin_unet', 'original_unet', 'time']

    param_to_vary = 'num_blocks'
    param_range = [6]

    assert param_to_vary in [*model_cfg.keys(), *trainer_cfg.keys()], \
        "param_to_vary must correspond to a key in cfg dictionaries"

    for param_value in param_range:

        model_cfg[param_to_vary] = param_value
        trainer_cfg['parameter_varied'] = param_to_vary

        runner = Runner(model_cfg = model_cfg, trainer_cfg = trainer_cfg)
        runner.do_experiment()
