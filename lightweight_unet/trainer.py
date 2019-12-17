from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import losses
from keras.utils import plot_model

from lightweight_unet.utils import *
import math
import os

class Trainer(object):

    def __init__(self, trainer_cfg = None, dataloader = None, saver = None, model_manager = None):

        self.model_manager = model_manager

        self.model = None
        self.history = None

        self.trainer_cfg = trainer_cfg
        self.dataloader = dataloader
        self.saver = saver

    def regular_train(self):

        # Initialise and compile model
        if self.trainer_cfg['mode'] == 'thin_unet':
            self.model = self.model_manager.get_thin_uNet(num_blocks = self.model_manager.model_cfg['num_blocks'])
        elif self.trainer_cfg['mode'] == 'original_unet':
            self.model = self.model_manager.get_original_uNet()
        else:
            # TODO: Insert except here
            pass

        optimizer = SGD(momentum=0.9, nesterov=True)
        scheduler = LearningRateScheduler(self.warm_start_step_decay)
        data = self.dataloader.get_data()

        if self.trainer_cfg['boundary_regularizer']:
            losses.my_loss = regularized_dice_loss
            self.model.compile(optimizer, loss = regularized_dice_loss, metrics = [regularized_dice_loss, dice_loss])
        else:
            self.model.compile(optimizer, loss=dice_loss, metrics=[dice_loss])


        # Define checkpoints
        model_fpath = os.path.join(self.saver.current_sub_dir, "u-net.hdf5")
        model_checkpoint = ModelCheckpoint(model_fpath, monitor = "val_dice_loss", verbose=1,
                                           save_best_only=True, save_weights_only=True)

        # Fit model
        print("Fitting model...")
        self.history = self.model.fit(data["train"]['imgs'], data["train"]['labels'],
                                    batch_size=self.trainer_cfg['batch_size'],
                                    epochs=self.trainer_cfg['total_epochs'],
                                    callbacks=[model_checkpoint, scheduler],
                                    validation_split=0.2)

        # End training and save some model info
        self.model.load_weights(model_fpath)
        with open(os.path.join(self.saver.current_sub_dir, "u-net.json"), 'w') as outfile:
            outfile.write(self.model.to_json())
        plot_model(self.model, to_file=os.path.join(self.saver.current_sub_dir, "u-net.png"), show_shapes=True)

    def distillation_train(self):

        self.model, teacher_model = self.model_manager.get_distillation_uNet(num_blocks = self.model_manager.model_cfg['num_blocks'])
        teacher_model.load_weights(self.model_manager.teacher_weight_path)

        data = self.dataloader.get_data()
        student_labels = teacher_model.predict(data["train"]["imgs"])[:-1]  # Labels for student's training contain intermediate activations of teacher
        student_labels.append(data["train"]["labels"])  # Labels for student's training also contain manual annotations ("labels_train")

        scheduler = LearningRateScheduler(self.warm_start_step_decay)

        # Compile model
        optimizer = SGD(momentum=0.9, nesterov=True)  # Define SGD optimizer

        if self.trainer_cfg['boundary_regularizer']:
            losses.my_loss = regularized_dice_loss

            loss_list, loss_weights = get_loss_list_and_weights(get_distillation_indices(self.model),
                                                                self.trainer_cfg['distill_loss_weight'],
                                                                self.model.layers, boundary_regularizer= self.trainer_cfg['boundary_regularizer'])

            self.model.compile(optimizer, loss=loss_list, loss_weights=loss_weights, metrics=[regularized_dice_loss, dice_loss, 'mse'])
        else:

            loss_list, loss_weights = get_loss_list_and_weights(get_distillation_indices(self.model),
                                                                self.trainer_cfg['distill_loss_weight'],
                                                                self.model.layers,
                                                                boundary_regularizer=self.trainer_cfg[
                                                                    'boundary_regularizer'])

            self.model.compile(optimizer, loss=loss_list, loss_weights=loss_weights,
                               metrics=[dice_loss, 'mse'])

        # Define checkpoints
        model_fpath = os.path.join(self.saver.current_sub_dir, "u-net.hdf5")
        model_checkpoint = ModelCheckpoint(model_fpath, monitor="val_out_dice_loss", verbose=1,
                                           save_best_only=True, save_weights_only=True)

        self.history = self.model.fit(data["train"]['imgs'], student_labels,
                                        batch_size=self.trainer_cfg['batch_size'],
                                        epochs=self.trainer_cfg['total_epochs'],
                                        callbacks=[model_checkpoint, scheduler],
                                        validation_split=0.2)

        # End training and save some model info
        self.model.load_weights(model_fpath)
        with open(os.path.join(self.saver.current_sub_dir, "u-net.json"), 'w') as outfile:
            outfile.write(self.model.to_json())
        #plot_model(self.model, to_file=os.path.join(self.saver.current_sub_dir, "u-net.png"), show_shapes=True)

    def time_model(self):

        import time

        print("Loading models and data to time...")
        optimizer = SGD(momentum=0.9, nesterov=True)
        data = self.dataloader.get_data()

        # Initialise and compile model
        if self.trainer_cfg['mode'] == 'thin_unet':

            self.model = self.model_manager.get_thin_uNet(num_blocks=self.model_manager.model_cfg['num_blocks'])
            self.model.compile(optimizer, loss=dice_loss, metrics=[dice_loss])

        elif self.trainer_cfg['mode'] == 'original_unet':

            self.model = self.model_manager.get_original_uNet()
            self.model.compile(optimizer, loss=dice_loss, metrics=[dice_loss])

        elif self.trainer_cfg['mode'] == 'distillation':

            self.model, teacher_model = self.model_manager.get_distillation_uNet(
                num_blocks=self.model_manager.model_cfg['num_blocks'])
            loss_list, loss_weights = get_loss_list_and_weights(get_distillation_indices(self.model),
                                                                self.trainer_cfg['distill_loss_weight'],
                                                                self.model.layers,
                                                                boundary_regularizer=self.trainer_cfg[
                                                                    'boundary_regularizer'])
            self.model.compile(optimizer, loss=loss_list, loss_weights=loss_weights,
                               metrics=[dice_loss, 'mse'])

        else:

            print(self.trainer_cfg['mode'])

        # Load model into memory
        dummy_preds = self.model.predict(data['eval']['imgs'][:5], batch_size = 1)

        t1 = time.time()
        preds = self.model.predict(data['eval']['imgs'][:10], batch_size = 1)
        t2 = time.time()

        num_ims = data['eval']['imgs'].shape[0]
        t_per_im = (t2 - t1)/10 # (t2 - t1)/num_ims

        cfg = {**self.model_manager.model_cfg, **self.trainer_cfg}

        if self.trainer_cfg['mode'] is not 'original_unet':
            param_varied = cfg['parameter_varied']
            param_value = cfg[param_varied]
        else:
            param_varied = '-'
            param_value = '-'

        print("Mode: {}| Param Varied: {}| Param Value: {}| Time Taken per Image: {}".format(self.trainer_cfg['mode'], param_varied, param_value, t_per_im))

    def warm_start_step_decay(self, epoch):
        initial_lrate = self.trainer_cfg['initial_lrate']
        epochs_drop = self.trainer_cfg['epochs_before_drop']
        epochs_per_run = self.trainer_cfg['epochs_per_warm_start_run']
        drop = 0.1

        epoch = epoch % epochs_per_run

        lrate = initial_lrate * math.pow(drop,
                                         math.floor((1 + epoch) / epochs_drop))

        return lrate

    def step_decay(self, epoch):
        initial_lrate = self.trainer_cfg['initial_lrate']
        epochs_drop = self.trainer_cfg['epochs_before_drop']
        drop = 0.1

        lrate = initial_lrate * math.pow(drop,
                                         math.floor((1 + epoch) / epochs_drop))

        return lrate

if __name__ == "__main__":

    trainer_cfg = {
        'batch_size': 64,
        'total_epochs': 0,
        'initial_lrate': 0.1,
        'epochs_before_drop': 15,
        'epochs_per_warm_start_run': 40,
        'dice_threshold': 0.5,
        'mode': 'distillation',
        'boundary_regularizer': False
    }

