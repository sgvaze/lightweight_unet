import numpy as np

class DataLoader(object):

    def __init__(self, trainer_cfg = None):

        self.all_data = self.init_data(trainer_cfg)
        self.data_for_exp = None

    def init_data(self, trainer_cfg):

        """
        Load data from numpy file, expects arrays of corresponding images and labels
        Split data into five validation folds and one test fold
        :param trainer_cfg: Cfg contains numpy arrays' file location
        :return: Dict of dicts, each sub-dict containing images and labels for one fold
        """
        images = np.load(trainer_cfg['input_data_path'])["arr_0"]
        labels = np.load(trainer_cfg['input_data_path'])["arr_1"]

        [train_imgs, test_imgs] = np.split(images, [int(0.8*images.shape[0])], axis=0)
        [train_labels, test_labels] = np.split(labels, [int(0.8*images.shape[0])], axis=0)

        train_img_val_folds = np.array_split(train_imgs, 5, axis = 0)
        train_labels_val_folds = np.array_split(train_labels, 5, axis=0)

        fold_names = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]

        data = {}

        for i, name in enumerate(fold_names):

            data[name] = {"imgs": train_img_val_folds[i],
                          "labels": train_labels_val_folds[i]
                          }

        data["Test"] = {
            "imgs": test_imgs,
            "labels": test_labels
        }

        return data

    def prepare_data(self, eval_fold = None):

        """
        Organise data into training and evaluation fold for specific experiment

        :param eval_fold: Which fold is the evaluation fold (either "Fold i" or "Test")
        :return: Dict of dicts, sub dicts for training and evaluation (each containing images and labels)
        """

        assert eval_fold in ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Test"], \
            "DataLoader.get_data() argument must be one of \"Fold 1\", \"Fold 2\", \"Fold 3\", \"Fold 4\", \"Fold 5\", \"Test\""

        print("Preparing data for fold: " + eval_fold + "...")

        train_imgs_list = []
        train_labels_list = []

        if eval_fold in ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]:

            for key, value in self.all_data.items():

                if key == "Test" or key == eval_fold:
                    pass
                else:
                    train_imgs_list.append(self.all_data[key]["imgs"])
                    train_labels_list.append(self.all_data[key]["labels"])

            train_imgs, train_labels = np.concatenate(train_imgs_list, axis = 0), np.concatenate(train_labels_list, axis = 0)
            eval_imgs, eval_labels = self.all_data[eval_fold]["imgs"], self.all_data[eval_fold]["labels"]

        else:

            for key, value in self.all_data.items():

                if key == eval_fold:
                    pass
                else:
                    train_imgs_list.append(self.all_data[key]["imgs"])
                    train_labels_list.append(self.all_data[key]["labels"])

            train_imgs, train_labels = np.concatenate(train_imgs_list, axis=0), np.concatenate(train_labels_list, axis=0)
            eval_imgs, eval_labels = self.all_data[eval_fold]["imgs"], self.all_data[eval_fold]["labels"]

        self.data_for_exp = {
            "train" : {
                "imgs": train_imgs,
                "labels": train_labels
            },

            "eval" : {
                "imgs": eval_imgs,
                "labels": eval_labels
            }
        }

        # Pre-process Data
        self.pre_process_data()

    def get_data(self):
        return self.data_for_exp

    def pre_process_data(self):

        """
        Preprocess data.
        Currently zero-means and unit stds the input images, and augments training labels for boundary regularization

        Add augmentation code here
        :return:
        """

        self.augment()  # TODO: Insert augmentation code here

        # Augment training labels for boundary regularization
        beta = 2
        boundaries = np.array(self.boundary_detector(self.data_for_exp["train"]["labels"]), dtype = np.int32)
        self.data_for_exp["train"]["labels"] += beta * boundaries

        # Zero mean the and unit std the training data. Apply same statistics to evaluation set
        # NOTE: Form of normalization could be changed (e.g per pixel normalization)
        mean = np.mean(self.data_for_exp["train"]["imgs"])
        std = np.mean(self.data_for_exp["train"]["imgs"])

        self.data_for_exp["train"]["imgs"] -= mean
        self.data_for_exp["train"]["imgs"] /= std

        self.data_for_exp["eval"]["imgs"] -= mean
        self.data_for_exp["eval"]["imgs"] /= std

    def augment(self):
        """
        Insert augmentation code here
        """
        pass

    # Function returns boundaries of the segmentation label
    def boundary_detector(self, labels):

        import skimage.morphology as sk

        size = np.shape(labels)
        boundaries = np.zeros(size)

        for i in np.arange(size[0]):
            label = labels[i, :, :]
            dilated = sk.dilation(label)
            eroded = sk.erosion(label)
            boundary = dilated - eroded
            boundaries[i, :, :] = boundary

        return boundaries