# from numpy.random import seed
# seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)
import numpy as np
import keras
from keras.layers import *
from keras.callbacks import *
from keras.regularizers import *
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


def manage_model(cmd_args, train_data):
    """If the user wants to train a model this function will train and return a model
    If the user just want to load a pre trained model the function will return a loaded model

    :parameter
      - `cmd_args`: The cmd_command line collected by argparse.
      - `train_data: TrainData object which will be used in the training process

    :returns
     - the function returns either a self trained model or a pre-trained model"""

    if cmd_args.loaded_model_location:
        return load_model(cmd_args.loaded_model_location)

    elif cmd_args.saved_model_location:
        model = Model(model_path=cmd_args.saved_model_location, input_shape=(train_data.selex_str_len, 4),
                      output_size=train_data.selex_files_num)
        # print("Hello...Test...")
        # print("train_data.shape",train_data.shape)
        # print("train_data.selex_str_len",train_data.selex_str_len.shape)
        model.create_model()
        model.model_train_and_save(data=train_data)
        return model.model


class Model:
    """
    A class used to build the model.

    Attributes
    ----------
    model : Sequen
        Sequential model
    model_path : str
        Where the model will be saved. Nedd to be with .h5 ending!!
    input_shape : tuple
        The model input shape
    output_size : int
        The model output size will be determined by the number of HT-SELEX cycles
    model_params_dict : dict
        Simple dictionary for the model architecture
    Methods
    -------
    create_model()
        Creates the sequential DL CNN model
    set_enrichment_matrix()
        Format the enrichment matrix in a way that will enable the matrix to be the model label
    model_train_and_save(DNA_string, data: TrainData)
        Train the model and save it in the end of the process
    """

    def __init__(self, model_path, input_shape, output_size):
        """
        Attributes
        ----------
        model : Sequen
            Sequential model
        model_path : str
            Where the model will be saved. Nedd to be with .h5 suffix!!
        input_shape : tuple
            The model input shape
        output_size : int
            The model output size will be determined by the number of HT-SELEX cycles
        model_params_dict : dict
            Simple dictionary for the model architecture
        """
        self.model = Sequential()
        self.model_path = model_path
        self.input_shape = input_shape
        self.output_size = output_size
        self.model_params_dict = {'ker_size': 8, 'pool_size': 5, 'layers': [64, 32, 32],
                                  'final_activation_function': 'sigmoid'}

    def create_model(self):
        """Builds the sequential model
        1. 1D conv layer
        2. Max-Pool
        3. Three Fully-Connected layers
        4. Output layer
        all the model arch params are from taken from model_params_dict
        """
        self.model.add(
            Conv1D(filters=512, kernel_size=self.model_params_dict['ker_size'], strides=1,
                   kernel_initializer='RandomNormal',
                   activation='relu',
                   kernel_regularizer=l2(5e-3), input_shape=self.input_shape, use_bias=True,
                   bias_initializer='RandomNormal'))

        self.model.add(MaxPooling1D(pool_size=self.model_params_dict['pool_size'], strides=None,
                               padding='valid',
                               data_format='channels_last'))

        self.model.add(Flatten())
        for layer_size in self.model_params_dict['layers']:
            print(f'the layer size is: {layer_size}')
            self.model.add(Dense(layer_size, activation='relu'))
        print("======> output_size", self.output_size)
        self.model.add(Dense(self.output_size, activation=self.model_params_dict['final_activation_function']))

    def model_train_and_save(self, data):
        """Train the model and save it in the end of the process
        The training process is short and relies on EarlyStopping callbacks
        The model is saved by model.model_path by EarlyStopping callback

        Parameters
        ----------
        data : TrainData
            This is the data object for the model traing process
        """
        Adam = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, decay=1e-5, amsgrad=False)
        self.model.compile(optimizer=Adam, loss='categorical_crossentropy')
        datasets = 3
        splitted_train_data, splitted_test_data = self.data_divider(data=data, datasets=datasets)
        for i in range(datasets):
            self.model.fit(splitted_train_data[i], splitted_test_data[i], epochs=30,
                        batch_size=64, verbose=1, shuffle=True, validation_split=0.3,
                        callbacks=[keras.callbacks.ModelCheckpoint(self.model_path, monitor='val_loss', verbose=0, save_best_only=True,
                                                                   save_weights_only=False, mode='auto', period=1),
                                   keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                 min_delta=0,
                                                                 patience=1,
                                                                 verbose=0, mode='auto', restore_best_weights=True)
                                   ])

    # def reshape_one_hot_data(self, data):
    #     # Convert the 1D array of 2D arrays into a single 3D array -- Dipesh
    #     print("=====> Inside reshape one hot data..")
    #     if isinstance(data.one_hot_data, np.ndarray) and data.one_hot_data.dtype == object:
    #         data.one_hot_data = np.array(list(data.one_hot_data))
    #     else:
    #         raise ValueError("data.one_hot_data is not in the expected format.")

    #     return data.one_hot_data
            
    def reshape_one_hot_data(self, data):
        print("=====> Inside reshape one hot data..")
        target_shape = (259, 4)
        if isinstance(data.one_hot_data, np.ndarray) and data.one_hot_data.dtype == object:
            # Pad sequences to the target length
            padded_sequences = pad_sequences(data.one_hot_data, maxlen=target_shape[0], padding='post', truncating='post', dtype='float32')
            data.one_hot_data = np.array(padded_sequences)
        else:
            raise ValueError("data.one_hot_data is not in the expected format.")
        return data.one_hot_data

    def data_divider(self, data, datasets=3):
        """Divide the data set into 3 (datasets) independent sets
        We found out this is the best way to train a model so it wont fit to the
        HT-SELEX technological characteristics

        Parameters
        ----------
        data : TrainData
            This is the data object which we will divide
        datasets : int
            This is the data object which we will divide
        """
        print("BEFORE=======", data.one_hot_data.shape)
        print("BEFORE=======", data.one_hot_data[0].shape)
        # Reshape one_hot_data if it's not 3-dimensional -- Dipesh
        if len(data.one_hot_data.shape) != 3:
            data.one_hot_data = self.reshape_one_hot_data(data)

        splitted_train_data = []
        splitted_test_data = []
        for i in range(0, datasets):
            print("=======", data.one_hot_data.shape)
            print("=======", data.one_hot_data[0].shape)
            print('the ranges in {i} are: {low}, {high}'.format(i=i,low=round((i/datasets)*len(data.one_hot_data)),high=round(((i+1)/datasets) * len(data.one_hot_data))))
            splitted_train_data.append(data.one_hot_data[round((i/datasets)*len(data.one_hot_data)):round(((i+1)/datasets) * len(data.one_hot_data)), :, :])
            splitted_test_data.append(data.enrichment_matrix[round((i/datasets)*len(data.enrichment_matrix)):round(((i+1)/datasets) * len(data.enrichment_matrix))])

        return splitted_train_data, splitted_test_data