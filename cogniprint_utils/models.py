from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import regularizers

class CogniprintClassifierCNN(Model):
    def __init__(self):
        input_layer = Input(shape=(75,100,3))
        
        # Convolutional layers
        x = Conv2D(10, kernel_size=(25, 1),  strides=(2, 2), activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(input_layer)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        
        # Dense Layers
        x = Dense(24,  activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=1e-4))(x)
        x = Dense(16,  activation='relu')(x)
        main_output = Dense(1, activation='sigmoid', name='main_output')(x)
        
        super().__init__(input_layer, main_output)
