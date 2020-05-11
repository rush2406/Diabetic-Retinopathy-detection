import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from random import shuffle
from keras.models import Model
from keras.layers import GlobalAveragePooling2D,Dense,Dropout
from keras import applications
from keras.preprocessing import image
from keras import optimizers
from keras.layers import Input
from keras.layers import Layer
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.models import save_model

#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SEED = 1
df_train = pd.read_csv('Resnet_colab/train.csv')
df_test = pd.read_csv('Resnet_colab/test.csv')
x = df_train['id_code']
y = df_train['diagnosis']

#x, y = shuffle(x, y, random_state=SEED)
df_train["id_code"] = df_train["id_code"].apply(lambda x: x + ".png")
df_test["id_code"] = df_test["id_code"].apply(lambda x: x + ".png")
df_train['diagnosis'] = df_train['diagnosis'].astype('str')
print(df_train)

from sklearn.model_selection import train_test_split
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.15,
                                                      stratify=y, random_state=1)
print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)
#train_y.hist()
#valid_y.hist()

BATCH_SIZE = 8
EPOCHS = 50
WARMUP_EPOCHS = 2
LEARNING_RATE = 1e-4
WARMUP_LEARNING_RATE = 1e-3
HEIGHT = 256
WIDTH = 256
CANAL = 3
N_CLASSES = df_train['diagnosis'].nunique()
ES_PATIENCE = 5
RLROP_PATIENCE = 3
DECAY_DROP = 0.5

def create_model(input_shape, n_out):
    input_tensor = Input(shape = input_shape)
    base_model = applications.DenseNet169(weights=None, 
                                       include_top=False,
                                       input_tensor=input_tensor)
    base_model.load_weights('Resnet_colab/DenseNet-BC-169-32-no-top.h5')
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.25)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.25)(x)
    final_output2 = Dense(n_out, activation='softmax',name='final_output2')(x)
    model = Model(input_tensor, final_output2)

    return model

def create_model1(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    base_model = applications.inception_v3.InceptionV3(weights='imagenet', 
                                       include_top=False,
                                       input_tensor=input_tensor)
    #base_model.load_weights('Resnet_colab/resnet50_weights.h5')
    
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.25)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.25)(x)
    final_output1 = Dense(n_out, activation='softmax',name='final_output1')(x)
    model = Model(input_tensor, final_output1)

    return model

model1 = create_model1(input_shape=(256, 256, 3),n_out=5)
model2 = create_model(input_shape=(256, 256, 3),n_out=5)

def get_config(self):

        config = super().get_config().copy()
        config.update({
            'vocab_size': self.vocab_size,
            'num_layers': self.num_layers,
            'units': self.units,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dropout': self.dropout
        })
        return config

class WeightedSum(Layer):
    def __init__(self, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
    def call(self, model_outputs):
        return 0.6 * model_outputs[0] + 0.4 * model_outputs[1]
    def compute_output_shape(self, input_shape):
        return input_shape[0]

out = WeightedSum()([model1.output, model2.output])
model = Model(inputs=[model1.input, model2.input], outputs=[out])

for layer in model.layers:
    layer.trainable = False

for i in range(-11, 0):
    model.layers[i].trainable = True

metric_list = ["accuracy"]
optimizer = optimizers.Adam(lr=WARMUP_LEARNING_RATE)
model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=metric_list)

#save_model(model,'avg.h5')
#model = load_model('avg.h5',custom_objects={'WeightedSum': WeightedSum()},compile=False)
#model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=metric_list)

input_imgen = ImageDataGenerator(validation_split=0.2)

test_imgen = ImageDataGenerator()

def generate_generator_multiple(generator):
    
    genX1 = generator.flow_from_dataframe(
                                    dataframe=df_train,
                                    directory="Resnet_colab/train_preprocess_1/",
                                    x_col="id_code",
                                    y_col="diagnosis",
                                    batch_size=BATCH_SIZE,
                                    class_mode="categorical",
                                    target_size=(HEIGHT, WIDTH),
                                    subset='training')
                                              
    while True:
            X1i = genX1.next()
            #X2i = genX2.next()
            yield [X1i[0], X1i[0]], X1i[1]  #Yield both images and their mutual label
            
            
inputgenerator=generate_generator_multiple(input_imgen)       
     
testgenerator=generate_generator_multiple(test_imgen) 



history_warmup = model.fit_generator(generator=inputgenerator,
                              steps_per_epoch=554,
                              validation_data=testgenerator,
                              validation_steps=138,
                              epochs=WARMUP_EPOCHS,
                              verbose=1).history


for layer in model.layers:
    layer.trainable = True

es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)
rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=RLROP_PATIENCE, factor=DECAY_DROP, min_lr=1e-6, verbose=1)

callback_list = [es, rlrop]
optimizer = optimizers.Adam(lr=LEARNING_RATE)
model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=metric_list)
history_finetunning = model.fit_generator(generator=inputgenerator,
                              steps_per_epoch=554,
                              validation_data=testgenerator,
                              validation_steps=138,
                              epochs=EPOCHS,
                              callbacks=callback_list,
                              verbose=1).history

import seaborn as sns
history = {'loss': history_warmup['loss'] + history_finetunning['loss'], 
           'val_loss': history_warmup['val_loss'] + history_finetunning['val_loss'], 
           'accuracy': history_warmup['accuracy'] + history_finetunning['accuracy'], 
           'val_accuracy': history_warmup['val_accuracy'] + history_finetunning['val_accuracy']}

sns.set_style("whitegrid")
fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', figsize=(20, 14))

ax1.plot(history['loss'], label='Train loss')
ax1.plot(history['val_loss'], label='Validation loss')
ax1.legend(loc='best')
ax1.set_title('Loss')

ax2.plot(history['accuracy'], label='Train Accuracy')
ax2.plot(history['val_accuracy'], label='Validation accuracy')
ax2.legend(loc='best')
ax2.set_title('Accuracy')

plt.xlabel('Epochs')
sns.despine()
fig.savefig('accuracy.jpg')

complete_datagen = ImageDataGenerator()

def generate_generator_complete(generator):
    
    genX1 = generator.flow_from_dataframe(
                                    dataframe=df_train,
                                    directory = "Resnet_colab/train_preprocess_1/",
                                    x_col="id_code",
                                    y_col=None,
                                    target_size=(HEIGHT, WIDTH),
                                    batch_size=1,
                                    shuffle=False,
                                    class_mode=None)
                                              
    while True:
            X1i = genX1.next()
            #X2i = genX2.next()
            yield [np.expand_dims(X1i[0], axis=0), np.expand_dims(X1i[0], axis=0)]  
            

complete_generator = generate_generator_complete(complete_datagen)

save_model(model,'avg.h5')
model = load_model('avg.h5',custom_objects={'WeightedSum': WeightedSum()},compile=False)
model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=metric_list)

train_preds = model.predict_generator(complete_generator, steps=5547)
train_preds = [np.argmax(pred) for pred in train_preds]

from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
labels = ['0 - No DR', '1 - Mild', '2 - Moderate', '3 - Severe', '4 - Proliferative DR']
cnf_matrix = confusion_matrix(df_train['diagnosis'].astype('int'), train_preds)
cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
df_cm = pd.DataFrame(cnf_matrix_norm, index=labels, columns=labels)
fig = plt.figure(figsize=(16, 7))
sns.heatmap(df_cm, annot=True, fmt='.2f', cmap="Blues")
fig.savefig('confusion_matrix.jpg')

print(classification_report(df_train['diagnosis'].astype('int'), train_preds)) 
