import numpy as np
import os
import tensorflow as tf
from keras.layers import Input, Lambda, Dropout, Dense, LeakyReLU, Layer, BatchNormalization, Concatenate, Softmax
from keras.models import Model
from keras.regularizers import l2,l1_l2,l1
from keras import backend as K
from keras import activations

class DenseTranspose(Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.dense = dense
        self.activation = activations.get(activation)
        super().__init__(**kwargs)
    def build(self, batch_input_shape):
        self.biases = self.add_weight(name='bias',shape=self.dense.input_shape[-1],initializer='zeros')
        super().build(batch_input_shape)
    def call(self, inputs):
        z = tf.matmul(inputs, self.dense.weights[0],transpose_b=True)
        return self.activation(z+self.biases)

def autoencoder(trainable, mean):
    #parameters:
    rate = 0.4
    dense_size = 200 
    glove_size = 300
    fMRI_size =  65730 
    reduced_size =  3221 
    gordon_areas = 333

    dense_size1 = 500
    dense_size2 = 200
    sizes = np.load(str(os.path.dirname(os.path.abspath(__file__))) + '/data/look_ups/sizes.npy')
    reduced = np.load(str(os.path.dirname(os.path.abspath(__file__))) + '/data/look_ups/reduced_sizes.npy')

    index1 = 0
    index = 0

    input_voxel = Input(shape=(fMRI_size,))

    # small ROI dense layers: Each ROI region has its own dense layer. The outputs are then concatenated and used in further layers

    branch_outputs = []
    dense_layers = []
    for i in range(gordon_areas):
        new_index = index + sizes[i]
        small_input = Lambda(lambda x: x[:,index:new_index],output_shape=(sizes[i],))(input_voxel)
        dense_layers.append(Dense(reduced[i]))
        small_out = dense_layers[i](small_input)
        small_out = LeakyReLU(alpha=0.3)(small_out)
        small_out = BatchNormalization()(small_out)
        branch_outputs.append(small_out)
        index = new_index
    Concat = Concatenate()(branch_outputs)
    print(type(Concat))
    print(Concat.shape)
    dense1 = BatchNormalization()(Concat)
    dense1 = Dropout(rate=rate)(dense1)
    print(f'small input: {small_input.shape}')
    print(f'small output: {small_out.shape}')
    print(f'dense1: {dense1}')
    #intermediate Layer: Reduce the output from the ROI small dense layer further. 
    # The output from this layer is also used for the autoencoder to reconstruct the fMRIs
    '''
    DENSE LAYER 1
    '''
    dense3 = Dense(dense_size1)   
    print(f'dense5: {dense3}')  
    out_furtherr1 = dense3(dense1)
    print(f'out_furtherr: {out_furtherr}')
    out_further1 = LeakyReLU(alpha=0.3)(out_furtherr1)
    out_further1 = BatchNormalization()(out_further1)
    out_further1 = Dropout(rate=rate)(out_further1)
    print(f'out_further: {out_further1}')
    '''
    DENSE LAYER 2
    '''
    dense5 = Dense(dense_size2)   
    print(f'dense5: {dense5}')  
    out_furtherr2 = dense5(dense3)
    print(f'out_furtherr: {out_furtherr2}')
    out_further2 = LeakyReLU(alpha=0.3)(out_furtherr2)
    out_further2 = BatchNormalization()(out_further2)
    out_further2 = Dropout(rate=rate)(out_further2)
    print(f'out_further: {out_further2}')

    #Glove layer: The output of this layer should represent the matching glove embeddings for their respective fMRI. 
    # A loss is only applied if the glove prediction model is run.
    
    dense_glove = Dense(300, trainable=trainable)
    print(f'dense_glove: {dense_glove}')
    out_glove = dense_glove(out_further2)
    print(f'out_glove: {out_glove}')
    out_gloverr = LeakyReLU(alpha=0.3)(out_glove)
    out_gloverr = BatchNormalization()(out_gloverr)
    out_gloverr = Dropout(rate=rate)(out_gloverr)
    print(f'out_gloverr: {out_gloverr}')

    #Classification layer: It returns a proability vector for a given fMRI belonging to a certainword out of the possible 180 words. 
    # The loss is only calculated if the classification model is run.

    out_mid = Dense(180, activation='softmax', trainable=trainable, kernel_regularizer=l2(0.005), bias_regularizer=l2(0.005))(out_gloverr)     
    print(f'out_mid: {out_mid}')
    dense4 = DenseTranspose(dense5)(out_further2)
    dense4 = LeakyReLU(alpha=0.3)(dense4)
    dense4 = BatchNormalization()(dense4)
    dense4 = Dropout(rate=rate)(dense4)
    print(f'dense4: {dense4}')
    branch_outputs1 = []
    for j in range(gordon_areas):
        new_index1 = index1+reduced[j] 
        small_input = Lambda(lambda x: x[:,index1:new_index1], output_shape=(reduced[j],))(dense4) 
        small_out = DenseTranspose(dense_layers[j])(small_input)
        small_out = LeakyReLU(alpha=0.3)(small_out)
        small_out = BatchNormalization()(small_out)
        branch_outputs1.append(small_out)
        index1 = new_index1
    out = Concatenate()(branch_outputs1)


    Concat_layer = Lambda(lambda t: t ,name = 'concat') (Concat)
    Dense_layer = Lambda(lambda t: t ,name = 'dense_mid') (out_furtherr2) 
    pred_class = Lambda(lambda t: t ,name = 'pred_class') (out_mid)
    pred_glove = Lambda(lambda t: t ,name = 'pred_glove') (out_glove)
    fMRI_rec = Lambda(lambda t: t, name='fMRI_rec')(out)

    if not mean:
        model= Model(inputs=[input_voxel],outputs=[fMRI_rec, pred_glove, pred_class])
    else:
        model = Model(inputs=[input_voxel],outputs=[fMRI_rec, pred_glove, pred_class, Concat_layer, Dense_layer])

    return model

def main():
    Model = autoencoder(True, True)

if __name__=="__main__":
    main()