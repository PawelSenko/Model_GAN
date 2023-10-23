from keras.datasets.cifar10 import load_data
from keras.utils import to_categorical, load_img, img_to_array, array_to_img
from keras.preprocessing import image
from keras.layers import Dense, Conv2D, Flatten, Dropout, LeakyReLU, Input, Reshape, Conv2DTranspose
from keras.models import Sequential, load_model, Model
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD,Adam,Adadelta,Adagrad, RMSprop
import pickle
import os
import random
import numpy as np 
import matplotlib.pyplot as plt


(x_train, y_train), (_, _) = load_data()

x_train = x_train[y_train.flatten() == 4]

latent_dim = 40
height = 32
width = 32
channels = 3

x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels))
x_train = x_train.astype('float32')
#normalizacja względem zera (-1,1)
x_train = (x_train - 127.5) / 127.5


plt.figure()
for i in range(9):
    plt.subplot(3, 3, 1 + i)
    #plt.axis('off')
    plt.imshow(x_train[i])
plt.show()



def generate_real_samples(dataset, n_samples):
    #i = np.random.randint(0, dataset.shape[0], n_samples)
    #X = dataset[i]
    X = dataset[np.random.choice(dataset.shape[0], n_samples, replace=True), :]
 # generate 'real' class labels (1)
    y = np.ones((n_samples, 1))
    return X, y


def generate_fake_samples(n_samples):
    X = np.random.rand(32 * 32 * 3 * n_samples)
    X = -1 + X * 2
    X = X.reshape((n_samples, 32, 32, 3))
    y = np.zeros((n_samples, 1))
    return X, y

#def train_discriminator(model, dataset, epoch=10, n_batch=64):
    half_batch = int(n_batch / 2)
    for i in range(epoch):
        X_real, y_real = generate_real_samples(dataset, 32)
        _, real_acc = model.fit(X_real, y_real)
        X_fake, y_fake = generate_fake_samples(half_batch)
        _, fake_acc = model.fit(X_fake, y_fake)
    print('>%d real=%.0f%% fake=%.0f%%' % (i+1, real_acc*100, fake_acc*100))

def discriminator_model():
    
    inp = Input(shape=(height, width, channels))
    x = inp
    x =Conv2D(filters=64,kernel_size=(3, 3), strides=(2,2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x =Conv2D(filters=128,kernel_size=(3, 3),strides=(2,2),padding='same')(x)
    x = LeakyReLU(alpha=0.3)(x)
    x =Conv2D(filters=256, kernel_size=(3, 3),strides=(2,2),padding='same')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Flatten()(x)
    #x = Dense(128, activation='linear')(x)
    x = Dropout(0.4)(x)
    x = Dense(1, activation='sigmoid')(x)
    discriminator = Model(inp, x)
    return discriminator


def generation_model():

    image_shape = (latent_dim)
    inp = Input(shape=image_shape)
    x= inp
    x = Dense(128 * 16 * 16)(x)
    x = LeakyReLU()(x)
    x = Reshape((16, 16, 128))(x)
    # Then, add a convolution layer
    x = Conv2D(256, 5, padding='same')(x)
    x = LeakyReLU()(x)
    #Upsample to 32x32
    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)
    # Few more conv layers
    x = Conv2D(256, 5, padding='same')(x)
    x = LeakyReLU()(x)
    # Produce a 32x32 1-channel feature map
    x = Conv2D(channels, 7, activation='tanh', padding='same')(x)
    generator =Model(inp, x)

    return generator


model_dis = discriminator_model()
model_gen = generation_model()

#plot_model(Model_dis, to_file='discriminator_plot.png', show_shapes=True, show_layer_names=True)
opt_dis = Adam(lr=0.0002, beta_1=0.5)
opt_gen = Adam(lr=0.0002, beta_1=0.5)



def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(g_model)
    # add the discriminator
    model.add(d_model)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input =  np.random.rand(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=200, n_batch=128):
    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    # manually enumerate epochs
    for i in range(n_epochs):
    # enumerate batches over the training set
        for j in range(bat_per_epo):
        # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # update discriminator model weights
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # generate 'fake' examples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # update discriminator model weights
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = np.ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
            (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))




model_dis.compile(optimizer=opt_dis, loss='binary_crossentropy',metrics=['accuracy'])
train_discriminator(model_dis,x_train)