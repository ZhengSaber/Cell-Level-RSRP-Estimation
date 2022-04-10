# import model_ult, cls_ult
import numpy as np
from tensorflow import keras
from    matplotlib import pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
from torch.utils.data import DataLoader

from TWC.gd_rsrp import Generator
batch_size = 10
weight = 64
height = 64
input_image_channel = 2
epochs = 100
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from sim_RadioGAN_train import RadioUNet_c

def RMSE(A,B):
    return np.sqrt(np.mean(np.power((A - B), 2)))
def MAPE(A,B):
    return np.nanmean(((A - B)/B)) * 100

def inference(dataset,phase):
    ShowPic = True
    for inputs, targets in dataloaders[phase]:
        inputs = inputs.numpy().astype('float')
        targets = targets.numpy().astype('float')
        input_image = np.reshape(inputs, [batch_size, weight, height, input_image_channel])
        building_img = np.reshape(input_image[:, :, :, 0], [batch_size, weight, height, 1])
        Pint_map = np.reshape(input_image[:, :, :, 1], [batch_size, weight, height, 1])
        target = np.reshape(targets, [batch_size, weight, height, 1])

        if ShowPic:
            plt.imshow(target[0,:, :,0])
            plt.colorbar()
            plt.show()
            plt.imshow(input_image[0,:, :, 0])
            plt.colorbar()
            plt.show()
            plt.imshow(input_image[0,:, :, 1])
            plt.colorbar()
            plt.show()

        prediction = generator(input_image, training=False)

        if ShowPic:
            plt.imshow(prediction[0,:, :,0])
            plt.colorbar()
            plt.show()

        rmse = RMSE(prediction[building_img>0], target[building_img>0])
        print(rmse)


if __name__=='__main__':
    Radio_val = RadioUNet_c(phase="test")

    image_datasets = {
        'val': Radio_val
    }

    # batch_size = 15

    dataloaders = {
        'val': DataLoader(Radio_val, batch_size=batch_size, shuffle=True, num_workers=1)
    }

    # i = 10
    # image_build_ant, image_gain = Radio_train[i]
    # print(np.shape(image_build_ant), np.shape(image_gain))
    # plt.imshow(image_gain[:,:,0])
    # plt.show()
    # plt.imshow(image_build_ant[:,:,0])
    # plt.show()
    # plt.imshow(image_build_ant[:,:,1])
    # plt.show()
    generator = Generator()
    # generator.load_weights("./model/RadioGAN_v2/generator")
    generator.load_weights("./model/Sim/generator")

    g_optimizer = keras.optimizers.Adam(learning_rate=2e-5, beta_1=0.5)
    d_optimizer = keras.optimizers.Adam(learning_rate=2e-5, beta_1=0.5)


    inference(dataloaders,'val')
    # print(image_build_ant[:,:,0].max(), image_build_ant[:,:,0].min())
    # print(image_build_ant[:,:,1].max(), image_build_ant[:,:,1].data)
    # print(image_gain[:,:,0].data, image_gain[:,:,0].data)