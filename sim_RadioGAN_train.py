# import model_ult, cls_ult
import numpy as np
import tensorflow as tf
import time
from tensorflow import keras
import os, cv2
from skimage import io, transform
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

from TWC.gd_rsrp import Discriminator, Generator
batch_size = 10
weight = 64
height = 64
input_image_channel = 2
epochs = 100

from torch.utils.data import Dataset, DataLoader
class RadioUNet_c(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""

    def __init__(self, maps_inds=np.zeros(1), phase="train",
                 ind1=0, ind2=0,
                 dir_dataset="/data/RadioUnet/",
                 numTx=80,
                 thresh=0.2,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,):
        """
        Args:
            maps_inds: optional shuffled sequence of the maps. Leave it as maps_inds=0 (default) for the standart split.
            phase:"train", "val", "test", "custom". If "train", "val" or "test", uses a standard split.
                  "custom" means that the loader will read maps ind1 to ind2 from the list maps_inds.
            ind1,ind2: First and last indices from maps_inds to define the maps of the loader, in case phase="custom".
            dir_dataset: directory of the RadioMapSeer dataset.
            numTx: Number of transmitters per map. Default and maximal value of numTx = 80.
            thresh: Pathlos threshold between 0 and 1. Defaoult is the noise floor 0.2.
            simulation:"DPM", "IRT2", "rand". Default= "DPM"
            carsSimul:"no", "yes". Use simulation with or without cars. Default="no".
            carsInput:"no", "yes". Take inputs with or without cars channel. Default="no".
            IRT2maxW: in case of "rand" simulation, the maximal weight IRT2 can take. Default=1.
            cityMap: "complete", "missing", "rand". Use the full city, or input map with missing buildings "rand" means that there is
                      a random number of missing buildings.
            missing: 1 to 4. in case of input map with missing buildings, and not "rand", the number of missing buildings. Default=1.
            transform: Transform to apply on the images of the loader.  Default= transforms.ToTensor())

        Output:
            inputs: The RadioUNet inputs.
            image_gain

        """

        # self.phase=phase

        if maps_inds.size == 1:
            self.maps_inds = np.arange(0, 700, 1, dtype=np.int16)
            # Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds = maps_inds

        if phase == "train":
            self.ind1 = 0
            self.ind2 = 500
        elif phase == "val":
            self.ind1 = 501
            self.ind2 = 600
        elif phase == "test":
            self.ind1 = 601
            self.ind2 = 699
        else:  # custom range
            self.ind1 = ind1
            self.ind2 = ind2

        self.dir_dataset = dir_dataset
        self.numTx = numTx
        self.thresh = thresh

        self.simulation = simulation
        self.carsSimul = carsSimul
        self.carsInput = carsInput
        if simulation == "DPM":
            if carsSimul == "no":
                self.dir_gain = self.dir_dataset + "gain/DPM/"
            else:
                self.dir_gain = self.dir_dataset + "gain/carsDPM/"
        elif simulation == "IRT2":
            if carsSimul == "no":
                self.dir_gain = self.dir_dataset + "gain/IRT2/"
            else:
                self.dir_gain = self.dir_dataset + "gain/carsIRT2/"
        elif simulation == "rand":
            if carsSimul == "no":
                self.dir_gainDPM = self.dir_dataset + "gain/DPM/"
                self.dir_gainIRT2 = self.dir_dataset + "gain/IRT2/"
            else:
                self.dir_gainDPM = self.dir_dataset + "gain/carsDPM/"
                self.dir_gainIRT2 = self.dir_dataset + "gain/carsIRT2/"

        self.IRT2maxW = IRT2maxW

        self.cityMap = cityMap
        self.missing = missing
        if cityMap == "complete":
            self.dir_buildings = self.dir_dataset + "png/buildings_complete/"
        else:
            self.dir_buildings = self.dir_dataset + "png/buildings_missing"  # a random index will be concatenated in the code
        # else:  #missing==number
        #    self.dir_buildings = self.dir_dataset+ "png/buildings_missing"+str(missing)+"/"

        self.transform = transform

        self.dir_Tx = self.dir_dataset + "png/antennas/"
        # later check if reading the JSON file and creating antenna images on the fly is faster
        if carsInput != "no":
            self.dir_cars = self.dir_dataset + "png/cars/"

        self.height = 256
        self.width = 256

    def __len__(self):
        return (self.ind2 - self.ind1 + 1) * self.numTx

    def __getitem__(self, idx):

        idxr = np.floor(idx / self.numTx).astype(int)
        idxc = idx - idxr * self.numTx
        dataset_map_ind = self.maps_inds[idxr + self.ind1] + 1
        # names of files that depend only on the map:
        name1 = str(dataset_map_ind) + ".png"
        # names of files that depend on the map and the Tx:
        name2 = str(dataset_map_ind) + "_" + str(idxc) + ".png"

        # Load buildings:
        if self.cityMap == "complete":
            img_name_buildings = os.path.join(self.dir_buildings, name1)
        else:
            if self.cityMap == "rand":
                self.missing = np.random.randint(low=1, high=5)
            version = np.random.randint(low=1, high=7)
            img_name_buildings = os.path.join(self.dir_buildings + str(self.missing) + "/" + str(version) + "/", name1)
            str(self.missing)
        image_buildings = np.asarray(io.imread(img_name_buildings))

        # Load Tx (transmitter):
        img_name_Tx = os.path.join(self.dir_Tx, name2)
        image_Tx = np.asarray(io.imread(img_name_Tx))

        # Load radio map:
        if self.simulation != "rand":
            img_name_gain = os.path.join(self.dir_gain, name2)
            image_gain = np.expand_dims(np.asarray(io.imread(img_name_gain)), axis=2) / 255
        else:  # random weighted average of DPM and IRT2
            img_name_gainDPM = os.path.join(self.dir_gainDPM, name2)
            img_name_gainIRT2 = os.path.join(self.dir_gainIRT2, name2)
            # image_gainDPM = np.expand_dims(np.asarray(io.imread(img_name_gainDPM)),axis=2)/255
            # image_gainIRT2 = np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)),axis=2)/255
            w = np.random.uniform(0, self.IRT2maxW)  # IRT2 weight of random average
            image_gain = w * np.expand_dims(np.asarray(io.imread(img_name_gainIRT2)), axis=2) / 256 \
                         + (1 - w) * np.expand_dims(np.asarray(io.imread(img_name_gainDPM)), axis=2) / 256

        # pathloss threshold transform
        if self.thresh > 0:
            mask = image_gain < self.thresh
            image_gain[mask] = self.thresh
            image_gain = image_gain - self.thresh * np.ones(np.shape(image_gain))
            image_gain = image_gain / (1 - self.thresh)

        image_buildings = cv2.resize(image_buildings, (64,64))
        image_Tx = cv2.resize(image_Tx, (64,64))
        image_gain = cv2.resize(image_gain, (64,64))

        image_buildings = np.reshape(image_buildings, [64, 64])
        image_Tx = np.reshape(image_Tx, [64, 64])
        image_gain = np.reshape(image_gain, [64,64,1])

        mask_rx = image_gain == image_gain.max()
        image_Tx = image_gain*mask_rx
        image_Tx = np.reshape(image_Tx, [64, 64])

        image_buildings = maxmin_norm(image_buildings)
        image_Tx = maxmin_norm(image_Tx)
        image_gain = maxmin_norm(image_gain)

        # inputs to radioUNet
        if self.carsInput == "no":
            inputs = np.stack([image_buildings, image_Tx], axis=2)
            # The fact that the buildings and antenna are normalized  256 and not 1 promotes convergence,
            # so we can use the same learning rate as RadioUNets
        else:  # cars
            # Normalization, so all settings can have the same learning rate
            image_buildings = image_buildings / 256
            image_Tx = image_Tx / 256
            img_name_cars = os.path.join(self.dir_cars, name1)
            image_cars = np.asarray(io.imread(img_name_cars)) / 256
            inputs = np.stack([image_buildings, image_Tx, image_cars], axis=2)
            # note that ToTensor moves the channel from the last asix to the first!

        return [inputs, image_gain]


def discriminator_loss(disc_real_output, disc_generated_output):
    # [1, 30, 30, 1] with [1, 30, 30, 1]
    # print(disc_real_output.shape, disc_generated_output.shape)
    real_loss = keras.losses.binary_crossentropy(
                    tf.ones_like(disc_real_output), disc_real_output, from_logits=True)

    generated_loss = keras.losses.binary_crossentropy(
                    tf.zeros_like(disc_generated_output), disc_generated_output, from_logits=True)

    real_loss = tf.reduce_mean(real_loss)
    generated_loss = tf.reduce_mean(generated_loss)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss



def generator_loss(disc_generated_output, gen_output, target):

    LAMBDA = 100

    gan_loss = keras.losses.binary_crossentropy(
                tf.ones_like(disc_generated_output), disc_generated_output, from_logits=True)
    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    # l1_loss = tf.reduce_mean(tf.losses.MSE(target,gen_output))

    gan_loss = tf.reduce_mean(gan_loss)

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    # total_gen_loss = l1_loss

    return total_gen_loss

def RMSE(A,B):
    return np.sqrt(np.mean(np.power((A - B), 2)))
def MAPE(A,B):
    return np.nanmean(((A - B)/B)) * 100

def main(dataloaders,phase):

    # # 以元组方式 生成Dataset数据集
    # dataset_tuple = tf.data.Dataset.from_tensor_slices(train_dataset)

    # best_score = 100
    best_rmse = 20

    for epoch in range(epochs):
        start = time.time()
        step = 0
        # db_tuple = dataset_tuple.shuffle(400).batch(batch_size)
        val_RMSE = []
        for phase in ['val','train']:
            ###[BHgt_map, mask_map, resi_map, BMsk_map]
            # for step, inputs in enumerate(db_tuple):
            for inputs, targets in dataloaders[phase]:
                step+=1
                inputs = inputs.numpy().astype('float')
                targets = targets.numpy().astype('float')

                # print(np.shape(inputs),np.shape(targets))
                input_image = np.reshape(inputs, [batch_size, weight, height, input_image_channel])
                building_img = np.reshape(input_image[:, :, :, 0], [batch_size, weight, height, 1])
                Pint_map = np.reshape(input_image[:, :, :, 1], [batch_size, weight, height, 1])
                target = np.reshape(targets, [batch_size, weight, height, 1])

                if True in np.isnan(input_image):
                    print("np.isnan(input_image)")
                    continue
                elif True in np.isnan(target):
                    print("np.isnan(target)")
                    continue


                # plt.imshow(target[0,:, :,0])
                # plt.colorbar()
                # plt.show()
                # plt.imshow(input_image[0,:, :, 0])
                # plt.colorbar()
                # plt.show()
                # plt.imshow(input_image[0,:, :, 1])
                # plt.colorbar()
                # plt.show()
                if phase == 'train':
                    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                        # get generated pixel2pixel image
                        gen_output = generator(input_image, training=True)


                        # fed real pixel2pixel image together with original image
                        disc_real_output = discriminator([building_img, target], training=True)
                        # fed generated/fake pixel2pixel image together with original image
                        disc_generated_output = discriminator([building_img, gen_output], training=True)



                        gen_loss = generator_loss(disc_generated_output, gen_output, target)
                        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
                        # disc_loss = 1

                    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
                    # generator_gradients = [tf.clip_by_norm(g, 15) for g in generator_gradients]
                    g_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

                    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
                    # discriminator_gradients = [tf.clip_by_norm(g, 15) for g in discriminator_gradients]
                    d_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

                    if step% 500 == 0:
                        # print(disc_loss.shape, gen_loss.shape)
                        print(epoch, step, float(disc_loss), float(gen_loss), RMSE(gen_output, target))
                        # generate_images([input_deltaH], prediction, target, epoch)
                        # log = open("./result/GAN_train_residuals.txt", 'a+')
                        # print(epoch, step, float(disc_loss), float(gen_loss), RMSE(gen_output, target), '\r\n',file = log)
                        # log.close()
                elif phase == 'val':
                    prediction = generator(input_image, training=False)
                    prediction = np.reshape(prediction,[batch_size * height * weight])
                    target = np.reshape(target,[batch_size * height * weight])
                    val_RMSE.append(RMSE(prediction, target))
                    # if step % 100 == 0:
                    #     print(step,np.mean(val_RMSE))


        print('===Time taken for epoch {} is {} sec.\n'
              '===The test RMSE is {}. '.format(
            epoch , time.time() - start, np.mean(val_RMSE)
        ))
        # log = open("./result/GAN_test_residuals.txt", 'a+')
        # print(epoch, float(disc_loss), float(gen_loss), np.mean(test_RMSE), '\r\n', file=log)
        # log.close()

        if np.mean(val_RMSE) < best_rmse:
            best_rmse = np.mean(val_RMSE)
            # best_rmse = np.mean(test_RMSE)
            # tf.saved_model.save(generator,'./model/GAN/generator')
            # generator.save_weights('./model/GAN_CI/gen_64')
            model_save_path = './model/Sim/'
            generator.save_weights(model_save_path + 'generator')
            discriminator.save_weights(model_save_path + 'discriminator')
            print("Nice weight, should be save in ", model_save_path)
        print('--------------------------------------------------')
        print("--------TEST BEST RMSE:", best_rmse, "-----------")
        print('--------------------------------------------------')



def get_filename(root_dir, debug=False):
    filenames = []
    sample_cnt = 0
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for name in files:
            sample_cnt += 1
            # print("files: ",os.path.join(root, name),name)
            file_name = name
            file_content = os.path.join(root, name)
            filenames.append([file_name,file_content])
            # if debug:
            #     if sample_cnt == 1:
            #         break
    return filenames

def maxmin_norm_mask(data, mask):
    data = (data-data[mask.astype('bool')].min()) \
                 / (data[mask.astype('bool')].max() - data[mask.astype('bool')].min())

    # data = (data-0.5)*2
    data = data * mask

    return data

def maxmin_norm(data, ):
    data = (data-data.min())/(data.max() - data.min())
    return data




if __name__=='__main__':
    Radio_train = RadioUNet_c(phase="train")
    Radio_val = RadioUNet_c(phase="val")

    image_datasets = {
        'train': Radio_train, 'val': Radio_val
    }

    # batch_size = 15

    dataloaders = {
        'train': DataLoader(Radio_train, batch_size=batch_size, shuffle=True, num_workers=1),
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
    generator.build(input_shape=(batch_size, weight, height, input_image_channel))
    generator.summary()
    discriminator = Discriminator()
    discriminator.build(
        input_shape=[(batch_size, weight, height, 1), (batch_size, weight, height, 1)])
    discriminator.summary()

    g_optimizer = keras.optimizers.Adam(learning_rate=2e-5, beta_1=0.5)
    d_optimizer = keras.optimizers.Adam(learning_rate=2e-5, beta_1=0.5)
    main(dataloaders,"train")
    # print(image_build_ant[:,:,0].max(), image_build_ant[:,:,0].min())
    # print(image_build_ant[:,:,1].max(), image_build_ant[:,:,1].data)
    # print(image_gain[:,:,0].data, image_gain[:,:,0].data)