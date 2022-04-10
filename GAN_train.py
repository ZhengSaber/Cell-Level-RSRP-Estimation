# import model_ult, cls_ult
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
import ult

from gd_rsrp import Discriminator, Generator
batch_size = 32
weight = 64
height = 64
input_image_channel = 2
epochs = 600

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


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
    #
    # real_loss = tf.reduce_mean(disc_real_output)
    # generated_loss = tf.reduce_mean(disc_generated_output)
    #
    # total_disc_loss = -real_loss + generated_loss

    return total_disc_loss



def generator_loss(disc_generated_output, gen_output, target):

    LAMBDA = 100

    gan_loss = keras.losses.binary_crossentropy(
                tf.ones_like(disc_generated_output), disc_generated_output, from_logits=True)
    # gan_loss = disc_generated_output

    # mean absolute error
    # l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    l2_loss = tf.reduce_mean(tf.square(target - gen_output))

    gan_loss = tf.reduce_mean(gan_loss)

    total_gen_loss = gan_loss + (LAMBDA * l2_loss)

    return total_gen_loss



def get_data(data):
    '''
        building_height_map,
       cross_point_map,
       building_mask_map,
       altitude_height_map,
       RSRP_map,
       SPM_map,
       fill_RSRP_map,
   '''

    building_height_map = data[:, :, :, 0]
    line_map = data[:, :, :, 1]
    building_mask_map = data[:, :, :, 2]
    altitude_height_map = data[:, :, :, 3]
    RSRP_map = data[:, :, :, 4]
    SPM_map = data[:, :, :, 5]
    fill_RSRP_map = data[:, :, :, 6]

    #Truncation measured RSRP and SPM prediction
    # diff = SPM_map - fill_RSRP_map
    # SPM_map = SPM_map - np.nanmedian(np.array(diff))
    fill_RSRP_map[fill_RSRP_map > -75] = -75
    fill_RSRP_map[fill_RSRP_map < -105] = -105
    SPM_map[SPM_map > -75] = -75
    SPM_map[SPM_map < -105] = -105

    # diff = SPM_map - fill_RSRP_map
    # SPM_map = SPM_map - np.nanmedian(np.array(diff))

    # resi_map = SPM_map - fill_RSRP_map
    resi_map = fill_RSRP_map

    #Norm.
    # fill_RSRP_map = ult.RSRP_maxmin_norm(fill_RSRP_map)
    # SPM_map = ult.RSRP_maxmin_norm(SPM_map)
    resi_map = ult.RSRP_maxmin_norm(resi_map)
    building_height_map = ult.maxmin_norm(building_height_map)
    input_image = np.stack([building_height_map, line_map], axis=3)

    BMsk_map = np.abs(building_mask_map - 1)
    # FSRP_map = fill_RSRP_map * BMsk_map

    # resi_map = resi_map * BMsk_map


    input_image = np.reshape(input_image, [-1, weight, height, input_image_channel])
    building_img = np.reshape(building_height_map, [-1, weight, height, 1])
    BMsk_map = np.reshape(BMsk_map, [-1, weight, height, 1])
    Pint_map = np.reshape(line_map, [-1, weight, height, 1])

    # target = np.reshape(FSRP_map, [-1, weight, height, 1])
    target = np.reshape(resi_map, [-1, weight, height, 1])

    mask = (RSRP_map != 0) + 0.
    mask = np.reshape(mask, [-1, weight, height, 1])
    # plt.imshow(SPM_map[0, :, :])
    # plt.colorbar()
    # plt.show()
    # plt.close('all')
    # plt.imshow(fill_RSRP_map[0, :, :])
    # plt.colorbar()
    # plt.show()
    # plt.close('all')
    # plt.imshow(target[0, :, :, 0])
    # plt.colorbar()
    # plt.show()
    # plt.close('all')

    return input_image, target, BMsk_map, mask


import GAN_inference
def main(traindataset,testdataset):

    best_rmse = 20.0
    MaxStep = 150
    step = 0
    epoch = 0
    for data in traindataset:
        if epoch>epochs:
            print("Well Done! My friend~")
            break
        if step>MaxStep:
            step = 0
            epoch += 1


        if len(data) != batch_size:
            continue
        step+=1

        # input_image, target, BMsk_map,_ = get_data(data)
        input_image, target, BMsk_map,mask = get_data(data)
        building_img = input_image[:,:,:,0]
        # signal_line_img = input_image[:, :, :, 1]
        # signal_line_img[signal_line_img>0] = 1
        # input_image = np.stack([building_img,signal_line_img], axis=3)

        building_img = np.reshape(building_img, [-1, weight, height, 1])
        # signal_line_img = np.reshape(building_img, [-1, weight, height, 1])

        if True in np.isnan(input_image):
            print("np.isnan(input_image)")
            continue
        elif True in np.isnan(target):
            print("np.isnan(target)")
            continue

        target_mask = target * BMsk_map



        #training:

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # get generated pixel2pixel image
            gen_output = generator(input_image, training=True)

            gen_output_mask = gen_output * BMsk_map

            # fed real pixel2pixel image together with original image
            disc_real_output = discriminator([building_img, target], training=True)
            # fed generated/fake pixel2pixel image together with original image
            disc_generated_output = discriminator([building_img, gen_output], training=True)



            gen_loss = generator_loss(disc_generated_output, gen_output_mask, target_mask)
            # disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
            disc_loss = 1

        generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        # generator_gradients = [tf.clip_by_value(g, -0.01, 0.01) for g in generator_gradients]
        g_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))

        # discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        # discriminator_gradients = [tf.clip_by_value(g, -0.01, 0.01) for g in discriminator_gradients]
        # d_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

        if step% 50 == 0:
            # print(disc_loss.shape, gen_loss.shape)
            print(step, float(disc_loss), float(gen_loss), ult.RMSE(gen_output, target))
            # generate_images([input_deltaH], prediction, target, epoch)
            log = open("./result/OurModel/GAN_train_residuals.txt", 'a+')
            print(epoch, step, float(disc_loss), float(gen_loss), ult.RMSE(gen_output, target), '\r\n',file = log)
            log.close()

        if step % MaxStep == 0:
            #testing


            RM, MA,SSIM,PSNR = [], [],[],[]
            for i in range(len(testdataset)):
                # if i == 5 :
                if i == 0 and epoch % 10 == 0:
                    Save_Result = True
                else:
                    Save_Result = False

                rmse, mape, ssim, psnr = GAN_inference.inference(test_data[i],
                                                        generator,
                                                        ep=epoch,
                                                        Save_Result=Save_Result,
                                                        save_dir_name='/data/RSRP_dataset/RadioGAN/TWC/result/OurModel/')

                # if i >3:
                #     break
                RM.append(rmse)
                MA.append(mape)
                SSIM.append(ssim)
                PSNR.append(psnr)

            RM = np.array(RM)
            MA = np.array(MA)
            SSIM = np.array(SSIM)
            PSNR = np.array(PSNR)
            # print(len(RM), RM.mean(), RM.std())
            # print(len(MA), MA.mean(), MA.std())


            # val_RMSE = RMSE(prediction, target)
            # if step % 100 == 0:
            #     print(step,np.mean(val_RMSE))

            print('===Epoch {} / Step {}'
                  '---The test RMSE, MAPE, SSIM, PSNR are {},{},{},{}. '.format(
                epoch, step, RM.mean(), MA.mean(), SSIM.mean(), PSNR.mean()
            ))
            log = open("./result/OurModel/GAN_test_residuals.txt", 'a+')
            print(epoch, step, float(disc_loss), float(gen_loss), RM.mean(), MA.mean(), '\r\n', file=log)
            log.close()

            if RM.mean() < best_rmse:

                best_rmse = RM.mean()
                # best_rmse = np.mean(test_RMSE)
                # tf.saved_model.save(generator,'./model/GAN/generator')
                # generator.save_weights('./model/GAN_CI/gen_64')
                model_save_path = './model/OurModel/'
                generator.save_weights(model_save_path + 'generator')
                discriminator.save_weights(model_save_path + 'discriminator')
                print("Nice weight, should be save in ", model_save_path)
            print('--------------------------------------------------')
            print("--------TEST BEST RMSE:", best_rmse, "-----------")
            print('--------------------------------------------------')





if __name__=='__main__':
    dataset_path = '/data/RSRP_dataset/RadioGAN/TWC/dataset/imgs/'

    train_data = ult.read_dataset(dataset_path,'trainset.npy')
    test_data = ult.read_dataset(dataset_path,'testset.npy')

    # i = 10
    # images = test_data[i]
    # print(np.shape(images))
    # plt.imshow(images[:,:,0])
    # plt.show()
    # plt.imshow(images[:,:,1])
    # plt.show()
    # plt.imshow(images[:,:,5])
    # plt.show()
    # plt.imshow(images[:,:,6])
    # plt.show()

    generator = Generator()
    generator.build(input_shape=(batch_size, weight, height, input_image_channel))
    generator.summary()
    discriminator = Discriminator()
    discriminator.build(
        input_shape=[(batch_size, weight, height, 1), (batch_size, weight, height, 1)])
    discriminator.summary()


    # generator.load_weights("/data/RSRP_dataset/RadioGAN/TWC/model/tanh/generator")
    # discriminator.load_weights("/data/RSRP_dataset/RadioGAN/TWC/model/tanh/discriminator")

    # g_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    # d_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    g_optimizer = keras.optimizers.RMSprop(1e-4)
    d_optimizer = keras.optimizers.RMSprop(5e-5)

    image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=180,
                                                                horizontal_flip=True,
                                                                vertical_flip=True
                                                                )
    # image_gen = tf.keras.preprocessing.image.ImageDataGenerator()
    # image_gen.fit(x=train_data)
    data_gen = image_gen.flow(train_data,batch_size=batch_size)  # 生成强化数据集迭代器


    main(data_gen, test_data)