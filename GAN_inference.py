# import model_ult, cls_ult
import numpy as np
from    matplotlib import pyplot as plt
import os,cv2
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
import ult
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from gd_rsrp import Generator
batch_size = 1
weight = 64
height = 64
input_image_channel = 2

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



def make_dataset(path):
    filenames = get_filename(path)
    data = []
    cnt = 0
    for name, context in filenames:
        data.append(np.load(path+name))
        cnt += 1

    return np.array(data)



def inference(data,generator,ep=0, Save_Result=False, save_dir_name=None):

    building_height_map = data[:, :, 0]
    line_map = data[:, :, 1]
    building_mask_map = data[:, :, 2]
    altitude_height_map = data[:, :, 3]
    RSRP_map = data[:, :, 4]
    SPM_map = data[:, :, 5]
    fill_RSRP_map = data[:, :, 6]


    # Truncation measured RSRP and SPM prediction
    # diff = SPM_map - fill_RSRP_map
    # SPM_map = SPM_map - np.nanmedian(np.array(diff))
    fill_RSRP_map[fill_RSRP_map > -75] = -75
    fill_RSRP_map[fill_RSRP_map < -105] = -105
    SPM_map[SPM_map > -75] = -75
    SPM_map[SPM_map < -105] = -105


    BMsk_map = np.abs(building_mask_map - 1)
    BMsk_map[BMsk_map==0] = np.nan

    mask = (RSRP_map != 0)*1.0
    mask[mask==0] = np.nan

    # resi_map = ult.RSRP_maxmin_norm(SPM_map - fill_RSRP_map)
    # # rsrp_map = (SPM_map) - (resi_map * 40 )
    # min = (SPM_map - fill_RSRP_map).min()
    # max = (SPM_map - fill_RSRP_map).max()
    # rsrp_map = SPM_map - (resi_map * (max - min) + min)
    # print("RMSE resi:", ult.RMSE(rsrp_map, fill_RSRP_map))

    # Norm.
    building_height_map = ult.maxmin_norm(building_height_map)
    # SPM_map = ult.RSRP_maxmin_norm(SPM_map)

    input_image = np.stack([building_height_map, line_map], axis=2)



    input_image = np.reshape(input_image, [1, weight, height, input_image_channel])
    prediction = generator(input_image, training=False)
    prediction = np.reshape(prediction,[64,64])

    RSRP_map = RSRP_map[mask>0]

    # map_min = -110
    # map_max = -70
    min = (fill_RSRP_map).min()
    max = (fill_RSRP_map).max()
    Pred_map = (prediction + 1) / 2
    Pred_map = Pred_map * (max - min) + min
    # Pred_map = (prediction + 1)/2
    # min = (SPM_map - fill_RSRP_map).min()
    # max = (SPM_map - fill_RSRP_map).max()
    # Pred_map = SPM_map - (Pred_map* (max-min) + min)


    ssim = ult.compute_ssim(Pred_map, fill_RSRP_map)
    Pred_map = Pred_map * BMsk_map
    FSRP_map = fill_RSRP_map * BMsk_map

    # plt.imshow(building_height_map)
    # plt.colorbar()
    # plt.show()
    #
    # plt.imshow(FSRP_map)
    # plt.colorbar()
    # plt.show()


    if Save_Result:
        # Pred_map = Pred_map[mask > 0]
        # FSRP_map = FSRP_map[mask > 0]
        # rmse = ult.RMSE(Pred_map, FSRP_map)
        # mape = ult.MAPE(Pred_map, FSRP_map)
        # print('logging:',rmse,mape)

        plot_2d(BMsk_map, FSRP_map, Pred_map,ep,save_dir_name)

    # plot_2d(BMsk_map, FSRP_map, Pred_map)

    # Pred_map = Pred_map[mask>0]
    # FSRP_map = FSRP_map[mask>0]
    rmse = ult.RMSE(Pred_map, FSRP_map)
    mape = ult.MAPE(Pred_map, FSRP_map)
    # print(rmse,mape)
    # SPM_map = SPM_map * BMsk_map
    psnr = ult.psnr(Pred_map, FSRP_map)

    # if rmse>8.0:
    #     print("watch out!")
    #     SPM_map = SPM_map * BMsk_map
    #     # plt.imshow(Pred_map)
    #     # plt.colorbar()
    #     # plt.show()
    #     plt.imshow(FSRP_map)
    #     plt.colorbar()
    #     plt.show()
    #     # plt.imshow(SPM_map)
    #     # plt.colorbar()
    #     # plt.show()
    #     # Pred_map = SPM_map

    return rmse, mape, ssim, psnr


def plot_2d(height_map,RSRP_map,pred_map,ep, save_dir_name=None):

    plt.subplot(221)
    plt.imshow(height_map,
                     # norm=vnorm,
                     alpha=1.0,
                     # cmap=cm.coolwarm
               )

    # 加上右侧的值的数值
    # fig.colorbar(surf, fraction=0.05, pad=0.05)

    col = np.nonzero(RSRP_map)
    pred_data = pred_map[RSRP_map.astype('bool')]
    RSRP_data = RSRP_map[RSRP_map.astype('bool')]

    k1 = plt.scatter(col[1], col[0],
                   c=RSRP_data,
                   s=5,
                   alpha=1.0,
                   label='measured RSRP')
    # plt.colorbar(k1, fraction=0.05, pad=0.05)
    # ax1.legend()
    plt.axis('off')  # 去坐标轴
    plt.title('measured RSRP')

    plt.subplot(222)
    plt.imshow(height_map,
                      # norm=vnorm,
                      alpha=1.0,
                      # cmap=cm.coolwarm
               )

    # 加上右侧的值的数值
    # fig.colorbar(surf, fraction=0.05, pad=0.05)

    k2 = plt.scatter(col[1], col[0],
                    c=pred_data,
                    s=5,
                    alpha=1.0,
                    label='predicted RSRP')
    plt.colorbar(k2, fraction=0.05, pad=0.05)
    # ax1.legend()
    plt.axis('off')  # 去坐标轴
    plt.title('predicted RSRP')

    # pred_bins = np.arange(np.floor(pred_data.min()), np.ceil(pred_data.max()), 1)
    # rsrp_bins = np.arange(np.floor(RSRP_data.min()), np.ceil(RSRP_data.max()), 1)
    pred_bins = np.arange(-110, -70, 1)
    rsrp_bins = np.arange(-110, -70, 1)
    plt.subplot(223)
    plt.hist(RSRP_map.ravel(), pred_bins)
    plt.subplot(224)
    plt.hist(pred_map.ravel(), rsrp_bins)
    if save_dir_name is not None:
        plt.savefig(save_dir_name+str(ep)+'.png')
    plt.close('all')
    # plt.show()

if __name__=='__main__':

    dataset_path = '/data/RSRP_dataset/RadioGAN/TWC/dataset/imgs/'
    test_data = ult.read_dataset(dataset_path, 'testset.npy')


    generator = Generator()
    # generator.load_weights("./model/RadioGAN_v2/generator")
    # generator.load_weights("./model/RadioGAN_v1/RadioGAN")

    generator.load_weights("./model/OurModel/generator")
    # generator.load_weights("./model/GAN_res/generator")

    RM, MA, SSIM,PSNR = [], [], [],[]
    for i in range(len(test_data)):
        print(i)

        # if i <= 10:
        rmse, mape, ssim,psnr = inference(test_data[i],
                                    generator,)

        print(rmse)

        # if i >5:
        #     break
        RM.append(rmse)
        MA.append(mape)
        SSIM.append(ssim)
        PSNR.append(psnr)
        break

    RM = np.array(RM)
    MA = np.array(MA)
    SSIM = np.array(SSIM)
    PSNR = np.array(PSNR)


    print(len(RM), RM.mean(), RM.std())
    print(len(MA), MA.mean(), MA.std())
    print(len(SSIM), SSIM.mean(), SSIM.std())
    print(len(PSNR), PSNR.mean(), PSNR.std())


    # rmse, mape = inference(test_data[-3],generator)