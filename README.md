# Cell-Level-RSRP-Estimation
Cell-Level RSRP Estimation with the Image-to-Image Wireless Propagation Model  Based on Measured data.  

This paper first proposes an image-to-image deep learning algorithm on real-world RSRP estimation. Different from the simulation data set, the measured RSRP data are sparse and fluctuating. First, we propose the mask method to solve the problem of data sparse. Second, we propose the residual method, which introduce the empirical model to provide the knowledge of radio propagation and guide the convergence direction of deep learning algorithm. The RSRP prediction results on measured dataset, our method can accelerate the convergence of deep learning algorithm and improve the accuracy of RSRP estimation.


## Requirement:
  
  Tensorflow 2.2.0 +

## Dataset download:
  
  [real-world dataset] Yi Zheng, March 9, 2022, "RSRPSet_urban: Radio map in dense urban ", IEEE Dataport, doi: https://dx.doi.org/10.21227/vmw5-c226.
  
  [simulation dataset] Levie, R., Yapar, Ç., Kutyniok, G. and Caire, G., 2021. RadioUNet: Fast radio map estimation with convolutional neural networks. IEEE Transactions on Wireless Communications.  
  
## Run:
    
  GAN_train.py                 # Running this code to training the Pix2Pix model on RSRPSet_urban for real-world RSRP estimation.
  
  GAN_inference.py             # inferencing the Pix2Pix model in RSRPSet_urban.
  
  UNet_train.py                # training the UNet model on RSRPSet_urban.
  
  UNet_inference.py            # training the UNet model on RSRPSet_urban.
  
  
  sim_RadioGAN_train.py        # training the Pix2Pix model on simulation dataset.
  
  sim_RadioGAN_inference.py    # inferencing the Pix2Pix model on simulation dataset.
  
  Note: before training the model, you may need to revise the dataset path and model save path in train code.
  
## Result:

  The visualization results and final performance of the training are placed in the folder "result".


![image](https://user-images.githubusercontent.com/22888185/162599726-fcd8acfc-5fc8-490a-b64f-55665cae8c40.png)

## Cite:
  This research has been published in IEEE Transactions on Cognitive Communications and Networking. if this research was helpful to you, please cite it as following format~
  
  **Y. Zheng, J. Wang, X. Li, J. Li and S. Liu, "Cell-Level RSRP Estimation With the Image-to-Image Wireless Propagation Model Based on Measured Data," in IEEE Transactions on Cognitive Communications and Networking, doi: 10.1109/TCCN.2023.3307945.**
  (https://ieeexplore.ieee.org/document/10227351)
  
  **Bib format:**
  @ARTICLE{Zheng2023,
  author={Zheng, Yi and Wang, Ji and Li, Xingwang and Li, Jiping and Liu, Shouyin},
  journal={IEEE Transactions on Cognitive Communications and Networking}, 
  title={Cell-Level RSRP Estimation With the Image-to-Image Wireless Propagation Model Based on Measured Data}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCCN.2023.3307945}}

  **Related work:**
  1. Z. Yi, L. Zhiwen, H. Rong, W. Ji, X. Wenwu and L. Shouyin, "Feature Extraction in Reference Signal Received Power Prediction Based on Convolution Neural Networks," in IEEE Communications Letters, vol. 25, no. 6, pp. 1751-1755, June 2021, doi: 10.1109/LCOMM.2021.3054862.
  2. Y. Zheng, T. Zhang, C. Liao, J. Wang, S. Liu, "RadioCycle: Deep Dual Learning based Radio Map Estimation," KSII Transactions on Internet and Information Systems, vol. 16, no. 11, pp. 3780-3797, 2022. DOI: 10.3837/tiis.2022.11.017.

