# Cell-Level-RSRP-Estimation
Cell-Level RSRP Estimation with the Image-to-Image Wireless Propagation Model  Based on Measured data.  

This paper first proposes an image-to-image deep learning algorithm on real-world RSRP estimation. Different from the simulation data set, the measured RSRP data are sparse and fluctuating. First, we propose the mask method to solve the problem of data sparse. Second, we propose the residual method, which introduce the empirical model to provide the knowledge of radio propagation and guide the convergence direction of deep learning algorithm. The RSRP prediction results on measured dataset, our method can accelerate the convergence of deep learning algorithm and improve the accuracy of RSRP estimation.


Requirement:
  
  Tensorflow 2.2.0 +

Dataset download:
  
  [real-world dataset] Yi Zheng, March 9, 2022, "RSRPSet_urban: Radio map in dense urban ", IEEE Dataport, doi: https://dx.doi.org/10.21227/vmw5-c226.
  
  [simulation dataset] Levie, R., Yapar, Ç., Kutyniok, G. and Caire, G., 2021. RadioUNet: Fast radio map estimation with convolutional neural networks. IEEE Transactions on Wireless Communications.  
  
Run:
    
  GAN_train.py                 # Running this code to training the Pix2Pix model on RSRPSet_urban for real-world RSRP estimation.
  
  GAN_inference.py             # inferencing the Pix2Pix model in RSRPSet_urban.
  
  UNet_train.py                # training the UNet model on RSRPSet_urban.
  
  UNet_inference.py            # training the UNet model on RSRPSet_urban.
  
  
  sim_RadioGAN_train.py        # training the Pix2Pix model on simulation dataset.
  
  sim_RadioGAN_inference.py    # inferencing the Pix2Pix model on simulation dataset.
  
  Note: before training the model, you may need to revise the dataset path and model save path in train code.
  
Result:

  The visualization results and final performance of the training are placed in the folder "result".


![image](https://user-images.githubusercontent.com/22888185/162599726-fcd8acfc-5fc8-490a-b64f-55665cae8c40.png)
