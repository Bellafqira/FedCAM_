# Identifying Malicious Models in Federated Learning Environments Using FedCam

As an emerging machine learning paradigm, federated learning (*FL*) allows multiple participants to collaboratively train a shared global model on decentralized data while safeguarding data privacy. However, traditional FL is susceptible to adversarial poisoning attacks. In an *FL* system, when the global model is poisoned by adversaries, it may fail to converge or demonstrate a degradation in accuracy. To counter these attacks, we propose *FedCam*, a robust framework wherein the central server utilizes a conditional variational autoencoder to detect and exclude malicious model updates. We utilize the reconstruction error of the distribution of activation maps as an anomaly score, as the reconstruction error of malicious updates is significantly larger than that of benign ones. Based on this concept, we formulate a dynamic threshold of reconstruction error to distinguish malicious updates from normal ones. FedCam has undergone rigorous testing through extensive experiments on IID federated benchmarks, demonstrating competitive performance compared to existing poisoning detection methods.

# Features
In this work, the poisoning detection methods and the poisoning attacks are configured for the image classification datasets [Mnist](https://www.image-net.org) (28x28 pixels, 10 classes) MNIST Database. 
We implemented the following *anomaly detection schemes*:
1. FedCam "our anomaly detection scheme"
1. [FedCVAE](https://ieeexplore.ieee.org/abstract/document/9460523) 

The following attacks are also implemented :

Same Value attack;
Add noise attack; 
Backdooring attack (add pattern attack and label flipping attack);
# Step-by-step Guide to Running the Code:
1. Configure Hyperparameters
2. Run the script using Python
   - python .\TestMain.py -algo fedCam
   or
   - python .\TestMain.py -algo fedCvae

# requirements : 
torch~=1.11.0
numpy~=1.21.5
scipy~=1.7.3
matplotlib~=3.4.1
torchvision~=0.12.0
tqdm~=4.60.0
geom_median

# License :
When using any code in this project, we would appreciate it if you could refer to this project.

# Contact :
Please send an email to reda.bellafqira@imt-atlantique.fr if you have any questions.
