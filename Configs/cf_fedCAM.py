# Hyperparameters for a federated learning system
# Language: Python

configs_fedCAM = {
    # Clients Settings
    "num_epochs": 14,  # Number of epochs for training
    "batch_size": 64,  # Batch size for training
    "lr": 1e-3,  # Learning rate
    "num_classes": 10,  # Number of classes in the dataset
    "wd": 1e-5,  # Weight decay for Clients model

    #FL Settings
    "data_dist": "IID", # For the moment we are considering only the IID data distribution
    "nb_rounds": 100,  # Maximum number of communication rounds for federated learning
    "aggregation": "FedAvg",  # Aggregation method for model updates
    "num_clients": 1000,  # Total number of clients in the federated learning system
    "nb_clients_per_round": 50,  # Number of clients selected for each round

    # CVAE Settings
    "condition_dim": 10,  # Dimension of the condition in CVAE
    "latent_dim": 8,  # Dimension of the latent space in CVAE
    "hidden_dim": 100,  # Dimension of the hidden layer in CVAE
    "cvae_input_dim": 64,  # Dimension of the input for CVAE and the size of the activation maps ("activation_size")
    # which is the output of FC2 in our case 128
    "cvae_nb_ep": 10,  # Number of epochs for training a CVAE model
    "cvae_lr": 1e-2,  # Learning rate for CVAE model
    "cvae_wd": 0,  # Weight decay for CVAE model 1e-5
    "cvae_gamma": 1,  # Gamma value for CVAE model

    # Attacks/Defenses Settings
    "with_defence": True,  # Flag indicating if defense mechanism is enabled
    "size_trigger": 100,  # Trigger size for defense mechanism
    "attacker_ratio": 0.3,  # Ratio of attackers in the system
    "attack_type": 'NaiveBackdoor',  # Type of attack (e.g., SameValue, AdditiveNoise)
    # 0: 'NoAttack' 1: 'AdditiveNoise', 2: 'SameValue', 3: 'SignFlip',  4: 'NaiveBackdoor', 5: 'SquareBackdoor'

    # Parameters of  NaiveBackdoor and SquareBackdoor attacks
    "source": 7,
    "target": 5,
    "square_size": 10,

    # GeoMEd parameters
    "eps": 1e-8,  # Epsilon value for numerical stability of goeMed
    "iter": 100,  # Maximum number of iterations

}