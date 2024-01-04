import unittest
import torch
from Models.MLP import MLP
from Configs.cf_fedCAM import configs_fedCAM
from Configs.cf_fedCVAE import configs_fedCVAE
import argparse

# Set the device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TestMain(unittest.TestCase):

    def test_main(self, algo="fedCam"):
        # Test for the fedCam algorithm
        if algo == "fedCam":
            from Defenses.FedCAM import Server
            model = MLP(configs_fedCAM["cvae_input_dim"]).to(device)
            server = Server(cf=configs_fedCAM, model=model)
            server.run()
        # Test for the fedCvae algorithm
        elif algo == "fedCvae":  # FedCVAE in this case
            from Defenses.FedCVAE import Server
            model = MLP(configs_fedCVAE["activation_size"]).to(device)
            server = Server(cf=configs_fedCVAE, model=model)
            server.run()
        else:
            # Print a message if the algorithm argument is not valid
            print("Please specify a valid -algo argument (e.g., fedCam, fedCvae)")

if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="This script corresponds to the implementation of FedCVAE and FedCAM")

    # Add an -algo argument to specify the algorithm
    parser.add_argument("-algo", type=str, help="The name of the defense system")

    # Parse the command line arguments
    args = parser.parse_args()

    # Create an instance of the TestMain class
    test_instance = TestMain()

    # Call the test_main function with the specified algorithm from the arguments
    if args.algo:
        test_instance.test_main(algo=args.algo)
    else:
        # Print a message if the -algo argument is not specified in the command line
        print("Please specify the -algo argument in the command line.")
