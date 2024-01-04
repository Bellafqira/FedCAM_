import copy
import os
import numpy as np
import torch
import torch.nn.functional as F

from geom_median.torch import compute_geometric_median

from Models.autoencoders import CVAE
from Models.MLP import MLP
from Utils.Utils import Utils


class Server:
    def __init__(self, cf=None, model=None):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cf = cf

        # Saving directory
        self.dir_path = f"Results/FedCVAE/{self.cf['data_dist']}_{int(self.cf['attacker_ratio'] * 100)}_{self.cf['attack_type']}"
        self.dir_path += "/Defence" if self.cf["with_defence"] else "/NoDefence"
        os.makedirs(self.dir_path, exist_ok=True)

        self.eps = cf["eps"]
        self.nb_rounds = cf["nb_rounds"]
        self.activation_size = cf["activation_size"]

        self.global_model = model.to(self.device) if model else MLP(self.activation_size).to(self.device)

        self.defence = cf["with_defence"]
        self.attack_type = cf["attack_type"]

        self.config_FL = {
            "num_clients": cf["num_clients"],
            "attackers_ratio": cf["attacker_ratio"],
            "nb_clients_per_round": cf["nb_clients_per_round"],
            "batch_size": cf["batch_size"]
        }

        self.train_data = Utils.distribute_iid_data_among_clients(self.config_FL["num_clients"], cf["batch_size"])
        self.clients = Utils.gen_clients(self.config_FL, self.attack_type, self.train_data)

        self.validation_loader, self.test_loader = Utils.get_test_data(cf["validation_size"])

        # Done
        total_weights = sum(param.numel() for param in self.global_model.parameters()  if param.dim() > 1)

        # selecting the indices that will be fed to the CVAE
        self.indices = np.random.choice(total_weights, self.cf["selected_weights_dim"], replace=False)

        self.cvae = CVAE(
            input_dim=self.cf["selected_weights_dim"],
            condition_dim=self.cf["condition_dim"],
            hidden_dim=self.cf["hidden_dim"],
            latent_dim=self.cf["latent_dim"]
        ).to(self.device)

        self.accuracy = []
        self.accuracy_backdoor = []
        self.nb_attackers_history = []
        self.nb_attackers_passed_defence_history = []
        self.nb_benign_history = []
        self.nb_benign_passed_defence_history = []

        self.histo_selected_clients = torch.tensor([])

    def one_hot_encoding(self, current_round):
        one_hot = torch.zeros(self.cf['condition_dim']).to(self.device)
        one_hot[current_round] = 1.0
        return one_hot

    def gen_surrogate_vectors(self, selected_clients):

        surrogate_vectors = [torch.cat([p.data.view(-1) for p in client.model.parameters() if p.dim() > 1])[
                self.indices].detach().cpu() for client in selected_clients]

        return surrogate_vectors

    def process_surrogate_vectors(self, surrogate_vectors):
        geo_median = compute_geometric_median(surrogate_vectors, weights=None, eps=self.cf["eps"], maxiter=self.cf["iter"])  # equivalent to `weights = torch.ones(n)`.
        geo_median = geo_median.median
        processed_vectors = [surrogate_vector - geo_median for surrogate_vector in surrogate_vectors]
        return processed_vectors

    def compute_reconstruction_error(self, processed_vectors, current_round):
        self.cvae.eval()
        clients_re = []
        condition = self.one_hot_encoding(current_round).unsqueeze(0).to(self.device)

        for processed_vector in processed_vectors:
            processed_vector = processed_vector.unsqueeze(0).to(self.device)
            recon_batch, _, _ = self.cvae(processed_vector, condition)
            mse = F.mse_loss(recon_batch, processed_vector, reduction='mean').item()
            clients_re.append(mse)

        return clients_re

    def run(self):
        for rounds in range(self.cf["nb_rounds"]):
            selected_clients = Utils.select_clients(self.clients, self.config_FL["nb_clients_per_round"])

            for client in selected_clients:
                client.set_model(copy.deepcopy(self.global_model).to(self.device))
                client.train(self.cf)

            if self.defence:
                surrogate_weights = self.gen_surrogate_vectors(selected_clients)
                processed_vectors = self.process_surrogate_vectors(surrogate_weights)

                clients_re = self.compute_reconstruction_error(processed_vectors, rounds)
                clients_re_np = np.array(clients_re)

                mean_of_re = np.mean(clients_re_np)

                selected_clients_np = np.array(selected_clients)
                good_updates = selected_clients_np[clients_re_np < mean_of_re]

            else:
                good_updates = selected_clients

            self.histo_selected_clients = torch.cat((self.histo_selected_clients,
                                                     torch.tensor([client.id for client in good_updates])))

            nb_attackers = np.array([client.is_attacker for client in selected_clients]).sum()
            nb_benign = np.array([not client.is_attacker for client in selected_clients]).sum()

            nb_attackers_passed = np.array([client.is_attacker for client in good_updates]).sum()
            nb_benign_passed = np.array([not client.is_attacker for client in good_updates]).sum()

            self.nb_attackers_history.append(nb_attackers)
            self.nb_attackers_passed_defence_history.append(nb_attackers_passed)
            self.nb_benign_history.append(nb_benign)
            self.nb_benign_passed_defence_history.append(nb_benign_passed)

            print("Total of Selected Clients ", len(selected_clients), ", Number of attackers ", nb_attackers,
                  ", Total of attackers passed defense ", nb_attackers_passed, " from ", len(good_updates))

            # Aggregation step
            self.global_model.load_state_dict(Utils.aggregate_models(good_updates))

            # Add the accuracy of the current global model to the accuracy list
            self.accuracy.append(Utils.test(self.global_model, self.device, self.test_loader))

            if self.attack_type == "NaiveBackdoor" or self.attack_type == "SquareBackdoor":
                self.accuracy_backdoor.append(Utils.test_backdoor(self.global_model, self.device, self.test_loader,
                                                                  self.attack_type, self.cf["source"],
                                                                  self.cf["target"], self.cf["square_size"]))

            print(f"Round {rounds + 1}/{self.cf['nb_rounds']} server accuracy: {self.accuracy[-1] * 100:.2f}%")
            if self.attack_type == "NaiveBackdoor" or self.attack_type == "SquareBackdoor":
                print(
                    f"Round {rounds + 1}/{self.cf['nb_rounds']} attacker accuracy: {self.accuracy_backdoor[-1] * 100:.2f}%")



        # Saving The accuracies of the Global model on the testing set and the backdoor set
        Utils.save_to_json(self.accuracy, self.dir_path, f"test_accuracy_{self.cf['nb_rounds']}")
        if self.attack_type == "NaiveBackdoor" or self.attack_type == "SquareBackdoor":
            Utils.save_to_json(self.accuracy_backdoor, self.dir_path, f"{self.attack_type}_accuracy_{self.cf['nb_rounds']}")

        save_path = f"{self.dir_path}/{self.attack_type}_{'With defence' if self.defence else 'No defence'}_clients_hist_{self.cf['nb_rounds']}.pdf"
        Utils.plot_hist(self.histo_selected_clients, x_info="Clients", y_info="Frequencies", title_info="", bins=1000,
                        save_path=save_path)

        # Plotting the testing accuracy of the global model
        title_info = f"Test Accuracy per Round for {self.cf['attacker_ratio'] * 100}% of {self.attack_type} with {('Defence' if self.defence else 'No Defence')}"
        save_path = f"{self.dir_path}/{self.attack_type}_{'With defence' if self.defence else 'No defence'}_Test_Accuracy_{self.cf['nb_rounds']}.pdf"
        Utils.plot_accuracy(self.accuracy, x_info='Round', y_info='Test Accuracy', title_info=title_info,
                            save_path=save_path)

        if self.attack_type == "NaiveBackdoor" or self.attack_type == "SquareBackdoor":
            # Plotting the backdoor accuracy
            title_info = f"Backdoor Accuracy per Round for {self.cf['attacker_ratio'] * 100}% of {self.attack_type} with {('Defence' if self.defence else 'No Defence')}"
            save_path = f"{self.dir_path}/{self.attack_type}_{'With defence' if self.defence else 'No defence'}_Backdoor_Accuracy_{self.cf['nb_rounds']}.pdf"
            Utils.plot_accuracy(self.accuracy_backdoor, x_info='Round', y_info='backdoor Accuracy',
                                title_info=title_info, save_path=save_path)

        # Plotting the histogram of the defense system
        Utils.plot_histogram(self.cf, self.nb_attackers_passed_defence_history, self.nb_attackers_history,
                             self.nb_benign_passed_defence_history, self.nb_benign_history, self.config_FL,
                             self.attack_type, self.defence, self.dir_path)

