from copy import deepcopy
import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
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
        self.dir_path = f"Results/FedCAM/{self.cf['data_dist']}_{int(self.cf['attacker_ratio'] * 100)}_{self.cf['attack_type']}"
        self.dir_path += "/Defence" if self.cf["with_defence"] else "/NoDefence"
        os.makedirs(self.dir_path, exist_ok=True)

        self.activation_size = cf["cvae_input_dim"]
        self.num_classes = cf["num_classes"]
        self.nb_rounds = cf["nb_rounds"]
        self.global_model = model.to(self.device) if model else MLP(self.activation_size).to(self.device)
        self.defence = cf["with_defence"]
        self.attack_type = cf["attack_type"]

        self.config_FL = {
            "num_clients": cf["num_clients"],
            "attackers_ratio": cf["attacker_ratio"],
            "nb_clients_per_round": cf["nb_clients_per_round"],
            "batch_size": cf["batch_size"]
        }

        self.config_cvae = {
            "cvae_nb_ep": cf["cvae_nb_ep"],
            "cvae_lr": cf["cvae_lr"],
            "cvae_wd": cf["cvae_wd"],
            "cvae_gamma": cf["cvae_gamma"],
        }

        self.train_data = Utils.distribute_iid_data_among_clients(self.config_FL["num_clients"], cf["batch_size"])
        self.clients = Utils.gen_clients(self.config_FL, self.attack_type, self.train_data)
        self.cvae_trained = False

        self.trigger_loader, self.test_loader = Utils.get_test_data(cf["size_trigger"])

        self.cvae = CVAE(
            input_dim=cf["cvae_input_dim"],
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

    def train_cvae(self):
        if self.cvae_trained:
            print("CVAE is already trained, skipping re-training.")
            return

        num_epochs = self.config_cvae["cvae_nb_ep"]
        optimizer = torch.optim.Adam(self.cvae.parameters(), lr=self.config_cvae["cvae_lr"], weight_decay=self.config_cvae["cvae_wd"])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 300], gamma=self.config_cvae["cvae_gamma"])

        model = deepcopy(self.global_model)
        model.eval()

        for epoch in range(num_epochs):
            train_loss = 0
            loop = tqdm(self.trigger_loader, leave=True)

            for batch_idx, (data, label) in enumerate(loop):
                data, label = data.to(self.device), label.to(self.device)
                activation = self.global_model.get_activations(data)
                activation = torch.sigmoid(activation)

                condition = Utils.one_hot_encoding(label, self.num_classes, self.device)

                recon_batch, mu, logvar = self.cvae(activation, condition)
                loss = Utils.cvae_loss(recon_batch, activation, mu, logvar)

                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(loss=train_loss / (batch_idx + 1))

            scheduler.step()
        self.cvae_trained = True

    def compute_reconstruction_error(self, selected_clients):
        self.cvae.eval()

        clients_re = []

        clients_act = torch.zeros(size=(len(selected_clients), self.cf["size_trigger"], self.activation_size)).to(self.device)
        labels_cat = torch.tensor([]).to(self.device)

        for client_nb, client_model in enumerate(selected_clients):
            labels_cat = torch.tensor([]).to(self.device)
            for data, label in self.trigger_loader:
                data, label = data.to(self.device), label.to(self.device)
                activation = client_model.model.get_activations(data)
                clients_act[client_nb] = activation
                labels_cat = label
                break

        gm = compute_geometric_median(clients_act.cpu(), weights=None)
        clients_act = clients_act - gm.median.to(self.device)
        clients_act = torch.abs(clients_act)
        clients_act = torch.sigmoid(clients_act)

        for client_act in clients_act:
            condition = Utils.one_hot_encoding(labels_cat, self.num_classes, self.device).to(self.device)
            recon_batch, _, _ = self.cvae(client_act, condition)
            mse = F.mse_loss(recon_batch, client_act, reduction='mean').item()
            clients_re.append(mse)

        return clients_re


    def run(self):

        if self.defence:
            if not self.cvae_trained:
                self.train_cvae()
                self.cvae_trained = True

        for rounds in range(self.cf["nb_rounds"]):
            torch.cuda.empty_cache()

            selected_clients = Utils.select_clients(self.clients, self.config_FL["nb_clients_per_round"])

            for client in selected_clients:
                client.set_model(deepcopy(self.global_model).to(self.device))
                client.train(self.cf)

            if self.defence:
                clients_re = self.compute_reconstruction_error(selected_clients)
                clients_re_np = np.array(clients_re)
                valid_values = clients_re_np[np.isfinite(clients_re_np)]

                max_of_re = np.max(valid_values)
                mean_of_re = np.mean(valid_values)

                clients_re_without_nan = np.where(np.isnan(clients_re_np) | (clients_re_np == np.inf), max_of_re,
                                                  clients_re_np)

                selected_clients_array = np.array(selected_clients)
                good_updates = selected_clients_array[clients_re_without_nan < mean_of_re]
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
            Utils.save_to_json(self.accuracy_backdoor, self.dir_path,
                               f"{self.attack_type}_accuracy_{self.cf['nb_rounds']}")

        save_path = f"{self.dir_path}/{self.attack_type}_{'With defence' if self.defence else 'No defence'}_clients_hist_{self.cf['nb_rounds']}.png"
        Utils.plot_hist(self.histo_selected_clients, x_info="Clients", y_info="Frequencies", title_info="", bins=1000,
                        save_path=save_path)

        # Plotting the testing accuracy of the global model
        title_info = f"Test Accuracy per Round for {self.cf['attacker_ratio'] * 100}% of {self.attack_type} with {('Defence' if self.defence else 'No Defence')}"
        save_path = f"{self.dir_path}/{self.attack_type}_{'With defence' if self.defence else 'No defence'}_Test_Accuracy_{self.cf['nb_rounds']}.png"
        Utils.plot_accuracy(self.accuracy, x_info='Round', y_info='Test Accuracy', title_info=title_info,
                            save_path=save_path)

        if self.attack_type == "NaiveBackdoor" or self.attack_type == "SquareBackdoor":
            # Plotting the backdoor accuracy
            title_info = f"Backdoor Accuracy per Round for {self.cf['attacker_ratio'] * 100}% of {self.attack_type} with {('Defence' if self.defence else 'No Defence')}"
            save_path = f"{self.dir_path}/{self.attack_type}_{'With defence' if self.defence else 'No defence'}_Backdoor_Accuracy_{self.cf['nb_rounds']}.png"
            Utils.plot_accuracy(self.accuracy_backdoor, x_info='Round', y_info='backdoor Accuracy',
                                title_info=title_info, save_path=save_path)

        # Plotting the histogram of the defense system
        Utils.plot_histogram(self.cf, self.nb_attackers_passed_defence_history, self.nb_attackers_history,
                             self.nb_benign_passed_defence_history, self.nb_benign_history, self.config_FL,
                             self.attack_type, self.defence, self.dir_path)