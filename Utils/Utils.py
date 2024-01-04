import os
import json
import numpy as np
import random
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets
from torchvision.transforms import transforms
import torch.nn.functional as F

from Client.Client import Client

class Utils:
    def __init__(self):
        pass

    @staticmethod
    def distribute_iid_data_among_clients(num_clients, batch_size):
        mnist = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        data_size = len(mnist) // num_clients
        return [
            DataLoader(Subset(mnist, range(i * data_size, (i + 1) * data_size)), batch_size=batch_size, shuffle=True)
            for i in range(num_clients)]

    @staticmethod
    def gen_database(num_clients, batch_size):
        return Utils.distribute_iid_data_among_clients(num_clients, batch_size)

    @staticmethod
    def gen_clients(config_fl, attack_type, train_data):
        total_clients = config_fl["num_clients"]
        num_attackers = int(total_clients * config_fl["attackers_ratio"])

        attacker_flags = [True] * num_attackers + [False] * (total_clients - num_attackers)
        np.random.shuffle(attacker_flags)

        clients = [Client(ids=i, dataloader=train_data[i], is_attacker=attacker_flags[i], attack_type=attack_type)
                   for i in range(total_clients)]
        return clients


    @staticmethod
    def cvae_loss(recon_x, x, mu, logvar):
        mse = F.mse_loss(recon_x, x, reduction='mean')
        # MSE = F.binary_cross_entropy(recon_x, x, reduction='mean')
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + kld


    @staticmethod
    def plot_accuracy(accuracy, x_info="round", y_info="Test Accuracy", title_info= "provide a title", save_path=None):
        plt.plot(range(1, len(accuracy) + 1), accuracy)
        plt.xlabel(x_info)
        plt.ylabel(y_info)
        plt.title(title_info)
        plt.savefig(save_path)
        plt.show()

    @staticmethod
    def plot_hist(data, x_info="Values", y_info="Frequencies", title_info= "provide a title", bins=1000, save_path=None):
        plt.title(title_info)
        plt.xlabel(x_info)
        plt.ylabel(y_info)
        plt.hist(data, bins=bins)
        plt.savefig(save_path)
        plt.show()


    @staticmethod
    def plot_histogram(hp, nb_attackers_passed_defence_history, nb_attackers_history,
                       nb_benign_passed_defence_history, nb_benign_history, config_fl,
                       attack_type, defence, dir_path):
        rounds = np.arange(1, hp["nb_rounds"] + 1)

        height_attackers_passed_defense = np.array(nb_attackers_passed_defence_history)
        height_remaining_attackers = np.array(nb_attackers_history) - height_attackers_passed_defense

        height_benign_passed_defense = np.array(nb_benign_passed_defence_history)
        height_remaining_benign = np.array(nb_benign_history) - height_benign_passed_defense

        plt.bar(rounds, height_attackers_passed_defense, color='red', edgecolor='black', alpha=0.5,
                label='Attackers Passed Defence')
        plt.bar(rounds, height_remaining_attackers, bottom=height_attackers_passed_defense, color='yellow',
                edgecolor='black', alpha=0.6, label='Total Attackers')

        plt.bar(rounds, height_benign_passed_defense, bottom=height_attackers_passed_defense + height_remaining_attackers, color='blue', edgecolor='black', alpha=0.5,
                label='Benign Clients Passed Defence')

        plt.bar(rounds, height_remaining_benign, bottom=height_benign_passed_defense + height_attackers_passed_defense + height_remaining_attackers, color='black',
                edgecolor='black', alpha=0.6, label='Total Benign Clients')

        plt.xlabel('Number of Rounds')
        plt.ylabel('Total Nb of Clients')
        plt.ylim(0, config_fl["nb_clients_per_round"])
        plt.title(f"Histogram for {hp['attacker_ratio'] * 100}% of {attack_type} "
                  f"with {'Defence' if defence else 'No Defence'}")

        plt.legend()
        plt.savefig(f"{dir_path}/{attack_type}_{'With defence' if defence else 'No defence'}_Histogram_{hp['nb_rounds']}.pdf")

        plt.show()

    @staticmethod
    def test(global_model, device, test_loader):
        global_model.to(device).eval()
        correct, total = 0, 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                output = global_model(data)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return accuracy

    @staticmethod
    def test_backdoor(global_model, device, test_loader, attack_type, source, target, square_size):
        global_model.to(device).eval()
        total_source_labels, misclassified_as_target = 0, 0
        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)

                tmp_data = data[labels==source].clone()
                tmp_labels = labels[labels==source].clone()

                if len(tmp_data) != 0 :
                    if attack_type == 'SquareBackdoor':
                        tmp_data[:, 0, :square_size, :square_size] = 1.0

                    outputs = global_model(tmp_data)
                    _, predicted = torch.max(outputs, 1)

                    total_source_labels += tmp_labels.size(0)
                    misclassified_as_target += (predicted == target).sum().item()

        effectiveness = misclassified_as_target / total_source_labels if total_source_labels > 0 else 0
        return effectiveness

    @staticmethod
    def get_test_data(size_trigger):
        mnist_test = datasets.MNIST(root='./data',
                                    train=False,
                                    transform=transforms.ToTensor(),
                                    download=True)

        size_test = len(mnist_test) - size_trigger
        trigger_set, validation_set = random_split(mnist_test, [size_trigger, size_test])
        # Create data loaders
        trigger_loader = DataLoader(trigger_set, batch_size=size_trigger, shuffle=False)
        test_loader = DataLoader(validation_set, batch_size=size_test, shuffle=False)
        return trigger_loader, test_loader

    @staticmethod
    def select_clients(clients, nb_clients_per_round):
        selected_clients = random.sample(clients, nb_clients_per_round)
        return selected_clients


    @staticmethod
    def save_to_json(accuracies, dir_path, file_name):
        file_name = f"{dir_path}/{file_name}.json"
        with open(file_name, "w") as f:
            json.dump(accuracies, f)

    @staticmethod
    def read_from_json(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data

    @staticmethod
    def aggregate_models(clients):
        aggregated_state_dict = {}
        total_samples = sum([client.num_samples for client in clients])

        # Initialize
        for name, param in clients[0].model.state_dict().items():
            aggregated_state_dict[name] = torch.zeros_like(param).float()
        # Aggregate the clients' models
        for client in clients:
            num_samples = client.num_samples
            weight_factor = num_samples / total_samples
            client_state_dict = client.get_model().state_dict()

            for name, param in client_state_dict.items():
                aggregated_state_dict[name] += weight_factor * param
        return aggregated_state_dict

    @staticmethod
    def one_hot_encoding(label, num_classes, device):
        one_hot = torch.eye(num_classes).to(device)[label]
        return one_hot.squeeze(1).to(device)

    # ****** Functions related to FedCVAE

    @staticmethod
    def get_prod_size(model):
        size = 0
        for param in model.parameters():
            size += np.prod(param.weight.shape)
        return size


