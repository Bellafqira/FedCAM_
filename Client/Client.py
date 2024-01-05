import torch

class Client:
    def __init__(self, ids=None, is_attacker=False, dataloader=None, model=None, attack_type=None, device=None) -> None:
        self.id = ids
        self.is_attacker = is_attacker
        self.dataloader = dataloader
        self.model = model
        self.attack_type = attack_type
        self.device = device
        self.num_samples = len(dataloader.dataset) if dataloader else 0

    def get_id(self):
        return self.id

    def is_attacker(self):
        return self.is_attacker

    def set_attack_type(self, attack_type):
        self.attack_type = attack_type

    def get_attack_type(self):
        return self.attack_type

    def get_model(self):
        return self.model

    def set_model(self, model):
        self.model = model.to(self.device)

    def set_data(self, data):
        self.dataloader = data
        self.num_samples = len(data.dataset)

    def train(self, hp):
        if self.model is None:
            raise ValueError("The model is not set. Use set_model method to set the model.")

        if self.is_attacker and self.attack_type not in ["NoAttack", "NaiveBackdoor", "SquareBackdoor"]:
            self.apply_attack()
            return self.model

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        for epoch in range(hp["num_epochs"]):
            for data, labels in self.dataloader:
                data, labels = data.to(device), labels.to(device)

                if self.is_attacker and self.attack_type == "NaiveBackdoor":
                    labels[labels == hp["source"]] = hp["target"]
                if self.is_attacker and self.attack_type == "SquareBackdoor":
                    data, labels = square_backdoor(data, labels, hp["source"], hp["target"], hp["square_size"])

                outputs = self.model(data)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        return self.model

    def apply_attack(self):
        if self.attack_type == "AdditiveNoise":
            self.additive_noise()
        elif self.attack_type == "SameValue":
            self.same_value()
        elif self.attack_type == "SignFlip":
            self.sign_flip()
        else:
            raise ValueError("Unknown or unsupported attack type for direct parameter manipulation.")

    def additive_noise(self):
        for param in self.model.parameters():
            noise = torch.normal(mean=0.0, std=20, size=param.data.shape, device=param.device)
            param.data += noise

    def same_value(self):
        for param in self.model.parameters():
            param.data.fill_(100)

    def sign_flip(self):
        for param in self.model.parameters():
            param.data *= -4


def square_backdoor(data, labels, source, target, square_size):
    # Create a white square
    data = data.clone()
    labels = labels.clone()
    data[labels == source][:, 0, :square_size, :square_size] = 1.0
    labels[labels == source] = target
    return data, labels