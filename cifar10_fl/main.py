import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
from tqdm import tqdm
from utils.supcon_loss_util import SupConLoss
from torchvision import datasets, transforms
import argparse
import wandb
from utils.general_utils import set_seed
import kornia.augmentation as K

DATA_DIR = '/IITP-Med-AI/datasets/cifar10'

contrast_transform = nn.Sequential(
    K.RandomResizedCrop((32, 32), scale=(0.5, 1.0)),
    K.RandomHorizontalFlip(),
    K.ColorJitter(0.4, 0.4, 0.4, 0.1),
    K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.5),
    K.Normalize(mean=torch.tensor([0.4914, 0.4822, 0.4465]),
                std=torch.tensor([0.2470, 0.2435, 0.2616]))
)

# no strong augmentations for vanilla version
fed_avg_transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2470, 0.2435, 0.2616))])


# Dirichlet partitioning
def partition_data_dirichlet(labels, num_clients, alpha):
    labels = np.array(labels)
    num_classes = labels.max() + 1
    idx_per_class = [np.where(labels == i)[0] for i in range(num_classes)]
    client_idx = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        np.random.shuffle(idx_per_class[c])
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idx_per_class[c])).astype(int)[:-1]
        splits = np.split(idx_per_class[c], proportions)
        for i in range(num_clients):
            client_idx[i] += splits[i].tolist()
    return client_idx


class ResNetWithHead(nn.Module):
    def __init__(self, backbone_name, pretrained, num_classes=10, proj_dim=128):
        super(ResNetWithHead, self).__init__()
        assert backbone_name in ['resnet18', 'resnet50'], f'Backbone {backbone_name} not supported'
        if backbone_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        feature_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.projection_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, proj_dim)
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_embed=False):
        feats = self.backbone(x)
        logits = self.classifier(feats)
        if return_embed:
            emb = self.projection_head(feats)
            emb = F.normalize(emb, dim=1)
            return logits, emb
        else:
            return logits


# --- NEW FUNCTION ---
def kd_loss_function(student_logits, teacher_logits, temperature):
    """
    Knowledge distillation loss using KL divergence.
    """
    student_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)

    # Use 'batchmean' reduction for KLDivLoss
    # (T^2) factor scales gradients, standard practice in KD
    loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    return loss


def local_update_supcon(model, train_loader, optimizer, device, epochs=1):
    model.train()
    criterion_ce = nn.CrossEntropyLoss()
    criterion_supcon = SupConLoss(temperature=0.07, device=device)
    contrast_transform.to(device)

    for epoch in range(epochs):
        for images, labels in train_loader:
            labels = labels.to(device)
            images_tensor = torch.stack([transforms.ToTensor()(img) for img in images]).to(device)

            img1 = contrast_transform(images_tensor)
            img2 = contrast_transform(images_tensor)
            logits1, emb1 = model(img1, return_embed=True)
            logits2, emb2 = model(img2, return_embed=True)

            features = torch.stack([emb1, emb2], dim=1)

            loss_contrast = criterion_supcon(features, labels)
            loss_ce1 = criterion_ce(logits1, labels)
            loss_ce2 = criterion_ce(logits2, labels)
            loss = loss_contrast + loss_ce1 + loss_ce2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.state_dict()


# --- NEW FUNCTION ---
def local_update_flism(model, global_model_teacher, train_loader, optimizer, device, epochs, kd_lambda, kd_temp):
    """
    Local update for FLISM: SupCon + CE + Global-to-Local KD
    """
    model.train()
    global_model_teacher.eval()  # Teacher model is in eval mode

    criterion_ce = nn.CrossEntropyLoss()
    criterion_supcon = SupConLoss(temperature=0.07, device=device)
    contrast_transform.to(device)

    for epoch in range(epochs):
        for images, labels in train_loader:
            labels = labels.to(device)
            images_tensor = torch.stack([transforms.ToTensor()(img) for img in images]).to(device)

            # Apply same augmentations
            img1 = contrast_transform(images_tensor)
            img2 = contrast_transform(images_tensor)

            # --- Student Pass (with gradients) ---
            logits1, emb1 = model(img1, return_embed=True)
            logits2, emb2 = model(img2, return_embed=True)

            # --- Teacher Pass (no gradients) ---
            with torch.no_grad():
                teacher_logits1, _ = global_model_teacher(img1, return_embed=True)
                teacher_logits2, _ = global_model_teacher(img2, return_embed=True)

            # --- Calculate Losses ---

            # 1. SupCon + CE Loss
            features = torch.stack([emb1, emb2], dim=1)
            loss_contrast = criterion_supcon(features, labels)
            loss_ce1 = criterion_ce(logits1, labels)
            loss_ce2 = criterion_ce(logits2, labels)
            loss_supcon_ce = loss_contrast + loss_ce1 + loss_ce2

            # 2. Knowledge Distillation Loss
            loss_kd1 = kd_loss_function(logits1, teacher_logits1, kd_temp)
            loss_kd2 = kd_loss_function(logits2, teacher_logits2, kd_temp)
            loss_kd = (loss_kd1 + loss_kd2) / 2.0

            # 3. Combined Loss
            loss = (1.0 - kd_lambda) * loss_supcon_ce + kd_lambda * loss_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model.state_dict()


def local_update_fedavg(model, train_loader, optimizer, device, epochs=1):
    model.train()
    criterion_ce = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for images, labels in train_loader:
            labels = labels.to(device)
            images = torch.stack([fed_avg_transform_train(img).to(device) for img in images])

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion_ce(outputs, labels)

            loss.backward()
            optimizer.step()
    return model.state_dict()


def pil_collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            pred = outputs.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return correct / total


def average_weights(client_weights):
    avg_weights = copy.deepcopy(client_weights[0])
    for k in avg_weights.keys():
        if avg_weights[k].dtype in [torch.float32, torch.float64, torch.float16]:
            avg_weights[k] = torch.stack([w[k] for w in client_weights], 0).mean(0)
    return avg_weights


def weighted_average_weights(client_weights, client_weight_coeffs):
    avg_weights = copy.deepcopy(client_weights[0])
    for k in avg_weights.keys():
        if avg_weights[k].dtype in [torch.float32, torch.float64, torch.float16]:
            stacked_weights = torch.stack([w[k].to(client_weight_coeffs.device) for w in client_weights], 0)
            coeffs_shape = [-1] + [1] * (stacked_weights.dim() - 1)
            coeffs = client_weight_coeffs.view(*coeffs_shape)
            avg_weights[k] = torch.sum(stacked_weights * coeffs, dim=0)
    return avg_weights


def calculate_entropy(model, loader, transform, device):
    model.eval()
    total_entropy = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = torch.stack([transform(img) for img in images]).to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            log_probs = torch.log(probs + 1e-8)
            entropy_batch = -torch.sum(probs * log_probs, dim=1)
            total_entropy += entropy_batch.sum().item()
            total_samples += labels.size(0)
    model.train()
    return total_entropy / total_samples if total_samples > 0 else 0


def federated_training(args):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2470, 0.2435, 0.2616))])

    train_data = datasets.CIFAR10(root=args.data_dir, train=True, download=True,
                                  transform=transforms.Lambda(lambda x: x))
    test_data = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)

    client_idx = partition_data_dirichlet(train_data.targets, args.num_users, args.alpha)

    client_loaders = [DataLoader(Subset(train_data, idx), batch_size=args.batch_size,
                                 shuffle=True, collate_fn=pil_collate_fn)
                      for idx in client_idx]
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    global_model = ResNetWithHead(args.backbone, args.pretrained, num_classes=10).to(device)
    global_model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
    global_model.backbone.maxpool = torch.nn.Identity()

    m = max(int(args.frac * args.num_users), 1)

    for round in range(1, args.rounds + 1):
        client_weights = []
        client_entropies = []  # Used by fedavg_supcon_entr and flism
        selected_clients = np.random.choice(range(args.num_users), m, replace=False)
        print('Round [{}/{}]: Clients {}'.format(round, args.rounds, selected_clients))

        for idx in tqdm(selected_clients):
            loader = client_loaders[idx]
            local_model = copy.deepcopy(global_model)
            optimizer = torch.optim.SGD(local_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

            # --- MODIFIED SECTION: Local Training ---
            if args.method == 'flism':
                global_model_teacher = copy.deepcopy(global_model).eval()  # Teacher is global model from start of round
                local_weights = local_update_flism(
                    local_model, global_model_teacher, loader, optimizer,
                    device, args.local_epochs, args.kd_lambda, args.kd_temp
                )
            elif args.method == 'fedavg_supcon' or args.method == 'fedavg_supcon_entr':
                local_weights = local_update_supcon(local_model, loader, optimizer, device, args.local_epochs)
            else:  # fedavg
                local_weights = local_update_fedavg(local_model, loader, optimizer, device, args.local_epochs)

            client_weights.append(copy.deepcopy(local_weights))

            # If using an entropy method, calculate entropy on the updated local model
            if args.method == 'fedavg_supcon_entr' or args.method == 'flism':
                entropy = calculate_entropy(local_model, loader, transform_test, device)
                client_entropies.append(entropy)
            # --- END MODIFIED SECTION ---

        # --- MODIFIED SECTION: Aggregation ---
        if args.method == 'fedavg_supcon_entr' or args.method == 'flism':
            entropies_tensor = torch.tensor(client_entropies, device=device).float()
            # Weight is inversely proportional to entropy: w_i = (1 / H_i)
            client_weight_coeffs = 1.0 / (entropies_tensor + 1e-8)
            # Normalize weights: w_i = w_i / sum(w)
            client_weight_coeffs = client_weight_coeffs / client_weight_coeffs.sum()

            print(f"Client aggregation weights: {client_weight_coeffs.cpu().numpy()}")
            global_weights = weighted_average_weights(client_weights, client_weight_coeffs)

        else:  # fedavg or fedavg_supcon
            global_weights = average_weights(client_weights)
        # --- END MODIFIED SECTION ---

        global_model.load_state_dict(global_weights)

        if round % args.eval_interval == 0 or round == 1:
            accuracy = evaluate(global_model, test_loader, device)
            print(f'Round [{round}/{args.rounds}] Test Accuracy: {accuracy:.4f}')
            if args.wandb:
                wandb.log({'test_acc': accuracy, 'round': round})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=DATA_DIR)
    parser.add_argument('--num_users', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--rounds', type=int, default=300)
    parser.add_argument('--local_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--eval_interval', type=int, default=2)
    parser.add_argument('--frac', type=float, default=0.1)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)

    # --- MODIFIED SECTION ---
    parser.add_argument('--method', type=str, default='fedavg_supcon',
                        choices=['fedavg_supcon', 'fedavg', 'fedavg_supcon_entr', 'flism'])
    # --- END MODIFIED SECTION ---

    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--note', type=str, default='')

    # Backbone-specific
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet50'])
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained model for backbone')

    # --- NEW SECTION ---
    # FLISM-specific (KD) args
    parser.add_argument('--kd_lambda', type=float, default=0.5, help='Balancing factor for KD loss')
    parser.add_argument('--kd_temp', type=float, default=2.0, help='Temperature for knowledge distillation')
    # --- END NEW SECTION ---

    args = parser.parse_args()

    set_seed(args.seed)

    if args.wandb:
        wandb.init(
            project='CIFAR10_FL_2025',
            config=vars(args),
        )

    federated_training(args)