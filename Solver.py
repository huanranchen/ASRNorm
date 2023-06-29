import torch
from torch import nn
from typing import Callable
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from optimizer import default_optimizer, default_lr_scheduler, CosineLRS, ALRS
from torch.utils.tensorboard import SummaryWriter


def default_loss(x, y):
    cross_entropy = F.cross_entropy(x, y)
    return cross_entropy


class Solver():
    def __init__(self, student: nn.Module,
                 loss_function: Callable or None = None,
                 optimizer: torch.optim.Optimizer or None = None,
                 scheduler=None,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 ):
        self.student = student
        self.criterion = loss_function if loss_function is not None else default_loss
        self.optimizer = optimizer if optimizer is not None else default_optimizer(self.student)
        self.scheduler = scheduler if scheduler is not None else ALRS(self.optimizer)
        self.device = device

        # initialization
        self.init()

    def init(self):
        # change device
        self.student.to(self.device)

        # # tensorboard
        # self.writer = SummaryWriter(log_dir="runs/Solver", flush_secs=120)

    def train(self,
              train_loader: DataLoader,
              validation_loader: DataLoader,
              total_epoch=500,
              fp16=False,
              ):
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        for epoch in range(1, total_epoch + 1):
            train_loss, train_acc, validation_loss, validation_acc = 0, 0, 0, 0
            self.student.train()
            # train
            pbar = tqdm(train_loader)
            for step, (x, y) in enumerate(pbar, 1):
                x, y = x.to(self.device), y.to(self.device)
                if fp16:
                    with autocast():
                        student_out = self.student(x)  # N, 60
                        _, pre = torch.max(student_out, dim=1)
                        loss = self.criterion(student_out, y)
                else:
                    student_out = self.student(x)  # N, 60
                    _, pre = torch.max(student_out, dim=1)
                    loss = self.criterion(student_out, y)
                if pre.shape != y.shape:
                    _, y = torch.max(y, dim=1)
                train_acc += (torch.sum(pre == y).item()) / y.shape[0]
                train_loss += loss.item()
                self.optimizer.zero_grad()

                if fp16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    # nn.utils.clip_grad_value_(self.student.parameters(), 0.1)
                    # nn.utils.clip_grad_norm(self.student.parameters(), max_norm=10)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    # nn.utils.clip_grad_value_(self.student.parameters(), 0.1)
                    # nn.utils.clip_grad_norm(self.student.parameters(), max_norm=10)
                    self.optimizer.step()

                if step % 10 == 0:
                    pbar.set_postfix_str(f'loss={train_loss / step}, acc={train_acc / step}')

            train_loss /= len(train_loader)
            train_acc /= len(train_loader)

            # validation
            vbar = tqdm(validation_loader, colour='yellow')
            self.student.eval()
            with torch.no_grad():
                for step, (x, y) in enumerate(vbar, 1):
                    x, y = x.to(self.device), y.to(self.device)
                    student_out = self.student(x)  # N, 60
                    _, pre = torch.max(student_out, dim=1)
                    loss = self.criterion(student_out, y)
                    if pre.shape != y.shape:
                        _, y = torch.max(y, dim=1)
                    validation_acc += (torch.sum(pre == y).item()) / y.shape[0]
                    validation_loss += loss.item()

                    if step % 10 == 0:
                        vbar.set_postfix_str(f'loss={validation_loss / step}, acc={validation_acc / step}')

                validation_loss /= len(validation_loader)
                validation_acc /= len(validation_loader)

            self.scheduler.step(train_loss, epoch)

            print(f'epoch {epoch}, train_loss = {train_loss}, train_acc = {train_acc}')
            print(f'epoch {epoch}, validation_loss = {validation_loss}, validation_acc = {validation_acc}')
            print('-' * 100)

            torch.save(self.student.state_dict(), 'student.pth')


def get_norm_layers(model:nn.Module, norm_name):
    norm_layers = []
    for module in model.modules():
        if isinstance(module, norm_name):
            norm_layers.append(module)
        elif isinstance(module, nn.ModuleList):
            for sub_module in module:
                if isinstance(sub_module, norm_name):
                    norm_layers.append(sub_module)
        elif isinstance(module, nn.Sequential):
            for sub_module in module.children():
                norm_layers.append(sub_module)
    return norm_layers


def freeze_weights(model:nn.Module, norm_name):
    for param in model.parameters():
        param.requires_grad = False
    for layer in get_norm_layers(model, norm_name):
        for param in layer.parameters():
            param.requires_grad = True


if __name__ == '__main__':
    import torchvision
    from torchvision.models import resnet18
    from data import get_PACS_train, get_PACS_test
    from backbones import pyramidnet272, pyramidnet164

    a = resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    a.fc = nn.Linear(512, 7)
    # # a = resnet18(num_classes=7)
    # a = pyramidnet164(num_classes=7)

    freeze_weights(a, nn.BatchNorm2d)


    train_loader = get_PACS_train(batch_size=64, target_domain='P')
    test_loader = get_PACS_test(batch_size=256, target_domain='P')

    w = Solver(a)
    w.train(train_loader, test_loader)
