import torch
from torch import nn
from typing import Callable
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from optimizer import default_optimizer, default_lr_scheduler, CosineLRS
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
                 eval_loader=None):
        self.student = student
        self.criterion = loss_function if loss_function is not None else default_loss
        self.optimizer = optimizer if optimizer is not None else default_optimizer(self.student)
        self.scheduler = scheduler if scheduler is not None else CosineLRS(self.optimizer)
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
              total_epoch=200,
              fp16=False,
              ):
        '''

        :param total_epoch:
        :param step_each_epoch: this 2 parameters is just a convention, for when output loss and acc, etc.
        :param fp16:
        :param generating_data_configuration:
        :return:
        '''
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        self.student.train()
        # lambda_1, lambda_2 = [], []
        for epoch in range(1, total_epoch + 1):
            # now_lambda_1, now_lambda_2 = [], []
            # from Normalizations import ASRNormBN
            # for module in self.student.modules():
            #     if isinstance(module, ASRNormBN):
            #         now_lambda_1.append(torch.sum(torch.sigmoid(module.lambda_1)).item() / module.lambda_1.shape[0])
            #         now_lambda_2.append(torch.sum(torch.sigmoid(module.lambda_2)).item() / module.lambda_2.shape[0])
            # lambda_1.append(sum(now_lambda_1) / len(now_lambda_1))
            # lambda_2.append(sum(now_lambda_2) / len(now_lambda_2))
            train_loss = 0
            train_acc = 0

            validation_loss = 0
            validation_acc = 0

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
                    nn.utils.clip_grad_value_(self.student.parameters(), 0.1)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_value_(self.student.parameters(), 0.1)
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
                    validation_acc += (torch.sum(pre == y).item()) / y.shape[0]
                    validation_loss += loss.item()

                    if step % 10 == 0:
                        vbar.set_postfix_str(f'loss={validation_loss / step}, acc={validation_acc / step}')

                validation_loss /= len(validation_loader)
                validation_acc /= len(validation_loader)

            # self.scheduler.step(train_loss)
            self.scheduler.step(epoch)

            print(f'epoch {epoch}, train_loss = {train_loss}, train_acc = {train_acc}')
            print(f'epoch {epoch}, validation_loss = {validation_loss}, validation_acc = {validation_acc}')

            torch.save(self.student.state_dict(), 'student.pth')

        # from matplotlib import pyplot as plt
        # plt.plot(list(range(len(lambda_1))), lambda_1)
        # plt.legend('lambda_1')
        # plt.plot(list(range(len(lambda_2))), lambda_2)
        # plt.legend('lambda_2')
        # plt.xlabel('epoch')
        # plt.ylabel('mean')
        # plt.show()
        # plt.savefig('1')



if __name__ == '__main__':
    from backbones import mobilenetV2, ShuffleV2, resnet32
    from torchvision.models import resnet50
    from Normalizations import ASRNormBN2d, ASRNormIN, ASRNormBN1d, ASRNormLN

    a = ShuffleV2(num_classes=10, norm_layer=ASRNormIN)
    from data import get_CIFAR100_train, get_CIFAR100_test, get_someset_loader, \
        get_CIFAR10_train, get_CIFAR10_test

    train_loader, validation_loader = get_CIFAR10_train(batch_size=256, augment=True)
    test_loader = get_CIFAR10_test(batch_size=256)

    w = Solver(a)
    w.train(train_loader, validation_loader)

    from tester import test_acc
    test_acc(w.student, test_loader)
