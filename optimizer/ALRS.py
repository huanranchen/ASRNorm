import torch


class ALRS():
    '''
    proposer: Huanran Chen
    theory: landscape
    Bootstrap Generalization Ability from Loss Landscape Perspective
    '''

    def __init__(self, optimizer, loss_threshold=0.01, loss_ratio_threshold=0.01, decay_rate=0.97,
                 max_epoch=200, lr_min=0, lr_max=0.1, warmup_epoch=0):
        self.optimizer = optimizer
        self.loss_threshold = loss_threshold
        self.decay_rate = decay_rate
        self.loss_ratio_threshold = loss_ratio_threshold

        self.warmup_epoch = warmup_epoch
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.max_epoch = max_epoch

        self.last_loss = 999

    def step(self, loss, current_epoch):
        if current_epoch < self.warmup_epoch:
            now_lr = self.lr_max * current_epoch / self.warmup_epoch
            for group in self.optimizer.param_groups:
                group['lr'] = now_lr
                print(f'now lr = {now_lr}')

        else:
            delta = self.last_loss - loss
            if delta < self.loss_threshold and delta / self.last_loss < self.loss_ratio_threshold:
                for group in self.optimizer.param_groups:
                    group['lr'] *= self.decay_rate
                    now_lr = group['lr']
                    print(f'now lr = {now_lr}')
            else:
                now_lr = self.optimizer.param_groups[0]['lr']
                print(f'now lr = {now_lr}')
            self.last_loss = loss
