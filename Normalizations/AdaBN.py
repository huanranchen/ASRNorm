from train import Model
import torch
from data.data import get_test_loader
from tqdm import tqdm
from data.dataUtils import write_result


from train import Model
import torch
from data.data import get_test_loader
from tqdm import tqdm
from data.dataUtils import write_result


def train(batch_size=64, total_epoch=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    state = torch.load('adabn.pth', map_location=device)
    model.model.load_state_dict(state['model_state'])
    train_image_path='../input/nico2022/track_1/public_dg_0416/train/'
    valid_image_path='../input/nico2022/track_1/public_dg_0416/train/'
    label2id_path='../input/nico2022/dg_label_id_mapping.json'
    test_image_path='../input/nico2022/track_1/public_dg_0416/public_test_flat/'
    loader,_ = get_test_loader(batch_size=batch_size,
                             transforms=None,
                             label2id_path=label2id_path,
                             test_image_path=test_image_path)
    model.train()
    with torch.no_grad():
        for epoch in range(1, total_epoch + 1):
            # train
            pbar = tqdm(loader)
            for x, _ in pbar:
                x=x.to(device)
                x = model(x)

    model.eval()
    result = {}
    with torch.no_grad():
        for x, name in tqdm(loader):
            x = x.to(device)
            y = model(x)  # N, D
            _, y = torch.max(y, dim=1)  # (N,)

            for i, name in enumerate(list(name)):
                result[name] = y[i].item()

    write_result(result)
    torch.save(model.model.state_dict(), 'model.pth')


if __name__ == '__main__':
    train()