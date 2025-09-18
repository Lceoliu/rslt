from ..training.data import build_dataloaders


def test_build_dataloaders_and_collate():
    cfg = {
        'data': {
            'batch_size': 3,
            'num_workers': 0,
            'T': 8,
            'train_length': 9,
            'val_length': 6,
        },
        'nclass': 5,
    }
    train_loader, val_loader, nclass = build_dataloaders(cfg)
    assert nclass == 5
    bt = next(iter(train_loader))
    parts, labels = bt
    assert labels.shape[0] == 3
    for k in ['body', 'face', 'left_hand', 'right_hand', 'fullbody']:
        assert k in parts
        assert parts[k].ndim == 4  # (B,T,V,C)
        assert parts[k].shape[0] == 3
        assert parts[k].shape[1] == 8
