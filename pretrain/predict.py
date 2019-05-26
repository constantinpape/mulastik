import numpy as np
import h5py
import torch
from inferno.trainers.basic import Trainer


def normalize(inp, eps=1e-6):
    inp = inp.astype('float32')
    mean, std = inp.mean(), inp.std()
    return (inp - mean) / (std + eps)


def predict(raw, gpu_id):
    trainer = Trainer().load('../networks/pretrain_v1/Weights',
                             best=True, map_location=torch.device(gpu_id))
    model = trainer.model

    out = []
    print("Prediction ...")
    with torch.no_grad():
        for z in range(raw.shape[0]):
            data = normalize(raw[z][None, None])
            data = torch.from_numpy(data).to(gpu_id)
            pred = model(data).cpu().numpy().squeeze()
            out.append(pred)
    out = np.concatenate([pred[:, None] for pred in out], axis=1).astype('float32')
    print(out.shape)
    print(out.min())
    print(out.max())
    return out


def predict_and_save(gpu_id):
    path = '/g/kreshuk/data/isbi2012_challenge/vnc_train_volume.h5'
    with h5py.File(path, 'r') as f:
        raw  = f['volumes/raw'][:]
    out = predict(raw, gpu_id)
    print("Saving ...")
    with h5py.File('data.h5') as f:
        f.create_dataset('raw', data=raw, compression='gzip')
        f.create_dataset('prediction', data=out, compression='gzip')


if __name__ == '__main__':
    predict_and_save(4)
