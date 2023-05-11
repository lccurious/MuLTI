from pathlib import Path
import torch
import pickle
import time
import lmdb
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import h5py
from PIL import Image
import multiprocessing as mp
from scipy.stats import ortho_group

from multi.utils.physics_engine import BallEngine, rand_float


__all__ = ["BallsDoubleViewDataset", "BallsTrajectoryDoubleViewDataset"]


def leaky_ReLU_1d(d, negSlope):
    if d > 0:
        return d
    else:
        return d * negSlope


leaky1d = np.vectorize(leaky_ReLU_1d)


def leaky_ReLU(D, negSlope):
    assert negSlope > 0
    return leaky1d(D, negSlope)


def store_data(data_names, data, path):
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()


def load_data(data_names, path):
    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            image = img.convert('RGB')
            image = image.resize((64, 64))
            return np.array(image, dtype=np.uint8)


def default_loader(path):
    return pil_loader(path)


def init_state(dim):
    # mean, std, count
    return np.zeros((dim, 3))


def combine_state(stat_0, stat_1):
    mean_0, std_0, n_0 = stat_0[:, 0], stat_0[:, 1], stat_0[:, 2]
    mean_1, std_1, n_1 = stat_1[:, 0], stat_1[:, 1], stat_1[:, 2]

    mean = (mean_0 * n_0 + mean_1 * n_1) / (n_0 + n_1)
    std = np.sqrt(
        (std_0 ** 2 * n_0 + std_1 ** 2 * n_1 + (mean_0 - mean) ** 2 * n_0 + (mean_1 - mean) ** 2 * n_1) / (n_0 + n_1))
    n = n_0 + n_1

    return np.stack([mean, std, n], axis=-1)


def generate_balls_double_view(info):
    thread_idx = info['thread_idx']
    data_root = info['data_root']
    data_names = info['data_names']
    n_rollout = info['n_rollout']
    time_steps = info['time_steps']
    n_balls = info['n_balls']
    param_load = info['param_load']
    dt = info['dt']
    latent_size = info['latent_size']

    # reset seeds for current thread
    np.random.seed(round(time.time() * 1000 + thread_idx) % 2 ** 32)

    attr_dim = 1  # radius
    state_dim = 4  # pos_x, pos_y, vel_x, vel_y
    action_dim = 2  # force_x, force_y

    # states mean the statistical variables: [mean, std, count]
    states = [init_state(attr_dim), init_state(
        state_dim), init_state(action_dim)]

    engine = BallEngine(dt, state_dim, action_dim)

    for i in tqdm(range(n_rollout)):
        rollout_idx = thread_idx * n_rollout + i
        rollout_dir = Path(data_root) / str(rollout_idx)
        if not rollout_dir.exists():
            rollout_dir.mkdir(parents=True)

        # use the same relations for every epsiode
        engine.init(n_balls, param_load=param_load)

        attrs_all = np.zeros((time_steps, n_balls, attr_dim))
        states_all = np.zeros((time_steps, n_balls, state_dim))
        actions_all = np.zeros((time_steps, n_balls, action_dim))
        rel_attrs_all = np.zeros((time_steps, engine.param_dim, 2))

        for t in range(time_steps):
            state = engine.get_state()

            vel_dim = state_dim // 2
            pos = state[:, :vel_dim]
            vel = state[:, vel_dim:]

            if t > 0:
                vel = (pos - states_all[t - 1, :, :vel_dim]) / dt

            attrs = np.zeros((n_balls, attr_dim))
            attrs = engine.radius

            attrs_all[t] = attrs
            states_all[t, :, :vel_dim] = pos
            states_all[t, :, vel_dim:] = vel
            rel_attrs_all[t] = engine.param

            # max_force: 300
            act = np.random.laplace(
                loc=0.0, scale=0.1, size=(n_balls, 2)) * 600
            act = np.clip(act, -1000, 1000)
            engine.step(act)

            actions_all[t] = act.copy()

        data = [attrs_all, states_all, actions_all, rel_attrs_all]
        store_data(data_names, data, str(rollout_dir) + '.h5')
        # Here only render the first 5 balls, there are two invisible
        engine.render(states_all[:, :5], actions_all,
                      engine.get_param(),
                      video=False, image=True,
                      path=rollout_dir,
                      image_prefix='view1',
                      draw_edge=False, verbose=False)
        # Here only render the last 5 balls, there are two invisible
        engine.render(states_all[:, -5:], actions_all,
                      engine.get_param(),
                      video=False, image=True,
                      path=rollout_dir,
                      image_prefix='view2',
                      draw_edge=False, verbose=False)

        data = [data[i].astype(np.float64) for i in range(len(data))]

        for j in range(len(states)):
            state = init_state(states[j].shape[0])
            state[:, 0] = np.mean(data[j], axis=(0, 1))[:]
            state[:, 1] = np.std(data[j], axis=(0, 1))[:]
            state[:, 2] = data[j].shape[0]
            states[j] = combine_state(states[j], state)
    return states


class BallsDoubleViewDataset(Dataset):
    def __init__(self, cfg, phase, transform=None, generate=False):
        super(BallsDoubleViewDataset, self).__init__()
        self.cfg = cfg
        self.phase = phase
        self.data_root = cfg.data_path
        self.lmdb_path = self.data_root
        if generate:
            if Path(self.lmdb_path).exists():
                print("Dataset already generated!")
            else:
                self.generate_data()
                self.dump_to_lmdb()

        if not Path(self.lmdb_path).exists() and not generate:
            raise FileNotFoundError(
                "Dataset not exists, you need generate one!")

        self.env = lmdb.open(self.lmdb_path, subdir=True,
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.num_samples = pickle.loads(txn.get(b'__len__'))
            self.keys = pickle.loads(txn.get(b'__keys__'))

        self.data_names = ['attrs', 'states', 'actions', 'rels']
        self.state_path = str(Path(self.data_root) / 'state.h5')
        self.attr_dim = 1
        self.state_dim = 4
        self.action_dim = 2
        self.transform = transform

        self.n_rollout = 250000  # int(n_rollout * ratio)

        self.length = cfg.time_lag + cfg.length

    def generate_data(self):
        n_rollout, time_steps, dt = self.n_rollout, self.length, 0.02
        assert n_rollout % self.cfg.workers == 0

        print("Generating data ... n_rollout={}, time_steps={}".format(
            n_rollout, time_steps
        ))

        # NOTE: we fix the ball generation rule here.
        num_balls = 8
        param_load = np.zeros((num_balls * (num_balls - 1) // 2, 2))
        edges = [
            (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4),
            (3, 5), (4, 7), (5, 7), (5, 6), (6, 7)
        ]
        cnt = 0
        for i in range(num_balls):
            for j in range(i):
                param_load[cnt, 0] = 0.
                if (i, j) in edges or (j, i) in edges:
                    param_load[cnt, 0] = 1.0
                    param_load[cnt, 1] = rand_float(20, 120)
                cnt += 1

        infos = []
        for i in range(self.cfg.workers):
            info = {
                'thread_idx': i,
                'data_root': self.data_root,
                'data_names': self.data_names,
                'n_balls': num_balls,
                'n_rollout': n_rollout // self.cfg.workers,
                'time_steps': time_steps,
                'dt': dt,
                'video': False,
                'phase': self.phase,
                'vis_height': self.cfg.dataset.m1_height_raw,
                'vis_width': self.cfg.dataset.m1_width_raw,
                'latent_size': self.cfg.dataset.m2_dim,
                'param_load': param_load
            }

            infos.append(info)

        cores = self.cfg.workers
        pool = mp.Pool(processes=cores)
        data = pool.map(generate_balls_double_view, infos)

        print("Training data generated, wrapping up states ...")

        if self.phase == 'train':
            # combine the statistical variables from different threads
            self.state = [
                init_state(self.attr_dim),
                init_state(self.state_dim),
                init_state(self.action_dim)
            ]
            for i in range(len(data)):
                for t in range(len(self.state)):
                    self.state[t] = combine_state(self.state[t], data[i][t])

            store_data(self.data_names[:len(self.state)],
                       self.state, self.state_path)
        else:
            print("Loading state from {} ...".format(self.state_path))
            self.state = load_data(self.data_names, self.state_path)

    def dump_to_lmdb(self):
        lmdb_path = self.lmdb_path
        db = lmdb.open(lmdb_path, subdir=True,
                       map_size=109951162776 * 2,
                       readonly=False,
                       meminit=False,
                       map_async=True)
        txn = db.begin(write=True)
        subfolders = [d for d in Path(self.data_root).iterdir() if d.is_dir()]
        for idx, img_root in enumerate(subfolders):
            video1_files = sorted(img_root.glob('view1*.png'))
            video2_files = sorted(img_root.glob('view2*.png'))
            video1 = np.array([pil_loader(d)
                              for d in video1_files], dtype=np.uint8)
            video2 = np.array([pil_loader(d)
                              for d in video2_files], dtype=np.uint8)
            metadata = load_data(self.data_names, str(img_root) + '.h5')

            txn.put(u'{}'.format(idx).encode('ascii'), pickle.dumps({
                'video1': video1, 'video2': video2, 'metadata': metadata
            }))

            if idx % 1000 == 0:
                print('[{}/{}] dumped into database'.format(idx, len(subfolders)))
                txn.commit()
                txn = db.begin(write=True)

        txn.commit()
        keys = [u'{}'.format(k).encode('ascii')
                for k in range(len(subfolders))]
        with db.begin(write=True) as txn:
            txn.put(b'__keys__', pickle.dumps(keys))
            txn.put(b'__len__', pickle.dumps(len(keys)))

        print("Flushing database ...")
        db.sync()
        db.close()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        env = self.env
        with env.begin(write=False) as txn:
            sample1_byteflow = txn.get(self.keys[idx])
        sample1_unpacked = pickle.loads(sample1_byteflow)

        sample1_video1 = sample1_unpacked['video1'][:self.length].transpose([
                                                                            0, 3, 1, 2]) / 255.0
        sample1_video2 = sample1_unpacked['video2'][:self.length].transpose([
                                                                            0, 3, 1, 2]) / 255.0
        sample1_metadata = sample1_unpacked['metadata']

        # ['attrs', 'states', 'actions', 'rels'], the box of balls are 80x80
        sample1_kpts = sample1_metadata[1] / 80.
        sample1_kpts[:, :, 1] *= -1
        sample1_latents = sample1_kpts[:self.length, :, :2].reshape(
            self.length, -1)

        sample1 = [
            torch.from_numpy((sample1_video1 - 0.5) / 0.5).float(),
            torch.from_numpy((sample1_video2 - 0.5) / 0.5).float(),
        ]
        ct = torch.from_numpy(sample1_latents).float()
        return sample1, ct
