import os
import torch
import torch.nn as nn
import numpy as np
import itertools
from torch.utils.data import Dataset
from torch.distributions.categorical import Categorical

from fastshap import FastSHAP
from fastshap import Surrogate, KLDivLoss
from fastshap.utils import MaskLayer1d


class MaskLayer1d(nn.Module):
    '''
    Masking for 1d inputs.

    Args:
      value: replacement value(s) for held out features.
      append: whether to append the mask along feature dimension.
    '''

    def __init__(self, value, append):
        super().__init__()
        self.value = value
        self.append = append

    def forward(self, input_tuple):
        x, S = input_tuple
        x = x * S + self.value * (1 - S)
        if self.append:
            x = torch.cat((x, S), dim=1)
        return x


class MaskLayer2d(nn.Module):
    '''
    Masking for 2d inputs.

    Args:
      value: replacement value(s) for held out features.
      append: whether to append the mask along channels dimension.
    '''

    def __init__(self, value, append):
        super().__init__()
        self.value = value
        self.append = append

    def forward(self, input_tuple):
        x, S = input_tuple
        if len(S.shape) == 3:
            S = S.unsqueeze(1)
        x = x * S + self.value * (1 - S)
        if self.append:
            x = torch.cat((x, S), dim=1)
        return x


class KLDivLoss(nn.Module):
    '''
    KL divergence loss that applies log softmax operation to predictions.

    Args:
      reduction: how to reduce loss value (e.g., 'batchmean').
      log_target: whether the target is expected as a log probabilities (or as
        probabilities).
    '''

    def __init__(self, reduction='batchmean', log_target=False):
        super().__init__()
        self.kld = nn.KLDivLoss(reduction=reduction, log_target=log_target)

    def forward(self, pred, target):
        '''
        Evaluate loss.

        Args:
          pred:
          target:
        '''
        return self.kld(pred.log_softmax(dim=1), target)


class DatasetRepeat(Dataset):
    '''
    A wrapper around multiple datasets that allows repeated elements when the
    dataset sizes don't match. The number of elements is the maximum dataset
    size, and all datasets must be broadcastable to the same size.

    Args:
      datasets: list of dataset objects.
    '''

    def __init__(self, datasets):
        # Get maximum number of elements.
        assert np.all([isinstance(dset, Dataset) for dset in datasets])
        items = [len(dset) for dset in datasets]
        num_items = np.max(items)

        # Ensure all datasets align.
        self.dsets = datasets
        self.num_items = num_items
        self.items = items

    def __getitem__(self, index):
        assert 0 <= index < self.num_items
        return_items = [dset[index % num] for dset, num in
                        zip(self.dsets, self.items)]
        return tuple(itertools.chain(*return_items))

    def __len__(self):
        return self.num_items


class DatasetInputOnly(Dataset):
    '''
    A wrapper around a dataset object to ensure that only the first element is
    returned.

    Args:
      dataset: dataset object.
    '''

    def __init__(self, dataset):
        assert isinstance(dataset, Dataset)
        self.dataset = dataset

    def __getitem__(self, index):
        return (self.dataset[index][0],)

    def __len__(self):
        return len(self.dataset)


class UniformSampler:
    '''
    For sampling player subsets with cardinality chosen uniformly at random.

    Args:
      num_players: number of players.
    '''

    def __init__(self, num_players):
        self.num_players = num_players

    def sample(self, batch_size):
        '''
        Generate sample.

        Args:
          batch_size: number of samples
        Return:
          S: tensor of shape batch_size X num_players with entries in {0, 1}
        '''
        rand = torch.rand(batch_size, self.num_players) # matrix in [0,1) with size batch x players
        thresh = torch.rand(batch_size, 1)
        S = (thresh > rand).float() # shape batch_size X num_players, belongs to 

        return S


class ShapleySampler:
    '''
    For sampling player subsets from the Shapley distribution.

    Args:
      num_players: number of players.
    '''

    def __init__(self, num_players):
        arange = torch.arange(1, num_players)
        w = 1 / (arange * (num_players - arange))
        w = w / torch.sum(w)
        self.categorical = Categorical(probs=w)
        self.num_players = num_players
        self.tril = np.tril(
            np.ones((num_players - 1, num_players), dtype=np.float32), k=0
        )
        self.rng = np.random.default_rng()

    def sample(self, batch_size, paired_sampling):
        '''
        Generate sample.

        Args:
          batch_size: number of samples.
          paired_sampling: whether to use paired sampling.
        '''
        num_included = 1 + self.categorical.sample([batch_size])
        S = self.tril[num_included - 1]
        S = self.rng.permuted(S, axis=1)  # Note: permutes each row.
        if paired_sampling:
            S[1::2] = 1 - S[0:(batch_size - 1):2]  # Note: allows batch_size % 2 == 1.
        return torch.from_numpy(S)

def build_surrogate(
    surrogate_path: str,
    produce_surr_model: callable,
    model,
    num_features,
    num_classes,
    device: torch.device,
    X_train = None,
    X_val = None,
    overwrite: bool = False
):
    surrogate_model = produce_surr_model(num_features, num_classes).to(device)

    if os.path.isfile(surrogate_path) and not overwrite:
        print('Loading saved surrogate model')
        state = torch.load(surrogate_path)
        surrogate_model.load_state_dict(state)
        surrogate = Surrogate(surrogate_model, num_features)
    else:
        surrogate = Surrogate(surrogate_model, num_features)
        print(surrogate)

        # Set up original model
        def original_model(x):
            pred = model.predict(x.cpu().numpy())
            if pred.shape[-1] == 1:
                pred = np.stack([1 - pred, pred]).T
            return torch.tensor(pred, dtype=torch.float32, device=x.device)

        # Train
        surrogate.train_original_model(
            X_train,
            X_val,
            original_model,
            batch_size=64,
            max_epochs=1000,
            loss_fn=KLDivLoss(),
            validation_samples=10,
            validation_batch_size=10000,
            verbose=True,
            lookback=200
            )
        surrogate_model.to("cpu")
        torch.save(surrogate_model.state_dict(), surrogate_path)
        surrogate_model.to(device)
    return surrogate

def standard_produce_surr_model(num_features, num_classes):
    return nn.Sequential(
        MaskLayer1d(value=0, append=True),
        nn.Linear(2 * num_features, 128),
        nn.ELU(inplace=True),
        nn.Linear(128, 128),
        nn.ELU(inplace=True),
        nn.Linear(128, num_classes))

def produce_expl_model(num_features, num_classes):
    return nn.Sequential(
        nn.Linear(num_features, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, num_classes * num_features))

def build_explainer(
    explainer_path,
    produce_expl_model,
    surrogate,
    num_features,
    num_classes,
    device: torch.device,
    X_train = None,
    X_val = None,
    overwrite: bool = False
):
    # Create explainer model
    explainer = produce_expl_model(num_features, num_classes).to(device)

    if os.path.isfile(explainer_path) and not overwrite:
        print('Loading saved explainer model')
        state = torch.load(explainer_path)

        explainer.load_state_dict(state)

        fastshap = FastSHAP(explainer, surrogate, normalization='additive',
                            link=nn.Softmax(dim=-1))

    else:

        # Set up FastSHAP object
        fastshap = FastSHAP(explainer, surrogate, normalization='additive',
                            link=nn.Softmax(dim=-1))

        # Train
        fastshap.train(
            X_train,
            X_val[:100],
            batch_size=32,
            num_samples=32,
            max_epochs=200,
            validation_samples=128,
            verbose=True,
            lookback=30)
        
        # Save explainer
        explainer.cpu()
        torch.save(explainer.state_dict(), explainer_path)
        explainer.to(device)
    return explainer, fastshap
