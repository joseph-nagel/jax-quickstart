'''FLAX modules.'''

from typing import Sequence

import flax.linen as nn


class MLP(nn.Module):
    '''MLP module.'''

    features: Sequence[int] # modules act as dataclasses

    @nn.compact
    def __call__(self, x):

        for idx, features in enumerate(self.features):
            x = nn.Dense(features, name=f'dense{idx + 1}')(x) # create inline submodules

            if idx + 1 < len(self.features):
                x = nn.relu(x)

        return x


class AutoEncoder(nn.Module):
    '''Autoencoder module.'''

    enc_features: Sequence[int]
    dec_features: Sequence[int]

    def setup(self):
        self.encoder = MLP(self.enc_features) # only accessible from inside .init or .apply
        self.decoder = MLP(self.dec_features)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def __call__(self, x):
        return self.decode(self.encode(x))

