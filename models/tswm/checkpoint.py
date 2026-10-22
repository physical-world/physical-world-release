"""Explicit mapping for two colliding handwritten class identities in TSWM files."""
import pickle
import torch
from models.dino import DINOV2_REPO
from models.tswm.proprio import ProprioceptiveEmbedding
from models.tswm.vqvae import VQVAE


class _TSWMUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('models.proprio', 'ProprioceptiveEmbedding'):
            return ProprioceptiveEmbedding
        if (module, name) == ('models.vqvae', 'VQVAE'):
            return VQVAE
        return super().find_class(module, name)


class _TSWMPickle:
    Unpickler = _TSWMUnpickler


def load_payload(path):
    """Restore trusted paper objects; the official pinned dependency provides DINO."""
    torch.hub.load(DINOV2_REPO, 'dinov2_vits14', pretrained=False, trust_repo=True, skip_validation=True)
    return torch.load(path, map_location='cpu', weights_only=False, pickle_module=_TSWMPickle)


class _TSWMPickler(pickle._Pickler):
    dispatch = pickle._Pickler.dispatch.copy()

    def save_global(self, obj, name=None):
        if obj is ProprioceptiveEmbedding:
            original = b'models.proprio\nProprioceptiveEmbedding\n'
        elif obj is VQVAE:
            original = b'models.vqvae\nVQVAE\n'
        else:
            return super().save_global(obj, name)
        self.write(pickle.GLOBAL + original)
        self.memoize(obj)

    dispatch[type] = save_global


_TSWMPickle.Pickler = _TSWMPickler


def save_payload(stream, payload):
    torch.save(payload, stream, pickle_module=_TSWMPickle)
