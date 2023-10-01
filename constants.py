import torch

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print('Using device: ', device)

actions = [
    ['right'],
    ['right', 'A']
]

level = 'SuperMarioBros-1-1-v0'
