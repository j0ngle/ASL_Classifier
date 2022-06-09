import torch
from torch import nn

class TransformerEncoder(nn.Module):
    def __init__(self, dim, n_heads, dropout=0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim*4, dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )


    def forward(self, x):
        x1 = self.norm(x)
        x1 = self.attention(x1)
        
        _x = x + x1

        x2 = self.norm(_x)
        x2 = self.mlp(x2)

        return _x + x2


class ViT(nn.Module):
    def __init__(self, 
                p_size=16,
                channels=3, 
                dim=512,
                n_heads=4,
                n_classes=100):

        super().__init__()
        self.p_size = p_size
        self.channels = channels
        self.dim = dim
        self.embed = nn.Linear((p_size**2)*channels, dim)
        self.transformer = TransformerEncoder(dim, n_heads)
        self.mlp_head = nn.Linear(dim, n_classes)

    def patch_image(self, img):
        size = img.size()
        patches = []
        for i in range(int(size[1]/self.p_size)):
            for j in range(int(size[2]/self.p_size)):
                xs = i*self.p_size
                xe = xs + self.p_size
                ys = j*self.p_size
                ye = ys + self.p_size
                patch = img[:, xs:xe, ys:ye]
                patches.append(patch)

        print(patches.shape)

        return patches

    def forward(self, x):
        patches = self.patch_image(x)
        for i in range(len(patches)):
            patches[i] = nn.Flatten(patches[i])

        embeddings = self.embed(x)
        encoded_projections = self.transformer(embeddings)
        return self.mlp_head(encoded_projections)