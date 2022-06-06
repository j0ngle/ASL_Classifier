import torch

def similarity(u, v):
    '''Returns cos(theta), where theta = angle between u and v'''

    dot = torch.dot(u, v)
    u_norm = torch.norm(u)
    v_norm = torch.norm(v)
    theta = dot / (u_norm*v_norm)
    return torch.acos(theta)

def nce(u, v):
    '''Noise Constrastive Estimation'''
    loss = 0
    for i in len(u):
        scores = []
        for j in len(v):
            sim = similarity(u[i], v[j])
            scores.append(torch.exp(sim))

        loss += -torch.log(scores[i] / torch.sum(scores))

    return loss

def nt_xent(u, v):
    '''Normalized Temperature-Scaled Cross Entropy'''
    pass_a = nce(u, v)
    pass_b = nce(v, u)

    return (pass_a + pass_b) / (2*len(u))

        

    