from pathlib import Path
import torch
from torch import nn

from models.models.superglue import normalize_keypoints, AttentionalPropagation, AttentionalGNN, \
    log_optimal_transport, arange_like

def MLP(channels: list, do_bn=True, do_leaky=False, last_layer=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1 if last_layer else n):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            if do_leaky:
                layers.append(nn.LeakyReLU())
            else:
                layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.feature_dim = feature_dim
        self.encoder = MLP([3] + layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = torch.cat([kpts.transpose(-2, -1), scores.unsqueeze(-2)], dim=-2)
        in_shape = inputs.shape
        out_shape = [*in_shape]
        out_shape[-2] = self.feature_dim
        return self.encoder(inputs.view(-1, in_shape[-2], in_shape[-1])).view(*out_shape)

class ConfidenceMLP(nn.Module):
    def __init__(self, feature_dim, in_dim, out_dim=1):
        super().__init__()
        self.feature_dim = feature_dim
        self.layers_f = MLP([feature_dim * 2, feature_dim * 2, feature_dim], last_layer=False)
        self.layers_c = MLP([in_dim, feature_dim, feature_dim], last_layer=False)
        self.layers = MLP([feature_dim, out_dim])
        
        nn.init.constant_(self.layers[-1].bias, 0.0)

    def forward(self, desc):
        inputs = torch.cat(desc[:-1], dim=-2)
        out_f = self.layers_f(inputs)
        out_c = self.layers_c(desc[-1])
        return torch.sigmoid(self.layers(out_f + out_c))

class MultiFrameAttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names
        self.combine_cross_attention = True

    def forward(self, desc, ids=None):
        if self.training:
            out_shape = desc.shape
            tuple_size = out_shape[0]
            in_shape_self = [-1, out_shape[-2], out_shape[-1]]
            if self.combine_cross_attention:
                in_shape_cross = [out_shape[1] * tuple_size, out_shape[2], -1]
            else:
                in_shape_cross = [out_shape[1], out_shape[2], -1]
            for layer, name in zip(self.layers, self.names):
                if name == 'cross':
                    if self.combine_cross_attention:
                        neighbor_ids = torch.stack([torch.cat([torch.arange(id), torch.arange(id + 1, tuple_size)], 0) for id in ids], 0)
                        delta = layer(desc[list(ids)].view(in_shape_cross), desc[neighbor_ids].permute(0, 2, 3, 1, 4).reshape(in_shape_cross))
                        delta = delta.view(out_shape)
                    else:
                        delta = torch.zeros_like(desc)
                        for id in ids:
                            neighbor_ids = torch.cat((torch.arange(id), torch.arange(id + 1, tuple_size)), 0)
                            delta[id] = layer(desc[id], desc[neighbor_ids].permute(1, 2, 0, 3).reshape(in_shape_cross))
                    desc = desc + delta
                else:  # if name == 'self':
                    desc = desc + layer(desc.view(in_shape_self), desc.view(in_shape_self)).view(out_shape)
        else:
            for layer, name in zip(self.layers, self.names):
                if name == 'cross':
                    delta = [None if d is None else torch.zeros_like(d) for d in desc]
                    for id in ids:
                        neighbor_ids = ids.copy()
                        neighbor_ids.remove(id)
                        neighbor_ids = torch.tensor(list(neighbor_ids))
                        delta[id] = layer(desc[id], torch.cat([desc[neighbor_id] for neighbor_id in neighbor_ids], -1))
                    for id in ids:
                        desc[id] = desc[id] + delta[id]
                else:  # if name == 'self':
                    for id in ids:
                        desc[id] = desc[id] + layer(desc[id], desc[id])
        return desc

class MultiViewMatcher(nn.Module):
    """ Multi-view feature matcher
    """
    default_config = {
        'descriptor_dim': 256,
        'weights': 'none',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'multi_frame_matching' : True,
        'full_output' : False,
        'conf_mlp' : True,
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        if self.config["multi_frame_matching"]:
            self.gnn = MultiFrameAttentionalGNN(
                self.config['descriptor_dim'], self.config['GNN_layers'])
        else:
            self.gnn = AttentionalGNN(self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        assert self.config['weights'] in ['indoor', 'outdoor', 'none']
        if self.config['weights'] != "none":
            path = Path(__file__).parent
            path = path / 'weights/superglue_{}.pth'.format(self.config['weights'])
            self.load_state_dict(torch.load(str(path)))
            print('Loaded SuperGlue model (\"{}\" weights)'.format(
                self.config['weights']))
        
        if self.config["conf_mlp"]:
            feat_dim = self.config['descriptor_dim']
            input_dim = 1
            self.conf_mlp = ConfidenceMLP(feat_dim, input_dim)

    def match(self, data, id0, id1):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors' + str(id0)], data['descriptors' + str(id1)]
        kpts0, kpts1 = data['keypoints' + str(id0)], data['keypoints' + str(id1)]

        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches{}_{}_{}'.format(id0, id0, id1) : kpts0.new_full(shape0, -1, dtype=torch.int),
                'matches{}_{}_{}'.format(id1, id0, id1) : kpts1.new_full(shape1, -1, dtype=torch.int),
                'matching_scores{}_{}_{}'.format(id0, id0, id1) : kpts0.new_zeros(shape0),
                'matching_scores{}_{}_{}'.format(id1, id0, id1) : kpts1.new_zeros(shape1),
            }

        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image' + str(id0)].shape)
        kpts1 = normalize_keypoints(kpts1, data['image' + str(id1)].shape)

        # Keypoint MLP encoder.
        desc0 = desc0 + self.kenc(kpts0, data['scores' + str(id0)])
        desc1 = desc1 + self.kenc(kpts1, data['scores' + str(id1)])

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5

        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])

        if not self.training or self.config["full_output"]:
            with torch.no_grad():
                max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
                indices0, indices1 = max0.indices, max1.indices
                mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
                mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
                zero = scores.new_tensor(0)
                mscores0 = torch.where(mutual0, max0.values.exp(), zero)
                mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
                valid0 = mutual0 & (mscores0 > 0.)
                valid1 = mutual1 & valid0.gather(1, indices1)
                indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
                indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

            # run confidence MLP
            if self.config["conf_mlp"]:
                batch_idx = torch.arange(indices0.shape[0]).unsqueeze(-1).repeat(1, indices0.shape[-1])
                additional_input = scores[batch_idx, torch.arange(indices0.shape[-1]), indices0].unsqueeze(-2)
                conf = self.conf_mlp([mdesc0, mdesc1.transpose(-2, -1)[batch_idx, indices0].transpose(-2, -1), additional_input]).transpose(-2, -1)
            return {
                'matches{}_{}_{}'.format(id0, id0, id1) : indices0, # use -1 for invalid match
                'matches{}_{}_{}'.format(id1, id0, id1) : indices1, # use -1 for invalid match
                'matching_scores{}_{}_{}'.format(id0, id0, id1) : mscores0,
                'matching_scores{}_{}_{}'.format(id1, id0, id1) : mscores1,
                'scores_' + str(id0) + '_' + str(id1): scores,
                'conf_scores_' + str(id0) + '_' + str(id1): conf if self.config["conf_mlp"] else None,
            }
        
        return {'scores_' + str(id0) + '_' + str(id1): scores}

    def multi_match(self, data, tuple_size):
        result = dict()
        if self.training:
            # tensors for fixed number of keypoints at train time
            desc = torch.stack([data['descriptors' + str(id)] for id in range(tuple_size)], 0)
            sp_desc = desc.clone()
            kpts = torch.stack([data['keypoints' + str(id)] for id in range(tuple_size)], 0)
            scores = torch.stack([data['scores' + str(id)] for id in range(tuple_size)], 0)
            pairs_with_kpts = [(id0, id1) for id1 in range(tuple_size) for id0 in range(id1) ]
            ids_with_kpts = {id for id in range(tuple_size)}
        else:
            # use list of tensors for variable number of keypoints at test time
            kpts = []
            desc = []
            sp_desc = []
            scores = []
            for id in range(tuple_size):
                kpts_id = data['keypoints' + str(id)]
                desc_id = data['descriptors' + str(id)]
                scores_id = data['scores' + str(id)]
                if kpts_id.shape[1] == 0:
                    kpts_id = None
                    desc_id = None
                    scores_id = None
                kpts.append(kpts_id)
                desc.append(desc_id)
                sp_desc.append(desc_id.clone())
                scores.append(scores_id)

            pairs_with_kpts = []
            ids_with_kpts = set()
            for id1 in range(tuple_size):
                for id0 in range(id1):
                    if kpts[id0] is None or kpts[id1] is None:
                        shape0, shape1 = kpts[id0].shape[:-1], kpts[id1].shape[:-1]
                        result.update({
                            'matches{}_{}_{}'.format(id0, id0, id1) : kpts[id0].new_full(shape0, -1, dtype=torch.int),
                            'matches{}_{}_{}'.format(id1, id0, id1) : kpts[id1].new_full(shape1, -1, dtype=torch.int),
                            'matching_scores{}_{}_{}'.format(id0, id0, id1) : kpts[id0].new_zeros(shape0),
                            'matching_scores{}_{}_{}'.format(id1, id0, id1) : kpts[id1].new_zeros(shape1),
                        })
                    else:
                        ids_with_kpts.update({id0, id1})
                        pairs_with_kpts.append((id0, id1))

        # Keypoint normalization and MLP encoder
        if self.training:
            kpts = normalize_keypoints(kpts, data['image0'].shape) # assume all images have the same size
            desc = desc + self.kenc(kpts, scores)
        else:
            for id in ids_with_kpts:
                kpts[id] = normalize_keypoints(kpts[id], data['image0'].shape) # assume all images have the same size
                desc[id] = desc[id] + self.kenc(kpts[id], scores[id])

        # Multi-layer Transformer network.
        desc = self.gnn(desc, ids_with_kpts)

        # Final MLP projection.
        for id0, id1 in pairs_with_kpts:
            mdesc0, mdesc1 = self.final_proj(desc[id0]), self.final_proj(desc[id1])

            # Compute matching descriptor distance.
            scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
            scores = scores / self.config['descriptor_dim']**.5

            # Run the optimal transport.
            scores = log_optimal_transport(
                scores, self.bin_score,
                iters=self.config['sinkhorn_iterations'])

            if not self.training or self.config["full_output"]:
                with torch.no_grad():

                    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
                    indices0, indices1 = max0.indices, max1.indices
                    mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
                    mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
                    zero = scores.new_tensor(0)
                    mscores0 = torch.where(mutual0, max0.values.exp(), zero)
                    mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
                    valid0 = mutual0 & (mscores0 > 0.)
                    valid1 = mutual1 & valid0.gather(1, indices1)
                    indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
                    indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))

                # run confidence MLP
                if self.config["conf_mlp"]:
                    batch_idx = torch.arange(indices0.shape[0]).unsqueeze(-1).repeat(1, indices0.shape[-1])
                    additional_input = scores[batch_idx, torch.arange(indices0.shape[-1]), indices0].unsqueeze(-2)
                    conf = self.conf_mlp([mdesc0, mdesc1.transpose(-2, -1)[batch_idx, indices0].transpose(-2, -1), additional_input]).transpose(-2, -1)

                result.update({
                    'matches{}_{}_{}'.format(id0, id0, id1) : indices0, # use -1 for invalid match
                    'matches{}_{}_{}'.format(id1, id0, id1) : indices1, # use -1 for invalid match
                    'matching_scores{}_{}_{}'.format(id0, id0, id1) : mscores0,
                    'matching_scores{}_{}_{}'.format(id1, id0, id1) : mscores1,
                    'scores_' + str(id0) + '_' + str(id1): scores,
                    'conf_scores_' + str(id0) + '_' + str(id1): conf if self.config["conf_mlp"] else None,
                })
            else:
                result.update({
                    'scores_' + str(id0) + '_' + str(id1): scores
                    })
        return result
    
    def forward(self, data):
        result = dict()
        tuple_size = len(data["ids"])
        if not self.config['multi_frame_matching']:
            # match all possible pairs
            for id1 in range(tuple_size):
                for id0 in range(id1):
                    result.update(self.match(data, id0, id1))
        else:
            result = self.multi_match(data, tuple_size)
        return result
