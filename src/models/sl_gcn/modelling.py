import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from transformers import PreTrainedModel, FeatureExtractionMixin
from transformers.modeling_outputs import ImageClassifierOutput
from .configuration import SLGCNConfig


class Graph:
    def __init__(self, num_node=27, strategy="spatial"):
        self.num_node = num_node
        self.strategy = strategy
        self.center = 0

        self.ori_idxs = [
            (5, 6), (5, 7), (6, 8), (8, 10), (7, 9), (9, 11),
            (12, 13), (12, 14), (12, 16), (12, 18), (12, 20),
            (14, 15), (16, 17), (18, 19), (20, 21), (22, 23),
            (22, 24), (22, 26), (22, 28), (22, 30), (24, 25),
            (26, 27), (28, 29), (30, 31), (10, 12), (11, 22),
        ]
        self.edges = [
            (v1 - 5, v2 - 5)
            for v1, v2 in self.ori_idxs
            if v1 - 5 < num_node and v2 - 5 < num_node
        ]

        self.get_adjacency_matrix()

    def get_adjacency_matrix(self):
        hop_dis = np.ones((self.num_node, self.num_node)) * np.inf
        for i in range(self.num_node):
            hop_dis[i, i] = 0
        for i, j in self.edges:
            hop_dis[i, j] = 1
            hop_dis[j, i] = 1

        for k in range(self.num_node):
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if hop_dis[i, j] > hop_dis[i, k] + hop_dis[k, j]:
                        hop_dis[i, j] = hop_dis[i, k] + hop_dis[k, j]

        if self.strategy == "spatial":
            A_0 = np.eye(self.num_node)
            A_1 = np.zeros((self.num_node, self.num_node))
            A_2 = np.zeros((self.num_node, self.num_node))
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if hop_dis[i, j] == 1:
                        if hop_dis[j, self.center] < hop_dis[i, self.center]:
                            A_1[i, j] = 1
                        else:
                            A_2[i, j] = 1
            self.A = np.stack([
                self.normalize_digraph(A_0),
                self.normalize_digraph(A_1),
                self.normalize_digraph(A_2)
            ])
        else:
            A = np.zeros((self.num_node, self.num_node))
            for i, j in self.edges:
                A[i, j] = 1
                A[j, i] = 1
            A = A + np.eye(self.num_node)
            self.A = np.stack([self.normalize_digraph(A)])

    def normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        AD = np.dot(A, Dn)
        return AD


class ConvTemporalGraphical(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )

    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum("nkctv,kvw->nctw", (x, A))
        return x.contiguous(), A


class st_gcn_block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dropout=0,
        residual=True,
    ):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, 1),
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res
        return self.relu(x), A


class SLGCN(nn.Module):
    def __init__(self, in_channels, num_classes, num_points, strategy="spatial", **kwargs):
        super().__init__()

        self.graph = Graph(num_node=num_points, strategy=strategy)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A", A)

        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * num_points)

        self.st_gcn_networks = nn.ModuleList((
            st_gcn_block(in_channels, 64, kernel_size, stride=1, residual=False),
            st_gcn_block(64, 64, kernel_size, stride=1),
            st_gcn_block(64, 64, kernel_size, stride=1),
            st_gcn_block(64, 128, kernel_size, stride=2),
            st_gcn_block(128, 128, kernel_size, stride=1),
            st_gcn_block(128, 128, kernel_size, stride=1),
            st_gcn_block(128, 256, kernel_size, stride=2),
            st_gcn_block(256, 256, kernel_size, stride=1),
            st_gcn_block(256, 256, kernel_size, stride=1),
        ))

        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones(self.A.size()))
            for _ in self.st_gcn_networks
        ])

        self.fcn = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        n, c, t, v, m = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # (N, M, V, C, T)
        x = x.view(n * m, v * c, t)  # (N*M, V*C, T)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # (N, M, C, T, V)
        x = x.view(n * m, c, t, v)  # (N*M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # Global average pooling
        x = F.avg_pool2d(x, x.size()[2:])  # (N*M, C, 1, 1)
        x = x.view(n, m, -1, 1, 1).mean(dim=1)  # (N, C, 1, 1)

        x = self.fcn(x)  # (N, num_class, 1, 1)
        x = x.view(n, -1)  # (N, num_class)

        return x


class SLGCNFeatureExtractor(FeatureExtractionMixin):
    def __init__(self, config: SLGCNConfig = SLGCNConfig(), **kwargs) -> None:
        super().__init__(**kwargs)
        self.arch = config.arch
        self.num_frames = config.window_size
        self.window_size = config.window_size
        self.num_points = config.num_points
        self.in_channels = config.in_channels
        self.num_people = config.num_people
        self.bone_stream = config.bone_stream
        self.motion_stream = config.motion_stream
        self.is_vector = config.is_vector


class SLGCNForGraphClassification(PreTrainedModel):
    config_class = SLGCNConfig

    def __init__(
        self,
        config: SLGCNConfig = SLGCNConfig(),
        label2id: dict = None,
        id2label: dict = None,
    ) -> None:
        super().__init__(config=config)
        self.label2id = label2id if label2id is not None else config.label2id
        self.id2label = id2label if id2label is not None else config.id2label
        self.num_classes = len(self.label2id) if self.label2id is not None else 400
        
        self.model = SLGCN(
            in_channels=self.config.in_channels,
            num_classes=self.num_classes,
            num_points=self.config.num_points,
            strategy=self.config.labeling_mode,
        )

        if self.config.pretrained and Path(self.config.pretrained).exists():
            state_dict = torch.load(self.config.pretrained)
            for key in list(state_dict.keys()):
                if key.startswith("model."):
                    state_dict[key[6:]] = state_dict.pop(key)
            self.model.load_state_dict(state_dict)

    def forward(
        self,
        poses: torch.Tensor,
        labels: torch.Tensor = None,
    ) -> ImageClassifierOutput:
        logits = self.model(poses)
        if labels is not None:
            labels = labels.to(logits.device, dtype=torch.long).view(-1)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return ImageClassifierOutput(loss=loss, logits=logits)
        return ImageClassifierOutput(logits=logits)
