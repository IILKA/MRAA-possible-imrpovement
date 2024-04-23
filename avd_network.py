
import torch
from torch import nn


class AVDNetwork(nn.Module):
    """
    Animation via Disentanglement network
    """
    def __init__(self, num_tps, id_bottle_size=64, pose_bottle_size=64):
        super(AVDNetwork, self).__init__()
        input_size = 5 * 2 * num_tps
        self.num_tps = num_tps

        self.id_encoder = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, padding=1), #(256, 128, 100)
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2), #(256, 64, 50)
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2), #(256, 32, 25)
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, id_bottle_size, kernel_size=3, padding=1)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128*25, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, id_bottle_size)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128*25, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, id_bottle_size)
        )

        self.pose_encoder = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=3, padding=1), #(256, 128, 100)
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2), #(256, 64, 50)
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2), #(256, 32, 25)
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, id_bottle_size, kernel_size=3, padding=1)
        )

        # self.decoder = nn.Sequential(
        #     nn.Linear(512, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Linear(256, input_size)
        # )
        self.decoder = nn.Sequential(
            nn.Linear(pose_bottle_size + id_bottle_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_size)
        )

    

    def forward(self, kp_source, kp_random):
        bs = kp_source['fg_kp'].shape[0]
        # print(kp_random['fg_kp'].size())
        # print(kp_source['fg_kp'].size())
        # print(kp_random['fg_kp'].view(bs, -1).size())
        # print(kp_source['fg_kp'].view(bs, -1).size())
        #transform the keypoint to (bs, 1, 5*2*num_tps)
        kpr = kp_random['fg_kp'].view(bs, 1, -1)
        kps = kp_source['fg_kp'].view(bs, 1, -1)
        # print("kpr",kpr.size())
        # print("kps",kps.size())
        pose_emb = self.pose_encoder(kpr)
        id_emb = self.id_encoder(kps)

        pose_emb = pose_emb.view(bs, -1)
        id_emb = id_emb.view(bs, -1)
        pose_emb = self.fc1(pose_emb)
        id_emb = self.fc2(id_emb)

        # print(pose_emb.size())
        # print(id_emb.size())
        # print("After cat", torch.cat([pose_emb, id_emb], dim=1).size())

        rec = self.decoder(torch.cat([pose_emb, id_emb], dim=1))

        rec =  {'fg_kp': rec.view(bs, self.num_tps*5, -1)}
        return rec

#original code
# class AVDNetwork2(nn.Module):
#     """
#     Animation via Disentanglement network
#     """

#     def __init__(self, num_tps, id_bottle_size=64, pose_bottle_size=64):
#         super(AVDNetwork, self).__init__()
#         input_size = 5*2 * num_tps
#         self.num_tps = num_tps

#         self.id_encoder = nn.Sequential(
#             nn.Linear(input_size, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, id_bottle_size)
#         )

#         self.pose_encoder = nn.Sequential(
#             nn.Linear(input_size, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#             nn.Linear(256, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(inplace=True),
#             nn.Linear(512, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, pose_bottle_size)
#         )

#         self.decoder = nn.Sequential(
#             nn.Linear(pose_bottle_size + id_bottle_size, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, input_size)
#         )

#     def forward(self, kp_source, kp_random):

#         bs = kp_source['fg_kp'].shape[0]
        
#         pose_emb = self.pose_encoder(kp_random['fg_kp'].view(bs, -1))
#         id_emb = self.id_encoder(kp_source['fg_kp'].view(bs, -1))

#         rec = self.decoder(torch.cat([pose_emb, id_emb], dim=1))

#         rec =  {'fg_kp': rec.view(bs, self.num_tps*5, -1)}
#         return rec


