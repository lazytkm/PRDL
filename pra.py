import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np

from utils import trunc_normal_


class PRAWrapper(nn.Module):
    def __init__(self, backbone, head, pra):
        super().__init__()

        self.backbone = backbone
        self.head = head
        self.pra = pra

    def forward(self, x, do_it_lists=None, pra=False):
        if pra:
            x = self.backbone(x)
            outputs, logvar, mu, sparse_loss, std_loss = self.pra(x, do_it_lists)
            return self.head(outputs), logvar, mu, sparse_loss, std_loss
        else:
            # convert to list
            if not isinstance(x, list):
                x = [x]
            idx_crops = torch.cumsum(torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1], 0)
            start_idx, output = 0, torch.empty(0).to(x[0].device)
            for end_idx in idx_crops:
                _out = self.backbone(torch.cat(x[start_idx: end_idx]))
                if torch.isnan(_out).sum():
                    print(x[start_idx: end_idx], force=True)
                    print(_out, force=True)
                # The output is a tuple with XCiT model. See:
                # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
                if isinstance(_out, tuple):
                    _out = _out[0]

                # accumulate outputs
                output = torch.cat((output, _out))
                start_idx = end_idx
            
            return self.head(output)


class PRA(nn.Module):
    def __init__(self, embed_dim, num_aug):
        super().__init__()

        self.log_var_head = nn.Linear(embed_dim, embed_dim)
        self.mu_head = nn.Linear(embed_dim, embed_dim)
        self.aug_masks = nn.Parameter(torch.zeros(num_aug, embed_dim))

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.log_var_head.weight, std=.02)
        if isinstance(self.log_var_head, nn.Linear) and self.log_var_head.bias is not None:
                nn.init.constant_(self.log_var_head.bias, 0)
        
        trunc_normal_(self.mu_head.weight, std=.02)
        if isinstance(self.mu_head, nn.Linear) and self.mu_head.bias is not None:
                nn.init.constant_(self.mu_head.bias, 0)

        trunc_normal_(self.aug_masks, mean=0.2, std=.02)

    def forward(self, x, do_it_lists):
        logvar = self.log_var_head(x)
        mu = self.mu_head(x)
        outputs = []
        sparse_loss = 0
        std_loss = 0
        for do_it_list in do_it_lists:
            do_it_index = torch.stack(do_it_list, dim=0)
            aug_masks = torch.nn.functional.sigmoid(self.aug_masks)
            aug_masks_mean = torch.stack([torch.mean(aug_masks[do_it_index[:, i]], dim=0) 
                                            for i in range(do_it_index.shape[1])], dim=0)
            eps = torch.randn(mu.shape[0], mu.shape[1]).to(x.device)
            sigma = torch.exp(logvar / 2)
            sigma_masked = sigma * aug_masks_mean
            output = mu + eps * sigma_masked
            outputs.append(output)

            aug_masks_c = aug_masks_mean - aug_masks_mean.mean(dim=1, keepdim=True)
            std_aug_masks = torch.sqrt(aug_masks_c.var(dim=1) + 0.0001)
            std_loss += torch.mean(torch.nn.functional.relu(1 - std_aug_masks))
            
            sparse_loss += torch.mean(torch.norm(aug_masks_mean, p=1, dim=-1))

        outputs = torch.cat(outputs)
        sparse_loss = sparse_loss / len(do_it_lists)
        std_loss = std_loss / len(do_it_lists)

        return outputs, logvar, mu, sparse_loss, std_loss

class PRDLLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, sample_temp=1):
        super().__init__()
        self.student_temp = student_temp
        self.sample_temp = sample_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, sample_ouput, student_output, teacher_output, log_var, mu, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        sample_ouput = sample_ouput / self.sample_temp
        sample_ouput = sample_ouput.chunk(2)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        sample_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
            for v in range(len(sample_ouput)):
                if v == iq:
                    loss = torch.sum(-q * F.log_softmax(sample_ouput[v], dim=-1), dim=-1)
                    sample_loss += loss.mean()
                    total_loss += loss.mean()
                    n_loss_terms += 1
        total_loss /= n_loss_terms
        ce_loss = total_loss
        self.update_center(teacher_output)
        
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - torch.exp(log_var), -1), 0)

        return total_loss, ce_loss, kl_loss, sample_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)