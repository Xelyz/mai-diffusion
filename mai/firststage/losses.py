# from mug.model.models import *
import torch
# torch.backends.cudnn.enabled = False
import torch.nn
import torch.nn.functional
import numpy as np

class MaimaiReconstructLoss(torch.nn.Module):
    def __init__(self, weight_tap=1.0, weight_start_offset=1.0, weight_holding=1.0, weight_end_offset=1.0, weight_touch=1.0, weight_touch_offset=1.0, weight_touch_holding=1.0, weight_touch_hold_end_offset=1.0, weight_star_pass_through=1.0, weight_star_end_offset=1.0,
                 label_smoothing=0.0, gamma=2.0):
        super(MaimaiReconstructLoss, self).__init__()
        self.weight_tap = weight_tap
        self.weight_start_offset = weight_start_offset
        self.weight_holding = weight_holding
        self.weight_end_offset = weight_end_offset
        self.weight_touch = weight_touch
        self.weight_touch_offset = weight_touch_offset
        self.weight_touch_holding = weight_touch_holding
        self.weight_touch_hold_end_offset = weight_touch_hold_end_offset
        self.weight_star_pass_through = weight_star_pass_through
        self.weight_star_end_offset = weight_star_end_offset
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.label_smoothing = label_smoothing
        self.gamma = gamma

    def label_smoothing_bce_loss(self, predicts, targets):
        # p = torch.sigmoid(predicts)
        # p_t = p * targets + (1 - p) * (1 - targets)
        return self.bce_loss(
            predicts,
            targets * (1 - 2 * self.label_smoothing) + self.label_smoothing,
        )# * ((1 - p_t) ** self.gamma)

    def get_key_loss(self, inputs: torch.Tensor, reconstructions: torch.Tensor, valid: torch.Tensor,
                     loss_func) -> torch.Tensor:
        loss = loss_func(
            reconstructions,
            inputs
        )
        return torch.mean(loss * valid) / torch.mean(valid + 1e-6)

    def classification_metrics(self, inputs, reconstructions, valid_flag):
        predict_start = reconstructions >= 0
        true_start = inputs
        tp = true_start == predict_start
        tp_valid = tp * valid_flag
        acc_start = (torch.sum(tp_valid) /
                     (torch.sum(valid_flag) + 1e-5)
                     ).item()
        precision_start = (torch.sum(tp_valid * predict_start) /
                           (torch.sum(predict_start * valid_flag) + 1e-5)
                           ).item()
        recall_start = (torch.sum(tp_valid * true_start) /
                        (torch.sum(true_start * valid_flag) + 1e-5)
                        ).item()
        return acc_start, precision_start, recall_start

    def forward(self, inputs: torch.Tensor, reconstructions: torch.Tensor,
                valid_flag: torch.Tensor):
        """
        inputs / reconstructions: [B, 4 * K, T]
        valid_flag: [B, T]
        Feature Layout:
            [is_start: 0/1] * key_count

            [offset_start: 0-1] * key_count
            valid only if is_start = 1

            [is_holding: 0/1] * key_count, (exclude start, include end),
            valid only if previous.is_start = 1 or previous.is_holding = 1

            [offset_end: 0-1]
            valid only if is_holding = 1 and latter.is_holding = 0
        """
        valid_flag = torch.ones_like(valid_flag) # TODO
        valid_flag = torch.unsqueeze(valid_flag, dim=1)  # [B, 1, T]

        lengths = {
            'tap': 8,
            'tap_offset': 8,
            'is_holding': 8,
            'hold_end_offset': 8,
            'touch': 33,
            'touch_offset': 33,
            'touch_holding': 1,
            'touch_hold_end_offset': 1,
            'slide_pass_through': 17,
            'slide_end_offset': 8
        }
        # 计算分割点
        split_indices = np.cumsum(list(lengths.values()))[:-1]
        # 使用np.split分割数组
        input_arrays = np.split(inputs, split_indices, 1)
        # 将分割后的数组与名称对应
        input_dict = dict(zip(lengths.keys(), input_arrays))
        recon_arrays = np.split(reconstructions, split_indices, 1)
        recon_dict = dict(zip(lengths.keys(), recon_arrays))

        is_start = input_dict['tap']  # [B, K, T]
        holding_pad = torch.nn.functional.pad(input_dict['is_holding'], (0, 1))  # [B, K, T + 1]
        is_end = (input_dict['is_holding'] -
                  holding_pad[:, :, 1:] > 0.5).int()

        start_loss = self.get_key_loss(input_dict['tap'], recon_dict['tap'], valid_flag,
                                       self.label_smoothing_bce_loss)
        offset_start_loss = self.get_key_loss(input_dict['tap_offset'], recon_dict['tap_offset'],
                                              valid_flag * is_start,
                                              self.mse_loss)
        holding_loss = self.get_key_loss(input_dict['is_holding'], input_dict['is_holding'], valid_flag,
                                         self.label_smoothing_bce_loss)
        offset_end_loss = self.get_key_loss(input_dict['hold_end_offset'], recon_dict['hold_end_offset'],
                                            valid_flag * is_end,
                                            self.mse_loss)
        
        is_touch_start = input_dict['touch']  # [B, K, T]
        touch_holding_pad = torch.nn.functional.pad(input_dict['touch_holding'], (0, 1))  # [B, K, T + 1]
        is_touch_hold_end = (input_dict['touch_holding'] -
                             touch_holding_pad[:, :, 1:] > 0.5).int()
        touch_loss = self.get_key_loss(input_dict['touch'], recon_dict['touch'], valid_flag,
                                    self.label_smoothing_bce_loss)
        touch_offset_loss = self.get_key_loss(input_dict['touch_offset'], recon_dict['touch_offset'],
                                                valid_flag * is_touch_start,
                                                self.mse_loss)
        touch_holding_loss = self.get_key_loss(input_dict['touch_holding'], recon_dict['touch_holding'],
                                               valid_flag,
                                               self.label_smoothing_bce_loss)
        touch_hold_end_offset_loss = self.get_key_loss(input_dict['touch_hold_end_offset'], recon_dict['touch_hold_end_offset'],
                                                       valid_flag * is_touch_hold_end,
                                                       self.mse_loss)
        
        slide_pass_through_loss = self.get_key_loss(input_dict['slide_pass_through'], recon_dict['slide_pass_through'], valid_flag,
                                                    self.label_smoothing_bce_loss)
        slide_end_offset_loss = self.get_key_loss(input_dict['slide_end_offset'], recon_dict['slide_end_offset'], valid_flag,
                                                    self.mse_loss)

        acc_start, precision_start, recall_start = self.classification_metrics(
            input_dict['tap'], recon_dict['tap'], valid_flag
        )
        acc_ln_start, precision_ln_start, recall_ln_start = self.classification_metrics(
            input_dict['is_holding'], recon_dict['is_holding'], valid_flag
        )
        acc_touch_start, precision_touch_start, recall_touch_start = self.classification_metrics(
            input_dict['touch'], recon_dict['touch'], valid_flag
        )
        acc_ln_touch_start, precision_ln_touch_start, recall_ln_touch_start = self.classification_metrics(
            input_dict['touch_holding'], recon_dict['touch_holding'], valid_flag
        )

        loss = (start_loss * self.weight_tap +
                offset_start_loss * self.weight_start_offset +
                holding_loss * self.weight_holding +
                offset_end_loss * self.weight_end_offset + 
                touch_loss * self.weight_touch +
                touch_offset_loss * self.weight_touch_offset +
                touch_holding_loss * self.weight_touch_holding +
                touch_hold_end_offset_loss * self.weight_touch_hold_end_offset +
                slide_pass_through_loss * self.weight_star_pass_through +
                slide_end_offset_loss * self.weight_star_end_offset)
        
        return loss, {
            'start_loss': start_loss.detach().item(),
            'offset_start_loss': offset_start_loss.detach().item(),
            'holding_loss': holding_loss.detach().item(),
            'offset_end_loss': offset_end_loss.detach().item(),
            'touch_loss': touch_loss.detach().item(),
            'touch_offset_loss': touch_offset_loss.detach().item(),
            'touch_holding_loss': touch_holding_loss.detach().item(),
            'touch_hold_end_offset_loss': touch_hold_end_offset_loss.detach().item(),
            'slide_pass_through_loss': slide_pass_through_loss.detach().item(),
            'slide_end_offset_loss': slide_end_offset_loss.detach().item(),
            "acc_rice": acc_start,
            "acc_ln": acc_ln_start,
            "acc_touch": acc_touch_start,
            "acc_ln_touch": acc_ln_touch_start,
            "precision_rice": precision_start,
            "precision_ln": precision_ln_start,
            "precision_touch": precision_touch_start,
            "precision_ln_touch": precision_ln_touch_start,
            "recall_rice": recall_start,
            "recall_ln": recall_ln_start,
            "recall_touch": recall_touch_start,
            "recall_ln_touch": recall_ln_touch_start,
        }
    
class MaimaiTapReconstructLoss(torch.nn.Module):
    def __init__(self, weight_tap=1.0, weight_start_offset=1.0, weight_holding=1.0, weight_end_offset=1.0,
                 label_smoothing=0.0, gamma=2.0):
        super(MaimaiTapReconstructLoss, self).__init__()
        self.weight_tap = weight_tap
        self.weight_start_offset = weight_start_offset
        self.weight_holding = weight_holding
        self.weight_end_offset = weight_end_offset
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.label_smoothing = label_smoothing
        self.gamma = gamma

    def label_smoothing_bce_loss(self, predicts, targets):
        # p = torch.sigmoid(predicts)
        # p_t = p * targets + (1 - p) * (1 - targets)
        return self.bce_loss(
            predicts,
            targets * (1 - 2 * self.label_smoothing) + self.label_smoothing,
        )# * ((1 - p_t) ** self.gamma)

    def get_key_loss(self, inputs: torch.Tensor, reconstructions: torch.Tensor, valid: torch.Tensor,
                     loss_func) -> torch.Tensor:
        loss = loss_func(
            reconstructions,
            inputs
        )
        return torch.mean(loss * valid) / torch.mean(valid + 1e-6)

    def classification_metrics(self, inputs, reconstructions, valid_flag):
        predict_start = reconstructions >= 0
        true_start = inputs
        tp = true_start == predict_start
        tp_valid = tp * valid_flag
        acc_start = (torch.sum(tp_valid) /
                     (torch.sum(valid_flag) + 1e-5)
                     ).item()
        precision_start = (torch.sum(tp_valid * predict_start) /
                           (torch.sum(predict_start * valid_flag) + 1e-5)
                           ).item()
        recall_start = (torch.sum(tp_valid * true_start) /
                        (torch.sum(true_start * valid_flag) + 1e-5)
                        ).item()
        return acc_start, precision_start, recall_start

    def forward(self, inputs: torch.Tensor, reconstructions: torch.Tensor,
                valid_flag: torch.Tensor):
        """
        inputs / reconstructions: [B, 4 * K, T]
        valid_flag: [B, T]
        Feature Layout:
            [is_start: 0/1] * key_count

            [offset_start: 0-1] * key_count
            valid only if is_start = 1

            [is_holding: 0/1] * key_count, (exclude start, include end),
            valid only if previous.is_start = 1 or previous.is_holding = 1

            [offset_end: 0-1]
            valid only if is_holding = 1 and latter.is_holding = 0
        """
        valid_flag = torch.ones_like(valid_flag) # TODO
        valid_flag = torch.unsqueeze(valid_flag, dim=1)  # [B, 1, T]

        lengths = {
            'tap': 8,
            'tap_offset': 8,
            'is_holding': 8,
            'hold_end_offset': 8,
        }
        # 计算分割点
        split_indices = np.cumsum(list(lengths.values()))[:-1]
        # 使用np.split分割数组
        input_arrays = np.split(inputs, split_indices, 1)
        # 将分割后的数组与名称对应
        input_dict = dict(zip(lengths.keys(), input_arrays))
        recon_arrays = np.split(reconstructions, split_indices, 1)
        recon_dict = dict(zip(lengths.keys(), recon_arrays))

        is_start = input_dict['tap']  # [B, K, T]
        holding_pad = torch.nn.functional.pad(input_dict['is_holding'], (0, 1))  # [B, K, T + 1]
        is_end = (input_dict['is_holding'] -
                  holding_pad[:, :, 1:] > 0.5).int()

        start_loss = self.get_key_loss(input_dict['tap'], recon_dict['tap'], valid_flag,
                                       self.label_smoothing_bce_loss)
        offset_start_loss = self.get_key_loss(input_dict['tap_offset'], recon_dict['tap_offset'],
                                              valid_flag * is_start,
                                              self.mse_loss)
        holding_loss = self.get_key_loss(input_dict['is_holding'], input_dict['is_holding'], valid_flag,
                                         self.label_smoothing_bce_loss)
        offset_end_loss = self.get_key_loss(input_dict['hold_end_offset'], recon_dict['hold_end_offset'],
                                            valid_flag * is_end,
                                            self.mse_loss)

        acc_start, precision_start, recall_start = self.classification_metrics(
            input_dict['tap'], recon_dict['tap'], valid_flag
        )
        acc_ln_start, precision_ln_start, recall_ln_start = self.classification_metrics(
            input_dict['is_holding'], recon_dict['is_holding'], valid_flag
        )

        loss = (start_loss * self.weight_tap +
                offset_start_loss * self.weight_start_offset +
                holding_loss * self.weight_holding +
                offset_end_loss * self.weight_end_offset)        
        return loss, {
            'loss': loss.detach().item(),
            'start_loss': start_loss.detach().item(),
            'offset_start_loss': offset_start_loss.detach().item(),
            'holding_loss': holding_loss.detach().item(),
            'offset_end_loss': offset_end_loss.detach().item(),
            "acc_rice": acc_start,
            "acc_ln": acc_ln_start,
            "precision_rice": precision_start,
            "precision_ln": precision_ln_start,
            "recall_rice": recall_start,
            "recall_ln": recall_ln_start,
        }

if __name__ == '__main__':
    loss_fn = MaimaiReconstructLoss(
        weight_tap=1.0,
        weight_start_offset=0.5,
        weight_holding=0.5,
        weight_end_offset=0.2,
        weight_touch=1.0,
        weight_touch_offset=0.5,
        weight_touch_holding=0.5,
        weight_touch_hold_end_offset=0.2,
        weight_star_pass_through=0.5,
        weight_star_end_offset=0.3,
        label_smoothing=0.001,
        gamma=2.0
    )
    print(loss_fn)