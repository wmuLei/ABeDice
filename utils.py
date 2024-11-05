import PIL
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from torch.nn import functional as F
import matplotlib.pyplot as plt


class ABeDice(nn.Module):
    def __init__(self, alpha=1, beta=0, n_classes=9):
        super(ABeDice, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha
        self.beta = beta

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _ABeDice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, logits, target):

        p = torch.softmax(logits, dim=1)

        t = 1
        for i in range(self.beta):
            t = 1 - p ** t
        p = p ** (self.alpha * t)

        target = self._one_hot_encoder(target)

        assert p.size() == target.size(), 'predict {} & target {} shape do not match'.format(p.size(), target.size())

        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(p[:, i], target[:, i])
            loss += dice
        return loss / self.n_classes


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class IoULoss(nn.Module):
    def __init__(self, n_classes):
        super(IoULoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _iou_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target)
        z_sum = torch.sum(score)
        loss = (intersect + smooth) / (z_sum + y_sum - intersect + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            iou = self._iou_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - iou.item())
            loss += iou * weight[i]
        return loss / self.n_classes


class GDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(GDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _gdice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        w = torch.sum(target, axis=[0, 2, 3])
        w = 1 / (w ** 2 + smooth)
        intersect = torch.sum(w * torch.sum(score * target, axis=[0, 2, 3]))
        union = torch.sum(w * torch.sum(score + target, axis=[0, 2, 3]))
        loss = 1 - (2. * intersect) / (union + smooth)
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        # for i in range(0, self.n_classes):
        #     gdice = self._gdice_loss(inputs[:, i], target[:, i])
        #     class_wise_dice.append(1.0 - gdice.item())
        #     loss += gdice * weight[i]

        target = target.float()
        smooth = 1e-5
        w = torch.sum(target, axis=[0, 2, 3])
        w = 1 / (w ** 2 + smooth)
        intersect = torch.sum(w * torch.sum(inputs * target, axis=[0, 2, 3]))
        union = torch.sum(w * torch.sum(inputs + target, axis=[0, 2, 3]))
        loss = 1 - (2. * intersect) / (union + smooth)

        return loss / self.n_classes


class HDLoss(nn.Module):    #Hardness-weighted Dice
    def __init__(self, n_classes):
        super(HDLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _HD_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target)
        z_sum = torch.sum(score)
        w = 0.5 * torch.abs(z_sum - y_sum) + (1.0 - 0.5)
        loss = (2 * w * intersect + smooth) / (w * (z_sum + y_sum) + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            hdloss = self._HD_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - hdloss.item())
            loss += hdloss * weight[i]
        return loss / self.n_classes



class FocalDiceLoss(nn.Module):     #focal dice loss
    def __init__(self, n_classes):
        super(FocalDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _focaldice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target)
        z_sum = torch.sum(score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = loss ** (1/2)    #β=2
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            focaldice = self._focaldice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - focaldice.item())
            loss += focaldice * weight[i]
        return loss / self.n_classes


class eDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(eDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _edice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)

        inputs = inputs ** (2 * ((1 - inputs) ** 2))

        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            edice = self._edice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - edice.item())
            loss += edice * weight[i]
        return loss / self.n_classes


class BoundaryDoULoss(nn.Module):
    def __init__(self, n_classes):
        super(BoundaryDoULoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _adaptive_size(self, score, target):
        kernel = torch.Tensor([[0,1,0], [1,1,1], [0,1,0]])
        padding_out = torch.zeros((target.shape[0], target.shape[-2]+2, target.shape[-1]+2))
        padding_out[:, 1:-1, 1:-1] = target
        h, w = 3, 3

        Y = torch.zeros((padding_out.shape[0], padding_out.shape[1] - h + 1, padding_out.shape[2] - w + 1)).cuda()
        for i in range(Y.shape[0]):
            Y[i, :, :] = torch.conv2d(target[i].unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0).cuda(), padding=1)
        Y = Y * target
        Y[Y == 5] = 0
        C = torch.count_nonzero(Y)
        S = torch.count_nonzero(target)
        smooth = 1e-5
        alpha = 1 - (C + smooth) / (S + smooth)
        alpha = 2 * alpha - 1

        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        alpha = min(alpha, 0.8)  ## We recommend using a truncated alpha of 0.8, as using truncation gives better results on some datasets and has rarely effect on others.
        loss = (z_sum + y_sum - 2 * intersect + smooth) / (z_sum + y_sum - (1 + alpha) * intersect + smooth)

        return loss

    def forward(self, inputs, target, softmax=False):
        inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())

        loss = 0.0
        for i in range(0, self.n_classes):
            loss += self._adaptive_size(inputs[:, i], target[:, i])
        return loss / self.n_classes



def sum_tensor(inp, axes, keepdim=False):
    # from: nnunet/utilities/tensor_utilities.py
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp

class SSLoss(nn.Module):
    def __init__(self, num_classes=None, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1., square=False):
        """
        Sensitivity-Specifity loss
        """
        super(SSLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.r = 0.1  # weight parameter in SS paper

    def forward(self, net_output, gt, softmax=False):
        shp_x = net_output.shape
        shp_y = gt.shape
        # class_num = shp_x[1]

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                gt = gt.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(shp_x)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)

        # no object value
        bg_onehot = 1 - y_onehot
        squared_error = (y_onehot - net_output) ** 2
        specificity_part = sum_tensor(squared_error * y_onehot, axes) / (sum_tensor(y_onehot, axes) + self.smooth)
        sensitivity_part = sum_tensor(squared_error * bg_onehot, axes) / (sum_tensor(bg_onehot, axes) + self.smooth)

        ss = self.r * specificity_part + (1 - self.r) * sensitivity_part

        if not self.do_bg:
            if self.batch_dice:
                ss = ss[1:]
            else:
                ss = ss[:, 1:]
        ss = ss.mean()

        return ss


class TverskyLoss(nn.Module):
    def __init__(self, n_classes):
        super(TverskyLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _tversky_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        a = 0.3
        b = 0.7
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target)
        z_sum = torch.sum(score)
        loss = (intersect + smooth) / (intersect + b * (z_sum - intersect) + a * (y_sum - intersect) + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            tversky = self._tversky_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - tversky.item())
            loss += tversky * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
        return dice, jc, hd95, asd
    # elif pred.sum() > 0 and gt.sum() == 0:
    #     return 1, 0, 0, 0
    else:
        return 0, 0, 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:

        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
    return metric_list


def test_single_volume2(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        x, y = image.shape[-2:]
        if x != patch_size[0] or y != patch_size[1]:
            slice = zoom(image, (1, patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
        input = torch.from_numpy(slice).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            outputs = net(input)
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
            else:
                pred = out
            prediction = pred
    else:

        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        # prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        # img_itk.SetSpacing((1, 1, z_spacing))
        # prd_itk.SetSpacing((1, 1, z_spacing))
        # lab_itk.SetSpacing((1, 1, z_spacing))
        # sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        # sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        # sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")

        image = (image * 255).astype(np.uint8).transpose(1, 2, 0)
        prediction = (prediction * 127.5).astype(np.uint8)
        label = (label * 127.5).astype(np.uint8)
        img_itk = PIL.Image.fromarray(image)
        prd_itk = PIL.Image.fromarray(prediction)
        lab_itk = PIL.Image.fromarray(label)
        prd_itk.save(test_save_path + '/' + case + "_pred.png")
        img_itk.save(test_save_path + '/' + case + "_img.png")
        lab_itk.save(test_save_path + '/' + case + "_gt.png")

    return metric_list
