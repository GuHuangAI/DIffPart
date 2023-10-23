import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from diff_nerf.data import VolumeDataset, default_collate
from torch.utils.data import DataLoader
import os
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

def get_inplanes():
    return [64, 128, 256, 512]

def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=5,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=10):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def generate_model(model_depth, path=None, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)
    if path is not None:
        ck = torch.load(path, map_location=lambda storage, loc: storage)
        model.load_state_dict(ck['model'])
    return model

def train(train_data, classifier, device, args):
    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad,
                                  classifier.parameters()), lr=args.lr)
    lr_schd = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(0.6 * args.epochs), int(0.9 * args.epochs)], 0.1)
    best_acc = 0
    # steps = 0
    last_epch = 0
    if args.resume:
        ck = torch.load(args.resume_model, map_location=lambda storage, loc: storage)
        classifier.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        lr_schd.load_state_dict(ck['lr_schd'])
        best_acc = ck['best_acc']
        last_epch = ck['last_epoch']

    for epoch in range(1 + last_epch, args.epochs + 1):
        classifier.train()
        for i, batch in enumerate(train_data):
            # print(type(batch[0]))
            # print(type(batch[1]))
            # print(type(batch[2]))
            dance = batch['input'] / 3.28
            label = batch['class_id']
            dance = dance.to(device)
            label = label.to(device)
            # print(label)
            # dance, label, _ = map(lambda x: x.to(device), batch)
            dance = dance.type(torch.cuda.FloatTensor)
            optimizer.zero_grad()
            logits = classifier(dance)
            loss = F.cross_entropy(logits, label)
            loss.backward()
            optimizer.step()
            # steps += 1

            if i % args.log_interval == 0:
                corrects = (torch.max(logits, 1)[1].view(label.size()).data == label.data).sum()
                train_acc = 100.0 * corrects / dance.shape[0]
                print('\rEpoch/Batch[{}/{}] - loss: {:.6f}  acc: {:.4f}%({}/{})  best_acc: {:.4f}'.format(epoch,
                                                                                        i,
                                                                                        loss.item(),
                                                                                        train_acc,
                                                                                        corrects,
                                                                                        dance.shape[0],
                                                                                        best_acc))
        lr_schd.step()
        # evaluate the model on test set at each epoch
        dev_acc = evaluate(train_data, classifier, device, args)
        if dev_acc > best_acc:
            best_acc = dev_acc
            print('Saving best model, acc: {:.4f}%\n'.format(best_acc))
            save(classifier, optimizer, lr_schd, best_acc, args.save_dir, args.save_model, epoch)

@torch.no_grad()
def evaluate(dev_data, classifier, device, args):
    classifier.eval()
    corrects, avg_loss, size = 0, 0, 0
    for i, batch in enumerate(dev_data):
        dance = batch['input'] / 3.28
        label = batch['class_id']
        dance = dance.to(device)
        label = label.to(device)
        # print(dance.shape)
        # dance, label, _ = map(lambda x: x.to(device), batch)
        dance = dance.type(torch.cuda.FloatTensor)
        logits = classifier(dance)
        loss = F.cross_entropy(logits, label)
        avg_loss += loss.item()
        corrects += (torch.max(logits, 1)
                     [1].view(label.size()).data == label.data).sum()
        size += dance.shape[0]

    avg_loss /= size
    accuracy = 100.0 * corrects / size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy

def save(model, optimizer, lr_schd, best_acc, save_dir, save_prefix, epoch):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    ck = {}
    save_path = os.path.join(save_dir, save_prefix)
    ck['model'] = model.state_dict()
    ck['optimizer'] = optimizer.state_dict()
    ck['lr_schd'] = lr_schd.state_dict()
    ck['best_acc'] = best_acc
    ck['last_epoch'] = epoch
    torch.save(ck, save_path)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tar_path', type=str, default='/media/huang/T7/data/diff_nerf/DVGO_results_64x64x64',
                        help='the directory of dance data')
    parser.add_argument('--image_path', type=str, default='/media/huang/T7/data/diff_nerf/ShapeNet_Render',
                        help='the directory of music feature data')
    parser.add_argument('--save_model', type=str, default='best_model_res18_rotate_non_normalize',
                        help='model name')
    parser.add_argument('--save_dir', metavar='PATH', default='/media/huang/T7/data/diff_nerf/classifier')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--test_interval', type=int, default=50)
    parser.add_argument('--interval', type=int, default=50)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--resume_model', type=str, default='')

    args = parser.parse_args()
    classifier = generate_model(18)
    dataset = VolumeDataset(tar_path=args.tar_path,
                            image_path=args.image_path,
                            load_rgb_net=False,
                            load_mask_cache=False,
                            use_rotate_ransform=True,
                            load_render_kwargs=False,
                            sample_num=1,
                            normalize=False)
    # dataset = dataset[:900]
    train_data = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                    pin_memory=True, num_workers=2, collate_fn=default_collate)
    device = torch.device('cuda')
    classifier = classifier.to(device)
    train(train_data, classifier, device, args)
    print('Training Complete!')

if __name__ == '__main__':
    main()