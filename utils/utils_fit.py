import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import evaluate


def fit_one_epoch(model_train, model, loss_history, loss, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                  Epoch, cuda, test_loader, test_half_loader, Batch_size, lfw_eval_flag, fp16, scaler, save_period, save_dir, local_rank):
    total_triple_loss = 0
    total_CE_loss = 0
    total_accuracy = 0

    val_total_triple_loss = 0
    val_total_CE_loss = 0
    val_total_accuracy = 0

    best_lfw_acc = 0.0
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        images, labels = batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                labels = labels.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            outputs1, outputs2 = model_train(images, "train")

            _triplet_loss = loss(outputs1, Batch_size)
            # _CE_loss        = nn.NLLLoss()(F.log_softmax(outputs2, dim = -1), labels)
            _CE_loss = nn.CrossEntropyLoss()(outputs2, labels)
            _loss = _triplet_loss + _CE_loss

            _loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs1, outputs2 = model_train(images, "train")
                # outputs1 = umap.UMAP().fit_transform(outputs1.cpu().detach())
                # outputs1 = torch.as_tensor(outputs1).cuda(local_rank)

                _triplet_loss   = loss(outputs1, Batch_size)
                #_triplet_loss = 0
                # _CE_loss        = nn.NLLLoss()(F.log_softmax(outputs2, dim = -1), labels)
                _CE_loss = nn.CrossEntropyLoss()(outputs2, labels)
                _loss = _triplet_loss + _CE_loss
            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            accuracy = torch.mean((torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

        total_triple_loss   += _triplet_loss.item()
        #total_triple_loss += 0
        total_CE_loss += _CE_loss.item()
        total_accuracy += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_triple_loss': total_triple_loss / (iteration + 1),
                                'total_CE_loss': total_CE_loss / (iteration + 1),
                                'accuracy': total_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, labels = batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                labels = labels.cuda(local_rank)

            optimizer.zero_grad()
            outputs1, outputs2 = model_train(images, "train")

            _triplet_loss   = loss(outputs1, Batch_size)
            #_triplet_loss = 0
            # _CE_loss        = nn.NLLLoss()(F.log_softmax(outputs2, dim = -1), labels)
            _CE_loss = nn.CrossEntropyLoss()(outputs2, labels)
            _loss = _triplet_loss + _CE_loss

            accuracy = torch.mean((torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

            val_total_triple_loss   += _triplet_loss.item()
            #val_total_triple_loss += 0
            val_total_CE_loss += _CE_loss.item()
            val_total_accuracy += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'val_total_triple_loss': val_total_triple_loss / (iteration + 1),
                                'val_total_CE_loss': val_total_CE_loss / (iteration + 1),
                                'val_accuracy': val_total_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if lfw_eval_flag:
        print("开始进行LFW数据集的验证。")
        labels, distances = [], []
        for _, (data_a, data_p, label) in enumerate(test_loader):
            with torch.no_grad():
                data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
                if cuda:
                    data_a, data_p = data_a.cuda(local_rank), data_p.cuda(local_rank)
                out_a, out_p = model_train(data_a), model_train(data_p)
                dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
            distances.append(dists.data.cpu().numpy())
            labels.append(label.data.cpu().numpy())

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for dist in distances for subdist in dist])
        _, _, accuracy, _, _, _, _ = evaluate(distances, labels)

    if lfw_eval_flag:
        print("开始进行LFW_Halftone数据集的验证。")
        labels, distances = [], []
        for _, (data_a, data_p, label) in enumerate(test_half_loader):
            with torch.no_grad():
                data_a, data_p = data_a.type(torch.FloatTensor), data_p.type(torch.FloatTensor)
                if cuda:
                    data_a, data_p = data_a.cuda(local_rank), data_p.cuda(local_rank)
                out_a, out_p = model_train(data_a), model_train(data_p)
                dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
            distances.append(dists.data.cpu().numpy())
            labels.append(label.data.cpu().numpy())

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for dist in distances for subdist in dist])
        _, _, accuracy_half, _, _, _, _ = evaluate(distances, labels)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')

        if lfw_eval_flag:
            print('LFW_Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
            print('LFW_Half_Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy_half), np.std(accuracy_half)))
        loss_history.append_loss(epoch, np.mean(accuracy) if lfw_eval_flag else total_accuracy / epoch_step, \
                                 (total_triple_loss + total_CE_loss) / epoch_step,
                                 (val_total_triple_loss + val_total_CE_loss) / epoch_step_val)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.4f' % ((total_triple_loss + total_CE_loss) / epoch_step))

        with open(os.path.join(save_dir, 'log.txt'), 'a') as f:
            f.write('Epoch:' + str(epoch + 1) + '/' + str(Epoch) + '\n')
            f.write('Total Loss: %.4f' % ((total_triple_loss + total_CE_loss) / epoch_step) + '\n')
            f.write('Val Total Loss: %.4f' % ((val_total_triple_loss + val_total_CE_loss) / epoch_step_val) + '\n')
            f.write('Total Accuracy: %.4f' % (total_accuracy / epoch_step) + '\n')
            f.write('Val Total Accuracy: %.4f' % (val_total_accuracy / epoch_step_val) + '\n')
            if lfw_eval_flag:
                f.write('LFW_Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)) + '\n')
                f.write('LFW_Half_Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy_half), np.std(accuracy_half)) + '\n')
            f.write('lr: %.6f' % get_lr(optimizer) + '\n')
            f.write('\n')

        cur_lfw_acc = np.mean(accuracy)
        if cur_lfw_acc > best_lfw_acc:
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
            best_lfw_acc = cur_lfw_acc
        if epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f_last.pth' % (Epoch,
                                                                                                             (total_triple_loss + total_CE_loss) / epoch_step,
                                                                                                             (val_total_triple_loss + val_total_CE_loss) / epoch_step_val)))
        return best_lfw_acc
