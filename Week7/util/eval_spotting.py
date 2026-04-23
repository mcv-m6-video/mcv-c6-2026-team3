"""
File containing main evaluation functions
"""

#Standard imports
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
import json
import os
from SoccerNet.Evaluation.ActionSpotting import average_mAP
import torch
import torch.nn.functional as F

#Local imports
from dataset.frame import FPS_SN

#Constants
INFERENCE_BATCH_SIZE = 4
INFERENCE_NUM_WORKERS = 4

@torch.no_grad()
def evaluate(model, dataset, batch_size=INFERENCE_BATCH_SIZE, num_workers=INFERENCE_NUM_WORKERS, nms_window = 5, delta=1):
    pred_dict = {}
    total_loss = 0.0
    num_batches = 0

    weights = torch.tensor([1.0] + [5.0] * (model._num_classes), dtype=torch.float32).to(model.device)

    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, len(dataset._class_dict)), np.float32), #scores matrix TxC (T with used stride)
            np.zeros(video_len, np.int32)) #support matrix T

    for clip in tqdm(DataLoader(
            dataset, num_workers=num_workers, pin_memory=True,
            batch_size=batch_size
    )):
        frames = clip['frame']
        labels = clip['label'].to(model.device).long()

        # Batched by dataloader
        batch_pred_scores, logits = model.predict(frames) # remove background class
        with torch.amp.autocast(model.device):
            loss_logits = logits.view(-1, model._num_classes + 1)
            loss_labels = labels.view(-1)
            loss = F.cross_entropy(loss_logits, loss_labels, reduction='mean', weight=weights)
            total_loss += loss.item()
            num_batches += 1

        for i in range(clip['frame'].shape[0]):
            video = clip['video'][i]
            scores, support = pred_dict[video]
            pred_scores = batch_pred_scores[i]
            start = clip['start'][i].item()
            if start < 0:
                pred_scores = pred_scores[-start:, :]
                start = 0
            end = start + pred_scores.shape[0]
            if end >= scores.shape[0]:
                end = scores.shape[0]
                pred_scores = pred_scores[:end - start, :]

            scores[start:end, :] += pred_scores[:, 1:] # remove background class
            support[start:end] += (pred_scores.sum(axis=1) != 0) * 1

    # For mAP evalaution
    detections_numpy = list()

    # Get detections_numpy (from predictions and applying NMS)
    for video, video_len, _ in dataset.videos:
        scores, support = pred_dict[video]
        support[support == 0] = 1
        scores = scores / support[:, np.newaxis] # mean over support predictions
        pred = apply_NMS(scores, nms_window, 0.05) # apply NMS
        detections_numpy.append(pred)

    targets_numpy = list()
    closests_numpy = list()
    # Get targets_numpy and closests_numpy (from ground truth)
    for video, video_len, _ in dataset.videos:
        targets = np.zeros((video_len, len(dataset._class_dict)), np.float32)
        labels = json.load(open(os.path.join(dataset._labels_dir, video, 'Labels-ball.json')))

        for annotation in labels["annotations"]:

            event = dataset._class_dict[annotation["label"]]
            frame = int(FPS_SN / dataset._stride * ( int(annotation["position"])/1000 )) #with the current framerate

            frame = min(frame, video_len-1)
            targets[frame, event-1] = 1

        targets_numpy.append(targets)

        closest_numpy = np.zeros(targets.shape) - 1
        # Get the closest action index
        for c in np.arange(targets.shape[-1]):
            indexes = np.where(targets[:, c] != 0)[0].tolist()
            if len(indexes) == 0:
                continue
            indexes.insert(0, -indexes[0])
            indexes.append(2 * closest_numpy.shape[0])
            for i in np.arange(len(indexes) - 2) + 1:
                start = max(0, (indexes[i - 1] + indexes[i]) // 2)
                stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
                closest_numpy[start:stop, c] = targets[indexes[i], c]
        closests_numpy.append(closest_numpy)

    avg_loss = total_loss / num_batches
    # Compute the performances
    mAP, AP_per_class, _, _, _, _ = (
        average_mAP(targets_numpy, detections_numpy, closests_numpy, FPS_SN / dataset._stride, deltas=np.array([delta]))
    )

    return mAP, AP_per_class, avg_loss


def apply_NMS(predictions, window, thresh=0.0):

    nf, nc = predictions.shape
    for i in range(nc):
        aux = predictions[:,i]
        aux2 = np.zeros(nf) -1
        while(np.max(aux) >= thresh):
            # Get the max remaining index and value
            max_value = np.max(aux)
            max_index = np.argmax(aux)
            # detections_NMS[max_index,i] = max_value

            nms_from = int(np.maximum(-(window/2)+max_index,0))
            nms_to = int(np.minimum(max_index+int(window/2), len(aux)))

            aux[nms_from:nms_to] = -1
            aux2[max_index] = max_value
        predictions[:,i] = aux2

    return predictions