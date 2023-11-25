#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Copyright: Copyright(c) 2018, seeprettyface.com, BUPT_GWY contributes the model.
Thanks to STTN provider: https://github.com/researchmm/STTN
Author: BUPT_GWY
Contact: a312863063@126.com
"""

import cv2
import numpy as np
import importlib
import argparse
import sys
import torch
import os
import subprocess
import platform
import threading
from pathlib import Path
from torchvision import transforms

# My libs
from core.utils import Stack, ToTorchFormatTensor

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])
total_memory = torch.cuda.get_device_properties(0).total_memory
fraction = 10*1024*1024*1024/total_memory
torch.cuda.set_per_process_memory_fraction(fraction, 0)

def get_parser():
    parser = argparse.ArgumentParser(description="STTN")

    parser.add_argument("-t", "--task", type=str, help='CHOOSE THE TASK：delogo or detext', default='detext')
    parser.add_argument("-v", "--video", type=str)
    parser.add_argument("-b", "--box", nargs='+', type=int,
                    help='Specify a mask box for the subtilte. Syntax: (top, bottom, left, right).')
    parser.add_argument("-e", "--exclude_ranges", nargs='+', type=str,
                        help='Specify exclude time ranges which not recognize. Syntax: (0_13880, 143360_146333).')
    parser.add_argument("-r", "--result",  type=str, default='result/')
    parser.add_argument("-d", "--dual",  type=bool, default=False, help='Whether to display the original video in the final video')
    parser.add_argument("-w", "--weight",   type=str, default='pretrained_weight/detext_trial.pth')

    parser.add_argument("--model", type=str, default='auto-sttn')
    parser.add_argument("-g", "--gap",   type=int, default=200, help='set it higher and get result better')
    parser.add_argument("-l", "--ref_length",   type=int, default=5)
    parser.add_argument("-n", "--neighbor_stride",   type=int, default=5)

    return parser

def read_frame_info_from_video(args):
    reader = cv2.VideoCapture(args.video)
    if not reader.isOpened():
        print("fail to open video in {}".format(args.video))
        sys.exit(1)
    frame_info = {'w_ori': int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)), 'h_ori': int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)), 'fps': reader.get(cv2.CAP_PROP_FPS), 'len': int(reader.get(cv2.CAP_PROP_FRAME_COUNT))}
    return reader, frame_info

def read_mask(path):
    img = cv2.imread(path, 0)
    ret, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    img = img[:, :, None]
    return img

# sample reference frames from the whole video
def get_ref_index(neighbor_ids, length, ref_length):
    ref_index = []
    for i in range(0, length, ref_length):
        if not i in neighbor_ids:
            ref_index.append(i)
    return ref_index

def pre_process(args):
    # prepare dataset, encode all frames into deep space
    reader, frame_info = read_frame_info_from_video(args)
    if not os.path.exists(args.result):
        os.makedirs(args.result)
    video_path = str(Path(args.result) / f"{Path(args.video).stem}_{args.task}.mp4")
    video_h = frame_info['h_ori'] if not args.dual else frame_info['h_ori'] * 2
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_info['fps'], (frame_info['w_ori'], video_h))
    print('Loading video from: {}'.format(args.video))
    print('Loading box from: {}'.format(args.box))
    print('args: {}'.format(args))
    print('--------------------------------------')

    clip_gap = args.gap  # processing how many frames during one period
    rec_times = frame_info['len'] // clip_gap if frame_info['len'] % clip_gap == 0 else frame_info['len'] // clip_gap + 1
    y11 = args.box[0]
    y12 = min(frame_info['h_ori'], args.box[1])
    x11 = args.box[2]
    x12 = min(frame_info['w_ori'], args.box[3])
    mask = np.zeros((frame_info['h_ori'], frame_info['w_ori']), dtype="uint8")
    site = np.array([[[x12, y12], [x11, y12], [x11, y11], [x12, y11]]], dtype=np.int32)
    cv2.polylines(mask, site, 1, 255)
    cv2.fillPoly(mask, site, 255)
    mask_encode = cv2.imencode('.png', mask)[1]
    img = cv2.imdecode(np.frombuffer(mask_encode.tobytes(), np.uint8), 0)
    _, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    mask = img[:, :, None]
    return clip_gap, frame_info, mask, reader, rec_times, video_path, writer

def process(args, frames, model, device, w, h):
    video_length = len(frames)
    feats = _to_tensors(frames).unsqueeze(0) * 2 - 1

    feats = feats.to(device)
    comp_frames = [None] * video_length

    with torch.no_grad():
        feats = model.encoder(feats.view(video_length, 3, h, w))
        _, c, feat_h, feat_w = feats.size()
        feats = feats.view(1, video_length, c, feat_h, feat_w)

    # completing holes by spatial-temporal transformers
    for f in range(0, video_length, args.neighbor_stride):
        neighbor_ids = [i for i in range(max(0, f - args.neighbor_stride), min(video_length, f + args.neighbor_stride + 1))]
        ref_ids = get_ref_index(neighbor_ids, video_length, args.ref_length)
        with torch.no_grad():
            pred_feat = model.infer(
                feats[0, neighbor_ids + ref_ids, :, :, :])
            pred_img = torch.tanh(model.decoder(
                pred_feat[:len(neighbor_ids), :, :, :])).detach()
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(
                    np.uint8)
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(
                        np.float32) * 0.5 + img.astype(np.float32) * 0.5
    return comp_frames

def get_inpaint_mode_for_detext(h_ori, h, mask):  # get inpaint segment
    mode = []
    to_h = from_h = h_ori   # the subtitles are usually underneath
    while from_h != 0:
        if to_h - h < 0:
            from_h = 0
            to_h = h
        else:
            from_h = to_h - h
        if not np.all(mask[from_h:to_h, :] == 0) and np.sum(mask[from_h:to_h, :]) > 10:
            if to_h != h_ori:
                move = 0
                while to_h + move < h_ori and not np.all(mask[to_h+move, :] == 0):
                    move += 1
                if to_h + move < h_ori and move < h:
                    to_h += move
                    from_h += move
            mode.append((from_h, to_h))
        to_h -= h
    return mode

class ProcessThread(threading.Thread):
    def __init__(self, args, k, comps, frames, model, device, w, h):
        threading.Thread.__init__(self)
        self.setDaemon(True)
        self.args = args
        self.k = k
        self.comps = comps
        self.frames = frames
        self.model = model
        self.device = device
        self.w = w
        self.h = h

    def run(self):
        self.comps[self.k] = process(self.args, self.frames[self.k], self.model, self.device, self.w, self.h)

def main(opts=None):  # detext
    parser = get_parser()
    args = parser.parse_args(opts)
    # set up models
    w, h = 640, 120
    clip_gap, frame_info, mask, reader, rec_times, video_path, writer = pre_process(args)
    gpu_count = torch.cuda.device_count()
    print('Task: ', args.task)
    models = []
    print('gpu_count:', gpu_count)
    for i in range(gpu_count):
        device = torch.device(f"cuda:{i}")
        net = importlib.import_module('model.' + args.model)
        model = net.InpaintGenerator().to(device)
        data = torch.load(args.weight, map_location=device)
        model.load_state_dict(data['netG'])
        model.eval()
        models.append((model, device))
    print('Loading weight from: {}'.format(args.weight))

    split_h = int(frame_info['w_ori'] * 3 / 16)
    mode = get_inpaint_mode_for_detext(frame_info['h_ori'], split_h, mask)

    ranges = [[int(i) for i in r.split('_')] for r in args.exclude_ranges]
    for i in range(rec_times):
        start_f = i * clip_gap
        end_f = min((i + 1) * clip_gap, frame_info['len'])
        print('Processing:', start_f+1, '-', end_f, ' / Total:', frame_info['len'])

        frames_hr = []
        frames = {}
        for k in range(len(mode)):
            frames[k] = []
        for j in range(start_f, end_f):
            success, image = reader.read()
            frames_hr.append(image)
            for k in range(len(mode)):
                image_crop = image[mode[k][0]:mode[k][1], :, :]
                image_resize = cv2.resize(image_crop, (w, h))
                frames[k].append(image_resize)

        _frames_hr = []
        _frames = {}
        _comps = {}
        _states = []
        for k in range(len(mode)):
            _frames[k] = []
            _comps[k] = []
        for j in range(len(frames_hr)):
            msec = (j + start_f) * 1000 / frame_info['fps']
            is_excluded = False
            for r in ranges:
                if r[0] <= msec <= r[1]:
                    is_excluded = True
                    break

            if len(_states) == 0 or _states[-1] != is_excluded:
                _states.append(is_excluded)
                _frames_hr.append([])
                for k in range(len(mode)):
                    _frames[k].append([])
                    _comps[k].append([])
            _frames_hr[-1].append(frames_hr[j])
            for k in range(len(mode)):
                _frames[k][-1].append(frames[k][j])
        print(f'Processing exclude time ranges, split into {len(_frames_hr)} segments')
        for j in range(len(_frames_hr)):
            if _states[j]:
                for frame in _frames_hr[j]:
                    writer.write(frame)
            else:
                frames_hr = _frames_hr[j]
                frames = {}
                comps = {}
                for k in range(len(mode)):
                    frames[k] = _frames[k][j]
                    comps[k] = _comps[k][j]

                threads = [None] * gpu_count
                for k in range(len(mode)):
                    need_waiting = True
                    for thread in threads:
                        if thread is None:
                            need_waiting = False
                            break
                    if need_waiting:
                        for thread in threads:
                            thread.join()
                        threads = [None] * gpu_count
                    idx = k % gpu_count
                    t = ProcessThread(args, k, comps, frames, models[idx][0], models[idx][1], w, h)
                    t.start()
                    threads[idx] = t
                    #comps[k] = process(args, frames[k], model, device, w, h)
                for thread in threads:
                    if thread is not None:
                        thread.join()

                if mode is not []:
                    for m in range(len(frames_hr)):
                        frame_ori = frames_hr[m].copy()
                        frame = frames_hr[m]
                        for k in range(len(mode)):
                            comp = cv2.resize(comps[k][m], (frame_info['w_ori'], split_h))
                            comp = cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB)
                            mask_area = mask[mode[k][0]:mode[k][1], :]
                            frame[mode[k][0]:mode[k][1], :, :] = mask_area * comp + (1 - mask_area) * frame[mode[k][0]:mode[k][1], :, :]
                        if args.dual:
                            frame = np.vstack([frame_ori, frame])
                        writer.write(frame)

    writer.release()
    out_path = str(Path(args.result) / f"{Path(args.video).stem}_out.mp4")
    command = 'ffmpeg -i {} -i {} -map 0:a -map 1:v -y {}'.format(args.video, video_path, out_path)
    subprocess.call(command, shell=platform.system() != 'Windows')
    print('--------------------------------------')
    print('Finish in {}'.format(out_path))

if __name__ == '__main__':
    main()
