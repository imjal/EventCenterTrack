from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import copy
from .prophesee_src.io.psee_loader import PSEELoader
from .prophesee_src.visualize.vis_utils import make_binary_histo
from ..event_generic_dataset import EventGenericDataset
from glob import glob
from itertools import chain
import time
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches



class EventStreamer:
    def __init__(self, opt, events, box_events, index, delta_t):
        self.delta_t = delta_t
        self.events = events
        self.box_events = box_events
        self.disc_events = []
        self.disc_boxevents = []
        self.event_index = index
        self.obj_id = 0
        self.idx = 0
        self.opt = opt
        while not self.events.done:
          self.disc_events += [self.events.load_delta_t(delta_t)]
          # self.disc_events += [ np.array([np.array(list(evs)) for evs in self.events.load_delta_t(delta_t)])]
        while not self.box_events.done:
          self.disc_boxevents += [self.box_events.load_delta_t(delta_t)]
    
    # dtype=[('t', '<u4'), ('x', '<u2'), ('y', '<u2'), ('p', 'u1')])
    def discretize(self, events, num_bins=9):
      def k_b(X):
          return np.maximum(0, 1 - np.abs(X))
      
      # break up by time
      V = np.zeros((256 * 320, num_bins), dtype=np.float32)

      if len(events) == 0:
        return V.reshape(256, 320, num_bins)
      t_0 = events[0][0]
      t_n = events[len(events)-1][0]
      # events n x 4 
      norm_t_res = ((events['t'] - t_0)/(t_n - t_0) ) * (num_bins-1)
      norm_t_integer = norm_t_res.astype(np.int32)
      neg1to1_res = 2* events['p'].astype(np.float32) -1 
      k_b_res = k_b(norm_t_integer - norm_t_res)
      res_val = neg1to1_res * k_b_res

      for bin in range(num_bins):
        X = events['y'][norm_t_integer == bin]
        Y = events['x'][norm_t_integer == bin]
        res = res_val[norm_t_integer == bin]
        np.put_along_axis(V[:, bin], indices = X*320 + Y, values=res, axis=0)

      # for i, x in enumerate(events):
      #     V[int(x[2]), int(x[1]), int(norm_t_res[i])] += res_val[i]
      V = V.reshape(256, 320, num_bins)
      return V

    def __next__(self):
        if self.idx >= len(self.disc_events) or self.idx >= len(self.disc_boxevents): 
          raise StopIteration
        evs, boxes = self.disc_events[self.idx], self.disc_boxevents[self.idx]
        while(len(evs) == 0):
          if self.idx >= len(self.disc_events) or self.idx >= len(self.disc_boxevents): 
            raise StopIteration
          self.idx +=1
          evs, boxes = self.disc_events[self.idx], self.disc_boxevents[self.idx]
          
        evs = evs[evs['t'] > (evs['t'][len(evs)-1] - 50000)] # only process the last 50,000 microseconds from the box timestamp
        
        cur_events = self.discretize(evs)

        # Debugging code to check if it works

        # img = make_binary_histo(evs)
        # cv2.imwrite("/home/jl5/tmp.png", ((np.sum(cur_events, axis=2) +1)/2) * 255)

        # if len(boxes) > 0:
        #   # Create figure and axes
        #   fig,ax = plt.subplots(1)
        #   # Display the image
        #   ax.imshow(((np.sum(cur_events, axis=2) +1)/2)* 255, cmap='gray') 

        anns = []
        for x in boxes:
            # dtype=[('ts', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), 
            #  ('class_id', 'u1'), ('confidence', '<f4'), ('track_id', '<u4')])
            ann = {
                "id": self.obj_id,
                "image_id": int(x[0]),
                "category_id": int(x[5]),
                "area": int(x[4]) * int(x[3]),
                "bbox": [int(x[1]), int(x[2]), int(x[3]), int(x[4])],
                "track_id": int(x[7])
            }
            self.obj_id += 1
            anns += [ann]
            # if ann['category_id'] == 0: 
            #   edge_color = 'r' # cars 
            # else:
            #   edge_color='b' # pedestrians
            # rect = patches.Rectangle((ann['bbox'][0],ann['bbox'][1]),ann['bbox'][2],ann['bbox'][3],linewidth=1, edgecolor=edge_color,facecolor='none')
            # ax.add_patch(rect)
        
        # if len(anns) > 0:
        #   plt.savefig('/home/jl5/tmp.png')
        #   print(self.idx)
        #   import pdb; pdb.set_trace()
        # else:
        #   plt.clf()
        
        # grab the last timestamp as the id, frame_id and video_id are pretty much unused
        if len(evs) == 0:
          img_info = {'id': 0, 'frame_id': self.idx, 'video_id': self.event_index}
        else:
          img_info = {'id': np.int64(evs[len(evs)-1][0]), 'frame_id': self.idx, 'video_id': self.event_index}
        self.idx+=1
        return cur_events, anns, img_info
        
    def __iter__(self):
        return self


class AggEventStreamer:
    def open_file(self, td_file):
        return open(td_file).read().splitlines()

    def __init__(self, opt, td_file_name, data_dir, delta_t=1000000):
        self.opt = opt
        self.loop = True
        self.idx = 0
        self.td_file_name = td_file_name
        self.td_files = [os.path.join(data_dir, x) for x in self.open_file(td_file_name)]
        video = PSEELoader(self.td_files[self.idx])
        box_video = PSEELoader(glob(self.td_files[self.idx].split('_td.dat')[0] +  '*.npy')[0])
        
        self.length = len(self.td_files)
        self.delta_t = delta_t
        self.streams = []
        # for d in range(self.length):
        #     print(self.td_files[d])
        #     input_stream = EventStreamer(self.videos[d], self.box_videos[d], index=d, delta_t=delta_t)
        self.streams.append(iter(EventStreamer(self.opt, video, box_video, index=self.idx, delta_t=delta_t)))
        self.seq_stream = self.streams[0]
        # self.seq_stream = chain(*(self.streams))

    def __next__(self):
      try: 
        out = next(self.seq_stream)
      except StopIteration:
        if self.idx >= self.length-1:
          raise StopIteration
        self.idx +=1
        video = PSEELoader(self.td_files[self.idx])
        box_video = PSEELoader(glob(self.td_files[self.idx].split('_td.dat')[0] +  '*.npy')[0])
        self.seq_stream = iter(EventStreamer(self.opt, video, box_video, index=self.idx, delta_t=self.delta_t))
        out = next(self.seq_stream)
      return out

    def __iter__(self):
        return self


class PropheseGen1(EventGenericDataset):
  default_resolution = [256, 320] # [240, 304]
  num_categories = 2
  class_name = ['car', 'person']
  _valid_ids = [0, 1]
  cat_ids = {v: i + 1 for i, v in enumerate(_valid_ids)}
  max_objs = 128

  def __init__(self, opt, split):
    # load annotations
    self.obj_id = 0
    self.opt = opt
    ann_data_dir = os.path.join(opt.ann_data_dir, 'detection_dataset_duration_60s_ratio_1.0')
    if opt.trainval:
      split = 'test'
      ann_path = os.path.join(ann_data_dir, 'test')
    else:
      ann_path = os.path.join(ann_data_dir, 'train')

    self.stream = iter(AggEventStreamer(self.opt, self.opt.data_stream_file, ann_path))
    super(PropheseGen1, self).__init__(opt, split)

    print('Loaded {} stream'.format(split))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    detections = []
    for image_id in all_bboxes:
      if type(all_bboxes[image_id]) != type({}):
        # newest format
        for j in range(len(all_bboxes[image_id])):
          item = all_bboxes[image_id][j]
          cat_id = item['class'] - 1
          category_id = self._valid_ids[cat_id]
          bbox = item['bbox']
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          bbox_out  = list(map(self._to_float, bbox[0:4]))
          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(item['score']))
          }
          detections.append(detection)
    return detections

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
                open('{}/results_coco.json'.format(save_dir), 'w'))
  
  def run_eval(self, results, save_dir):
    self.save_results(results, save_dir)
    coco_dets = self.coco.loadRes('{}/results_prophesee.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
  