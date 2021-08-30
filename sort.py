"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)

total_ids= []
Total_Pool = []

occlusion_ids = []
Occlusion_Pool = []
evaporation_ids = []
Evaporation_Pool = []

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])                                      
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)                                              
  return(o)  


def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = iou_batch(detections, trackers)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold
    self.trackers = []
    self.frame_count = 0

  def occlu_iou(self, my_trk, other_trk):
    """
    겹쳐지는지 여부 반환
    """
    left_top_x_max = np.maximum(my_trk[0][0], other_trk[0][0])
    left_top_y_max = np.maximum(my_trk[0][1], other_trk[0][1])
    right_bottom_x_min = np.minimum(my_trk[0][2], other_trk[0][2])
    right_bottom_y_min = np.minimum(my_trk[0][3], other_trk[0][3])
    w = np.maximum(0., right_bottom_x_min - left_top_x_max)
    h = np.maximum(0., right_bottom_y_min - left_top_y_max)
    wh = w * h
    over = wh / ((my_trk[0][2] - my_trk[0][0]) * (my_trk[0][3] - my_trk[0][1])
            + (other_trk[0][2] - other_trk[0][0]) * (other_trk[0][3] - other_trk[0][1]) - wh)
    return over

  def evap_iou(self, new_trk, over_trk):
    """
    겹쳐지는지 여부 반환
    """
    left_top_x_max = np.maximum(new_trk[0][0], over_trk['left_top_x'])
    left_top_y_max = np.maximum(new_trk[0][1], over_trk['left_top_y'])
    right_bottom_x_min = np.minimum(new_trk[0][2], over_trk['right_bottom_x'])
    right_bottom_y_min = np.minimum(new_trk[0][3], over_trk['right_bottom_y'])
    w = np.maximum(0., right_bottom_x_min - left_top_x_max)
    h = np.maximum(0., right_bottom_y_min - left_top_y_max)
    wh = w * h
    over = wh / ((new_trk[0][2] - new_trk[0][0]) * (new_trk[0][3] - new_trk[0][1])
            + (over_trk['right_bottom_x'] - over_trk['left_top_x']) * (over_trk['right_bottom_y'] - over_trk['left_top_y']) - wh)
    return over

  def merge_sort(self, array):
	  if len(array) < 2:
	  	return array
	  mid = len(array) // 2
	  low_arr = self.merge_sort(array[:mid])
	  high_arr = self.merge_sort(array[mid:])

	  merged_arr = []
	  l = h = 0
	  while l < len(low_arr) and h < len(high_arr):
	  	if low_arr[l]['iou'] > high_arr[h]['iou']:
	  		merged_arr.append(low_arr[l])
	  		l += 1
	  	else:
	  		merged_arr.append(high_arr[h])
	  		h += 1
	  merged_arr += low_arr[l:]
	  merged_arr += high_arr[h:]
	  return merged_arr

  def occlusion_overlapping(self, over, trk):
    """
    새로운 아이디가 기존 아이디와 같은지 아닌지 반환
    """
    global occlusion_ids, Occlusion_Pool
    # 가장 많이 겹쳐지는 iou의 추적 아이디 할당
    overlap_max = over
    idx_max = over
    for oc_idx, oc_trk in enumerate(Occlusion_Pool): # 새로운 추적 아이디 유지            
      o_iou = self.occlu_iou(trk, oc_trk)
      if overlap_max < o_iou:
        overlap_max = o_iou
        idx_max = oc_idx
    if overlap_max > 0:
      # self.trackers에 있는 추적 아이디도 바꿔줘야함
      for s_trk in self.trackers:
        if s_trk.id+1 == trk[0][4]:
          s_trk.id = int(Occlusion_Pool[idx_max][0][4])-1
          break
      # OP에 있는 해당 가려진 인물의 추적 아이디 재할당
      trk[0][4] = Occlusion_Pool[idx_max][0][4]
      # OP로부터 재할당한 정보 삭제 및 제거
      occlusion_ids.remove(trk[0][4])
      del Occlusion_Pool[idx_max]
    return overlap_max

  def evaporation_overlapping(self, trk, result):
    """
    새로운 아이디가 기존 아이디와 같은지 아닌지 반환
    """
    global evaporation_ids, Evaporation_Pool
    # iou를 계산했을 때 가장 높은 순서대로 재할당
    # 이미 할당 되어 있는 tracker이면 다음 순서의 tracker 할당
    # 0만 남아 있으면 바로 0 반환
    overlaps = []
    for ev_idx, ev_trk in enumerate(Evaporation_Pool): # 새로운 추적 아이디 유지
      overlaps.append({'ep_idx': ev_idx, 'trk_id': ev_trk['trk_id'], 'iou': self.evap_iou(trk, ev_trk)})
    if len(overlaps) == 0: # EP에 아무것도 없다면 0 반환
      return 0
    else:
      overlaps_sorted = self.merge_sort(overlaps)
      for over_sorted in overlaps_sorted:
        if over_sorted['iou'] == 0: # 내림차순 정렬했는데도 0이면 바로 0 반환
          return 0
        else:
          already_assigned = False # 이미 할당되어 있다면 그 다음 순서로 할당
          for res_trk in result:
            if over_sorted['trk_id'] == res_trk[0][4]:
              already_assigned = True
              break
          if not already_assigned:
            # self.trackers에 있는 추적 아이디도 바꿔줘야함
            for s_trk in self.trackers:
              if s_trk.id+1 == trk[0][4]:
                s_trk.id = int(over_sorted['trk_id'])-1
                break
            # EP에 있는 사라진 인물의 추적 아이디 재할당
            trk[0][4] = over_sorted['trk_id']
            # EP로부터 재할당한 정보 삭제 및 제거
            evaporation_ids.remove(trk[0][4])
            del Evaporation_Pool[over_sorted['ep_idx']]
            return over_sorted['iou']

  def update(self, dets=np.empty((0, 5))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    # if self.frame_count == 5276:
    #   print(self.frame_count)
    if(len(ret)>0):
      # 0825 가려지거나 사라져서 재식별하는 문제를 해결하려고 OP,EP를 사용해서 재할당하도록 보완
      global total_ids, Total_Pool, occlusion_ids, Occlusion_Pool, evaporation_ids, Evaporation_Pool
      result = []
      success = []
      failure = []
      for trk in ret:
        result.append(trk) # 추적 성공
        if trk[0][4] in total_ids:
          success.append(trk[0][4])
      # ret 크기는 0 이고 total_ids 크기가 아직 있으면 추적 실패한 인물이 있음
      for id in total_ids:
        if id not in success:
          failure.append(id)
      if len(failure) > 0: # 추적 실패
        for fail in failure:
          # OP에 있으면 패쓰
          if fail not in occlusion_ids:
            if fail not in evaporation_ids:
              for trk in Total_Pool:
                if fail == trk['trk_id']:
                  evaporation_ids.append(fail)
                  Evaporation_Pool.append(trk) # OP에 없음, EP에 추가

      for idx, trk in enumerate(result): # 추적 성공, trk[0][4]가 추적 아이디임
        if trk[0][4] not in total_ids: # 새로운 인물?, 새로운 추적 아이디 유지
          # OP에 있는 겹친 BBox 현재 영역과 겹쳐진 BBox 과거 영역의 합집합 영역과 겹침?
          occlu_over = self.occlusion_overlapping(0, trk) # over 초기값은 무조건 0
          # EP에 있는 사라진 BBox 과거 영역과 겹침?
          evap_over = self.evaporation_overlapping(trk, result)
          if (occlu_over == 0) & (evap_over == 0): # 완전히 새로운 인물
            tracker = {'trk_id' : trk[0][4], 'left_top_x' : trk[0][0], 'left_top_y' : trk[0][1], 'right_bottom_x' : trk[0][2], 'right_bottom_y' : trk[0][3]}
            total_ids.append(trk[0][4])
            Total_Pool.append(tracker)
        else: # 기존 추적 아이디 유지
          for tracker in Total_Pool:
            if tracker['trk_id'] == trk[0][4]: # 새로운 정보로 업데이트
              tracker['left_top_x'] = trk[0][0]
              tracker['left_top_y'] = trk[0][1]
              tracker['right_bottom_x'] = trk[0][2]
              tracker['right_bottom_y'] = trk[0][3]
        # 다른 BBox와 겹침?
        over_lap = 0
        for over_idx, over_trk in enumerate(result):
          if over_idx != idx:
            over_lap = self.occlu_iou(trk, over_trk)
            if over_lap > 0: # 겹쳐진 인물의 추적 아이디를 OP에 삽입
              if trk[0][4] not in occlusion_ids:
                occlusion_ids.append(trk[0][4])
                Occlusion_Pool.append(trk)
              else:
                for occ_trk in Occlusion_Pool:
                  if trk[0][4] == occ_trk[0][4]:
                    occ_trk[0][0] = trk[0][0]
                    occ_trk[0][1] = trk[0][1]
                    occ_trk[0][2] = trk[0][2]
                    occ_trk[0][3] = trk[0][3]
        if over_lap == 0:
        # OP에 이미 삽입됨?
          if trk[0][4] in occlusion_ids:
            for idx, oc_trk in enumerate(Occlusion_Pool):
              if trk[0][4] == oc_trk[0][4]: # OP에서 삭제 및 제거
                occlusion_ids.remove(trk[0][4])
                del Occlusion_Pool[idx]
                break
        # EP에 이미 삽입됨?
        if (over_lap > 0) | ((over_lap == 0) & (trk[0][4] in occlusion_ids)):
          for idx, ev_trk in enumerate(Evaporation_Pool):
            if trk[0][4] == ev_trk['trk_id']: # EP에서 삭제 및 제거
              evaporation_ids.remove(trk[0][4])
              del Evaporation_Pool[idx]
              break
        # 다음 프레임으로 이동
      return np.concatenate(result)
    return np.empty((0,5))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    # parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='my_data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
  # all train
  args = parse_args()
  display = args.display
  phase = args.phase
  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(32, 3) #used only for display
  if(display):
    if not os.path.exists('mot_benchmark'):
      print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
      exit()
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect='equal')

  if not os.path.exists('output'):
    os.makedirs('output')
  pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')
  for seq_dets_fn in glob.glob(pattern):
    mot_tracker = Sort(max_age=args.max_age, 
                       min_hits=args.min_hits,
                       iou_threshold=args.iou_threshold) #create instance of the SORT tracker
    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
    seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]
    
    with open(os.path.join('output', '%s.txt'%(seq)),'w') as out_file:
      print("Processing %s."%(seq))
      for frame in range(int(seq_dets[:,0].max())):
        # frame += 1 #detection and frame numbers begin at 1
        dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        total_frames += 1

        if(display):
          # fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg'%(frame))
          fn = os.path.join('mot_benchmark', phase, seq, seq + '_000%09d_rendered.png'%(frame))
          im =io.imread(fn)
          ax1.imshow(im)
          plt.title(seq + ' Tracked Targets, frame: %d'%(frame))

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
          if(display):
            d = d.astype(np.int32)
            ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))

        if(display):
          fig.canvas.flush_events()
          plt.draw()
          ax1.cla()

  print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

  if(display):
    print("Note: to get real runtime results run without the option: --display")
