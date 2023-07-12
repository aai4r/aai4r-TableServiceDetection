# ---------------------------------------------------------
# IOU Tracker
# Copyright (c) 2019 TU Berlin, Communication Systems Group
# Licensed under The MIT License [see LICENSE for details]
# Written by Erik Bochinski
# ---------------------------------------------------------
import this

import cv2
import numpy as np
from lapsolver import solve_dense
from tqdm import tqdm
from time import time
import pdb

try:
    from util import iou, load_mot
    from vis_tracker import VisTracker
except:
    from iouTracker.util import iou, load_mot
    from iouTracker.vis_tracker import VisTracker

def track_viou_alternative(frames_path, detections, sigma_l, sigma_h, sigma_iou, t_min, ttl, tracker_type, keep_upper_height_ratio):
    if tracker_type == 'NONE':
        assert ttl == 1, "ttl should not be larger than 1 if no visual tracker is selected"

    tracks_active = []
    tracks_extendable = []
    tracks_finished = []
    frame_buffer = []

    for frame_num, detections_frame in enumerate(tqdm(detections), start=1):
        # load frame and put into buffer
        frame_path = frames_path.format(frame_num)
        frame = cv2.imread(frame_path)
        assert frame is not None, "could not read '{}'".format(frame_path)

        track_viou_1by1()


    # finish all remaining active and extendable tracks
    tracks_finished = tracks_finished + \
                      [track for track in tracks_active + tracks_extendable
                       if track['max_score'] >= sigma_h and track['det_counter'] >= t_min]

    # remove last visually tracked frames and compute the track classes
    for track in tracks_finished:
        if ttl != track['ttl']:
            track['bboxes'] = track['bboxes'][:-(ttl - track['ttl'])]
        track['class'] = max(set(track['classes']), key=track['classes'].count)

        del track['visual_tracker']

    return tracks_finished

def track_viou_1by1(detections_frame, frame, frame_num,
                    sigma_h, sigma_l, sigma_iou, t_min, ttl, tracker_type, keep_upper_height_ratio,
                    tracks_active, tracks_extendable, tracks_finished, frame_buffer, current_track_id,
                    amount_pred=False):
    if tracker_type == 'NONE':
        assert ttl == 1, "ttl should not be larger than 1 if no visual tracker is selected"

    frame_buffer.append(frame)
    if len(frame_buffer) > ttl + 1:
        frame_buffer.pop(0)

    # apply low threshold to detections
    dets = [det for det in detections_frame if det['score'] >= sigma_l]

    track_ids, det_ids = associate(tracks_active, dets, sigma_iou)
    updated_tracks = []
    for track_id, det_id in zip(track_ids, det_ids):
        tracks_active[track_id]['bboxes'].append(dets[det_id]['bbox'])
        tracks_active[track_id]['max_score'] = max(tracks_active[track_id]['max_score'],
                                                   dets[det_id]['score'])
        tracks_active[track_id]['classes'].append(dets[det_id]['class'])
        tracks_active[track_id]['det_counter'] += 1
        tracks_active[track_id]['cap_times'].append(dets[det_id]['cap_time'])

        if amount_pred:
            tracks_active[track_id]['amount_pred'].append(dets[det_id]['amount_pred'])
            tracks_active[track_id]['amount_pred_w'].append(dets[det_id]['amount_pred_w'])
            # tracks_active[track_id]['classes_index'].append(dets[det_id]['class_index'])

        if tracks_active[track_id]['ttl'] != ttl:
            # reset visual tracker if active
            tracks_active[track_id]['ttl'] = ttl
            tracks_active[track_id]['visual_tracker'] = None

        updated_tracks.append(tracks_active[track_id])

    tracks_not_updated = [tracks_active[idx] for idx in
                          set(range(len(tracks_active))).difference(set(track_ids))]

    for track in tracks_not_updated:
        if track['ttl'] > 0:
            if track['ttl'] == ttl:
                # init visual tracker
                track['visual_tracker'] = VisTracker(tracker_type, track['bboxes'][-1],
                                                     frame_buffer[-2],
                                                     keep_upper_height_ratio)
            # viou forward update
            ok, bbox = track['visual_tracker'].update(frame)

            if not ok:
                # visual update failed, track can still be extended
                tracks_extendable.append(track)
                continue

            track['ttl'] -= 1
            track['bboxes'].append(bbox)
            updated_tracks.append(track)
        else:
            tracks_extendable.append(track)

    # update the list of extendable tracks. tracks that are too old are moved to the finished_tracks. this should
    # not be necessary but may improve the performance for large numbers of tracks (eg. for mot19)
    tracks_extendable_updated = []
    for track in tracks_extendable:
        if track['start_frame'] + len(track['bboxes']) + ttl - track['ttl'] >= frame_num:
            tracks_extendable_updated.append(track)
        elif track['max_score'] >= sigma_h and track['det_counter'] >= t_min:
            tracks_finished.append(track)
    tracks_extendable = tracks_extendable_updated

    new_dets = [dets[idx] for idx in set(range(len(dets))).difference(set(det_ids))]
    dets_for_new = []

    for det in new_dets:
        finished = False
        # go backwards and track visually
        boxes = []
        vis_tracker = VisTracker(tracker_type, det['bbox'], frame, keep_upper_height_ratio)

        for f in reversed(frame_buffer[:-1]):
            ok, bbox = vis_tracker.update(f)
            if not ok:
                # can not go further back as the visual tracker failed
                break
            boxes.append(bbox)

            # sorting is not really necessary but helps to avoid different behaviour for different orderings
            # preferring longer tracks for extension seems intuitive, LAP solving might be better
            for track in sorted(tracks_extendable, key=lambda x: len(x['bboxes']),
                                reverse=True):

                offset = track['start_frame'] + len(track['bboxes']) + len(boxes) - frame_num
                # association not optimal (LAP solving might be better)
                # association is performed at the same frame, not adjacent ones
                if 1 <= offset <= ttl - track['ttl'] and iou(track['bboxes'][-offset],
                                                             bbox) >= sigma_iou:
                    if offset > 1:
                        # remove existing visually tracked boxes behind the matching frame
                        track['bboxes'] = track['bboxes'][:-offset + 1]
                    track['bboxes'] += list(reversed(boxes))[1:]
                    track['bboxes'].append(tuple(det['bbox']))
                    track['max_score'] = max(track['max_score'], det['score'])
                    track['classes'].append(det['class'])
                    track['ttl'] = ttl
                    track['visual_tracker'] = None
                    track['cap_times'].append(det['cap_time'])

                    if amount_pred:
                        # track['amount_pred'].append(tuple(det['amount_pred']))
                        track['amount_pred'].append(det['amount_pred'])
                        track['amount_pred_w'].append(det['amount_pred_w'])
                        # track['classes_index'].append(det['class_index'])

                    tracks_extendable.remove(track) # list.remove and del[idx] is same. del can remove multiple items

                    try:
                        if track in tracks_finished:    # dict is managed as ref. check id(variable)
                            del tracks_finished[tracks_finished.index(track)]
                    except:
                        pass

                    updated_tracks.append(track)

                    finished = True
                    break
            if finished:
                break
        if not finished:
            dets_for_new.append(det)

    # create new tracks
    if amount_pred:
        new_tracks = [{'bboxes': [tuple(det['bbox'])], 'max_score': det['score'], 'start_frame': frame_num,
                       'track_id': current_track_id,
                       'ttl': ttl,
                       'amount_pred': [det['amount_pred']], #[tuple(det['amount_pred'])],
                       'amount_pred_w': [det['amount_pred_w']],
                       'cap_times': [det['cap_time']],
                       # 'classes_index': [det['class_index']],
                       'classes': [det['class']], 'det_counter': 1, 'visual_tracker': None} for det
                      in dets_for_new]
    else:
        new_tracks = [{'bboxes': [tuple(det['bbox'])], 'max_score': det['score'], 'start_frame': frame_num,
                       'track_id': current_track_id,
                       'ttl': ttl,
                       'cap_times': [det['cap_time']],
                       'classes': [det['class']], 'det_counter': 1, 'visual_tracker': None} for det
                      in dets_for_new]
    current_track_id += 1
    tracks_active = []
    for track in updated_tracks + new_tracks:
        if track['ttl'] == 0:
            tracks_extendable.append(track)
        else:
            tracks_active.append(track)

    return tracks_active, tracks_extendable, tracks_finished, frame_buffer, current_track_id


def track_viou(frames_path, detections, sigma_l, sigma_h, sigma_iou, t_min, ttl, tracker_type, keep_upper_height_ratio):
    """ V-IOU Tracker.
    See "Extending IOU Based Multi-Object Tracking by Visual Information by E. Bochinski, T. Senst, T. Sikora" for
    more information.

    Args:
         frames_path (str): path to ALL frames.
                            string must contain a placeholder like {:07d} to be replaced with the frame numbers.
         detections (list): list of detections per frame, usually generated by util.load_mot
         sigma_l (float): low detection threshold.
         sigma_h (float): high detection threshold.
         sigma_iou (float): IOU threshold.
         t_min (float): minimum track length in frames.
         ttl (float): maximum number of frames to perform visual tracking.
                      this can fill 'gaps' of up to 2*ttl frames (ttl times forward and backward).
         tracker_type (str): name of the visual tracker to use. see VisTracker for more details.
         keep_upper_height_ratio (float): float between 0.0 and 1.0 that determines the ratio of height of the object
                                          to track to the total height of the object used for visual tracking.

    Returns:
        list: list of tracks.
    """
    if tracker_type == 'NONE':
        assert ttl == 1, "ttl should not be larger than 1 if no visual tracker is selected"

    tracks_active = []
    tracks_extendable = []
    tracks_finished = []
    frame_buffer = []

    for frame_num, detections_frame in enumerate(tqdm(detections), start=1):
        # load frame and put into buffer
        frame_path = frames_path.format(frame_num)
        frame = cv2.imread(frame_path)
        assert frame is not None, "could not read '{}'".format(frame_path)
        frame_buffer.append(frame)
        if len(frame_buffer) > ttl + 1:
            frame_buffer.pop(0)

        # apply low threshold to detections
        dets = [det for det in detections_frame if det['score'] >= sigma_l]

        track_ids, det_ids = associate(tracks_active, dets, sigma_iou)
        updated_tracks = []
        for track_id, det_id in zip(track_ids, det_ids):
            tracks_active[track_id]['bboxes'].append(dets[det_id]['bbox'])
            tracks_active[track_id]['max_score'] = max(tracks_active[track_id]['max_score'], dets[det_id]['score'])
            tracks_active[track_id]['classes'].append(dets[det_id]['class'])
            tracks_active[track_id]['det_counter'] += 1

            if tracks_active[track_id]['ttl'] != ttl:
                # reset visual tracker if active
                tracks_active[track_id]['ttl'] = ttl
                tracks_active[track_id]['visual_tracker'] = None

            updated_tracks.append(tracks_active[track_id])

        tracks_not_updated = [tracks_active[idx] for idx in set(range(len(tracks_active))).difference(set(track_ids))]

        for track in tracks_not_updated:
            if track['ttl'] > 0:
                if track['ttl'] == ttl:
                    # init visual tracker
                    track['visual_tracker'] = VisTracker(tracker_type, track['bboxes'][-1], frame_buffer[-2],
                                                         keep_upper_height_ratio)
                # viou forward update
                ok, bbox = track['visual_tracker'].update(frame)

                if not ok:
                    # visual update failed, track can still be extended
                    tracks_extendable.append(track)
                    continue

                track['ttl'] -= 1
                track['bboxes'].append(bbox)
                updated_tracks.append(track)
            else:
                tracks_extendable.append(track)

        # update the list of extendable tracks. tracks that are too old are moved to the finished_tracks. this should
        # not be necessary but may improve the performance for large numbers of tracks (eg. for mot19)
        tracks_extendable_updated = []
        for track in tracks_extendable:
            if track['start_frame'] + len(track['bboxes']) + ttl - track['ttl'] >= frame_num:
                tracks_extendable_updated.append(track)
            elif track['max_score'] >= sigma_h and track['det_counter'] >= t_min:
                tracks_finished.append(track)
        tracks_extendable = tracks_extendable_updated

        new_dets = [dets[idx] for idx in set(range(len(dets))).difference(set(det_ids))]
        dets_for_new = []

        for det in new_dets:
            finished = False
            # go backwards and track visually
            boxes = []
            vis_tracker = VisTracker(tracker_type, det['bbox'], frame, keep_upper_height_ratio)

            for f in reversed(frame_buffer[:-1]):
                ok, bbox = vis_tracker.update(f)
                if not ok:
                    # can not go further back as the visual tracker failed
                    break
                boxes.append(bbox)

                # sorting is not really necessary but helps to avoid different behaviour for different orderings
                # preferring longer tracks for extension seems intuitive, LAP solving might be better
                for track in sorted(tracks_extendable, key=lambda x: len(x['bboxes']), reverse=True):

                    offset = track['start_frame'] + len(track['bboxes']) + len(boxes) - frame_num
                    # association not optimal (LAP solving might be better)
                    # association is performed at the same frame, not adjacent ones
                    if 1 <= offset <= ttl - track['ttl'] and iou(track['bboxes'][-offset], bbox) >= sigma_iou:
                        if offset > 1:
                            # remove existing visually tracked boxes behind the matching frame
                            track['bboxes'] = track['bboxes'][:-offset+1]
                        track['bboxes'] += list(reversed(boxes))[1:]
                        track['bboxes'].append(det['bbox'])
                        track['max_score'] = max(track['max_score'], det['score'])
                        track['classes'].append(det['class'])
                        track['ttl'] = ttl
                        track['visual_tracker'] = None

                        tracks_extendable.remove(track)
                        if track in tracks_finished:
                            del tracks_finished[tracks_finished.index(track)]
                        updated_tracks.append(track)

                        finished = True
                        break
                if finished:
                    break
            if not finished:
                dets_for_new.append(det)

        # create new tracks
        new_tracks = [{'bboxes': [det['bbox']], 'max_score': det['score'], 'start_frame': frame_num, 'ttl': ttl,
                       'classes': [det['class']], 'det_counter': 1, 'visual_tracker': None} for det in dets_for_new]
        tracks_active = []
        for track in updated_tracks + new_tracks:
            if track['ttl'] == 0:
                tracks_extendable.append(track)
            else:
                tracks_active.append(track)

    # finish all remaining active and extendable tracks
    tracks_finished = tracks_finished + \
                      [track for track in tracks_active + tracks_extendable
                       if track['max_score'] >= sigma_h and track['det_counter'] >= t_min]

    # remove last visually tracked frames and compute the track classes
    for track in tracks_finished:
        if ttl != track['ttl']:
            track['bboxes'] = track['bboxes'][:-(ttl - track['ttl'])]
        track['class'] = max(set(track['classes']), key=track['classes'].count)

        del track['visual_tracker']

    return tracks_finished


def associate(tracks, detections, sigma_iou):
    """ perform association between tracks and detections in a frame.
    Args:
        tracks (list): input tracks
        detections (list): input detections
        sigma_iou (float): minimum intersection-over-union of a valid association

    Returns:
        (tuple): tuple containing:

        track_ids (numpy.array): 1D array with indexes of the tracks
        det_ids (numpy.array): 1D array of the associated indexes of the detections
    """
    costs = np.empty(shape=(len(tracks), len(detections)), dtype=np.float32)
    for row, track in enumerate(tracks):
        for col, detection in enumerate(detections):
            costs[row, col] = 1 - iou(track['bboxes'][-1], detection['bbox'])

    np.nan_to_num(costs)
    costs[costs > 1 - sigma_iou] = np.nan
    track_ids, det_ids = solve_dense(costs)
    return track_ids, det_ids


def track_viou_matlab_wrapper(frames_path, detections, sigma_l, sigma_h, sigma_iou, t_min, ttl, tracker_type,
                              keep_upper_height_ratio=1.):
    """
    Matlab wrapper of the v-iou tracker for the detrac evaluation toolkit.

    Args:
         detections (numpy.array): numpy array of detections, usually supplied by run_tracker.m
         sigma_l (float): low detection threshold.
         sigma_h (float): high detection threshold.
         sigma_iou (float): IOU threshold.
         t_min (float): minimum track length in frames.

    Returns:
        float: speed in frames per second.
        list: list of tracks.
    """

    detections = detections.reshape((7, -1)).transpose()
    dets = load_mot(detections, with_classes=False)
    start = time()
    tracks = track_viou(frames_path+"img{:05d}.jpg", dets, sigma_l, sigma_h, sigma_iou, int(t_min), int(ttl), tracker_type, keep_upper_height_ratio)
    end = time()

    id_ = 1
    out = []
    for track in tracks:
        for i, bbox in enumerate(track['bboxes']):
            out += [float(bbox[0]), float(bbox[1]), float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1]),
                    float(track['start_frame'] + i), float(id_)]
        id_ += 1

    num_frames = len(dets)
    speed = num_frames / (end - start)

    return speed, out
