# 1. filtering and smoothing
# 2. tracking and accumulation
# 3. generate final result

import pdb
import numpy as np
from iouTracker.iou_tracker import track_iou_1by1
from iouTracker.viou_tracker import track_viou_1by1
import statistics

def det_to_trk(labels, scores, boxes, amounts_pred, amounts_pred_weighted, index_to_classname,
               cap_time=None):
    if cap_time is None:
        cap_time_sec = 0
    else:
        cap_time_sec = 3600 * cap_time[0] + 60 * cap_time[1] + cap_time[2]
    det_trk = []
    for det in zip(labels, scores, boxes, amounts_pred, amounts_pred_weighted):
        class_name = index_to_classname[det[0].cpu().item()]
        if class_name in ['food', 'drink']:   # food(1) and drink(2)
            trk = {
                # 'class_index': det[0].cpu().item(),
                # 'class': str(det[0].cpu().item()),
                'class': class_name,
                'score': det[1].cpu().item(),
                'bbox': det[2].cpu().numpy(),
                'amount_pred': det[3].cpu().numpy()[0],
                'amount_pred_w': det[4].cpu().item(),
                'cap_time': cap_time_sec
            }
            # print(det[3].cpu().numpy())
        else:
            trk = {
                # 'class_index': det[0].cpu().item(),
                # 'class': str(det[0].cpu().item()),
                'class': class_name,
                'score': det[1].cpu().item(),
                'bbox': det[2].cpu().numpy(),
                'amount_pred': 0., # np.array([0., 0., 0.], dtype=np.float32),
                'amount_pred_w': 0.,
                'cap_time': cap_time_sec
            }
            # print(det[3].cpu().numpy())
        det_trk.append(trk)

    return det_trk


class postprocessor():
    def __init__(self, class_names, target_class_list):
        super().__init__()
        # filtering
        self.class_names = [''] + class_names
        self.target_class_list = target_class_list

        # tracking
        self.sigma_l = 0        # low detection threshold
        self.sigma_h = 0.5      # high detection threshold
        self.sigma_iou = 0.5
        self.t_min = 15     # minimum track length (frame)
        self.ttl = 15       # time-to-live length (frame), possible live length
        self.tracker_type = 'MEDIANFLOW' # [BOOSTING, MIL, KCF, KCF2, TLD, MEDIANFLOW, GOTURN, NONE]
        self.keep_upper_height_ratio = 1.0

        self.tracks_active = []
        self.tracks_finished = []
        self.tracks_extendable = []
        self.frame_buffer = []
        self.frame_num = 1      # start frame number
        self.starting_track_id = 2     # 1 is reserved for dessert alarm

        # area
        self.points = []

        self.smoothing_amount_duration_frame = 15

        # alarm
        self.alarm_duration_sec = 10 # sec
        self.alarm_refill_amount = 0.15 # under this amount
        self.alarm_min_num_trk = 5
        # median is less than 0.15

        #
        self.progresses = {
            'probs': [],
            'cap_times': []
        }

        self.dict_class_names = {item: ith for ith, item in enumerate(self.class_names)}
        # self.target_class_index_list = [self.dict_class_names[t_name] for t_name in self.target_class_list]

    def set_point(self, x1, y1):
        self.points.append([x1, y1])

    def process(self, detections, frame=None):
        # 1. filtering (and smoothing)
        detections = self.filtering(detections=detections)

        # 2. tracking and accumulation
        tracks = self.tracking(detections=detections, frame=frame)

        # 3. generate final result (segmenting results)
        tracks = self.segment_and_sort(tracks=tracks)

        # 4. smoothing amout_pred
        tracks = self.smoothing(tracks=tracks)

        return tracks

    def segmenting(self, tracks):
        for trk in tracks:
            cx = sum(trk['bboxes'][-1][0::2]) / 2
            cy = sum(trk['bboxes'][-1][1::2]) / 2

            trk['area_id'] = 0

            for i_a, area in enumerate(self.areas, start=1):
                if cx > area[0] and cx < area[2] and cy > area[1] and cy < area[3]:
                    trk['area_id'] = i_a
                    break

        return tracks

    def most_frequent(self, l):
        return max(set(l), key=l.count)

    def segment_and_sort(self, tracks):
        # every point has the nearest dishes tracks
        for trk in tracks:
            cx = sum(trk['bboxes'][-1][0::2]) / 2
            cy = sum(trk['bboxes'][-1][1::2]) / 2

            trk['area_id'] = 0

            min_dist = 100000000.
            for i_p, pt in enumerate(self.points, start=1):
                ip_dist = pow(pow(pt[0] - cx, 2) + pow(pt[1] - cy, 2), 1/2)
                if ip_dist < min_dist:
                    trk['area_id'] = i_p
                    min_dist = ip_dist

        # every track has its rep. class name
        for trk in tracks:
            trk['rep_class'] = self.most_frequent(trk['classes'])

        # food/drink/empty_dish is matched with dish track
        # and give updating info to dish/cup track
        list_dishcup = []
        list_foodrink = []
        for trk in tracks:
            # if trk['classes_index'][-1] in [1, 2, 5]: # food(1), drink(2), empty_container(5)
            if trk['rep_class'] in ['food', 'drink']:
                list_foodrink.append(trk)
            # elif trk['classes_index'][-1] in [3, 4]:  # dish(3), cup(4)
            elif trk['rep_class'] in ['dish', 'cup']:
                list_dishcup.append(trk)

        for trk_dc in list_dishcup:
            find_food = False
            for trk_fd in list_foodrink:
                if self.check_a_in_b(trk_fd['bboxes'][-1], trk_dc['bboxes'][-1]) > 0.8:
                    self.update_dish_info(trk_dc, trk_fd)
                    find_food = True
                    break

            if find_food is True:
                self.update_dish_info(trk_dc, trk_fd)
            else:
                self.update_dish_info(trk_dc, None)

        return tracks

    def update_dish_info(self, track_dishcup, track_fooddrink):
        # update info (amount)
        if 'fooddrink_amount_pred' not in track_dishcup.keys():
            track_dishcup['fooddrink_amount_pred'] = []
            track_dishcup['fooddrink_amount_pred_w'] = []

        if track_fooddrink is None:
            track_dishcup['fooddrink_amount_pred'].append(0.)
            track_dishcup['fooddrink_amount_pred_w'].append(0.)
        else:
            track_dishcup['fooddrink_amount_pred'].append(track_fooddrink['amount_pred'][-1])
            track_dishcup['fooddrink_amount_pred_w'].append(track_fooddrink['amount_pred_w'][-1])


    def check_a_in_b(self, boxA, boxB):
        boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        # boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        boxO = [max(boxA[0], boxB[0]), max(boxA[1], boxB[1]),
                min(boxA[2], boxB[2]), min(boxA[3], boxB[3])]

        boxO_area = max(0, boxO[2] - boxO[0] + 1) * max(0, boxO[3] - boxO[1] + 1)

        return boxO_area / boxA_area

    def smoothing(self, tracks):
        for trk in tracks:
            if trk['rep_class'] in ['dish', 'cup']:
                trk['amount_final'] = np.array(trk['fooddrink_amount_pred'][-self.smoothing_amount_duration_frame:]).mean()
                trk['amount_w_final'] = np.array(trk['fooddrink_amount_pred_w'][-self.smoothing_amount_duration_frame:]).mean()
            else:
                trk['amount_final'] = trk['amount_pred'][-1]
                trk['amount_w_final'] = trk['amount_pred_w'][-1]

        return tracks

    def check_alarm(self, tracks, progress_prob, cap_time):
        # check time
        alarms = {
            'refill': [],
            'dessert': [],
            'lost': [],
            'trash': [],
        }

        cap_time_in_sec = cap_time[0] * 3600 + cap_time[1] * 60 + cap_time[2]
        self.progresses['probs'].append(progress_prob.cpu().numpy())
        self.progresses['cap_times'].append(cap_time_in_sec)

        # four service scenarios

        # # 4) provide dessert
        # # process is 0.0 0.0 1.0 is lasting for 10 seconds
        # lasting_time = self.progresses['cap_times'][-1] - np.array(self.progresses['cap_times'])
        # over_duration = lasting_time >= self.alarm_duration_sec
        # over_duration_t_index = [ith for ith, item in enumerate(over_duration) if item]
        # if len(over_duration_t_index) > 0:
        #     start_index = max(over_duration_t_index)
        #     end_index = len(self.progresses['cap_times'])
        #     num_progress_T = end_index - start_index + 1
        #
        #     progress_mat = self.progresses['probs'][start_index:end_index]
        #     progress_prob_mean = np.mean(progress_mat, axis=1)
        #
        #     if progress_prob_mean[2] > 0.9 and num_progress_T > self.alarm_min_num_trk \
        #             and max(lasting_time) > 10 * 60:
        #
        #         alarm_dessert = {
        #             'bboxes': [[0, 0, -1, -1]],
        #             'activate': True,
        #             'track_id': '1'
        #         }
        #         alarms['dessert'].append(alarm_dessert)
        #     # else:
        #     #     print('alarm dessert is not activate')
        #     #     print()

        for trk in tracks:
            lasting_time = trk['cap_times'][-1] - np.array(trk['cap_times'])
            over_duration = lasting_time >= self.alarm_duration_sec
            over_duration_t_index = [ith for ith, item in enumerate(over_duration) if item]

            if len(over_duration_t_index) > 0:
                start_index = max(over_duration_t_index)
                end_index = len(trk['cap_times'])
                num_trk_T = end_index - start_index + 1

                # 1) refill dishes
                if trk['rep_class'] in ['dish', 'cup'] and num_trk_T > self.alarm_min_num_trk:
                    try:
                        amount_pred_median = statistics.median(trk['fooddrink_amount_pred'][start_index:end_index])
                        amount_pred_last3 = statistics.mean(trk['fooddrink_amount_pred'][start_index:end_index][-3:])
                        # max_fooddrink_amount_pred = max(trk['fooddrink_amount_pred'])

                        print('{}: \n'.format(trk['rep_class']))
                        print('\tamount_pred_median: {} < {}'.format(amount_pred_median, self.alarm_refill_amount))
                        print('\tamount_pred_last3: {} < {}'.format(amount_pred_last3, self.alarm_refill_amount))
                        # print('\tmax_fooddrink_amount_pred: {} > 0.0'.format(max_fooddrink_amount_pred))

                        if amount_pred_median < self.alarm_refill_amount and \
                               amount_pred_last3 < self.alarm_refill_amount:
                                # max_fooddrink_amount_pred > 0.0:
                            trk['refill_cue'] = {
                                'amount_pred_median': amount_pred_median,
                                'amount_pred_last3': amount_pred_last3,
                                # 'max_fooddrink_amount_pred': max_fooddrink_amount_pred,
                                'start_index': start_index,
                                'end_index': end_index
                            }
                            alarms['refill'].append(trk)
                    except:
                        print('exception is raised')

                # # 2) clean trash
                # if trk['rep_class'] in ['trash'] and over_duration.any() and num_trk_T > self.alarm_min_num_trk:
                #     trk['trash_cue'] = {
                #         'over_duration': over_duration,
                #         'num_trk_T': num_trk_T,
                #         'alarm_min_num_trk': self.alarm_min_num_trk
                #     }
                #     alarms['trash'].append(trk)
                #
                # # 3) find lost belongings/things
                # if trk['rep_class'] in ['mobile_phone', 'wallet'] and over_duration.any() and \
                #         num_trk_T > self.alarm_min_num_trk:
                #     # TODO: yochin, check no person, check finish meals
                #     if progress_prob_mean[2] > 0.9:
                #         trk['lost_cue'] = {
                #             'over_duration': over_duration,
                #             'num_trk_T': num_trk_T,
                #             'progress_prob_mean': progress_prob_mean
                #         }
                #         alarms['lost'].append(trk)

        return alarms

    def tracking(self, detections, frame):
        # detections: list of detections_frame
        # detections_frame: list of det
        # det: dict of objects, score, bbox, class_name

        # convert form
        # to ['frame', 'id', 'x', 'y', 'w', 'h', 'score', 'class_id'] numpy
        if False:
            self.tracks_active, self.tracks_finished = track_iou_1by1(detections, self.frame_num,
                                                                      self.tracks_active, self.tracks_finished,
                                                                      self.sigma_h, self.sigma_l, self.sigma_iou, self.t_min, amount_pred=True)
        else:
            self.tracks_active, self.tracks_extendable, self.tracks_finished, \
            self.frame_buffer, self.starting_track_id = track_viou_1by1(detections, frame, self.frame_num,
                                                                       self.sigma_h, self.sigma_l,
                                                                       self.sigma_iou, self.t_min,
                                                                       self.ttl, self.tracker_type,
                                                                       self.keep_upper_height_ratio,
                                                                       self.tracks_active,
                                                                       self.tracks_extendable,
                                                                       self.tracks_finished,
                                                                       self.frame_buffer,
                                                                       current_track_id=self.starting_track_id,
                                                                       amount_pred=True)
        self.frame_num += 1

        return self.tracks_active

    def tracking_finish(self, tracks_active, tracks_finished, sigma_h, t_min):
        # finish all remaining active tracks
        tracks_finished += [track for track in tracks_active
                            if track['max_score'] >= sigma_h and len(track['bboxes']) >= t_min]

        return tracks_finished

    def filtering(self, detections):
        ret = []

        if len(self.target_class_list) == 0:
            for item in detections:
                ret.append(item)
        else:
            for item in detections:
                # if item['class_index'] in self.target_class_index_list:
                if item['class'] in self.target_class_list:
                    ret.append(item)

        return ret
