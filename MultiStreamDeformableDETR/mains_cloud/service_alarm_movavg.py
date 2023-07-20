import pdb
import numpy as np

class table_service_alarm_movavg:
    def __init__(self):

        self.classes = tuple(['dish', 'food', 'cup', 'drink'])
        self.num_dets = {item: [] for item in self.classes}
        self.amount_th = 0.15   # if amount is less than amount_th, consider not-detected (empty)
        self.prev_len = 5
        self.cur_len = 5
        self.over_len = 5
        self.max_len = self.prev_len + self.cur_len + self.over_len
        self.var_th = 0.7

        self.list_prev_avg = {item: [] for item in self.classes}
        self.list_cur_avg = {item: [] for item in self.classes}

    def process(self, labels, scores, boxes, amount_pred_weighted, class_index_to_name):
        # put default number, zero
        for class_name in self.classes:
            self.num_dets[class_name].append(0)

        # put the result
        for index, item_amt in zip(labels, amount_pred_weighted):
            class_name = class_index_to_name[index-1]

            if class_name in self.classes:
                if class_name in ['food', 'drink']:
                    if item_amt > self.amount_th:
                        self.num_dets[class_name][-1] += 1
                else:
                    self.num_dets[class_name][-1] += 1

        # pop if the lengh is over than max_len
        for class_name in self.classes:
            if len(self.num_dets[class_name]) > self.max_len:
                self.num_dets[class_name].pop(0)

        # get avg values
        prev_avg = {item: 0 for item in self.classes}
        cur_avg = {item: 0 for item in self.classes}

        prev_b = -(self.prev_len + self.over_len + self.cur_len)
        prev_e = -(self.cur_len)
        cur_b = -(self.over_len + self.cur_len)

        for class_name in self.classes:
            if len(self.num_dets[class_name]) >= self.max_len:
                prev_avg[class_name] = np.mean(np.array(self.num_dets[class_name][prev_b:prev_e]))
                cur_avg[class_name] = np.mean(np.array(self.num_dets[class_name][cur_b:]))

                print(class_name, self.num_dets[class_name])
                print('\tprev:', prev_avg[class_name])
                print('\tcur:', cur_avg[class_name])

                self.list_prev_avg[class_name].append(prev_avg[class_name])
                self.list_cur_avg[class_name].append(cur_avg[class_name])

                if len(self.list_prev_avg[class_name]) > self.max_len:
                    self.list_prev_avg[class_name].pop(0)
                if len(self.list_cur_avg[class_name]) > self.max_len:
                    self.list_cur_avg[class_name].pop(0)

        # final decision
        # service_results[0]: refill
        service_results = [0.] * 4  # reset
        # repr_service_index = -1
        # repr_service_name = 'none'

        # if (prev_avg['food'] - cur_avg['food']) > self.var_th > abs(cur_avg['dish'] - prev_avg['dish']):
        #     service_results[0] = 1.0

        for contents, container in zip(['food', 'drink'], ['dish', 'cup']):
            num_empty_food_cur = cur_avg[container] - cur_avg[contents]
            num_empty_food_prev = prev_avg[container] - prev_avg[contents]
            print(f'{contents}(prev, cur): {num_empty_food_prev}, {num_empty_food_cur}')
            if num_empty_food_cur > self.var_th and \
                abs(num_empty_food_cur) - abs(num_empty_food_prev) > self.var_th:
                service_results[0] = 1.0

        # return service_results, repr_service_index, repr_service_name
        return service_results
