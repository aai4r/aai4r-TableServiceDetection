import pdb


class ServiceInfo:
    def __init__(self, service_name='none', threshold=0.5):
        self.service_name = service_name

        self.prob = 0.
        self.threshold = threshold
        self.current_decision = False
        self.current_secs = -1

        self.re_active_limit_secs = 60 * 5
        self.valid_time_provide_dessert = 60 * 15

        self.last_activated_secs = -self.re_active_limit_secs

    def set_prob(self, prob, secs):
        self.prob = prob
        self.current_secs = secs

        if self.prob >= self.threshold:
            self.current_decision = True
        else:
            self.current_decision = False

    def check_limits(self):
        if self.service_name == 'provide_dessert':
            if self.current_secs <= self.valid_time_provide_dessert:
                self.current_decision = False

        if self.current_secs - self.last_activated_secs <= self.re_active_limit_secs:
            self.current_decision = False

    def update(self):
        if self.current_decision:
            self.last_activated_secs = self.current_secs


class ServiceManager:
    def __init__(self, list_service_name=('no_service',
                                          'refill_food', 'found_trash',
                                          'provide_dessert', 'found_lost'),
                 start_time_in_sec=0., list_threshold=None):
        self.start_time_in_sec = start_time_in_sec

        self.sname_to_index = {service_name: idx for idx, service_name in enumerate(list_service_name)}
        self.list_services = []
        for ith, service_name in enumerate(list_service_name):
            if list_threshold is None:
                self.list_services.append(ServiceInfo(service_name))
            else:
                self.list_services.append(ServiceInfo(service_name, threshold=list_threshold[ith]))

    def set_start_time(self, start_time_in_sec=0.):
        self.start_time_in_sec = start_time_in_sec

    def process(self, pred_service_prob, current_time_in_sec):
        duration_time_in_sec = current_time_in_sec - self.start_time_in_sec
        # put and decide
        self.list_services[self.sname_to_index['no_service']].set_prob(1. - max(pred_service_prob), duration_time_in_sec)
        for ith, ith_service in enumerate(self.list_services[1:]):
            ith_service.set_prob(pred_service_prob[ith], duration_time_in_sec)  # set prob and get a decision

        # add manual decision
        # apply 'refill_food' prob by multiplying 'provide_dessert' prob
        # print('before: ', self.list_services[self.sname_to_index['refill_food']].prob)
        if duration_time_in_sec > self.list_services[self.sname_to_index['provide_dessert']].valid_time_provide_dessert:
            scale_for_refill = 1. - self.list_services[self.sname_to_index['provide_dessert']].prob
            rescaled_refill_food_prob = scale_for_refill * self.list_services[self.sname_to_index['refill_food']].prob
            self.list_services[self.sname_to_index['refill_food']].set_prob(rescaled_refill_food_prob,
                                                                            duration_time_in_sec)
        # print('scale: ', scale_for_refill)
        # print('after: ', self.list_services[self.sname_to_index['refill_food']].prob)

        # print('current status of service_manager')

        # check limits
        for ith in range(len(pred_service_prob)):
            self.list_services[ith+1].check_limits()

        # return only-one service with max prob
        ret_max_prob = -1
        ret_service = self.list_services[0].service_name
        ret_service_index = -1

        for ith, ith_service in enumerate(self.list_services[1:]):
            if ith_service.current_decision and ith_service.prob > ret_max_prob:
                ret_max_prob = ith_service.prob
                ret_service = ith_service.service_name
                ret_service_index = ith

        # remove other activations (T to F)
        for ith_service in self.list_services[1:]:
            if ith_service.service_name != ret_service:
                ith_service.current_decision = False

        # update
        for ith_service in self.list_services[1:]:
            ith_service.update()

        return ret_service_index, ret_service
