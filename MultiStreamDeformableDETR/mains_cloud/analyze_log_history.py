import os
import json
import matplotlib.pyplot as plt
import pdb

def find_earlyStop_acc_checkpoint(output_dir, log_filename, list_keys, x_key='epoch', wait_epoch=2):
    print('find_earlyStop_acc_checkpoint: ', os.getpid())
    logs = []
    for line in open(os.path.join(output_dir, log_filename), "r"):
        logs.append(json.loads(line))

    ongoing_update_es_epooch = True

    list_values = {key: {} for key in list_keys}

    earlyStop_epoch = -1
    max_acc_over_all_epochs = -1
    list_acc_total = {}
    for log_at_epoch in logs:
        list_acc_at_epoch = []

        # to get the averaged accuracy
        epoch_now = log_at_epoch[x_key]
        for key in list_keys:
            if key in log_at_epoch.keys():
                value = log_at_epoch[key]

                if isinstance(value, list):
                    value = sum(value) / len(value)

                list_values[key][epoch_now] = value
                list_acc_at_epoch.append(value)
            # else:
            #     print(f'{key} is not in log_at_epoch')

        # list_value[key][epoch_now]
        # list_acc_at_epoch = [value_key1, value_key2, ...]

        if len(list_acc_at_epoch) > 0:  # to avoid training log
            list_acc_total[epoch_now] = sum(list_acc_at_epoch) / len(list_acc_at_epoch)
            # print('list_acc_at_epoch:', list_acc_at_epoch)
            # print('list_acc_total:', list_acc_total)

            if epoch_now <= (earlyStop_epoch + wait_epoch):
                ongoing_update_es_epooch = True
                # print('0.max_acc_over_all_epochs: ', max_acc_over_all_epochs)
                if list_acc_total[epoch_now] > max_acc_over_all_epochs:
                    earlyStop_epoch = epoch_now
                    # print('1.max_acc_over_all_epochs: ', max_acc_over_all_epochs)
                    # print(list_acc_total)
                    max_acc_over_all_epochs = list_acc_total[epoch_now]
                    # print('2.max_acc_over_all_epochs: ', max_acc_over_all_epochs)
            else:
                ongoing_update_es_epooch = False
                # print(f'skip epoch {epoch_now}, larger than es {earlyStop_epoch} + wait_epoch {wait_epoch}')

    print(list_acc_total)
    print(f'earlyStop epoch: {earlyStop_epoch}, {max_acc_over_all_epochs}')
    print(f'max acc: ', max(list_acc_total.values()))

    for key in list_keys:
        if earlyStop_epoch in list_values[key].keys():
            print(f'earlyStop epoch: {earlyStop_epoch}, {key}, {list_values[key][earlyStop_epoch]}')
        else:
            print(f'key {earlyStop_epoch} is not in {list_values[key].keys()}')

    print(f'\n -- ongoing_update_es_epooch: {ongoing_update_es_epooch} till epoch {(earlyStop_epoch + wait_epoch)}--')

    return earlyStop_epoch, ongoing_update_es_epooch


def log_analyzer(output_dir, log_filename, list_keys, x_key='epoch', es_epoch=None):
    logs = []
    for line in open(os.path.join(output_dir, log_filename), "r"):
        logs.append(json.loads(line))

    list_values = {key: {} for key in list_keys}
    # list_epoch = {key: [] for key in list_keys}

    for key in list_keys:
        max_acc = 0
        max_acc_list = []
        epoch_max = -1
        es_acc = 0
        es_acc_list = []

        value = 0
        epoch_now = -1
        value_list = []

        for log_at_epoch in logs:
            if key in log_at_epoch.keys():
                if x_key is None:
                    epoch_now = 4510
                else:
                    epoch_now = log_at_epoch[x_key]
                value_list = log_at_epoch[key]

                if isinstance(value_list, list):
                    value = sum(value_list) / len(value_list)
                else:
                    value = value_list

                list_values[key][epoch_now] = value

                if value > max_acc:
                    max_acc = value
                    max_acc_list = value_list
                    epoch_max = epoch_now

                if es_epoch is not None and es_epoch == epoch_now:
                    es_acc = value
                    es_acc_list = value_list

                # print(f'{log_at_epoch[x_key]}: {value}')
        print(f'{key}')
        print(f'\tlast of {key} : {round(value, 4)} @ {epoch_now} := {value_list}')
        print(f'\tmax of {key} : {round(max_acc, 4)} @ {epoch_max} := {max_acc_list}')
        if es_epoch is not None:
            print(f'\tearly stop of {key} : {round(es_acc, 4)} @ {es_epoch} := {es_acc_list}')

    for key, values in list_values.items():
        plt.plot(values.keys(), values.values(), label=key)

    plt.xlabel(f'x - {x_key}')
    plt.ylabel(f'y - losses')
    # plt.title(project_name[:21] + '\n' + project_name[21:])
    # plt.title(project_name)
    plt.legend()
    # plt.show()


if __name__ == '__main__':
    path_to_base = '/home/yochin/Desktop/TableServiceDetector/MultiStreamDeformableDETR/mains_cloud/exps'

    # # through training
    # # project_name = 'ETRIGJHall/Imageavgp_Encv2avgp_pca1dlcnmsSoftmaxAttnSimple_frzDETR_wDet21c_v4'
    # project_name = 'ETRIGJHall_Seq/T10avgp_imageavgp_encv2avgp_pca1dlcnmsSoftmaxAttnSimple_frzDETR_wDet21c_v3_allTraining_EarlyStop_MultiGPUs'
    # log_file = 'log_per_epoch_backup.txt'
    #
    # # project_name = 'ETRIGJHall_Seq/T5attnsimple_imageavgp_pca1dlcnmsSoftmaxAttnSimple_frzDETR_wDet21c_v3'
    # # log_file = 'log_per_epoch.txt'
    # # log_file = 'log.txt'
    #
    # path_to_output = os.path.join(path_to_base, project_name)
    #
    # list_keys = ['test_sacs_acc', 'test_sacs_f1score', 'test_sacs_auc', 'test_sacs_prauc']   #
    #
    # es_epoch = find_earlyStop_acc_checkpoint(path_to_output, log_file, list_keys, wait_epoch=20)
    # log_analyzer(path_to_output, log_file, list_keys, es_epoch=es_epoch)

    # through testing
    project_name = 'ETRIGJHallDeploy/T10avgp_imageavgp_encv2avgp_pca1dlcnmsSoftmaxAttnSimple_frzDETR_wDet21c_v3/list_captured1/checkpoint0029'
    log_file = 'eval.txt'

    path_to_output = os.path.join(path_to_base, project_name)

    list_keys = ['test_sacs_acc', 'test_sacs_f1score', 'test_sacs_auc', 'test_sacs_prauc']
    log_analyzer(path_to_output, log_file, list_keys, x_key=None, es_epoch=None)
