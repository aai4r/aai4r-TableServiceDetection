import pdb

import torch
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageDraw, ImageFont
import datasets.transforms as T
from util.box_ops import box_cxcywh_to_xyxy

mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
rev_normalize = T.Normalize((-mean / std), (1.0 / std))


def tensor_to_pil(tensor_image, orig_size=None):
    # cv_im_data = convert_tensor_to_image(im_data_chw_bgr, add_pixel_mean=True)
    tensor_image_tuple = rev_normalize(tensor_image)
    pil_image = to_pil_image(tensor_image_tuple[0])

    if orig_size is not None:
        pil_image = pil_image.resize(orig_size)

    return pil_image

def draw_bboxes_on_pil(pil_image, boxes, labels, scores=None, vis_th=0.5, no_bbox=False, box_color=(0, 0, 255)):
    # font = ImageFont.truetype('/usr/share/fonts/NanumGothic/NanumGothic-Regular.ttf', 20)

    num_boxes = boxes.size(0)
    draw = ImageDraw.Draw(pil_image)
    for i in range(num_boxes):
        if scores is not None:
            if scores[i] >= vis_th:
                if no_bbox is False:
                    draw.rectangle((int(boxes[i, 0]), int(boxes[i, 1]),
                                    int(boxes[i, 2]), int(boxes[i, 3])),
                                   outline=box_color, width=2)
                # draw.text((int(boxes[i, 0]), int(boxes[i, 1])), '%s_%.2f' % (labels[i], scores[i]),
                #           font=font)
                draw.text((int(boxes[i, 0]), int(boxes[i, 1])), '%s_%.2f' % (labels[i], scores[i]))

        else:
            if no_bbox is False:
                draw.rectangle((int(boxes[i, 0]), int(boxes[i, 1]),
                                int(boxes[i, 2]), int(boxes[i, 3])),
                               outline=box_color, width=2)

            # draw.text((int(boxes[i, 0]), int(boxes[i, 1])), str(int(labels[i])), font=font)
            draw.text((int(boxes[i, 0]), int(boxes[i, 1])), str(int(labels[i])))

    return pil_image

# g_list_colors = ['', 'green', 'blue', 'yellow', 'white', 'black']
# g_font = ImageFont.truetype('/usr/share/fonts/NanumGothic/NanumGothic-Regular.ttf', 20)
#
# def draw_bboxes_on_pil_amount(pil_image, boxes, labels, scores=None, amounts=None, vis_th=0.5):
#     num_boxes = boxes.size(0)
#     draw = ImageDraw.Draw(pil_image)
#
#     for i in range(num_boxes):
#         # set string
#         class_name = labels[i]
#         if scores is None:
#             str_result = '%s' % class_name
#         else:
#             score = scores[i]
#             if score >= vis_th:
#                 if amounts is None:
#                     str_result = '%s (%.1f)' % (class_name, score)
#                 else:
#                     amount_pred = amounts[i]
#                     str_result = '%s (%.1f) \nam: %.0f' % (class_name, score, amount_pred)
#
#         # draw string
#         draw.text
#
#         # draw bbox
# if class_name in ['dish', 'cup']:
#     draw.rectangle([xmin, ymin - 10,
#                     xmin + (xmax - xmin) * amount_pred_weighted, ymin - 5],
#                    fill=list_colors[area_id], width=2)
#
#             draw.rectangle((int(boxes[i, 0]), int(boxes[i, 1]),
#                             int(boxes[i, 2]), int(boxes[i, 3])),
#                            outline=(0, 0, 255), width=2)
#             draw.text((int(boxes[i, 0]), int(boxes[i, 1])), ,
#                       font=font)
#             draw.text((int(boxes[i, 0]), int(boxes[i, 1])), , font=font)
#
#     return pil_image




    #
    #     draw.multiline_text((xmin, ymin), str_result, fill=list_colors[area_id],
    #                         spacing=2)
    #
    #     # class_name = pps.class_names[label]
    #     draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=list_colors[area_id],
    #                    width=2)



def save_tensor_as_image(path_to_file, tensor_image, boxes, labels, scores=None, orig_size=None,
                         vis_th=0.5):
    # boxes: [n_boxes, 4] in (0, 1)
    # cv_im_data = convert_tensor_to_image(im_data_chw_bgr, add_pixel_mean=True)
    # tensor_image_tuple = rev_normalize(tensor_image)
    # pil_image = to_pil_image(tensor_image_tuple[0])
    pil_image = tensor_to_pil(tensor_image, orig_size)
    pil_image = draw_bboxes_on_pil(pil_image, boxes, labels, scores=scores, vis_th=vis_th)
    pil_image.save(path_to_file, 'JPEG')


def crop_masked_image(image, mask):
    mask = ~mask
    image_flatten = image.reshape(3, -1)
    mask_flatten = mask.reshape(-1)

    # masked_image_flatten = image_flatten * mask_flatten
    nonzero_index = mask_flatten.nonzero(as_tuple=True)[0]
    masked_image_flatten = image_flatten[:, nonzero_index]

    height = torch.sum(mask, dim=0).max()
    width = torch.sum(mask, dim=1).max()
    mask_image = torch.reshape(masked_image_flatten, (3, height, width))

    return mask_image


def convert_sample_and_boxes(draw_image, draw_mask, scaled_boxes=None):
    masked_image = crop_masked_image(draw_image, draw_mask)
    # masked_image = draw_image

    im_c, im_h, im_w = masked_image.shape
    target_sizes = torch.tensor([[im_h, im_w]])
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

    if scaled_boxes is not None:
        scaled_boxes = box_cxcywh_to_xyxy(scaled_boxes)
        unscaled_boxes = scaled_boxes * scale_fct

        return masked_image, unscaled_boxes

    return masked_image, None

import glob
import cv2
import os
def images_to_video(path_to_images, path_to_video, in_file_type = 'jpg', fps=15):
    list_files = glob.glob(os.path.join(path_to_images, f'*.{in_file_type}'))
    list_files = sorted(list_files)

    img_array = []
    for filename in list_files:
        img = cv2.imread(filename)
        h, w, c = img.shape
        size = (w, h)
        img_array.append(img)

    out = cv2.VideoWriter(path_to_video, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print(f'{path_to_video} is generated')


## time to second
import time
import datetime
def hms_to_sec(hour, min, sec, msec=0.0):
    return 3600*hour + 60*min + sec + 0.000001*msec

def get_data_from_string(str_date):
    year, month, day, hour, min, sec, msec = str_date.split('-')
    cur_day = f'{year}-{month}-{day}'
    cur_time = hms_to_sec(int(hour), int(min), int(sec), int(msec))

    return cur_day, cur_time

dict_start_time = {
    '2022-09-19': hms_to_sec(11, 42, 38.012746),
    '2022-09-21': hms_to_sec(11, 44, 07.009727),
    '2022-09-26': hms_to_sec(11, 40, 18.005954),
    '2022-09-28': hms_to_sec(11, 49, 42.019936),
    '2022-09-30': hms_to_sec(11, 41, 59.065387),
    '2022-10-05': hms_to_sec(11, 39, 39.056581),
    '2022-10-07': hms_to_sec(11, 45, 25.064612),
    '2022-10-12': hms_to_sec(11, 48, 38.198142),
    '2022-10-14': hms_to_sec(11, 46, 37.161493),

    '2023-01-18': hms_to_sec(11, 50, 07.181543),
    '2023-01-19': hms_to_sec(12, 16, 41.160705),
    '2023-01-26': hms_to_sec(12,  4, 31.184115),
    '2023-01-27': hms_to_sec(11, 58, 29.002064),
    '2023-01-30': hms_to_sec(12,  2, 12.033583),
    '2023-01-31': hms_to_sec(11, 58, 01.118151),
    '2023-02-01': hms_to_sec(12,  6, 46.185397),
    '2023-02-03': hms_to_sec(11, 56, 19.099524),
    '2023-02-06': hms_to_sec(11, 59,  3.284310),
    '2023-02-07': hms_to_sec(12,  5, 40.321468),
    '2023-02-09': hms_to_sec(11, 59, 10.230106),
    '2023-02-10': hms_to_sec(12,  6, 45.175896),
    '2023-02-14': hms_to_sec(11, 59, 49.195300),
    '2023-02-15': hms_to_sec(11, 55, 33.299030),
    '2023-02-16': hms_to_sec(12,  2, 22.222812)
}

def get_start_time_by_date(key_date):
    return dict_start_time[key_date]

def get_duration_using_startDate(str_date, start_date):
    start_day, start_secs = get_data_from_string(start_date)
    cur_day, cur_secs = get_data_from_string(str_date)

    assert start_day == cur_day

    duration_sec = cur_secs - start_secs
    duration_norm = get_duration_norm(duration_sec)

    return duration_norm

def get_duration(str_date):
    key, cur_time = get_data_from_string(str_date)
    duration_norm, _ = get_duration_key_time(key, cur_time)

    return duration_norm

def get_duration_key_time(key, cur_time):
    start_time = get_start_time_by_date(key)
    duration_sec = cur_time - start_time
    duration_norm = get_duration_norm(duration_sec)

    return duration_norm, duration_sec


def get_duration_norm(duration_seconds):
    duration_norm = duration_seconds / (60 * 30)
    if torch.is_tensor(duration_norm):
        duration_norm = duration_norm.clone().detach()
    else:
        duration_norm = torch.tensor(duration_norm)
    duration_norm = torch.clamp(duration_norm, 0.0, 1.0)

    return duration_norm