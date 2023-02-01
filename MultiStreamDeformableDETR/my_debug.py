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
    font = ImageFont.truetype('/usr/share/fonts/NanumGothic/NanumGothic-Regular.ttf', 20)

    num_boxes = boxes.size(0)
    draw = ImageDraw.Draw(pil_image)
    for i in range(num_boxes):
        if scores is not None:
            if scores[i] >= vis_th:
                if no_bbox is False:
                    draw.rectangle((int(boxes[i, 0]), int(boxes[i, 1]),
                                    int(boxes[i, 2]), int(boxes[i, 3])),
                                   outline=box_color, width=2)
                draw.text((int(boxes[i, 0]), int(boxes[i, 1])), '%s_%.2f' % (labels[i], scores[i]),
                          font=font)
        else:
            if no_bbox is False:
                draw.rectangle((int(boxes[i, 0]), int(boxes[i, 1]),
                                int(boxes[i, 2]), int(boxes[i, 3])),
                               outline=box_color, width=2)

            draw.text((int(boxes[i, 0]), int(boxes[i, 1])), str(int(labels[i])), font=font)

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
