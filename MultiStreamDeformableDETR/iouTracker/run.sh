python demo.py -d data/mot17/train/MOT17-04-SDP/det/det.txt -o res/iou-tracker/MOT17-04-SDP-1by1.txt
#demo.py -f '/path/to/VisDrone2018-MOT-val/sequences/uav0000137_00458_v/{:07d}.jpg' -d /path/to/VisDrone2018-MOT-val/detections/uav0000137_00458_v.txt -o results/VisDrone2018-MOT-val/uav0000137_00458_v.txt -v MEDIANFLOW -sl 0.9 -sh 0.98 -si 0.1 -tm 23 --ttl 8 --nms 0.6 -fmt visdrone

#$ ./demo.py -h
#usage: demo.py [-h] [-v VISUAL] [-hr KEEP_UPPER_HEIGHT_RATIO] [-f FRAMES_PATH]
#               -d DETECTION_PATH -o OUTPUT_PATH [-sl SIGMA_L] [-sh SIGMA_H]
#               [-si SIGMA_IOU] [-tm T_MIN] [-ttl TTL] [-nms NMS] [-fmt FORMAT]
#
#IOU/V-IOU Tracker demo script
#
#optional arguments:
#  -h, --help            show this help message and exit
#  -v VISUAL, --visual VISUAL
#                        visual tracker for V-IOU. Currently supported are
#                        [BOOSTING, MIL, KCF, KCF2, TLD, MEDIANFLOW, GOTURN,
#                        NONE] see README.md for furthert details
#  -hr KEEP_UPPER_HEIGHT_RATIO, --keep_upper_height_ratio KEEP_UPPER_HEIGHT_RATIO
#                        Ratio of height of the object to track to the total
#                        height of the object for visual tracking. e.g. upper
#                        30%
#  -f FRAMES_PATH, --frames_path FRAMES_PATH
#                        sequence frames with format
#                        '/path/to/frames/frame_{:04d}.jpg' where '{:04d}' will
#                        be replaced with the frame id. (zero_padded to 4
#                        digits, use {:05d} for 5 etc.)
#  -d DETECTION_PATH, --detection_path DETECTION_PATH
#                        full path to CSV file containing the detections
#  -o OUTPUT_PATH, --output_path OUTPUT_PATH
#                        output path to store the tracking results (MOT
#                        challenge/Visdrone devkit compatible format)
#  -sl SIGMA_L, --sigma_l SIGMA_L
#                        low detection threshold
#  -sh SIGMA_H, --sigma_h SIGMA_H
#                        high detection threshold
#  -si SIGMA_IOU, --sigma_iou SIGMA_IOU
#                        intersection-over-union threshold
#  -tm T_MIN, --t_min T_MIN
#                        minimum track length
#  -ttl TTL, --ttl TTL   time to live parameter for v-iou
#  -nms NMS, --nms NMS   nms for loading multi-class detections
#  -fmt FORMAT, --format FORMAT
#                        format of the detections [motchallenge, visdrone]