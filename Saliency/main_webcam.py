# -------------------------------------------------------------------------------
# Name:        main_webcam
# Purpose:     Testing the package pySaliencyMap with your own webcams
#
# Author:      Akisato Kimura <akisato@ieee.org>
#
# Created:     May 14, 2016
# Copyright:   (c) Akisato Kimura 2016-
# Licence:     All rights reserved
# -------------------------------------------------------------------------------

import cv2
import sys
from Saliency import pySaliencyMap

def saliency(Prog, frame, prev_frame):
    frame_size = frame.shape
    frame_width = frame_size[1]
    frame_height = frame_size[0]

    sm = pySaliencyMap.pySaliencyMap(frame_width, frame_height)
    # computation
    saliency_map = sm.SMGetSM(Prog, frame, prev_frame)
    # Visualize
    cv2.imshow('Input image', cv2.flip(frame, 1))
    cv2.imshow('Saliency map', cv2.flip(saliency_map, 1))
    # exit if the key "q" is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # break
        return saliency_map

    return saliency_map


    # cv2.destroyAllWindows()
