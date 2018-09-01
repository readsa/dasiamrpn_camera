# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import cv2  # imread
import torch
import numpy as np

from net import SiamRPNBIG
from run_SiamRPN import SiamRPN_init, SiamRPN_track

def main():
    # load net
    net = SiamRPNBIG()
    net.load_state_dict(torch.load('SiamRPNBIG.model'))
    net.eval().cuda()
    
    # warm up
    for i in range(10):
        net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cuda())
        net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)).cuda())
    
    # start to track
    cap = cv2.VideoCapture(0)
    while True:
        success,im=cap.read()
        if success is not True:
            break
        cv2.imshow('cam',im)
        k = cv2.waitKey(1)
        if k==27:
            x,y,w,h = cv2.selectROI('cam',im)
            cx,cy = (x+w/2,y+h/2)
            break
    
    target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
    state = SiamRPN_init(im, target_pos, target_sz, net)  # init tracker
    while True:
        success,im=cap.read()  # HxWxC
        if success is not True:
            break
        state = SiamRPN_track(state, im)  # track
        bbox = np.array([state['target_pos'][0]-state['target_sz'][0]/2, 
                         state['target_pos'][1]-state['target_sz'][1]/2, 
                         state['target_pos'][0]+state['target_sz'][0]/2, 
                         state['target_pos'][1]+state['target_sz'][1]/2]).astype(int)
        pt = (state['target_pos'],state['target_sz'])
        im_bbox = cv2.rectangle(im,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),3)
        cv2.imshow('cam',im_bbox)
        k = cv2.waitKey(1)
        if k==27:
            cap.release()
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()