#-------------------------------------------------------------------------------
# Name:        pySaliencyMap
# Purpose:     Extracting a saliency map from a single still image
#
# Author:      Akisato Kimura <akisato@ieee.org>
#
# Created:     April 24, 2014
# Copyright:   (c) Akisato Kimura 2014-
# Licence:     All rights reserved
#-------------------------------------------------------------------------------

import cv2
import numpy as np
import sys
sys.path.append('/home/ubuntu/Documentos/ActionRecognition/Code/GP/Saliency')   #path to directory that contains outdir
from . import pySaliencyMapDefs

class pySaliencyMap:
    # initialization
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.prev_frame = None
        self.SM = None
        #self.GaborKernel0   = np.array(pySaliencyMapDefs.GaborKernel_0)
        #self.GaborKernel45  = np.array(pySaliencyMapDefs.GaborKernel_45)
        #self.GaborKernel90  = np.array(pySaliencyMapDefs.GaborKernel_90)
        #self.GaborKernel135 = np.array(pySaliencyMapDefs.GaborKernel_135)

    # extracting color channels
    def SMExtractRGBI(self, inputImage):
        # convert scale of array elements
        src = np.float32(inputImage) * 1./255
        # split
        (B, G, R) = cv2.split(src)
        # extract an intensity image
        I = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # return
        return R, G, B, I

    # feature maps
    ## constructing a Gaussian pyramid
    def FMCreateGaussianPyr(self, src):
        dst = list()
        dst.append(src)
        for i in range(1, 9):
            nowdst = cv2.pyrDown(dst[i-1])
            dst.append(nowdst)
        return dst

    ## taking center-surround differences
    def FMCenterSurroundDiff(self, GaussianMaps):
        dst = list()
        for s in range(2,5):
            now_size = GaussianMaps[s].shape
            now_size = (now_size[1], now_size[0])  ## (width, height)
            tmp = cv2.resize(GaussianMaps[s+3], now_size, interpolation=cv2.INTER_LINEAR)
            nowdst = cv2.absdiff(GaussianMaps[s], tmp)
            dst.append(nowdst)
            tmp = cv2.resize(GaussianMaps[s+4], now_size, interpolation=cv2.INTER_LINEAR)
            nowdst = cv2.absdiff(GaussianMaps[s], tmp)
            dst.append(nowdst)
        return dst

    ## constructing a Gaussian pyramid + taking center-surround differences
    def FMGaussianPyrCSD(self, src):
        GaussianMaps = self.FMCreateGaussianPyr(src)
        dst = self.FMCenterSurroundDiff(GaussianMaps)
        return dst

    ## intensity feature maps
    def IFMGetFM(self, I):
        # Center-Surround Difference
        return self.FMGaussianPyrCSD(I)

    ## color feature map
    # Leon Dozal - CentroGeo - 15/10/2019
    def CFMGetFM(self, prog, TermsCol):
        # Extract GP-color features
        CFM = prog.TreeCol.Eval(TermsCol)
        #prog.PrintTree()
        # obtain feature maps in the same way as intensity
        return self.FMGaussianPyrCSD(CFM)


    ## orientation feature maps
    # Leon Dozal - CentroGeo - 16/10/2019
    def OFMGetFM(self, prog, TermsOri):
        # Extract GP-color features
        OFM = prog.TreeOri.Eval(TermsOri)
        # prog.PrintTree()
        # obtain feature maps in the same way as intensity
        if OFM.size == 1:
            return self.FMGaussianPyrCSD(OFM)
        return self.FMGaussianPyrCSD(OFM)

        # creating a Gaussian pyramid
        #GaussianI = self.FMCreateGaussianPyr(src)
        # convoluting a Gabor filter with an intensity image to extract orientation features
        # GaborOutput0   = [ np.empty((1,1)), np.empty((1,1)) ]  # dummy data: any kinds of np.array()s are OK
        # GaborOutput45  = [ np.empty((1,1)), np.empty((1,1)) ]
        # GaborOutput90  = [ np.empty((1,1)), np.empty((1,1)) ]
        # GaborOutput135 = [ np.empty((1,1)), np.empty((1,1)) ]
        # for j in range(2,9):
            # GaborOutput0.append(   cv2.filter2D(GaussianI[j], cv2.CV_32F, self.GaborKernel0) )
            # GaborOutput45.append(  cv2.filter2D(GaussianI[j], cv2.CV_32F, self.GaborKernel45) )
            # GaborOutput90.append(  cv2.filter2D(GaussianI[j], cv2.CV_32F, self.GaborKernel90) )
            # GaborOutput135.append( cv2.filter2D(GaussianI[j], cv2.CV_32F, self.GaborKernel135) )
        # calculating center-surround differences for every orientation
        # CSD0   = self.FMCenterSurroundDiff(GaborOutput0)
        # CSD45  = self.FMCenterSurroundDiff(GaborOutput45)
        # CSD90  = self.FMCenterSurroundDiff(GaborOutput90)
        # CSD135 = self.FMCenterSurroundDiff(GaborOutput135)
        # concatenate
        # dst = list(CSD0)
        # dst.extend(CSD45)
        # dst.extend(CSD90)
        # dst.extend(CSD135)
        # return dst

    ## Motion feature maps
    def MFMGetFM(self, prog, TermsMot):
        # Extract GP-Motion features
        MFM = prog.TreeMot.Eval(TermsMot)
        return self.FMGaussianPyrCSD(MFM)

    # conspicuity maps
    ## standard range normalization
    def SMRangeNormalize(self, src):
        minn, maxx, dummy1, dummy2 = cv2.minMaxLoc(src)
        if maxx != minn:
            dst = src/(maxx-minn) + minn/(minn-maxx)
        else:
            dst = src - minn
        return dst

    ## computing an average of local maxima
    def SMAvgLocalMax(self, src):
        # size
        stepsize = pySaliencyMapDefs.default_step_local
        width = src.shape[1]
        height = src.shape[0]
        # find local maxima
        numlocal = 0
        lmaxmean = 0
        for y in range(0, height-stepsize, stepsize):
            for x in range(0, width-stepsize, stepsize):
                localimg = src[y:y+stepsize, x:x+stepsize]
                lmin, lmax, dummy1, dummy2 = cv2.minMaxLoc(localimg)
                lmaxmean += lmax
                numlocal += 1

        if numlocal == 0:
            return 0
        else:
            # averaging over all the local regions
            return lmaxmean / numlocal

    ## normalization specific for the saliency map model
    def SMNormalization(self, src):
        dst = self.SMRangeNormalize(src)
        lmaxmean = self.SMAvgLocalMax(dst)
        normcoeff = (1-lmaxmean)*(1-lmaxmean)
        return dst * normcoeff

    ## normalizing feature maps
    def normalizeFeatureMaps(self, FM):
        NFM = list()
        for i in range(0, 6):
            normalizedImage = self.SMNormalization(FM[i])
            nownfm = cv2.resize(normalizedImage, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            NFM.append(nownfm)
        return NFM

    ## intensity conspicuity map
    def ICMGetCM(self, IFM):
        NIFM = self.normalizeFeatureMaps(IFM)
        ICM = sum(NIFM)
        return ICM

    #def CCMGetCM(self, CFM_RG, CFM_BY):
        # extracting a conspicuity map for every color opponent pair
    #    CCM_RG = self.ICMGetCM(CFM_RG)
    #    CCM_BY = self.ICMGetCM(CFM_BY)
        # merge
    #    CCM = CCM_RG + CCM_BY
        # return
    #    return CCM

    # Leon Dozal - CentroGeo - 17/10/2019
    ## orientation conspicuity map
    #def OCMGetCM(self, OFM):
    #    return self.ICMGetCM(OFM)
        # OCM = np.zeros((self.height, self.width))
        # for i in range (0,4):
            # slicing
            # nowofm = OFM[i*6:(i+1)*6]  # angle = i*45
            # extracting a conspicuity map for every angle
            # NOFM = self.ICMGetCM(nowofm)
            # normalize
            # NOFM2 = self.SMNormalization(NOFM)
            # accumulate
            # OCM += NOFM2
        # return OCM

    # Leon Dozal - CentroGeo - 18/10/2019
    ## motion conspicuity map
    # def MCMGetCM(self, MFM_X, MFM_Y):
    #def MCMGetCM(self, MFM):
        #return self.CCMGetCM(MFM_X, MFM_Y)
    #    return self.ICMGetCM(MFM)

    # core
    def SMGetSM(self, prog, src, prev_frame):
        # definitions
        size = src.shape
        width = size[1]
        height = size[0]
        # check
#        if(width != self.width or height != self.height):
#            sys.exit("size mismatch")
        # extracting individual color channels
        R, G, B, I = self.SMExtractRGBI(src)
        p_R, p_G, p_B, p_I = self.SMExtractRGBI(prev_frame)
        # Leon Dozal - CentroGeo - 15/10/2019
        # Get terminals set
        TermsCol = {'R': R, 'G': G, 'B': B, 'I': I}
        TermsOri = {'R': R, 'G': G, 'B': B, 'I': I}

        ## Optical Flow terminals set

        # calculating optical flows
        # if self.prev_frame is not None:
        farne_pyr_scale = pySaliencyMapDefs.farne_pyr_scale
        farne_levels = pySaliencyMapDefs.farne_levels
        farne_winsize = pySaliencyMapDefs.farne_winsize
        farne_iterations = pySaliencyMapDefs.farne_iterations
        farne_poly_n = pySaliencyMapDefs.farne_poly_n
        farne_poly_sigma = pySaliencyMapDefs.farne_poly_sigma
        farne_flags = pySaliencyMapDefs.farne_flags

        flow_R = cv2.calcOpticalFlowFarneback(
            prev=np.uint8(p_R * 255),
            next=np.uint8(R * 255),
            flow=None,
            pyr_scale=farne_pyr_scale,
            levels=farne_levels,
            winsize=farne_winsize,
            iterations=farne_iterations,
            poly_n=farne_poly_n,
            poly_sigma=farne_poly_sigma,
            flags=farne_flags
        )
        flow_R_mag, flow_R_ang = cv2.cartToPolar(flow_R[..., 0], flow_R[..., 1])

        flow_G = cv2.calcOpticalFlowFarneback(
            prev=np.uint8(p_G * 255),
            next=np.uint8(G * 255),
            pyr_scale=farne_pyr_scale,
            levels=farne_levels,
            winsize=farne_winsize,
            iterations=farne_iterations,
            poly_n=farne_poly_n,
            poly_sigma=farne_poly_sigma,
            flags=farne_flags,
            flow=None
        )
        flow_G_mag, flow_G_ang = cv2.cartToPolar(flow_G[..., 0], flow_G[..., 1])

        flow_B = cv2.calcOpticalFlowFarneback(
            prev=np.uint8(p_B * 255),
            next=np.uint8(B * 255),
            pyr_scale=farne_pyr_scale,
            levels=farne_levels,
            winsize=farne_winsize,
            iterations=farne_iterations,
            poly_n=farne_poly_n,
            poly_sigma=farne_poly_sigma,
            flags=farne_flags,
            flow=None
        )
        flow_B_mag, flow_B_ang = cv2.cartToPolar(flow_B[..., 0], flow_B[..., 1])

        flow_I = cv2.calcOpticalFlowFarneback(
            prev=np.uint8(p_I * 255),
            next=np.uint8(I * 255),
            pyr_scale=farne_pyr_scale,
            levels=farne_levels,
            winsize=farne_winsize,
            iterations=farne_iterations,
            poly_n=farne_poly_n,
            poly_sigma=farne_poly_sigma,
            flags=farne_flags,
            flow=None
        )
        flow_I_mag, flow_I_ang = cv2.cartToPolar(flow_I[..., 0], flow_I[..., 1])

        # create Gaussian pyramids

        TermsMot = {'f_R_m': flow_R_mag, 'f_R_a': flow_R_ang, 'f_G_m': flow_G_mag, 'f_G_a': flow_G_ang,
                    'f_B_m': flow_B_mag, 'f_B_a': flow_B_ang, 'f_I_m': flow_I_mag, 'f_I_a': flow_I_ang}

        # Extract feature maps (FM, scales pyramid) by applying the corresponding individual's tree
        # Features Maps:
        # Intensity
        IFM = self.IFMGetFM(I)
        # Color
        # Leon Dozal - CentroGeo - 15/10/2019
        CFM = self.CFMGetFM(prog, TermsCol)
        # Orientation
        # Leon Dozal - CentroGeo - 16/10/2019
        OFM = self.OFMGetFM(prog, TermsOri)
        # Motion Feature Maps
        # Leon Dozal - CentroGeo - 07/03/2023
        MFM = self.MFMGetFM(prog, TermsMot)

        # Extract conspicuity maps (CM)
        # Normalize FM pyramids and sum all levels
        # Leon Dozal - CentroGeo - 07/03/2023
        ICM = self.ICMGetCM(IFM)
        CCM = self.ICMGetCM(CFM)
        OCM = self.ICMGetCM(OFM)
        MCM = self.ICMGetCM(MFM)

        # Add all conspicuity maps to form a saliency map
        # wi = pySaliencyMapDefs.weight_intensity
        # wc = pySaliencyMapDefs.weight_color
        # wo = pySaliencyMapDefs.weight_orientation
        # wm = pySaliencyMapDefs.weight_motion
        # Features Lineal Combination Saliency Map
        # SMMat = wi*ICM + wc*CCM + wo*OCM + wm*MCM

        # Terminal set of the Combination tree
        TermsComb = {'ICM': ICM, 'CCM': CCM, 'OCM': OCM, 'MCM': MCM}

        # Features Combination to get the Saliency Map through the individual's combination tree
        SMMat = prog.TreeComb.Eval(TermsComb)

        # normalize
        normalizedSM = self.SMRangeNormalize(SMMat)
        normalizedSM2 = normalizedSM.astype(np.float32)
        smoothedSM = cv2.bilateralFilter(normalizedSM2, 7, 3, 1.55)
        self.SM = cv2.resize(smoothedSM, (width,height), interpolation=cv2.INTER_NEAREST)
        # return
        return self.SM

    def SMGetBinarizedSM(self, src):
        # get a saliency map
        if self.SM is None:
            self.SM = self.SMGetSM(src)
        # convert scale
        SM_I8U = np.uint8(255 * self.SM)
        # binarize
        thresh, binarized_SM = cv2.threshold(SM_I8U, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return binarized_SM

    def SMGetSalientRegion(self, src):
        # get a binarized saliency map
        binarized_SM = self.SMGetBinarizedSM(src)
        # GrabCut
        img = src.copy()
        mask =  np.where((binarized_SM!=0), cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype('uint8')
        bgdmodel = np.zeros((1,65),np.float64)
        fgdmodel = np.zeros((1,65),np.float64)
        rect = (0,0,1,1)  # dummy
        iterCount = 1
        cv2.grabCut(img, mask=mask, rect=rect, bgdModel=bgdmodel, fgdModel=fgdmodel, iterCount=iterCount, mode=cv2.GC_INIT_WITH_MASK)
        # post-processing
        mask_out = np.where((mask==cv2.GC_FGD) + (mask==cv2.GC_PR_FGD), 255, 0).astype('uint8')
        output = cv2.bitwise_and(img,img,mask=mask_out)
        return output
