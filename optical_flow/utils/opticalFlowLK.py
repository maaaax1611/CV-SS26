import numpy as np
import cv2

from utils.util import computeBilinerWeights, computeGaussianWeights, invertMatrix2x2


class OpticalFlowLK:

    def __init__(self, winsize, epsilon, iterations):

        self.winsize = winsize
        self.epsilon = epsilon
        self.iterations = iterations

    def compute(self, prevImg, nextImg, prevPts):

        assert prevImg.size != 0 and nextImg.size != 0, "check prevImg and nextImg"
        assert prevImg.shape[0] == nextImg.shape[0], "size mismatch, rows."
        assert prevImg.shape[1] == nextImg.shape[1], "size mismatch, cols."

        N = prevPts.shape[0]
        status = np.ones(N, dtype=int)
        nextPts = np.copy(prevPts)
        
        # Compute the spacial derivatives of prev using the Scharr function
        # cv2.Scharr(src, ddepth, dx, dy) -> dst
        prevDerivx = cv2.Scharr(prevImg, cv2.CV_64F, 1, 0) / 32.0
        prevDerivy = cv2.Scharr(prevImg, cv2.CV_64F, 0, 1) / 32.0


        halfWin = np.array([(self.winsize[0] - 1) * 0.5, (self.winsize[1] - 1) * 0.5])
        weights = computeGaussianWeights(self.winsize, 0.3)

        # iterate over all keypoints
        for ptidx in range(N):
            # remember: u0 is a subpixel position
            u0 = prevPts[ptidx]
            u0 -= halfWin # convert centered coords back (move to top-left corner of the window)

            u = u0
            # floor to get back to full pixel coordinates
            iu0 = [int(np.floor(u0[0])), int(np.floor(u0[1]))]

            # check if window is inside image bounds
            # if not set status to 0 and continue with next point
            if iu0[0] < 0 or \
                    iu0[0] + self.winsize[0] >= prevImg.shape[1] - 1 or \
                    iu0[1] < 0 or \
                    (iu0[1] + self.winsize[1] >= (prevImg.shape[0] - 1)):
                status[ptidx] = 0
                continue
            
            # interpolation weights for current window
            bw = computeBilinerWeights(u0)

            bprev = np.zeros((self.winsize[0] * self.winsize[1], 1)) # (625, 1) contains bightness values
            A = np.zeros((self.winsize[0] * self.winsize[1], 2)) # (625,2) contains gradients
            AtWA = np.zeros((2, 2)) # (2x2) structure matrix
            invAtWA = np.zeros((2, 2)) # (2x2) inverse of structure matrix

            for y in range(self.winsize[1]):
                for x in range(self.winsize[0]):
                    # convert centered to global coordinates
                    gx = int(iu0[0] + x)
                    gy = int(iu0[1] + y)

                    # we need the index bc we are storing values in 1D arrays
                    idx = y * self.winsize[0] + x
                    
                    # subpixel interpolation of the brightness values of the previous image
                    bprev[idx] = (bw[0] * prevImg[gy, gx] + bw[1] * prevImg[gy, gx + 1] +
                                  bw[2] * prevImg[gy + 1, gx] + bw[3] * prevImg[gy + 1, gx + 1])
                    
                    # subpixel interpolation of the gradients of the previous image
                    A[idx,0] = (bw[0] * prevDerivx[gy, gx] + bw[1] * prevDerivx[gy, gx + 1] +
                                bw[2] * prevDerivx[gy + 1, gx] + bw[3] * prevDerivx[gy + 1, gx + 1])
                    
                    A[idx,1] = (bw[0] * prevDerivy[gy, gx] + bw[1] * prevDerivy[gy, gx + 1] +
                                bw[2] * prevDerivy[gy + 1, gx] + bw[3] * prevDerivy[gy + 1, gx + 1])
                    
                    # AtWA computed with the outer product of A[idx] and the weight for the current pixel
                    AtWA += weights[y, x] * np.outer(A[idx], A[idx]) # (2x2)

            # invert structure matrix
            invAtWA = invertMatrix2x2(AtWA) #(2x2)

            # Estimate the target point with the previous point
            u = u0

            ## Iterative solver
            for _ in range(self.iterations):
                iu = [int(np.floor(u[0])), int(np.floor(u[1]))]

                if iu[0] < 0 or iu[0] + self.winsize[0] >= prevImg.shape[1] - 1 \
                        or iu[1] < 0 or iu[1] + self.winsize[1] >= prevImg.shape[0] - 1:
                    status[ptidx] = 0
                    break

                bw = computeBilinerWeights(u)
                AtWbnbp = np.zeros((2, 1))

                for y in range(self.winsize[1]):
                    for x in range(self.winsize[0]):
                        gx = iu[0] + x
                        gy = iu[1] + y

                        idx = self.winsize[0] * y + x

                        # subpixel interpolation of the brightness values of the next image
                        I_next = (bw[0] * nextImg[gy, gx] + bw[1] * nextImg[gy, gx + 1] +
                                  bw[2] * nextImg[gy + 1, gx] + bw[3] * nextImg[gy + 1, gx + 1])
                        
                        # intensity difference between the next image and the previous image at the current pixel
                        diff = I_next - bprev[idx, 0]
                        # accumulate
                        # AtWbnbp describes how and in what direction brightness changes between two images
                        AtWbnbp += weights[y, x] * np.outer(A[idx], diff)
                
                # check for convergence (solve for deltaU and update u)
                deltaU = -np.matmul(invAtWA, AtWbnbp)
                u = u + deltaU.flatten()

                # early termination
                if np.sum(deltaU**2) < self.epsilon:
                    break

            nextPts[ptidx] = u + halfWin

        return nextPts, status
