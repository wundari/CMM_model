
import numpy as np
from RDS_v2 import RDS

import matplotlib.pyplot as plt


w_bg = 512
h_bg = 256
w_ct = w_bg // 2
h_ct = h_bg // 2
dotDens = 0.25
rDot = 5 # dot radius in pixel
overlap_flag = 0 # dots are not allowed to overlap
n_rds = 50
rds = RDS(n_rds, w_bg, h_bg,
          w_ct, h_ct,
          dotDens,
          rDot,
          overlap_flag)

disp_ct_pix = [-30, 30]

########################################################################
# ards: 0, crds: 1, hmrds: 0.5, urds: -1
dotMatch_ct = 0.5
rds_with_bg = rds.create_rds_batch(disp_ct_pix, dotMatch_ct)

## remapping rds into [n_trial, nx, ny, left-right]
rds_left = np.zeros((n_rds, 
                     3, # rgb channels
                     rds.size_rds_bg[0],
                     rds.size_rds_bg[1]),
                    dtype=np.float32)
rds_right = np.zeros((n_rds, 
                      3, # rgb channels
                      rds.size_rds_bg[0],
                      rds.size_rds_bg[1]),
                     dtype=np.float32)
rds_label = np.zeros(n_rds, dtype=np.int8)
count = 0
for d in range(len(disp_ct_pix)):
    
    if disp_ct_pix[d] < 0:
        depth_label = 0
    else:
        depth_label = 1
        
    for t in range(n_rds):
        
        temp = rds_with_bg[0][t, d]
        temp = np.roll(temp, disp_ct_pix[1]//2, axis=1)
        rds_left[count, 0] = temp*0.3
        rds_left[count, 0] = temp*0.3
        rds_left[count, 0] = temp*0.3
        
        temp = rds_with_bg[1][t, d]
        temp = np.roll(temp, -disp_ct_pix[1]//2, axis=1)
        rds_right[count, 0] = temp*0.3
        rds_right[count, 1] = temp*0.3
        rds_right[count, 2] = temp*0.3

        rds_label[count] = depth_label

        count += 1

## plot rds
fig, ax = plt.subplots(figsize=(15, 10), 
                       nrows=2, ncols=2)
fig.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.0)
fig.text(0.5, 1.0,
         "hmRDS",
         fontsize=24,
         ha="center")

ax[0, 0].imshow(rds_left[0][0], cmap="gray")
ax[0, 0].axis("off")
ax[0, 1].imshow(rds_right[0][0], cmap="gray")
ax[0, 1].axis("off")

ax[1, 0].imshow(rds_left[-1][0], cmap="gray")
ax[1, 0].axis("off")
ax[1, 1].imshow(rds_right[-1][0], cmap="gray")
ax[1, 1].axis("off")

fig.savefig("results/hmrds_{}.png"
            .format(dotDens))


#### test gc-net ####
h = 256
w = 512
maxdisp = 160 #gc_net.py also need to change  must be a multiple of 32...maybe can cancel the outpadding of deconv
net = GCNet(h, w, maxdisp)
checkpoint = torch.load('./checkpoint/ckpt.t7')
net.load_state_dict(checkpoint['net'])
net.to(device)
        
# predict disparity map
batch_test = 2
pred_disp = np.zeros((2*n_rds_trial,
                      size_rds_bg_pix_height,
                      size_rds_bg_pix_width), 
                     dtype=np.float32)

for i in range(2*n_rds_trial):
        
    id_start = i * batch_test
    id_end = id_start + batch_test
    
    input_left = rds_left[id_start:id_end]
    input_right = rds_right[id_start:id_end]
    temp = net.predict(torch.tensor(input_left).to(device),
                       torch.tensor(input_right).to(device))

    pred_disp[id_start:id_end] = temp.cpu().detach().numpy()

pred_disp_near = pred_disp[0:n_rds_trial]
pred_disp_far = pred_disp[n_rds_trial:]

## plot disparity map
fig, ax = plt.subplots(figsize=(15, 10),
                       nrows=2, ncols=2)
ax[0, 0].imshow(pred_disp_near[0])
predict_disp_line = pred_disp_near[0][128]
ax[1, 0].plot(predict_disp_line)

ax[0, 1].imshow(pred_disp_far[0])
predict_disp_line = pred_disp_far[0][128]
ax[1, 1].plot(predict_disp_line)
plt.savefig("results/disp_map_hmrds_dotDens_{}.png"
            .format(dotDens))



## center - background
id_mask_start = (63, 157)
id_mask_end = (195, 413)
mask_center = np.zeros((size_rds_bg_pix_height,
                       size_rds_bg_pix_width),
                       dtype=np.float32)
mask_center[id_mask_start[0]:id_mask_end[0],
            id_mask_start[1]:id_mask_end[1]] = 1
mask_bg = np.ones((size_rds_bg_pix_height,
                   size_rds_bg_pix_width), dtype=np.float32) - mask_center

depth_expect = np.zeros(2*n_rds_trial, dtype=np.float32)
for i in range(2*n_rds_trial):
    
    # expected depth in the center
    temp = pred_disp[i]
    depth_per_pix_center = np.sum(temp*mask_center)/np.sum(mask_center)
    
    # expected depth in the background
    depth_per_pix_bg = np.sum(temp*mask_bg)/np.sum(mask_bg)
    
    depth_expect[i] = depth_per_pix_center - depth_per_pix_bg

plt.plot(depth_expect)


##



###########################################################################
#### rds without background ####
############################################################################

# ards: 0, crds: 1, hmrds: 0.5, urds: -1
dotMatch_ct = 0
rds_wo_bg = rds.create_rds_without_bg_batch(disp_ct_pix, dotMatch_ct)
# np.save("crds_wo_bg", rds_wo_bg)

## remapping rds into [n_trial, nx, ny, left-right]
rds_left = np.zeros((n_rds, 
                     3,
                     rds.size_rds_bg[0],
                     rds.size_rds_bg[1]),
                    dtype=np.float32)
rds_right = np.zeros((n_rds, 
                      3,
                      rds.size_rds_bg[0],
                      rds.size_rds_bg[1]),
                     dtype=np.float32)
rds_label = np.zeros(n_rds, dtype=np.int8)
count = 0
for d in range(len(disp_ct_pix)):
    
    if disp_ct_pix[d] < 0:
        depth_label = 0
    else:
        depth_label = 1
        
    for t in range(n_rds_trial):
        
        temp = rds_wo_bg[0][t, d]
        temp = np.roll(temp, disp_ct_pix[1]//2, axis=1)
        rds_left[count, 0] = temp*0.3
        rds_left[count, 0] = temp*0.3
        rds_left[count, 0] = temp*0.3
        
        temp = rds_wo_bg[1][t, d]
        temp = np.roll(temp, -disp_ct_pix[1]//2, axis=1)
        rds_right[count, 0] = temp*0.3
        rds_right[count, 1] = temp*0.3
        rds_right[count, 2] = temp*0.3

        rds_label[count] = depth_label

        count += 1
        
## plot rds
fig, ax = plt.subplots(figsize=(15, 10), 
                       nrows=2, ncols=2)
fig.tight_layout()
plt.subplots_adjust(wspace=0.05, hspace=0.0)
fig.text(0.5, 1.0,
         "hmRDS without background",
         fontsize=24,
         ha="center")

ax[0, 0].imshow(rds_left[0][0])
ax[0, 0].axis("off")
ax[0, 1].imshow(rds_right[0][0])
ax[0, 1].axis("off")

ax[1, 0].imshow(rds_left[-1][0])
ax[1, 0].axis("off")
ax[1, 1].imshow(rds_right[-1][0])
ax[1, 1].axis("off")

fig.savefig("../rds_images/hmrds_wo_bg.png")

    
#### test gc-net ####
h = 256
w = 512
maxdisp = 160 #gc_net.py also need to change  must be a multiple of 32...maybe can cancel the outpadding of deconv
net = GCNet(h, w, maxdisp)
checkpoint = torch.load('./checkpoint/ckpt.t7')
net.load_state_dict(checkpoint['net'])
net.to(device)
        
# predict disparity map
batch_test = 2
pred_disp = np.zeros((2*n_rds_trial,
                      size_rds_bg_pix_height,
                      size_rds_bg_pix_width), 
                     dtype=np.float32)

for i in range(2*n_rds_trial):
        
    id_start = i * batch_test
    id_end = id_start + batch_test
    
    input_left = rds_left[id_start:id_end]
    input_right = rds_right[id_start:id_end]
    temp = net.predict(torch.tensor(input_left).to(device),
                       torch.tensor(input_right).to(device))

    pred_disp[id_start:id_end] = temp.cpu().detach().numpy()

pred_disp_near = pred_disp[0:n_rds_trial]
pred_disp_far = pred_disp[n_rds_trial:]

fig, ax = plt.subplots(figsize=(15, 10),
                       nrows=2, ncols=2)
ax[0, 0].imshow(pred_disp_near.mean(axis=0))
predict_disp_line = pred_disp_near.mean(axis=0)[128]
ax[1, 0].plot(predict_disp_line)

ax[0, 1].imshow(pred_disp_far.mean(axis=0))
predict_disp_line = pred_disp_far.mean(axis=0)[128]
ax[1, 1].plot(predict_disp_line)