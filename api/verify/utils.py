import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import os
from django.conf import settings

media_folder = settings.MEDIA_ROOT
model_path = os.getcwd() + "/static/aimodels/RRDB_ESRGAN_x4.pth"  # models/RRDB_ESRGAN_x4.pth OR models/RRDB_PSNR_x4.pth
device = torch.device('cpu')  # if you want to run on CPU, change 'cuda' -> cpu
# device = torch.device('cpu')
userid = 1
test_img_folder = media_folder + "images/user_" + str(userid) + "/*"
test_user_folder = media_folder + "media/images/user_" + str(userid) + "/"
print(test_img_folder)
print(test_user_folder)
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
for path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(path))[0]
    print(idx, base)
    # read images
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    cv2.imwrite('{pth}{file}_rlt.png'.format(pth=test_user_folder, file=base), output)
