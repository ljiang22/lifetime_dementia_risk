from nibabel import load, save
from nibabel.openers import ImageOpener
#from nibabel.mghformat import MGHHeader, MGHError, MGHImage
from nibabel.tmpdirs import InTemporaryDirectory
from nibabel.fileholders import FileHolder
from nibabel.spatialimages import HeaderDataError
from nibabel.volumeutils import sys_is_le
from nibabel.wrapstruct import WrapStructError
from nibabel import imageglobals
import matplotlib.pyplot as plt
from matplotlib import pyplot, image, transforms
import numpy as np
from scipy import ndimage


#mgz_name = 'wm.mgz'
#mgz_name = './data/sub-OAS30001_ses-d2430_acq-AV45_pet.nii'
mgz_name = './data/pet_proc/OAS30001_AV45_d2430.4dfp.hdr'
mgz = load(mgz_name)

    # header
h = mgz.header
print(h)
#print(h['version'], h['type'])
"""assert h['version'] == 1
assert h['type'] == 3
assert h['dof'] == 0
assert h['goodRASFlag'] == 1"""
"""assert_array_equal(h['dims'], [3, 4, 5, 2])
assert_almost_equal(h['tr'], 2.0)
assert_almost_equal(h['flip_angle'], 0.0)
assert_almost_equal(h['te'], 0.0)
assert_almost_equal(h['ti'], 0.0)
assert_array_almost_equal(h.get_zooms(), [1, 1, 1, 2])
assert_array_almost_equal(h.get_vox2ras(), v2r)
assert_array_almost_equal(h.get_vox2ras_tkr(), v2rtkr)"""

# data. will be different for your own mri_volsynth invocation
v = mgz.get_fdata()
#assert_almost_equal(v[1, 2, 3, 0], -0.3047, 4)
#assert_almost_equal(v[1, 2, 3, 1], 0.0018, 4)

print(v.shape)
#print(v[:, :, 0])
print(np.max(v))

for i in range(256):
    if i % 5 == 0:
        plt.figure(i)
        #img_tmp = v[:, :, i, 0] # MRI
        img_tmp = v[:, :, i, 15]
        rotated_img = ndimage.rotate(img_tmp, 90)
        plt.imshow(rotated_img, cmap='jet')
        plt.show()
