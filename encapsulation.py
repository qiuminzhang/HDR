import cv2
import numpy as np


def readImagesAndTimes():
    times = np.array([1 / 30.0, 0.25, 2.5, 15.0], dtype=np.float32)

    filenames = ["img_0.033.jpg", "img_0.25.jpg", "img_2.5.jpg", "img_15.jpg"]

    images = []
    for filename in filenames:
        im = cv2.imread(filename)
        images.append(im)

    return images, times


def align_images(images):
    """
    Converts all the images to median threshold bitmaps (MTB),  the MTBs can be aligned without requiring us to
    specify the exposure time.
    :return:
    """
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(images, images)
    return images


def obtain_camera_response_function(images, times):
    calibrateDebevec = cv2.createCalibrateDebevec()
    responseDebevec = calibrateDebevec.process(images, times)
    return responseDebevec


def generate_hdr_image(images, times, responseDebevec):
    """
    Merge multiple exposure images into one HDR image
    :return:
    """
    mergeDebevec = cv2.createMergeDebevec()
    hdrDebevec = mergeDebevec.process(images, times, responseDebevec)
    return hdrDebevec


# ----------- Tone Mapping-----------------
# Tone mapping is the process to convert a HDR image to an 8-bit per channel image while
# preserving as much detail as possible.

# There are multiple tone-mapping algorithms. OpenCV implements four of them.
def tone_mapping_Drago(hdrDebevec):
    tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
    ldrDrago = tonemapDrago.process(hdrDebevec)
    ldrDrago = 3 * ldrDrago
    return ldrDrago * 255


def tone_mapping_durand(hdrDebevec):
    tonemapDurand = cv2.createTonemapDurand(1.5, 4, 1.0, 1, 1)
    ldrDurand = tonemapDurand.process(hdrDebevec)
    ldrDurand = 3 * ldrDurand
    return ldrDurand * 255


def tone_mapping_reinhard(hdrDebevec):
    tonemapReinhard = cv2.createTonemapReinhard(1.5, 0, 0, 0)
    ldrReinhard = tonemapReinhard.process(hdrDebevec)
    return ldrReinhard * 255


def tone_mapping_Mantiuk(hdrDebevec):
    tonemapMantiuk = cv2.createTonemapMantiuk(2.2, 0.85, 1.2)
    ldrMantiuk = tonemapMantiuk.process(hdrDebevec)
    ldrMantiuk = 3 * ldrMantiuk
    return ldrMantiuk * 255


def save_image(outputname, outputfile):
    cv2.imwrite(outputname, outputfile)


def main():
    # read images and exposure times
    images, times = readImagesAndTimes()

    # Align input images
    images = align_images(images)

    # Obtain Camera Response Function (CRF)
    responseDebevec = obtain_camera_response_function(images, times)

    # Merge images into an HDR linear image
    hdrDebevec = generate_hdr_image(images, times, responseDebevec)
    # Save hdr image
    save_image("hdr.hdr", hdrDebevec)

    # ----------------- 4 options of tone mapping -----------------
    # 1. Drago's method
    outputimage = tone_mapping_Drago(hdrDebevec)
    # Save
    save_image("Drago.jpg", outputimage)

    # 2. Durand's method
    outputimage = tone_mapping_durand(hdrDebevec)
    # Save
    save_image("Durand.jpg", outputimage)

    # 3. Reinhard's method
    outputimage = tone_mapping_reinhard(hdrDebevec)
    # Save
    save_image("Reinhard.jpg", outputimage)

    # 4. Mantiuk's method
    outputimage = tone_mapping_Mantiuk(hdrDebevec)
    # Save
    save_image("Mantiuk.jpg", outputimage)


if __name__ == '__main__':
    main()

