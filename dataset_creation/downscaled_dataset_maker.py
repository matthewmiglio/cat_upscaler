import shutil
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
import uuid
import os


IMAGE_RESIZE = (1920, 1080)


def resize_image(image, scale_factor):
    # Calculate the new dimensions based on the scale factor
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    new_dim = (new_width, new_height)

    # Resize image using bicubic interpolation
    resized_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_CUBIC)
    return resized_image


def gaussian_blur(image, kernel_size):
    # Apply Gaussian Blur to the image
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image


def motion_blur(image, kernel_size, angle):
    # Create a motion blur kernel
    M = np.zeros((kernel_size, kernel_size))
    # Set the direction of the blur
    center = kernel_size // 2
    for i in range(kernel_size):
        M[center, i] = 1  # Horizontal motion blur

    # Rotate the kernel if necessary
    M = cv2.warpAffine(
        M,
        cv2.getRotationMatrix2D((center, center), angle, 1),
        (kernel_size, kernel_size),
    )

    # Apply the motion blur kernel
    motion_blurred_image = cv2.filter2D(image, -1, M)
    return motion_blurred_image


def pixelate(image, block_size):
    # Get the dimensions of the image
    height, width, _ = image.shape

    # Resize the image to the block size (downscale)
    small = cv2.resize(
        image,
        (width // block_size, height // block_size),
        interpolation=cv2.INTER_LINEAR,
    )

    # Resize back to the original size (upscale)
    pixelated_image = cv2.resize(
        small, (width, height), interpolation=cv2.INTER_NEAREST
    )
    return pixelated_image


def add_gaussian_noise(image, mean=0, stddev=25):
    # Generate Gaussian noise
    noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8)

    # Add noise to the original image
    noisy_image = cv2.add(image, noise)
    return noisy_image


def add_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    # Create a copy of the image to add noise
    noisy_image = image.copy()

    # Salt noise (white pixels)
    salt = np.random.rand(*image.shape[:2]) < salt_prob
    noisy_image[salt] = 255  # White pixels

    # Pepper noise (black pixels)
    pepper = np.random.rand(*image.shape[:2]) < pepper_prob
    noisy_image[pepper] = 0  # Black pixels

    return noisy_image


def add_jpeg_compression(image, quality=30):
    # Encode the image to a JPEG format with the specified quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, compressed_image = cv2.imencode(".jpg", image, encode_param)

    # Decode the image back into an array
    decompressed_image = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)
    return decompressed_image


def chroma_subsampling(image, factor=2):
    # Convert the image from BGR to YUV color space
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # Subsample the chroma channels (U, V) by reducing the resolution by the given factor
    yuv_image[::factor, ::factor, 1] = yuv_image[::factor, ::factor, 1]  # U channel
    yuv_image[::factor, ::factor, 2] = yuv_image[::factor, ::factor, 2]  # V channel

    # Convert back to BGR color space
    subsampled_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)
    return subsampled_image


def show_image_matplotlib(cv2_image):
    plt.imshow(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def do_random_image_operation(image):
    random_operation = random.randint(0, 6)
    if random_operation == 0:
        random_resize = random.randint(30, 90) / 100
        image = resize_image(image, random_resize)
    elif random_operation == 1:
        this_blur = random.choice([1, 3, 5, 9])
        image = gaussian_blur(image, this_blur)
    elif random_operation == 2:
        motion_blur_angle = random.randint(0, 90)
        image = motion_blur(image, 1, motion_blur_angle)
    elif random_operation == 3:
        pixelate_block_size = random.randint(1, 6)
        image = pixelate(image, pixelate_block_size)
    elif random_operation == 4:
        gaussian_mean = random.randint(1, 1)
        gaussian_stddev = random.randint(0, 10)
        image = add_gaussian_noise(image, mean=gaussian_mean, stddev=gaussian_stddev)
    elif random_operation == 5:
        pepper_percent = random.randint(1, 5) / 100
        salt_percent = random.randint(1, 5) / 100
        image = add_salt_and_pepper_noise(image, salt_percent, pepper_percent)
    elif random_operation == 6:
        jpg_compression = random.randint(1, 10)
        image = add_jpeg_compression(image, quality=jpg_compression)

    return image

def do_random_downsample_operation(image):
    jpg_compression = random.randint(2, 10)
    random_resize = random.randint(30, 90) / 100
    this_blur = random.choice([3, 5, 9])
    motion_blur_angle = random.randint(0, 90)
    pixelate_block_size = random.randint(2, 6)

    this_operation_index = random.randint(0,4)
    if this_operation_index == 0:image = add_jpeg_compression(image, quality=jpg_compression)
    if this_operation_index == 1:image = resize_image(image, random_resize)
    if this_operation_index == 2:image = gaussian_blur(image, this_blur)
    if this_operation_index == 3:image = motion_blur(image, 1, motion_blur_angle)
    if this_operation_index == 4:image = pixelate(image, pixelate_block_size)

    return image


def make_lr_images(image,only_quality_related = False):
    images = [image]

    for i in range(random.randint(1, 18)):
        if only_quality_related is False: image = do_random_image_operation(image)
        else: image = do_random_downsample_operation(image)
        images.append(image)

    return images


def image2data(input_cv2_image):
    """
    Takes in a cv2 image, returns the HR image and LR image
    """

    images = make_lr_images(input_cv2_image,only_quality_related=True)  # 0 is highest res, -1 is lowest res
    lr_image = random.choice(images[1:])
    hr_image = images[0]

    lr_image_resized = cv2.resize(lr_image, IMAGE_RESIZE, interpolation=cv2.INTER_CUBIC)
    hr_image_resized = cv2.resize(hr_image, IMAGE_RESIZE, interpolation=cv2.INTER_CUBIC)

    return lr_image_resized, hr_image_resized


def raw_images_to_data_images(
    raw_images_dir, export_lr_folder, export_hr_folder, image_process_count=None
):
    def progress_printout(good_count, fail_count, total_count):
        if good_count + fail_count == 0:
            return

        current_total = good_count + fail_count
        good_percent = (
            round((good_count / current_total * 100), 2) if good_count > 0 else 0
        )
        fail_percent = (
            round((fail_count / current_total * 100), 2) if fail_count > 0 else 0
        )
        progress_percent = (
            round((current_total / total_count * 100), 2) if total_count > 0 else 0
        )

        out_string = ""
        out_string += f"\n\n\n\n\n\n\n\n\nGood processes: {good_count} {good_percent}%"
        out_string += f"\nFail processes: {fail_count} {fail_percent}%"
        out_string += f"\nTotal processes: {current_total}"
        out_string += (
            f"\nProgress {current_total} / {total_count} = {progress_percent}%\n"
        )
        print(out_string, end="\r")

    def make_uid():
        return str(uuid.uuid4().hex)

    good_count = 0
    fail_count = 0
    fail_paths = []

    for folder_path in [export_lr_folder, export_hr_folder]:
        os.makedirs(folder_path, exist_ok=True)

    raw_image_paths = [
        os.path.join(raw_images_dir, f) for f in os.listdir(raw_images_dir)
    ]

    if image_process_count is not None:
        raw_image_paths = random.sample(raw_image_paths, image_process_count)

    for image_path in raw_image_paths:
        try:
            image = cv2.imread(image_path)
            lr_image, hr_image = image2data(image)

            uid = make_uid()

            lr_image_path = os.path.join(export_lr_folder, f"{uid}.jpg")
            hr_image_path = os.path.join(export_hr_folder, f"{uid}.jpg")

            cv2.imwrite(lr_image_path, lr_image)
            cv2.imwrite(hr_image_path, hr_image)
        except:
            fail_count += 1
            fail_paths.append(image_path)
            continue
        good_count += 1

        progress_printout(good_count, fail_count, len(raw_image_paths))

    if fail_count > 0:
        print("\n\n\nFailed paths:")
        for path in fail_paths:
            print("\t", path)

        print(f"A total of {fail_count} images were failed to process.")
        if input("Would you like to remove the erroring images? (y/n)") == "y":
            for path in fail_paths:
                os.remove(path)


if __name__ == "__main__":
    dataset_name = r"true_quality_only_dataset"
    if os.path.exists(dataset_name):
        shutil.rmtree(dataset_name)

    export_lr_folder = os.path.join(dataset_name, "LR")
    export_hr_folder = os.path.join(dataset_name, "HR")
    raw_images_folder = r"H:\my_files\my_programs\cat_upscaler\cats"
    raw_images_to_data_images(
        raw_images_folder, export_lr_folder, export_hr_folder, image_process_count=None
    )

    def test_downscale_dataset_maker():
        lr_images = make_lr_images(only_quality_related=True,image=cv2.imread(r'H:\my_files\my_programs\cat_upscaler\cats\0a2cb91f-16a5-4ce5-8e10-6a12aa1ca349.jpg'))
        for lr_image in lr_images:
            show_image_matplotlib(lr_image)

    def test_do_random_downsample_operation():
        test_image_path = r'H:\my_files\my_programs\cat_upscaler\cats\0a3fd8c0-a815-48b0-8cb9-a4ffa6749120.jpg'
        test_image = cv2.imread(test_image_path)

        downsampled_image = do_random_downsample_operation(test_image)
        print('Youre seeing the raw input image\n')
        show_image_matplotlib(test_image)
        print('\n'*50,'Youre seeing the downsampled image\n')
        show_image_matplotlib(downsampled_image)

    def test_image2data():
        test_image_folder = 'cats'
        for file in random.sample(os.listdir(test_image_folder),5):
            test_image_path = os.path.join(test_image_folder,file)
            test_image = cv2.imread(test_image_path)
            lr_image, hr_image = image2data(test_image)
            print('\n'*50,'Youre seeing the low quality\n')
            show_image_matplotlib(lr_image)
            print('\n'*50,'Youre seeing the high quality image\n')
            show_image_matplotlib(hr_image)


    # test_downscale_dataset_maker()

    # test_do_random_downsample_operation()

    # test_image2data()
