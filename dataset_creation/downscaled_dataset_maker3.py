import os
import cv2
import matplotlib.pyplot as plt
import random
import time
import uuid


def show_cv2_image_as_matplotlib_plot(cv2image):
    plt.imshow(cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB))
    plt.show()


def show_lr_and_hr_images(lr_image, hr_image):
    # Convert images from BGR to RGB for correct color display
    lr_image_rgb = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
    hr_image_rgb = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Show LR image
    axes[0].imshow(lr_image_rgb)
    axes[0].set_title("Low Resolution (LR)", fontsize=14)
    axes[0].axis("off")  # Turn off axis

    # Show HR image
    axes[1].imshow(hr_image_rgb)
    axes[1].set_title("High Resolution (HR)", fontsize=14)
    axes[1].axis("off")  # Turn off axis

    # Display the plot
    plt.tight_layout()
    plt.show()


def downscale_image(
    cv2image,
    factor: float,
):
    """
    Where factor is (0-1) where 1 represents the original image, 0.5 represents half the size of the original image
    """
    start_width, start_height = cv2image.shape[1], cv2image.shape[0]
    new_width = int(start_width * factor)
    new_height = int(start_height * factor)

    # shrink the image to the new dims
    downsized_image = cv2.resize(
        cv2image, (new_width, new_height), interpolation=cv2.INTER_AREA
    )

    # expand the image to the original dims
    normalized_downsized_image = cv2.resize(
        downsized_image, (start_width, start_height), interpolation=cv2.INTER_CUBIC
    )

    return normalized_downsized_image


def create_datum(image_path):
    factor_difference = 0.20
    first_ds_factor_upper_range = 1 - factor_difference

    image = cv2.imread(image_path)
    ds_factor_1 = random.randint(1, 100 * first_ds_factor_upper_range) / 100
    if ds_factor_1 < 0.1:
        ds_factor_2 = ds_factor_1 + 0.01
    elif ds_factor_1 < 0.23:
        ds_factor_2 = ds_factor_1 + factor_difference * 0.5
    elif ds_factor_1 < 0.4:
        ds_factor_2 = ds_factor_1 + factor_difference * 0.6
    elif ds_factor_1 < 0.5:
        ds_factor_2 = ds_factor_1 + factor_difference * 0.9
    else:
        ds_factor_2 = ds_factor_1 + factor_difference * 1

    lr_image = downscale_image(image, ds_factor_1)
    hr_image = downscale_image(image, ds_factor_2)

    return lr_image, hr_image


def save_datum_to_dataset(lr_image, hr_image, data_folder, export_dims):
    # set up file organization
    hr_folder = os.path.join(data_folder, "hr")
    lr_folder = os.path.join(data_folder, "lr")
    for folder in [data_folder, hr_folder, lr_folder]:
        os.makedirs(folder, exist_ok=True)

    u = str(uuid.uuid4())
    lr_path = os.path.join(lr_folder, u + ".jpg")
    hr_path = os.path.join(hr_folder, u + ".jpg")

    # resize images to export dims
    lr_image = cv2.resize(lr_image, export_dims, interpolation=cv2.INTER_CUBIC)
    hr_image = cv2.resize(hr_image, export_dims, interpolation=cv2.INTER_CUBIC)

    cv2.imwrite(lr_path, lr_image)
    cv2.imwrite(hr_path, hr_image)


def progress_printout(start_time, current_index, total_count):
    progress = current_index / total_count if total_count != 0 else 0
    progress_string = f"{progress*100:.2f}%"
    time_taken = time.time() - start_time
    time_per_item = time_taken / progress if progress != 0 else 0
    time_remaining = (1 - progress) * time_per_item
    print(
        f"\n\n\n\n\n{progress_string}\ntime taken: {time_taken:.2f}s\ntime remaining: {time_remaining:.2f}s\nindex: {current_index}/{total_count}\n\n",
        end="\r",
    )


def get_datetime():
    return time.strftime("%m_%d")


if __name__ == "__main__":
    test_cat_image_path = r"H:\my_files\my_programs\cat_upscaler\cats\0a3fd8c0-a815-48b0-8cb9-a4ffa6749120.jpg"
    test_cat_image = cv2.imread(test_cat_image_path)
    cats_images_folder = r"H:\my_files\my_programs\cat_upscaler\datasets\raw_cat_images"
    export_dims = (640, 640)
    dataset_name = f"downscale3_{get_datetime()}"
    print("Creating a dataset under the name of ", dataset_name)
    export_dataset_folder = os.path.join(
        r"H:\my_files\my_programs\cat_upscaler\datasets", dataset_name
    )
    start_time = time.time()
    image_file_paths = [
        os.path.join(cats_images_folder, f) for f in os.listdir(cats_images_folder)
    ]
    random.shuffle(image_file_paths)
    for i, image_path in enumerate(image_file_paths):
        lr, hr = create_datum(image_path)
        # show_lr_and_hr_images(lr, hr)
        save_datum_to_dataset(lr, hr, export_dataset_folder, export_dims)
        progress_printout(start_time, i, len(image_file_paths))
