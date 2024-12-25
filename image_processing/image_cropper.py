import cv2
from concurrent.futures import ThreadPoolExecutor
import os


def crop_image(image, x, y, width, height):
    cropped_image = image[y : y + height, x : x + width]
    return cropped_image


def stretch_image(image, width, height):
    try:
        stretched_image = cv2.resize(image, (width, height))
        return stretched_image
    except Exception as e:
        pass

    return False


def batch_crop_images(images, x, y, width, height):
    print("batch cropping images...")
    with ThreadPoolExecutor() as executor:
        cropped_images = list(
            executor.map(lambda img: crop_image(img, x, y, width, height), images)
        )
    return cropped_images


def batch_stretch_images(images, width, height):
    print("batch stretching images...")
    with ThreadPoolExecutor() as executor:
        stretched_images = list(
            executor.map(lambda img: stretch_image(img, width, height), images)
        )
        false_count = len([i for i in stretched_images if i is False])
        print(f"Failed to stretch {false_count} images")
        stretched_images = [i for i in stretched_images if i is not False]
    return stretched_images


def load_images(all_image_paths):
    images = []
    count = len(all_image_paths)
    for i, image_path in enumerate(all_image_paths):
        print(f"loading image: {i} / {count}", end="\r")
        image = cv2.imread(image_path)
        images.append(image)

    return images


import random


def batch_stretch_folder(folder_path, export_folder_path, width, height):
    def progress_printout(good_count, fail_count, total_count):
        if 0 in [good_count, fail_count, total_count]:
            good_count += 1
            fail_count += 1
            total_count += 1

        current_total = good_count + fail_count
        good_percent = round((good_count / current_total * 100), 2)
        fail_percent = round((fail_count / current_total * 100), 2)
        progress_percent = round((current_total / total_count * 100), 2)

        out_string = ""
        out_string += f"\n\n\nGood processes: {good_count} {good_percent}%"
        out_string += f"\nFail processes: {fail_count} {fail_percent}%"
        out_string += f"\nTotal processes: {current_total} 100%"
        out_string += (
            f"\nProgress {current_total} / {total_count} = {progress_percent}%"
        )
        print(out_string, end="\r")

    # grab all images in the folder
    existing_files = os.listdir(export_folder_path)
    all_input_file_names = os.listdir(folder_path)
    all_unprocessed_input_file_names = [
        f for f in all_input_file_names if f not in existing_files
    ]
    all_unprocessed_input_file_paths = [
        os.path.join(folder_path, f) for f in all_unprocessed_input_file_names
    ]

    random.shuffle(all_unprocessed_input_file_paths)
    print(
        f"There are {len(existing_files)} images already processed, and {len(all_unprocessed_input_file_paths)} images left to process"
    )

    good_processes = 0
    fail_processes = 0
    fail_image_paths = []

    for image_path in all_unprocessed_input_file_paths:
        image = cv2.imread(image_path)
        stretched_image = stretch_image(image, width, height)
        if stretched_image is False:
            fail_image_paths.append(image_path)
            fail_processes += 1
            continue

        export_path = os.path.join(export_folder_path, os.path.basename(image_path))
        cv2.imwrite(export_path, stretched_image)
        good_processes += 1
        if random.randint(0, 10) == 10:
            progress_printout(
                good_processes, fail_processes, len(all_unprocessed_input_file_paths)
            )

    if len((fail_image_paths)) > 0:
        print("Failed to process the following images:")
        for i in fail_image_paths:
            print("\t", i)
        print(f"thats a total of {len(fail_image_paths)} failed processes")
        if input("Would you like to remove these invalid images? (y/n)") == "y":
            for i in fail_image_paths:
                os.remove(i)
            print("All invalid images removed")

    print(f"Finished batch stretching {len(all_unprocessed_input_file_paths)} images")


if __name__ == "__main__":
    folder_path = r"C:\Users\matmi\Desktop\my files\my programs\upscaler_train\HR"
    export_folder_path = (
        r"C:\Users\matmi\Desktop\my files\my programs\upscaler_train\HR_stretched"
    )
    width = 1920
    height = 1080
    batch_stretch_folder(folder_path, export_folder_path, width, height)
