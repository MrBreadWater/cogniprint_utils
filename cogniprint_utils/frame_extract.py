"""CLI script to extract frames from a folder of videos.

This script is a bit shit. I wrote it when I was 12, and relatively new to Python.
I've cleaned it up a little bit, but I don't recommend using it if you have an alternative.

At least I had the sense to write comments back then.

Example:
     $ python frame_extract.py -v a_folder_with_videos/ -i frame_output_folder/

Copyright © 2020 Michael Paniagua (MrBreadWater) <mrbreadwater@yahoo.com>
This work is free. You can redistribute it and/or modify it under the
terms of the Do What The Fuck You Want To Public License, Version 2,
as published by Sam Hocevar. See http://www.wtfpl.net/ for more details.
"""

import gc
import os
import time
import cv2
import numpy as np

VIDEO_FOLDER = 'timelapses/'
IMAGE_FOLDER = 'images/'
DEBUG = False


def calc_frames(seconds, multiplier=0.5):
    """Calculates the number of frames which should be extracted from each video.

    The equation is as follows: floor(sqrt(m*s))

    Written so that timelapse length has diminishing returns on number of frames extracted.
    Why? Good question.

    Args:
        seconds (numpy.array): An array containing the length of each video.
        multiplier (float): Scales down (or up, I suppose) the number of frames extracted

    Returns:
        numpy.array: The number of frames to extract
    """

    frame_count_array = np.sqrt(seconds * multiplier).astype(int)

    if DEBUG:
        print("seconds = ", seconds)
        print("frame_count_array = ", frame_count_array)

    return frame_count_array


def gen_frame_count(timelapse_count):
    """Gets frame counts for each timelapse

    Why is it called "gen" frame count? Who the fuck knows.

    Args:
        timelapse_count (int): Number of timelapses. Also corresponds the timelapse names for some reason.

    Returns:
        2-tuple of numpy.arrays: Arrays containing the number of frames per video and the length of each video in seconds
    """

    print("Generating FrameCount array")
    time_start = time.time()
    video_fps_values = []  # number of frames per second for each video
    video_frame_nums = []  # number of frames within each video

    # TODO: Replace with actual iteration over a list of paths
    for i in range(timelapse_count):
        print("\rGenerating values for timelapse #", i + 1, " of ", timelapse_count, end="")
        try:
            # Hacky way to use either mpg or mp4 - 2015 me
            # Why. Why would you ever. *ever do this.* - 2020 me
            clip = cv2.VideoCapture(os.path.abspath(VIDEO_FOLDER) + '/' + str(i) + '.mkv')
            if not clip.read()[0]:  # clip.read()[0] will be true or false depending on success
                print("Loading from MP4")
                clip = cv2.VideoCapture(os.path.abspath(VIDEO_FOLDER) + '/' + str(i) + ".mp4")

            # Calculate frames
            video_frame_nums.append(int(clip.get(cv2.CAP_PROP_FRAME_COUNT)))
            video_fps_values.append(int(clip.get(cv2.CAP_PROP_FPS)))

        # TODO: Use real error handling. (Seriously what the fuck dude.)
        except Exception as e:
            print("Exception gen_frame_count! Fix, or your results are fucked!")
            print(e)

    video_times = np.divide(video_frame_nums, video_fps_values)
    frames_array = calc_frames(video_times)  # calculate array for # of extracted frames per video
    if DEBUG:
        print("video fps vals = ", video_fps_values, "\n video frame nums = ", video_frame_nums)
        print("Video times = ", video_times)
        print("Generated array in %s seconds!" % round(time.time() - time_start, 2))
    return frames_array, video_times


def gen_times_array(timelapse_count):
    '''Gets the frequency in seconds at which extractions should occur
    
    Args:
        timelapse_count (int): Number of timelapses. Also corresponds the timelapse names for some reason.

    Returns:
        2-tuple of numpy.arrays: Arrays containing the frequency in seconds at which extractions should occur and the length of each video
    '''

    frames_array, video_times = gen_frame_count(timelapse_count)  # get value arrays for videos

    time_start = time.time()  # Timing it for some reason. Think I wanted to test how fast using numpy made it?

    print("Generating Interval Array... This may take some time.")

    interval_array = np.divide(video_times, frames_array)  # generate extraction interval

    if DEBUG:
        print("Generated interval array in ", str(round(time.time() - time_start, 2)), "seconds!")
        print(frames_array, video_times)
        print("Interval Array = ", interval_array)

    return interval_array, frames_array


def extract_video_frames(timelapse_count):
    '''Extracts video frames into apropriate folders

    Args:
        timelapse_count (int): Number of timelapses. Also corresponds the timelapse names for some reason.

    Returns:
        int: number of extracted images
    '''

    interval_array, loop_counts = gen_times_array(timelapse_count)
    image_num = 0
    timelapse_path = os.path.abspath(VIDEO_FOLDER) + '/'
    image_path = (os.path.abspath(IMAGE_FOLDER) + '/')

    print(timelapse_count, "timelapses to extract from.")
    print("Extracting images... this will take some time.")

    for i in range(timelapse_count):
        print("Extracting images for timelapse #", i, " of ", timelapse_count)
        gc.collect()  # collect and remove unused variables
        vid_time = 0  # reset time for cv2 to grab frame
        try:
            # TODO: Fix this mess. Use file path array.
            for j in range(int(loop_counts[i])):
                vidcap = cv2.VideoCapture(timelapse_path + str(i) + '.mp4')
                success = vidcap.read()[0]  # gets if video was succesfully read and image value
                image_file = image_path + str(image_num) + ".jpg"  # set new image file path and name

                if not success:  # if file not found
                    vidcap = cv2.VideoCapture(timelapse_path + str(i) + '.mpg')  # change to mpg
                    success = vidcap.read()[0]  # gets if video was succesfully read and image value
                if not success:
                    vidcap = cv2.VideoCapture(timelapse_path + str(i) + '.mkv')  # change to mkv
                    success = vidcap.read()[0]  # gets if video was succesfully read and image value
                if success:
                    vidcap.set(cv2.CAP_PROP_POS_MSEC, vid_time * 1000)
                    print("writing image: " + str(image_num) + '.jpg')
                    image = vidcap.read()[1]
                    cv2.imwrite(image_file, image)  # save frame as JPEG file
                else:
                    print("NOT SUCCESSFUL: ", )
                vid_time = vid_time + int(interval_array[i])
                print("Vid Time = ", vid_time)
                image_num = image_num + 1
        # TODO: Error handling again
        except Exception as e:
            print(e)
    return image_num


def foo(x: int, y: str) -> str:
    print(x,y)
    return "%s %s" % (y, x)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video-folder', default=VIDEO_FOLDER, help='The folder where the videos are saved.',
                        required=True)
    parser.add_argument('-i', '--image-folder', default=IMAGE_FOLDER, help='Image save folder', required=True)
    args = parser.parse_args()

    VIDEO_FOLDER = args.video_folder
    IMAGE_FOLDER = args.image_folder

    formats = ['.mp4', '.mpg', '.mkv']

    print(os.path.abspath(VIDEO_FOLDER))

    extract_video_frames(
        len(list(filter(lambda filename: any(ext in filename for ext in formats), os.listdir(VIDEO_FOLDER)))))
