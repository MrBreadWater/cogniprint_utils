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
import glob
import numpy as np
from typing import List
from multiprocessing import Pool

VIDEO_FOLDER = 'timelapses/'
IMAGE_FOLDER = 'images/'
DEBUG = False


def calc_frames(seconds: int, multiplier:float = 0.5):
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


def gen_frame_count(*args):
    i, file = args

    if DEBUG:
        print("\rGenerating values for timelapse #", i + 1, " of ", timelapse_count, end="")

    try:
        clip = cv2.VideoCapture(file)

        # Calculate frames
        return int(clip.get(cv2.CAP_PROP_FRAME_COUNT)), int(clip.get(cv2.CAP_PROP_FPS))

    # TODO: Use real error handling. (Seriously what the fuck dude.)
    except Exception as e:
        print("Exception gen_frame_counts! Fix, or your results are fucked!")
        print(e)


def gen_frame_counts(files: List[str], pool: Pool):
    """Gets frame counts for each timelapse

    Why is it called "gen" frame count? Who the fuck knows.

    Args:
        timelapse_count (int): Number of timelapses. Also corresponds the timelapse names for some reason.

    Returns:
        2-tuple of numpy.arrays: Arrays containing the number of frames per video and the length of each video in seconds
    """

    print("Generating FrameCount array")
    time_start = time.time()

    video_info = pool.map(gen_frame_count, enumerate(files))
    video_frame_nums,  video_fps_values = list(zip(*video_info))

    video_times = np.divide(video_frame_nums, video_fps_values)

    frames_array = calc_frames(video_times)  # calculate array for # of extracted frames per video

    if DEBUG:
        print("video fps vals = ", video_fps_values, "\n video frame nums = ", video_frame_nums)
        print("Video times = ", video_times)
        print("Generated array in %s seconds!" % round(time.time() - time_start, 2))

    return frames_array, video_times


def gen_times_array(timelapse_count: int, pool: Pool):
    '''Gets the frequency in seconds at which extractions should occur

    Args:
        timelapse_count (int): Number of timelapses. Also corresponds the timelapse names for some reason.

    Returns:
        2-tuple of numpy.arrays: Arrays containing the frequency in seconds at which extractions should occur and the length of each video
    '''

    frames_array, video_times = gen_frame_counts(timelapse_count, pool)  # get value arrays for videos

    time_start = time.time()  # Timing it for some reason. Think I wanted to test how fast using numpy made it?

    print("Generating Interval Array... This may take some time.")

    interval_array = np.divide(video_times, frames_array)  # generate extraction interval

    if DEBUG:
        print("Generated interval array in ", str(round(time.time() - time_start, 2)), "seconds!")
        print(frames_array, video_times)
        print("Interval Array = ", interval_array)

    return interval_array, frames_array

def extract_single_video_frames()

def extract_video_frames(files: List[str], pool: Pool):
    '''Extracts video frames into apropriate folders

    Args:
        files (int): Number of timelapses. Also corresponds the timelapse names for some reason.

    Returns:
        int: number of extracted images
    '''

    timelapse_count = len(files)

    interval_array, loop_counts = gen_times_array(timelapse_count, pool)

    image_num = 0

    print(timelapse_count, "timelapses to extract from.")
    print("Extracting images... this will take some time.")

    for i, file, interval, loop_count in enumerate(zip(files, interval_array, loop_counts)):
        print("Extracting images for timelapse #", i, " of ", timelapse_count)
        gc.collect()  # collect and remove unused variables
        vid_time = 0  # reset time for cv2 to grab frame
        vidcap = cv2.VideoCapture(file)
        success = vidcap.read()[0]  # gets if video was succesfully read and image value
        if success:
            for j in range(loop_count):
                image_file = "%s/%s.jpg" % (image_path, file.split('/')[-1])  # set new image file path and name
                vidcap.set(cv2.CAP_PROP_POS_MSEC, vid_time * 1000)
                if DEBUG:
                    print("writing image: " + str(image_num) + '.jpg')
                image = vidcap.read()[1]
                cv2.imwrite(image_file, image)  # save frame as JPEG file
                vid_time += int(interval)
                image_num += 1

    return image_num



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video-folder', default=VIDEO_FOLDER, help='The folder where the videos are saved.',
                        required=True)
    parser.add_argument('-i', '--image-folder', default=IMAGE_FOLDER, help='Image save folder', required=True)
    parser.add_argument('-d', '--debug', default=DEBUG, help='Verbosity argument, but -v was already taken so.')
    args = parser.parse_args()

    VIDEO_FOLDER = args.video_folder
    DEBUG = args.debug
    IMAGE_FOLDER = args.image_folder

    timelapse_path = os.path.abspath(VIDEO_FOLDER) + '/'
    image_path = os.path.abspath(IMAGE_FOLDER)

    files = glob.glob(timelapse_path+"*.mp4")
    files.extend(glob.glob(timelapse_path+"*.mpg"))
    files.extend(glob.glob(timelapse_path+"*.mov"))
    files.extend(glob.glob(timelapse_path+"*.mkv"))

    pool = Pool(16)
    print(os.path.abspath(VIDEO_FOLDER))

    extract_video_frames(files, pool)
