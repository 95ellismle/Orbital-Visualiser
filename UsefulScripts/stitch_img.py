"""
Can be used to stitch frames into a movie if required.
"""

import os
import random
import numpy as np
import re

# Settings
img_folder = "."
ffmpeg_binary = "ffmpeg"
movie_length = 25
movie_name = "movie.mp4"

img_ext = "jpg"
img_prefix = "frame"


# init
randStr = ''.join(random.sample("absdefghijklmnopqrstu1234567890", 20))
all_img = [i for i in os.listdir(img_folder) if f'.{img_ext}' in i]
name_map = {}

if not all_img:
    raise SystemExit("I couldn't find the images you were looking for. Check the extension is correct!")

# Sort the files
if f'.{img_ext}' in all_img[0] and img_prefix in all_img[0]:
    fnums = [re.findall("[0-9]+[0-9.]*", i) for i in all_img]
    all_nums = []
    for i, nums in enumerate(fnums):
        if len(nums) != 1:
            raise SystemExit(f"Can't find the right amount of numbers in file {all_img[i]}")
        else:
            all_nums.append(float(nums[0]))
    all_img = [i[1] for i in sorted(zip(all_nums, all_img))]

if not all_img:
    raise SystemExit("Can't find any images!")

# Make sure we aren't overwriting files etc...
all_files = []
num_padding = int(np.ceil(np.log10(len(all_img) + 1)))
for i, f in enumerate(all_img):
    new_name = "%s_%s.%s" % (randStr, str(i).zfill(num_padding), img_ext)
    os.rename(f, new_name)
    name_map[new_name] = f
    all_files.append(new_name)


# Now actually rename them
final_files = []
for i, f in enumerate(all_files):
    new_name = "%s.%s" % (str(i).zfill(num_padding), img_ext)
    os.rename(f, new_name)
    name_map[new_name] = name_map[f]
    final_files.append(new_name)

if not all_files:
    raise SystemExit("Lost the files!")

# Now stitch the files
framerate = int(len(all_files) / movie_length)
# Pad string will add a pixel of width or height if needed as libx264 doesn't allow them to be odd integers
pad_string = '-vf "pad=ceil(iw/2)*2:ceil(ih/2)*2"'
ffmpeg_cmd = "%s -f image2 -framerate %i -i %s/%%0%id.%s -vcodec libx264 -acodec aac -pix_fmt yuv420p %s -preset slow %s" % (ffmpeg_binary,
                        framerate,
                        img_folder,
                        num_padding,
                        img_ext,
                        pad_string,
                        movie_name)

print("\n\n\n\nFFMPEG COMMAND = %s\n\n\n\n" % ffmpeg_cmd)
os.system(ffmpeg_cmd)


# Change the name back to the original ones
for f in final_files:
    orig_name = name_map[f]
    os.rename(f, orig_name)
