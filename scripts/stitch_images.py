import os
import random
import numpy as np

# Settings
img_folder = "."
ffmpeg_binary = "ffmpeg"
movie_length = 30


# init
randStr = ''.join(random.sample("absdefghijklmnopqrstu1234567890", 20))
all_jpg = [i for i in os.listdir(img_folder) if '.jpg' in i]
name_map = {}

# Sort the files
if '_img.jpg' in all_jpg[0]:
    fnums = [float(i.replace("_img.jpg", "").replace(",", ".")) for i in all_jpg]
    all_jpg = [i[1] for i in sorted(zip(fnums, all_jpg))]


# Make sure we aren't overwriting files etc...
all_files = []
num_padding = int(np.ceil(np.log10(len(all_jpg) + 1)))
for i, f in enumerate(all_jpg):
    new_name = "%s_%s.jpg" % (randStr, str(i).zfill(num_padding))
    os.rename(f, new_name)
    name_map[new_name] = f
    all_files.append(new_name)


# Now actually rename them
final_files = []
for i, f in enumerate(all_files):
    new_name = "%s.jpg" % str(i).zfill(num_padding)
    os.rename(f, new_name)
    name_map[new_name] = name_map[f]
    final_files.append(new_name)


# Now stitch the files
framerate = int(len(all_files) / movie_length)
ffmpeg_cmd = "%s -f image2 -framerate %i -i %s/%%0%id.jpg -vcodec libx264 -acodec aac -pix_fmt yuv420p -preset slow test.mp4" % (ffmpeg_binary,
                                                                                                                                 framerate,
                                                                                                                                 img_folder,
                                                                                                                                 num_padding)
print("\n\n\n\nFFMPEG COMMAND = %s\n\n\n\n" % ffmpeg_cmd)
os.system(ffmpeg_cmd)


# Change the name back to the original ones
for f in final_files:
    orig_name = name_map[f]
    os.rename(f, orig_name)

