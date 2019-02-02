MOV_LENGTH='15'           # Movie length in seconds
INPUTMOVIE='../Docs/Perm_img/Plane2.mp4'           # Which file to slow down
OUTPUTNAME='./test.mp4'           # Name of output file



FFMPEGBINARY=ffmpeg        # Location of the ffmpeg binary (to slow down movie)



## No need to change things beyond this point ##
#########################################################################################


NUMFRAMES=`ffmpeg -i $INPUTMOVIE -map 0:v:0 -c copy -f null -y /dev/null 2>&1 | grep -Eo 'frame= *[0-9]+ *' | grep -Eo '[0-9]+' | tail -1`
FRAMERATE=`echo "print(int($NUMFRAMES/$MOV_LENGTH))" | python` 

$FFMPEGBINARY -r $FRAMERATE -i $INPUTMOVIE -y $OUTPUTNAME
