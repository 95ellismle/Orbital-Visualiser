new_fold="all_ad_states"
adState_data_folds="/home/matt/Data/Work/NewPentaceneSlab/24_25_5/FSSH_TMP/state_data"

# Create the folder
if [ -d "$new_fold" ]
then
  rm -rf $new_fold

fi
mkdir $new_fold


escape_for_sed() {
  # Will print a string that has all special characters escaped with \
  KEYWORD="$1";
  printf '%s\n' "$KEYWORD" | sed -e 's/[]\/$*.^[]/\\&/g';
}



#all_folds=`find /home/matt/Data/Work/restraint_CP2K/AdStates/run-fssh-0/state_data -name "AdState_*"`
for state_num in `seq 0 200`
do
  fold="$adState_data_folds/AdState_$state_num"
  esc_fold=`escape_for_sed $fold`

  sed s/"ST_FOLDER"/"$esc_fold"/ settingsO.inp > settings.inp


  state_num=`python -c "print('$fold'['$fold'.rfind('/'):].replace('AdState_', '')[1:])"`
  python3 main.py


  mv img/Calibration/0,00_img.jpg $new_fold/$state_num.jpg
done
