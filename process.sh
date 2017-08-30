# Process new training data

FOLDER=$1
#cat $FOLDER/driving_log.csv >> driving_data/driving_log.csv
mogrify -crop 320x80+0+60 $FOLDER/IMG/*.jpg
mogrify -resize 80x80\! $FOLDER/IMG/*.jpg
