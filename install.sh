#!/bin/bash
CURDIR="$(pwd)"
    if [ ! -d "$CURDIR/dataset/" ]
    then
	echo "Pulling training and test set into $CURDIR/dataset/"
	newDir=$CURDIR/dataset/
	mkdir -p $newDir
	cd $newDir
	wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ihmLsGFHgVfTAH5QJwTjJPBz-srGkpXB' -O input_data.txt
	wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yZr01IaybHRCCoDC8LcVqpVyYri_hA22' -O input_test_data.txt
	wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ch1gDKhxF9y7v6ZksOxxBw7EL5svOOiM' -O output_test_data.txt
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1yK41CLkd9gdjYPmUA5qDtcICFU_XrFzk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1yK41CLkd9gdjYPmUA5qDtcICFU_XrFzk" -O output_data.txt && rm -rf /tmp/cookies.txt
    else
	echo "Folder $CURDIR/dataset/ exist. If no data exist in that folder remove it and rerun this script"
    fi

