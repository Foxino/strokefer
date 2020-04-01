A quick run down of the scripts within this folder, because it's kinda a mess

the main application is FaceEmotionDetection.py, this can be ran with two parameters [webcam] [fp], running normally without parameters will use
the unmodified dataset and detect 7 emotions from a demo video, including webcam (python FaceEmotionDetection.py webcam) will switch this to the webcam
as source, to use the modified dataset you will have to state fp as the last parameter (python FaceEmotionDetection.py webcam fp or python FaceEmotionDetection.py demo fp)

Note that demo is not needed to run the video but something will need to be put into the parameter as this is just a quickly thrown together way of calling without editing the
code for debugging purposes.

Other scripts

imagesToFaces.py checks a source folder for pictures and extracts any faces found on them into the output

fetchFromYt.py takes videos from youtube, runs the facial classifier against them and pulls faces from them

deeplearn_fer.py the training script, training with fer2013 will train only with the 7 base emotions, using fer2013_fpalsy will include the facial palsy FaceEmotionDetection, ensure these go to their correct model_json

other stuff

alarm.h5, alarm.json - model and weights for 8 emotion

fer_without_fp.hs, fer_without_fp.json - model and weights for base 7 emotions

fer2013_fpalsy - modified dataset

fer2013 - unmodified dataset

The folder facialpalsy_extendedDataSet contains the images before entering the .csv with a script imagestocsv.py to enter them into the .read_csv
