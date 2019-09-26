Instruction:
For detecting exit sign in a image need two
arguments. The first one is the file name of the testing image
in the test_img folder. The second one is the flag for using
red color red color extraction. If it is '1' then the program
extract the red from the testing image, if it is '0' then the program
uses the testing image without extract the red
example:
'python find_object_location.py test1_1.jpg 1'
In the above example it runs the program on testing image test1_1.jpg
and extracts the red color from the image before processing.

In red color extraction mode, the program has 5 stages:
original image -> red image -> grey image -> edge image -> result image
In the regular mode, the program has 4 stages:
original image -> gray image -> edge image -> result image
each stage will show a corresponding image, press any key to enter the
next stage.
Results of each stage will be saved in the corresponding folder.
While running pixels which are being processed will be print out,
after done running the contour image of E, X and the result image will be
shown.


For processing a video, it needs one arguments.

example:
'python process_video.py testing_video.mp4'
While running pixels which are being processed will be print out,
after done running the result video will be saved in the video_result folder