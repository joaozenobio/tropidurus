import glob
import os
import random

train_size = 0.9

images = [filename for filename in glob.glob("images/*.jpg")]
random.shuffle(images)
train = images[0:int(len(images) * train_size)]
test = images[int(len(images) * train_size):]
with open('train.txt', 'w') as file:
	for filename in train:
		file.write(filename + "\n")
with open('test.txt', 'w') as file:
	for filename in test:
		file.write(filename + "\n")
