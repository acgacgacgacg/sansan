import numpy as np
import csv

# Convert image data to numpy arrays and store for convenience
# Input: path: string where data is stored
#		 s: 'train' or 'test'
def importData(path, s='train'):
	#some variables
	DataH=[]
	DataV=[]
	labelsH=[]
	labelsV=[]
	#import training data from images
	with open(path+s+'.csv', 'rb') as trainCsv:
		reader=csv.DictReader(trainCsv)
		for row in reader:
			img=cv2.imread(imgPath+row['filename'])

			# get crop position
			top=int(row['top'])
			bottom=int(row['bottom'])
			left=int(row['left'])
			right=int(row['right'])

			# get labels
			if s =='train':
				company_name=int(row['company_name'])
				full_name=int(row['full_name'])
				position_name=int(row['position_name'])
				address=int(row['address'])
				phone_number=int(row['phone_number'])
				fax=int(row['fax'])
				mobile=int(row['mobile'])
				email=int(row['email'])
				url=int(row['url'])

			# crop image
			cimg=img[top:bottom, left:right]

			# horizontal or vertical?
			if bottom - top < right - left:
				#resize: mean size of H: 341, 39
				cimg=cv2.resize(cimg, (341, 39))
				DataH.append(cimg.reshape([39*341, 3]))
				if s == 'train':
					labelsH.append([company_name, full_name, position_name, address, phone_number, fax, mobile, email, url])
			else:
				#resize: mean size of H: 50, 299
				cimg=cv2.resize(cimg, (50, 299))
				DataV.append(cimg.reshape([299*50, 3]))
				if s == 'train':
					labelsV.append([company_name, full_name, position_name, address, phone_number, fax, mobile, email, url])

	np.save(path+s+'DataH.npy', DataH)
	np.save(path+s+'DataV.npy', DataV)

	if s == 'train':
		np.save(path+s+'LabelsH.npy', labelsH)
		np.save(path+s+'LabelsV.npy', labelsV)