import matplotlib.pyplot as plt
import cv2
import easyocr
import sys
import random
import os
#from pylab import rcParams
#from IPython.display import Image
#rcParams['figure.figsize'] = 8, 16

reader = easyocr.Reader(['en'])

#print(sys.argv[1])

#Image((sys.argv[1]))

output = reader.readtext(sys.argv[1])
#print(output)
list1 = []

image = cv2.imread(sys.argv[1])
for i in range(0,len(output)):
    cord = output[i][0]
    x_min, y_min = [int(min(idx)) for idx in zip(*cord)]
    x_max, y_max = [int(max(idx)) for idx in zip(*cord)]
    r, g, b = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) # Random Color
    cv2.rectangle(image,(x_min,y_min),(x_max,y_max),(r,g,b),2)
    cv2.putText(image, output[i][1], (x_min,y_min), cv2.FONT_HERSHEY_SIMPLEX, 2, (r,g,b), 2)
    list1.append(output[i][1])

path_file = os.getcwd()
with open('media/new.txt', 'w') as f:
    for item in list1:
        f.write("%s\n" % item)

""" cord = output[0][0]
print(cord)

x_min, y_min = [int(min(idx)) for idx in zip(*cord)]

x_max, y_max = [int(max(idx)) for idx in zip(*cord)] 

image = cv2.imread(sys.argv[1])
cv2.rectangle(image,(x_min,y_min),(x_max,y_max),(0,0,255),2)"""
print(list)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
path = 'media/1/'+sys.argv[2]
plt.imsave(path, image)
#plt.show()

