from helpers import max_cosine_images, max_cosine_text
import os
os.system("wget https://upload.wikimedia.org/wikipedia/commons/5/53/2008-03-11_Bicyclist_in_Carrboro.jpg ")

print(max_cosine_text('2008-03-11_Bicyclist_in_Carrboro.jpg',[ 'мотоцикл','велосипед', 'крсное вино']))#велосипед
