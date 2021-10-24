from helpers import max_cosine_images, max_cosine_text
import os
try:
  os.system("wget https://upload.wikimedia.org/wikipedia/commons/5/53/2008-03-11_Bicyclist_in_Carrboro.jpg ")
  os.system("wget https://upload.wikimedia.org/wikipedia/commons/2/2f/Prm_VJ_fig1_featureTypesWithAlpha.png")
except:
  print('что то пошло не так')

print(max_cosine_text('2008-03-11_Bicyclist_in_Carrboro.jpg',[ 'мотоцикл','велосипед', 'крсное вино']))#велосипед

_, _ ,path = max_cosine_images(['/content/Prm_VJ_fig1_featureTypesWithAlpha.png','/content/2008-03-11_Bicyclist_in_Carrboro.jpg'],'велосипед')

print(path)

