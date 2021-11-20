from helpers import max_cosine_images, max_cosine_text
try:
  print(max_cosine_text('2008-03-11_Bicyclist_in_Carrboro.jpg',[ 'мотоцикл','велосипед', 'крсное вино']))#велосипед
  _, _ ,path = max_cosine_images(['Prm_VJ_fig1_featureTypesWithAlpha.png','2008-03-11_Bicyclist_in_Carrboro.jpg'],'велосипед')
  print(path)
except:
  print('Для тестирования загрузите изображения по следующим ссылкам в директорию, где находится helpers.py')
  print("wget https://upload.wikimedia.org/wikipedia/commons/5/53/2008-03-11_Bicyclist_in_Carrboro.jpg ")
  print("wget https://upload.wikimedia.org/wikipedia/commons/2/2f/Prm_VJ_fig1_featureTypesWithAlpha.png")
