
import os
import zipfile
os.system('kaggle datasets download -d uciml/red-wine-quality-cortez-et-al-2009')
zipfile.ZipFile('red-wine-quality-cortez-et-al-2009.zip').extract('winequality-red.csv')
