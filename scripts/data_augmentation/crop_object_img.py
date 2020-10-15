from PIL import Image
import os.path
import glob

def convertjpg(pngfile,outdir,width=640,height=480):
  img=Image.open(jpgfile)
  width = img.size[0]
  height = img.size[1]
  while(width > 160) | (height > 100) :
      # print(width)
      # print(height)
      img = img.resize((int(width/2), int(height/2)),Image.BILINEAR)
      width = img.size[0]
      height = img.size[1]
  print(width)
  print(height)
  img.save(os.path.join(outdir, os.path.basename(pngfile)))
  # try:
  #   new_img=img.resize((width,height),Image.BILINEAR)
  #   new_img.save(os.path.join(outdir,os.path.basename(pngfile)))
  # except Exception as e:
  #   print(e)


for jpgfile in glob.glob("/media/shenyl/Elements/sweeper/dataset/0716/objects/dogshit/*.png"):
  convertjpg(jpgfile, "/media/shenyl/Elements/sweeper/dataset/0716/objects/dogshit_crop")