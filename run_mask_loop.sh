#!/bin/bash

rm images/positivos/*.jpg
rm images/infractores/*.jpg

while [ True ];
do
  echo "Capturando imagen..."
  ./take_snap.sh
  echo "Analizando imagen..."
  python3 inference.py -i images/cam_image.jpg -m mask_classifier.tflite
done