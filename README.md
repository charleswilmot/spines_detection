## Dependencies

```
pip3 install --user tensorflow==1.5
pip3 install --user socket
...
```


## Generate the datasets

```
python3 make_tf_records.py \
  -i /home/kaschube-shared/spines/SpineMeasurements_XYcoordinates/SR51N1/ \
  -l /home/kaschube-shared/spines/SpineMeasurements_XYcoordinates/SR51N1/SR51*eu*end*ay*.mat \
  --name SR51N1_fake_training \
  --image-size 32             \
  --cells-size-xy 8           \
  --cells-stride-xy 4         \
  --cells-size-z 4            \
  --cells-stride-z 2          \
  --cells-padding SAME        \

python3 make_tf_records.py \
  -i /home/kaschube-shared/spines/SpineMeasurements_XYcoordinates/SR51N1/ \
  -l /home/kaschube-shared/spines/SpineMeasurements_XYcoordinates/SR51N1/SR51*eu*end*ay*.mat \
  --name SR51N1_fake_training \
  --image-size 64             \
  --cells-size-xy 16          \
  --cells-stride-xy 8         \
  --cells-size-z 4            \
  --cells-stride-z 2          \
  --cells-padding SAME        \

python3 make_tf_records.py \
  -i /home/kaschube-shared/spines/SpineMeasurements_XYcoordinates/SR51N1/ \
  -l /home/kaschube-shared/spines/SpineMeasurements_XYcoordinates/SR51N1/SR51*eu*end*ay*.mat \
  --name SR51N1_fake_training \
  --image-size 128            \
  --cells-size-xy 32          \
  --cells-stride-xy 16        \
  --cells-size-z 4            \
  --cells-stride-z 2          \
  --cells-padding SAME        \

python3 make_tf_records.py \
  -i /home/kaschube-shared/spines/SpineMeasurements_XYcoordinates/SR51N1/ \
  -l /home/kaschube-shared/spines/SpineMeasurements_XYcoordinates/SR51N1/SR51*eu*end*ay*.mat \
  --name SR51N1_fake_training \
  --image-size 32             \
  --cells-size-xy 4           \
  --cells-stride-xy 2         \
  --cells-size-z 4            \
  --cells-stride-z 2          \
  --cells-padding SAME        \

python3 make_tf_records.py \
  -i /home/kaschube-shared/spines/SpineMeasurements_XYcoordinates/SR51N1/ \
  -l /home/kaschube-shared/spines/SpineMeasurements_XYcoordinates/SR51N1/SR51*eu*end*ay*.mat \
  --name SR51N1_fake_training \
  --image-size 64             \
  --cells-size-xy 8           \
  --cells-stride-xy 4         \
  --cells-size-z 4            \
  --cells-stride-z 2          \
  --cells-padding SAME        \

python3 make_tf_records.py \
  -i /home/kaschube-shared/spines/SpineMeasurements_XYcoordinates/SR51N1/ \
  -l /home/kaschube-shared/spines/SpineMeasurements_XYcoordinates/SR51N1/SR51*eu*end*ay*.mat \
  --name SR51N1_fake_training \
  --image-size 128            \
  --cells-size-xy 16          \
  --cells-stride-xy 8         \
  --cells-size-z 4            \
  --cells-stride-z 2          \
  --cells-padding SAME        \
```



## Start training

```
p3 training_local.py \
  ../tfrecords/../tfrecords/cell_size_004_004__cell_strides_002_002__padding_SAME__image_size_032/SR51N1_fake_training.tfrecord \
  --name test \
  --n-epochs 1
```
