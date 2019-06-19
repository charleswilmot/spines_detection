## Dependencies

```
pip3 install --user tensorflow==1.5
pip3 install --user socket
...
```


## Generate the data *

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

```
