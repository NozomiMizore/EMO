# EMO
This project was developed and maintained by 徐志文(XuZhiwen), 王昊天(WangHaotian)

## Warning
The dataset(fer2013.csv 287MB) can't be uploaded because of the file size limit(100MB)

## Dir_tree
```
│  .gitignore
│  emo_rec.py
│  generator.py
│  gui.py
│  load_process.py
│  myui.ui
│  README.md
│  run.py
│  test.py
│  training.py
│  tree.txt
│  
├─data
│  └─fer2013
│          fer2013.csv
│          
├─haarcascade_files
│      haarcascade_frontalface_default.xml
│      
├─img
│  │  anger.jpg
│  │  background.png
│  │  camera.png
│  │  folder.png
│  │  icon.png
│  │  model.png
│  │  result.png
│  │  result_bg.png
│  │  result_bg_png.py
│  │  sad.jpg
│  │  scan.gif
│  │  smile.jpg
│  │  speed.png
│  │  
│  └─__pycache__
│          result_bg_png.cpython-37.pyc
│          
├─model
│  │  CNN.98-0.62.hdf5
│  │  cnn.py
│  │  CNN_training.log
│  │  miniXCEPTION_training.log
│  │  mini_XCEPTION.79-0.64.hdf5
│  │  
│  └─__pycache__
│          cnn.cpython-37.pyc
│          
└─__pycache__
        emo_rec.cpython-37.pyc
        gui.cpython-37.pyc
        load_process.cpython-37.pyc
        slice_png.cpython-37.pyc
```
## Result
- The mini_Xception model has reached 64.3% accuracy with validation
- The simpler_cnn model has reached 61.5% accuracy with validation

## Ref
The CNN model structure refers to https://github.com/oarriaga/face_classification