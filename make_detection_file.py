import glob
import os
import json



ANNOTATION_PATH='/Users/jiangts/Documents/stanford/cs231n/final_project/semantic_annotations'
IMAGE_PATH='/Users/jiangts/Documents/stanford/cs231n/final_project/combined'

def get_img_path(annot_path):
    base=os.path.basename(annot_path)
    num,ext = os.path.splitext(base)
    return os.path.join(IMAGE_PATH, num+'.jpg')




label_mapping = {
        "Advertisement": 0,
        "Background Image": 1,
        "Bottom Navigation": 2,
        "Button Bar": 3,
        "Card": 4,
        "Checkbox": 5,
        "Date Picker": 6,
        "Drawer": 7,
        "Icon": 8,
        "Image": 9,
        "Input": 10,
        "List Item": 11,
        "Map View": 12,
        "Modal": 13,
        "Multi-Tab": 14,
        "Number Stepper": 15,
        "On/Off Switch": 16,
        "Pager Indicator": 17,
        "Radio Button": 18,
        "Slider": 19,
        "Text": 20,
        "Text Button": 21,
        "Text Buttonn": 22,
        "Toolbar": 23,
        "Video": 24,
        "Web View": 25
        }


def get_boxes(annot, boxes):
    if isinstance(annot, dict):
        if 'componentLabel' in annot and 'bounds' in annot:
            boxes.append({'bounds': annot['bounds'], 'label': annot['componentLabel']})
        if 'children' in annot:
            get_boxes(annot['children'], boxes)
    elif isinstance(annot, list):
        for child in annot:
            get_boxes(child, boxes)
    return boxes


# https://github.com/qqwweee/keras-yolo3
# Row format: image_file_path box1 box2 ... boxN;
# Box format: x_min,y_min,x_max,y_max,class_id (no space).
# path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3

def yolov3_line(img_path, boxes):
    row = [img_path]
    for box in boxes:
        b = map(str, box['bounds'])
        l = box['label']
        s = ','.join(b) + ',' + str(label_mapping[l])
        row.append(s)
    return ' '.join(row)


# https://github.com/kbardool/keras-frcnn
# filepath,x1,y1,x2,y2,class_name
# /data/imgs/img_001.jpg,837,346,981,456,cow
def frcnn_line(img_path, boxes):
    rows = []
    for box in boxes:
        b = map(str, box['bounds'])
        l = box['label']
        s = img_path + ',' + ','.join(b) + ',' + l
        rows.append(s)
    return '\n'.join(rows)


def classify_line(img_path, boxes):
    rows = []

    base=os.path.basename(img_path)
    num,ext = os.path.splitext(base)
    img_path = 'gs://ui-scene-seg_training/data/combined/combined/'+ num+'.jpg'

    for box in boxes:
        b = map(str, box['bounds'])
        l = box['label']
        s = img_path + ',' + ','.join(b) + ',' + str(label_mapping[l])
        rows.append(s)
    return '\n'.join(rows)


def process(path, yolov3, frcnn, classify):
    boxes = []
    img_path = get_img_path(path)

    with open(path) as f:
        annot = f.read()
        annot = json.loads(annot)
        boxes = get_boxes(annot, [])
        yolov3.write(yolov3_line(img_path, boxes)+'\n')
        frcnn.write(frcnn_line(img_path, boxes)+'\n')
        classify.write(classify_line(img_path, boxes)+'\n')



files = glob.glob(os.path.join(ANNOTATION_PATH, './*json'))

yolov3 = open('yolov3_annot.txt', 'a')
frcnn = open('frcnn_annot.txt', 'a')
classify = open('classify.txt', 'a')

for i, path in enumerate(files):
    process(path, yolov3, frcnn, classify)
    print('processed', i)

yolov3.close()
frcnn.close()
classify.close()
