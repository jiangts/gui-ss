import glob
import mmap

files = glob.glob('./*json')

labels = dict()
def process(path):
    with open(path) as f:
        # s = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        # if s.find('componentLabel') != -1:
        #     print('true')
        datafile = f.readlines()
        for line in datafile:
            if 'componentLabel' in line:
                labels[line.strip(' ,\t')] = 1

for path in files:
    lcnt = len(labels)
    process(path)
    if len(labels) > lcnt:
        print(labels)


{'Multi-Tab': 1,
 'Modal': 1,
 'Drawer': 1,
 'Bottom Navigation': 1,
 'Input': 1,
 'Icon': 1,
 'Date Picker': 1,
 'Button Bar': 1,
 'Text Buttonn': 1,
 'Image': 1,
 'Checkbox': 1,
 'Text': 1,
 'List Item': 1,
 'Web View': 1,
 'On/Off Switch': 1,
 'Text Button': 1,
 'Card': 1,
 'Advertisement': 1,
 'Pager Indicator': 1,
 'Radio Button': 1,
 'Video': 1,
 'Toolbar': 1,
 'Background Image': 1,
 'Slider': 1,
 'Number Stepper': 1,
 'Map View': 1}

