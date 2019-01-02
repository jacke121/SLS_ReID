import os
import cv2
import xml.etree.ElementTree as ET
def _load_pascal_annotation(xml_name,jpeg_name,path_limage = None):
    im = cv2.imread(jpeg_name)
    if im is None:
        return
    if im.shape[1]<1 or im.shape[0]<1:
        return
    tree = ET.parse(xml_name)
    file_objs = tree.findall('filename')
    # non_diff_objs = [objf for objf in file_objs if int(file_objs.find('difficult').text) == 0]
    filetext = file_objs[0].text
    objs = tree.findall('object')

    objs_diffi = tree.findall('object/difficult')
    if len(objs_diffi)==0:
        print(xml_name,objs_diffi,len(objs_diffi))
    # non_diff_objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
    # objs = non_diff_objs
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        if obj.find("difficult") is not None and obj.find("difficult").text == '1':
            continue
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = int(bbox.find('xmin').text)
        y1 = int(bbox.find('ymin').text)
        x2 = int(bbox.find('xmax').text)
        y2 = int(bbox.find('ymax').text)
        minY = max(y1, 0)
        maxY = min(y2, im.shape[0])
        minX = max(x1, 0)
        maxX = min(x2, im.shape[1])
        cropImg = im[minY:maxY, minX: maxX]
        name = os.path.basename(jpeg_name).split('.')[0]
        cv2.imwrite(os.path.join(path_limage,'%s_%d.jpg'%(name,ix)),cropImg)
if __name__ == '__main__':
    all_path = r'\\192.168.55.73\Team-CV\dataset\origin_all_datas_0813\_2test'
    all_files = [list(map(lambda x: os.path.join(root, x), files)) for root, _, files in
                             os.walk(all_path, topdown=False) if os.path.basename(root) == 'Annotations']
    label_files = []
    for i in range(len(all_files)):
        label_files += all_files[i]
    img_files = [file.replace('Annotations', 'JPEGImages').replace('xml', 'jpg') for file in
                                  label_files]

    assert len(img_files) == len(label_files),'error files'
    for i,label_file in enumerate(label_files):
        dir_path = os.path.dirname(label_files[i])
        path_limage = os.path.join(dir_path,'../tiny_pic')
        os.makedirs(path_limage,exist_ok=True)
        print(label_files[i])
        _load_pascal_annotation(label_files[i], img_files[i], path_limage)
