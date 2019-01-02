
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import shutil
import traceback


ext='jpg|jpeg|bmp|png|ppm'


def gen_data(rootDir):
    global mouse_count,mouse_index
    for root,dirs,files in os.walk(rootDir):
        if len(files)>0:
            mouse_count+=1
            mouse_index=0
        for file in files:
            mouse_index+=1
            if re.match(r'([\w]+\.(?:' + ext + '))', file):
                ori_file=os.path.join(root,file)
                new_file=os.path.join(root,'%03d_%03d.jpg' % (mouse_count,mouse_index))
                os.rename(ori_file, new_file)
                print("....",new_file)
        for dir in dirs:
            if "JPEGImages" in dir or "Annotations" in dir :
                try:
                    for c_root, dirs, files in os.walk(dir):
                        for name in files:
                            # delete the log and test result
                            del_file = os.path.join(c_root, name)
                            os.remove(del_file)
                    shutil.rmtree(os.path.join(root, dir))
                except Exception as e:
                    traceback.print_exc()
            else:
                gen_data(dir)

def list_data(rootDir):
    global mouse_count,mouse_index
    for root,dirs,files in os.walk(rootDir):
        for file in files:
                ori_file=os.path.join(root,file)
                print(ori_file)
        for dir in dirs:
            list_data(dir)
if __name__ == '__main__':
    mouse_count = -1
    mouse_index = 0
    path = r"\\192.168.55.73\Team-CV\dataset\origin_all_datas_0814\_2train"
    gen_data(path)
    # list_data(path)
    print("mouse_count",mouse_count+1)
    # path = r"\\192.168.55.73\Team-CV\dataset\origin_all_datas_0814\_2train\sh_wd\two_mouse\tiny_pic_two"
    # files= os.listdir(path)
    # mouse_count=0
    # for file in files:
    #     ori_file = os.path.join(path, file)
    #     new_file = os.path.join(path, '%03d_%03d.jpg' % (0, mouse_index))
    #     print(new_file)
    #     os.rename(ori_file, new_file)
    #     mouse_index+=1


# print(sorted([os.path.join(root, f)
#                for root, _, files in os.walk(path) for f in files
#                if re.match(r'([\w]+\.(?:' + ext + '))', f)]))