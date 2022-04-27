import os
import sys
import argparse
import shutil
import xml.dom.minidom
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import random


class Cvat2ImageNet(object):
    def __init__(self, cvat_label_file, classes_file, output_dir):

        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        print("Parsing XML...")
        image_label_name = []
        tree = ET.parse(cvat_label_file)
        root = tree.getroot()
        for node in root:
            if node.tag == "image":
                image_name = node.attrib["name"]
                for item in node:
                    label_name = item.attrib["label"]
                image_label_name.append("{} {}".format(image_name, label_name))
        print("Parse XML Done.\n")
    
        with open(classes_file, 'r') as f:
            self.classes = f.readlines()
        categories_id = {}
        for c in self.classes:
            c = c.strip()
            categories_id[c] = len(categories_id)
        
        # Print categories
        print("Categories:")
        for c, _id in categories_id.items():
            print("{}: {}".format(c, _id))
        print("")
        # Convert label name to label id
        self.image_label_id = []
        for item in image_label_name:
            image_name, label_name = item.split(" ")
            self.image_label_id.append("{} {}".format(image_name, categories_id[label_name]))

    def split_data(self, train, valid, shuffle=True):
        if train + valid == 1.0:
            test = 0.0
        else:
            test = 1 - train - valid

        # shuffle image_label
        if shuffle:
            random.shuffle(self.image_label_id)

        train_num = int(len(self.image_label_id) * train)
        valid_num = int(len(self.image_label_id) * valid)
        test_num = len(self.image_label_id) - train_num - valid_num
        print("Total:{}".format(len(self.image_label_id)))
        print("Train:{}, Valid:{}, Test:{}".format(train_num, valid_num, test_num))

        self.train_data = self.image_label_id[:train_num] # train
        self.valid_data = self.image_label_id[train_num: train_num + valid_num] # valid
        if test != 0.0:
            self.test_data = self.image_label_id[train_num + valid_num: ]
        else:
            test = []
        with open(os.path.join(self.output_dir, "train.txt"), 'w') as f1:
            for item in self.train_data:
                f1.writelines("{}{}".format(item, "\n"))
        with open(os.path.join(self.output_dir, "valid.txt"), 'w') as f2:
            for item in self.valid_data:
                f2.writelines("{}{}".format(item, "\n"))
        with open(os.path.join(self.output_dir, "test.txt"), 'w') as f3:
            for item in self.test_data:
                f3.writelines("{}{}".format(item, "\n"))
    
    def copy_images(self, image_dir):
        print("Copy images...")
        os.makedirs(os.path.join(self.output_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "valid"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "test"), exist_ok=True)
        
        class_id2name = {}
        for item in self.classes:
            item = item.strip()
            class_id2name[len(class_id2name)] = item
            os.makedirs(os.path.join(self.output_dir+"/train", item), exist_ok=True)
        print(class_id2name)
        for train_image_labelid in self.train_data:
            image_name, label_id = train_image_labelid.split(" ")
            src_image = os.path.join(image_dir, image_name)
            dst_image = os.path.join(
                                     os.path.join(self.output_dir+"/train", class_id2name[int(label_id)]),
                                     image_name
            )
            shutil.copy(src_image, dst_image)

        for item in self.valid_data:
            image_name, label_id = item.split(" ")
            src_image = os.path.join(image_dir, image_name)
            dst_image = os.path.join(self.output_dir, "valid", image_name)
            shutil.copy(src_image, dst_image)
        for item in self.test_data:
            image_name, label_id = item.split(" ")
            src_image = os.path.join(image_dir, image_name)
            dst_image = os.path.join(self.output_dir, "test", image_name)
            shutil.copy(src_image, dst_image)
        print("Copy images Done.\n")


parser = argparse.ArgumentParser()
parser.add_argument("--cvat", type=str, default="cvat_data/single_door/annotations.xml", 
                    help="CVAT label file")
parser.add_argument("--classes", type=str, default="data/classes.txt",
                    help="classes file, each line for one class name")
parser.add_argument("--train", type=float, default=0.7,
                    help="train data percent, between 0 and 1")
parser.add_argument("--valid", type=float, default=0.2,
                    help="valid data percent, betwenn 0 and 1")
parser.add_argument("--image-dir", type=str, default="cvat_data/single_door",
                    help="cvat image directory")
parser.add_argument("--output-dir", type=str, default="data/single_door",
                    help="output directory of images in ImageNet format")
args = parser.parse_args()

annotations_file = args.cvat
classes_file = args.classes
output_dir = args.output_dir

cvat_label_info = Cvat2ImageNet(annotations_file, classes_file, output_dir)
cvat_label_info.split_data(args.train, args.valid)
cvat_label_info.copy_images(args.image_dir)

