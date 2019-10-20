import zipfile
import os
import io

import cv2
import numpy as np

from cv2 import IMREAD_COLOR, IMREAD_GRAYSCALE, IMREAD_UNCHANGED

mode_list = {
    'RGB': IMREAD_COLOR,
    'BGR': IMREAD_COLOR,
    'P': IMREAD_UNCHANGED
}

class ZipReader(object):
    zip_origin_bank = dict()
    zip_bank = dict()
    zip_filelist_bank = dict()

    def __init__(self):
        super(ZipReader, self).__init__()

    @staticmethod
    def get_zipfile(path):
        zip_bank = ZipReader.zip_bank
        zip_origin_bank = ZipReader.zip_origin_bank
        if path in zip_bank:
            return zip_bank[path]
        else:
            # print("creating new zip_bank")
            # zfile = zipfile.ZipFile(path, 'r')
            # TODO: a tmp fixup way
            # https://discuss.pytorch.org/t/dataloader-with-zipfile-failed/42795
            with open(path, 'rb') as f:
                zip_origin_bank[path] = f.read()
            zfile = zipfile.ZipFile(io.BytesIO(zip_origin_bank[path]), 'r')
            zip_bank[path] = zfile
            return zip_bank[path]

    @staticmethod
    def split_zip_style_path(path):
        pos_at = path.index('@')
        if pos_at == -1:
            print("character '@' is not found from the given path '%s'" %
                  (path))
            assert 0
        zip_path = path[0:pos_at]
        folder_path = path[pos_at + 1:]
        folder_path = str.strip(folder_path, '/')
        return zip_path, folder_path

    @staticmethod
    def exist_file(path):
        zip_path, file_path = ZipReader.split_zip_style_path(path)

        for file_name in ZipReader.zip_namelist(zip_path):
            if file_path == file_name:
                return True
        return False

    @staticmethod
    def list_files(path):
        zip_path, folder_path = ZipReader.split_zip_style_path(path)
        file_lists = []

        for file_folder_name in ZipReader.zip_namelist(zip_path):
            if folder_path in file_folder_name:
                file_path = file_folder_name.split(folder_path + '/')[1]
                if file_path != '':
                    file_lists.append(file_path)
        return file_lists

    @staticmethod
    def zip_namelist(zip_path):
        zip_filelist_bank = ZipReader.zip_filelist_bank
        if zip_path in zip_filelist_bank:
            return zip_filelist_bank[zip_path]
        else:
            zfile = ZipReader.get_zipfile(zip_path)
            zip_filelist_bank[zip_path] = zfile.namelist()
            return zip_filelist_bank[zip_path]

    @staticmethod
    def imread(path, mode):
        zip_path, path_img = ZipReader.split_zip_style_path(path)
        zfile = ZipReader.get_zipfile(zip_path)
        data = zfile.read(path_img)
        im = cv2.imdecode(np.frombuffer(data, np.uint8), mode_list[mode])
        return im

    @staticmethod
    def read(path):
        zip_path, path_img = ZipReader.split_zip_style_path(path)
        zfile = ZipReader.get_zipfile(zip_path)
        data = zfile.read(path_img)
        return data
