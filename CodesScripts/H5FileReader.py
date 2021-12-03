import h5py
import os

import numpy as np


def _remove_prefix(text: str, prefix):
    if text.startswith(prefix):
        text = text[len(prefix):]
    return text


def _remove_from_char(text: str, marker, leave_front=True):
    """

    :param text:
    :param marker: marker is omitted
    :param leave_front:
    :return:
    """
    for i in range(len(text)):
        if text[i] == marker:
            if leave_front:
                text = text[:i]
            else:
                text = text[i + 1:]
            break
    return text


class H5FileReader:
    """An auxiliary class to help convert h5 data to nd array, with other relative information needed.

    Attributes:
        file_ndarray:    array-like, shape(param_num, file_length).
        index_alloc:     dict {parameter name, [parameter_num_start, parameter_num_end]}.
    """
    file_ndarray: np.array
    index_alloc: dict
    file_name: str
    f: h5py.File
    reftable: np.array

    def __init__(self):
        """
        Attributes:
            index_alloc: only initialized for one time at the first reading
            branch_idx_alloc:
        """
        self.file_length = 0
        self.parameter_num = 0
        # {array_name: [start_idx,end_idx]}, both indices are included
        self.index_alloc = {}
        # {array_name: {leaf_name: [start_idx, end_idx]}}. start and end should be identical
        self.branch_idx_alloc = {}
        # records father branches
        self.leaves = {}
        # {datax: offset}
        self.glob_datablockoffset = {}
        self.SAMPLE_FILE = 'example.h5'
        self.CLIP_OVERHEAD = 400  # cycles
        self.CLIP_TAIL = 200  # include event frame
        # self.EVENT_AGGREGATE_TIME = 100  # event frames with fewer distance than it will be aggregated. use OVERHEAD

    def read_file(self, file_name):
        """
            Convert h5 file into a simple nd-array. All data within are regarded as FLOAT.
            Current types in the Group: 0--float; 3,5,6,7--int.
            Data_start_idx should by default follow the lb_end_idx
        :param file_name:
        :return:
        """
        self.f = h5py.File(os.path.join(file_name), 'r')
        self.file_name = file_name
        self.file_length = len(self.f['data0'][0])
        self.reftable = self.f['reftable']
        self.parameter_num = len(self.reftable)
        self.file_ndarray = self._get_ndarray_type_order()
        if self.index_alloc == {}:
            # print("allocation initialized")
            self._get_index_alloc()
            self._get_branch_idx_alloc()
        # decode("utf-8")
        # print("Frames: ", self.file_length)
        # print("Indices of obj_mnvs: ", self.index_alloc.get("obj_mnvs"))
        # print("branch_idx_alloc: ", self.branch_idx_alloc.get('output_obj_daten').keys())
        return self.file_ndarray

    def _get_ndarray_of_idx_clip(self, start_idx, end_idx):
        """OBSOLETE
            Get data clip according to given index of parameters, as ndarray
        :param start_idx: index in h5py.File['reftable']
        :param end_idx: included
        :return: array-like, shape(end_idx - start_idx + 1, file_length)
        """
        # invalid index pair
        if start_idx > end_idx or start_idx < 0 or end_idx >= self.parameter_num:
            return np.zeros(1)
        clip_len = end_idx - start_idx + 1
        output = np.zeros((clip_len, self.file_length))
        for i in range(clip_len):
            data_branch = 'data{}'.format(self.reftable[start_idx + i][1].decode("utf-8"))
            data_blockoffset = self.reftable[start_idx + i][2]
            output[i] = self.f[data_branch][data_blockoffset].astype(float)
        return output

    def _get_ndarray_type_order(self):
        key_list = list(self.f.keys())
        file_ndarray = np.array(self.f[key_list[0]])
        self.glob_datablockoffset[key_list[0]] = 0
        for i in range(1, len(key_list) - 1):
            self.glob_datablockoffset[key_list[i]] = file_ndarray.shape[0]
            #file_ndarray = np.concatenate((file_ndarray, self.f[key_list[i]]), axis=0, dtype=float)
            file_ndarray = np.concatenate((file_ndarray, self.f[key_list[i]]), axis=0)
        return file_ndarray

    def _get_index_alloc(self):
        """
            Extract index dictionary from 'reftable' group. We assume that all parameter names are
            properly with whole [index] as part of an array
        :return: dict, {parameter name: [start, end]}
        """
        is_array_branch = False
        # index of current array entry
        cur_arr_idx = -1
        # start_idx = 0  # for array elements allocation
        # end_idx = 0
        # arr_name = " "
        last_arr_name = " "
        i = 0
        while i < self.parameter_num:
            entry_name = self.reftable[i][0].decode("utf-8")
            entry_name = _remove_prefix(entry_name, "PythonInterpreter.check_lc.return.")
            name_len = len(entry_name)
            is_array = False
            j = 0
            # check for array syntax
            while j < name_len - 2:
                # find the first element of an array
                if entry_name[j] == '[':
                    if entry_name[j + 2] == ']':
                        new_arr_idx = int(entry_name[j + 1])
                    else:
                        new_arr_idx = int(entry_name[j + 1: j + 3])
                    arr_name = _remove_from_char(entry_name, '[')
                    start_idx = i
                    end_idx = i
                    # initialize an array entry
                    if not self.index_alloc.__contains__(arr_name):
                        # if former entry is the end of a branch
                        if is_array_branch:
                            self.index_alloc[last_arr_name + str(cur_arr_idx)] = self.index_alloc.get(last_arr_name)
                            self.index_alloc.pop(last_arr_name)
                        self.index_alloc[arr_name] = [start_idx, end_idx]
                        cur_arr_idx = new_arr_idx
                        is_array_branch = False
                    # else only update the end_idx
                    else:
                        # if it is a branch, set the flag
                        if cur_arr_idx == new_arr_idx:
                            is_array_branch = True
                            self.index_alloc[arr_name] = [self.index_alloc.get(arr_name)[0], end_idx]
                        # if it is the next branch of same array, rename the covered branch
                        elif is_array_branch:
                            self.index_alloc[arr_name + str(cur_arr_idx)] = self.index_alloc.get(arr_name)
                            self.index_alloc[arr_name] = [start_idx, end_idx]
                            cur_arr_idx = new_arr_idx
                        # if it is normal array
                        else:
                            self.index_alloc[arr_name] = [self.index_alloc.get(arr_name)[0], end_idx]
                    last_arr_name = arr_name
                    is_array = True
                    break
                j += 1
            if not is_array:
                self.index_alloc[entry_name] = [i, i]
            i += 1
        if is_array_branch:
            self.index_alloc[last_arr_name + str(cur_arr_idx)] = self.index_alloc.get(last_arr_name)
            self.index_alloc.pop(last_arr_name)

    def _get_branch_idx_alloc(self):
        """Update index of branch arrays.

        Notes:
            assume that non-branch array name never ends with 0.

        Returns:

        """
        key_list = list(self.index_alloc.keys())
        i = 0
        while i < len(key_list):
            branch = key_list[i]
            # check for keys with names ended with 0, indicating it's branch array
            # ord('0') = 48, ord('9') = 57
            if 48 == ord(branch[len(branch) - 1]) and \
                    (ord(branch[len(branch) - 2]) < 48 or ord(branch[len(branch) - 2]) > 57):
                indices = self.index_alloc.get(branch)
                # ignore non-branch array names and covered branches
                if self.branch_idx_alloc.__contains__(branch):
                    i += 1
                    continue
                self.branch_idx_alloc[_remove_from_char(branch, '0')] = {}
                for idx in range(indices[0], indices[1] + 1):
                    entry_name = self.reftable[idx][0].decode("utf-8")
                    leaf_name = _remove_from_char(entry_name, ']', leave_front=False)
                    leaf_name = _remove_from_char(leaf_name, '.', leave_front=False)
                    branch_name = _remove_from_char(branch, '0')
                    self.leaves[leaf_name] = branch_name
                    #self.leaves.get(leaf_name).append(branch_name)
                    self.branch_idx_alloc.get(branch_name)[leaf_name] = \
                        [idx - indices[0], idx - indices[0]]
            i += 1

    def _get_blockinfo(self, reftable_idx):
        blocknum = self.reftable[reftable_idx][1].decode('utf-8')
        blockoffset = self.reftable[reftable_idx][2]
        return blocknum, blockoffset

    def _reset_is_auto_labeled(self):
        """
        Reset all is_auto_labeled_idx to 1
        :return:
        """
        is_auto_labeled_idx = self.index_alloc.get('is_auto_labeled')[0]
        (blocknum, blockoffset) = self._get_blockinfo(is_auto_labeled_idx)
        global_blockoffset = self.glob_datablockoffset.get('data{}'.format(blocknum))
        zero_frames = np.argwhere(self.f['data{}'.format(blocknum)][blockoffset] == 0)
        for frame in zero_frames:
            self.f['data{}'.format(blocknum)][blockoffset, frame] = 1

    def get_data_by_params(self, params: []):
        """

        :param params: list, parameter names. If name of a branch array is given, all its leaves are included.
                             If name of a leaf is given, leaves in all containing branches are included
                             Branches must be given explicitly with integrated index.
                             If len(list)==0 then return whole array by order of data types.
        :return: array-like, shape=(selected params, file length)
        """
        if len(params) <= 0:
            return self.file_ndarray
        f_idx = []
        for param in params:
            # if it's branch/normal array name
            if self.index_alloc.__contains__(param):
                reftable_idx = self.index_alloc.get(param)
                reftable_idx = np.arange(reftable_idx[0], reftable_idx[len(reftable_idx) - 1] + 1)
            else:
                print("parameter {} is not found".format(param))
                continue
            for idx in reftable_idx:
                #data_type = self.reftable[idx][1].decode('utf-8')
                (blocknum, blockoffset) = self._get_blockinfo(idx)
                global_blockoffset = self.glob_datablockoffset.get('data{}'.format(blocknum))
                #blockoffset = int(self.reftable[idx][2])
                #global_blockoffset = self.glob_datablockoffset.get('data{}'.format(data_type))
                f_idx.append(global_blockoffset + blockoffset)
        return self.file_ndarray[f_idx, :]

    def set_label(self, obj_mnvs, frame, is_label_changed=False):
        """
            Set obj_mnv and the resulted event_type and event_obj_vid_indx for given frame.
        :param obj_mnvs: list, len==OBJ_ID_NUM, signed int
        :param frame: int, starts with 0, supports ndarray indexing
        :param is_label_changed: boolean, if true then it's done by manual check
        :return:
        """
        obj_mnvs_idx_stnd = self.index_alloc.get('obj_mnvs')
        event_type_idx = self.index_alloc.get('event_type')[0]
        is_auto_labeled_idx = self.index_alloc.get('is_auto_labeled')[0]
        event_obj_vid_indx_idx = self.index_alloc.get('event_obj_vid_indx')[0]
        obj_mnvs_idx = np.arange(obj_mnvs_idx_stnd[0], obj_mnvs_idx_stnd[1] + 1)
        event_type = 0
        event_idx = -1
        i = 0
        # update obj_mnvs
        for ind in obj_mnvs_idx:
            (blocknum, blockoffset) = self._get_blockinfo(ind)
            global_blockoffset = self.glob_datablockoffset.get('data{}'.format(blocknum))
            self.f['data{}'.format(blocknum)][blockoffset, frame] = obj_mnvs[i]
            self.file_ndarray[blockoffset + global_blockoffset, frame] = obj_mnvs[i]
            # only record the LAST event of current frame
            # if event_type == 0 and obj_mnvs[i] != 0:
            if obj_mnvs[i] != 0:
                event_type = obj_mnvs[i]
                event_idx = i
            i += 1
        # update event_type
        (blocknum, blockoffset) = self._get_blockinfo(event_type_idx)
        global_blockoffset = self.glob_datablockoffset.get('data{}'.format(blocknum))
        self.f['data{}'.format(blocknum)][blockoffset, frame] = int(event_type)
        self.file_ndarray[blockoffset + global_blockoffset, frame] = int(event_type)
        # update event_idx
        (blocknum, blockoffset) = self._get_blockinfo(event_obj_vid_indx_idx)
        global_blockoffset = self.glob_datablockoffset.get('data{}'.format(blocknum))
        self.f['data{}'.format(blocknum)][blockoffset, frame] = int(event_idx)
        self.file_ndarray[blockoffset + global_blockoffset, frame] = int(event_idx)
        # update is_auto_labeled
        (blocknum, blockoffset) = self._get_blockinfo(is_auto_labeled_idx)
        global_blockoffset = self.glob_datablockoffset.get('data{}'.format(blocknum))
        self.f['data{}'.format(blocknum)][blockoffset, frame] = 0 if is_label_changed else 1
        self.file_ndarray[blockoffset + global_blockoffset, frame] = 0 if is_label_changed else 1

    def get_labeled_frames(self):
        """
            Get frames whose event_type != 0
        :return: array-like, shape: (labeled frames #, )
        """
        event_type_idx = self.index_alloc.get('event_type')[0]
        blocknum = self.reftable[event_type_idx][1].decode('utf-8')
        blockoffset = self.reftable[event_type_idx][2]
        event_type_arr = np.array(self.f['data{}'.format(blocknum)][blockoffset])
        return np.argwhere(event_type_arr != 0).flatten()

    def get_training_set(self, desired_params, use_attrs=False, negative_ratio=1.2):
        """Get positive and false positive cases in clips with information described in h5.attrs. The h5.attrs are
        generated after reading bookmarks, which is not a must yet.

        Args:
            desired_params: array-like, desired parameters to be included in the training set
            use_attrs:
            negative_ratio: negative case num / positive case num Todo: negative cases

        Returns:
            list of TrainingDataClip
        """
        desired_data = self.get_data_by_params(desired_params)
        labels = self.get_data_by_params(["obj_mnvs"])
        f_attrs = self.f.attrs
        labeled_frames = self.get_labeled_frames()
        labeled_len = len(labeled_frames)
        output = []
        if labeled_len == 0:
            return output
        attrs_keys = list(f_attrs.keys())
        evt_head = max(labeled_frames[0] - self.CLIP_OVERHEAD, 0)
        evt_tail = min(self.file_length, labeled_frames[0] + self.CLIP_TAIL)
        prev_evt_frame = labeled_frames[0]
        if not attrs_keys.__contains__(prev_evt_frame):
            # print('frame {:d} is missing in attributes'.format(prev_evt_frame))
            attr = [""]
        else:
            attr = f_attrs.get(prev_evt_frame)
        attributes = [attr]
        evt_frames = [prev_evt_frame]
        for evt_frame in labeled_frames[1:]:
            # if current frame is separatable with previous one, record previous one
            if evt_frame - prev_evt_frame >= self.CLIP_OVERHEAD:
                output.append(TrainingDataClip(desired_data[:, evt_head: evt_tail], labels[:, evt_head: evt_tail],
                                               attributes, evt_frames))
                attributes = []
                evt_frames = []
                evt_head = max(evt_frame - self.CLIP_OVERHEAD, 0)
            evt_tail = min(self.file_length, evt_frame + self.CLIP_TAIL)
            if not attrs_keys.__contains__(evt_frame):
                # print('frame {:d} is missing in attributes'.format(evt_frame))
                attr = [""]
            else:
                attr = f_attrs.get(evt_frame)
            attributes.append(attr)
            evt_frames.append(evt_frame)
            prev_evt_frame = evt_frame
        # append the last clip
        output.append(TrainingDataClip(desired_data[:, evt_head: evt_tail], labels[:, evt_head: evt_tail],
                                       attributes, evt_frames))
        return output

    def close(self):
        self.f.close()


class TrainingDataClip:
    def __init__(self, sliced_data, label_data, attributes, evt_frames):
        """
        An auxiliary class used to record an (aggregated) event.

        :param sliced_data: ndarray, raw sliced data from original set excluding debug and naive labels
        :param label_data: ndarray, the idx-wise labels for each frame of sliced_data
        :param attributes: list, h5py.File.attrs, including manually checked comments, are comments of evt_frames
        :param evt_frames: list, of evt_frames, in case of event aggregation
        """
        self.sliced_data = sliced_data
        self.label_data = label_data
        self.attrs_dict = {}
        if len(attributes) != len(evt_frames):
            print("number of axis 0 of attributes and evt_frames must be identical.")
            return
        for i in range(len(evt_frames)):
            self.attrs_dict[evt_frames[i]] = attributes[i]


"""
    QUICK TEST
"""
# h5rd = H5FileReader()
# h5rd.read_file(h5rd.SAMPLE_FILE)
# branch_names = list(h5rd.branch_idx_alloc.keys())
# for branch_name in branch_names:
#     leaf_names = list(h5rd.branch_idx_alloc.get(branch_name).keys())
#     for leaf_name in leaf_names:
#         print(branch_name, ": ", leaf_name)

# obj_mnvs = np.zeros(20)
# h5rd.set_label(obj_mnvs, 0)
# h5rd.set_label(obj_mnvs, 2)
# print("event type: ", h5rd.get_data_by_params(['event_type', 'test'])[0])
# print("event idx: ", h5rd.get_data_by_params(['event_obj_vid_indx'])[0])
# print("obj mnvs at frame: ", h5rd.get_data_by_params(['obj_mnvs'])[:, 1])
