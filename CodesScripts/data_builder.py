import random

from H5FileReader import H5FileReader as H5rd
import os, numpy as np
import torch
from torch import tensor
from progressbar import ProgressBar
DEFAULT_MAIN_PARAMS = np.array(['host_data.EgoVehALatFild_f', 'host_data.EgoVehALatFildVar_f',
                                'host_data.EgoVehALgtFild_f', 'host_data.EgoVehALgtFildVar_f',
                                'host_data.EgoVehCrvt_f', 'host_data.EgoVehCrvtVar_f',
                               'host_data.EgoVehSpdFild_f', 'host_data.EgoVehSpdFildVar_f',
                                'host_data.EgoVehYawRateFild_f', 'host_data.EgoVehYawRateFildVar_f',
                                'host_data.EgoVehTurnIndcrSts_u8', 'host_path_mean',
                                'host_path_std', 'lane_data.Mean_Coefficient_arr_f32',
                                'lane_data.egolane_width_mean_f32',
                                'lane_data.Confdnc_arr_f32', 'path_data.path_estipoly_coeff_mean_f32',
                                'output_obj_daten0', 'output_obj_daten1', 'output_obj_daten2', 'output_obj_daten3',
                                'output_obj_daten4', 'output_obj_daten5', 'output_obj_daten6', 'output_obj_daten7',
                                'output_obj_daten8', 'output_obj_daten9', 'output_obj_daten10', 'output_obj_daten11',
                                'output_obj_daten12', 'output_obj_daten13', 'output_obj_daten14', 'output_obj_daten15',
                                'output_obj_daten16', 'output_obj_daten17', 'output_obj_daten18', 'output_obj_daten19'])
# obj_daten idx: [8, 27]
DEFAULT_OBJ_PARAMS = np.arange(3, 25, dtype=int)
TEST_H5_AMOUNT = 200
# DEFAULT_OBJ_PARAMS = np.array([3, 4, 5, 7, 11, 16, 19, 23], dtype=int)
#DEFAULT_H5_DIR = r'./hdf_data/'
DEFAULT_H5_DIR = '/tmp/tmpvcol_yw3/'
DEFAULT_OBJ_NUM = 3
DEFAULT_DATA_FPS = 20.
INVALID_REG_TIME = 0.  # seconds
OBJ_MAX_NUM = 20
DEFAULT_CLIP_LENGTH = 32
DEFAULT_BATCH_NUM = 2  # maximum valid batch number
EARLIEST_LABEL_FRAME = 200  # 32-frame data clips within 200 frames before auto-labeled frame are marked as positive
EARLIEST_CLIP_FRAME = 0  # frame from which clip slide-window starts on the sliced 30s training set
NON_OBJ_DATA_LEN = 30
DEFAULT_NEG_POS_RATIO = 0.1
EVENT_PROCESSING_OFFSET = 20  # frame, need 20 frames to settle the hypocrisy
VID_OBJ_INVALID_ID = 0
SLIDE_STEP = 10
IV = 0
SLNIV = 1
FLNIV = 2


def get_desired_obj_param_indices(raw_train_obj_dim, num_obj=3, wanted_obj_idx=-1, for_output_tensor=False):
    """
    Get indices of parameters of first 3 obj_daten in raw_data.

    Args:
        raw_train_obj_dim:
        num_obj:
        wanted_obj_idx: index of the wanted specific objects' parameters.
        for_output_tensor: true if the indices correspond to the output tensor of 3*32 dimensions

    Returns:
        list of indices

    """
    output = np.array([])
    if not for_output_tensor:
        obj_dim_len = raw_train_obj_dim
        obj_param_loc = DEFAULT_OBJ_PARAMS
    else:
        obj_param_loc = np.arange(0, len(DEFAULT_OBJ_PARAMS))
        obj_dim_len = len(obj_param_loc)
    if wanted_obj_idx == -1:
        for obj_idx in range(num_obj):
            output = np.concatenate((output, NON_OBJ_DATA_LEN +
                                     obj_idx * obj_dim_len + obj_param_loc))
    else:
        output = np.concatenate((output, NON_OBJ_DATA_LEN +
                                 wanted_obj_idx * obj_dim_len + obj_param_loc))
    return output.astype(int)


def choose_obj(raw_train_obj_dim, num_obj, sliced_data, obj_ids_all, clip_head, clip_end):
    """Choose the num_obj positional objects based on their position at the regarded frame. Index scope:
    clip.

    Notes:
        This function will return invalid obj_id and its output, as long as not all 3 positional objects are found.
    Args:
        raw_train_obj_dim:
        num_obj:
        sliced_data:
        obj_ids_all:
        clip_head:
        clip_end: the frame that determines which objects should be among 3 positional ones

    Returns:
         list of parameter clips of 3 positional data, note that it is a simple slice based on regarded frame
         for all frames, and positional obj_ids:.
    """
    obj_iv = np.zeros((DEFAULT_OBJ_PARAMS.shape[0], sliced_data.shape[1]))
    obj_slniv = np.zeros((DEFAULT_OBJ_PARAMS.shape[0], sliced_data.shape[1]))
    obj_flniv = np.zeros((DEFAULT_OBJ_PARAMS.shape[0], sliced_data.shape[1]))
    obj_iv_idx = IV
    obj_slniv_idx = SLNIV
    obj_flniv_idx = FLNIV
    remaining_pos = list(np.arange(0, num_obj, step=1, dtype=int))
    # check each of 20 objects for num_obj frontal objects
    for i in range(OBJ_MAX_NUM):
        obj_param_idx = get_desired_obj_param_indices(raw_train_obj_dim, wanted_obj_idx=i)
        # default obj_num == 3
        if num_obj == DEFAULT_OBJ_NUM:
            # SLNIV, 0< dst_to_left_line < 3
            if 0 < sliced_data[obj_param_idx[-2], clip_end - 1] < 3 and remaining_pos.__contains__(SLNIV):
                obj_slniv = sliced_data[obj_param_idx]
                obj_slniv_idx = i
                remaining_pos.remove(SLNIV)
            # IV
            elif sliced_data[obj_param_idx[-2], clip_end - 1] < 0 and \
                    sliced_data[obj_param_idx[-1], clip_end - 1] > 0 and \
                    remaining_pos.__contains__(IV):
                obj_iv = sliced_data[obj_param_idx]
                obj_iv_idx = i
                remaining_pos.remove(IV)
            # FLNIV
            elif -3 < sliced_data[obj_param_idx[-1], clip_end - 1] < 0 and remaining_pos.__contains__(FLNIV):
                obj_flniv = sliced_data[obj_param_idx]
                obj_flniv_idx = i
                remaining_pos.remove(FLNIV)
            elif len(remaining_pos) == 0:
                break

    obj_data = [obj_iv, obj_slniv, obj_flniv]
    obj_ids = obj_ids_all[[obj_iv_idx, obj_slniv_idx, obj_flniv_idx]]
    return obj_data, obj_ids


def filter_data(train_data, *labels, neg_case_ratio=DEFAULT_NEG_POS_RATIO):
    """
        Filter negative cases to make sure they are not overwhelming.
    Args:
        train_data:
        labels: contains cls and reg labels by DEFAULT
        neg_case_ratio: == #neg / #pos

    Returns:
        filtered data with [positive cases, negative cases]
    """
    # clip_idx = set(range(len(labels)))
    labels_cls, labels_reg = labels
    labels_cls = np.array(labels_cls)
    # todo: magic numbers
    # pos_case_idx = np.argwhere(0 < labels_cls < 3)[:, 0]
    idle_case_idx = set(np.argwhere(labels_cls == 0)[:, 0])
    left_case_idx = set(np.argwhere(labels_cls == 1)[:, 0])
    right_case_idx = set(np.argwhere(labels_cls == 2)[:, 0])
    pos_case_idx = np.array(list(left_case_idx.union(right_case_idx)))
    pos_case_num = pos_case_idx.shape[0]
    neg_case_num = int(pos_case_num * neg_case_ratio)
    neg_case_idx = np.array(list(idle_case_idx.difference(pos_case_idx)))
    np.random.shuffle(neg_case_idx)
    filtered_neg_case_idx = neg_case_idx[:neg_case_num]
    np.random.shuffle(pos_case_idx)
    np.random.shuffle(filtered_neg_case_idx)
    filtered_case_idx = np.concatenate((pos_case_idx, filtered_neg_case_idx))
    # np.random.shuffle(filtered_case_idx)
    print("################ Training Data Num", len(filtered_case_idx), " ##################")
    print("idle: ", neg_case_num,
          "left: ", left_case_idx.__len__(), "right: ", right_case_idx.__len__())
    return tensor(train_data)[filtered_case_idx].type('torch.FloatTensor'), \
        tensor(labels_cls)[filtered_case_idx].type('torch.LongTensor'), \
        tensor(labels_reg)[filtered_case_idx].type('torch.FloatTensor'), \
        pos_case_num


def assign_labels(label_cls_batch, label_reg_batch, event_label_loc, clip_head, clip_end, obj_idx,
                  obj_id, obj_ids_all, fixed_pos, raw_data, dbg_label_ids_batch):
    """Assign labels for a given obj_id. Label values are dependent on evt label and relative position between sliding
    window and evt_process.

    Args:
        label_cls_batch:
        label_reg_batch:
        event_label_loc:
        clip_head:
        clip_end:
        obj_idx:
        obj_id:
        obj_ids_all:
        fixed_pos:
        raw_data:
        dbg_label_ids_batch:

    Returns:

    """
    clip_obj_id_cls = 0
    clip_obj_id_reg = 20
    # since we extract data clip based on events, this should always be true
    if event_label_loc.shape[0] != 0:
        for evt in event_label_loc:
            # if event is passed
            if clip_head > evt[1] - EVENT_PROCESSING_OFFSET:
                continue
            # if event has not started
            if evt[1] - EVENT_PROCESSING_OFFSET - clip_head > EARLIEST_LABEL_FRAME:
                continue
            # NOTE: ASSUMING its signal is consistent. Signal interruption is possible but rare. Todo
            # NOTE: for most cut out cases, event_obj_id will be lost before event processes
            # if the position of obj_id's existence is consistent throughout clip Todo
            if obj_id == obj_ids_all[obj_idx, clip_end - 1]:
                # if an event obj_id is detected as one of the num_obj in this clip
                if obj_ids_all[evt[0], evt[1]] == obj_id:
                    # record label_reg
                    event_frame = evt[1] - EVENT_PROCESSING_OFFSET
                    if event_frame > clip_end:
                        # label_reg_batch[-1][fixed_pos] = min(label_reg_batch[-1][fixed_pos],
                        #                                      (event_frame - clip_end) / DEFAULT_DATA_FPS)
                        clip_obj_id_reg = min(clip_obj_id_reg,
                                              (event_frame - clip_end) / DEFAULT_DATA_FPS)
                    # record label_cls
                    # todo: corrupted data
                    # label_cls_batch[-1][fixed_pos] = min(2, raw_data.label_data[evt[0], evt[1]])
                    clip_obj_id_cls = min(2, raw_data.label_data[evt[0], evt[1]])
                    dbg_label_ids_batch[-1][fixed_pos] = obj_id
    if obj_id != VID_OBJ_INVALID_ID:
        # record label_reg
        label_reg_batch[-1][fixed_pos] = clip_obj_id_reg
        # record label_cls
        label_cls_batch[-1][fixed_pos] = clip_obj_id_cls


class DataBuilder:
    """

    Notes:
        Label of value NUM_CLASS - 1 represents label for invalid data (empty input).

    Attributes:
        raw_data: list of TrainingDataClips with (desired_data, labels, attributes, evt_frames) for each event
        num_obj: int, indices bzgl. the first num_obj objects are calculated
        clip_len: int, number of frames per training clip. Flawed clips will be discarded.
    """
    def __init__(self, h5_directory, main_parameters=DEFAULT_MAIN_PARAMS,
                 obj_parameter_indices=DEFAULT_OBJ_PARAMS, num_obj=DEFAULT_OBJ_NUM,
                 num_obj_max=OBJ_MAX_NUM,
                 batch_num=DEFAULT_BATCH_NUM, clip_len=DEFAULT_CLIP_LENGTH, data_fps=DEFAULT_DATA_FPS):
        """

        Args:
            h5_directory:
            main_parameters:
            obj_parameter_indices:
        """
        self.h5rd = H5rd()
        self.num_obj = num_obj
        self.num_obj_max = num_obj_max
        self.clip_len = clip_len
        self.batch_num = batch_num
        self.data_fps = data_fps
        self.raw_data = self._generate_raw_data(h5_directory, main_parameters)
        if len(self.raw_data) != 0:
            self.raw_train_data_dim = self.raw_data[0].sliced_data.shape[0]
            self.raw_train_obj_dim = int((self.raw_train_data_dim - NON_OBJ_DATA_LEN) / OBJ_MAX_NUM)
            self.sliced_param_indices = np.concatenate((range(0, NON_OBJ_DATA_LEN),
                                                        get_desired_obj_param_indices(self.raw_train_obj_dim)))
            self.tensor_id_indices = self._get_obj_id_indices()

    def _get_obj_id_indices(self):
        output = np.array([])
        obj_id_loc = np.zeros(1)
        for obj_idx in range(self.num_obj_max):
            output = np.concatenate((output, NON_OBJ_DATA_LEN + obj_idx * self.raw_train_obj_dim + obj_id_loc))
        return output.astype(int)

    def _generate_raw_data(self, h5_directory, main_parameters):
        raw_data = []
        print("################ Scanning H5 Files ##################")
        for i, filename in enumerate(os.listdir(h5_directory)):
            if filename.endswith(".h5"):
                if i >= TEST_H5_AMOUNT:
                    break
                self.h5rd.read_file(os.path.join(h5_directory + filename))
                temp = self.h5rd.get_training_set(main_parameters)
                raw_data = np.concatenate((raw_data, temp))
                # DEBUG
        return raw_data

    @staticmethod
    def build_random(batch_num=20, batch_size=2):
        labels = []
        train_data = np.random.randn(batch_num, batch_size, 96, 32)
        for i in range(batch_num):
            label_batch = [[random.randrange(0, 3, step=1), random.randrange(0, 3, step=1), 0]]
            for _ in range(batch_size - 1):
                label_batch.append([random.randrange(0, 3, step=1), random.randrange(0, 3, step=1), 0])
            labels.append(label_batch)
        return tensor(train_data).type('torch.FloatTensor'), tensor(labels).type('torch.FloatTensor')

    def build(self):
        labels_cls = []
        labels_reg = []
        train_data = []
        dbg_label_ids = []
        # raw_data: TrainingDataClip
        pgbar = ProgressBar().start()
        total = len(self.raw_data)
        for i, raw_data in enumerate(self.raw_data):
            non_obj_data = raw_data.sliced_data[list(range(NON_OBJ_DATA_LEN))]
            obj_ids_all = raw_data.sliced_data[self.tensor_id_indices]
            # generate batches
            # batch format: (event, ~400, 32 frames, parameter data per frame)
            event_label_loc = np.argwhere(raw_data.label_data != 0)
            recorded_obj_id = {}  # id: fixed position
            free_pos = list(range(self.num_obj))
            clip_head = EARLIEST_CLIP_FRAME
            clip_end = EARLIEST_CLIP_FRAME + DEFAULT_CLIP_LENGTH
            # event_batch = [[non_obj_data, <reorganized obj_data>], ...]
            event_batch = []
            label_cls_batch = []
            label_reg_batch = []
            dbg_label_ids_batch = []
            # the sliding window starts from EARLIEST_CLIP_FRAME
            # the relative position of data and label of an object is fixed into the 3 slots
            while clip_end <= raw_data.sliced_data.shape[1]:
                # labels format: <Clip#, Behavior/Class>
                obj_data, obj_ids = choose_obj(self.raw_train_obj_dim, self.num_obj, raw_data.sliced_data,
                                               obj_ids_all, clip_head, clip_end)
                # todo: magic number
                # initialize invalid classification labels
                label_cls_batch.append(np.full(self.num_obj, 3))
                # initialize invalid/idle regression labels
                label_reg_batch.append(np.full(self.num_obj, INVALID_REG_TIME, float))
                dbg_label_ids_batch.append(np.full(self.num_obj, -1))
                # initialize event_clip (aka "batch" hereby)
                event_batch.append(np.zeros((96, self.clip_len)))
                event_batch[-1][list(range(NON_OBJ_DATA_LEN))] = non_obj_data[:, clip_head: clip_end]
                # clear recorded_obj_id and free_pos list for lost ids
                keys = list(recorded_obj_id.keys())
                for key in keys:
                    if not obj_ids[:, clip_head].__contains__(key) or \
                       not obj_ids[:, clip_end - 1].__contains__(key):
                        free_pos.append(recorded_obj_id.get(key))
                        recorded_obj_id.pop(key)
                # assign obj_id to a random free fixed_pos
                # always record the num_obj objects, set label for event_obj
                # todo NOTE: obj_id that was removed above might be reassigned in a different fixed_pos below
                for obj_idx in range(DEFAULT_OBJ_NUM):
                    obj_id = obj_ids[obj_idx, clip_head]
                    if recorded_obj_id.__contains__(obj_id):
                        # fixed_pos: fixed position in the output as training data
                        fixed_pos = recorded_obj_id.get(obj_id)
                    else:
                        fixed_pos = random.choice(free_pos)
                        recorded_obj_id[obj_id] = fixed_pos
                        free_pos.remove(fixed_pos)
                    fixed_pos_indices = get_desired_obj_param_indices(self.raw_train_obj_dim,
                                                                      wanted_obj_idx=fixed_pos, for_output_tensor=True)
                    # todo: do tracking for whole clip in choose_data()
                    event_batch[-1][fixed_pos_indices] = obj_data[obj_idx][:, clip_head: clip_end]
                    assign_labels(label_cls_batch, label_reg_batch, event_label_loc, clip_head, clip_end,
                                  obj_idx, obj_id, obj_ids_all, fixed_pos, raw_data, dbg_label_ids_batch)
                # event_batch[-1] = event_batch[-1].flatten()
                clip_end += SLIDE_STEP
                clip_head += SLIDE_STEP
            train_data.append(event_batch)
            labels_cls.append(label_cls_batch)
            labels_reg.append(label_reg_batch)
            dbg_label_ids.append(dbg_label_ids_batch)
            pgbar.update(int((i / (total - 1)) * 100))
        pgbar.finish()
        # indices = torch.randperm(len(train_data))
        # manually flatten
        train_data = [clip for batch in train_data for clip in batch]
        labels_cls = [clip_label for batch in labels_cls for clip_label in batch]
        labels_reg = [clip_label for batch in labels_reg for clip_label in batch]
        # return tensor(train_data).type('torch.FloatTensor'), tensor(labels).type('torch.LongTensor')
        return filter_data(train_data, labels_cls, labels_reg)
