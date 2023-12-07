# Code modified and taken from VideoMAEv2 datasets.py

import os
import numpy as np
from torch.utils.data import Dataset
from VideoMAEv2.dataset import video_transforms, volume_transforms
from VideoMAEv2.dataset.loader import get_video_loader

class ProposalDataset(Dataset):
    """Generates and loads proposal frames across three videos (diff. camera views) for a single video id"""

    def __init__(self,
                 cfg,
                 anno_df):
        self.cfg = cfg
        self.anno_df = anno_df
        self.frame_sample_rate = cfg.DATA.SAMPLING_RATE
        self.clip_len = cfg.DATA.NUM_FRAMES
        self.num_segment=1

        self.video_loader = get_video_loader()

        mean = [0.485, 0.456, 0.406] if cfg.MODEL.MODEL_NAME == "VideoMAEv2" else cfg.DATA.MEAN
        std = [0.229, 0.224, 0.225] if cfg.MODEL.MODEL_NAME == "VideoMAEv2" else cfg.DATA.STD

        self.data_resize = video_transforms.Compose([
            video_transforms.Resize(
                cfg.DATA.TEST_CROP_SIZE, interpolation='bilinear'),
            video_transforms.CenterCrop(
                size=(cfg.DATA.TEST_CROP_SIZE, cfg.DATA.TEST_CROP_SIZE))
        ])

        self.data_transform = video_transforms.Compose([
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=mean, 
                                       std=std)
        ])

        # list of tuples of start/end times for each proposal by index
        self.proposal_times = list(zip(self.anno_df["start_time"].to_list(), 
                                  self.anno_df["end_time"].to_list()))
        
        self.proposal_paths_by_view = {
            "Dashboard": self.anno_df["Dashboard"].to_list(),
            "Rear_view": self.anno_df["Rear_view"].to_list(),
            "Right_side_window": self.anno_df["Right_side_window"].to_list(),
        }
    
        assert len(set(map(lambda x:len(x), self.proposal_paths_by_view.values()))) == 1, "Video ID has mismatch number of proposal clips"

    def __getitem__(self, index):
        proposal_frames_dict = {}
        clip_paths = []

        for cam_view, clips in self.proposal_paths_by_view.items():
            # retrieve proposal clip path by index
            clip_path = clips[index]
            clip_paths.append(clip_path)

            # load proposal frames (T H W C)
            frames = self.load_video(clip_path)     
            frames = self.data_resize(frames)
            frames = self.data_transform(frames)

            # C T H W
            proposal_frames_dict[cam_view] = frames
        
        # final check to see if clips are from same proposal
        assert len(set(map(lambda x: x.rpartition('id_')[-1].partition('-')[0], clip_paths))) == 1, "Error: videos do not have the same id"
        assert len(set(map(lambda x: x.partition("-start")[-1].partition('-')[0], clip_paths))) == 1,\
        f"Clips for sample {index} do not represent the same temporal interval"
        
        return proposal_frames_dict, index
        

    def load_video(self, sample, sample_rate_scale=1):
        """
        Load video frames corresponding to given proposal 

        Returns numpy array of frames
        """
        fname = sample
        try:
            vr = self.video_loader(fname)
        except Exception as e:
            print(f"Failed to load video from {fname} with error {e}!")
            return []

        length = len(vr)
        
        # handle temporal segments
        converted_len = int(self.clip_len * self.frame_sample_rate)
        seg_len = length // self.num_segment

        all_index = []
        for i in range(self.num_segment):
            if seg_len <= converted_len:
                index = np.linspace(
                    0, seg_len, num=seg_len // self.frame_sample_rate)
                index = np.concatenate(
                    (index,
                     np.ones(self.clip_len - seg_len // self.frame_sample_rate)
                     * seg_len))
                index = np.clip(index, 0, seg_len - 1).astype(np.int64)
            else:
                end_idx = (converted_len + seg_len) // 2
                str_idx = end_idx - converted_len
                index = np.linspace(str_idx, end_idx, num=self.clip_len)
                index = np.clip(index, str_idx, end_idx - 1).astype(np.int64)
            index = index + i * seg_len
            all_index.extend(list(index))

        all_index = all_index[::int(sample_rate_scale)]
        vr.seek(0)
        buffer = vr.get_batch(all_index).asnumpy()
        return buffer
    

    def __len__(self):
        return len(self.proposal_times)
