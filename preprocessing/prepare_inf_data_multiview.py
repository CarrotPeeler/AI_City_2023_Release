import os
import pandas as pd
from tqdm.auto import tqdm
from glob import glob
from prepare_train_data_multiview import timestamp_to_seconds


def extract_video_duration_seconds(video_filepath:str):
    """
    Extract video duration in seconds 
    """
    video_metadata = os.popen(f'ffmpeg -i {video_filepath} 2>&1 | grep "Duration"').read()
    timestamp = video_metadata.partition('.')[0].partition(':')[-1].strip()
    return timestamp_to_seconds(timestamp)


def generate_video_proposals(total_frames, video_fps, proposal_length, proposal_stride):
    """
    Generates proposals given total frames for a video and length and stride of proposals
    """
    proposals = []
    for j in range(0, total_frames - proposal_length, proposal_stride):
        # generate proposal start and end frame index
        start_frame_idx = j
        end_frame_idx = j + proposal_length

        # calculate start and end times in format ss.ms (seconds, milliseconds)
        start_time = round(start_frame_idx / video_fps, 2)
        end_time = round(end_frame_idx / video_fps, 2)

        proposals.append((start_time, end_time))
    return proposals


def videos_to_proposal_clips(video_dir: str, 
                        video_ids_csv_path: str,
                        clip_dir: str,
                        anno_name: str,
                        video_fps: int,
                        proposal_length: int,
                        proposal_stride: int,
                        re_encode:bool, 
                        clip_resolution:str, 
                        encode_speed:str):
    """
    Extracts video clips from videos 
    """
    # create save dir for clips if it doesn't exist
    if(not os.path.exists(clip_dir)):
        os.mkdir(clip_dir)

    # gather testing video ids and filenames
    videos_df = pd.read_csv(video_ids_csv_path)

    # output csv columns
    video_ids = []
    start_times = []
    end_times = []
    views = {"Dashboard": [],
             "Rear_view": [],
             "Right_side_window": []}

    # run tal for each video_id in the test dataset
    for i in tqdm(range(len(videos_df))):
        # get names to all videos corresponding to same video_id
        video_names = videos_df.iloc[i,1:4].to_list()
        video_id = i + 1
        id_views = {"Dashboard": [],
                    "Rear_view": [],
                    "Right_side_window": []}
        
        proposal_sets = []
        for video_name in video_names: # for each video, parse out its individual csv data from the original csv for the group of videos 
            video_path = glob(video_dir + "**/**/" + video_name, recursive=True)[0]      
            total_frames = extract_video_duration_seconds(video_path) * video_fps

            proposals = generate_video_proposals(total_frames, video_fps, proposal_length, proposal_stride)
            proposal_sets.append(proposals)

            # make dir for all clips in a video
            clip_subdir = f"{clip_dir}/id_{video_id}/{video_name}"

            if not os.path.exists(clip_subdir):
                os.makedirs(clip_subdir, exist_ok=True)
                
            # write ffmpeg commands to bash script for parallel execution & add labels and file paths to lists for annotation 
            with open(clip_dir + "/ffmpeg_commands.sh", 'a+') as f:

                for (start, end) in proposals: # for each action in the video, extract video clip of action based on timestamps

                    # extract only the portion of the video between start_time and end_time
                    clip_filepath = f"{clip_subdir}/{video_name}-start{start}-end{end}.MP4"
                    id_views[video_name.rpartition('/')[-1].partition('_user')[0]].append(clip_filepath)

                    if(encode_speed == "default"): 
                        preset = ""
                    else:
                        preset = "-preset " + encode_speed + " "
                    
                    # # no re-encoding (typically much faster than with re-encoding)
                    if(re_encode == False):
                        f.writelines(f"ffmpeg -loglevel quiet -y -ss {start} -to {end} -i {video_path} -c:v copy {clip_filepath}\n")
                    else:       
                        f.writelines(f"ffmpeg -loglevel quiet -y -ss {start} -to {end} -i {video_path} -vf scale={clip_resolution} -c:v libx264 {preset}{clip_filepath}\n")
        
        # find shortest video among all views and extract start and end times
        min_proposal_set = min(proposal_sets, key=len)
        start, end = zip(*min_proposal_set)
        start_times.extend(start)
        end_times.extend(end)
        video_ids.extend([video_id] * len(min_proposal_set))
        for view in views: views[view].extend(id_views[view][:len(min_proposal_set)])

    df = pd.DataFrame()
    df["video_id"] = video_ids
    df["start_time"] = start_times
    df["end_time"] = end_times
    df["Dashboard"] = views["Dashboard"]
    df["Rear_view"] = views["Rear_view"]
    df["Right_side_window"] = views["Right_side_window"]
    df.to_csv(clip_dir + '/' + anno_name, sep=" ", header=True, index=False)

    # parallelize ffmpeg commands
    os.system(f"parallel --eta < {clip_dir}/ffmpeg_commands.sh")


if __name__ == '__main__':

    # Edit params
    videos_loadpath = "/home/vislab-001/Jared/SET-A2" # where test videos are saved
    video_ids_csv_path = "/home/vislab-001/Jared/Envy_AI_City/slowfast/inference/video_ids.csv" # csv with names and ids of videos
    clips_savepath = os.getcwd() + "/data/data_inf"

    # truncate each train video into frames (truncate_size = num of frames per video)
    videos_to_proposal_clips(video_dir=videos_loadpath, 
                            video_ids_csv_path=video_ids_csv_path,
                            clip_dir=clips_savepath, 
                            anno_name="proposals.csv",
                            proposal_length=64,
                            proposal_stride=16,
                            video_fps=30,
                            re_encode=True,
                            encode_speed = "ultrafast",
                            clip_resolution="512:512")

    ############################################################################################

    print("All videos have been successfully processed into clips.")