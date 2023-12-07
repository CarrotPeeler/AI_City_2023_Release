import pandas as pd
import os
import decord
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

# suppress pandas chain assignment warnings
pd.options.mode.chained_assignment = None

"""
Given text file with class indices and corresponding name/activity, create dictionary for key-val pairs
"""
def getClassNamesDict(class_names_txt_path):
    d = dict()
    with open(class_names_txt_path) as txt:
        for line in txt:
            (key, val) = line.strip().split(",")
            d[int(key)] = val
    return d


"""
Converts timestamp in string format "hh:mm:ss" to total seconds
Returns an integer representing total seconds
"""
def timestamp_to_seconds(timestamp:str):
    hours, mins, secs = timestamp.split(':')
    return int(hours)*3600 + int(mins)*60 + int(secs)


"""
Extracts video clips from videos and generates csv with paths and labels
"""
def videos_to_clips(video_dir: str, clip_dir: str, annotation_filename: str, video_extension: str, re_encode:bool, clip_resolution:str, encode_speed:str):
    # create save dir for clips if it doesn't exist
    if(not os.path.exists(clip_dir)):
        os.mkdir(clip_dir)
    
    csv_filepaths = glob(video_dir + "/**/*.csv", recursive=True) # search for all .csv files (each dir. of videos should only have ONE)
    clip_filepaths = [] # stores image (frame) names
    classes = [] # stores class labels for each frame

    for i in tqdm(range(len(csv_filepaths))): # for each csv file corresponding to a group of videos, split each video into clips
        annotation_df = pd.read_csv(csv_filepaths[i])
        videos = glob(csv_filepaths[i].rpartition('/')[0] + "/*" + video_extension)

        for video in videos: # for each video, parse out its individual csv data from the original csv for the group of videos 
            video_filename = video.rpartition('/')[-1].partition('.')[0] 
            parsed_video_df = parse_data_from_csv(video, annotation_df)

            # extract start and end timestamps from df and convert each to seconds
            start_times = list(map(timestamp_to_seconds, parsed_video_df['Start Time'].to_list()))
            end_times = list(map(timestamp_to_seconds, parsed_video_df['End Time'].to_list()))
            labels = list(map(lambda str:int(str.rpartition(' ')[-1]), parsed_video_df['Label (Primary)'].to_list()))
                
            # write ffmpeg commands to bash script for parallel execution & add labels and file paths to lists for annotation 
            with open(clip_dir + "/ffmpeg_commands.sh", 'a+') as f:

                for k in range(len(labels)): # for each action in the video, extract video clip of action based on timestamps

                    # extract only the portion of the video between start_time and end_time
                    clip_filepath = clip_dir + f"/{video_filename}" + f"-start{start_times[k]}" + f"-end{end_times[k]}" + ".MP4"

                    if(encode_speed == "default"): 
                        preset = ""
                    else:
                        preset = "-preset " + encode_speed + " "
                    
                    # # no re-encoding (typically much faster than with re-encoding)
                    if(re_encode == False):
                        f.writelines(f"ffmpeg -loglevel quiet -y -ss {start_times[k]} -to {end_times[k]} -i {video} -c:v copy {clip_filepath}\n")
                    else:       
                        f.writelines(f"ffmpeg -loglevel quiet -y -ss {start_times[k]} -to {end_times[k]} -i {video} -vf scale={clip_resolution} -c:v libx264 {preset}{clip_filepath}\n")
                    
                    clip_filepaths.append(clip_filepath)
                    classes.append(labels[k])
    
    # parallelize ffmpeg commands
    os.system(f"parallel --eta < {clip_dir}/ffmpeg_commands.sh")

    # create annotation csv to store clip file paths and their labels
    data = pd.DataFrame()
    data['clip'] = clip_filepaths
    data['class'] = classes  
    data.to_csv(clip_dir + "/" + annotation_filename, sep=" ", header=False, index=False)


"""
Parses out the data for a single video, given its file path and the entire annotation csv for the user recorded in the video
NOTE: each annotation file has multiple videos, each with their own data
"""
def parse_data_from_csv(video_filepath, annotation_dataframe):
    df = annotation_dataframe

    video_view = video_filepath.rpartition('/')[-1].partition('_')[0] # retrieve camera view angle
    video_endnum = video_filepath.rpartition('_')[-1].partition('.')[0] # retrieve block appearance number (last number in the file name)

    video_start_rows = df.loc[df["Filename"].notnull()] # create dataframe with only rows having non-null file names
    video_start_rows.reset_index(inplace=True)

    for index, row in video_start_rows.iterrows(): # rename each row; only include the camera view type and block number
        video_start_rows["Filename"][index] = row["Filename"].partition('_')[0] + "_" + row["Filename"].rpartition('_')[-1]

    # with sub-dataframe, retrieve zero-based index for the current video 
    video_index = video_start_rows.index[video_start_rows["Filename"].str.contains(video_view, case=False) &
                                        video_start_rows["Filename"].str.contains(video_endnum, case=False)].to_list()[0]
    
    video_index_orig = video_start_rows.iloc[[video_index]]["index"].to_list()[0] # find the original dataframe index 

    next_video_index_orig = -1

    if video_index + 1 < len(video_start_rows): # if there's data for other videos after this video, grab the index where the next video's data starts
        next_video_index_orig = video_start_rows.iloc[[video_index + 1]]["index"].to_list()[0]
    else:
        next_video_index_orig = len(df) - 1 # otherwise, this video's data is last in the csv, simply set the 

    parsed_video_data = df.iloc[video_index_orig:next_video_index_orig] # create a sub-dataframe of the original with only this video's data

    return parsed_video_data


"""
Because of encoding, the videos may take several hours to process; use the command below to run the script in the background:
python3 prepare_data.py < /dev/null > ffmpeg_log.txt 2>&1 &
"""
if __name__ == '__main__':

    # Edit params
    videos_loadpath = "/home/vislab-001/Jared/SET-A1"
    clips_savepath = os.getcwd() + "/data/data_normal"
    annotation_filename = "annotation.csv"

    # truncate each train video into frames (truncate_size = num of frames per video)
    videos_to_clips(video_dir=videos_loadpath, 
                  clip_dir=clips_savepath, 
                  video_extension=".MP4", 
                  annotation_filename=annotation_filename,
                  re_encode=True,
                  encode_speed = "ultrafast",
                  clip_resolution="512:512")

    ############################################################################################

    print("All videos have been successfully processed into clips. Creating annotation split...")

    df = pd.read_csv(clips_savepath + "/" + annotation_filename, sep=" ", names=["clip", "class"])

    # clean data for videos with no time duration
    zero_sec_clips = df.index[df['clip'].str.rpartition("start")[2].str.partition('-')[0] == df['clip'].str.rpartition("end")[2].str.partition('.')[0]].to_list()
    df.drop(zero_sec_clips, inplace=True)

    # Generate 5-fold annotations for each camera view 
    num_folds = 5
    cam_views = ['Dashboard', 'Rear_view', 'Right_side_window']
    
    for cam_view in cam_views:
        # retrieve df for cam view clips
        cam_view_df = df.loc[df["clip"].str.rpartition('/')[2].str.partition('_user')[0] == cam_view]

        # split data into train and val sets 
        splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state = 42)
        splits = splitter.split(X=cam_view_df['clip'], y=cam_view_df['class'])

        for fold, (train_indexes, test_indexes) in enumerate(splits):
            train_df = cam_view_df.iloc[train_indexes]
            test_df = cam_view_df.iloc[test_indexes]

            cam_dir = os.getcwd() + f"/slowfast/annotations/{cam_view}"

            if(not os.path.exists(cam_dir)):
                os.makedirs(cam_dir, exist_ok=True)

            fold_dir = f"{cam_dir}/fold_{fold}"

            if(not os.path.exists(fold_dir)):
                os.makedirs(fold_dir, exist_ok=True)

            train_df.to_csv(fold_dir + "/train.csv", sep=" ", header=False, index=False)
            test_df.to_csv(fold_dir + "/val.csv", sep=" ", header=False, index=False)

        # final train split with combined train and val data after 5-fold CV
        all_dir = f"{cam_dir}/all"
        if(not os.path.exists(all_dir)):
            os.makedirs(all_dir, exist_ok=True)
        cam_view_df.to_csv(all_dir + "/train.csv", sep=" ", header=False, index=False)
        # dummy val csv 
        cam_view_df.to_csv(all_dir + "/val.csv", sep=" ", header=False, index=False)