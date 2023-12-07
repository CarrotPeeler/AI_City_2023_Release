import pandas as pd

if __name__ == "__main__":
    bound_tolerance = 6 # +- seconds

    root = "/home/vislab-001/Jared/Envy_AI_City/slowfast/inference/submission_files/"
    sub_1_file = f"{root}/videomae_sub.txt"
    sub_2_file = f"{root}/submission_file.txt"

    colnames = ["video_id", "activity_id", "start_time", "end_time"]
    results_2_df = pd.read_csv(sub_2_file, sep=" ", names=colnames)
    results_1_df = pd.read_csv(sub_1_file, sep=" ", names=colnames).sort_values(by=["video_id", "activity_id"])

    diff_df = pd.DataFrame()
    video_ids = []
    activity_ids = []
    start_time_diffs = []
    end_time_diffs = []

    for idx, row in results_1_df.iterrows():
        video_id = row["video_id"]
        activity_id = row["activity_id"]

        start_time_1 = row["start_time"]
        end_time_1 = row["end_time"]

        results_2_row = results_2_df.loc[(results_2_df["video_id"] == video_id) 
                                       & (results_2_df["activity_id"] == activity_id)]
        if len(results_2_row) == 0:
            start_time_2 = 10000
            end_time_2 = 10000
        else:
            start_time_2 = results_2_row["start_time"].to_list()[0]
            end_time_2 = results_2_row["end_time"].to_list()[0]

        start_diff = abs(start_time_2 - start_time_1)
        end_diff = abs(end_time_2 - end_time_1)

        video_ids.append(video_id)
        activity_ids.append(activity_id)
        start_time_diffs.append(start_diff)
        end_time_diffs.append(end_diff)
    
    diff_df["video_id"] = video_ids
    diff_df["activity_id"] = activity_ids
    diff_df["start_time_diff"] = start_time_diffs 
    diff_df["end_time_diff"] = end_time_diffs
    
    bad_rows = diff_df.loc[(diff_df["start_time_diff"] > bound_tolerance) 
            | (diff_df["end_time_diff"] > bound_tolerance)]
    bad_rows.to_csv(f"{root}/bad_rows.txt", sep=" ", header=False, index=False)