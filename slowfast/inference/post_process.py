import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.optimize import minimize

# cd slowfast beforehand
        

def generate_gaussian_weights(sigma, length):
    """
    Generates Gaussian weights

    params:
        sigma: sigma term of Gaussian filter
        length: window size (number of samples to consider for filtering)
    returns:
        NParray of Gaussian weights for a window size of 'length'
    """
    center = length // 2
    x = np.linspace(-center, center, length)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    return kernel


def apply_gaussian_weights(mats, sigma):
    """
    Given list of prob matrices, computes the Gaussian Weighted Mean matrix
    """
    prob_mats = np.vstack(mats)

    weights = generate_gaussian_weights(sigma, len(prob_mats))
    weighted_prob_mats = []

    for i, prob_mat in enumerate(prob_mats):
        weighted_prob_mats.append(prob_mat * weights[i])

    gaussian_avged_mat = np.sum(weighted_prob_mats, axis=0) / np.sum(weights, axis=0)

    return gaussian_avged_mat


def calibrate_video_frames(probs, prop_length, prop_stride, gaussian_weights):
    """
    Calibrate all frames in a video by aggregating overlapping frame probabilities

    params:
        probs: NxN (row = proposal probs, col = class probs) np.ndarray of all overlapping proposal probs
        prop_length: frame length of a proposal
        prop_stride: how many frames between proposals
        gaussian_weights: array/list of gaussian weights; length must must proposal length
    returns:
        new np.ndarray of calibrated probs for each frame in video
    """
    overlap_factor = int(prop_length / prop_stride)

    # init probs matrix with offset padding for each overlap ratio group
    all_frame_probs_by_overlap_ratio = [
        [np.zeros((int(prop_length * 0), probs.shape[1]))],
        [np.zeros((int(prop_length * 0.25), probs.shape[1]))],
        [np.zeros((int(prop_length * 0.5), probs.shape[1]))],
        [np.zeros((int(prop_length * 0.75), probs.shape[1]))]
    ]
    # init weights matrix with offset padding for each overlap ratio group
    all_frame_weights_by_overlap_ratio = [
        [np.zeros((int(prop_length * 0), 1))],
        [np.zeros((int(prop_length * 0.25), 1))],
        [np.zeros((int(prop_length * 0.5), 1))],
        [np.zeros((int(prop_length * 0.75), 1))]
    ]
    # hstack probs and weights onto their respective overlap group
    for i in range(len(probs)):
        idx = i % overlap_factor
        # generate weighted probs for each frame in proposal via Gaussian distribution
        all_frame_probs = np.expand_dims(probs[i], axis=0)\
                        * np.expand_dims(gaussian_weights, axis=1) 
        
        # stack horizontally with already existing matrix
        all_frame_probs_by_overlap_ratio[idx].append(all_frame_probs)
        all_frame_weights_by_overlap_ratio[idx].append(np.expand_dims(gaussian_weights, axis=1))
        
    all_frame_probs_by_overlap_ratio = list(map(lambda x:np.vstack(x), all_frame_probs_by_overlap_ratio))
    all_frame_weights_by_overlap_ratio = list(map(lambda x:np.vstack(x), all_frame_weights_by_overlap_ratio))

    num_total_frames = len(max(all_frame_probs_by_overlap_ratio, key=len))
    all_frame_probs = np.zeros((num_total_frames, probs.shape[1]))
    all_frame_weights = np.zeros((num_total_frames, 1))

    # add padding to make mats same length and aggregate probs
    for i in range(len(all_frame_probs_by_overlap_ratio)):
        p = all_frame_probs_by_overlap_ratio[i]
        w = all_frame_weights_by_overlap_ratio[i]

        p = np.vstack([p, np.zeros((num_total_frames - len(p), probs.shape[1]))])
        w = np.vstack([w, np.zeros((num_total_frames - len(w), 1))])

        all_frame_probs += p
        all_frame_weights += w

    # aggregate overlapping probs via Gaussian weighted average
    calibrated_frame_probs = all_frame_probs / all_frame_weights
    return calibrated_frame_probs


def smooth_probs(probs, window_size):
    """
    Perform moving average over the proposal probabilities for a given window size
    params:
        probs: NxT matrix (class probs x timestamps)
        window_size: num of timestamps to average together at once
    returns:
        average probs
    """
    return np.convolve(probs, np.ones(window_size)/window_size, mode='valid')


def merge_sequence(sequence, proposal_length_secs):
    """
    Given a sequence of start times which correspond to a single peak,
    merge times and output final start and end times for peak

    params:
        sequence: list-like of start times which correspond to single peak
        proposal_length_secs: length of a proposal in seconds
    return:
        tuple (start time, end time) in seconds
    """
    return (sequence[0], sequence[-1] + proposal_length_secs)


# def find_peaks(probs, start_times, proposal_length_secs, prob_threshold, link_threshold=2):
#     """
#     Finds the start and end bounds of every peak; uses relative thresholding for processing

#     params:
#         probs: the data to process bounds for
#         start_times: array of start times that represent each probability datapoint
#         proposal_length_secs: length of a proposal in seconds
#         prob_threshold; threshold to use for probabilities
#         link_threshold: max difference in seconds between two proposals for them 
#                         to be considered a part of same action sequence
#     returns:
#         dict of tuples; key = (start time, end time), value = list of probabilities in bounds
#     """
#     # filter via relative thresholding
#     probs[probs < prob_threshold] = 0
#     peak_idxs = np.where(probs > 0)
#     peak_start_times = np.squeeze(np.take(start_times, peak_idxs), axis=0)
#     filtered_probs = np.squeeze(np.take(probs, peak_idxs), axis=0)

#     all_bounds = {}
#     sequence_start_times = [peak_start_times[0]]
#     sequence_probs = [filtered_probs[0]]
    
#     for i in range(1, len(peak_start_times)):
#         # if time b/w curr start and prev end is too large, merge current sequence times and reset sequence
#         if peak_start_times[i] - (sequence_start_times[-1] + proposal_length_secs) > link_threshold:
#             # merge start times
#             bounds = merge_sequence(sequence_start_times, proposal_length_secs)
#             # add merged bounds and its probs to dict
#             all_bounds[bounds] = sequence_probs
#             # reset sequence and probs
#             sequence_start_times = []
#             sequence_probs = []

#         sequence_start_times.append(peak_start_times[i])
#         sequence_probs.append(filtered_probs[i])

#     if len(sequence_start_times) > 0:
#         bounds = merge_sequence(sequence_start_times, proposal_length_secs)
#         all_bounds[bounds] = sequence_probs

#     return all_bounds


def merge_bounds(action_probs, 
                 action_start_times,
                 proposal_length_secs,
                 link_threshold):
    """
    Given start times of proposals where an action occurs, merges start times into bounded segments

    params:
        probs: the data to process bounds for
        start_times: array of start times that represent each probability datapoint
        proposal_length_secs: length of a proposal in seconds
        prob_threshold; threshold to use for probabilities
        link_threshold: max difference in seconds between two proposals for them 
                        to be considered a part of same action sequence
    returns:
        dict of tuples; key = (start time, end time), value = list of probabilities in bounds
    """
    all_bounds = {}
    sequence_start_times = [action_start_times[0]]
    sequence_probs = [action_probs[0]]
    
    if len(action_start_times) > 1:
        for i in range(1, len(action_start_times)):
            # if time b/w curr start and prev end is too large, merge current sequence times and reset sequence
            if action_start_times[i] - (sequence_start_times[-1] + proposal_length_secs) > link_threshold:
                # merge start times
                bounds = merge_sequence(sequence_start_times, proposal_length_secs)
                # add merged bounds and its probs to dict
                all_bounds[bounds] = sequence_probs
                # reset sequence and probs
                sequence_start_times = []
                sequence_probs = []

            sequence_start_times.append(action_start_times[i])
            sequence_probs.append(action_probs[i])

    if len(sequence_start_times) > 0:
        bounds = merge_sequence(sequence_start_times, proposal_length_secs)
        all_bounds[bounds] = sequence_probs

    return all_bounds


def relative_threshold_filter(probs, threshold_percentage):
    """
    Filters out action instances where prob is lower than a relative threshold

    params:
        probs: NxC array of probabilities (N = num proposals, C = num classes)
        threshold_percentage: in range [0, 1]; value is multiplied by max prob to obtain relative threshold
    returns:
        tuple (indices for proposal probs >= threshold, threshold)
    """
    prob_threshold = max(probs)*threshold_percentage
    action_idxs = np.where(probs >= prob_threshold)
    return action_idxs, prob_threshold


def argmax_filter(probs, activity_id):
    cls = np.argmax(probs, axis=-1) # argmax all probs
    # get idxs where argmax == action
    action_idxs = np.where(cls == activity_id)
    return action_idxs


def locate_action_bounds(action_idxs,
                         probs, 
                         start_times, 
                         proposal_length_secs,  
                         link_threshold=2):
    """
    Locate action bounds

    params:
        action_idxs: list of filtered indices of the proposals where the action instance most likely occurs
        probs: (N,) numpy array of ALL probabilities for a single action class in a video
        start_times: start times for ALL proposals in a video
        proposal_length_secs: length of a single proposal in seconds
        link_threshold: max difference in seconds between two proposals for them 
                        to be considered a part of same action sequence
    """
    # get predicted start times according to idxs
    action_start_times = start_times[action_idxs]
    # get action probs according to proposal idxs
    action_probs = probs[action_idxs]
    # merge start times into bounded segments
    action_bounds = merge_bounds(action_probs, 
                                 action_start_times, 
                                 proposal_length_secs,
                                 link_threshold)
    return action_bounds


def filter_action_bounds(bounds_dict, length_threshold, bonus_per_sec=0.05, sigma_per_proposal=0.55):
    """
    Filters peak bounds based on confidence and length

    params:
        bounds_dict: dict with key = (start, end) in secs of bound
                               value = list of probs in bounds
        bonus_per_sec: constant score applied per second of the boundary length
        sigma_per_proposal: sigma refers to Gaussian sigma parameter; computed as sigma_per_proposal * num proposals
                            sigma_per_proposal is a proportion of how much each proposal contributes to the sigma parameter
        length_threshold: if length (secs) of segment is lower than threshold, it is filtered out
    returns:
        the bounds with the highest score or None if no bounds detected
    """
    bound_scores_dict = {}

    for bound, bound_probs in bounds_dict.items():
        # apply Gaussian weighted average proportionally over bound probs
        # gaussian_weighted_avg = apply_gaussian_weights(bound_probs, sigma_per_proposal*len(bound_probs))
        mean_prob = np.array(bound_probs).mean()
        # get length of bound
        bound_length_secs = bound[1] - bound[0]
        # determine score by factoring in length
        if bound_length_secs >= length_threshold:  
            # bound_score = gaussian_weighted_avg + bound_length_secs*bonus_per_sec
            bound_score = mean_prob + bound_length_secs*bonus_per_sec
            bound_scores_dict[bound] = bound_score
    if len(bound_scores_dict) == 0:
        best_bound = None
        print(bounds_dict)
    else:
        best_bound = max(bound_scores_dict.keys(), key=lambda x: bound_scores_dict[x])
    return best_bound


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_left_right_search_bounds(best_bound, probs, start_times, proposal_length_secs):
    """
    Helper function: Fetches left and right search bounds for linking
    
    params:
        best_bound: tuple of best bound detected (start time, end time)
        bounds_dict: dict of all detected bounds (key=tuple[start time, end time], value=list of probs)
        probs: (N,) numpy array of ALL probabilities for a single action class in a video
        start_times: start times for ALL proposals in a video

    returns:
        ndarray of probs and start times for left and right search bounds, respectively
    """
    left_bound = start_times[0] # grab start time
    right_bound = start_times[-1] # grab start time

    # parse all probs and start times within search bounds
    left_bound_idx = np.where(start_times == left_bound)[0][0]
    bb_idx = np.where(start_times == best_bound[0])[0][0] # use start time of best seg to bound left side search
    left_probs = probs[left_bound_idx:bb_idx+1]
    left_start_times = start_times[left_bound_idx:bb_idx+1]

    right_bound_idx = np.where(start_times == right_bound)[0][0]
    best_bound_end_time = find_nearest(start_times, best_bound[1] - proposal_length_secs)
    bb_idx = np.where(start_times == best_bound_end_time)[0][0] # use end time of best seg to bound right side search
    right_probs = probs[bb_idx:right_bound_idx+1]
    right_start_times = start_times[bb_idx:right_bound_idx+1]

    return [(left_probs, left_start_times), (right_probs, right_start_times)]


def NMS_linking(bounds, bound_probs, conf_thresholds, side):
    """
    Performs Non-Maximum Suppresion on bounds detected over multiple confidence thresholds

    params:
        bounds: list of tuples (start time, end time) representing action bounds across multiple conf thresholds
        bound_probs: (NxP) numpy array of floats representing proposal probabilities within each bound
                     N = num of bounds, P = num proposals
        conf_thresholds: list of all confidence thresholds (must correspond with order of results from bounds)
        side: "left" or "right
    returns:
        single tuple (start, end) representing the final bound after NMS is applied
    """
    bound_weights = list(map(lambda x:np.array(x).mean(), bound_probs))
    bound_weights = np.array(bound_weights) * conf_thresholds

    if side == "left":
        # all bounds should have same end time since search ends at the best bound
        end_times = list(set(map(lambda x:x[1], bounds)))
        assert len(end_times) == 1, f"Error: left side end times are different {end_times}" 
        end_time = end_times[0]

        # analyze changes in start time 
        start_times = list(map(lambda x:x[0], bounds)) 
        # perform fusion
        start_time = sum([start_times[i] * bound_weights[i] for i in range(len(start_times))])
        start_time /= sum(bound_weights)

    elif side == "right":
        # all bounds should have same start time since search begins from best bound onward
        start_times = list(set(map(lambda x:x[0], bounds)))
        assert len(start_times) == 1, f"Error: right side start times are different {start_times}" 
        start_time = start_times[0]

        # analyze changes in end time 
        end_times = list(map(lambda x:x[1], bounds))
        # perform fusion
        end_time = sum([end_times[i] * bound_weights[i] for i in range(len(end_times))])
        end_time /= sum(bound_weights)

    return (start_time, end_time)


def link_action_bounds(best_bound,
                       probs, 
                       start_times, 
                       proposal_length_secs,
                       link_threshold=2,
                       conf_thresholds=np.round(np.arange(0.1, 1.1, 0.1), 1)):
    """
    Refines best bound by searching for adjacent bounds to potentially link to it
    Performs adjacency search at multiple confidence levels

    params:
        best_bound: tuple of best bound detected (start time, end time)
        bounds_dict: dict of all detected bounds (key=tuple[start time, end time], value=list of probs)
        probs: (N,) numpy array of ALL probabilities for a single action class in a video
        start_times: start times for ALL proposals in a video
        proposal_length_secs: length of a single proposal in seconds
        link_threshold: max difference in seconds between two proposals for them 
                    to be considered a part of same action sequence
        conf_thresholds: list of threshold floats that determine the confidence level search space

    returns:
        refined best bound tuple 
    """
    left_right_bounds = get_left_right_search_bounds(best_bound,
                                                     probs, 
                                                     start_times,
                                                     proposal_length_secs)

    # store bounds detected at all confidence levels
    multiconf_left_bounds = []
    multiconf_left_probs = []
    multiconf_right_bounds = []
    multiconf_right_probs = []

    # generate linking search space by checking proposals/bounds to left and right of best bound
    for i, adjacent_bounds in enumerate(left_right_bounds):
        action_probs, action_start_times = adjacent_bounds
        # perform multiple confidence level linking
        for threshold in conf_thresholds:
            action_idxs = np.where(action_probs >= threshold)
            results_dict = locate_action_bounds(action_idxs,
                                                action_probs, 
                                                action_start_times, 
                                                proposal_length_secs, 
                                                link_threshold)
            if i == 0: # append left bounds results
                left_bound = list(results_dict.keys())[-1] # last bound detected is the updated best bound
                                                           # last bound is the end bound of the left search space = best bound
                # print("l",left_bound)
                left_probs = results_dict[left_bound]
                multiconf_left_bounds.append(left_bound)
                multiconf_left_probs.append(left_probs)
            else: # append right bounds results
                right_bound = list(results_dict.keys())[0] # first bound detected is the updated best bound
                                                           # first bound is the start bound of the right search space = best bound
                # print("r",right_bound)
                right_probs = results_dict[right_bound]
                multiconf_right_bounds.append(right_bound)
                multiconf_right_probs.append(right_probs)

    # print("Best",best_bound)
    # print("l",multiconf_left_bounds)
    # print("r",multiconf_right_bounds)

    # perform NMS on bounds detected across multiple confidence levels
    final_left_bound = NMS_linking(multiconf_left_bounds,
                                   multiconf_left_probs,
                                   conf_thresholds,
                                   side="left")
    final_right_bound = NMS_linking(multiconf_right_bounds,
                                    multiconf_right_probs,
                                    conf_thresholds,
                                    side="right")
    # merge left and right final bounds to get refined bound
    refined_bound = (final_left_bound[0], final_right_bound[1])

    return refined_bound            
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Post-process')

    parser.add_argument('--video_fps', type=int, default=30,
                        help='FPS of the videos being processed')
    parser.add_argument('--probs_dir', type=str, default="./inference/probs",
                        help='Folder where probabilities are stored')
    parser.add_argument('--proposal_length', type=int, default=64,
                        help='Number of frames in a proposal')
    parser.add_argument('--proposal_stride', type=int, default=16,
                        help='The rate at which proposals are generated at (in frames)')
    parser.add_argument('--num_videos', type=int, default=10,
                        help='Number of videos to be processed')
    parser.add_argument('--gaussian_sigma', type=int, default=30,
                        help='Sigma hyperparameter to use for Gaussian weights')
    parser.add_argument('--sub_file_path', type=str, default="./inference/submission_files/submission_file.txt",
                        help='Submission file output path')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    
    args = parser.parse_args()
    gaussian_weights = generate_gaussian_weights(args.gaussian_sigma, args.proposal_length)

    # set random seed
    np.random.seed(args.seed)

    # set action linking threshold (max downtime b/w peaks before considering them as different segments)
    noise_link_thresholds = {
                            1:2,
                            2:2,
                            3:2,
                            4:2,
                            5:2,
                            6:4,
                            7:2,
                            8:2,
                            9:2,
                            10:2,
                            11:2,
                            12:5,
                            13:10,
                            14:5,
                            15:15
                            }
    # for linking algorithm
    noise_link_thresholds_2 = {
                            1:.5,
                            2:2,
                            3:2,
                            4:4,
                            5:.5,
                            6:.5,
                            7:4,
                            8:.5,
                            9:.5,
                            10:.5,
                            11:2,
                            12:2,
                            13:5,
                            14:5,
                            15:.5,
                            }
    # min length of segments for different actions
    noise_length_thresholds = {
                            1:0,
                            2:4,
                            3:4,
                            4:2,
                            5:4,
                            6:4,
                            7:0,
                            8:0,
                            9:0,
                            10:0,
                            11:4,
                            12:4,
                            13:2,
                            14:2,
                            15:4
                            }
    # for each action, probabilities are thresholded relative to a percentage of the highest probability
    relative_prob_threshold = .75
    
    # # delete existing submission_file.txt if exists
    if os.path.exists(args.sub_file_path):
        print("Found existing submission file. Deleting it...")
        os.remove(args.sub_file_path)

    for video_id in tqdm(range(1, args.num_videos + 1)):
        probs_by_view = {}
        for view in os.listdir(args.probs_dir):
            # parse file paths
            view_dir = args.probs_dir + '/' + view
            # obtain path for probs corresponding to given video id
            probs_files = [file for file in os.listdir(view_dir) if f"video_{video_id}_" in file]

            # if multiple prob files found, perform K-Fold CV ensemble averaging
            video_probs = []
            for probs_file in probs_files:
                # reconstruct path   
                probs_file = view_dir + '/' + probs_file
                # load probs as numpy matrix
                with open(probs_file, "rb") as f:
                    video_probs.append(np.load(f))
            # perform ensemble averaging
            probs_by_view[view] = np.array(video_probs).mean(axis=0)

        ensemble_probs = .31 * probs_by_view["dashboard"] +\
                         .35 * probs_by_view["rear_view"] +\
                         .34 * probs_by_view["right_side_window"]
        ensemble_probs[:, 8] = probs_by_view["right_side_window"][:, 8]
        ensemble_probs[:, [13,14]] = probs_by_view["rear_view"][:, [13,14]]

        ensemble_cls = np.argmax(ensemble_probs, axis=-1)

        # ensemble_probs = ensemble_probs[::(args.proposal_length//args.proposal_stride)]
        # ensemble_probs = probs_by_view["dashboard"]
        
        fig, axs = plt.subplots(4, 4, figsize=(40,40))

        # process probs for each class separately
        for activity_id in range(1, 16):
            # print(activity_id)
            probs = ensemble_probs[:, activity_id]
            start_times_secs = np.array(range(len(probs)))*args.proposal_stride/args.video_fps
            # start_times_secs = np.array(range(len(probs)))*args.proposal_length/args.video_fps

            row = (activity_id)//4
            col = ((activity_id+1) % 4) - 1

            axs[row,col].plot(start_times_secs, probs, c="blue", alpha=0.5)
            axs[row,col].plot(start_times_secs, 
                              len(probs)*[max(probs)*relative_prob_threshold], 
                              linestyle="dotted", c='orange', alpha=0.75)

            # action_idxs = argmax_filter(ensemble_probs, activity_id)
            action_idxs, prob_threshold = relative_threshold_filter(probs, relative_prob_threshold)
            bounds_dict = locate_action_bounds(action_idxs,
                                               probs, 
                                               start_times_secs, 
                                               args.proposal_length/args.video_fps, 
                                               link_threshold=noise_link_thresholds[activity_id]) 
            best_bound = filter_action_bounds(bounds_dict, noise_length_thresholds[activity_id])

            if best_bound is not None:
                # refine/link bounds
                # if video_id == 7 and activity_id == 4:
                max_conf = prob_threshold * 10 // 1 / 10 # round down max conf to 1st decimal place
                conf_thresholds = np.round(np.arange(0.2, max_conf + 0.1, 0.1), 1)
                best_bound = link_action_bounds(best_bound,
                                                probs,
                                                start_times_secs,
                                                args.proposal_length/args.video_fps,
                                                link_threshold=noise_link_thresholds_2[activity_id],
                                                conf_thresholds=conf_thresholds)
                with open(args.sub_file_path, "a+") as sub_f:
                    # output video id, action class, start time, end time
                    sub_f.writelines(f"{video_id} {activity_id} {round(best_bound[0])} {round(best_bound[1])}\n")

                # plot results
                for z, x_b in enumerate(best_bound):
                    axs[row,col].axvline(x_b, linestyle="dashed", c="green", alpha=1)   
                    axs[row,col].text(x_b, .2 if z == 0 else .25, f"{x_b:.2f}", verticalalignment='center')

            axs[row,col].title.set_text(f"Class {activity_id}")
            axs[row,col].set_xlabel("Time (seconds)")
            axs[row,col].set_ylabel("Prediction Probability")
            # axs[row,col].set_xlim(50, 200)
            
        plt.savefig(f"video_{video_id}_results.jpg")
        plt.clf()
    print("Results saved.")