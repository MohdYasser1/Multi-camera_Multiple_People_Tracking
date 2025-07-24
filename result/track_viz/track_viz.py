import pandas as pd
import os
import os.path as osp
import sys
import numpy as np
from tqdm import tqdm

import cv2

def annotate_frame(frame, bbox, obj_id, color=(0, 255, 0)):
    x_min, y_min, width, height = bbox
    x_min, y_min, width, height = int(x_min), int(y_min), int(width), int(height)
    x_max, y_max = x_min + width, y_min + height

    # Draw the bounding box
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

    # Put the text label
    label = f'ID: {obj_id}'
    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def generate_colors(num_colors):
    np.random.seed(42)  # For reproducibility
    colors = np.random.randint(0, 255, size=(num_colors, 3), dtype=int)
    return {i: tuple(map(int, color)) for i, color in enumerate(colors)}


def main():
    scene_name = sys.argv[1]

    current_file_path = os.path.abspath(__file__)
    path_arr = current_file_path.split('/')[:-2]
    root_path = '/'.join(path_arr)

    track_dir = osp.join(root_path, 'track')
    track_data = np.loadtxt(osp.join(track_dir, scene_name+'.txt'))

    obj_ids = track_data[:, 1].astype(int)
    camera_ids = track_data[:, 0].astype(int)

    # Create a dictionary to count unique cameras for each obj_id
    obj_camera_dict = {}
    for obj_id, camera_id in zip(obj_ids, camera_ids):
        if obj_id not in obj_camera_dict:
            obj_camera_dict[obj_id] = set()
        obj_camera_dict[obj_id].add(camera_id)

    # Filter obj_ids tracked by more than 2 cameras
    tracked_by_multiple_cameras = {obj_id: cameras for obj_id, cameras in obj_camera_dict.items() if len(cameras) > 2}

    # # If there are no such obj_ids, handle the case
    # if not tracked_by_multiple_cameras:
    #     print("No person is tracked by more than 2 cameras.")
    # else:
    #     # Determine the most tracked person among those tracked by more than 2 cameras
    #     most_tracked_person = max(tracked_by_multiple_cameras, key=lambda obj_id: len(tracked_by_multiple_cameras[obj_id]))
    #     most_tracked_count = np.sum(obj_ids == most_tracked_person)

    #     print(f"The most tracked person across more than 2 cameras is obj_id {most_tracked_person} with {most_tracked_count} occurrences.")

    most_tracked_person = max(obj_camera_dict, key=lambda obj_id: len(obj_camera_dict[obj_id]))
    print(f"The person tracked by the most cameras is obj_id {most_tracked_person}")


    obj_id_to_track = most_tracked_person 
    dataset_dir = 'dataset/test'
    scene_dir = dataset_dir

    person_results = track_data[track_data[:, 1] == obj_id_to_track]

    unique_ids = np.unique(obj_ids)
    colors = generate_colors(len(unique_ids))
    id_to_color = {obj_id: colors[i] for i, obj_id in enumerate(unique_ids)}
    for cam_id in np.unique(person_results[:, 0]):
        camera_dir = os.path.join(scene_dir, f'camera_0{int(cam_id)}')
        video_path = os.path.join(camera_dir, 'video.mp4')

        cap = cv2.VideoCapture(video_path)

        # Get the frames where the object is present
        frames_with_object = person_results[person_results[:, 0] == cam_id][:, 2].astype(int)

        for frame_id in frames_with_object:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if not ret:
                break

            frame_results = track_data[(track_data[:, 0] == cam_id) & (track_data[:, 2] == frame_id)]

            for result in frame_results:
                if int(result[0]) == cam_id:
                    bbox = result[3:7]
                    bbox = result[3:7]
                    obj_id = int(result[1])
                    color = (0, 0, 255) if obj_id == obj_id_to_track else (0, 255, 0)
                    annotate_frame(frame, bbox, obj_id, color)
                    

            cv2.imshow(f'Camera {cam_id}', frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):  # Press 'q' to quit
                break

        cap.release()
        cv2.destroyAllWindows()

    for cam_id in tqdm(np.unique(camera_ids)):
        camera_dir = os.path.join(scene_dir, f'Camera_0{int(cam_id)}')
        video_path = camera_dir+ '.mp4'
        # print(f'Processing camera {cam_id} video: {video_path}')
        output_dir = osp.join(track_dir, scene_name)
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
        output_path = osp.join(output_dir, f'Camera_0{int(cam_id)}_tracked.mp4')

        cap = cv2.VideoCapture(video_path)

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        pbar = tqdm(desc=f'Camera {cam_id}', total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            frame_results = track_data[(track_data[:, 0] == cam_id-1) & (track_data[:, 2] == frame_id)]

            for result in frame_results:
                bbox = result[3:7]
                obj_id = int(result[1])
                color = (0, 0, 255) if obj_id == 1 else (0, 255, 0)
                # color = id_to_color[obj_id]
                annotate_frame(frame, bbox, obj_id, color)

            pbar.update(1)
            cv2.imshow(f'Camera {cam_id}', frame)
            print("Frame ", frame_id)
            if cv2.waitKey(100) & 0xFF == ord('q'):  # Press 'q' to quit
                break
            out.write(frame)

        cap.release()
        out.release()
        # cv2.destroyAllWindows()
        out.release()
    cap.release()
    # cv2.destroyAllWindows()




    # # # Generate colors for each unique ID
    # unique_ids = np.unique(obj_ids)
    # colors = generate_colors(len(unique_ids))
    # id_to_color = {obj_id: colors[i] for i, obj_id in enumerate(unique_ids)}

    # # Open video captures for all cameras
    # caps = {}
    # for cam_id in np.unique(camera_ids):
    #     camera_dir = os.path.join(scene_dir, f'Camera_0{int(cam_id)}')
    #     video_path = camera_dir+ '.mp4'
    #     caps[cam_id] = cv2.VideoCapture(video_path)

    # # Get screen resolution
    # screen_width = 2800
    # screen_height = 1080

    # # Calculate window size
    # num_cameras = len(caps)
    # window_width = screen_width
    # window_height = screen_height

    # while True:
    #     frames = {}
    #     for cam_id, cap in caps.items():
    #         ret, frame = cap.read()
    #         if not ret:
    #             continue

    #         frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    #         frame_results = track_data[(track_data[:, 0] == cam_id-1) & (track_data[:, 2] == frame_id)]

    #         for result in frame_results:
    #             bbox = result[3:7]
    #             obj_id = int(result[1])
    #             color = id_to_color[obj_id]
    #             annotate_frame(frame, bbox, obj_id, color)

    #         frames[cam_id] = frame
    #         resized_frame = cv2.resize(frame, (window_width, window_height))
    #         cv2.imshow(f'Camera {cam_id}', resized_frame)

    #     # Check if any window is closed or 'q' is pressed
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # # Release all captures and close windows
    # for cap in caps.values():
    #     cap.release()
    # cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()
