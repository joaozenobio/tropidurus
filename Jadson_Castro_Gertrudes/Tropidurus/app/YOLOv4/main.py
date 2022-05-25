import cv2 as cv
import os
import json
import tempfile
import shutil


def save_frames(frames_dir_path):
    videos_path = ["../videos/" + filename for filename in os.listdir("../videos")]
    for path in videos_path:
        video = cv.VideoCapture(path)
        frame_counter = 0
        while True:
            ok, frame = video.read()
            if not ok:
                break
            cv.imwrite(f"{frames_dir_path}/{path.split('/')[2].split('.')[0]}_counter-{frame_counter}.jpg", frame)
            frame_counter += 1


def create_frames_paths_txt(frames_dir_path):
    with open(os.getcwd() + "/frames_paths.txt", 'w') as f:
        for filename in os.listdir(frames_dir_path):
            f.write(f"{frames_dir_path}/{filename}\n")


def darknet(frames_dir_path):
    create_frames_paths_txt(frames_dir_path)
    thresh_hold = "0.56"
    project_path = str(os.getcwd()) + "/"
    full_input_path = project_path + "frames_paths.txt"
    full_output_path = project_path + "output.json"
    command = f"./darknet detector -dont_show test my_data/images.data my_data/yolo.cfg my_data/backup/yolo-obj_best.weights" \
              f" -thresh {thresh_hold} -out {full_output_path} < {full_input_path}"
    os.system(f"cd ../darknet && {command}")


def convert_coordinates(size, box):
    x_min = abs(int(size[0] * (box[0] - box[2])))
    y_min = abs(int(size[1] * (box[1] - box[3])))
    x_max = abs(int(size[0] * (box[0] + box[2])))
    y_max = abs(int(size[1] * (box[1] + box[3])))
    return x_min, y_min, x_max, y_max


def draw_rectangles(frames_temp_dir_path):
    with open('output.json') as output_file:
        data = json.load(output_file)
        for frame in data:
            img = cv.imread(frame['filename'])
            height, width, _ = img.shape
            head_confidence = .0
            body_confidence = .0
            for i in range(len(frame['objects'])):
                name = frame['objects'][i]['name']
                if name == 'head':
                    if float(frame['objects'][i]['confidence']) > head_confidence:
                        head_confidence = float(frame['objects'][i]['confidence'])
                        head_temp = frame['objects'][i]
                if name == 'body':
                    if float(frame['objects'][i]['confidence']) > body_confidence:
                        body_confidence = float(frame['objects'][i]['confidence'])
                        body_temp = frame['objects'][i]
            x = head_temp['relative_coordinates']['center_x']
            y = head_temp['relative_coordinates']['center_y']
            w = head_temp['relative_coordinates']['width']
            h = head_temp['relative_coordinates']['height']
            box = (float(x), float(y), float(w), float(h))
            size = (width, height)
            x_min, y_min, x_max, y_max = convert_coordinates(size, box)
            cv.rectangle(img, (x_min, y_min), (x_max, y_max), 255, 2)
            x = body_temp['relative_coordinates']['center_x']
            y = body_temp['relative_coordinates']['center_y']
            w = body_temp['relative_coordinates']['width']
            h = body_temp['relative_coordinates']['height']
            box = (float(x), float(y), float(w), float(h))
            size = (width, height)
            x_min, y_min, x_max, y_max = convert_coordinates(size, box)
            cv.rectangle(img, (x_min, y_min), (x_max, y_max), 255, 2)
            cv.imwrite(f"{frames_temp_dir_path}/{frame['filename'].split('/')[-1].split('.')[0]}_result.jpg", img)


def make_videos(frames_temp_dir_path, resultados_dir_path):
    draw_rectangles(frames_temp_dir_path)
    videos_path = ["../videos/" + filename for filename in os.listdir("../videos")]
    for path in videos_path:
        video = cv.VideoCapture(path)
        width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
        video_output_path = f"{resultados_dir_path}/{path.split('/')[-1].split('.')[0]}_result.mp4"
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_writer = cv.VideoWriter(video_output_path, fourcc, 30.0, (width, height))
        i = 0
        frame_path = f"{path.split('/')[-1].split('.')[0].split('_counter-')[0]}_counter-{i}_result.jpg"
        while frame_path in os.listdir(frames_temp_dir_path):
            img = cv.imread(f"{frames_temp_dir_path}/{frame_path}")
            video_writer.write(img)
            i += 1
            frame_path = f"{path.split('/')[-1].split('.')[0].split('_counter-')[0]}_counter-{i}_result.jpg"
        video.release()
        video_writer.release()

os.chdir(os.getcwd() + "/YOLOv4/")
frames_dir_path = tempfile.mkdtemp()
frames_temp_dir_path = tempfile.mkdtemp()
results_dir_path = "results"
if not os.path.exists(results_dir_path):
    os.mkdir(results_dir_path)

save_frames(frames_dir_path)
darknet(frames_dir_path)
make_videos(frames_temp_dir_path, results_dir_path)

shutil.rmtree(frames_dir_path)
shutil.rmtree(frames_temp_dir_path)
